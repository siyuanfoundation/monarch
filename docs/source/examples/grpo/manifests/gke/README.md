# How to Setup RDMA on GKE

* See the tutorial of [Allocate network resources by using GKE managed DRANET](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/allocate-network-resources-dra) for the most up-to-date instructions.

1. Find a zone that supports the target GPUs with RDMA

```bash
# find a zone that has the accelerator type you want
gcloud compute accelerator-types list
# find a zone that has network profiles for RDMA
gcloud alpha compute network-profiles list
```

2. Create a cluster with dataplane v2 enabled

```bash
CLUSTER_NAME=${USER}-gpu-rdma
ZONE=us-central1-b # pick the zone found in step 1
gcloud container clusters create $CLUSTER_NAME \
  --zone=$ZONE \
  --machine-type=n2-standard-4 \
  --enable-dataplane-v2  # required for RDMA with DRANET

# Wait for the cluster to be ready, then you can get the credentials
gcloud container clusters get-credentials $CLUSTER_NAME --zone $ZONE
```

3. Create a node pool with RDMA support

```bash
# use h200 as an example, you can change it to other GPU types
gcloud container node-pools create h200-rdma-pool \
  --region=$ZONE \
  --cluster=$CLUSTER_NAME \
  --accelerator type=nvidia-h200-141gb,count=8 \
  --machine-type=a3-ultragpu-8g \
  --num-nodes=1 --spot \
  --accelerator-network-profile=auto \
  --node-labels=cloud.google.com/gke-networking-dra-driver=true
```