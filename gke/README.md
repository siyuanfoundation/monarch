# How to Setup RDMA on GKE

## Create a cluster with dataplane v2 enabled

```bash
CLUSTER_NAME=${USER}-gpu-rdma
ZONE=us-central1-b
gcloud container clusters create $CLUSTER_NAME \
  --zone=$ZONE \
  --addons=RayOperator \
  --machine-type=n2-standard-16 \
  --enable-image-streaming \
  --enable-dataplane-v2 \
  --cluster-ipv4-cidr=172.20.0.0/20  # Explicitly request a smaller /20 block

gcloud container clusters get-credentials $CLUSTER_NAME --zone $ZONE
```

## Create a node pool with RDMA support

```bash
gcloud container node-pools create h200-rdma-pool \
  --region=$ZONE \
  --cluster=$CLUSTER_NAME \
  --accelerator type=nvidia-h200-141gb,count=8 \
  --machine-type=a3-ultragpu-8g \
  --num-nodes=1 --spot \
  --accelerator-network-profile=auto \
  --node-labels=cloud.google.com/gke-networking-dra-driver=true
```

## Test 2 Pods on the same node with RDMA

```bash
kubectl apply -f two-pods-rdma.yaml
```
