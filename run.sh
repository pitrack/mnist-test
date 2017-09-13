# source /export/a07/zach/libs/miniconda3/bin/activate mnist-test
source /home/pxia/anaconda3/envs/mnist-test/bin/activate mnist-test
timeout 10m nvidia-smi --query-gpu=index,timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,clocks.sm,clocks.mem,clocks.gr --format=csv --filename=/home/zwood/nvidia-stats.txt --loop-ms=1 &
echo "Running on `free-gpu`" >> gpu.txt
echo "Running on `free-gpu`"
CUDA_VISIBLE_DEVICES=`free-gpu` python /export/a07/zach/dic/mnist-test/pytorch_test.py
