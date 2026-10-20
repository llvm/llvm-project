// RUN: mlir-opt %s -convert-agx-to-agxir \
// RUN:   -convert-gpu-to-agxir='use-bare-ptr-memref-call-conv=1' \
// RUN:   -reconcile-unrealized-casts -o %t.mlir
// RUN: %run_agx --grid 32 --wg 32 %t.mlir f32:@%S/Inputs/identity-fragments.txt \
// RUN:   f32:iota*64 f32*64 | FileCheck %s

// I * B + B = 2 B: each thread's two elements, in fragment order, are
// twice the inputs it holds.
// CHECK: 2: 0 2 4 6 8 10 {{.*}} 122 124 126{{$}}

gpu.module @kernels {
  gpu.func @mma(%a: memref<32xvector<2xf32>, 1>, %b: memref<32xvector<2xf32>, 1>,
                %out: memref<32xvector<2xf32>, 1>) kernel {
    %t = gpu.lane_id
    %x = memref.load %a[%t] : memref<32xvector<2xf32>, 1>
    %y = memref.load %b[%t] : memref<32xvector<2xf32>, 1>
    %d = agx.mma %x * %y + %y : vector<2xf32>, vector<2xf32>
    memref.store %d, %out[%t] : memref<32xvector<2xf32>, 1>
    gpu.return
  }
}
