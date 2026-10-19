// RUN: mlir-opt %s -convert-agx-to-agxir \
// RUN:   -convert-gpu-to-agxir='use-bare-ptr-memref-call-conv=1' \
// RUN:   -reconcile-unrealized-casts -o %t.mlir
// RUN: %run_agx --grid 32 --wg 32 %t.mlir i32*32 i32*32 | FileCheck %s

// Each thread reads the lane index of its xor-1 neighbour, and the ballot of
// the even lanes is 0x55555555.
// CHECK: 0: 1 0 3 2 5 4 {{.*}} 31 30{{$}}
// CHECK: 1: 1431655765 1431655765 {{.*}} 1431655765{{$}}

gpu.module @kernels {
  gpu.func @k(%out: memref<32xi32, 1>, %mask: memref<32xi32, 1>) kernel {
    %t = gpu.lane_id
    %ti = arith.index_cast %t : index to i32
    %c1 = arith.constant 1 : i32
    %s = agx.shuffle xor %ti, %c1 : i32
    memref.store %s, %out[%t] : memref<32xi32, 1>
    %odd = arith.andi %ti, %c1 : i32
    %c0 = arith.constant 0 : i32
    %even = arith.cmpi eq, %odd, %c0 : i32
    %m = agx.ballot %even : i32
    memref.store %m, %mask[%t] : memref<32xi32, 1>
    gpu.return
  }
}
