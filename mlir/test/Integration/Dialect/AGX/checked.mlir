// RUN: mlir-opt %s -convert-gpu-to-agxir='use-bare-ptr-memref-call-conv=1' \
// RUN:   -reconcile-unrealized-casts -ensure-debug-info-scope-on-llvm-func \
// RUN:   -mlir-print-debuginfo -o %t.mlir
// RUN: %run_agx --checked --cpu agx-g14c --grid 32 --wg 32 %t.mlir \
// RUN:   i32:iota*24 i32*32 index=32 | FileCheck %s
// RUN: %run_agx --checked --cpu agx-g14c --grid 32 --wg 32 %t.mlir \
// RUN:   i32:iota*32 i32*32 index=32 | FileCheck %s --check-prefix=OK

// Checked mode on an MLIR kernel: out[i] = 2 * in[i] for i < n, with the
// input 8 elements shorter than n. The first bad load is reported with the
// thread that made it and the line of the memref.load in this file; built
// without the bug, the kernel reports nothing.

// CHECK: checked: out of bounds load of 4 bytes at 0x1000060, thread (24, 0, 0)
// CHECK-SAME: checked.mlir:[[# @LINE + 9]]:12
// OK: checked: no error
// OK: 1: 0 2 4 6 {{.*}} 62{{$}}

gpu.module @kernels {
  gpu.func @k(%in: memref<32xi32, 1>, %out: memref<32xi32, 1>, %n: index) kernel {
    %i = gpu.thread_id x
    %c = arith.cmpi ult, %i, %n : index
    scf.if %c {
      %x = memref.load %in[%i] : memref<32xi32, 1>
      %y = arith.addi %x, %x : i32
      memref.store %y, %out[%i] : memref<32xi32, 1>
    }
    gpu.return
  }
}
