// RUN: not mlir-opt %s 2>&1 | FileCheck %s
// CHECK: 'nvvm.tcgen05.alloc' op is not supported on sm_90

module {
    gpu.module @mod [#nvvm.target<chip = "sm_90">] {
        func.func @tcgen05_alloc(%arg0: !llvm.ptr<7>, %arg1: i32) {
             nvvm.tcgen05.alloc %arg0, %arg1 : !llvm.ptr<7>, i32
             return
        }
    }
}
