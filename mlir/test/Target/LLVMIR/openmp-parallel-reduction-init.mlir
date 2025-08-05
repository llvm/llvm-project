// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// Regression test for https://github.com/llvm/llvm-project/issues/120254.

omp.declare_reduction @add_reduction : !llvm.ptr alloc {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x !llvm.struct<(ptr)> : (i64) -> !llvm.ptr
  omp.yield(%1 : !llvm.ptr)
} init {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  %6 = llvm.mlir.constant(1 : i32) : i32
  "llvm.intr.memcpy"(%arg1, %arg0, %6) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
  omp.yield(%arg1 : !llvm.ptr)
} combiner {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  omp.yield(%arg0 : !llvm.ptr)
} cleanup {
^bb0(%arg0: !llvm.ptr):
  omp.yield
}

llvm.func @use_reduction() attributes {fir.bindc_name = "test"} {
  %6 = llvm.mlir.constant(1 : i32) : i32
  omp.parallel {
    %18 = llvm.mlir.constant(1 : i64) : i64
    %19 = llvm.alloca %18 x !llvm.struct<(ptr)> : (i64) -> !llvm.ptr
    omp.wsloop reduction(byref @add_reduction %19 -> %arg0 : !llvm.ptr) {
      omp.loop_nest (%arg1) : i32 = (%6) to (%6) inclusive step (%6) {
        omp.yield
      }
    }
    omp.terminator
  }
  llvm.return
}

// CHECK: omp.par.entry:
// CHECK:   %[[RED_REGION_ALLOC:.*]] = alloca { ptr }, i64 1, align 8

// CHECK: omp.par.region:
// CHECK:   br label %omp.par.region1

// CHECK: omp.par.region1:
// CHECK:   %[[PAR_REG_VAL:.*]] = alloca { ptr }, i64 1, align 8
// CHECK:   br label %omp.reduction.init

// CHECK: omp.reduction.init:
// CHECK:   call void @llvm.memcpy{{.*}}(ptr %[[RED_REGION_ALLOC]], ptr %[[PAR_REG_VAL]], {{.*}})
