// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s


// Tests inlining behavior of `omp.private` ops for `omp.wsloop` ops nested in
// `omp.target` ops.

llvm.func @foo(%arg0: !llvm.ptr)

omp.private {type = private} @impure_alloca_privatizer : !llvm.ptr init {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  llvm.call @foo(%arg0) : (!llvm.ptr) -> ()
  omp.yield(%arg1 : !llvm.ptr)
}

llvm.func @test_alloca_ip_workaround() {
  omp.target {
    %65 = llvm.mlir.constant(1 : i32) : i32
    %66 = llvm.alloca %65 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %67 = llvm.mlir.constant(0 : index) : i64
    %68 = llvm.mlir.constant(10 : i32) : i32
    %69 = llvm.mlir.constant(1 : i32) : i32
    omp.wsloop private(@impure_alloca_privatizer %66 -> %arg6 : !llvm.ptr) {
      omp.loop_nest (%arg8) : i32 = (%69) to (%68) inclusive step (%69) {
        omp.yield
      }
    }
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define {{.*}} @__omp_offloading_{{.*}}_test_alloca_ip_workaround
// CHECK:       entry:
// CHECK:         %[[PRIV_ALLOC:.*]] = alloca ptr, align 8

// CHECK:       omp.target:
// CHECK:         %[[ALLOC_REG_ARG:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, align 8
// Verify that the `init` region of the privatizer is inlined within the `target`
// region body not before that.
// CHECK:         call void @foo(ptr %[[ALLOC_REG_ARG]])
// CHECK:         br label %omp_loop.preheader
