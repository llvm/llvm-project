// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Tests the "impure" alloc region workaround until `omp.private` is updated.
// See
// https://discourse.llvm.org/t/rfc-openmp-supporting-delayed-task-execution-with-firstprivate-variables/83084/7
// and https://discourse.llvm.org/t/delayed-privatization-for-omp-wsloop/83989
// for more info.

omp.private {type = private} @impure_alloca_privatizer : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "i", pinned} : (i64) -> !llvm.ptr
  %3 = llvm.getelementptr %arg0[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr)>
  omp.yield(%1 : !llvm.ptr)
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

// CHECK:       omp.target:
// CHECK:         %[[ALLOC_REG_ARG:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, align 8
// CHECK:         br label %omp.private.latealloc

// CHECK:       omp.private.latealloc:
// CHECK:         %[[PRIV_ALLOC:.*]] = alloca i32, i64 1, align 4
// The usage of `ALLOC_REG_ARG` in the inlined alloc region is the reason for
// introducing the workaround.
// CHECK:         %{{.*}} = getelementptr { ptr }, ptr %[[ALLOC_REG_ARG]], i32 0
// CHECK:         br label %omp.region.after_defining_block


