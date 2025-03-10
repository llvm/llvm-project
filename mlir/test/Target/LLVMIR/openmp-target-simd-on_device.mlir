// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {omp.is_target_device = true} {
  omp.private {type = private} @simd_privatizer : !llvm.ptr init {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    omp.yield(%arg0 : !llvm.ptr)
  }

  llvm.func @test_target_simd() {
    omp.target {
      %5 = llvm.mlir.constant(1 : i32) : i32
      %x = llvm.alloca %5 x i32 {bindc_name = "x"} : (i32) -> !llvm.ptr
      omp.simd private(@simd_privatizer %x -> %arg1 : !llvm.ptr) {
        omp.loop_nest (%arg2) : i32 = (%5) to (%5) inclusive step (%5) {
          omp.yield
        }
      }
      omp.terminator
    }
    llvm.return
  }

}

// CHECK-LABEL: define {{.*}} @__omp_offloading_{{.*}}_test_target_simd_{{.*}}

// CHECK:         %[[INT:.*]] = alloca i32, align 4
// CHECK:         br label %[[LATE_ALLOC_BB:.*]]

// CHECK:       [[LATE_ALLOC_BB]]:
// CHECK:         br label %[[AFTER_ALLOC_BB:.*]]

// CHECK:       [[AFTER_ALLOC_BB]]:
// CHECK:         br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
