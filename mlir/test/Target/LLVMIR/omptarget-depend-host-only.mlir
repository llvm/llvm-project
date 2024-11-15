// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {omp.is_target_device = false} {
  llvm.func @omp_target_depend_() {
    %0 = llvm.mlir.constant(39 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.constant(40 : index) : i64
    %3 = omp.map.bounds lower_bound(%1 : i64) upper_bound(%0 : i64) extent(%2 : i64) stride(%1 : i64) start_idx(%1 : i64)
    %4 = llvm.mlir.addressof @_QFEa : !llvm.ptr
    %5 = omp.map.info var_ptr(%4 : !llvm.ptr, !llvm.array<40 x i32>) map_clauses(from) capture(ByRef) bounds(%3) -> !llvm.ptr {name = "a"}
    omp.target depend(taskdependin -> %4 : !llvm.ptr) map_entries(%5 -> %arg0 : !llvm.ptr) {
      %6 = llvm.mlir.constant(100 : index) : i32
      llvm.store %6, %arg0 : i32, !llvm.ptr
      omp.terminator
    }
    llvm.return
  }

  llvm.mlir.global internal @_QFEa() {addr_space = 0 : i32} : !llvm.array<40 x i32> {
    %0 = llvm.mlir.zero : !llvm.array<40 x i32>
    llvm.return %0 : !llvm.array<40 x i32>
  }
}

// CHECK: define void @omp_target_depend_()
// CHECK-NOT: define {{.*}} @
// CHECK-NOT: call i32 @__tgt_target_kernel({{.*}})
// CHECK: call void @__omp_offloading_[[DEV:.*]]_[[FIL:.*]]_omp_target_depend__l[[LINE:.*]](ptr {{.*}})
// CHECK-NEXT: ret void

// CHECK: define internal void @__omp_offloading_[[DEV]]_[[FIL]]_omp_target_depend__l[[LINE]](ptr %[[ADDR_A:.*]])
// CHECK: store i32 100, ptr %[[ADDR_A]], align 4
