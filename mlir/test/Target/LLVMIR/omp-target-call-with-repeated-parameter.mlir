// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @caller_()  {
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %x_host = llvm.alloca %c1 x f32 {bindc_name = "x"} : (i64) -> !llvm.ptr
  %i_host = llvm.alloca %c1 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
  %x_map = omp.map.info var_ptr(%x_host : !llvm.ptr, f32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !llvm.ptr {name = "x"}
  %i_map = omp.map.info var_ptr(%i_host : !llvm.ptr, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !llvm.ptr {name = "i"}
  omp.target map_entries(%x_map -> %x_arg, %i_map -> %i_arg : !llvm.ptr, !llvm.ptr) {
    %1 = llvm.load %i_arg : !llvm.ptr -> i32
    %2 = llvm.sitofp %1 : i32 to f32
    llvm.store %2, %x_arg : f32, !llvm.ptr
    // The call instruction uses %x_arg more than once. Hence modifying users
    // while iterating them invalidates the iteration. Which is what is tested
    // by this test.
    llvm.call @callee_(%x_arg, %x_arg) : (!llvm.ptr, !llvm.ptr) -> ()
    omp.terminator
  }
  llvm.return
}

llvm.func @callee_(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  llvm.return
}


// CHECK: define internal void @__omp_offloading_{{.*}}_caller__{{.*}}(ptr %[[X_PARAM:.*]], ptr %[[I_PARAM:.*]]) {

// CHECK:   %[[I_VAL:.*]] = load i32, ptr %[[I_PARAM]], align 4
// CHECK:   %[[I_VAL_FL:.*]] = sitofp i32 %[[I_VAL]] to float
// CHECK:   store float %[[I_VAL_FL]], ptr %[[X_PARAM]], align 4
// CHECK:   call void @callee_(ptr %[[X_PARAM]], ptr %[[X_PARAM]])
// CHECK:   br label %[[REGION_CONT:.*]]

// CHECK: [[REGION_CONT]]:
// CHECK:   ret void
// CHECK: }
