// RUN: not mlir-translate -mlir-to-llvmir -split-input-file %s 2>&1 | FileCheck %s

llvm.func @_QPopenmp_target_data_update() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "i", in_type = i32, operand_segment_sizes = array<i32: 0, 0>, uniq_name = "_QFopenmp_target_dataEi"} : (i64) -> !llvm.ptr
  %2 = omp.map.info var_ptr(%1 : !llvm.ptr, i32)   map_clauses(to) capture(ByRef) -> !llvm.ptr {name = ""}

  // CHECK: error: `nowait` is not supported yet
  omp.target_update map_entries(%2 : !llvm.ptr) nowait

  llvm.return
}

// -----

llvm.func @_QPopenmp_target_data_enter() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "i", in_type = i32, operand_segment_sizes = array<i32: 0, 0>, uniq_name = "_QFopenmp_target_dataEi"} : (i64) -> !llvm.ptr
  %2 = omp.map.info var_ptr(%1 : !llvm.ptr, i32)   map_clauses(to) capture(ByRef) -> !llvm.ptr {name = ""}

  // CHECK: error: `nowait` is not supported yet
  omp.target_enter_data map_entries(%2 : !llvm.ptr) nowait

  llvm.return
}


// -----

llvm.func @_QPopenmp_target_data_exit() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "i", in_type = i32, operand_segment_sizes = array<i32: 0, 0>, uniq_name = "_QFopenmp_target_dataEi"} : (i64) -> !llvm.ptr
  %2 = omp.map.info var_ptr(%1 : !llvm.ptr, i32)   map_clauses(from) capture(ByRef) -> !llvm.ptr {name = ""}

  // CHECK: error: `nowait` is not supported yet
  omp.target_exit_data map_entries(%2 : !llvm.ptr) nowait

  llvm.return
}
