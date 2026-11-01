// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-DAG: c";x;test.f90;3;5;;\00"
llvm.func @map_name_from_attr(%a : !llvm.ptr) {
  %m = omp.map.info var_ptr(%a : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) name("x") -> !llvm.ptr loc("test.f90":3:5)
  omp.target_data map_entries(%m : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// CHECK-DAG: c";y;test.f90;4;6;;\00"
llvm.func @map_name_from_nameloc(%a : !llvm.ptr) {
  %m = omp.map.info var_ptr(%a : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) name("") -> !llvm.ptr loc("y"("test.f90":4:6))
  omp.target_data map_entries(%m : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// CHECK-DAG: c";unknown;test.f90;5;7;;\00"
llvm.func @map_name_unknown(%a : !llvm.ptr) {
  %m = omp.map.info var_ptr(%a : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) name("") -> !llvm.ptr loc("test.f90":5:7)
  omp.target_data map_entries(%m : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}
