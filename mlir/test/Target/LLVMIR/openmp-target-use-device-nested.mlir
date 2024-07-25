// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// This tests check that target code nested inside a target data region which
// has only use_device_ptr mapping corectly generates code on the device pass.

// CHECK-NOT: call void @__tgt_target_data_begin_mapper
// CHECK: store i32 999, ptr {{.*}}
module attributes {omp.is_target_device = true } {
  llvm.func @_QQmain() attributes {fir.bindc_name = "main"} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %a = llvm.alloca %0 x !llvm.ptr : (i64) -> !llvm.ptr
    %map = omp.map.info var_ptr(%a : !llvm.ptr, !llvm.ptr)   map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
    omp.target_data use_device_ptr(%map : !llvm.ptr)  {
    ^bb0(%arg0: !llvm.ptr):
      %map1 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.ptr)   map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
      omp.target map_entries(%map1 : !llvm.ptr){
      ^bb0(%arg1: !llvm.ptr):
        %1 = llvm.mlir.constant(999 : i32) : i32
        %2 = llvm.load %arg1 : !llvm.ptr -> !llvm.ptr
        llvm.store %1, %2 : i32, !llvm.ptr
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }
}
