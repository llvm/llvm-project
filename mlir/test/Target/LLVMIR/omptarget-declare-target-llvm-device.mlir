// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// This test the generation of additional load operations for declare target link variables
// inside of target op regions when lowering to IR for device. Unfortunately as the host file is not
// passed as a module attribute, we miss out on the metadata and entryinfo.
//
// Unfortunately, only so much can be tested as the device side is dependent on a *.bc
// file created by the host and appended as an attribute to the module.

module attributes {omp.is_target_device = true} {
  // CHECK-DAG: @_QMtest_0Esp_decl_tgt_ref_ptr = weak global ptr null, align 8
  llvm.mlir.global external @_QMtest_0Esp() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} : i32 {
    %0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %0 : i32
  }

  llvm.func @_QQmain() attributes {} {
    %0 = llvm.mlir.addressof @_QMtest_0Esp : !llvm.ptr

  // CHECK-DAG:   omp.target:                                       ; preds = %user_code.entry
  // CHECK-DAG: %[[V:.*]] = load ptr, ptr @_QMtest_0Esp_decl_tgt_ref_ptr, align 8
  // CHECK-DAG: store i32 1, ptr %[[V]], align 4
  // CHECK-DAG: br label %omp.region.cont
    %map = omp.map.info var_ptr(%0 : !llvm.ptr, i32)   map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
    omp.target   map_entries(%map -> %arg0 : !llvm.ptr) {
      %1 = llvm.mlir.constant(1 : i32) : i32
      llvm.store %1, %arg0 : i32, !llvm.ptr
      omp.terminator
    }

    llvm.return
  }
}
