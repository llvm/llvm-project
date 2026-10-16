// RUN: mlir-opt %s -test-dlti-query-op-interface --mlir-print-op-generic | FileCheck %s
// RUN: mlir-opt %s -emit-bytecode | mlir-opt --mlir-print-op-generic | FileCheck --check-prefix=BYTECODE %s

// BYTECODE: "builtin.module"() <{dlti = #dlti.map<"module" = 1 : i32>}> ({
module attributes {dlti = #dlti.map<"module" = 1 : i32>} {
  // CHECK: "test.op_with_data_layout"() <{dlti = #dlti.map<"replaced" = 2 : i32, i32 = 32 : i32, "inserted" = 1 : i32>}> ({
  // CHECK-NOT: test.dlti_action
  "test.op_with_data_layout"() ({
  }) {dlti = #dlti.map<"replaced" = 0 : i32,
                       "removed" = 0 : i32,
                       i32 = 0 : i32>,
      test.dlti_action = "update"} : () -> ()

  // CHECK: "test.op_with_data_layout"() ({
  // CHECK-NOT: dlti =
  // CHECK-NOT: test.dlti_action
  "test.op_with_data_layout"() ({
  }) {dlti = #dlti.map<"only" = 0 : i32>,
      test.dlti_action = "clear"} : () -> ()

  // CHECK: "test.op_with_data_layout"() <{dlti = #dlti.dl_spec<"immutable" = 0 : i32>}> ({
  // CHECK: }) {test.set_dlti_failed}
  "test.op_with_data_layout"() ({
  }) {dlti = #dlti.dl_spec<"immutable" = 0 : i32>,
      test.dlti_action = "fail"} : () -> ()

  // CHECK: "test.op_with_data_layout"() <{dlti = #dlti.map<"kept" = 0 : i32>}> ({
  // CHECK-NOT: test.dlti_action
  "test.op_with_data_layout"() ({
  }) {dlti = #dlti.map<"kept" = 0 : i32>,
      test.dlti_action = "invalid"} : () -> ()
}
