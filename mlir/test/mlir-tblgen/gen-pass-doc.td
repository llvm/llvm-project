// RUN: mlir-tblgen -gen-pass-doc -I %S/../../include -dialect=test %s | FileCheck %s

include "mlir/Pass/PassBase.td"

def TestPassDoc : Pass<"test-pass-doc"> {
  let summary = "pass summary";
  let description = [{
    Pass description
  }];

  let options = [
    ListOption<"option", "option", "std::string", "pass option">
  ];
}

// CHECK: `-test-pass-doc`
// CHECK: _Pass summary_
// CHECK: Pass description
// CHECK: Options
// CHECK: -option : pass option
