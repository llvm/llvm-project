// RUN: mlir-tblgen -gen-pass-doc -I %S/../../include -dialect=test %s | FileCheck %s

include "mlir/Pass/PassBase.td"

def TestPassDocA : Pass<"test-pass-doc-a"> {
  let summary = "pass summary";
  let description = [{
    Pass description
  }];

  let options = [
    ListOption<"option", "option", "std::string", "pass option">
  ];
}

def TestPassDocB : Pass<"test-pass-doc-b"> {
}

// Ensure there are empty lines between individual pass docs.

// CHECK:      `-test-pass-doc-a`
// CHECK-EMPTY:
// CHECK-NEXT:  _Pass summary_
// CHECK-EMPTY:
// CHECK-NEXT:  Pass description
// CHECK-EMPTY:
// CHECK-NEXT:  Options
// CHECK-EMPTY:
// CHECK-NEXT:  ```
// CHECK-NEXT:  -option : pass option
// CHECK-NEXT:  ```
// CHECK-EMPTY:
// CHECK-NEXT:  `-test-pass-doc-b`
