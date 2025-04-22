// RUN: not mlir-tblgen -gen-attrdef-decls -I %S/../../include %s 2>&1 | FileCheck %s

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def Test_Dialect: Dialect {
  let name = "TestDialect";
  let cppNamespace = "::test";
}

def RawAPIntAttr : AttrDef<Test_Dialect, "RawAPInt"> {
  let mnemonic = "raw_ap_int";
  let parameters = (ins "APInt":$value);
  let hasCustomAssemblyFormat = 1;
}

// CHECK: apint-param-error.td:11:5: error: Using a raw APInt parameter
