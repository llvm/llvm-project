// RUN: not mlir-tblgen -gen-typedef-defs -I %S/../../include %s 2>&1 | FileCheck %s -DFILE=%s

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "TestDialect";
  let cppNamespace = "::test";
}

class InvalidType<string name> : TypeDef<Test_Dialect, name> {
  let mnemonic = ?;
}

// This definition should not generate an error due to the use in `InvalidTypeA`
def TestParameter : TypeParameter<"int", "int parameter">;

// Test builder uses wrong record class.
// CHECK: [[FILE]]:[[@LINE+1]]:5: error: Initializer of 'typeName' in 'InvalidTypeA' could not be fully resolved: !strconcat("TestDialect", !strconcat(".", mnemonic))
def InvalidTypeA : InvalidType<"InvalidTypeA"> {
  let parameters = (ins "int":$v0);
  let builders = [
    TypeBuilder<(ins TestParameter:$arg0), [{
      return $_get($_ctxt, arg0);
    }]>
  ];
}

