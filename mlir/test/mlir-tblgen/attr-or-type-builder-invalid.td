// RUN: not mlir-tblgen -gen-typedef-defs -I %S/../../include %s 2>&1 | FileCheck %s

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
// CHECK-NOT: Record `TestParameter' does not have a field named `type'!
def TestParameter : TypeParameter<"int", "int parameter">;

// Test builder uses wrong record class.
def InvalidTypeA : InvalidType<"InvalidTypeA"> {
  let parameters = (ins "int":$v0);
  let builders = [
    // CHECK: Builder DAG arguments must be either strings or defs which inherit from CArg
    TypeBuilder<(ins TestParameter:$arg0), [{
      return $_get($_ctxt, arg0);
    }]>
  ];
}

