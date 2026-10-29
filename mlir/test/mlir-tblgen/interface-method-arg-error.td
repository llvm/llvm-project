// RUN: not mlir-tblgen -gen-op-interface-decls -I %S/../../include %s 2>&1 | FileCheck %s

// Test that a non-string argument type in an interface method gives a readable
// error instead of a crash. See https://github.com/llvm/llvm-project/issues/61869

include "mlir/IR/OpBase.td"

def Foo {}

// CHECK: error: expected string type for interface method argument #0 ('arg')
def Bar : OpInterface<"Bar"> {
  let methods = [InterfaceMethod<[{}], "void", "baz", (ins Foo:$arg)>];
}
