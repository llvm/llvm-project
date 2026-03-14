// RUN: not mlir-tblgen -on-deprecated=error -gen-op-decls -I %S/../../include -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s

include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test_dialect";
}

#ifdef ERROR1
def OpTraitA : NativeOpTrait<"OpTraitA">, Deprecated<"use `bar` instead">;

// ERROR1: warning: Using deprecated def `OpTraitA`
// ERROR1: use `bar` instead
def OpTraitWithoutDependentTrait : Op<Test_Dialect, "default_value", [OpTraitA]> {}
#endif
