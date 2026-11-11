// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s
// RUN: mlir-tblgen -gen-op-defs -op-exclude-regex='test.canonicalize' -I %S/../../include %s | FileCheck %s --check-prefix=NOCANON

include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "::test";
}

def OpTraitA : NativeOpTrait<"OpTraitA"> {
  let cppIncludes = ["test/OpTraitA.h"];
}

// Op definitions include the headers they need beyond the declarations, but
// not the headers of the traits, which are included by the declarations. A
// synthesized `getCanonicalizationPatterns` additionally needs the pattern
// header, which is otherwise not included.

// CHECK-LABEL: #ifdef GET_OP_CLASSES
// CHECK-NEXT: #undef GET_OP_CLASSES
// CHECK-EMPTY:
// CHECK-NEXT: #include "mlir/IR/Builders.h"
// CHECK-NEXT: #include "mlir/IR/BuiltinAttributes.h"
// CHECK-NEXT: #include "mlir/IR/BuiltinTypes.h"
// CHECK-NEXT: #include "mlir/IR/OpImplementation.h"
// CHECK-NEXT: #include "mlir/IR/PatternMatch.h"
// CHECK-EMPTY:
// CHECK-NOT: #include

// NOCANON-LABEL: #ifdef GET_OP_CLASSES
// NOCANON-NEXT: #undef GET_OP_CLASSES
// NOCANON-EMPTY:
// NOCANON-NEXT: #include "mlir/IR/Builders.h"
// NOCANON-NEXT: #include "mlir/IR/BuiltinAttributes.h"
// NOCANON-NEXT: #include "mlir/IR/BuiltinTypes.h"
// NOCANON-NEXT: #include "mlir/IR/OpImplementation.h"
// NOCANON-EMPTY:
// NOCANON-NOT: #include

def TraitOp : Op<Test_Dialect, "trait", [OpTraitA]>;

def CanonicalizeOp : Op<Test_Dialect, "canonicalize"> {
  let hasCanonicalizeMethod = 1;
}
