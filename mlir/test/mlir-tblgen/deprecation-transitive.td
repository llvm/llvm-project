// RUN: mlir-tblgen -I %S/../../include %s 2>&1 | FileCheck %s --implicit-check-not warning:

include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test_dialect";
  let useFoldAPI = kEmitFoldAdaptorFolder;
}

def OpTraitA : NativeOpTrait<"OpTraitA">, Deprecated<"use `bar` instead">;

// CHECK: :[[# @LINE + 1]]:{{[0-9]+}}: warning: Using deprecated def `OpTraitA`
def A_AsField {
  NativeOpTrait value = OpTraitA;
}

// CHECK: :[[# @LINE + 1]]:{{[0-9]+}}: warning: Using deprecated def `OpTraitA`
def B_InList {
  list<NativeOpTrait> value = [OpTraitA];
}

// CHECK: :[[# @LINE + 1]]:{{[0-9]+}}: warning: Using deprecated def `OpTraitA`
def C_InListList {
    list<list<NativeOpTrait>> value = [[OpTraitA]];
}

class Base;

// CHECK: :[[# @LINE + 1]]:{{[0-9]+}}: warning: Using deprecated def `OpTraitA`
def D_InDagAsOperator {
  dag value = (OpTraitA $test);
}

// CHECK: :[[# @LINE + 1]]:{{[0-9]+}}: warning: Using deprecated def `OpTraitA`
def E_InDagAsArg : Base {
  dag value = (ins OpTraitA:$test);
}

class ThingTakingList<list<NativeOpTrait> l> {
  list<NativeOpTrait> i = l; // avoid unused variable warning.
}

// CHECK: :[[# @LINE + 1]]:{{[0-9]+}}: warning: Using deprecated def `OpTraitA`
def F_AsFieldWithinAnonymousDef {
  ThingTakingList value = ThingTakingList<[OpTraitA]>;
}

// CHECK: :[[# @LINE + 1]]:{{[0-9]+}}: warning: Using deprecated def `OpTraitA`
def G_InDagAsAnonymousDefOperator {
  dag value = (ThingTakingList<[OpTraitA]> $test);
}

def H_ButNotTransitivelyInNonAnonymousDef {
  Base value = E_InDagAsArg;
}
