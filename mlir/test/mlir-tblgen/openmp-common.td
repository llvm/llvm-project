// RUN: mlir-tblgen -gen-directive-decl -I %S/../../../llvm/include %s | FileCheck -match-full-lines %s

include "llvm/Frontend/Directive/DirectiveBase.td"

def TDLCV_vala : ClauseVal<"vala",1,1> {}
def TDLCV_valb : ClauseVal<"valb",2,1> {}
def TDLCV_valc : ClauseVal<"valc",3,0> { let isDefault = 1; }

def TDLC_ClauseA : Clause<"clausea"> {
  let flangClass = "TdlClauseA";
  let enumClauseValue = "AKind";
  let allowedClauseValues = [
    TDLCV_vala,
    TDLCV_valb,
    TDLCV_valc
  ];
}

// CHECK: def AKindvala : StrEnumAttrCase<"vala">;
// CHECK: def AKindvalb : StrEnumAttrCase<"valb">;
// CHECK: def AKind: StrEnumAttr<
// CHECK:   "ClauseAKind",
// CHECK:   "AKind Clause",
// CHECK:   [AKindvala,AKindvalb]> {
// CHECK:     let cppNamespace = "::mlir::omp";
// CHECK: }
