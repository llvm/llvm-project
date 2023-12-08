// RUN: mlir-tblgen -gen-directive-decl -directives-dialect=TDL -I %S/../../../llvm/include %s | FileCheck -match-full-lines %s

include "llvm/Frontend/Directive/DirectiveBase.td"

def TestDirectiveLanguage : DirectiveLanguage {
  let name = "Tdl";
  let cppNamespace = "tdl";
}

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

// CHECK: def AKindvala : I32EnumAttrCase<"Vala", 0, "vala">;
// CHECK: def AKindvalb : I32EnumAttrCase<"Valb", 1, "valb">;
// CHECK: def AKind: I32EnumAttr<
// CHECK:   "ClauseAKind",
// CHECK:   "AKind Clause",
// CHECK:   [AKindvala,AKindvalb]> {
// CHECK:     let cppNamespace = "::mlir::tdl";
// CHECK: }
// CHECK: def AKindAttr : EnumAttr<TDL_Dialect, AKind, "akind">;
