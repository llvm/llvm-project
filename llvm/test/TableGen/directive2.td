// RUN: llvm-tblgen -gen-directive-decls -I %p/../../include %s | FileCheck %s

include "llvm/Frontend/Directive/DirectiveBase.td"

def TestDirectiveLanguage : DirectiveLanguage {
  let name = "tdl";

  let cppNamespace = "tdl";
  let directivePrefix = "TDLD_";
  let clausePrefix = "TDLC_";
}

def TDLC_ClauseA : Clause<"clausea"> {}
def TDLC_ClauseB : Clause<"clauseb"> {}

def TDL_DirA : Directive<"dira"> {
  let allowedClauses = [TDLC_ClauseA, TDLC_ClauseB];
}

// CHECK: #ifndef LLVM_tdl_INC
// CHECK-NEXT: #define LLVM_tdl_INC
// CHECK-NEXT: namespace llvm {
// CHECK-NEXT: namespace tdl {
// CHECK-NEXT: enum class Directive {
// CHECK-NEXT: TDLD_dira,
// CHECK-NEXT: }
// CHECK-NEXT: static constexpr std::size_t Directive_enumSize = 1;
// CHECK-NEXT: enum class Clause {
// CHECK-NEXT: TDLC_clausea,
// CHECK-NEXT: TDLC_clauseb,
// CHECK-NEXT: }
// CHECK-NEXT: static constexpr std::size_t Clause_enumSize = 2;
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: #endif
