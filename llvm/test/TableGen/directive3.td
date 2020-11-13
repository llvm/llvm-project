// RUN: not llvm-tblgen -gen-directive-decl -I %p/../../include %s 2>&1 | FileCheck -match-full-lines %s
// RUN: not llvm-tblgen -gen-directive-impl -I %p/../../include %s 2>&1 | FileCheck -match-full-lines %s
// RUN: not llvm-tblgen -gen-directive-gen -I %p/../../include %s 2>&1 | FileCheck -match-full-lines %s

include "llvm/Frontend/Directive/DirectiveBase.td"

def TestDirectiveLanguage : DirectiveLanguage {
  let name = "TdlError";
}

def TDLC_ClauseA : Clause<"clausea"> {
  let isDefault = 1;
}

def TDLC_ClauseB : Clause<"clauseb"> {
}

def TDLC_ClauseC : Clause<"clausec"> {
}

def TDLC_ClauseD : Clause<"claused"> {
}

def TDL_DirA : Directive<"dira"> {
  let allowedClauses = [
    VersionedClause<TDLC_ClauseA>,
    VersionedClause<TDLC_ClauseB>,
    VersionedClause<TDLC_ClauseD>
  ];
  let allowedOnceClauses = [
    VersionedClause<TDLC_ClauseA>,
    VersionedClause<TDLC_ClauseC>
  ];
  let requiredClauses = [
    VersionedClause<TDLC_ClauseC>,
    VersionedClause<TDLC_ClauseD>
  ];
  let isDefault = 1;
}

// CHECK: error: Clause TDLC_ClauseA already defined on directive TDL_DirA
// CHECK: error: Clause TDLC_ClauseD already defined on directive TDL_DirA
// CHECK: error: One or more clauses are defined multiple times on directive TDL_DirA
