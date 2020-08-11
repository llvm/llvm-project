// RUN: llvm-tblgen -gen-directive-decl -I %p/../../include %s | FileCheck -match-full-lines %s
// RUN: llvm-tblgen -gen-directive-impl -I %p/../../include %s | FileCheck -match-full-lines %s -check-prefix=IMPL
// RUN: llvm-tblgen -gen-directive-gen -I %p/../../include %s | FileCheck -match-full-lines %s -check-prefix=GEN

include "llvm/Frontend/Directive/DirectiveBase.td"

def TestDirectiveLanguage : DirectiveLanguage {
  let name = "Tdl";

  let cppNamespace = "tdl";
  let directivePrefix = "TDLD_";
  let clausePrefix = "TDLC_";
  let includeHeader = "tdl.h.inc";
}

def TDLC_ClauseA : Clause<"clausea"> {
  let isImplicit = 1;
}
def TDLC_ClauseB : Clause<"clauseb"> {
  let isDefault = 1;
}

def TDL_DirA : Directive<"dira"> {
  let allowedClauses = [
    VersionedClause<TDLC_ClauseA, 2, 4>,
    VersionedClause<TDLC_ClauseB, 2>
  ];
  let isDefault = 1;
}

// CHECK:       #ifndef LLVM_Tdl_INC
// CHECK-NEXT:  #define LLVM_Tdl_INC
// CHECK-EMPTY:
// CHECK-NEXT:  namespace llvm {
// CHECK-NEXT:  class StringRef;
// CHECK-NEXT:  namespace tdl {
// CHECK-EMPTY:
// CHECK-NEXT:  enum class Directive {
// CHECK-NEXT:    TDLD_dira,
// CHECK-NEXT:  };
// CHECK-EMPTY:
// CHECK-NEXT:  static constexpr std::size_t Directive_enumSize = 1;
// CHECK-EMPTY:
// CHECK-NEXT:  enum class Clause {
// CHECK-NEXT:    TDLC_clausea,
// CHECK-NEXT:    TDLC_clauseb,
// CHECK-NEXT:  };
// CHECK-EMPTY:
// CHECK-NEXT:  static constexpr std::size_t Clause_enumSize = 2;
// CHECK-EMPTY:
// CHECK-NEXT:  // Enumeration helper functions
// CHECK-NEXT:  Directive getTdlDirectiveKind(llvm::StringRef Str);
// CHECK-EMPTY:
// CHECK-NEXT:  llvm::StringRef getTdlDirectiveName(Directive D);
// CHECK-EMPTY:
// CHECK-NEXT:  Clause getTdlClauseKind(llvm::StringRef Str);
// CHECK-EMPTY:
// CHECK-NEXT:  llvm::StringRef getTdlClauseName(Clause C);
// CHECK-EMPTY:
// CHECK-NEXT:  /// Return true if \p C is a valid clause for \p D in version \p Version.
// CHECK-NEXT:  bool isAllowedClauseForDirective(Directive D, Clause C, unsigned Version);
// CHECK-EMPTY:
// CHECK-NEXT:  } // namespace tdl
// CHECK-NEXT:  } // namespace llvm
// CHECK-NEXT:  #endif // LLVM_Tdl_INC

// IMPL:       #include "tdl.h.inc"
// IMPL-EMPTY:
// IMPL-NEXT:  #include "llvm/ADT/StringRef.h"
// IMPL-NEXT:  #include "llvm/ADT/StringSwitch.h"
// IMPL-NEXT:  #include "llvm/Support/ErrorHandling.h"
// IMPL-EMPTY:
// IMPL-NEXT:  using namespace llvm;
// IMPL-NEXT:  using namespace tdl;
// IMPL-EMPTY:
// IMPL-NEXT:  Directive llvm::tdl::getTdlDirectiveKind(llvm::StringRef Str) {
// IMPL-NEXT:    return llvm::StringSwitch<Directive>(Str)
// IMPL-NEXT:      .Case("dira",TDLD_dira)
// IMPL-NEXT:      .Default(TDLD_dira);
// IMPL-NEXT:  }
// IMPL-EMPTY:
// IMPL-NEXT:  llvm::StringRef llvm::tdl::getTdlDirectiveName(Directive Kind) {
// IMPL-NEXT:    switch (Kind) {
// IMPL-NEXT:      case TDLD_dira:
// IMPL-NEXT:        return "dira";
// IMPL-NEXT:    }
// IMPL-NEXT:    llvm_unreachable("Invalid Tdl Directive kind");
// IMPL-NEXT:  }
// IMPL-EMPTY:
// IMPL-NEXT:  Clause llvm::tdl::getTdlClauseKind(llvm::StringRef Str) {
// IMPL-NEXT:    return llvm::StringSwitch<Clause>(Str)
// IMPL-NEXT:      .Case("clausea",TDLC_clauseb)
// IMPL-NEXT:      .Case("clauseb",TDLC_clauseb)
// IMPL-NEXT:      .Default(TDLC_clauseb);
// IMPL-NEXT:  }
// IMPL-EMPTY:
// IMPL-NEXT:  llvm::StringRef llvm::tdl::getTdlClauseName(Clause Kind) {
// IMPL-NEXT:    switch (Kind) {
// IMPL-NEXT:      case TDLC_clausea:
// IMPL-NEXT:        return "clausea";
// IMPL-NEXT:      case TDLC_clauseb:
// IMPL-NEXT:        return "clauseb";
// IMPL-NEXT:    }
// IMPL-NEXT:    llvm_unreachable("Invalid Tdl Clause kind");
// IMPL-NEXT:  }
// IMPL-EMPTY:
// IMPL-NEXT:  bool llvm::tdl::isAllowedClauseForDirective(Directive D, Clause C, unsigned Version) {
// IMPL-NEXT:    assert(unsigned(D) <= llvm::tdl::Directive_enumSize);
// IMPL-NEXT:    assert(unsigned(C) <= llvm::tdl::Clause_enumSize);
// IMPL-NEXT:    switch (D) {
// IMPL-NEXT:      case TDLD_dira:
// IMPL-NEXT:        switch (C) {
// IMPL-NEXT:          case TDLC_clausea:
// IMPL-NEXT:            return 2 <= Version && 4 >= Version;
// IMPL-NEXT:          case TDLC_clauseb:
// IMPL-NEXT:            return 2 <= Version && 2147483647 >= Version;
// IMPL-NEXT:          default:
// IMPL-NEXT:            return false;
// IMPL-NEXT:        }
// IMPL-NEXT:        break;
// IMPL-NEXT:    }
// IMPL-NEXT:    llvm_unreachable("Invalid Tdl Directive kind");
// IMPL-NEXT:  }


// GEN:      #ifdef GEN_FLANG_DIRECTIVE_CLAUSE_SETS
// GEN-NEXT: #undef GEN_FLANG_DIRECTIVE_CLAUSE_SETS
// GEN-EMPTY:
// GEN-NEXT: namespace llvm {
// GEN-NEXT: namespace tdl {
// GEN-EMPTY:
// GEN-NEXT:   // Sets for dira
// GEN-EMPTY:
// GEN-NEXT:   static  allowedClauses_TDLD_dira {
// GEN-NEXT:     llvm::tdl::Clause::TDLC_clausea,
// GEN-NEXT:     llvm::tdl::Clause::TDLC_clauseb,
// GEN-NEXT:   };
// GEN-EMPTY:
// GEN-NEXT:   static  allowedOnceClauses_TDLD_dira {
// GEN-NEXT:   };
// GEN-EMPTY:
// GEN-NEXT:   static  allowedExclusiveClauses_TDLD_dira {
// GEN-NEXT:   };
// GEN-EMPTY:
// GEN-NEXT:   static  requiredClauses_TDLD_dira {
// GEN-NEXT:   };
// GEN-NEXT: } // namespace tdl
// GEN-NEXT: } // namespace llvm
// GEN-EMPTY:
// GEN-NEXT: #endif // GEN_FLANG_DIRECTIVE_CLAUSE_SETS
// GEN-EMPTY:
// GEN-NEXT: #ifdef GEN_FLANG_DIRECTIVE_CLAUSE_MAP
// GEN-NEXT: #undef GEN_FLANG_DIRECTIVE_CLAUSE_MAP
// GEN-EMPTY:
// GEN-NEXT: {
// GEN-NEXT:   {llvm::tdl::Directive::TDLD_dira,
// GEN-NEXT:     {
// GEN-NEXT:       llvm::tdl::allowedClauses_TDLD_dira,
// GEN-NEXT:       llvm::tdl::allowedOnceClauses_TDLD_dira,
// GEN-NEXT:       llvm::tdl::allowedExclusiveClauses_TDLD_dira,
// GEN-NEXT:       llvm::tdl::requiredClauses_TDLD_dira,
// GEN-NEXT:     }
// GEN-NEXT:   },
// GEN-NEXT: }
// GEN-EMPTY:
// GEN-NEXT: #endif // GEN_FLANG_DIRECTIVE_CLAUSE_MAP
