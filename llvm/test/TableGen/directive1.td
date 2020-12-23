// RUN: llvm-tblgen -gen-directive-decl -I %p/../../include %s | FileCheck -match-full-lines %s
// RUN: llvm-tblgen -gen-directive-impl -I %p/../../include %s | FileCheck -match-full-lines %s -check-prefix=IMPL
// RUN: llvm-tblgen -gen-directive-gen -I %p/../../include %s | FileCheck -match-full-lines %s -check-prefix=GEN

include "llvm/Frontend/Directive/DirectiveBase.td"

def TestDirectiveLanguage : DirectiveLanguage {
  let name = "Tdl";

  let cppNamespace = "tdl";
  let directivePrefix = "TDLD_";
  let clausePrefix = "TDLC_";
  let makeEnumAvailableInNamespace = 1;
  let enableBitmaskEnumInNamespace = 1;
  let flangClauseBaseClass = "TdlClause";
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

def TDLC_ClauseB : Clause<"clauseb"> {
  let flangClassValue = "IntExpr";
  let isValueOptional = 1;
  let isDefault = 1;
}

def TDL_DirA : Directive<"dira"> {
  let allowedClauses = [
    VersionedClause<TDLC_ClauseA>,
    VersionedClause<TDLC_ClauseB>
  ];
  let isDefault = 1;
}

// CHECK:       #ifndef LLVM_Tdl_INC
// CHECK-NEXT:  #define LLVM_Tdl_INC
// CHECK-EMPTY:
// CHECK-NEXT:  #include "llvm/ADT/BitmaskEnum.h"
// CHECK-EMPTY:
// CHECK-NEXT:  namespace llvm {
// CHECK-NEXT:  class StringRef;
// CHECK-NEXT:  namespace tdl {
// CHECK-EMPTY:
// CHECK-NEXT:  LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();
// CHECK-EMPTY:
// CHECK-NEXT:  enum class Directive {
// CHECK-NEXT:    TDLD_dira,
// CHECK-NEXT:  };
// CHECK-EMPTY:
// CHECK-NEXT:  static constexpr std::size_t Directive_enumSize = 1;
// CHECK-EMPTY:
// CHECK-NEXT:  constexpr auto TDLD_dira = llvm::tdl::Directive::TDLD_dira;
// CHECK-EMPTY:
// CHECK-NEXT:  enum class Clause {
// CHECK-NEXT:    TDLC_clausea,
// CHECK-NEXT:    TDLC_clauseb,
// CHECK-NEXT:  };
// CHECK-EMPTY:
// CHECK-NEXT:  static constexpr std::size_t Clause_enumSize = 2;
// CHECK-EMPTY:
// CHECK-NEXT:  constexpr auto TDLC_clausea = llvm::tdl::Clause::TDLC_clausea;
// CHECK-NEXT:  constexpr auto TDLC_clauseb = llvm::tdl::Clause::TDLC_clauseb;
// CHECK-EMPTY:
// CHECK-NEXT:  enum class AKind {
// CHECK-NEXT:    TDLCV_vala=1,
// CHECK-NEXT:    TDLCV_valb=2,
// CHECK-NEXT:    TDLCV_valc=3,
// CHECK-NEXT:  };
// CHECK-EMPTY:
// CHECK-NEXT:  constexpr auto TDLCV_vala = llvm::tdl::AKind::TDLCV_vala;
// CHECK-NEXT:  constexpr auto TDLCV_valb = llvm::tdl::AKind::TDLCV_valb;
// CHECK-NEXT:  constexpr auto TDLCV_valc = llvm::tdl::AKind::TDLCV_valc;
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
// CHECK-NEXT:  AKind getAKind(StringRef);
// CHECK-NEXT:  llvm::StringRef getTdlAKindName(AKind);
// CHECK-EMPTY:
// CHECK-NEXT:  } // namespace tdl
// CHECK-NEXT:  } // namespace llvm
// CHECK-NEXT:  #endif // LLVM_Tdl_INC


// IMPL:       #include "llvm/ADT/StringRef.h"
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
// IMPL-NEXT:      .Case("clausea",TDLC_clausea)
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
// IMPL-NEXT:  AKind llvm::tdl::getAKind(llvm::StringRef Str) {
// IMPL-NEXT:    return llvm::StringSwitch<AKind>(Str)
// IMPL-NEXT:      .Case("vala",TDLCV_vala)
// IMPL-NEXT:      .Case("valb",TDLCV_valb)
// IMPL-NEXT:      .Case("valc",TDLCV_valc)
// IMPL-NEXT:      .Default(TDLCV_valc);
// IMPL-NEXT:  }
// IMPL-EMPTY:
// IMPL-NEXT:  llvm::StringRef llvm::tdl::getTdlAKindName(llvm::tdl::AKind x) {
// IMPL-NEXT:    switch (x) {
// IMPL-NEXT:      case TDLCV_vala:
// IMPL-NEXT:        return "vala";
// IMPL-NEXT:      case TDLCV_valb:
// IMPL-NEXT:        return "valb";
// IMPL-NEXT:      case TDLCV_valc:
// IMPL-NEXT:        return "valc";
// IMPL-NEXT:    }
// IMPL-NEXT:    llvm_unreachable("Invalid Tdl AKind kind");
// IMPL-NEXT:  }
// IMPL-EMPTY:
// IMPL-NEXT:  bool llvm::tdl::isAllowedClauseForDirective(Directive D, Clause C, unsigned Version) {
// IMPL-NEXT:    assert(unsigned(D) <= llvm::tdl::Directive_enumSize);
// IMPL-NEXT:    assert(unsigned(C) <= llvm::tdl::Clause_enumSize);
// IMPL-NEXT:    switch (D) {
// IMPL-NEXT:      case TDLD_dira:
// IMPL-NEXT:        switch (C) {
// IMPL-NEXT:          case TDLC_clausea:
// IMPL-NEXT:            return 1 <= Version && 2147483647 >= Version;
// IMPL-NEXT:          case TDLC_clauseb:
// IMPL-NEXT:            return 1 <= Version && 2147483647 >= Version;
// IMPL-NEXT:          default:
// IMPL-NEXT:            return false;
// IMPL-NEXT:        }
// IMPL-NEXT:        break;
// IMPL-NEXT:    }
// IMPL-NEXT:    llvm_unreachable("Invalid Tdl Directive kind");
// IMPL-NEXT:  }
// IMPL-EMPTY:



// GEN:       #ifdef GEN_FLANG_DIRECTIVE_CLAUSE_SETS
// GEN-NEXT:  #undef GEN_FLANG_DIRECTIVE_CLAUSE_SETS
// GEN-EMPTY:
// GEN-NEXT:  namespace llvm {
// GEN-NEXT:  namespace tdl {
// GEN-EMPTY:
// GEN-NEXT:    // Sets for dira
// GEN-EMPTY:
// GEN-NEXT:    static  allowedClauses_TDLD_dira {
// GEN-NEXT:      llvm::tdl::Clause::TDLC_clausea,
// GEN-NEXT:      llvm::tdl::Clause::TDLC_clauseb,
// GEN-NEXT:    };
// GEN-EMPTY:
// GEN-NEXT:    static  allowedOnceClauses_TDLD_dira {
// GEN-NEXT:    };
// GEN-EMPTY:
// GEN-NEXT:    static  allowedExclusiveClauses_TDLD_dira {
// GEN-NEXT:    };
// GEN-EMPTY:
// GEN-NEXT:    static  requiredClauses_TDLD_dira {
// GEN-NEXT:    };
// GEN-NEXT:  } // namespace tdl
// GEN-NEXT:  } // namespace llvm
// GEN-EMPTY:
// GEN-NEXT:  #endif // GEN_FLANG_DIRECTIVE_CLAUSE_SETS
// GEN-EMPTY:
// GEN-NEXT:  #ifdef GEN_FLANG_DIRECTIVE_CLAUSE_MAP
// GEN-NEXT:  #undef GEN_FLANG_DIRECTIVE_CLAUSE_MAP
// GEN-EMPTY:
// GEN-NEXT:  {
// GEN-NEXT:    {llvm::tdl::Directive::TDLD_dira,
// GEN-NEXT:      {
// GEN-NEXT:        llvm::tdl::allowedClauses_TDLD_dira,
// GEN-NEXT:        llvm::tdl::allowedOnceClauses_TDLD_dira,
// GEN-NEXT:        llvm::tdl::allowedExclusiveClauses_TDLD_dira,
// GEN-NEXT:        llvm::tdl::requiredClauses_TDLD_dira,
// GEN-NEXT:      }
// GEN-NEXT:    },
// GEN-NEXT:  }
// GEN-EMPTY:
// GEN-NEXT:  #endif // GEN_FLANG_DIRECTIVE_CLAUSE_MAP
// GEN-EMPTY:
// GEN-NEXT:  #ifdef GEN_FLANG_CLAUSE_PARSER_CLASSES
// GEN-NEXT:  #undef GEN_FLANG_CLAUSE_PARSER_CLASSES
// GEN-EMPTY:
// GEN-NEXT:  WRAPPER_CLASS(Clauseb, std::optional<IntExpr>);
// GEN-EMPTY:
// GEN-NEXT:  #endif // GEN_FLANG_CLAUSE_PARSER_CLASSES
// GEN-EMPTY:
// GEN-NEXT:  #ifdef GEN_FLANG_CLAUSE_PARSER_CLASSES_LIST
// GEN-NEXT:  #undef GEN_FLANG_CLAUSE_PARSER_CLASSES_LIST
// GEN-EMPTY:
// GEN-NEXT:  TdlClauseA
// GEN-NEXT:  , Clauseb
// GEN-EMPTY:
// GEN-NEXT:  #endif // GEN_FLANG_CLAUSE_PARSER_CLASSES_LIST
// GEN-EMPTY:
// GEN-NEXT:  #ifdef GEN_FLANG_DUMP_PARSE_TREE_CLAUSES
// GEN-NEXT:  #undef GEN_FLANG_DUMP_PARSE_TREE_CLAUSES
// GEN-EMPTY:
// GEN-NEXT:  NODE(TdlClause, Clauseb)
// GEN-EMPTY:
// GEN-NEXT:  #endif // GEN_FLANG_DUMP_PARSE_TREE_CLAUSES
// GEN-EMPTY:
// GEN-NEXT:  #ifdef GEN_FLANG_CLAUSE_UNPARSE
// GEN-NEXT:  #undef GEN_FLANG_CLAUSE_UNPARSE
// GEN-EMPTY:
// GEN-NEXT:  void Unparse(const TdlClause::Clauseb &x) {
// GEN-NEXT:    Word("CLAUSEB");
// GEN-NEXT:    Walk("(", x.v, ")");
// GEN-NEXT:  }
// GEN-EMPTY:
// GEN-NEXT:  #endif // GEN_FLANG_CLAUSE_UNPARSE
