//===- DirectiveEmitter.cpp - Directive Language Emitter ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DirectiveEmitter uses the descriptions of directives and clauses to construct
// common code declarations to be used in Frontends.
//
//===----------------------------------------------------------------------===//

#include "llvm/TableGen/DirectiveEmitter.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#include <numeric>
#include <vector>

using namespace llvm;

namespace {
// Simple RAII helper for defining ifdef-undef-endif scopes.
class IfDefScope {
public:
  IfDefScope(StringRef Name, raw_ostream &OS) : Name(Name), OS(OS) {
    OS << "#ifdef " << Name << "\n"
       << "#undef " << Name << "\n";
  }

  ~IfDefScope() { OS << "\n#endif // " << Name << "\n\n"; }

private:
  StringRef Name;
  raw_ostream &OS;
};
} // namespace

namespace {
enum class Frontend { LLVM, Flang, Clang };

StringRef getFESpelling(Frontend FE) {
  switch (FE) {
  case Frontend::LLVM:
    return "llvm";
  case Frontend::Flang:
    return "flang";
  case Frontend::Clang:
    return "clang";
  }
  llvm_unreachable("unknown FE kind");
}
} // namespace

// Get the full namespace qualifier for the directive language.
static std::string getQualifier(const DirectiveLanguage &DirLang,
                                Frontend FE = Frontend::LLVM) {
  return (Twine(getFESpelling(FE)) + "::" + DirLang.getCppNamespace().str() +
          "::")
      .str();
}

// Get prefixed formatted name, e.g. for "target data", get "OMPD_target_data".
// This should work for any Record as long as BaseRecord::getFormattedName
// works.
static std::string getIdentifierName(const Record *Rec, StringRef Prefix) {
  return Prefix.str() + BaseRecord(Rec).getFormattedName();
}

static void generateEnumExports(ArrayRef<const Record *> Records,
                                raw_ostream &OS, StringRef Enum,
                                StringRef Prefix) {
  for (const Record *R : Records) {
    std::string N = getIdentifierName(R, Prefix);
    OS << "constexpr auto " << N << " = " << Enum << "::" << N << ";\n";
  }
}

// Generate enum class. Entries are emitted in the order in which they appear
// in the `Records` vector.
static void generateEnumClass(ArrayRef<const Record *> Records, raw_ostream &OS,
                              StringRef Enum, StringRef Prefix,
                              bool ExportEnums) {
  OS << "\n";
  OS << "enum class " << Enum << " {\n";
  for (const Record *R : Records) {
    OS << "  " << getIdentifierName(R, Prefix) << ",\n";
  }
  OS << "};\n";
  OS << "\n";
  OS << "static constexpr std::size_t " << Enum
     << "_enumSize = " << Records.size() << ";\n";

  // Make the enum values available in the defined namespace. This allows us to
  // write something like Enum_X if we have a `using namespace <CppNamespace>`.
  // At the same time we do not loose the strong type guarantees of the enum
  // class, that is we cannot pass an unsigned as Directive without an explicit
  // cast.
  if (ExportEnums) {
    OS << "\n";
    generateEnumExports(Records, OS, Enum, Prefix);
  }
}

// Generate enum class with values corresponding to different bit positions.
// Entries are emitted in the order in which they appear in the `Records`
// vector.
static void generateEnumBitmask(ArrayRef<const Record *> Records,
                                raw_ostream &OS, StringRef Enum,
                                StringRef Prefix, bool ExportEnums) {
  assert(Records.size() <= 64 && "Too many values for a bitmask");
  StringRef Type = Records.size() <= 32 ? "uint32_t" : "uint64_t";
  StringRef TypeSuffix = Records.size() <= 32 ? "U" : "ULL";

  OS << "\n";
  OS << "enum class " << Enum << " : " << Type << " {\n";
  std::string LastName;
  for (auto [I, R] : llvm::enumerate(Records)) {
    LastName = getIdentifierName(R, Prefix);
    OS << "  " << LastName << " = " << (1ull << I) << TypeSuffix << ",\n";
  }
  OS << "  LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/" << LastName << ")\n";
  OS << "};\n";
  OS << "\n";
  OS << "static constexpr std::size_t " << Enum
     << "_enumSize = " << Records.size() << ";\n";

  // Make the enum values available in the defined namespace. This allows us to
  // write something like Enum_X if we have a `using namespace <CppNamespace>`.
  // At the same time we do not loose the strong type guarantees of the enum
  // class, that is we cannot pass an unsigned as Directive without an explicit
  // cast.
  if (ExportEnums) {
    OS << "\n";
    generateEnumExports(Records, OS, Enum, Prefix);
  }
}

// Generate enums for values that clauses can take.
// Also generate function declarations for get<Enum>Name(StringRef Str).
static void generateEnumClauseVal(ArrayRef<const Record *> Records,
                                  raw_ostream &OS,
                                  const DirectiveLanguage &DirLang,
                                  std::string &EnumHelperFuncs) {
  for (const Record *R : Records) {
    Clause C(R);
    const auto &ClauseVals = C.getClauseVals();
    if (ClauseVals.size() <= 0)
      continue;

    StringRef Enum = C.getEnumName();
    if (Enum.empty()) {
      PrintError("enumClauseValue field not set in Clause" +
                 C.getFormattedName() + ".");
      return;
    }

    OS << "\n";
    OS << "enum class " << Enum << " {\n";
    for (const ClauseVal CVal : ClauseVals)
      OS << "  " << CVal.getRecordName() << "=" << CVal.getValue() << ",\n";
    OS << "};\n";

    if (DirLang.hasMakeEnumAvailableInNamespace()) {
      OS << "\n";
      for (const auto &CV : ClauseVals) {
        OS << "constexpr auto " << CV->getName() << " = " << Enum
           << "::" << CV->getName() << ";\n";
      }
      EnumHelperFuncs += (Twine("LLVM_ABI ") + Twine(Enum) + Twine(" get") +
                          Twine(Enum) + Twine("(StringRef Str);\n"))
                             .str();

      EnumHelperFuncs +=
          (Twine("LLVM_ABI StringRef get") + Twine(DirLang.getName()) +
           Twine(Enum) + Twine("Name(") + Twine(Enum) + Twine(" x);\n"))
              .str();
    }
  }
}

static bool hasDuplicateClauses(ArrayRef<const Record *> Clauses,
                                const Directive &Directive,
                                StringSet<> &CrtClauses) {
  bool HasError = false;
  for (const VersionedClause VerClause : Clauses) {
    const auto InsRes = CrtClauses.insert(VerClause.getClause().getName());
    if (!InsRes.second) {
      PrintError("Clause " + VerClause.getClause().getRecordName() +
                 " already defined on directive " + Directive.getRecordName());
      HasError = true;
    }
  }
  return HasError;
}

// Check for duplicate clauses in lists. Clauses cannot appear twice in the
// three allowed list. Also, since required implies allowed, clauses cannot
// appear in both the allowedClauses and requiredClauses lists.
static bool
hasDuplicateClausesInDirectives(ArrayRef<const Record *> Directives) {
  bool HasDuplicate = false;
  for (const Directive Dir : Directives) {
    StringSet<> Clauses;
    // Check for duplicates in the three allowed lists.
    if (hasDuplicateClauses(Dir.getAllowedClauses(), Dir, Clauses) ||
        hasDuplicateClauses(Dir.getAllowedOnceClauses(), Dir, Clauses) ||
        hasDuplicateClauses(Dir.getAllowedExclusiveClauses(), Dir, Clauses)) {
      HasDuplicate = true;
    }
    // Check for duplicate between allowedClauses and required
    Clauses.clear();
    if (hasDuplicateClauses(Dir.getAllowedClauses(), Dir, Clauses) ||
        hasDuplicateClauses(Dir.getRequiredClauses(), Dir, Clauses)) {
      HasDuplicate = true;
    }
    if (HasDuplicate)
      PrintFatalError("One or more clauses are defined multiple times on"
                      " directive " +
                      Dir.getRecordName());
  }

  return HasDuplicate;
}

// Check consitency of records. Return true if an error has been detected.
// Return false if the records are valid.
bool DirectiveLanguage::HasValidityErrors() const {
  if (getDirectiveLanguages().size() != 1) {
    PrintFatalError("A single definition of DirectiveLanguage is needed.");
    return true;
  }

  return hasDuplicateClausesInDirectives(getDirectives());
}

// Count the maximum number of leaf constituents per construct.
static size_t getMaxLeafCount(const DirectiveLanguage &DirLang) {
  size_t MaxCount = 0;
  for (const Directive D : DirLang.getDirectives())
    MaxCount = std::max(MaxCount, D.getLeafConstructs().size());
  return MaxCount;
}

// Generate the declaration section for the enumeration in the directive
// language.
static void emitDirectivesDecl(const RecordKeeper &Records, raw_ostream &OS) {
  const auto DirLang = DirectiveLanguage(Records);
  if (DirLang.HasValidityErrors())
    return;

  StringRef Lang = DirLang.getName();

  OS << "#ifndef LLVM_" << Lang << "_INC\n";
  OS << "#define LLVM_" << Lang << "_INC\n";
  OS << "\n#include \"llvm/ADT/ArrayRef.h\"\n";

  if (DirLang.hasEnableBitmaskEnumInNamespace())
    OS << "#include \"llvm/ADT/BitmaskEnum.h\"\n";

  OS << "#include \"llvm/Support/Compiler.h\"\n";
  OS << "#include <cstddef>\n"; // for size_t
  OS << "\n";
  OS << "namespace llvm {\n";
  OS << "class StringRef;\n";

  // Open namespaces defined in the directive language
  SmallVector<StringRef, 2> Namespaces;
  SplitString(DirLang.getCppNamespace(), Namespaces, "::");
  for (auto Ns : Namespaces)
    OS << "namespace " << Ns << " {\n";

  if (DirLang.hasEnableBitmaskEnumInNamespace())
    OS << "\nLLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();\n";

  // Emit Directive associations
  std::vector<const Record *> Associations;
  copy_if(DirLang.getAssociations(), std::back_inserter(Associations),
          // Skip the "special" value
          [](const Record *Def) { return Def->getName() != "AS_FromLeaves"; });
  generateEnumClass(Associations, OS, "Association",
                    /*Prefix=*/"", /*ExportEnums=*/false);

  generateEnumClass(DirLang.getCategories(), OS, "Category", /*Prefix=*/"",
                    /*ExportEnums=*/false);

  generateEnumBitmask(DirLang.getSourceLanguages(), OS, "SourceLanguage",
                      /*Prefix=*/"", /*ExportEnums=*/false);

  // Emit Directive enumeration
  generateEnumClass(DirLang.getDirectives(), OS, "Directive",
                    DirLang.getDirectivePrefix(),
                    DirLang.hasMakeEnumAvailableInNamespace());

  // Emit Clause enumeration
  generateEnumClass(DirLang.getClauses(), OS, "Clause",
                    DirLang.getClausePrefix(),
                    DirLang.hasMakeEnumAvailableInNamespace());

  // Emit ClauseVal enumeration
  std::string EnumHelperFuncs;
  generateEnumClauseVal(DirLang.getClauses(), OS, DirLang, EnumHelperFuncs);

  // Generic function signatures
  OS << "\n";
  OS << "// Enumeration helper functions\n";
  OS << "LLVM_ABI Directive get" << Lang << "DirectiveKind(StringRef Str);\n";
  OS << "\n";

  // For OpenMP the signature is
  //   getOpenMPDirectiveName(Directive D, unsigned V)
  OS << "LLVM_ABI StringRef get" << DirLang.getName()
     << "DirectiveName(Directive D";
  if (DirLang.getCppNamespace() == "omp")
    OS << ", unsigned = 0";
  OS << ");\n";
  OS << "\n";

  OS << "LLVM_ABI Clause get" << Lang << "ClauseKind(StringRef Str);\n";
  OS << "\n";
  OS << "LLVM_ABI StringRef get" << Lang << "ClauseName(Clause C);\n";
  OS << "\n";
  OS << "/// Return true if \\p C is a valid clause for \\p D in version \\p "
     << "Version.\n";
  OS << "LLVM_ABI bool isAllowedClauseForDirective(Directive D, "
     << "Clause C, unsigned Version);\n";
  OS << "\n";
  OS << "constexpr std::size_t getMaxLeafCount() { return "
     << getMaxLeafCount(DirLang) << "; }\n";
  OS << "LLVM_ABI Association getDirectiveAssociation(Directive D);\n";
  OS << "LLVM_ABI Category getDirectiveCategory(Directive D);\n";
  OS << "LLVM_ABI SourceLanguage getDirectiveLanguages(Directive D);\n";
  if (EnumHelperFuncs.length() > 0) {
    OS << EnumHelperFuncs;
    OS << "\n";
  }

  // Closing namespaces
  for (auto Ns : reverse(Namespaces))
    OS << "} // namespace " << Ns << "\n";

  OS << "} // namespace llvm\n";

  OS << "#endif // LLVM_" << Lang << "_INC\n";
}

// Generate function implementation for get<Enum>Name(StringRef Str)
static void generateGetName(ArrayRef<const Record *> Records, raw_ostream &OS,
                            StringRef Enum, const DirectiveLanguage &DirLang,
                            StringRef Prefix) {
  StringRef Lang = DirLang.getName();
  std::string Qual = getQualifier(DirLang);
  // For OpenMP the "Directive" signature is
  //   getOpenMPDirectiveName(Directive D, unsigned V)
  OS << "\n";
  OS << "llvm::StringRef " << Qual << "get" << Lang << Enum << "Name(" << Qual
     << Enum << " Kind";
  if (DirLang.getCppNamespace() == "omp" && Enum == "Directive")
    OS << ", unsigned";
  OS << ") {\n";
  OS << "  switch (Kind) {\n";
  for (const Record *R : Records) {
    OS << "    case " << getIdentifierName(R, Prefix) << ":\n";
    OS << "      return \"" << BaseRecord(R).getName() << "\";\n";
  }
  OS << "  }\n"; // switch
  OS << "  llvm_unreachable(\"Invalid " << Lang << " " << Enum << " kind\");\n";
  OS << "}\n";
}

// Generate function implementation for get<Enum>Kind(StringRef Str)
static void generateGetKind(ArrayRef<const Record *> Records, raw_ostream &OS,
                            StringRef Enum, const DirectiveLanguage &DirLang,
                            StringRef Prefix, bool ImplicitAsUnknown) {

  const auto *DefaultIt = find_if(
      Records, [](const Record *R) { return R->getValueAsBit("isDefault"); });

  if (DefaultIt == Records.end()) {
    PrintError("At least one " + Enum + " must be defined as default.");
    return;
  }

  BaseRecord DefaultRec(*DefaultIt);
  std::string Qual = getQualifier(DirLang);
  std::string DefaultName = getIdentifierName(*DefaultIt, Prefix);

  OS << "\n";
  OS << Qual << Enum << " " << Qual << "get" << DirLang.getName() << Enum
     << "Kind(llvm::StringRef Str) {\n";
  OS << "  return StringSwitch<" << Enum << ">(Str)\n";

  for (const Record *R : Records) {
    BaseRecord Rec(R);
    if (ImplicitAsUnknown && R->getValueAsBit("isImplicit")) {
      OS << "    .Case(\"" << Rec.getName() << "\"," << DefaultName << ")\n";
    } else {
      OS << "    .Case(\"" << Rec.getName() << "\","
         << getIdentifierName(R, Prefix) << ")\n";
    }
  }
  OS << "    .Default(" << DefaultName << ");\n";
  OS << "}\n";
}

// Generate function implementation for get<ClauseVal>Kind(StringRef Str)
static void generateGetKindClauseVal(const DirectiveLanguage &DirLang,
                                     raw_ostream &OS) {
  StringRef Lang = DirLang.getName();
  std::string Qual = getQualifier(DirLang);

  for (const Clause C : DirLang.getClauses()) {
    const auto &ClauseVals = C.getClauseVals();
    if (ClauseVals.size() <= 0)
      continue;

    auto DefaultIt = find_if(ClauseVals, [](const Record *CV) {
      return CV->getValueAsBit("isDefault");
    });

    if (DefaultIt == ClauseVals.end()) {
      PrintError("At least one val in Clause " + C.getFormattedName() +
                 " must be defined as default.");
      return;
    }
    const auto DefaultName = (*DefaultIt)->getName();

    StringRef Enum = C.getEnumName();
    if (Enum.empty()) {
      PrintError("enumClauseValue field not set in Clause" +
                 C.getFormattedName() + ".");
      return;
    }

    OS << "\n";
    OS << Qual << Enum << " " << Qual << "get" << Enum
       << "(llvm::StringRef Str) {\n";
    OS << "  return StringSwitch<" << Enum << ">(Str)\n";
    for (const auto &CV : ClauseVals) {
      ClauseVal CVal(CV);
      OS << "    .Case(\"" << CVal.getFormattedName() << "\"," << CV->getName()
         << ")\n";
    }
    OS << "    .Default(" << DefaultName << ");\n";
    OS << "}\n";

    OS << "\n";
    OS << "llvm::StringRef " << Qual << "get" << Lang << Enum << "Name(" << Qual
       << Enum << " x) {\n";
    OS << "  switch (x) {\n";
    for (const auto &CV : ClauseVals) {
      ClauseVal CVal(CV);
      OS << "    case " << CV->getName() << ":\n";
      OS << "      return \"" << CVal.getFormattedName() << "\";\n";
    }
    OS << "  }\n"; // switch
    OS << "  llvm_unreachable(\"Invalid " << Lang << " " << Enum
       << " kind\");\n";
    OS << "}\n";
  }
}

static void generateCaseForVersionedClauses(ArrayRef<const Record *> VerClauses,
                                            raw_ostream &OS,
                                            const DirectiveLanguage &DirLang,
                                            StringSet<> &Cases) {
  StringRef Prefix = DirLang.getClausePrefix();
  for (const Record *R : VerClauses) {
    VersionedClause VerClause(R);
    std::string Name =
        getIdentifierName(VerClause.getClause().getRecord(), Prefix);
    if (Cases.insert(Name).second) {
      OS << "        case " << Name << ":\n";
      OS << "          return " << VerClause.getMinVersion()
         << " <= Version && " << VerClause.getMaxVersion() << " >= Version;\n";
    }
  }
}

// Generate the isAllowedClauseForDirective function implementation.
static void generateIsAllowedClause(const DirectiveLanguage &DirLang,
                                    raw_ostream &OS) {
  std::string Qual = getQualifier(DirLang);

  OS << "\n";
  OS << "bool " << Qual << "isAllowedClauseForDirective(" << Qual
     << "Directive D, " << Qual << "Clause C, unsigned Version) {\n";
  OS << "  assert(unsigned(D) <= Directive_enumSize);\n";
  OS << "  assert(unsigned(C) <= Clause_enumSize);\n";

  OS << "  switch (D) {\n";

  StringRef Prefix = DirLang.getDirectivePrefix();
  for (const Record *R : DirLang.getDirectives()) {
    Directive Dir(R);
    OS << "    case " << getIdentifierName(R, Prefix) << ":\n";
    if (Dir.getAllowedClauses().empty() &&
        Dir.getAllowedOnceClauses().empty() &&
        Dir.getAllowedExclusiveClauses().empty() &&
        Dir.getRequiredClauses().empty()) {
      OS << "      return false;\n";
    } else {
      OS << "      switch (C) {\n";

      StringSet<> Cases;

      generateCaseForVersionedClauses(Dir.getAllowedClauses(), OS, DirLang,
                                      Cases);

      generateCaseForVersionedClauses(Dir.getAllowedOnceClauses(), OS, DirLang,
                                      Cases);

      generateCaseForVersionedClauses(Dir.getAllowedExclusiveClauses(), OS,
                                      DirLang, Cases);

      generateCaseForVersionedClauses(Dir.getRequiredClauses(), OS, DirLang,
                                      Cases);

      OS << "        default:\n";
      OS << "          return false;\n";
      OS << "      }\n"; // End of clauses switch
    }
    OS << "      break;\n";
  }

  OS << "  }\n"; // End of directives switch
  OS << "  llvm_unreachable(\"Invalid " << DirLang.getName()
     << " Directive kind\");\n";
  OS << "}\n"; // End of function isAllowedClauseForDirective
}

static void emitLeafTable(const DirectiveLanguage &DirLang, raw_ostream &OS,
                          StringRef TableName) {
  // The leaf constructs are emitted in a form of a 2D table, where each
  // row corresponds to a directive (and there is a row for each directive).
  //
  // Each row consists of
  // - the id of the directive itself,
  // - number of leaf constructs that will follow (0 for leafs),
  // - ids of the leaf constructs (none if the directive is itself a leaf).
  // The total number of these entries is at most MaxLeafCount+2. If this
  // number is less than that, it is padded to occupy exactly MaxLeafCount+2
  // entries in memory.
  //
  // The rows are stored in the table in the lexicographical order. This
  // is intended to enable binary search when mapping a sequence of leafs
  // back to the compound directive.
  // The consequence of that is that in order to find a row corresponding
  // to the given directive, we'd need to scan the first element of each
  // row. To avoid this, an auxiliary ordering table is created, such that
  //   row for Dir_A = table[auxiliary[Dir_A]].

  ArrayRef<const Record *> Directives = DirLang.getDirectives();
  DenseMap<const Record *, int> DirId; // Record * -> llvm::omp::Directive

  for (auto [Idx, Rec] : enumerate(Directives))
    DirId.try_emplace(Rec, Idx);

  using LeafList = std::vector<int>;
  int MaxLeafCount = getMaxLeafCount(DirLang);

  // The initial leaf table, rows order is same as directive order.
  std::vector<LeafList> LeafTable(Directives.size());
  for (auto [Idx, Rec] : enumerate(Directives)) {
    Directive Dir(Rec);
    std::vector<const Record *> Leaves = Dir.getLeafConstructs();

    auto &List = LeafTable[Idx];
    List.resize(MaxLeafCount + 2);
    List[0] = Idx;           // The id of the directive itself.
    List[1] = Leaves.size(); // The number of leaves to follow.

    for (int I = 0; I != MaxLeafCount; ++I)
      List[I + 2] =
          static_cast<size_t>(I) < Leaves.size() ? DirId.at(Leaves[I]) : -1;
  }

  // Some Fortran directives are delimited, i.e. they have the form of
  // "directive"---"end directive". If "directive" is a compound construct,
  // then the set of leaf constituents will be nonempty and the same for
  // both directives. Given this set of leafs, looking up the corresponding
  // compound directive should return "directive", and not "end directive".
  // To avoid this problem, gather all "end directives" at the end of the
  // leaf table, and only do the search on the initial segment of the table
  // that excludes the "end directives".
  // It's safe to find all directives whose names begin with "end ". The
  // problem only exists for compound directives, like "end do simd".
  // All existing directives with names starting with "end " are either
  // "end directives" for an existing "directive", or leaf directives
  // (such as "end declare target").
  DenseSet<int> EndDirectives;
  for (auto [Rec, Id] : DirId) {
    if (Directive(Rec).getName().starts_with_insensitive("end "))
      EndDirectives.insert(Id);
  }

  // Avoid sorting the vector<vector> array, instead sort an index array.
  // It will also be useful later to create the auxiliary indexing array.
  std::vector<int> Ordering(Directives.size());
  std::iota(Ordering.begin(), Ordering.end(), 0);

  sort(Ordering, [&](int A, int B) {
    auto &LeavesA = LeafTable[A];
    auto &LeavesB = LeafTable[B];
    int DirA = LeavesA[0], DirB = LeavesB[0];
    // First of all, end directives compare greater than non-end directives.
    int IsEndA = EndDirectives.count(DirA), IsEndB = EndDirectives.count(DirB);
    if (IsEndA != IsEndB)
      return IsEndA < IsEndB;
    if (LeavesA[1] == 0 && LeavesB[1] == 0)
      return DirA < DirB;
    return std::lexicographical_compare(&LeavesA[2], &LeavesA[2] + LeavesA[1],
                                        &LeavesB[2], &LeavesB[2] + LeavesB[1]);
  });

  // Emit the table

  // The directives are emitted into a scoped enum, for which the underlying
  // type is `int` (by default). The code above uses `int` to store directive
  // ids, so make sure that we catch it when something changes in the
  // underlying type.
  StringRef Prefix = DirLang.getDirectivePrefix();
  std::string Qual = getQualifier(DirLang);
  std::string DirectiveType = Qual + "Directive";
  OS << "\nstatic_assert(sizeof(" << DirectiveType << ") == sizeof(int));\n";

  OS << "[[maybe_unused]] static const " << DirectiveType << ' ' << TableName
     << "[][" << MaxLeafCount + 2 << "] = {\n";
  for (size_t I = 0, E = Directives.size(); I != E; ++I) {
    auto &Leaves = LeafTable[Ordering[I]];
    OS << "    {" << Qual << getIdentifierName(Directives[Leaves[0]], Prefix);
    OS << ", static_cast<" << DirectiveType << ">(" << Leaves[1] << "),";
    for (size_t I = 2, E = Leaves.size(); I != E; ++I) {
      int Idx = Leaves[I];
      if (Idx >= 0)
        OS << ' ' << Qual << getIdentifierName(Directives[Leaves[I]], Prefix)
           << ',';
      else
        OS << " static_cast<" << DirectiveType << ">(-1),";
    }
    OS << "},\n";
  }
  OS << "};\n\n";

  // Emit a marker where the first "end directive" is.
  auto FirstE = find_if(Ordering, [&](int RowIdx) {
    return EndDirectives.count(LeafTable[RowIdx][0]);
  });
  OS << "[[maybe_unused]] static auto " << TableName
     << "EndDirective = " << TableName << " + "
     << std::distance(Ordering.begin(), FirstE) << ";\n\n";

  // Emit the auxiliary index table: it's the inverse of the `Ordering`
  // table above.
  OS << "[[maybe_unused]] static const int " << TableName << "Ordering[] = {\n";
  OS << "   ";
  std::vector<int> Reverse(Ordering.size());
  for (int I = 0, E = Ordering.size(); I != E; ++I)
    Reverse[Ordering[I]] = I;
  for (int Idx : Reverse)
    OS << ' ' << Idx << ',';
  OS << "\n};\n";
}

static void generateGetDirectiveAssociation(const DirectiveLanguage &DirLang,
                                            raw_ostream &OS) {
  enum struct Association {
    None = 0, // None should be the smallest value.
    Block,    // The values of the rest don't matter.
    Declaration,
    Delimited,
    Loop,
    Separating,
    FromLeaves,
    Invalid,
  };

  ArrayRef<const Record *> Associations = DirLang.getAssociations();

  auto GetAssocValue = [](StringRef Name) -> Association {
    return StringSwitch<Association>(Name)
        .Case("AS_Block", Association::Block)
        .Case("AS_Declaration", Association::Declaration)
        .Case("AS_Delimited", Association::Delimited)
        .Case("AS_Loop", Association::Loop)
        .Case("AS_None", Association::None)
        .Case("AS_Separating", Association::Separating)
        .Case("AS_FromLeaves", Association::FromLeaves)
        .Default(Association::Invalid);
  };

  auto GetAssocName = [&](Association A) -> StringRef {
    if (A != Association::Invalid && A != Association::FromLeaves) {
      const auto *F = find_if(Associations, [&](const Record *R) {
        return GetAssocValue(R->getName()) == A;
      });
      if (F != Associations.end())
        return (*F)->getValueAsString("name"); // enum name
    }
    llvm_unreachable("Unexpected association value");
  };

  auto ErrorPrefixFor = [&](Directive D) -> std::string {
    return (Twine("Directive '") + D.getName() + "' in namespace '" +
            DirLang.getCppNamespace() + "' ")
        .str();
  };

  auto Reduce = [&](Association A, Association B) -> Association {
    if (A > B)
      std::swap(A, B);

    // Calculate the result using the following rules:
    //   x + x = x
    //   AS_None + x = x
    //   AS_Block + AS_Loop = AS_Loop
    if (A == Association::None || A == B)
      return B;
    if (A == Association::Block && B == Association::Loop)
      return B;
    if (A == Association::Loop && B == Association::Block)
      return A;
    return Association::Invalid;
  };

  DenseMap<const Record *, Association> AsMap;

  auto CompAssocImpl = [&](const Record *R, auto &&Self) -> Association {
    if (auto F = AsMap.find(R); F != AsMap.end())
      return F->second;

    Directive D(R);
    Association AS = GetAssocValue(D.getAssociation()->getName());
    if (AS == Association::Invalid) {
      PrintFatalError(ErrorPrefixFor(D) +
                      "has an unrecognized value for association: '" +
                      D.getAssociation()->getName() + "'");
    }
    if (AS != Association::FromLeaves) {
      AsMap.try_emplace(R, AS);
      return AS;
    }
    // Compute the association from leaf constructs.
    std::vector<const Record *> Leaves = D.getLeafConstructs();
    if (Leaves.empty()) {
      errs() << D.getName() << '\n';
      PrintFatalError(ErrorPrefixFor(D) +
                      "requests association to be computed from leaves, "
                      "but it has no leaves");
    }

    Association Result = Self(Leaves[0], Self);
    for (int I = 1, E = Leaves.size(); I < E; ++I) {
      Association A = Self(Leaves[I], Self);
      Association R = Reduce(Result, A);
      if (R == Association::Invalid) {
        PrintFatalError(ErrorPrefixFor(D) +
                        "has leaves with incompatible association values: " +
                        GetAssocName(A) + " and " + GetAssocName(R));
      }
      Result = R;
    }

    assert(Result != Association::Invalid);
    assert(Result != Association::FromLeaves);
    AsMap.try_emplace(R, Result);
    return Result;
  };

  for (const Record *R : DirLang.getDirectives())
    CompAssocImpl(R, CompAssocImpl); // Updates AsMap.

  OS << '\n';

  StringRef Prefix = DirLang.getDirectivePrefix();
  std::string Qual = getQualifier(DirLang);

  OS << Qual << "Association " << Qual << "getDirectiveAssociation(" << Qual
     << "Directive Dir) {\n";
  OS << "  switch (Dir) {\n";
  for (const Record *R : DirLang.getDirectives()) {
    if (auto F = AsMap.find(R); F != AsMap.end()) {
      OS << "  case " << getIdentifierName(R, Prefix) << ":\n";
      OS << "    return Association::" << GetAssocName(F->second) << ";\n";
    }
  }
  OS << "  } // switch (Dir)\n";
  OS << "  llvm_unreachable(\"Unexpected directive\");\n";
  OS << "}\n";
}

static void generateGetDirectiveCategory(const DirectiveLanguage &DirLang,
                                         raw_ostream &OS) {
  std::string Qual = getQualifier(DirLang);

  OS << '\n';
  OS << Qual << "Category " << Qual << "getDirectiveCategory(" << Qual
     << "Directive Dir) {\n";
  OS << "  switch (Dir) {\n";

  StringRef Prefix = DirLang.getDirectivePrefix();

  for (const Record *R : DirLang.getDirectives()) {
    Directive D(R);
    OS << "  case " << getIdentifierName(R, Prefix) << ":\n";
    OS << "    return Category::" << D.getCategory()->getValueAsString("name")
       << ";\n";
  }
  OS << "  } // switch (Dir)\n";
  OS << "  llvm_unreachable(\"Unexpected directive\");\n";
  OS << "}\n";
}

static void generateGetDirectiveLanguages(const DirectiveLanguage &DirLang,
                                          raw_ostream &OS) {
  std::string Qual = getQualifier(DirLang);

  OS << '\n';
  OS << Qual << "SourceLanguage " << Qual << "getDirectiveLanguages(" << Qual
     << "Directive D) {\n";
  OS << "  switch (D) {\n";

  StringRef Prefix = DirLang.getDirectivePrefix();

  for (const Record *R : DirLang.getDirectives()) {
    Directive D(R);
    OS << "  case " << getIdentifierName(R, Prefix) << ":\n";
    OS << "    return ";
    llvm::interleave(
        D.getSourceLanguages(), OS,
        [&](const Record *L) {
          StringRef N = L->getValueAsString("name");
          OS << "SourceLanguage::" << BaseRecord::formatName(N);
        },
        " | ");
    OS << ";\n";
  }
  OS << "  } // switch(D)\n";
  OS << "  llvm_unreachable(\"Unexpected directive\");\n";
  OS << "}\n";
}

// Generate a simple enum set with the give clauses.
static void generateClauseSet(ArrayRef<const Record *> VerClauses,
                              raw_ostream &OS, StringRef ClauseSetPrefix,
                              const Directive &Dir,
                              const DirectiveLanguage &DirLang, Frontend FE) {

  OS << "\n";
  OS << "static " << DirLang.getClauseEnumSetClass() << " " << ClauseSetPrefix
     << DirLang.getDirectivePrefix() << Dir.getFormattedName() << " {\n";

  StringRef Prefix = DirLang.getClausePrefix();

  for (const VersionedClause VerClause : VerClauses) {
    Clause C = VerClause.getClause();
    if (FE == Frontend::Flang) {
      OS << "  Clause::" << getIdentifierName(C.getRecord(), Prefix) << ",\n";
    } else {
      assert(FE == Frontend::Clang);
      assert(DirLang.getName() == "OpenACC");
      OS << "  OpenACCClauseKind::" << C.getClangAccSpelling() << ",\n";
    }
  }
  OS << "};\n";
}

// Generate an enum set for the 4 kinds of clauses linked to a directive.
static void generateDirectiveClauseSets(const DirectiveLanguage &DirLang,
                                        Frontend FE, raw_ostream &OS) {

  std::string IfDefName{"GEN_"};
  IfDefName += getFESpelling(FE).upper();
  IfDefName += "_DIRECTIVE_CLAUSE_SETS";
  IfDefScope Scope(IfDefName, OS);

  StringRef Namespace =
      getFESpelling(FE == Frontend::Flang ? Frontend::LLVM : FE);
  OS << "\n";
  // The namespace has to be different for clang vs flang, as 2 structs with the
  // same name but different layout is UB.  So just put the 'clang' on in the
  // clang namespace.
  OS << "namespace " << Namespace << " {\n";

  // Open namespaces defined in the directive language.
  SmallVector<StringRef, 2> Namespaces;
  SplitString(DirLang.getCppNamespace(), Namespaces, "::");
  for (auto Ns : Namespaces)
    OS << "namespace " << Ns << " {\n";

  for (const Directive Dir : DirLang.getDirectives()) {
    OS << "\n";
    OS << "// Sets for " << Dir.getName() << "\n";

    generateClauseSet(Dir.getAllowedClauses(), OS, "allowedClauses_", Dir,
                      DirLang, FE);
    generateClauseSet(Dir.getAllowedOnceClauses(), OS, "allowedOnceClauses_",
                      Dir, DirLang, FE);
    generateClauseSet(Dir.getAllowedExclusiveClauses(), OS,
                      "allowedExclusiveClauses_", Dir, DirLang, FE);
    generateClauseSet(Dir.getRequiredClauses(), OS, "requiredClauses_", Dir,
                      DirLang, FE);
  }

  // Closing namespaces
  for (auto Ns : reverse(Namespaces))
    OS << "} // namespace " << Ns << "\n";

  OS << "} // namespace " << Namespace << "\n";
}

// Generate a map of directive (key) with DirectiveClauses struct as values.
// The struct holds the 4 sets of enumeration for the 4 kinds of clauses
// allowances (allowed, allowed once, allowed exclusive and required).
static void generateDirectiveClauseMap(const DirectiveLanguage &DirLang,
                                       Frontend FE, raw_ostream &OS) {
  std::string IfDefName{"GEN_"};
  IfDefName += getFESpelling(FE).upper();
  IfDefName += "_DIRECTIVE_CLAUSE_MAP";
  IfDefScope Scope(IfDefName, OS);

  OS << "\n";
  OS << "{\n";

  // The namespace has to be different for clang vs flang, as 2 structs with the
  // same name but different layout is UB.  So just put the 'clang' on in the
  // clang namespace.
  std::string Qual =
      getQualifier(DirLang, FE == Frontend::Flang ? Frontend::LLVM : FE);
  StringRef Prefix = DirLang.getDirectivePrefix();

  for (const Record *R : DirLang.getDirectives()) {
    Directive Dir(R);
    std::string Name = getIdentifierName(R, Prefix);

    OS << "  {";
    if (FE == Frontend::Flang) {
      OS << Qual << "Directive::" << Name << ",\n";
    } else {
      assert(FE == Frontend::Clang);
      assert(DirLang.getName() == "OpenACC");
      OS << "clang::OpenACCDirectiveKind::" << Dir.getClangAccSpelling()
         << ",\n";
    }

    OS << "    {\n";
    OS << "      " << Qual << "allowedClauses_" << Name << ",\n";
    OS << "      " << Qual << "allowedOnceClauses_" << Name << ",\n";
    OS << "      " << Qual << "allowedExclusiveClauses_" << Name << ",\n";
    OS << "      " << Qual << "requiredClauses_" << Name << ",\n";
    OS << "    }\n";
    OS << "  },\n";
  }

  OS << "}\n";
}

// Generate classes entry for Flang clauses in the Flang parse-tree
// If the clause as a non-generic class, no entry is generated.
// If the clause does not hold a value, an EMPTY_CLASS is used.
// If the clause class is generic then a WRAPPER_CLASS is used. When the value
// is optional, the value class is wrapped into a std::optional.
static void generateFlangClauseParserClass(const DirectiveLanguage &DirLang,
                                           raw_ostream &OS) {

  IfDefScope Scope("GEN_FLANG_CLAUSE_PARSER_CLASSES", OS);

  OS << "\n";

  for (const Clause Clause : DirLang.getClauses()) {
    if (!Clause.getFlangClass().empty()) {
      OS << "WRAPPER_CLASS(" << Clause.getFormattedParserClassName() << ", ";
      if (Clause.isValueOptional() && Clause.isValueList()) {
        OS << "std::optional<std::list<" << Clause.getFlangClass() << ">>";
      } else if (Clause.isValueOptional()) {
        OS << "std::optional<" << Clause.getFlangClass() << ">";
      } else if (Clause.isValueList()) {
        OS << "std::list<" << Clause.getFlangClass() << ">";
      } else {
        OS << Clause.getFlangClass();
      }
    } else {
      OS << "EMPTY_CLASS(" << Clause.getFormattedParserClassName();
    }
    OS << ");\n";
  }
}

// Generate a list of the different clause classes for Flang.
static void generateFlangClauseParserClassList(const DirectiveLanguage &DirLang,
                                               raw_ostream &OS) {

  IfDefScope Scope("GEN_FLANG_CLAUSE_PARSER_CLASSES_LIST", OS);

  OS << "\n";
  interleaveComma(DirLang.getClauses(), OS, [&](const Record *C) {
    Clause Clause(C);
    OS << Clause.getFormattedParserClassName() << "\n";
  });
}

// Generate dump node list for the clauses holding a generic class name.
static void generateFlangClauseDump(const DirectiveLanguage &DirLang,
                                    raw_ostream &OS) {

  IfDefScope Scope("GEN_FLANG_DUMP_PARSE_TREE_CLAUSES", OS);

  OS << "\n";
  for (const Clause Clause : DirLang.getClauses()) {
    OS << "NODE(" << DirLang.getFlangClauseBaseClass() << ", "
       << Clause.getFormattedParserClassName() << ")\n";
  }
}

// Generate Unparse functions for clauses classes in the Flang parse-tree
// If the clause is a non-generic class, no entry is generated.
static void generateFlangClauseUnparse(const DirectiveLanguage &DirLang,
                                       raw_ostream &OS) {

  IfDefScope Scope("GEN_FLANG_CLAUSE_UNPARSE", OS);

  StringRef Base = DirLang.getFlangClauseBaseClass();
  OS << "\n";

  for (const Clause Clause : DirLang.getClauses()) {
    if (Clause.skipFlangUnparser())
      continue;
    std::string Parser = Clause.getFormattedParserClassName();
    std::string Upper = Clause.getName().upper();

    if (!Clause.getFlangClass().empty()) {
      if (Clause.isValueOptional() && Clause.getDefaultValue().empty()) {
        OS << "void Unparse(const " << Base << "::" << Parser << " &x) {\n";
        OS << "  Word(\"" << Upper << "\");\n";

        OS << "  Walk(\"(\", x.v, \")\");\n";
        OS << "}\n";
      } else if (Clause.isValueOptional()) {
        OS << "void Unparse(const " << Base << "::" << Parser << " &x) {\n";
        OS << "  Word(\"" << Upper << "\");\n";
        OS << "  Put(\"(\");\n";
        OS << "  if (x.v.has_value())\n";
        if (Clause.isValueList())
          OS << "    Walk(x.v, \",\");\n";
        else
          OS << "    Walk(x.v);\n";
        OS << "  else\n";
        OS << "    Put(\"" << Clause.getDefaultValue() << "\");\n";
        OS << "  Put(\")\");\n";
        OS << "}\n";
      } else {
        OS << "void Unparse(const " << Base << "::" << Parser << " &x) {\n";
        OS << "  Word(\"" << Upper << "\");\n";
        OS << "  Put(\"(\");\n";
        if (Clause.isValueList())
          OS << "  Walk(x.v, \",\");\n";
        else
          OS << "  Walk(x.v);\n";
        OS << "  Put(\")\");\n";
        OS << "}\n";
      }
    } else {
      OS << "void Before(const " << Base << "::" << Parser << " &) { Word(\""
         << Upper << "\"); }\n";
    }
  }
}

// Generate check in the Enter functions for clauses classes.
static void generateFlangClauseCheckPrototypes(const DirectiveLanguage &DirLang,
                                               raw_ostream &OS) {

  IfDefScope Scope("GEN_FLANG_CLAUSE_CHECK_ENTER", OS);

  OS << "\n";
  for (const Clause Clause : DirLang.getClauses()) {
    OS << "void Enter(const parser::" << DirLang.getFlangClauseBaseClass()
       << "::" << Clause.getFormattedParserClassName() << " &);\n";
  }
}

// Generate the mapping for clauses between the parser class and the
// corresponding clause Kind
static void generateFlangClauseParserKindMap(const DirectiveLanguage &DirLang,
                                             raw_ostream &OS) {

  IfDefScope Scope("GEN_FLANG_CLAUSE_PARSER_KIND_MAP", OS);

  StringRef Prefix = DirLang.getClausePrefix();
  std::string Qual = getQualifier(DirLang);

  OS << "\n";
  for (const Record *R : DirLang.getClauses()) {
    Clause C(R);
    OS << "if constexpr (std::is_same_v<A, parser::"
       << DirLang.getFlangClauseBaseClass()
       << "::" << C.getFormattedParserClassName();
    OS << ">)\n";
    OS << "  return " << Qual << "Clause::" << getIdentifierName(R, Prefix)
       << ";\n";
  }

  OS << "llvm_unreachable(\"Invalid " << DirLang.getName()
     << " Parser clause\");\n";
}

static bool compareClauseName(const Record *R1, const Record *R2) {
  Clause C1(R1);
  Clause C2(R2);
  return (C1.getName() > C2.getName());
}

// Generate the parser for the clauses.
static void generateFlangClausesParser(const DirectiveLanguage &DirLang,
                                       raw_ostream &OS) {
  std::vector<const Record *> Clauses = DirLang.getClauses();
  // Sort clauses in reverse alphabetical order so with clauses with same
  // beginning, the longer option is tried before.
  sort(Clauses, compareClauseName);
  IfDefScope Scope("GEN_FLANG_CLAUSES_PARSER", OS);
  StringRef Base = DirLang.getFlangClauseBaseClass();

  OS << "\n";
  unsigned Index = 0;
  unsigned LastClauseIndex = Clauses.size() - 1;
  OS << "TYPE_PARSER(\n";
  for (const Clause Clause : Clauses) {
    const std::vector<StringRef> &Aliases = Clause.getAliases();
    if (Aliases.empty()) {
      OS << "  \"" << Clause.getName() << "\"";
    } else {
      OS << "  ("
         << "\"" << Clause.getName() << "\"_tok";
      for (StringRef Alias : Aliases) {
        OS << " || \"" << Alias << "\"_tok";
      }
      OS << ")";
    }

    StringRef FlangClass = Clause.getFlangClass();
    OS << " >> construct<" << Base << ">(construct<" << Base
       << "::" << Clause.getFormattedParserClassName() << ">(";
    if (FlangClass.empty()) {
      OS << "))";
      if (Index != LastClauseIndex)
        OS << " ||";
      OS << "\n";
      ++Index;
      continue;
    }

    if (Clause.isValueOptional())
      OS << "maybe(";
    OS << "parenthesized(";
    if (Clause.isValueList())
      OS << "nonemptyList(";

    if (!Clause.getPrefix().empty())
      OS << "\"" << Clause.getPrefix() << ":\" >> ";

    // The common Flang parser are used directly. Their name is identical to
    // the Flang class with first letter as lowercase. If the Flang class is
    // not a common class, we assume there is a specific Parser<>{} with the
    // Flang class name provided.
    SmallString<128> Scratch;
    StringRef Parser =
        StringSwitch<StringRef>(FlangClass)
            .Case("Name", "name")
            .Case("ScalarIntConstantExpr", "scalarIntConstantExpr")
            .Case("ScalarIntExpr", "scalarIntExpr")
            .Case("ScalarExpr", "scalarExpr")
            .Case("ScalarLogicalExpr", "scalarLogicalExpr")
            .Default(("Parser<" + FlangClass + ">{}").toStringRef(Scratch));
    OS << Parser;
    if (!Clause.getPrefix().empty() && Clause.isPrefixOptional())
      OS << " || " << Parser;
    if (Clause.isValueList()) // close nonemptyList(.
      OS << ")";
    OS << ")"; // close parenthesized(.

    if (Clause.isValueOptional()) // close maybe(.
      OS << ")";
    OS << "))";
    if (Index != LastClauseIndex)
      OS << " ||";
    OS << "\n";
    ++Index;
  }
  OS << ")\n";
}

// Generate the implementation section for the enumeration in the directive
// language
static void emitDirectivesClangImpl(const DirectiveLanguage &DirLang,
                                    raw_ostream &OS) {
  // Currently we only have work to do for OpenACC, so skip otherwise.
  if (DirLang.getName() != "OpenACC")
    return;

  generateDirectiveClauseSets(DirLang, Frontend::Clang, OS);
  generateDirectiveClauseMap(DirLang, Frontend::Clang, OS);
}
// Generate the implementation section for the enumeration in the directive
// language
static void emitDirectivesFlangImpl(const DirectiveLanguage &DirLang,
                                    raw_ostream &OS) {
  generateDirectiveClauseSets(DirLang, Frontend::Flang, OS);

  generateDirectiveClauseMap(DirLang, Frontend::Flang, OS);

  generateFlangClauseParserClass(DirLang, OS);

  generateFlangClauseParserClassList(DirLang, OS);

  generateFlangClauseDump(DirLang, OS);

  generateFlangClauseUnparse(DirLang, OS);

  generateFlangClauseCheckPrototypes(DirLang, OS);

  generateFlangClauseParserKindMap(DirLang, OS);

  generateFlangClausesParser(DirLang, OS);
}

static void generateClauseClassMacro(const DirectiveLanguage &DirLang,
                                     raw_ostream &OS) {
  // Generate macros style information for legacy code in clang
  IfDefScope Scope("GEN_CLANG_CLAUSE_CLASS", OS);

  StringRef Prefix = DirLang.getClausePrefix();
  OS << "\n";

  OS << "#ifndef CLAUSE\n";
  OS << "#define CLAUSE(Enum, Str, Implicit)\n";
  OS << "#endif\n";
  OS << "#ifndef CLAUSE_CLASS\n";
  OS << "#define CLAUSE_CLASS(Enum, Str, Class)\n";
  OS << "#endif\n";
  OS << "#ifndef CLAUSE_NO_CLASS\n";
  OS << "#define CLAUSE_NO_CLASS(Enum, Str)\n";
  OS << "#endif\n";
  OS << "\n";
  OS << "#define __CLAUSE(Name, Class)                      \\\n";
  OS << "  CLAUSE(" << Prefix << "##Name, #Name, /* Implicit */ false) \\\n";
  OS << "  CLAUSE_CLASS(" << Prefix << "##Name, #Name, Class)\n";
  OS << "#define __CLAUSE_NO_CLASS(Name)                    \\\n";
  OS << "  CLAUSE(" << Prefix << "##Name, #Name, /* Implicit */ false) \\\n";
  OS << "  CLAUSE_NO_CLASS(" << Prefix << "##Name, #Name)\n";
  OS << "#define __IMPLICIT_CLAUSE_CLASS(Name, Str, Class)  \\\n";
  OS << "  CLAUSE(" << Prefix << "##Name, Str, /* Implicit */ true)    \\\n";
  OS << "  CLAUSE_CLASS(" << Prefix << "##Name, Str, Class)\n";
  OS << "#define __IMPLICIT_CLAUSE_NO_CLASS(Name, Str)      \\\n";
  OS << "  CLAUSE(" << Prefix << "##Name, Str, /* Implicit */ true)    \\\n";
  OS << "  CLAUSE_NO_CLASS(" << Prefix << "##Name, Str)\n";
  OS << "\n";

  for (const Clause C : DirLang.getClauses()) {
    std::string Name = C.getFormattedName();
    if (C.getClangClass().empty()) { // NO_CLASS
      if (C.isImplicit()) {
        OS << "__IMPLICIT_CLAUSE_NO_CLASS(" << Name << ", \"" << Name
           << "\")\n";
      } else {
        OS << "__CLAUSE_NO_CLASS(" << Name << ")\n";
      }
    } else { // CLASS
      if (C.isImplicit()) {
        OS << "__IMPLICIT_CLAUSE_CLASS(" << Name << ", \"" << Name << "\", "
           << C.getClangClass() << ")\n";
      } else {
        OS << "__CLAUSE(" << Name << ", " << C.getClangClass() << ")\n";
      }
    }
  }

  OS << "\n";
  OS << "#undef __IMPLICIT_CLAUSE_NO_CLASS\n";
  OS << "#undef __IMPLICIT_CLAUSE_CLASS\n";
  OS << "#undef __CLAUSE_NO_CLASS\n";
  OS << "#undef __CLAUSE\n";
  OS << "#undef CLAUSE_NO_CLASS\n";
  OS << "#undef CLAUSE_CLASS\n";
  OS << "#undef CLAUSE\n";
}

// Generate the implemenation for the enumeration in the directive
// language. This code can be included in library.
void emitDirectivesBasicImpl(const DirectiveLanguage &DirLang,
                             raw_ostream &OS) {
  IfDefScope Scope("GEN_DIRECTIVES_IMPL", OS);

  StringRef DPrefix = DirLang.getDirectivePrefix();
  StringRef CPrefix = DirLang.getClausePrefix();

  OS << "\n#include \"llvm/Support/ErrorHandling.h\"\n";

  // getDirectiveKind(StringRef Str)
  generateGetKind(DirLang.getDirectives(), OS, "Directive", DirLang, DPrefix,
                  /*ImplicitAsUnknown=*/false);

  // getDirectiveName(Directive Kind)
  generateGetName(DirLang.getDirectives(), OS, "Directive", DirLang, DPrefix);

  // getClauseKind(StringRef Str)
  generateGetKind(DirLang.getClauses(), OS, "Clause", DirLang, CPrefix,
                  /*ImplicitAsUnknown=*/true);

  // getClauseName(Clause Kind)
  generateGetName(DirLang.getClauses(), OS, "Clause", DirLang, CPrefix);

  // get<ClauseVal>Kind(StringRef Str)
  generateGetKindClauseVal(DirLang, OS);

  // isAllowedClauseForDirective(Directive D, Clause C, unsigned Version)
  generateIsAllowedClause(DirLang, OS);

  // getDirectiveAssociation(Directive D)
  generateGetDirectiveAssociation(DirLang, OS);

  // getDirectiveCategory(Directive D)
  generateGetDirectiveCategory(DirLang, OS);

  // getDirectiveLanguages(Directive D)
  generateGetDirectiveLanguages(DirLang, OS);

  // Leaf table for getLeafConstructs, etc.
  emitLeafTable(DirLang, OS, "LeafConstructTable");
}

// Generate the implemenation section for the enumeration in the directive
// language.
static void emitDirectivesImpl(const RecordKeeper &Records, raw_ostream &OS) {
  const auto DirLang = DirectiveLanguage(Records);
  if (DirLang.HasValidityErrors())
    return;

  emitDirectivesFlangImpl(DirLang, OS);

  emitDirectivesClangImpl(DirLang, OS);

  generateClauseClassMacro(DirLang, OS);

  emitDirectivesBasicImpl(DirLang, OS);
}

static TableGen::Emitter::Opt
    X("gen-directive-decl", emitDirectivesDecl,
      "Generate directive related declaration code (header file)");

static TableGen::Emitter::Opt
    Y("gen-directive-impl", emitDirectivesImpl,
      "Generate directive related implementation code");
