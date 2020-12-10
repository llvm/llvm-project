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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

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
} // end anonymous namespace

namespace llvm {

// Generate enum class
void GenerateEnumClass(const std::vector<Record *> &Records, raw_ostream &OS,
                       StringRef Enum, StringRef Prefix,
                       const DirectiveLanguage &DirLang) {
  OS << "\n";
  OS << "enum class " << Enum << " {\n";
  for (const auto &R : Records) {
    BaseRecord Rec{R};
    OS << "  " << Prefix << Rec.getFormattedName() << ",\n";
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
  if (DirLang.hasMakeEnumAvailableInNamespace()) {
    OS << "\n";
    for (const auto &R : Records) {
      BaseRecord Rec{R};
      OS << "constexpr auto " << Prefix << Rec.getFormattedName() << " = "
         << "llvm::" << DirLang.getCppNamespace() << "::" << Enum
         << "::" << Prefix << Rec.getFormattedName() << ";\n";
    }
  }
}

// Generate enums for values that clauses can take.
// Also generate function declarations for get<Enum>Name(StringRef Str).
void GenerateEnumClauseVal(const std::vector<Record *> &Records,
                           raw_ostream &OS, const DirectiveLanguage &DirLang,
                           std::string &EnumHelperFuncs) {
  for (const auto &R : Records) {
    Clause C{R};
    const auto &ClauseVals = C.getClauseVals();
    if (ClauseVals.size() <= 0)
      continue;

    const auto &EnumName = C.getEnumName();
    if (EnumName.size() == 0) {
      PrintError("enumClauseValue field not set in Clause" +
                 C.getFormattedName() + ".");
      return;
    }

    OS << "\n";
    OS << "enum class " << EnumName << " {\n";
    for (const auto &CV : ClauseVals) {
      ClauseVal CVal{CV};
      OS << "  " << CV->getName() << "=" << CVal.getValue() << ",\n";
    }
    OS << "};\n";

    if (DirLang.hasMakeEnumAvailableInNamespace()) {
      OS << "\n";
      for (const auto &CV : ClauseVals) {
        OS << "constexpr auto " << CV->getName() << " = "
           << "llvm::" << DirLang.getCppNamespace() << "::" << EnumName
           << "::" << CV->getName() << ";\n";
      }
      EnumHelperFuncs += (llvm::Twine(EnumName) + llvm::Twine(" get") +
                          llvm::Twine(EnumName) + llvm::Twine("(StringRef);\n"))
                             .str();
    }
  }
}

bool HasDuplicateClauses(const std::vector<Record *> &Clauses,
                         const Directive &Directive,
                         llvm::StringSet<> &CrtClauses) {
  bool HasError = false;
  for (const auto &C : Clauses) {
    VersionedClause VerClause{C};
    const auto insRes = CrtClauses.insert(VerClause.getClause().getName());
    if (!insRes.second) {
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
bool HasDuplicateClausesInDirectives(const std::vector<Record *> &Directives) {
  bool HasDuplicate = false;
  for (const auto &D : Directives) {
    Directive Dir{D};
    llvm::StringSet<> Clauses;
    // Check for duplicates in the three allowed lists.
    if (HasDuplicateClauses(Dir.getAllowedClauses(), Dir, Clauses) ||
        HasDuplicateClauses(Dir.getAllowedOnceClauses(), Dir, Clauses) ||
        HasDuplicateClauses(Dir.getAllowedExclusiveClauses(), Dir, Clauses)) {
      HasDuplicate = true;
    }
    // Check for duplicate between allowedClauses and required
    Clauses.clear();
    if (HasDuplicateClauses(Dir.getAllowedClauses(), Dir, Clauses) ||
        HasDuplicateClauses(Dir.getRequiredClauses(), Dir, Clauses)) {
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

  return HasDuplicateClausesInDirectives(getDirectives());
}

// Generate the declaration section for the enumeration in the directive
// language
void EmitDirectivesDecl(RecordKeeper &Records, raw_ostream &OS) {
  const auto DirLang = DirectiveLanguage{Records};
  if (DirLang.HasValidityErrors())
    return;

  OS << "#ifndef LLVM_" << DirLang.getName() << "_INC\n";
  OS << "#define LLVM_" << DirLang.getName() << "_INC\n";

  if (DirLang.hasEnableBitmaskEnumInNamespace())
    OS << "\n#include \"llvm/ADT/BitmaskEnum.h\"\n";

  OS << "\n";
  OS << "namespace llvm {\n";
  OS << "class StringRef;\n";

  // Open namespaces defined in the directive language
  llvm::SmallVector<StringRef, 2> Namespaces;
  llvm::SplitString(DirLang.getCppNamespace(), Namespaces, "::");
  for (auto Ns : Namespaces)
    OS << "namespace " << Ns << " {\n";

  if (DirLang.hasEnableBitmaskEnumInNamespace())
    OS << "\nLLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();\n";

  // Emit Directive enumeration
  GenerateEnumClass(DirLang.getDirectives(), OS, "Directive",
                    DirLang.getDirectivePrefix(), DirLang);

  // Emit Clause enumeration
  GenerateEnumClass(DirLang.getClauses(), OS, "Clause",
                    DirLang.getClausePrefix(), DirLang);

  // Emit ClauseVal enumeration
  std::string EnumHelperFuncs;
  GenerateEnumClauseVal(DirLang.getClauses(), OS, DirLang, EnumHelperFuncs);

  // Generic function signatures
  OS << "\n";
  OS << "// Enumeration helper functions\n";
  OS << "Directive get" << DirLang.getName()
     << "DirectiveKind(llvm::StringRef Str);\n";
  OS << "\n";
  OS << "llvm::StringRef get" << DirLang.getName()
     << "DirectiveName(Directive D);\n";
  OS << "\n";
  OS << "Clause get" << DirLang.getName()
     << "ClauseKind(llvm::StringRef Str);\n";
  OS << "\n";
  OS << "llvm::StringRef get" << DirLang.getName() << "ClauseName(Clause C);\n";
  OS << "\n";
  OS << "/// Return true if \\p C is a valid clause for \\p D in version \\p "
     << "Version.\n";
  OS << "bool isAllowedClauseForDirective(Directive D, "
     << "Clause C, unsigned Version);\n";
  OS << "\n";
  if (EnumHelperFuncs.length() > 0) {
    OS << EnumHelperFuncs;
    OS << "\n";
  }

  // Closing namespaces
  for (auto Ns : llvm::reverse(Namespaces))
    OS << "} // namespace " << Ns << "\n";

  OS << "} // namespace llvm\n";

  OS << "#endif // LLVM_" << DirLang.getName() << "_INC\n";
}

// Generate function implementation for get<Enum>Name(StringRef Str)
void GenerateGetName(const std::vector<Record *> &Records, raw_ostream &OS,
                     StringRef Enum, const DirectiveLanguage &DirLang,
                     StringRef Prefix) {
  OS << "\n";
  OS << "llvm::StringRef llvm::" << DirLang.getCppNamespace() << "::get"
     << DirLang.getName() << Enum << "Name(" << Enum << " Kind) {\n";
  OS << "  switch (Kind) {\n";
  for (const auto &R : Records) {
    BaseRecord Rec{R};
    OS << "    case " << Prefix << Rec.getFormattedName() << ":\n";
    OS << "      return \"";
    if (Rec.getAlternativeName().empty())
      OS << Rec.getName();
    else
      OS << Rec.getAlternativeName();
    OS << "\";\n";
  }
  OS << "  }\n"; // switch
  OS << "  llvm_unreachable(\"Invalid " << DirLang.getName() << " " << Enum
     << " kind\");\n";
  OS << "}\n";
}

// Generate function implementation for get<Enum>Kind(StringRef Str)
void GenerateGetKind(const std::vector<Record *> &Records, raw_ostream &OS,
                     StringRef Enum, const DirectiveLanguage &DirLang,
                     StringRef Prefix, bool ImplicitAsUnknown) {

  auto DefaultIt = std::find_if(Records.begin(), Records.end(), [](Record *R) {
    return R->getValueAsBit("isDefault") == true;
  });

  if (DefaultIt == Records.end()) {
    PrintError("At least one " + Enum + " must be defined as default.");
    return;
  }

  BaseRecord DefaultRec{(*DefaultIt)};

  OS << "\n";
  OS << Enum << " llvm::" << DirLang.getCppNamespace() << "::get"
     << DirLang.getName() << Enum << "Kind(llvm::StringRef Str) {\n";
  OS << "  return llvm::StringSwitch<" << Enum << ">(Str)\n";

  for (const auto &R : Records) {
    BaseRecord Rec{R};
    if (ImplicitAsUnknown && R->getValueAsBit("isImplicit")) {
      OS << "    .Case(\"" << Rec.getName() << "\"," << Prefix
         << DefaultRec.getFormattedName() << ")\n";
    } else {
      OS << "    .Case(\"" << Rec.getName() << "\"," << Prefix
         << Rec.getFormattedName() << ")\n";
    }
  }
  OS << "    .Default(" << Prefix << DefaultRec.getFormattedName() << ");\n";
  OS << "}\n";
}

// Generate function implementation for get<ClauseVal>Kind(StringRef Str)
void GenerateGetKindClauseVal(const DirectiveLanguage &DirLang,
                              raw_ostream &OS) {
  for (const auto &R : DirLang.getClauses()) {
    Clause C{R};
    const auto &ClauseVals = C.getClauseVals();
    if (ClauseVals.size() <= 0)
      continue;

    auto DefaultIt =
        std::find_if(ClauseVals.begin(), ClauseVals.end(), [](Record *CV) {
          return CV->getValueAsBit("isDefault") == true;
        });

    if (DefaultIt == ClauseVals.end()) {
      PrintError("At least one val in Clause " + C.getFormattedName() +
                 " must be defined as default.");
      return;
    }
    const auto DefaultName = (*DefaultIt)->getName();

    const auto &EnumName = C.getEnumName();
    if (EnumName.size() == 0) {
      PrintError("enumClauseValue field not set in Clause" +
                 C.getFormattedName() + ".");
      return;
    }

    OS << "\n";
    OS << EnumName << " llvm::" << DirLang.getCppNamespace() << "::get"
       << EnumName << "(llvm::StringRef Str) {\n";
    OS << "  return llvm::StringSwitch<" << EnumName << ">(Str)\n";
    for (const auto &CV : ClauseVals) {
      ClauseVal CVal{CV};
      OS << "    .Case(\"" << CVal.getFormattedName() << "\"," << CV->getName()
         << ")\n";
    }
    OS << "    .Default(" << DefaultName << ");\n";
    OS << "}\n";
  }
}

void GenerateCaseForVersionedClauses(const std::vector<Record *> &Clauses,
                                     raw_ostream &OS, StringRef DirectiveName,
                                     const DirectiveLanguage &DirLang,
                                     llvm::StringSet<> &Cases) {
  for (const auto &C : Clauses) {
    VersionedClause VerClause{C};

    const auto ClauseFormattedName = VerClause.getClause().getFormattedName();

    if (Cases.find(ClauseFormattedName) == Cases.end()) {
      Cases.insert(ClauseFormattedName);
      OS << "        case " << DirLang.getClausePrefix() << ClauseFormattedName
         << ":\n";
      OS << "          return " << VerClause.getMinVersion()
         << " <= Version && " << VerClause.getMaxVersion() << " >= Version;\n";
    }
  }
}

// Generate the isAllowedClauseForDirective function implementation.
void GenerateIsAllowedClause(const DirectiveLanguage &DirLang,
                             raw_ostream &OS) {
  OS << "\n";
  OS << "bool llvm::" << DirLang.getCppNamespace()
     << "::isAllowedClauseForDirective("
     << "Directive D, Clause C, unsigned Version) {\n";
  OS << "  assert(unsigned(D) <= llvm::" << DirLang.getCppNamespace()
     << "::Directive_enumSize);\n";
  OS << "  assert(unsigned(C) <= llvm::" << DirLang.getCppNamespace()
     << "::Clause_enumSize);\n";

  OS << "  switch (D) {\n";

  for (const auto &D : DirLang.getDirectives()) {
    Directive Dir{D};

    OS << "    case " << DirLang.getDirectivePrefix() << Dir.getFormattedName()
       << ":\n";
    if (Dir.getAllowedClauses().size() == 0 &&
        Dir.getAllowedOnceClauses().size() == 0 &&
        Dir.getAllowedExclusiveClauses().size() == 0 &&
        Dir.getRequiredClauses().size() == 0) {
      OS << "      return false;\n";
    } else {
      OS << "      switch (C) {\n";

      llvm::StringSet<> Cases;

      GenerateCaseForVersionedClauses(Dir.getAllowedClauses(), OS,
                                      Dir.getName(), DirLang, Cases);

      GenerateCaseForVersionedClauses(Dir.getAllowedOnceClauses(), OS,
                                      Dir.getName(), DirLang, Cases);

      GenerateCaseForVersionedClauses(Dir.getAllowedExclusiveClauses(), OS,
                                      Dir.getName(), DirLang, Cases);

      GenerateCaseForVersionedClauses(Dir.getRequiredClauses(), OS,
                                      Dir.getName(), DirLang, Cases);

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

// Generate a simple enum set with the give clauses.
void GenerateClauseSet(const std::vector<Record *> &Clauses, raw_ostream &OS,
                       StringRef ClauseSetPrefix, Directive &Dir,
                       const DirectiveLanguage &DirLang) {

  OS << "\n";
  OS << "  static " << DirLang.getClauseEnumSetClass() << " " << ClauseSetPrefix
     << DirLang.getDirectivePrefix() << Dir.getFormattedName() << " {\n";

  for (const auto &C : Clauses) {
    VersionedClause VerClause{C};
    OS << "    llvm::" << DirLang.getCppNamespace()
       << "::Clause::" << DirLang.getClausePrefix()
       << VerClause.getClause().getFormattedName() << ",\n";
  }
  OS << "  };\n";
}

// Generate an enum set for the 4 kinds of clauses linked to a directive.
void GenerateDirectiveClauseSets(const DirectiveLanguage &DirLang,
                                 raw_ostream &OS) {

  IfDefScope Scope("GEN_FLANG_DIRECTIVE_CLAUSE_SETS", OS);

  OS << "\n";
  OS << "namespace llvm {\n";

  // Open namespaces defined in the directive language.
  llvm::SmallVector<StringRef, 2> Namespaces;
  llvm::SplitString(DirLang.getCppNamespace(), Namespaces, "::");
  for (auto Ns : Namespaces)
    OS << "namespace " << Ns << " {\n";

  for (const auto &D : DirLang.getDirectives()) {
    Directive Dir{D};

    OS << "\n";
    OS << "  // Sets for " << Dir.getName() << "\n";

    GenerateClauseSet(Dir.getAllowedClauses(), OS, "allowedClauses_", Dir,
                      DirLang);
    GenerateClauseSet(Dir.getAllowedOnceClauses(), OS, "allowedOnceClauses_",
                      Dir, DirLang);
    GenerateClauseSet(Dir.getAllowedExclusiveClauses(), OS,
                      "allowedExclusiveClauses_", Dir, DirLang);
    GenerateClauseSet(Dir.getRequiredClauses(), OS, "requiredClauses_", Dir,
                      DirLang);
  }

  // Closing namespaces
  for (auto Ns : llvm::reverse(Namespaces))
    OS << "} // namespace " << Ns << "\n";

  OS << "} // namespace llvm\n";
}

// Generate a map of directive (key) with DirectiveClauses struct as values.
// The struct holds the 4 sets of enumeration for the 4 kinds of clauses
// allowances (allowed, allowed once, allowed exclusive and required).
void GenerateDirectiveClauseMap(const DirectiveLanguage &DirLang,
                                raw_ostream &OS) {

  IfDefScope Scope("GEN_FLANG_DIRECTIVE_CLAUSE_MAP", OS);

  OS << "\n";
  OS << "{\n";

  for (const auto &D : DirLang.getDirectives()) {
    Directive Dir{D};
    OS << "  {llvm::" << DirLang.getCppNamespace()
       << "::Directive::" << DirLang.getDirectivePrefix()
       << Dir.getFormattedName() << ",\n";
    OS << "    {\n";
    OS << "      llvm::" << DirLang.getCppNamespace() << "::allowedClauses_"
       << DirLang.getDirectivePrefix() << Dir.getFormattedName() << ",\n";
    OS << "      llvm::" << DirLang.getCppNamespace() << "::allowedOnceClauses_"
       << DirLang.getDirectivePrefix() << Dir.getFormattedName() << ",\n";
    OS << "      llvm::" << DirLang.getCppNamespace()
       << "::allowedExclusiveClauses_" << DirLang.getDirectivePrefix()
       << Dir.getFormattedName() << ",\n";
    OS << "      llvm::" << DirLang.getCppNamespace() << "::requiredClauses_"
       << DirLang.getDirectivePrefix() << Dir.getFormattedName() << ",\n";
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
void GenerateFlangClauseParserClass(const DirectiveLanguage &DirLang,
                                    raw_ostream &OS) {

  IfDefScope Scope("GEN_FLANG_CLAUSE_PARSER_CLASSES", OS);

  OS << "\n";

  for (const auto &C : DirLang.getClauses()) {
    Clause Clause{C};
    // Clause has a non generic class.
    if (!Clause.getFlangClass().empty())
      continue;
    if (!Clause.getFlangClassValue().empty()) {
      OS << "WRAPPER_CLASS(" << Clause.getFormattedParserClassName() << ", ";
      if (Clause.isValueOptional() && Clause.isValueList()) {
        OS << "std::optional<std::list<" << Clause.getFlangClassValue()
           << ">>";
      } else if (Clause.isValueOptional()) {
        OS << "std::optional<" << Clause.getFlangClassValue() << ">";
      } else if (Clause.isValueList()) {
        OS << "std::list<" << Clause.getFlangClassValue() << ">";
      } else {
        OS << Clause.getFlangClassValue();
      }
    } else {
      OS << "EMPTY_CLASS(" << Clause.getFormattedParserClassName();
    }
    OS << ");\n";
  }
}

// Generate a list of the different clause classes for Flang.
void GenerateFlangClauseParserClassList(const DirectiveLanguage &DirLang,
                                        raw_ostream &OS) {

  IfDefScope Scope("GEN_FLANG_CLAUSE_PARSER_CLASSES_LIST", OS);

  OS << "\n";
  llvm::interleaveComma(DirLang.getClauses(), OS, [&](Record *C) {
    Clause Clause{C};
    if (Clause.getFlangClass().empty())
      OS << Clause.getFormattedParserClassName() << "\n";
    else
      OS << Clause.getFlangClass() << "\n";
  });
}

// Generate dump node list for the clauses holding a generic class name.
void GenerateFlangClauseDump(const DirectiveLanguage &DirLang,
                             raw_ostream &OS) {

  IfDefScope Scope("GEN_FLANG_DUMP_PARSE_TREE_CLAUSES", OS);

  OS << "\n";
  for (const auto &C : DirLang.getClauses()) {
    Clause Clause{C};
    // Clause has a non generic class.
    if (!Clause.getFlangClass().empty())
      continue;

    OS << "NODE(" << DirLang.getFlangClauseBaseClass() << ", "
       << Clause.getFormattedParserClassName() << ")\n";
  }
}

// Generate Unparse functions for clauses classes in the Flang parse-tree
// If the clause is a non-generic class, no entry is generated.
void GenerateFlangClauseUnparse(const DirectiveLanguage &DirLang,
                                raw_ostream &OS) {

  IfDefScope Scope("GEN_FLANG_CLAUSE_UNPARSE", OS);

  OS << "\n";

  for (const auto &C : DirLang.getClauses()) {
    Clause Clause{C};
    // Clause has a non generic class.
    if (!Clause.getFlangClass().empty())
      continue;
    if (!Clause.getFlangClassValue().empty()) {
      if (Clause.isValueOptional() && Clause.getDefaultValue().empty()) {
        OS << "void Unparse(const " << DirLang.getFlangClauseBaseClass()
           << "::" << Clause.getFormattedParserClassName() << " &x) {\n";
        OS << "  Word(\"" << Clause.getName().upper() << "\");\n";

        OS << "  Walk(\"(\", x.v, \")\");\n";
        OS << "}\n";
      } else if (Clause.isValueOptional()) {
        OS << "void Unparse(const " << DirLang.getFlangClauseBaseClass()
           << "::" << Clause.getFormattedParserClassName() << " &x) {\n";
        OS << "  Word(\"" << Clause.getName().upper() << "\");\n";
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
        OS << "void Unparse(const " << DirLang.getFlangClauseBaseClass()
           << "::" << Clause.getFormattedParserClassName() << " &x) {\n";
        OS << "  Word(\"" << Clause.getName().upper() << "\");\n";
        OS << "  Put(\"(\");\n";
        if (Clause.isValueList())
          OS << "  Walk(x.v, \",\");\n";
        else
          OS << "  Walk(x.v);\n";
        OS << "  Put(\")\");\n";
        OS << "}\n";
      }
    } else {
      OS << "void Before(const " << DirLang.getFlangClauseBaseClass()
         << "::" << Clause.getFormattedParserClassName() << " &) { Word(\""
         << Clause.getName().upper() << "\"); }\n";
    }
  }
}

// Generate the implemenation section for the enumeration in the directive
// language
void EmitDirectivesFlangImpl(const DirectiveLanguage &DirLang,
                             raw_ostream &OS) {

  GenerateDirectiveClauseSets(DirLang, OS);

  GenerateDirectiveClauseMap(DirLang, OS);

  GenerateFlangClauseParserClass(DirLang, OS);

  GenerateFlangClauseParserClassList(DirLang, OS);

  GenerateFlangClauseDump(DirLang, OS);

  GenerateFlangClauseUnparse(DirLang, OS);
}

// Generate the implemenation section for the enumeration in the directive
// language.
void EmitDirectivesGen(RecordKeeper &Records, raw_ostream &OS) {
  const auto DirLang = DirectiveLanguage{Records};
  if (DirLang.HasValidityErrors())
    return;

  EmitDirectivesFlangImpl(DirLang, OS);
}

// Generate the implemenation for the enumeration in the directive
// language. This code can be included in library.
void EmitDirectivesImpl(RecordKeeper &Records, raw_ostream &OS) {
  const auto DirLang = DirectiveLanguage{Records};
  if (DirLang.HasValidityErrors())
    return;

  if (!DirLang.getIncludeHeader().empty())
    OS << "#include \"" << DirLang.getIncludeHeader() << "\"\n\n";

  OS << "#include \"llvm/ADT/StringRef.h\"\n";
  OS << "#include \"llvm/ADT/StringSwitch.h\"\n";
  OS << "#include \"llvm/Support/ErrorHandling.h\"\n";
  OS << "\n";
  OS << "using namespace llvm;\n";
  llvm::SmallVector<StringRef, 2> Namespaces;
  llvm::SplitString(DirLang.getCppNamespace(), Namespaces, "::");
  for (auto Ns : Namespaces)
    OS << "using namespace " << Ns << ";\n";

  // getDirectiveKind(StringRef Str)
  GenerateGetKind(DirLang.getDirectives(), OS, "Directive", DirLang,
                  DirLang.getDirectivePrefix(), /*ImplicitAsUnknown=*/false);

  // getDirectiveName(Directive Kind)
  GenerateGetName(DirLang.getDirectives(), OS, "Directive", DirLang,
                  DirLang.getDirectivePrefix());

  // getClauseKind(StringRef Str)
  GenerateGetKind(DirLang.getClauses(), OS, "Clause", DirLang,
                  DirLang.getClausePrefix(),
                  /*ImplicitAsUnknown=*/true);

  // getClauseName(Clause Kind)
  GenerateGetName(DirLang.getClauses(), OS, "Clause", DirLang,
                  DirLang.getClausePrefix());

  // get<ClauseVal>Kind(StringRef Str)
  GenerateGetKindClauseVal(DirLang, OS);

  // isAllowedClauseForDirective(Directive D, Clause C, unsigned Version)
  GenerateIsAllowedClause(DirLang, OS);
}

} // namespace llvm
