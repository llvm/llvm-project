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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
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

// Get Directive or Clause name formatted by replacing whitespaces with
// underscores.
std::string getFormattedName(StringRef Name) {
  std::string N = Name.str();
  std::replace(N.begin(), N.end(), ' ', '_');
  return N;
}

// Generate enum class
void GenerateEnumClass(const std::vector<Record *> &Records, raw_ostream &OS,
                       StringRef Enum, StringRef Prefix, StringRef CppNamespace,
                       bool MakeEnumAvailableInNamespace) {
  OS << "\n";
  OS << "enum class " << Enum << " {\n";
  for (const auto &R : Records) {
    const auto Name = R->getValueAsString("name");
    OS << "  " << Prefix << getFormattedName(Name) << ",\n";
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
  if (MakeEnumAvailableInNamespace) {
    OS << "\n";
    for (const auto &R : Records) {
      const auto FormattedName = getFormattedName(R->getValueAsString("name"));
      OS << "constexpr auto " << Prefix << FormattedName << " = "
         << "llvm::" << CppNamespace << "::" << Enum << "::" << Prefix
         << FormattedName << ";\n";
    }
  }
}

// Generate the declaration section for the enumeration in the directive
// language
void EmitDirectivesDecl(RecordKeeper &Records, raw_ostream &OS) {

  const auto &DirectiveLanguages =
      Records.getAllDerivedDefinitions("DirectiveLanguage");

  if (DirectiveLanguages.size() != 1) {
    PrintError("A single definition of DirectiveLanguage is needed.");
    return;
  }

  const auto &DirectiveLanguage = DirectiveLanguages[0];
  StringRef LanguageName = DirectiveLanguage->getValueAsString("name");
  StringRef DirectivePrefix =
      DirectiveLanguage->getValueAsString("directivePrefix");
  StringRef ClausePrefix = DirectiveLanguage->getValueAsString("clausePrefix");
  StringRef CppNamespace = DirectiveLanguage->getValueAsString("cppNamespace");
  bool MakeEnumAvailableInNamespace =
      DirectiveLanguage->getValueAsBit("makeEnumAvailableInNamespace");
  bool EnableBitmaskEnumInNamespace =
      DirectiveLanguage->getValueAsBit("enableBitmaskEnumInNamespace");

  OS << "#ifndef LLVM_" << LanguageName << "_INC\n";
  OS << "#define LLVM_" << LanguageName << "_INC\n";

  if (EnableBitmaskEnumInNamespace)
    OS << "\n#include \"llvm/ADT/BitmaskEnum.h\"\n";

  OS << "\n";
  OS << "namespace llvm {\n";
  OS << "class StringRef;\n";

  // Open namespaces defined in the directive language
  llvm::SmallVector<StringRef, 2> Namespaces;
  llvm::SplitString(CppNamespace, Namespaces, "::");
  for (auto Ns : Namespaces)
    OS << "namespace " << Ns << " {\n";

  if (EnableBitmaskEnumInNamespace)
    OS << "\nLLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();\n";

  // Emit Directive enumeration
  const auto &Directives = Records.getAllDerivedDefinitions("Directive");
  GenerateEnumClass(Directives, OS, "Directive", DirectivePrefix, CppNamespace,
                    MakeEnumAvailableInNamespace);

  // Emit Clause enumeration
  const auto &Clauses = Records.getAllDerivedDefinitions("Clause");
  GenerateEnumClass(Clauses, OS, "Clause", ClausePrefix, CppNamespace,
                    MakeEnumAvailableInNamespace);

  // Generic function signatures
  OS << "\n";
  OS << "// Enumeration helper functions\n";
  OS << "Directive get" << LanguageName
     << "DirectiveKind(llvm::StringRef Str);\n";
  OS << "\n";
  OS << "llvm::StringRef get" << LanguageName
     << "DirectiveName(Directive D);\n";
  OS << "\n";
  OS << "Clause get" << LanguageName << "ClauseKind(llvm::StringRef Str);\n";
  OS << "\n";
  OS << "llvm::StringRef get" << LanguageName << "ClauseName(Clause C);\n";
  OS << "\n";
  OS << "/// Return true if \\p C is a valid clause for \\p D in version \\p "
     << "Version.\n";
  OS << "bool isAllowedClauseForDirective(Directive D, "
     << "Clause C, unsigned Version);\n";
  OS << "\n";

  // Closing namespaces
  for (auto Ns : llvm::reverse(Namespaces))
    OS << "} // namespace " << Ns << "\n";

  OS << "} // namespace llvm\n";

  OS << "#endif // LLVM_" << LanguageName << "_INC\n";
}

// Generate function implementation for get<Enum>Name(StringRef Str)
void GenerateGetName(const std::vector<Record *> &Records, raw_ostream &OS,
                     StringRef Enum, StringRef Prefix, StringRef LanguageName,
                     StringRef Namespace) {
  OS << "\n";
  OS << "llvm::StringRef llvm::" << Namespace << "::get" << LanguageName << Enum
     << "Name(" << Enum << " Kind) {\n";
  OS << "  switch (Kind) {\n";
  for (const auto &R : Records) {
    const auto Name = R->getValueAsString("name");
    const auto AlternativeName = R->getValueAsString("alternativeName");
    OS << "    case " << Prefix << getFormattedName(Name) << ":\n";
    OS << "      return \"";
    if (AlternativeName.empty())
      OS << Name;
    else
      OS << AlternativeName;
    OS << "\";\n";
  }
  OS << "  }\n"; // switch
  OS << "  llvm_unreachable(\"Invalid " << LanguageName << " " << Enum
     << " kind\");\n";
  OS << "}\n";
}

// Generate function implementation for get<Enum>Kind(StringRef Str)
void GenerateGetKind(const std::vector<Record *> &Records, raw_ostream &OS,
                     StringRef Enum, StringRef Prefix, StringRef LanguageName,
                     StringRef Namespace, bool ImplicitAsUnknown) {

  auto DefaultIt = std::find_if(Records.begin(), Records.end(), [](Record *R) {
    return R->getValueAsBit("isDefault") == true;
  });

  if (DefaultIt == Records.end()) {
    PrintError("A least one " + Enum + " must be defined as default.");
    return;
  }

  const auto FormattedDefaultName =
      getFormattedName((*DefaultIt)->getValueAsString("name"));

  OS << "\n";
  OS << Enum << " llvm::" << Namespace << "::get" << LanguageName << Enum
     << "Kind(llvm::StringRef Str) {\n";
  OS << "  return llvm::StringSwitch<" << Enum << ">(Str)\n";

  for (const auto &R : Records) {
    const auto Name = R->getValueAsString("name");
    if (ImplicitAsUnknown && R->getValueAsBit("isImplicit")) {
      OS << "    .Case(\"" << Name << "\"," << Prefix << FormattedDefaultName
         << ")\n";
    } else {
      OS << "    .Case(\"" << Name << "\"," << Prefix << getFormattedName(Name)
         << ")\n";
    }
  }
  OS << "    .Default(" << Prefix << FormattedDefaultName << ");\n";
  OS << "}\n";
}

void GenerateCaseForVersionedClauses(const std::vector<Record *> &Clauses,
                                     raw_ostream &OS, StringRef DirectiveName,
                                     StringRef DirectivePrefix,
                                     StringRef ClausePrefix,
                                     llvm::StringSet<> &Cases) {
  for (const auto &C : Clauses) {
    const auto MinVersion = C->getValueAsInt("minVersion");
    const auto MaxVersion = C->getValueAsInt("maxVersion");
    const auto SpecificClause = C->getValueAsDef("clause");
    const auto ClauseName =
        getFormattedName(SpecificClause->getValueAsString("name"));

    if (Cases.find(ClauseName) == Cases.end()) {
      Cases.insert(ClauseName);
      OS << "        case " << ClausePrefix << ClauseName << ":\n";
      OS << "          return " << MinVersion << " <= Version && " << MaxVersion
         << " >= Version;\n";
    }
  }
}

// Generate the isAllowedClauseForDirective function implementation.
void GenerateIsAllowedClause(const std::vector<Record *> &Directives,
                             raw_ostream &OS, StringRef LanguageName,
                             StringRef DirectivePrefix, StringRef ClausePrefix,
                             StringRef CppNamespace) {
  OS << "\n";
  OS << "bool llvm::" << CppNamespace << "::isAllowedClauseForDirective("
     << "Directive D, Clause C, unsigned Version) {\n";
  OS << "  assert(unsigned(D) <= llvm::" << CppNamespace
     << "::Directive_enumSize);\n";
  OS << "  assert(unsigned(C) <= llvm::" << CppNamespace
     << "::Clause_enumSize);\n";

  OS << "  switch (D) {\n";

  for (const auto &D : Directives) {

    const auto DirectiveName = D->getValueAsString("name");
    const auto &AllowedClauses = D->getValueAsListOfDefs("allowedClauses");
    const auto &AllowedOnceClauses =
        D->getValueAsListOfDefs("allowedOnceClauses");
    const auto &AllowedExclusiveClauses =
        D->getValueAsListOfDefs("allowedExclusiveClauses");
    const auto &RequiredClauses = D->getValueAsListOfDefs("requiredClauses");

    OS << "    case " << DirectivePrefix << getFormattedName(DirectiveName)
       << ":\n";
    if (AllowedClauses.size() == 0 && AllowedOnceClauses.size() == 0 &&
        AllowedExclusiveClauses.size() == 0 && RequiredClauses.size() == 0) {
      OS << "      return false;\n";
    } else {
      OS << "      switch (C) {\n";

      llvm::StringSet<> Cases;

      GenerateCaseForVersionedClauses(AllowedClauses, OS, DirectiveName,
                                      DirectivePrefix, ClausePrefix, Cases);

      GenerateCaseForVersionedClauses(AllowedOnceClauses, OS, DirectiveName,
                                      DirectivePrefix, ClausePrefix, Cases);

      GenerateCaseForVersionedClauses(AllowedExclusiveClauses, OS,
                                      DirectiveName, DirectivePrefix,
                                      ClausePrefix, Cases);

      GenerateCaseForVersionedClauses(RequiredClauses, OS, DirectiveName,
                                      DirectivePrefix, ClausePrefix, Cases);

      OS << "        default:\n";
      OS << "          return false;\n";
      OS << "      }\n"; // End of clauses switch
    }
    OS << "      break;\n";
  }

  OS << "  }\n"; // End of directives switch
  OS << "  llvm_unreachable(\"Invalid " << LanguageName
     << " Directive kind\");\n";
  OS << "}\n"; // End of function isAllowedClauseForDirective
}

// Generate a simple enum set with the give clauses.
void GenerateClauseSet(const std::vector<Record *> &Clauses, raw_ostream &OS,
                       StringRef ClauseEnumSetClass, StringRef ClauseSetPrefix,
                       StringRef DirectiveName, StringRef DirectivePrefix,
                       StringRef ClausePrefix, StringRef CppNamespace) {

  OS << "\n";
  OS << "  static " << ClauseEnumSetClass << " " << ClauseSetPrefix
     << DirectivePrefix << getFormattedName(DirectiveName) << " {\n";

  for (const auto &C : Clauses) {
    const auto SpecificClause = C->getValueAsDef("clause");
    const auto ClauseName = SpecificClause->getValueAsString("name");
    OS << "    llvm::" << CppNamespace << "::Clause::" << ClausePrefix
       << getFormattedName(ClauseName) << ",\n";
  }
  OS << "  };\n";
}

// Generate an enum set for the 4 kinds of clauses linked to a directive.
void GenerateDirectiveClauseSets(const std::vector<Record *> &Directives,
                                 raw_ostream &OS, StringRef LanguageName,
                                 StringRef ClauseEnumSetClass,
                                 StringRef DirectivePrefix,
                                 StringRef ClausePrefix,
                                 StringRef CppNamespace) {

  IfDefScope Scope("GEN_FLANG_DIRECTIVE_CLAUSE_SETS", OS);

  OS << "\n";
  OS << "namespace llvm {\n";

  // Open namespaces defined in the directive language.
  llvm::SmallVector<StringRef, 2> Namespaces;
  llvm::SplitString(CppNamespace, Namespaces, "::");
  for (auto Ns : Namespaces)
    OS << "namespace " << Ns << " {\n";

  for (const auto &D : Directives) {
    const auto DirectiveName = D->getValueAsString("name");

    const auto &AllowedClauses = D->getValueAsListOfDefs("allowedClauses");
    const auto &AllowedOnceClauses =
        D->getValueAsListOfDefs("allowedOnceClauses");
    const auto &AllowedExclusiveClauses =
        D->getValueAsListOfDefs("allowedExclusiveClauses");
    const auto &RequiredClauses = D->getValueAsListOfDefs("requiredClauses");

    OS << "\n";
    OS << "  // Sets for " << DirectiveName << "\n";

    GenerateClauseSet(AllowedClauses, OS, ClauseEnumSetClass, "allowedClauses_",
                      DirectiveName, DirectivePrefix, ClausePrefix,
                      CppNamespace);
    GenerateClauseSet(AllowedOnceClauses, OS, ClauseEnumSetClass,
                      "allowedOnceClauses_", DirectiveName, DirectivePrefix,
                      ClausePrefix, CppNamespace);
    GenerateClauseSet(AllowedExclusiveClauses, OS, ClauseEnumSetClass,
                      "allowedExclusiveClauses_", DirectiveName,
                      DirectivePrefix, ClausePrefix, CppNamespace);
    GenerateClauseSet(RequiredClauses, OS, ClauseEnumSetClass,
                      "requiredClauses_", DirectiveName, DirectivePrefix,
                      ClausePrefix, CppNamespace);
  }

  // Closing namespaces
  for (auto Ns : llvm::reverse(Namespaces))
    OS << "} // namespace " << Ns << "\n";

  OS << "} // namespace llvm\n";
}

// Generate a map of directive (key) with DirectiveClauses struct as values.
// The struct holds the 4 sets of enumeration for the 4 kinds of clauses
// allowances (allowed, allowed once, allowed exclusive and required).
void GenerateDirectiveClauseMap(const std::vector<Record *> &Directives,
                                raw_ostream &OS, StringRef LanguageName,
                                StringRef ClauseEnumSetClass,
                                StringRef DirectivePrefix,
                                StringRef ClausePrefix,
                                StringRef CppNamespace) {

  IfDefScope Scope("GEN_FLANG_DIRECTIVE_CLAUSE_MAP", OS);

  OS << "\n";
  OS << "struct " << LanguageName << "DirectiveClauses {\n";
  OS << "  const " << ClauseEnumSetClass << " allowed;\n";
  OS << "  const " << ClauseEnumSetClass << " allowedOnce;\n";
  OS << "  const " << ClauseEnumSetClass << " allowedExclusive;\n";
  OS << "  const " << ClauseEnumSetClass << " requiredOneOf;\n";
  OS << "};\n";

  OS << "\n";

  OS << "std::unordered_map<llvm::" << CppNamespace << "::Directive, "
     << LanguageName << "DirectiveClauses>\n";
  OS << "    directiveClausesTable = {\n";

  for (const auto &D : Directives) {
    const auto FormattedDirectiveName =
        getFormattedName(D->getValueAsString("name"));
    OS << "  {llvm::" << CppNamespace << "::Directive::" << DirectivePrefix
       << FormattedDirectiveName << ",\n";
    OS << "    {\n";
    OS << "      llvm::" << CppNamespace << "::allowedClauses_"
       << DirectivePrefix << FormattedDirectiveName << ",\n";
    OS << "      llvm::" << CppNamespace << "::allowedOnceClauses_"
       << DirectivePrefix << FormattedDirectiveName << ",\n";
    OS << "      llvm::" << CppNamespace << "::allowedExclusiveClauses_"
       << DirectivePrefix << FormattedDirectiveName << ",\n";
    OS << "      llvm::" << CppNamespace << "::requiredClauses_"
       << DirectivePrefix << FormattedDirectiveName << ",\n";
    OS << "    }\n";
    OS << "  },\n";
  }

  OS << "};\n";
}

// Generate the implemenation section for the enumeration in the directive
// language
void EmitDirectivesFlangImpl(const std::vector<Record *> &Directives,
                             raw_ostream &OS, StringRef LanguageName,
                             StringRef ClauseEnumSetClass,
                             StringRef DirectivePrefix, StringRef ClausePrefix,
                             StringRef CppNamespace) {

  GenerateDirectiveClauseSets(Directives, OS, LanguageName, ClauseEnumSetClass,
                              DirectivePrefix, ClausePrefix, CppNamespace);

  GenerateDirectiveClauseMap(Directives, OS, LanguageName, ClauseEnumSetClass,
                             DirectivePrefix, ClausePrefix, CppNamespace);
}

// Generate the implemenation section for the enumeration in the directive
// language.
void EmitDirectivesGen(RecordKeeper &Records, raw_ostream &OS) {

  const auto &DirectiveLanguages =
      Records.getAllDerivedDefinitions("DirectiveLanguage");

  if (DirectiveLanguages.size() != 1) {
    PrintError("A single definition of DirectiveLanguage is needed.");
    return;
  }

  const auto &DirectiveLanguage = DirectiveLanguages[0];
  StringRef DirectivePrefix =
      DirectiveLanguage->getValueAsString("directivePrefix");
  StringRef LanguageName = DirectiveLanguage->getValueAsString("name");
  StringRef ClausePrefix = DirectiveLanguage->getValueAsString("clausePrefix");
  StringRef CppNamespace = DirectiveLanguage->getValueAsString("cppNamespace");
  StringRef ClauseEnumSetClass =
      DirectiveLanguage->getValueAsString("clauseEnumSetClass");

  const auto &Directives = Records.getAllDerivedDefinitions("Directive");

  EmitDirectivesFlangImpl(Directives, OS, LanguageName, ClauseEnumSetClass,
                          DirectivePrefix, ClausePrefix, CppNamespace);
}

// Generate the implemenation for the enumeration in the directive
// language. This code can be included in library.
void EmitDirectivesImpl(RecordKeeper &Records, raw_ostream &OS) {

  const auto &DirectiveLanguages =
      Records.getAllDerivedDefinitions("DirectiveLanguage");

  if (DirectiveLanguages.size() != 1) {
    PrintError("A single definition of DirectiveLanguage is needed.");
    return;
  }

  const auto &DirectiveLanguage = DirectiveLanguages[0];
  StringRef DirectivePrefix =
      DirectiveLanguage->getValueAsString("directivePrefix");
  StringRef LanguageName = DirectiveLanguage->getValueAsString("name");
  StringRef ClausePrefix = DirectiveLanguage->getValueAsString("clausePrefix");
  StringRef CppNamespace = DirectiveLanguage->getValueAsString("cppNamespace");
  const auto &Directives = Records.getAllDerivedDefinitions("Directive");
  const auto &Clauses = Records.getAllDerivedDefinitions("Clause");

  StringRef IncludeHeader =
      DirectiveLanguage->getValueAsString("includeHeader");

  if (!IncludeHeader.empty())
    OS << "#include \"" << IncludeHeader << "\"\n\n";

  OS << "#include \"llvm/ADT/StringRef.h\"\n";
  OS << "#include \"llvm/ADT/StringSwitch.h\"\n";
  OS << "#include \"llvm/Support/ErrorHandling.h\"\n";
  OS << "\n";
  OS << "using namespace llvm;\n";
  llvm::SmallVector<StringRef, 2> Namespaces;
  llvm::SplitString(CppNamespace, Namespaces, "::");
  for (auto Ns : Namespaces)
    OS << "using namespace " << Ns << ";\n";

  // getDirectiveKind(StringRef Str)
  GenerateGetKind(Directives, OS, "Directive", DirectivePrefix, LanguageName,
                  CppNamespace, /*ImplicitAsUnknown=*/false);

  // getDirectiveName(Directive Kind)
  GenerateGetName(Directives, OS, "Directive", DirectivePrefix, LanguageName,
                  CppNamespace);

  // getClauseKind(StringRef Str)
  GenerateGetKind(Clauses, OS, "Clause", ClausePrefix, LanguageName,
                  CppNamespace, /*ImplicitAsUnknown=*/true);

  // getClauseName(Clause Kind)
  GenerateGetName(Clauses, OS, "Clause", ClausePrefix, LanguageName,
                  CppNamespace);

  // isAllowedClauseForDirective(Directive D, Clause C, unsigned Version)
  GenerateIsAllowedClause(Directives, OS, LanguageName, DirectivePrefix,
                          ClausePrefix, CppNamespace);
}

} // namespace llvm
