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

// Wrapper class that contains DirectiveLanguage's information defined in
// DirectiveBase.td and provides helper methods for accessing it.
class DirectiveLanguage {
public:
  explicit DirectiveLanguage(const llvm::Record *Def) : Def(Def) {}

  StringRef getName() const { return Def->getValueAsString("name"); }

  StringRef getCppNamespace() const {
    return Def->getValueAsString("cppNamespace");
  }

  StringRef getDirectivePrefix() const {
    return Def->getValueAsString("directivePrefix");
  }

  StringRef getClausePrefix() const {
    return Def->getValueAsString("clausePrefix");
  }

  StringRef getIncludeHeader() const {
    return Def->getValueAsString("includeHeader");
  }

  StringRef getClauseEnumSetClass() const {
    return Def->getValueAsString("clauseEnumSetClass");
  }

  bool hasMakeEnumAvailableInNamespace() const {
    return Def->getValueAsBit("makeEnumAvailableInNamespace");
  }

  bool hasEnableBitmaskEnumInNamespace() const {
    return Def->getValueAsBit("enableBitmaskEnumInNamespace");
  }

private:
  const llvm::Record *Def;
};

// Base record class used for Directive and Clause class defined in
// DirectiveBase.td.
class BaseRecord {
public:
  explicit BaseRecord(const llvm::Record *Def) : Def(Def) {}

  StringRef getName() const { return Def->getValueAsString("name"); }

  StringRef getAlternativeName() const {
    return Def->getValueAsString("alternativeName");
  }

  // Returns the name of the directive formatted for output. Whitespace are
  // replaced with underscores.
  std::string getFormattedName() {
    StringRef Name = Def->getValueAsString("name");
    std::string N = Name.str();
    std::replace(N.begin(), N.end(), ' ', '_');
    return N;
  }

  bool isDefault() const { return Def->getValueAsBit("isDefault"); }

protected:
  const llvm::Record *Def;
};

// Wrapper class that contains a Directive's information defined in
// DirectiveBase.td and provides helper methods for accessing it.
class Directive : public BaseRecord {
public:
  explicit Directive(const llvm::Record *Def) : BaseRecord(Def) {}

  std::vector<Record *> getAllowedClauses() const {
    return Def->getValueAsListOfDefs("allowedClauses");
  }

  std::vector<Record *> getAllowedOnceClauses() const {
    return Def->getValueAsListOfDefs("allowedOnceClauses");
  }

  std::vector<Record *> getAllowedExclusiveClauses() const {
    return Def->getValueAsListOfDefs("allowedExclusiveClauses");
  }

  std::vector<Record *> getRequiredClauses() const {
    return Def->getValueAsListOfDefs("requiredClauses");
  }
};

// Wrapper class that contains Clause's information defined in DirectiveBase.td
// and provides helper methods for accessing it.
class Clause : public BaseRecord {
public:
  explicit Clause(const llvm::Record *Def) : BaseRecord(Def) {}

  // Optional field.
  StringRef getClangClass() const {
    return Def->getValueAsString("clangClass");
  }

  // Optional field.
  StringRef getFlangClass() const {
    return Def->getValueAsString("flangClass");
  }

  bool isImplict() const { return Def->getValueAsBit("isImplicit"); }
};

// Wrapper class that contains VersionedClause's information defined in
// DirectiveBase.td and provides helper methods for accessing it.
class VersionedClause {
public:
  explicit VersionedClause(const llvm::Record *Def) : Def(Def) {}

  // Return the specific clause record wrapped in the Clause class.
  Clause getClause() const { return Clause{Def->getValueAsDef("clause")}; }

  int64_t getMinVersion() const { return Def->getValueAsInt("minVersion"); }

  int64_t getMaxVersion() const { return Def->getValueAsInt("maxVersion"); }

private:
  const llvm::Record *Def;
};

// Generate enum class
void GenerateEnumClass(const std::vector<Record *> &Records, raw_ostream &OS,
                       StringRef Enum, StringRef Prefix,
                       DirectiveLanguage &DirLang) {
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

// Generate the declaration section for the enumeration in the directive
// language
void EmitDirectivesDecl(RecordKeeper &Records, raw_ostream &OS) {

  const auto &DirectiveLanguages =
      Records.getAllDerivedDefinitions("DirectiveLanguage");

  if (DirectiveLanguages.size() != 1) {
    PrintError("A single definition of DirectiveLanguage is needed.");
    return;
  }

  DirectiveLanguage DirLang{DirectiveLanguages[0]};

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
  const auto &Directives = Records.getAllDerivedDefinitions("Directive");
  GenerateEnumClass(Directives, OS, "Directive", DirLang.getDirectivePrefix(),
                    DirLang);

  // Emit Clause enumeration
  const auto &Clauses = Records.getAllDerivedDefinitions("Clause");
  GenerateEnumClass(Clauses, OS, "Clause", DirLang.getClausePrefix(), DirLang);

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

  // Closing namespaces
  for (auto Ns : llvm::reverse(Namespaces))
    OS << "} // namespace " << Ns << "\n";

  OS << "} // namespace llvm\n";

  OS << "#endif // LLVM_" << DirLang.getName() << "_INC\n";
}

// Generate function implementation for get<Enum>Name(StringRef Str)
void GenerateGetName(const std::vector<Record *> &Records, raw_ostream &OS,
                     StringRef Enum, DirectiveLanguage &DirLang,
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
                     StringRef Enum, DirectiveLanguage &DirLang,
                     StringRef Prefix, bool ImplicitAsUnknown) {

  auto DefaultIt = std::find_if(Records.begin(), Records.end(), [](Record *R) {
    return R->getValueAsBit("isDefault") == true;
  });

  if (DefaultIt == Records.end()) {
    PrintError("A least one " + Enum + " must be defined as default.");
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

void GenerateCaseForVersionedClauses(const std::vector<Record *> &Clauses,
                                     raw_ostream &OS, StringRef DirectiveName,
                                     DirectiveLanguage &DirLang,
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
void GenerateIsAllowedClause(const std::vector<Record *> &Directives,
                             raw_ostream &OS, DirectiveLanguage &DirLang) {
  OS << "\n";
  OS << "bool llvm::" << DirLang.getCppNamespace()
     << "::isAllowedClauseForDirective("
     << "Directive D, Clause C, unsigned Version) {\n";
  OS << "  assert(unsigned(D) <= llvm::" << DirLang.getCppNamespace()
     << "::Directive_enumSize);\n";
  OS << "  assert(unsigned(C) <= llvm::" << DirLang.getCppNamespace()
     << "::Clause_enumSize);\n";

  OS << "  switch (D) {\n";

  for (const auto &D : Directives) {
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
                       DirectiveLanguage &DirLang) {

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
void GenerateDirectiveClauseSets(const std::vector<Record *> &Directives,
                                 raw_ostream &OS, DirectiveLanguage &DirLang) {

  IfDefScope Scope("GEN_FLANG_DIRECTIVE_CLAUSE_SETS", OS);

  OS << "\n";
  OS << "namespace llvm {\n";

  // Open namespaces defined in the directive language.
  llvm::SmallVector<StringRef, 2> Namespaces;
  llvm::SplitString(DirLang.getCppNamespace(), Namespaces, "::");
  for (auto Ns : Namespaces)
    OS << "namespace " << Ns << " {\n";

  for (const auto &D : Directives) {
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
void GenerateDirectiveClauseMap(const std::vector<Record *> &Directives,
                                raw_ostream &OS, DirectiveLanguage &DirLang) {

  IfDefScope Scope("GEN_FLANG_DIRECTIVE_CLAUSE_MAP", OS);

  OS << "\n";
  OS << "{\n";

  for (const auto &D : Directives) {
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

// Generate the implemenation section for the enumeration in the directive
// language
void EmitDirectivesFlangImpl(const std::vector<Record *> &Directives,
                             raw_ostream &OS,
                             DirectiveLanguage &DirectiveLanguage) {

  GenerateDirectiveClauseSets(Directives, OS, DirectiveLanguage);

  GenerateDirectiveClauseMap(Directives, OS, DirectiveLanguage);
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

  const auto &Directives = Records.getAllDerivedDefinitions("Directive");
  DirectiveLanguage DirectiveLanguage{DirectiveLanguages[0]};
  EmitDirectivesFlangImpl(Directives, OS, DirectiveLanguage);
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

  const auto &Directives = Records.getAllDerivedDefinitions("Directive");

  DirectiveLanguage DirLang = DirectiveLanguage{DirectiveLanguages[0]};

  const auto &Clauses = Records.getAllDerivedDefinitions("Clause");

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
  GenerateGetKind(Directives, OS, "Directive", DirLang,
                  DirLang.getDirectivePrefix(), /*ImplicitAsUnknown=*/false);

  // getDirectiveName(Directive Kind)
  GenerateGetName(Directives, OS, "Directive", DirLang,
                  DirLang.getDirectivePrefix());

  // getClauseKind(StringRef Str)
  GenerateGetKind(Clauses, OS, "Clause", DirLang, DirLang.getClausePrefix(),
                  /*ImplicitAsUnknown=*/true);

  // getClauseName(Clause Kind)
  GenerateGetName(Clauses, OS, "Clause", DirLang, DirLang.getClausePrefix());

  // isAllowedClauseForDirective(Directive D, Clause C, unsigned Version)
  GenerateIsAllowedClause(Directives, OS, DirLang);
}

} // namespace llvm
