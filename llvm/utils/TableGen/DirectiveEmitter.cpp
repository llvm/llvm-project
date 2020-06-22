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
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

namespace llvm {
void EmitDirectivesEnums(RecordKeeper &Records, raw_ostream &OS) {

  const auto &DirectiveLanguages =
      Records.getAllDerivedDefinitions("DirectiveLanguage");

  if (DirectiveLanguages.size() != 1) {
    PrintError("A single definition of DirectiveLanguage is needed.");
    return;
  }

  const auto &DirectiveLanguage = DirectiveLanguages[0];
  StringRef languageName = DirectiveLanguage->getValueAsString("name");
  StringRef DirectivePrefix =
      DirectiveLanguage->getValueAsString("directivePrefix");
  StringRef ClausePrefix = DirectiveLanguage->getValueAsString("clausePrefix");
  StringRef CppNamespace = DirectiveLanguage->getValueAsString("cppNamespace");
  bool MakeEnumAvailableInNamespace =
      DirectiveLanguage->getValueAsBit("makeEnumAvailableInNamespace");
  bool EnableBitmaskEnumInNamespace =
      DirectiveLanguage->getValueAsBit("enableBitmaskEnumInNamespace");

  OS << "#ifndef LLVM_" << languageName << "_INC\n";
  OS << "#define LLVM_" << languageName << "_INC\n";

  if (EnableBitmaskEnumInNamespace)
    OS << "#include \"llvm/ADT/BitmaskEnum.h\"\n";

  OS << "namespace llvm {\n";

  // Open namespaces defined in the directive language
  llvm::SmallVector<StringRef, 2> Namespaces;
  llvm::SplitString(CppNamespace, Namespaces, "::");
  for (auto Ns : Namespaces)
    OS << "namespace " << Ns << " {\n";

  if (EnableBitmaskEnumInNamespace)
    OS << "LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();\n";

  // Emit Directive enumeration
  OS << "enum class Directive {\n";
  const auto &Directives = Records.getAllDerivedDefinitions("Directive");
  for (const auto &D : Directives) {
    const auto Name = D->getValueAsString("name");
    std::string N = Name.str();
    std::replace(N.begin(), N.end(), ' ', '_');
    OS << DirectivePrefix << N << ",\n";
  }
  OS << "};\n";

  OS << "static constexpr std::size_t Directive_enumSize = "
     << Directives.size() << ";\n";

  // Emit Clause enumeration
  OS << "enum class Clause {\n";
  const auto &Clauses = Records.getAllDerivedDefinitions("Clause");
  for (const auto &C : Clauses) {
    const auto Name = C->getValueAsString("name");
    OS << ClausePrefix << Name << ",\n";
  }
  OS << "};\n";

  OS << "static constexpr std::size_t Clause_enumSize = " << Clauses.size()
     << ";\n";

  // Make the enum values available in the defined namespace. This allows us to
  // write something like Enum_X if we have a `using namespace <CppNamespace>`.
  // At the same time we do not loose the strong type guarantees of the enum
  // class, that is we cannot pass an unsigned as Directive without an explicit
  // cast.
  if (MakeEnumAvailableInNamespace) {
    for (const auto &D : Directives) {
      const auto Name = D->getValueAsString("name");
      std::string N = Name.str();
      std::replace(N.begin(), N.end(), ' ', '_');
      OS << "constexpr auto " << DirectivePrefix << N << " = " << CppNamespace
         << "::Directive::" << DirectivePrefix << N << ";\n";
    }

    for (const auto &C : Clauses) {
      const auto Name = C->getValueAsString("name");
      OS << "constexpr auto " << ClausePrefix << Name << " = " << CppNamespace
         << "::Clause::" << ClausePrefix << Name << ";\n";
    }
  }

  // Closing namespaces
  for (auto Ns : llvm::reverse(Namespaces))
    OS << "} // namespace " << Ns << "\n";

  OS << "} // namespace llvm\n";

  OS << "#endif";
}
} // namespace llvm
