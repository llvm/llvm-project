//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLDBTableGenBackends.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;
using namespace lldb_private;

static void emitEnumValue(const Record *EnumValue, StringRef Prefix,
                          raw_ostream &OS) {
  OS << "  {";
  OS << Prefix << EnumValue->getValueAsString("Value") << ',';
  OS << '"' << EnumValue->getValueAsString("Name") << "\",";
  OS << '"';
  printEscapedString(EnumValue->getValueAsString("Description"), OS);
  OS << '"';
  OS << "},\n";
}

static void emitEnum(const Record *Enum, raw_ostream &OS) {
  StringRef ID = Enum->getName();

  std::string NeededMacro = ("LLDB_ENUMS_" + ID).str();
  StringRef Prefix = Enum->getValueAsString("Prefix");
  OS << "#ifdef " << NeededMacro << "\n";
  OS << "constexpr static OptionEnumValueElement g_" << ID << "[] = {\n";
  for (const Record *EnumValue : Enum->getValueAsListOfDefs("Values"))
    emitEnumValue(EnumValue, Prefix, OS);
  OS << "};\n";
  OS << "#undef " << NeededMacro << "\n";
  OS << "#endif\n";
}

void lldb_private::EmitOptionEnumDefs(const RecordKeeper &Records,
                                      raw_ostream &OS) {
  emitSourceFileHeader("Enums for LLDB command line options and properties.",
                       OS, Records);

  ArrayRef<const Record *> Enums = Records.getAllDerivedDefinitions("EnumDef");
  for (const Record *Enum : Enums)
    emitEnum(Enum, OS);
}
