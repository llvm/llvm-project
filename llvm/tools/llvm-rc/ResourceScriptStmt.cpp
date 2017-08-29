//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
//
// This implements methods defined in ResourceScriptStmt.h.
//
// Ref: msdn.microsoft.com/en-us/library/windows/desktop/aa380599(v=vs.85).aspx
//
//===---------------------------------------------------------------------===//

#include "ResourceScriptStmt.h"

namespace llvm {
namespace rc {

raw_ostream &operator<<(raw_ostream &OS, const IntOrString &Item) {
  if (Item.IsInt)
    return OS << Item.Data.Int;
  else
    return OS << Item.Data.String;
}

raw_ostream &OptionalStmtList::log(raw_ostream &OS) const {
  for (const auto &Stmt : Statements) {
    OS << "  Option: ";
    Stmt->log(OS);
  }
  return OS;
}

raw_ostream &LanguageResource::log(raw_ostream &OS) const {
  return OS << "Language: " << Lang << ", Sublanguage: " << SubLang << "\n";
}

raw_ostream &CursorResource::log(raw_ostream &OS) const {
  return OS << "Cursor (" << ResName << "): " << CursorLoc << "\n";
}

raw_ostream &IconResource::log(raw_ostream &OS) const {
  return OS << "Icon (" << ResName << "): " << IconLoc << "\n";
}

raw_ostream &HTMLResource::log(raw_ostream &OS) const {
  return OS << "HTML (" << ResName << "): " << HTMLLoc << "\n";
}

raw_ostream &StringTableResource::log(raw_ostream &OS) const {
  OS << "StringTable:\n";
  OptStatements.log(OS);
  for (const auto &String : Table)
    OS << "  " << String.first << " => " << String.second << "\n";
  return OS;
}

raw_ostream &CharacteristicsStmt::log(raw_ostream &OS) const {
  return OS << "Characteristics: " << Value << "\n";
}

raw_ostream &VersionStmt::log(raw_ostream &OS) const {
  return OS << "Version: " << Value << "\n";
}

} // namespace rc
} // namespace llvm
