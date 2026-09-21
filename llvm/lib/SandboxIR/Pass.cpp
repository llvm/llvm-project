//===- Pass.cpp - Passes that operate on Sandbox IR -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/Pass.h"
#include "llvm/Support/Debug.h"

using namespace llvm::sandboxir;

#ifndef NDEBUG
void Pass::dump() const {
  print(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

bool AuxPassArg::set(bool NewVal) {
  Registry->Entries[ArgIdx].Val = NewVal;
  return NewVal;
}

bool AuxPassArg::get() const { return Registry->Entries[ArgIdx].Val; }

llvm::StringRef AuxPassArg::getFlagStr() const {
  return Registry->Entries[ArgIdx].FlagStr;
}

#ifndef NDEBUG
void AuxPassArg::print(raw_ostream &OS) const {
  auto &E = Registry->Entries[ArgIdx];
  OS << E.FlagStr << " : " << E.Val;
}

void AuxPassArg::dump() const {
  print(dbgs());
  dbgs() << "\n";
}
#endif

AuxPassArgsRegistry::Entry *AuxPassArgsRegistry::getEntry(StringRef Flag) {
  for (Entry &E : Entries) {
    if (E.FlagStr == Flag)
      return &E;
  }
  return nullptr;
}

AuxPassArg AuxPassArgsRegistry::createArg(StringRef Flag) {
  AuxPassArg NewArg(Entries.size(), this);
  Entries.emplace_back(Flag, false);
  return NewArg;
}

void AuxPassArgsRegistry::parse(StringRef ArgsStr) {
  SmallVector<StringRef> Parts;
  ArgsStr.split(Parts, ',');
  for (StringRef Part : Parts) {
    if (Part.empty())
      continue;
    Entry *E = getEntry(Part);
    if (E == nullptr) {
      std::string ErrStr;
      raw_string_ostream ErrSS(ErrStr);
      ErrSS << "Unsupported argument: '" << Part
            << "'. List of supported args:\n";
      for (const auto &[ArgStr, Val] : Entries)
        ErrSS << "  '" << ArgStr << "'\n";
      reportFatalUsageError(ErrStr.c_str());
    }
    *E = {Part, true};
  }
}

#ifndef NDEBUG
void AuxPassArgsRegistry::print(raw_ostream &OS) const {
  if (Entries.empty()) {
    OS << "No args created yet!\n";
    return;
  }
  for (const Entry &E : Entries)
    OS << E.FlagStr << " : " << E.Val << "\n";
}

void AuxPassArgsRegistry::dump() const {
  print(dbgs());
  dbgs() << "\n";
}
#endif
