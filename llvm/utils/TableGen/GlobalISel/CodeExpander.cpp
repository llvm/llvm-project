//===- CodeExpander.cpp - Expand variables in a string --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file Expand the variables in a string.
//
//===----------------------------------------------------------------------===//

#include "CodeExpander.h"
#include "CodeExpansions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"

using namespace llvm;

void CodeExpander::emit(raw_ostream &OS) const {
  StringRef Current = Code;

  while (!Current.empty()) {
    size_t Pos = Current.find_first_of("$\n\\");
    if (Pos == StringRef::npos) {
      OS << Current;
      Current = "";
      continue;
    }

    OS << Current.substr(0, Pos);
    Current = Current.substr(Pos);

    if (Current.startswith("\n")) {
      OS << "\n" << Indent;
      Current = Current.drop_front(1);
      continue;
    }

    if (Current.startswith("\\$") || Current.startswith("\\\\")) {
      OS << Current[1];
      Current = Current.drop_front(2);
      continue;
    }

    if (Current.startswith("\\")) {
      Current = Current.drop_front(1);
      continue;
    }

    if (Current.startswith("${")) {
      StringRef StartVar = Current;
      Current = Current.drop_front(2);
      StringRef Var;
      std::tie(Var, Current) = Current.split("}");

      // Warn if we split because no terminator was found.
      StringRef EndVar = StartVar.drop_front(2 /* ${ */ + Var.size());
      if (EndVar.empty()) {
        size_t LocOffset = StartVar.data() - Code.data();
        PrintWarning(
            Loc.size() > 0 && Loc[0].isValid()
                ? SMLoc::getFromPointer(Loc[0].getPointer() + LocOffset)
                : SMLoc(),
            "Unterminated expansion");
      }

      auto ValueI = Expansions.find(Var);
      if (ValueI == Expansions.end()) {
        size_t LocOffset = StartVar.data() - Code.data();
        PrintError(Loc.size() > 0 && Loc[0].isValid()
                       ? SMLoc::getFromPointer(Loc[0].getPointer() + LocOffset)
                       : SMLoc(),
                   "Attempting to expand an undeclared variable " + Var);
      }
      if (ShowExpansions)
        OS << "/*$" << Var << "{*/";
      OS << Expansions.lookup(Var);
      if (ShowExpansions)
        OS << "/*}*/";
      continue;
    }

    size_t LocOffset = Current.data() - Code.data();
    PrintWarning(Loc.size() > 0 && Loc[0].isValid()
                     ? SMLoc::getFromPointer(Loc[0].getPointer() + LocOffset)
                     : SMLoc(),
                 "Assuming missing escape character");
    OS << "$";
    Current = Current.drop_front(1);
  }
}
