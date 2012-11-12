//===- Error.cpp - tblgen error handling helper routines --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains error handling helper routines to pretty-print diagnostic
// messages from tblgen.
//
//===----------------------------------------------------------------------===//

#include "llvm/TableGen/Error.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

namespace llvm {

SourceMgr SrcMgr;

static void PrintMessage(ArrayRef<SMLoc> Loc, SourceMgr::DiagKind Kind,
                         const Twine &Msg) {
  SMLoc NullLoc;
  if (Loc.empty())
    Loc = NullLoc;
  SrcMgr.PrintMessage(Loc.front(), Kind, Msg);
  for (unsigned i = 1; i < Loc.size(); ++i)
    SrcMgr.PrintMessage(Loc[i], SourceMgr::DK_Note,
                        "instantiated from multiclass");
}

void PrintWarning(ArrayRef<SMLoc> WarningLoc, const Twine &Msg) {
  PrintMessage(WarningLoc, SourceMgr::DK_Warning, Msg);
}

void PrintWarning(const char *Loc, const Twine &Msg) {
  SrcMgr.PrintMessage(SMLoc::getFromPointer(Loc), SourceMgr::DK_Warning, Msg);
}

void PrintWarning(const Twine &Msg) {
  errs() << "warning:" << Msg << "\n";
}

void PrintError(ArrayRef<SMLoc> ErrorLoc, const Twine &Msg) {
  PrintMessage(ErrorLoc, SourceMgr::DK_Error, Msg);
}

void PrintError(const char *Loc, const Twine &Msg) {
  SrcMgr.PrintMessage(SMLoc::getFromPointer(Loc), SourceMgr::DK_Error, Msg);
}

void PrintError(const Twine &Msg) {
  errs() << "error:" << Msg << "\n";
}

void PrintFatalError(const std::string &Msg) {
  PrintError(Twine(Msg));
  std::exit(1);
}

void PrintFatalError(ArrayRef<SMLoc> ErrorLoc, const std::string &Msg) {
  PrintError(ErrorLoc, Msg);
  std::exit(1);
}

} // end namespace llvm
