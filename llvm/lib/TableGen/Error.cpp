//===- Error.cpp - tblgen error handling helper routines --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains error handling helper routines to pretty-print diagnostic
// messages from tblgen.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/WithColor.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <cstdlib>

namespace llvm {

SourceMgr SrcMgr;
unsigned ErrorsPrinted = 0;

static void PrintMessage(ArrayRef<SMLoc> Locs, SourceMgr::DiagKind Kind,
                         const Twine &Msg) {
  // Count the total number of errors printed.
  // This is used to exit with an error code if there were any errors.
  if (Kind == SourceMgr::DK_Error)
    ++ErrorsPrinted;

  SMLoc NullLoc;
  if (Locs.empty())
    Locs = NullLoc;
  SrcMgr.PrintMessage(Locs.consume_front(), Kind, Msg);
  for (SMLoc Loc : Locs)
    SrcMgr.PrintMessage(Loc, SourceMgr::DK_Note,
                        "instantiated from multiclass");
}

// Run file cleanup handlers and then exit fatally (with non-zero exit code).
[[noreturn]] inline static void fatal_exit() {
  // The following call runs the file cleanup handlers.
  sys::RunInterruptHandlers();
  std::exit(1);
}

// Functions to print notes.

void PrintNote(const Twine &Msg) {
  WithColor::note() << Msg << "\n";
}

void PrintNote(function_ref<void(raw_ostream &OS)> PrintMsg) {
  PrintMsg(WithColor::note());
}

void PrintNote(ArrayRef<SMLoc> NoteLoc, const Twine &Msg) {
  PrintMessage(NoteLoc, SourceMgr::DK_Note, Msg);
}

// Functions to print fatal notes.

void PrintFatalNote(const Twine &Msg) {
  PrintNote(Msg);
  fatal_exit();
}

void PrintFatalNote(ArrayRef<SMLoc> NoteLoc, const Twine &Msg) {
  PrintNote(NoteLoc, Msg);
  fatal_exit();
}

// This method takes a Record and uses the source location
// stored in it.
void PrintFatalNote(const Record *Rec, const Twine &Msg) {
  PrintNote(Rec->getLoc(), Msg);
  fatal_exit();
}

// This method takes a RecordVal and uses the source location
// stored in it.
void PrintFatalNote(const RecordVal *RecVal, const Twine &Msg) {
  PrintNote(RecVal->getLoc(), Msg);
  fatal_exit();
}

// Functions to print warnings.

void PrintWarning(const Twine &Msg) { WithColor::warning() << Msg << "\n"; }

void PrintWarning(ArrayRef<SMLoc> WarningLoc, const Twine &Msg) {
  PrintMessage(WarningLoc, SourceMgr::DK_Warning, Msg);
}

void PrintWarning(const char *Loc, const Twine &Msg) {
  SrcMgr.PrintMessage(SMLoc::getFromPointer(Loc), SourceMgr::DK_Warning, Msg);
}

// Functions to print errors.

void PrintError(const Twine &Msg) { WithColor::error() << Msg << "\n"; }

void PrintError(function_ref<void(raw_ostream &OS)> PrintMsg) {
  PrintMsg(WithColor::error());
}

void PrintError(ArrayRef<SMLoc> ErrorLoc, const Twine &Msg) {
  PrintMessage(ErrorLoc, SourceMgr::DK_Error, Msg);
}

void PrintError(const char *Loc, const Twine &Msg) {
  SrcMgr.PrintMessage(SMLoc::getFromPointer(Loc), SourceMgr::DK_Error, Msg);
}

// This method takes a Record and uses the source location
// stored in it.
void PrintError(const Record *Rec, const Twine &Msg) {
  PrintMessage(Rec->getLoc(), SourceMgr::DK_Error, Msg);
}

// This method takes a RecordVal and uses the source location
// stored in it.
void PrintError(const RecordVal *RecVal, const Twine &Msg) {
  PrintMessage(RecVal->getLoc(), SourceMgr::DK_Error, Msg);
}

// Functions to print fatal errors.

void PrintFatalError(const Twine &Msg) {
  PrintError(Msg);
  fatal_exit();
}

void PrintFatalError(function_ref<void(raw_ostream &OS)> PrintMsg) {
  PrintError(PrintMsg);
  fatal_exit();
}

void PrintFatalError(ArrayRef<SMLoc> ErrorLoc, const Twine &Msg) {
  PrintError(ErrorLoc, Msg);
  fatal_exit();
}

// This method takes a Record and uses the source location
// stored in it.
void PrintFatalError(const Record *Rec, const Twine &Msg) {
  PrintError(Rec->getLoc(), Msg);
  fatal_exit();
}

// This method takes a RecordVal and uses the source location
// stored in it.
void PrintFatalError(const RecordVal *RecVal, const Twine &Msg) {
  PrintError(RecVal->getLoc(), Msg);
  fatal_exit();
}

// Check an assertion: Obtain the condition value and be sure it is true.
// If not, print a nonfatal error along with the message.
bool CheckAssert(SMLoc Loc, const Init *Condition, const Init *Message) {
  auto *CondValue = dyn_cast_or_null<IntInit>(Condition->convertInitializerTo(
      IntRecTy::get(Condition->getRecordKeeper())));
  if (!CondValue) {
    PrintError(Loc, "assert condition must of type bit, bits, or int.");
    return true;
  }
  if (!CondValue->getValue()) {
    auto *MessageInit = dyn_cast<StringInit>(Message);
    StringRef AssertMsg = MessageInit ? MessageInit->getValue()
                                      : "(assert message is not a string)";
    PrintError(Loc, "assertion failed: " + AssertMsg);
    return true;
  }
  return false;
}

// Dump a message to stderr.
void dumpMessage(SMLoc Loc, const Init *Message) {
  if (auto *MessageInit = dyn_cast<StringInit>(Message))
    PrintNote(Loc, MessageInit->getValue());
  else
    PrintError(Loc, "dump value is not of type string");
}

} // end namespace llvm
