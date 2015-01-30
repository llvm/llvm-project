//===- PrettyStackTrace.cpp - Pretty Crash Handling -----------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines some helpful functions for dealing with the possibility of
// Unix signals occurring while your program is running.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/PrettyStackTrace.h"
#include "llvm-c/Core.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Config/config.h"     // Get autoconf configuration settings
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/ThreadLocal.h"
#include "llvm/Support/Watchdog.h"
#include "llvm/Support/raw_ostream.h"

#ifdef HAVE_CRASHREPORTERCLIENT_H
#include <CrashReporterClient.h>
#endif

using namespace llvm;

static ManagedStatic<sys::ThreadLocal<const PrettyStackTraceEntry> > PrettyStackTraceHead;

static unsigned PrintStack(const PrettyStackTraceEntry *Entry, raw_ostream &OS){
  unsigned NextID = 0;
  if (Entry->getNextEntry())
    NextID = PrintStack(Entry->getNextEntry(), OS);
  OS << NextID << ".\t";
  {
    sys::Watchdog W(5);
    Entry->print(OS);
  }
  
  return NextID+1;
}

/// PrintCurStackTrace - Print the current stack trace to the specified stream.
static void PrintCurStackTrace(raw_ostream &OS) {
  // Don't print an empty trace.
  if (!PrettyStackTraceHead->get()) return;
  
  // If there are pretty stack frames registered, walk and emit them.
  OS << "Stack dump:\n";
  
  PrintStack(PrettyStackTraceHead->get(), OS);
  OS.flush();
}

// Integrate with crash reporter libraries.
#if defined (__APPLE__) && HAVE_CRASHREPORTERCLIENT_H
//  If any clients of llvm try to link to libCrashReporterClient.a themselves,
//  only one crash info struct will be used.
extern "C" {
CRASH_REPORTER_CLIENT_HIDDEN 
struct crashreporter_annotations_t gCRAnnotations 
        __attribute__((section("__DATA," CRASHREPORTER_ANNOTATIONS_SECTION))) 
        = { CRASHREPORTER_ANNOTATIONS_VERSION, 0, 0, 0, 0, 0, 0 };
}
#elif defined (__APPLE__) && HAVE_CRASHREPORTER_INFO
static const char *__crashreporter_info__ = 0;
asm(".desc ___crashreporter_info__, 0x10");
#endif


/// CrashHandler - This callback is run if a fatal signal is delivered to the
/// process, it prints the pretty stack trace.
static void CrashHandler(void *) {
#ifndef __APPLE__
  // On non-apple systems, just emit the crash stack trace to stderr.
  PrintCurStackTrace(errs());
#else
  // Otherwise, emit to a smallvector of chars, send *that* to stderr, but also
  // put it into __crashreporter_info__.
  SmallString<2048> TmpStr;
  {
    raw_svector_ostream Stream(TmpStr);
    PrintCurStackTrace(Stream);
  }
  
  if (!TmpStr.empty()) {
#ifdef HAVE_CRASHREPORTERCLIENT_H
    // Cast to void to avoid warning.
    (void)CRSetCrashLogMessage(std::string(TmpStr.str()).c_str());
#elif HAVE_CRASHREPORTER_INFO 
    __crashreporter_info__ = strdup(std::string(TmpStr.str()).c_str());
#endif
    errs() << TmpStr.str();
  }
  
#endif
}

PrettyStackTraceEntry::PrettyStackTraceEntry() {
  // Link ourselves.
  NextEntry = PrettyStackTraceHead->get();
  PrettyStackTraceHead->set(this);
}

PrettyStackTraceEntry::~PrettyStackTraceEntry() {
  // Do nothing if PrettyStackTraceHead is uninitialized. This can only happen
  // if a shutdown occurred after we created the PrettyStackTraceEntry. That
  // does occur in the following idiom:
  //
  // PrettyStackTraceProgram X(...);
  // llvm_shutdown_obj Y;
  //
  // Without this check, we may end up removing ourselves from the stack trace
  // after PrettyStackTraceHead has already been destroyed.
  if (!PrettyStackTraceHead.isConstructed())
    return;
  
  assert(PrettyStackTraceHead->get() == this &&
         "Pretty stack trace entry destruction is out of order");
  PrettyStackTraceHead->set(getNextEntry());
}

void PrettyStackTraceString::print(raw_ostream &OS) const {
  OS << Str << "\n";
}

void PrettyStackTraceProgram::print(raw_ostream &OS) const {
  OS << "Program arguments: ";
  // Print the argument list.
  for (unsigned i = 0, e = ArgC; i != e; ++i)
    OS << ArgV[i] << ' ';
  OS << '\n';
}

static bool RegisterCrashPrinter() {
  sys::AddSignalHandler(CrashHandler, nullptr);
  return false;
}

void llvm::EnablePrettyStackTrace() {
  // The first time this is called, we register the crash printer.
  static bool HandlerRegistered = RegisterCrashPrinter();
  (void)HandlerRegistered;
}

void LLVMEnablePrettyStackTrace() {
  EnablePrettyStackTrace();
}
