//===- Signals.cpp - Generic Unix Signals Implementation -----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines some helpful functions for dealing with the possibility of
// Unix signals occuring while your program is running.
//
//===----------------------------------------------------------------------===//

#include "Unix.h"
#include "llvm/ADT/STLExtras.h"
#include <vector>
#include <algorithm>
#if HAVE_EXECINFO_H
# include <execinfo.h>         // For backtrace().
#endif
#if HAVE_SIGNAL_H
#include <signal.h>
#endif
#if HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif
#if HAVE_DLFCN_H && __GNUG__
#include <dlfcn.h>
#include <cxxabi.h> 
#endif
using namespace llvm;

namespace {

static bool StackTraceRequested = false; 

/// InterruptFunction - The function to call if ctrl-c is pressed.
static void (*InterruptFunction)() = 0;

static std::vector<sys::Path> *FilesToRemove = 0 ;
static std::vector<sys::Path> *DirectoriesToRemove = 0;

// IntSigs - Signals that may interrupt the program at any time.
static const int IntSigs[] = {
  SIGHUP, SIGINT, SIGQUIT, SIGPIPE, SIGTERM, SIGUSR1, SIGUSR2
};
static const int *const IntSigsEnd =
  IntSigs + sizeof(IntSigs) / sizeof(IntSigs[0]);

// KillSigs - Signals that are synchronous with the program that will cause it
// to die.
static const int KillSigs[] = {
  SIGILL, SIGTRAP, SIGABRT, SIGFPE, SIGBUS, SIGSEGV, SIGSYS, SIGXCPU, SIGXFSZ
#ifdef SIGEMT
  , SIGEMT
#endif
};
static const int *const KillSigsEnd =
  KillSigs + sizeof(KillSigs) / sizeof(KillSigs[0]);

#ifdef HAVE_BACKTRACE
static void* StackTrace[256];
#endif

// PrintStackTrace - In the case of a program crash or fault, print out a stack
// trace so that the user has an indication of why and where we died.
//
// On glibc systems we have the 'backtrace' function, which works nicely, but
// doesn't demangle symbols.  
static void PrintStackTrace() {
#ifdef HAVE_BACKTRACE
  // Use backtrace() to output a backtrace on Linux systems with glibc.
  int depth = backtrace(StackTrace,
                        static_cast<int>(array_lengthof(StackTrace)));
#if HAVE_DLFCN_H && __GNUG__
  int width = 0;
  for (int i = 0; i < depth; ++i) {
    Dl_info dlinfo;
    dladdr(StackTrace[i], &dlinfo);
    char* name = strrchr(dlinfo.dli_fname, '/');

    int nwidth;
    if (name == NULL) nwidth = strlen(dlinfo.dli_fname);
    else              nwidth = strlen(name) - 1;

    if (nwidth > width) width = nwidth;
  }

  for (int i = 0; i < depth; ++i) {
    Dl_info dlinfo;
    dladdr(StackTrace[i], &dlinfo);

    fprintf(stderr, "%-3d", i);

    char* name = strrchr(dlinfo.dli_fname, '/');
    if (name == NULL) fprintf(stderr, " %-*s", width, dlinfo.dli_fname);
    else              fprintf(stderr, " %-*s", width, name+1);

    fprintf(stderr, " %#0*lx",
            (int)(sizeof(void*) * 2) + 2, (unsigned long)StackTrace[i]);

    if (dlinfo.dli_sname != NULL) {
      int res;
      fputc(' ', stderr);
      char* d = abi::__cxa_demangle(dlinfo.dli_sname, NULL, NULL, &res);
      if (d == NULL) fputs(dlinfo.dli_sname, stderr);
      else           fputs(d, stderr);
      free(d);

      fprintf(stderr, " + %tu",(char*)StackTrace[i]-(char*)dlinfo.dli_saddr);
    }
    fputc('\n', stderr);
  }
#else
  backtrace_symbols_fd(StackTrace, depth, STDERR_FILENO);
#endif
#endif
}

// SignalHandler - The signal handler that runs...
static RETSIGTYPE SignalHandler(int Sig) {
  if (FilesToRemove != 0)
    while (!FilesToRemove->empty()) {
      FilesToRemove->back().eraseFromDisk(true);
      FilesToRemove->pop_back();
    }

  if (DirectoriesToRemove != 0)
    while (!DirectoriesToRemove->empty()) {
      DirectoriesToRemove->back().eraseFromDisk(true);
      DirectoriesToRemove->pop_back();
    }

  if (std::find(IntSigs, IntSigsEnd, Sig) != IntSigsEnd) {
    if (InterruptFunction) {
      void (*IF)() = InterruptFunction;
      InterruptFunction = 0;
      IF();        // run the interrupt function.
      return;
    } else {
      exit(1);   // If this is an interrupt signal, exit the program
    }
  }

  // Otherwise if it is a fault (like SEGV) output the stacktrace to
  // STDERR (if we can) and reissue the signal to die...
  if (StackTraceRequested)
    PrintStackTrace();
  signal(Sig, SIG_DFL);
}

// Just call signal
static void RegisterHandler(int Signal) { 
  signal(Signal, SignalHandler); 
}

}


void sys::SetInterruptFunction(void (*IF)()) {
  InterruptFunction = IF;
  RegisterHandler(SIGINT);
}

// RemoveFileOnSignal - The public API
bool sys::RemoveFileOnSignal(const sys::Path &Filename, std::string* ErrMsg) {
  if (FilesToRemove == 0)
    FilesToRemove = new std::vector<sys::Path>;

  FilesToRemove->push_back(Filename);

  std::for_each(IntSigs, IntSigsEnd, RegisterHandler);
  std::for_each(KillSigs, KillSigsEnd, RegisterHandler);
  return false;
}

// RemoveDirectoryOnSignal - The public API
bool sys::RemoveDirectoryOnSignal(const sys::Path& path, std::string* ErrMsg) {
  // Not a directory?
  struct stat buf;
  if (0 != stat(path.c_str(), &buf)) {
    MakeErrMsg(ErrMsg, path.toString() + ": can't get status of file");
    return true;
  }

  if (!S_ISDIR(buf.st_mode)) {
    if (ErrMsg)
      *ErrMsg = path.toString() + " is not a directory";
    return true;
  }

  if (DirectoriesToRemove == 0)
    DirectoriesToRemove = new std::vector<sys::Path>;

  DirectoriesToRemove->push_back(path);

  std::for_each(IntSigs, IntSigsEnd, RegisterHandler);
  std::for_each(KillSigs, KillSigsEnd, RegisterHandler);
  return false;
}

/// PrintStackTraceOnErrorSignal - When an error signal (such as SIBABRT or
/// SIGSEGV) is delivered to the process, print a stack trace and then exit.
void sys::PrintStackTraceOnErrorSignal() {
  StackTraceRequested = true;
  std::for_each(KillSigs, KillSigsEnd, RegisterHandler);
}
