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
// Unix signals occurring while your program is running.
//
//===----------------------------------------------------------------------===//

#include "Unix.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/UniqueLock.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <string>
#if HAVE_EXECINFO_H
# include <execinfo.h>         // For backtrace().
#endif
#if HAVE_SIGNAL_H
#include <signal.h>
#endif
#if HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif
#if HAVE_CXXABI_H
#include <cxxabi.h>
#endif
#if HAVE_DLFCN_H
#include <dlfcn.h>
#endif
#if HAVE_MACH_MACH_H
#include <mach/mach.h>
#endif
#if HAVE_LINK_H
#include <link.h>
#endif

using namespace llvm;

static RETSIGTYPE SignalHandler(int Sig);  // defined below.

static ManagedStatic<SmartMutex<true> > SignalsMutex;

/// InterruptFunction - The function to call if ctrl-c is pressed.
static void (*InterruptFunction)() = nullptr;

static ManagedStatic<std::vector<std::string>> FilesToRemove;

// IntSigs - Signals that represent requested termination. There's no bug
// or failure, or if there is, it's not our direct responsibility. For whatever
// reason, our continued execution is no longer desirable.
static const int IntSigs[] = {
  SIGHUP, SIGINT, SIGPIPE, SIGTERM, SIGUSR1, SIGUSR2
};

// KillSigs - Signals that represent that we have a bug, and our prompt
// termination has been ordered.
static const int KillSigs[] = {
  SIGILL, SIGTRAP, SIGABRT, SIGFPE, SIGBUS, SIGSEGV, SIGQUIT
#ifdef SIGSYS
  , SIGSYS
#endif
#ifdef SIGXCPU
  , SIGXCPU
#endif
#ifdef SIGXFSZ
  , SIGXFSZ
#endif
#ifdef SIGEMT
  , SIGEMT
#endif
};

static unsigned NumRegisteredSignals = 0;
static struct {
  struct sigaction SA;
  int SigNo;
} RegisteredSignalInfo[array_lengthof(IntSigs) + array_lengthof(KillSigs)];


static void RegisterHandler(int Signal) {
  assert(NumRegisteredSignals < array_lengthof(RegisteredSignalInfo) &&
         "Out of space for signal handlers!");

  struct sigaction NewHandler;

  NewHandler.sa_handler = SignalHandler;
  NewHandler.sa_flags = SA_NODEFER|SA_RESETHAND;
  sigemptyset(&NewHandler.sa_mask);

  // Install the new handler, save the old one in RegisteredSignalInfo.
  sigaction(Signal, &NewHandler,
            &RegisteredSignalInfo[NumRegisteredSignals].SA);
  RegisteredSignalInfo[NumRegisteredSignals].SigNo = Signal;
  ++NumRegisteredSignals;
}

static void RegisterHandlers() {
  // We need to dereference the signals mutex during handler registration so
  // that we force its construction. This is to prevent the first use being
  // during handling an actual signal because you can't safely call new in a
  // signal handler.
  *SignalsMutex;

  // If the handlers are already registered, we're done.
  if (NumRegisteredSignals != 0) return;

  for (auto S : IntSigs) RegisterHandler(S);
  for (auto S : KillSigs) RegisterHandler(S);
}

static void UnregisterHandlers() {
  // Restore all of the signal handlers to how they were before we showed up.
  for (unsigned i = 0, e = NumRegisteredSignals; i != e; ++i)
    sigaction(RegisteredSignalInfo[i].SigNo,
              &RegisteredSignalInfo[i].SA, nullptr);
  NumRegisteredSignals = 0;
}


/// RemoveFilesToRemove - Process the FilesToRemove list. This function
/// should be called with the SignalsMutex lock held.
/// NB: This must be an async signal safe function. It cannot allocate or free
/// memory, even in debug builds.
static void RemoveFilesToRemove() {
  // Avoid constructing ManagedStatic in the signal handler.
  // If FilesToRemove is not constructed, there are no files to remove.
  if (!FilesToRemove.isConstructed())
    return;

  // We avoid iterators in case of debug iterators that allocate or release
  // memory.
  std::vector<std::string>& FilesToRemoveRef = *FilesToRemove;
  for (unsigned i = 0, e = FilesToRemoveRef.size(); i != e; ++i) {
    const char *path = FilesToRemoveRef[i].c_str();

    // Get the status so we can determine if it's a file or directory. If we
    // can't stat the file, ignore it.
    struct stat buf;
    if (stat(path, &buf) != 0)
      continue;

    // If this is not a regular file, ignore it. We want to prevent removal of
    // special files like /dev/null, even if the compiler is being run with the
    // super-user permissions.
    if (!S_ISREG(buf.st_mode))
      continue;

    // Otherwise, remove the file. We ignore any errors here as there is nothing
    // else we can do.
    unlink(path);
  }
}

// SignalHandler - The signal handler that runs.
static RETSIGTYPE SignalHandler(int Sig) {
  // Restore the signal behavior to default, so that the program actually
  // crashes when we return and the signal reissues.  This also ensures that if
  // we crash in our signal handler that the program will terminate immediately
  // instead of recursing in the signal handler.
  UnregisterHandlers();

  // Unmask all potentially blocked kill signals.
  sigset_t SigMask;
  sigfillset(&SigMask);
  sigprocmask(SIG_UNBLOCK, &SigMask, nullptr);

  {
    unique_lock<SmartMutex<true>> Guard(*SignalsMutex);
    RemoveFilesToRemove();

    if (std::find(std::begin(IntSigs), std::end(IntSigs), Sig)
        != std::end(IntSigs)) {
      if (InterruptFunction) {
        void (*IF)() = InterruptFunction;
        Guard.unlock();
        InterruptFunction = nullptr;
        IF();        // run the interrupt function.
        return;
      }

      Guard.unlock();
      raise(Sig);   // Execute the default handler.
      return;
   }
  }

  // Otherwise if it is a fault (like SEGV) run any handler.
  llvm::sys::RunSignalHandlers();

#ifdef __s390__
  // On S/390, certain signals are delivered with PSW Address pointing to
  // *after* the faulting instruction.  Simply returning from the signal
  // handler would continue execution after that point, instead of
  // re-raising the signal.  Raise the signal manually in those cases.
  if (Sig == SIGILL || Sig == SIGFPE || Sig == SIGTRAP)
    raise(Sig);
#endif
}

void llvm::sys::RunInterruptHandlers() {
  sys::SmartScopedLock<true> Guard(*SignalsMutex);
  RemoveFilesToRemove();
}

void llvm::sys::SetInterruptFunction(void (*IF)()) {
  {
    sys::SmartScopedLock<true> Guard(*SignalsMutex);
    InterruptFunction = IF;
  }
  RegisterHandlers();
}

// RemoveFileOnSignal - The public API
bool llvm::sys::RemoveFileOnSignal(StringRef Filename,
                                   std::string* ErrMsg) {
  {
    sys::SmartScopedLock<true> Guard(*SignalsMutex);
    FilesToRemove->push_back(Filename);
  }

  RegisterHandlers();
  return false;
}

// DontRemoveFileOnSignal - The public API
void llvm::sys::DontRemoveFileOnSignal(StringRef Filename) {
  sys::SmartScopedLock<true> Guard(*SignalsMutex);
  std::vector<std::string>::reverse_iterator RI =
    std::find(FilesToRemove->rbegin(), FilesToRemove->rend(), Filename);
  std::vector<std::string>::iterator I = FilesToRemove->end();
  if (RI != FilesToRemove->rend())
    I = FilesToRemove->erase(RI.base()-1);
}

/// AddSignalHandler - Add a function to be called when a signal is delivered
/// to the process.  The handler can have a cookie passed to it to identify
/// what instance of the handler it is.
void llvm::sys::AddSignalHandler(void (*FnPtr)(void *), void *Cookie) {
  CallBacksToRun->push_back(std::make_pair(FnPtr, Cookie));
  RegisterHandlers();
}

#if defined(HAVE_BACKTRACE) && defined(ENABLE_BACKTRACES) && HAVE_LINK_H &&    \
    (defined(__linux__) || defined(__FreeBSD__) ||                             \
     defined(__FreeBSD_kernel__) || defined(__NetBSD__))
struct DlIteratePhdrData {
  void **StackTrace;
  int depth;
  bool first;
  const char **modules;
  intptr_t *offsets;
  const char *main_exec_name;
};

static int dl_iterate_phdr_cb(dl_phdr_info *info, size_t size, void *arg) {
  DlIteratePhdrData *data = (DlIteratePhdrData*)arg;
  const char *name = data->first ? data->main_exec_name : info->dlpi_name;
  data->first = false;
  for (int i = 0; i < info->dlpi_phnum; i++) {
    const auto *phdr = &info->dlpi_phdr[i];
    if (phdr->p_type != PT_LOAD)
      continue;
    intptr_t beg = info->dlpi_addr + phdr->p_vaddr;
    intptr_t end = beg + phdr->p_memsz;
    for (int j = 0; j < data->depth; j++) {
      if (data->modules[j])
        continue;
      intptr_t addr = (intptr_t)data->StackTrace[j];
      if (beg <= addr && addr < end) {
        data->modules[j] = name;
        data->offsets[j] = addr - info->dlpi_addr;
      }
    }
  }
  return 0;
}

/// If this is an ELF platform, we can find all loaded modules and their virtual
/// addresses with dl_iterate_phdr.
static bool findModulesAndOffsets(void **StackTrace, int Depth,
                                  const char **Modules, intptr_t *Offsets,
                                  const char *MainExecutableName,
                                  StringSaver &StrPool) {
  DlIteratePhdrData data = {StackTrace, Depth,   true,
                            Modules,    Offsets, MainExecutableName};
  dl_iterate_phdr(dl_iterate_phdr_cb, &data);
  return true;
}
#else
/// This platform does not have dl_iterate_phdr, so we do not yet know how to
/// find all loaded DSOs.
static bool findModulesAndOffsets(void **StackTrace, int Depth,
                                  const char **Modules, intptr_t *Offsets,
                                  const char *MainExecutableName,
                                  StringSaver &StrPool) {
  return false;
}
#endif // defined(HAVE_BACKTRACE) && defined(ENABLE_BACKTRACES) && ...

// PrintStackTrace - In the case of a program crash or fault, print out a stack
// trace so that the user has an indication of why and where we died.
//
// On glibc systems we have the 'backtrace' function, which works nicely, but
// doesn't demangle symbols.
void llvm::sys::PrintStackTrace(raw_ostream &OS) {
#if defined(HAVE_BACKTRACE) && defined(ENABLE_BACKTRACES)
  static void* StackTrace[256];
  // Use backtrace() to output a backtrace on Linux systems with glibc.
  int depth = backtrace(StackTrace,
                        static_cast<int>(array_lengthof(StackTrace)));
  if (printSymbolizedStackTrace(StackTrace, depth, OS))
    return;
#if HAVE_DLFCN_H && __GNUG__
  int width = 0;
  for (int i = 0; i < depth; ++i) {
    Dl_info dlinfo;
    dladdr(StackTrace[i], &dlinfo);
    const char* name = strrchr(dlinfo.dli_fname, '/');

    int nwidth;
    if (!name) nwidth = strlen(dlinfo.dli_fname);
    else       nwidth = strlen(name) - 1;

    if (nwidth > width) width = nwidth;
  }

  for (int i = 0; i < depth; ++i) {
    Dl_info dlinfo;
    dladdr(StackTrace[i], &dlinfo);

    OS << format("%-2d", i);

    const char* name = strrchr(dlinfo.dli_fname, '/');
    if (!name) OS << format(" %-*s", width, dlinfo.dli_fname);
    else       OS << format(" %-*s", width, name+1);

    OS << format(" %#0*lx", (int)(sizeof(void*) * 2) + 2,
                 (unsigned long)StackTrace[i]);

    if (dlinfo.dli_sname != nullptr) {
      OS << ' ';
#  if HAVE_CXXABI_H
      int res;
      char* d = abi::__cxa_demangle(dlinfo.dli_sname, nullptr, nullptr, &res);
#  else
      char* d = NULL;
#  endif
      if (!d) OS << dlinfo.dli_sname;
      else    OS << d;
      free(d);

      // FIXME: When we move to C++11, use %t length modifier. It's not in
      // C++03 and causes gcc to issue warnings. Losing the upper 32 bits of
      // the stack offset for a stack dump isn't likely to cause any problems.
      OS << format(" + %u",(unsigned)((char*)StackTrace[i]-
                                      (char*)dlinfo.dli_saddr));
    }
    OS << '\n';
  }
#else
  backtrace_symbols_fd(StackTrace, depth, STDERR_FILENO);
#endif
#endif
}

static void PrintStackTraceSignalHandler(void *) {
  PrintStackTrace(llvm::errs());
}

void llvm::sys::DisableSystemDialogsOnCrash() {}

/// PrintStackTraceOnErrorSignal - When an error signal (such as SIGABRT or
/// SIGSEGV) is delivered to the process, print a stack trace and then exit.
void llvm::sys::PrintStackTraceOnErrorSignal(bool DisableCrashReporting) {
  AddSignalHandler(PrintStackTraceSignalHandler, nullptr);

#if defined(__APPLE__) && defined(ENABLE_CRASH_OVERRIDES)
  // Environment variable to disable any kind of crash dialog.
  if (DisableCrashReporting || getenv("LLVM_DISABLE_CRASH_REPORT")) {
    mach_port_t self = mach_task_self();

    exception_mask_t mask = EXC_MASK_CRASH;

    kern_return_t ret = task_set_exception_ports(self,
                             mask,
                             MACH_PORT_NULL,
                             EXCEPTION_STATE_IDENTITY | MACH_EXCEPTION_CODES,
                             THREAD_STATE_NONE);
    (void)ret;
  }
#endif
}


/***/

// On Darwin, raise sends a signal to the main thread instead of the current
// thread. This has the unfortunate effect that assert() and abort() will end up
// bypassing our crash recovery attempts. We work around this for anything in
// the same linkage unit by just defining our own versions of the assert handler
// and abort.

#if defined(__APPLE__) && defined(ENABLE_CRASH_OVERRIDES)

#include <signal.h>
#include <pthread.h>

int raise(int sig) {
  return pthread_kill(pthread_self(), sig);
}

void __assert_rtn(const char *func,
                  const char *file,
                  int line,
                  const char *expr) {
  if (func)
    fprintf(stderr, "Assertion failed: (%s), function %s, file %s, line %d.\n",
            expr, func, file, line);
  else
    fprintf(stderr, "Assertion failed: (%s), file %s, line %d.\n",
            expr, file, line);
  abort();
}

void abort() {
  raise(SIGABRT);
  usleep(1000);
  __builtin_trap();
}

#endif
