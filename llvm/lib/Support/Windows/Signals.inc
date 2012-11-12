//===- Win32/Signals.cpp - Win32 Signals Implementation ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides the Win32 specific implementation of the Signals class.
//
//===----------------------------------------------------------------------===//

#include "Windows.h"
#include <stdio.h>
#include <vector>
#include <algorithm>

#ifdef __MINGW32__
 #include <imagehlp.h>
#else
 #include <dbghelp.h>
#endif
#include <psapi.h>

#ifdef _MSC_VER
 #pragma comment(lib, "psapi.lib")
 #pragma comment(lib, "dbghelp.lib")
#elif __MINGW32__
 #if ((HAVE_LIBIMAGEHLP != 1) || (HAVE_LIBPSAPI != 1))
  #error "libimagehlp.a & libpsapi.a should be present"
 #endif
 // The version of g++ that comes with MinGW does *not* properly understand
 // the ll format specifier for printf. However, MinGW passes the format
 // specifiers on to the MSVCRT entirely, and the CRT understands the ll
 // specifier. So these warnings are spurious in this case. Since we compile
 // with -Wall, this will generate these warnings which should be ignored. So
 // we will turn off the warnings for this just file. However, MinGW also does
 // not support push and pop for diagnostics, so we have to manually turn it
 // back on at the end of the file.
 #pragma GCC diagnostic ignored "-Wformat"
 #pragma GCC diagnostic ignored "-Wformat-extra-args"

 #if !defined(__MINGW64_VERSION_MAJOR)
 // MinGW.org does not have updated support for the 64-bit versions of the
 // DebugHlp APIs. So we will have to load them manually. The structures and
 // method signatures were pulled from DbgHelp.h in the Windows Platform SDK,
 // and adjusted for brevity.
 typedef struct _IMAGEHLP_LINE64 {
   DWORD    SizeOfStruct;
   PVOID    Key;
   DWORD    LineNumber;
   PCHAR    FileName;
   DWORD64  Address;
 } IMAGEHLP_LINE64, *PIMAGEHLP_LINE64;

 typedef struct _IMAGEHLP_SYMBOL64 {
   DWORD   SizeOfStruct;
   DWORD64 Address;
   DWORD   Size;
   DWORD   Flags;
   DWORD   MaxNameLength;
   CHAR    Name[1];
 } IMAGEHLP_SYMBOL64, *PIMAGEHLP_SYMBOL64;

 typedef struct _tagADDRESS64 {
   DWORD64       Offset;
   WORD          Segment;
   ADDRESS_MODE  Mode;
 } ADDRESS64, *LPADDRESS64;

 typedef struct _KDHELP64 {
   DWORD64   Thread;
   DWORD   ThCallbackStack;
   DWORD   ThCallbackBStore;
   DWORD   NextCallback;
   DWORD   FramePointer;
   DWORD64   KiCallUserMode;
   DWORD64   KeUserCallbackDispatcher;
   DWORD64   SystemRangeStart;
   DWORD64   KiUserExceptionDispatcher;
   DWORD64   StackBase;
   DWORD64   StackLimit;
   DWORD64   Reserved[5];
 } KDHELP64, *PKDHELP64;

 typedef struct _tagSTACKFRAME64 {
   ADDRESS64   AddrPC;
   ADDRESS64   AddrReturn;
   ADDRESS64   AddrFrame;
   ADDRESS64   AddrStack;
   ADDRESS64   AddrBStore;
   PVOID       FuncTableEntry;
   DWORD64     Params[4];
   BOOL        Far;
   BOOL        Virtual;
   DWORD64     Reserved[3];
   KDHELP64    KdHelp;
 } STACKFRAME64, *LPSTACKFRAME64;

typedef BOOL (__stdcall *PREAD_PROCESS_MEMORY_ROUTINE64)(HANDLE hProcess,
                      DWORD64 qwBaseAddress, PVOID lpBuffer, DWORD nSize,
                      LPDWORD lpNumberOfBytesRead);

typedef PVOID (__stdcall *PFUNCTION_TABLE_ACCESS_ROUTINE64)( HANDLE ahProcess,
                      DWORD64 AddrBase);

typedef DWORD64 (__stdcall *PGET_MODULE_BASE_ROUTINE64)(HANDLE hProcess,
                      DWORD64 Address);

typedef DWORD64 (__stdcall *PTRANSLATE_ADDRESS_ROUTINE64)(HANDLE hProcess,
                      HANDLE hThread, LPADDRESS64 lpaddr);

typedef BOOL (WINAPI *fpStackWalk64)(DWORD, HANDLE, HANDLE, LPSTACKFRAME64,
                      PVOID, PREAD_PROCESS_MEMORY_ROUTINE64,
                      PFUNCTION_TABLE_ACCESS_ROUTINE64,
                      PGET_MODULE_BASE_ROUTINE64,
                      PTRANSLATE_ADDRESS_ROUTINE64);
static fpStackWalk64 StackWalk64;

typedef DWORD64 (WINAPI *fpSymGetModuleBase64)(HANDLE, DWORD64);
static fpSymGetModuleBase64 SymGetModuleBase64;

typedef BOOL (WINAPI *fpSymGetSymFromAddr64)(HANDLE, DWORD64,
                      PDWORD64, PIMAGEHLP_SYMBOL64);
static fpSymGetSymFromAddr64 SymGetSymFromAddr64;

typedef BOOL (WINAPI *fpSymGetLineFromAddr64)(HANDLE, DWORD64,
                      PDWORD, PIMAGEHLP_LINE64);
static fpSymGetLineFromAddr64 SymGetLineFromAddr64;

typedef PVOID (WINAPI *fpSymFunctionTableAccess64)(HANDLE, DWORD64);
static fpSymFunctionTableAccess64 SymFunctionTableAccess64;

static bool load64BitDebugHelp(void) {
  HMODULE hLib = ::LoadLibrary("Dbghelp.dll");
  if (hLib) {
    StackWalk64 = (fpStackWalk64)
                      ::GetProcAddress(hLib, "StackWalk64");
    SymGetModuleBase64 = (fpSymGetModuleBase64)
                      ::GetProcAddress(hLib, "SymGetModuleBase64");
    SymGetSymFromAddr64 = (fpSymGetSymFromAddr64)
                      ::GetProcAddress(hLib, "SymGetSymFromAddr64");
    SymGetLineFromAddr64 = (fpSymGetLineFromAddr64)
                      ::GetProcAddress(hLib, "SymGetLineFromAddr64");
    SymFunctionTableAccess64 = (fpSymFunctionTableAccess64)
                     ::GetProcAddress(hLib, "SymFunctionTableAccess64");
  }
  return StackWalk64 != NULL;
}
 #endif // !defined(__MINGW64_VERSION_MAJOR)
#endif // __MINGW32__

// Forward declare.
static LONG WINAPI LLVMUnhandledExceptionFilter(LPEXCEPTION_POINTERS ep);
static BOOL WINAPI LLVMConsoleCtrlHandler(DWORD dwCtrlType);

// InterruptFunction - The function to call if ctrl-c is pressed.
static void (*InterruptFunction)() = 0;

static std::vector<llvm::sys::Path> *FilesToRemove = NULL;
static std::vector<std::pair<void(*)(void*), void*> > *CallBacksToRun = 0;
static bool RegisteredUnhandledExceptionFilter = false;
static bool CleanupExecuted = false;
static bool ExitOnUnhandledExceptions = false;
static PTOP_LEVEL_EXCEPTION_FILTER OldFilter = NULL;

// Windows creates a new thread to execute the console handler when an event
// (such as CTRL/C) occurs.  This causes concurrency issues with the above
// globals which this critical section addresses.
static CRITICAL_SECTION CriticalSection;

namespace llvm {

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only Win32 specific code
//===          and must not be UNIX code
//===----------------------------------------------------------------------===//

#ifdef _MSC_VER
/// CRTReportHook - Function called on a CRT debugging event.
static int CRTReportHook(int ReportType, char *Message, int *Return) {
  // Don't cause a DebugBreak() on return.
  if (Return)
    *Return = 0;

  switch (ReportType) {
  default:
  case _CRT_ASSERT:
    fprintf(stderr, "CRT assert: %s\n", Message);
    // FIXME: Is there a way to just crash? Perhaps throw to the unhandled
    // exception code? Perhaps SetErrorMode() handles this.
    _exit(3);
    break;
  case _CRT_ERROR:
    fprintf(stderr, "CRT error: %s\n", Message);
    // FIXME: Is there a way to just crash? Perhaps throw to the unhandled
    // exception code? Perhaps SetErrorMode() handles this.
    _exit(3);
    break;
  case _CRT_WARN:
    fprintf(stderr, "CRT warn: %s\n", Message);
    break;
  }

  // Don't call _CrtDbgReport.
  return TRUE;
}
#endif

static void RegisterHandler() {
#if __MINGW32__ && !defined(__MINGW64_VERSION_MAJOR)
  // On MinGW.org, we need to load up the symbols explicitly, because the
  // Win32 framework they include does not have support for the 64-bit
  // versions of the APIs we need.  If we cannot load up the APIs (which
  // would be unexpected as they should exist on every version of Windows
  // we support), we will bail out since there would be nothing to report.
  if (!load64BitDebugHelp()) {
    assert(false && "These APIs should always be available");
    return;
  }
#endif

  if (RegisteredUnhandledExceptionFilter) {
    EnterCriticalSection(&CriticalSection);
    return;
  }

  // Now's the time to create the critical section.  This is the first time
  // through here, and there's only one thread.
  InitializeCriticalSection(&CriticalSection);

  // Enter it immediately.  Now if someone hits CTRL/C, the console handler
  // can't proceed until the globals are updated.
  EnterCriticalSection(&CriticalSection);

  RegisteredUnhandledExceptionFilter = true;
  OldFilter = SetUnhandledExceptionFilter(LLVMUnhandledExceptionFilter);
  SetConsoleCtrlHandler(LLVMConsoleCtrlHandler, TRUE);

  // Environment variable to disable any kind of crash dialog.
  if (getenv("LLVM_DISABLE_CRASH_REPORT")) {
#ifdef _MSC_VER
    _CrtSetReportHook(CRTReportHook);
#endif
    SetErrorMode(SEM_FAILCRITICALERRORS |
                 SEM_NOGPFAULTERRORBOX |
                 SEM_NOOPENFILEERRORBOX);
    ExitOnUnhandledExceptions = true;
  }

  // IMPORTANT NOTE: Caller must call LeaveCriticalSection(&CriticalSection) or
  // else multi-threading problems will ensue.
}

// RemoveFileOnSignal - The public API
bool sys::RemoveFileOnSignal(const sys::Path &Filename, std::string* ErrMsg) {
  RegisterHandler();

  if (CleanupExecuted) {
    if (ErrMsg)
      *ErrMsg = "Process terminating -- cannot register for removal";
    return true;
  }

  if (FilesToRemove == NULL)
    FilesToRemove = new std::vector<sys::Path>;

  FilesToRemove->push_back(Filename);

  LeaveCriticalSection(&CriticalSection);
  return false;
}

// DontRemoveFileOnSignal - The public API
void sys::DontRemoveFileOnSignal(const sys::Path &Filename) {
  if (FilesToRemove == NULL)
    return;

  RegisterHandler();

  FilesToRemove->push_back(Filename);
  std::vector<sys::Path>::reverse_iterator I =
  std::find(FilesToRemove->rbegin(), FilesToRemove->rend(), Filename);
  if (I != FilesToRemove->rend())
    FilesToRemove->erase(I.base()-1);

  LeaveCriticalSection(&CriticalSection);
}

/// PrintStackTraceOnErrorSignal - When an error signal (such as SIBABRT or
/// SIGSEGV) is delivered to the process, print a stack trace and then exit.
void sys::PrintStackTraceOnErrorSignal() {
  RegisterHandler();
  LeaveCriticalSection(&CriticalSection);
}


void sys::SetInterruptFunction(void (*IF)()) {
  RegisterHandler();
  InterruptFunction = IF;
  LeaveCriticalSection(&CriticalSection);
}


/// AddSignalHandler - Add a function to be called when a signal is delivered
/// to the process.  The handler can have a cookie passed to it to identify
/// what instance of the handler it is.
void sys::AddSignalHandler(void (*FnPtr)(void *), void *Cookie) {
  if (CallBacksToRun == 0)
    CallBacksToRun = new std::vector<std::pair<void(*)(void*), void*> >();
  CallBacksToRun->push_back(std::make_pair(FnPtr, Cookie));
  RegisterHandler();
  LeaveCriticalSection(&CriticalSection);
}
}

static void Cleanup() {
  EnterCriticalSection(&CriticalSection);

  // Prevent other thread from registering new files and directories for
  // removal, should we be executing because of the console handler callback.
  CleanupExecuted = true;

  // FIXME: open files cannot be deleted.

  if (FilesToRemove != NULL)
    while (!FilesToRemove->empty()) {
      FilesToRemove->back().eraseFromDisk();
      FilesToRemove->pop_back();
    }

  if (CallBacksToRun)
    for (unsigned i = 0, e = CallBacksToRun->size(); i != e; ++i)
      (*CallBacksToRun)[i].first((*CallBacksToRun)[i].second);

  LeaveCriticalSection(&CriticalSection);
}

void llvm::sys::RunInterruptHandlers() {
  Cleanup();
}

static LONG WINAPI LLVMUnhandledExceptionFilter(LPEXCEPTION_POINTERS ep) {
  Cleanup();

  // Initialize the STACKFRAME structure.
  STACKFRAME64 StackFrame;
  memset(&StackFrame, 0, sizeof(StackFrame));

  DWORD machineType;
#if defined(_M_X64)
  machineType = IMAGE_FILE_MACHINE_AMD64;
  StackFrame.AddrPC.Offset = ep->ContextRecord->Rip;
  StackFrame.AddrPC.Mode = AddrModeFlat;
  StackFrame.AddrStack.Offset = ep->ContextRecord->Rsp;
  StackFrame.AddrStack.Mode = AddrModeFlat;
  StackFrame.AddrFrame.Offset = ep->ContextRecord->Rbp;
  StackFrame.AddrFrame.Mode = AddrModeFlat;
#elif defined(_M_IX86)
  machineType = IMAGE_FILE_MACHINE_I386;
  StackFrame.AddrPC.Offset = ep->ContextRecord->Eip;
  StackFrame.AddrPC.Mode = AddrModeFlat;
  StackFrame.AddrStack.Offset = ep->ContextRecord->Esp;
  StackFrame.AddrStack.Mode = AddrModeFlat;
  StackFrame.AddrFrame.Offset = ep->ContextRecord->Ebp;
  StackFrame.AddrFrame.Mode = AddrModeFlat;
#endif

  HANDLE hProcess = GetCurrentProcess();
  HANDLE hThread = GetCurrentThread();

  // Initialize the symbol handler.
  SymSetOptions(SYMOPT_DEFERRED_LOADS|SYMOPT_LOAD_LINES);
  SymInitialize(hProcess, NULL, TRUE);

  while (true) {
    if (!StackWalk64(machineType, hProcess, hThread, &StackFrame,
                   ep->ContextRecord, NULL, SymFunctionTableAccess64,
                   SymGetModuleBase64, NULL)) {
      break;
    }

    if (StackFrame.AddrFrame.Offset == 0)
      break;

    // Print the PC in hexadecimal.
    DWORD64 PC = StackFrame.AddrPC.Offset;
#if defined(_M_X64)
    fprintf(stderr, "0x%016llX", PC);
#elif defined(_M_IX86)
    fprintf(stderr, "0x%08lX", static_cast<DWORD>(PC));
#endif

    // Print the parameters.  Assume there are four.
#if defined(_M_X64)
    fprintf(stderr, " (0x%016llX 0x%016llX 0x%016llX 0x%016llX)",
                StackFrame.Params[0],
                StackFrame.Params[1],
                StackFrame.Params[2],
                StackFrame.Params[3]);
#elif defined(_M_IX86)
    fprintf(stderr, " (0x%08lX 0x%08lX 0x%08lX 0x%08lX)",
                static_cast<DWORD>(StackFrame.Params[0]),
                static_cast<DWORD>(StackFrame.Params[1]),
                static_cast<DWORD>(StackFrame.Params[2]),
                static_cast<DWORD>(StackFrame.Params[3]));
#endif
    // Verify the PC belongs to a module in this process.
    if (!SymGetModuleBase64(hProcess, PC)) {
      fputs(" <unknown module>\n", stderr);
      continue;
    }

    // Print the symbol name.
    char buffer[512];
    IMAGEHLP_SYMBOL64 *symbol = reinterpret_cast<IMAGEHLP_SYMBOL64 *>(buffer);
    memset(symbol, 0, sizeof(IMAGEHLP_SYMBOL64));
    symbol->SizeOfStruct = sizeof(IMAGEHLP_SYMBOL64);
    symbol->MaxNameLength = 512 - sizeof(IMAGEHLP_SYMBOL64);

    DWORD64 dwDisp;
    if (!SymGetSymFromAddr64(hProcess, PC, &dwDisp, symbol)) {
      fputc('\n', stderr);
      continue;
    }

    buffer[511] = 0;
    if (dwDisp > 0)
      fprintf(stderr, ", %s() + 0x%llX bytes(s)", symbol->Name, dwDisp);
    else
      fprintf(stderr, ", %s", symbol->Name);

    // Print the source file and line number information.
    IMAGEHLP_LINE64 line;
    DWORD dwLineDisp;
    memset(&line, 0, sizeof(line));
    line.SizeOfStruct = sizeof(line);
    if (SymGetLineFromAddr64(hProcess, PC, &dwLineDisp, &line)) {
      fprintf(stderr, ", %s, line %lu", line.FileName, line.LineNumber);
      if (dwLineDisp > 0)
        fprintf(stderr, " + 0x%lX byte(s)", dwLineDisp);
    }

    fputc('\n', stderr);
  }

  if (ExitOnUnhandledExceptions)
    _exit(ep->ExceptionRecord->ExceptionCode);

  // Allow dialog box to pop up allowing choice to start debugger.
  if (OldFilter)
    return (*OldFilter)(ep);
  else
    return EXCEPTION_CONTINUE_SEARCH;
}

static BOOL WINAPI LLVMConsoleCtrlHandler(DWORD dwCtrlType) {
  // We are running in our very own thread, courtesy of Windows.
  EnterCriticalSection(&CriticalSection);
  Cleanup();

  // If an interrupt function has been set, go and run one it; otherwise,
  // the process dies.
  void (*IF)() = InterruptFunction;
  InterruptFunction = 0;      // Don't run it on another CTRL-C.

  if (IF) {
    // Note: if the interrupt function throws an exception, there is nothing
    // to catch it in this thread so it will kill the process.
    IF();                     // Run it now.
    LeaveCriticalSection(&CriticalSection);
    return TRUE;              // Don't kill the process.
  }

  // Allow normal processing to take place; i.e., the process dies.
  LeaveCriticalSection(&CriticalSection);
  return FALSE;
}

#if __MINGW32__
 // We turned these warnings off for this file so that MinGW-g++ doesn't
 // complain about the ll format specifiers used.  Now we are turning the
 // warnings back on.  If MinGW starts to support diagnostic stacks, we can
 // replace this with a pop.
 #pragma GCC diagnostic warning "-Wformat"
 #pragma GCC diagnostic warning "-Wformat-extra-args"
#endif
