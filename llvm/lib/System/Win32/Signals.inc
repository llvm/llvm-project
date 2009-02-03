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

#include "Win32.h"
#include <stdio.h>
#include <vector>

#ifdef __MINGW32__
 #include <imagehlp.h>
#else
 #include <dbghelp.h>
#endif
#include <psapi.h>

#ifdef __MINGW32__
 #if ((HAVE_LIBIMAGEHLP != 1) || (HAVE_LIBPSAPI != 1))
  #error "libimagehlp.a & libpsapi.a should be present"
 #endif
#else
 #pragma comment(lib, "psapi.lib")
 #pragma comment(lib, "dbghelp.lib")
#endif

// Forward declare.
static LONG WINAPI LLVMUnhandledExceptionFilter(LPEXCEPTION_POINTERS ep);
static BOOL WINAPI LLVMConsoleCtrlHandler(DWORD dwCtrlType);

// InterruptFunction - The function to call if ctrl-c is pressed.
static void (*InterruptFunction)() = 0;

static std::vector<llvm::sys::Path> *FilesToRemove = NULL;
static std::vector<llvm::sys::Path> *DirectoriesToRemove = NULL;
static bool RegisteredUnhandledExceptionFilter = false;
static bool CleanupExecuted = false;
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


static void RegisterHandler() { 
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

// RemoveDirectoryOnSignal - The public API
bool sys::RemoveDirectoryOnSignal(const sys::Path& path, std::string* ErrMsg) {
  // Not a directory?
  WIN32_FILE_ATTRIBUTE_DATA fi;
  if (!GetFileAttributesEx(path.c_str(), GetFileExInfoStandard, &fi)) {
    MakeErrMsg(ErrMsg, path.toString() + ": can't get status of file");
    return true;
  }
    
  if (!(fi.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
    if (ErrMsg)
      *ErrMsg = path.toString() + ": not a directory";
    return true;
  }

  RegisterHandler();

  if (CleanupExecuted) {
    if (ErrMsg)
      *ErrMsg = "Process terminating -- cannot register for removal";
    return true;
  }

  if (DirectoriesToRemove == NULL)
    DirectoriesToRemove = new std::vector<sys::Path>;
  DirectoriesToRemove->push_back(path);

  LeaveCriticalSection(&CriticalSection);
  return false;
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
}

static void Cleanup() {
  EnterCriticalSection(&CriticalSection);

  // Prevent other thread from registering new files and directories for
  // removal, should we be executing because of the console handler callback.
  CleanupExecuted = true;

  // FIXME: open files cannot be deleted.

  if (FilesToRemove != NULL)
    while (!FilesToRemove->empty()) {
      try {
        FilesToRemove->back().eraseFromDisk();
      } catch (...) {
      }
      FilesToRemove->pop_back();
    }

  if (DirectoriesToRemove != NULL)
    while (!DirectoriesToRemove->empty()) {
      try {
        DirectoriesToRemove->back().eraseFromDisk(true);
      } catch (...) {
      }
      DirectoriesToRemove->pop_back();
    }

  LeaveCriticalSection(&CriticalSection);
}

static LONG WINAPI LLVMUnhandledExceptionFilter(LPEXCEPTION_POINTERS ep) {
  try {
    Cleanup();
    
#ifdef _WIN64
  // TODO: provide a x64 friendly version of the following
#else
    
    // Initialize the STACKFRAME structure.
    STACKFRAME StackFrame;
    memset(&StackFrame, 0, sizeof(StackFrame));

    StackFrame.AddrPC.Offset = ep->ContextRecord->Eip;
    StackFrame.AddrPC.Mode = AddrModeFlat;
    StackFrame.AddrStack.Offset = ep->ContextRecord->Esp;
    StackFrame.AddrStack.Mode = AddrModeFlat;
    StackFrame.AddrFrame.Offset = ep->ContextRecord->Ebp;
    StackFrame.AddrFrame.Mode = AddrModeFlat;

    HANDLE hProcess = GetCurrentProcess();
    HANDLE hThread = GetCurrentThread();

    // Initialize the symbol handler.
    SymSetOptions(SYMOPT_DEFERRED_LOADS|SYMOPT_LOAD_LINES);
    SymInitialize(hProcess, NULL, TRUE);

    while (true) {
      if (!StackWalk(IMAGE_FILE_MACHINE_I386, hProcess, hThread, &StackFrame,
                     ep->ContextRecord, NULL, SymFunctionTableAccess,
                     SymGetModuleBase, NULL)) {
        break;
      }

      if (StackFrame.AddrFrame.Offset == 0)
        break;

      // Print the PC in hexadecimal.
      DWORD PC = StackFrame.AddrPC.Offset;
      fprintf(stderr, "%08X", PC);

      // Print the parameters.  Assume there are four.
      fprintf(stderr, " (0x%08X 0x%08X 0x%08X 0x%08X)", StackFrame.Params[0],
              StackFrame.Params[1], StackFrame.Params[2], StackFrame.Params[3]);

      // Verify the PC belongs to a module in this process.
      if (!SymGetModuleBase(hProcess, PC)) {
        fputs(" <unknown module>\n", stderr);
        continue;
      }

      // Print the symbol name.
      char buffer[512];
      IMAGEHLP_SYMBOL *symbol = reinterpret_cast<IMAGEHLP_SYMBOL *>(buffer);
      memset(symbol, 0, sizeof(IMAGEHLP_SYMBOL));
      symbol->SizeOfStruct = sizeof(IMAGEHLP_SYMBOL);
      symbol->MaxNameLength = 512 - sizeof(IMAGEHLP_SYMBOL);

      DWORD dwDisp;
      if (!SymGetSymFromAddr(hProcess, PC, &dwDisp, symbol)) {
        fputc('\n', stderr);
        continue;
      }

      buffer[511] = 0;
      if (dwDisp > 0)
        fprintf(stderr, ", %s()+%04d bytes(s)", symbol->Name, dwDisp);
      else
        fprintf(stderr, ", %s", symbol->Name);

      // Print the source file and line number information.
      IMAGEHLP_LINE line;
      memset(&line, 0, sizeof(line));
      line.SizeOfStruct = sizeof(line);
      if (SymGetLineFromAddr(hProcess, PC, &dwDisp, &line)) {
        fprintf(stderr, ", %s, line %d", line.FileName, line.LineNumber);
        if (dwDisp > 0)
          fprintf(stderr, "+%04d byte(s)", dwDisp);
      }

      fputc('\n', stderr);
    }

#endif

  } catch (...) {
      assert(!"Crashed in LLVMUnhandledExceptionFilter");
  }

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

