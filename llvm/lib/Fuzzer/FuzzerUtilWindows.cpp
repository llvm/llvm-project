//===- FuzzerUtilWindows.cpp - Misc utils for Windows. --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Misc utils implementation for Windows.
//===----------------------------------------------------------------------===//

#include "FuzzerDefs.h"
#if LIBFUZZER_WINDOWS
#include "FuzzerIO.h"
#include "FuzzerInternal.h"
#include <Psapi.h>
#include <cassert>
#include <chrono>
#include <cstring>
#include <errno.h>
#include <iomanip>
#include <signal.h>
#include <sstream>
#include <stdio.h>
#include <sys/types.h>
#include <windows.h>

namespace fuzzer {

LONG WINAPI SEGVHandler(PEXCEPTION_POINTERS ExceptionInfo) {
  switch (ExceptionInfo->ExceptionRecord->ExceptionCode) {
  case EXCEPTION_ACCESS_VIOLATION:
  case EXCEPTION_ARRAY_BOUNDS_EXCEEDED:
  case EXCEPTION_STACK_OVERFLOW:
    Fuzzer::StaticCrashSignalCallback();
    break;
  }
  return EXCEPTION_CONTINUE_SEARCH;
}

LONG WINAPI BUSHandler(PEXCEPTION_POINTERS ExceptionInfo) {
  switch (ExceptionInfo->ExceptionRecord->ExceptionCode) {
  case EXCEPTION_DATATYPE_MISALIGNMENT:
  case EXCEPTION_IN_PAGE_ERROR:
    Fuzzer::StaticCrashSignalCallback();
    break;
  }
  return EXCEPTION_CONTINUE_SEARCH;
}

LONG WINAPI ILLHandler(PEXCEPTION_POINTERS ExceptionInfo) {
  switch (ExceptionInfo->ExceptionRecord->ExceptionCode) {
  case EXCEPTION_ILLEGAL_INSTRUCTION:
  case EXCEPTION_PRIV_INSTRUCTION:
    Fuzzer::StaticCrashSignalCallback();
    break;
  }
  return EXCEPTION_CONTINUE_SEARCH;
}

LONG WINAPI FPEHandler(PEXCEPTION_POINTERS ExceptionInfo) {
  switch (ExceptionInfo->ExceptionRecord->ExceptionCode) {
  case EXCEPTION_FLT_DENORMAL_OPERAND:
  case EXCEPTION_FLT_DIVIDE_BY_ZERO:
  case EXCEPTION_FLT_INEXACT_RESULT:
  case EXCEPTION_FLT_INVALID_OPERATION:
  case EXCEPTION_FLT_OVERFLOW:
  case EXCEPTION_FLT_STACK_CHECK:
  case EXCEPTION_FLT_UNDERFLOW:
  case EXCEPTION_INT_DIVIDE_BY_ZERO:
  case EXCEPTION_INT_OVERFLOW:
    Fuzzer::StaticCrashSignalCallback();
    break;
  }
  return EXCEPTION_CONTINUE_SEARCH;
}

BOOL WINAPI INTHandler(DWORD dwCtrlType) {
  switch (dwCtrlType) {
  case CTRL_C_EVENT:
    Fuzzer::StaticInterruptCallback();
    return TRUE;
  default:
    return FALSE;
  }
}

BOOL WINAPI TERMHandler(DWORD dwCtrlType) {
  switch (dwCtrlType) {
  case CTRL_BREAK_EVENT:
    Fuzzer::StaticInterruptCallback();
    return TRUE;
  default:
    return FALSE;
  }
}

void CALLBACK AlarmHandler(PVOID, BOOLEAN) {
  Fuzzer::StaticAlarmCallback();
}

class TimerQ {
  HANDLE TimerQueue;
 public:
  TimerQ() : TimerQueue(NULL) {};
  ~TimerQ() {
    if (TimerQueue)
      DeleteTimerQueueEx(TimerQueue, NULL);
  };
  void SetTimer(int Seconds) {
    if (!TimerQueue) {
      TimerQueue = CreateTimerQueue();
      if (!TimerQueue) {
        Printf("libFuzzer: CreateTimerQueue failed.\n");
        exit(1);
      }
    }
    HANDLE Timer;
    if (!CreateTimerQueueTimer(&Timer, TimerQueue, AlarmHandler, NULL,
        Seconds*1000, Seconds*1000, 0)) {
      Printf("libFuzzer: CreateTimerQueueTimer failed.\n");
      exit(1);
    }
  };
};

static TimerQ Timer;

void SetTimer(int Seconds) {
  Timer.SetTimer(Seconds);
  return;
}

void SetSigSegvHandler() {
  if (!AddVectoredExceptionHandler(1, SEGVHandler)) {
    Printf("libFuzzer: AddVectoredExceptionHandler failed.\n");
    exit(1);
  }
}

void SetSigBusHandler() {
  if (!AddVectoredExceptionHandler(1, BUSHandler)) {
    Printf("libFuzzer: AddVectoredExceptionHandler failed.\n");
    exit(1);
  }
}

static void CrashHandler(int) { Fuzzer::StaticCrashSignalCallback(); }

void SetSigAbrtHandler() { signal(SIGABRT, CrashHandler); }

void SetSigIllHandler() {
  if (!AddVectoredExceptionHandler(1, ILLHandler)) {
    Printf("libFuzzer: AddVectoredExceptionHandler failed.\n");
    exit(1);
  }
}

void SetSigFpeHandler() {
  if (!AddVectoredExceptionHandler(1, FPEHandler)) {
    Printf("libFuzzer: AddVectoredExceptionHandler failed.\n");
    exit(1);
  }
}

void SetSigIntHandler() {
  if (!SetConsoleCtrlHandler(INTHandler, TRUE)) {
    DWORD LastError = GetLastError();
    Printf("libFuzzer: SetConsoleCtrlHandler failed (Error code: %lu).\n",
           LastError);
    exit(1);
  }
}

void SetSigTermHandler() {
  if (!SetConsoleCtrlHandler(TERMHandler, TRUE)) {
    DWORD LastError = GetLastError();
    Printf("libFuzzer: SetConsoleCtrlHandler failed (Error code: %lu).\n",
           LastError);
    exit(1);
  }
}

void SleepSeconds(int Seconds) { Sleep(Seconds * 1000); }

int GetPid() { return GetCurrentProcessId(); }

size_t GetPeakRSSMb() {
  PROCESS_MEMORY_COUNTERS info;
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info)))
    return 0;
  return info.PeakWorkingSetSize >> 20;
}

FILE *OpenProcessPipe(const char *Command, const char *Mode) {
  return _popen(Command, Mode);
}

int ExecuteCommand(const std::string &Command) {
  return system(Command.c_str());
}

const void *SearchMemory(const void *Data, size_t DataLen, const void *Patt,
                         size_t PattLen) {
  // TODO: make this implementation more efficient.
  const char *Cdata = (const char *)Data;
  const char *Cpatt = (const char *)Patt;

  if (!Data || !Patt || DataLen == 0 || PattLen == 0 || DataLen < PattLen)
    return NULL;

  if (PattLen == 1)
    return memchr(Data, *Cpatt, DataLen);

  const char *End = Cdata + DataLen - PattLen;

  for (const char *It = Cdata; It < End; ++It)
    if (It[0] == Cpatt[0] && memcmp(It, Cpatt, PattLen) == 0)
      return It;

  return NULL;
}

} // namespace fuzzer
#endif // LIBFUZZER_WINDOWS
