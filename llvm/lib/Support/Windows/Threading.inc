//===- Windows/Threading.inc - Win32 Threading Implementation - -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides the Win32 specific implementation of Threading functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"

#include "llvm/Support/Windows/WindowsSupport.h"
#include <process.h>

// Windows will at times define MemoryFence.
#ifdef MemoryFence
#undef MemoryFence
#endif

static unsigned __stdcall threadFuncSync(void *Arg) {
  SyncThreadInfo *TI = static_cast<SyncThreadInfo *>(Arg);
  TI->UserFn(TI->UserData);
  return 0;
}

static unsigned __stdcall threadFuncAsync(void *Arg) {
  std::unique_ptr<AsyncThreadInfo> Info(static_cast<AsyncThreadInfo *>(Arg));
  (*Info)();
  return 0;
}

static void
llvm_execute_on_thread_impl(unsigned (__stdcall *ThreadFunc)(void *), void *Arg,
                            llvm::Optional<unsigned> StackSizeInBytes,
                            JoiningPolicy JP) {
  HANDLE hThread = (HANDLE)::_beginthreadex(
      NULL, StackSizeInBytes.getValueOr(0), ThreadFunc, Arg, 0, NULL);

  if (!hThread) {
    ReportLastErrorFatal("_beginthreadex failed");
  }

  if (JP == JoiningPolicy::Join) {
    if (::WaitForSingleObject(hThread, INFINITE) == WAIT_FAILED) {
      ReportLastErrorFatal("WaitForSingleObject failed");
    }
  }
  if (::CloseHandle(hThread) == FALSE) {
    ReportLastErrorFatal("CloseHandle failed");
  }
}

uint64_t llvm::get_threadid() {
  return uint64_t(::GetCurrentThreadId());
}

uint32_t llvm::get_max_thread_name_length() { return 0; }

#if defined(_MSC_VER)
static void SetThreadName(DWORD Id, LPCSTR Name) {
  constexpr DWORD MS_VC_EXCEPTION = 0x406D1388;

#pragma pack(push, 8)
  struct THREADNAME_INFO {
    DWORD dwType;     // Must be 0x1000.
    LPCSTR szName;    // Pointer to thread name
    DWORD dwThreadId; // Thread ID (-1 == current thread)
    DWORD dwFlags;    // Reserved.  Do not use.
  };
#pragma pack(pop)

  THREADNAME_INFO info;
  info.dwType = 0x1000;
  info.szName = Name;
  info.dwThreadId = Id;
  info.dwFlags = 0;

  __try {
    ::RaiseException(MS_VC_EXCEPTION, 0, sizeof(info) / sizeof(ULONG_PTR),
      (ULONG_PTR *)&info);
  }
  __except (EXCEPTION_EXECUTE_HANDLER) {
  }
}
#endif

void llvm::set_thread_name(const Twine &Name) {
#if defined(_MSC_VER)
  // Make sure the input is null terminated.
  SmallString<64> Storage;
  StringRef NameStr = Name.toNullTerminatedStringRef(Storage);
  SetThreadName(::GetCurrentThreadId(), NameStr.data());
#endif
}

void llvm::get_thread_name(SmallVectorImpl<char> &Name) {
  // "Name" is not an inherent property of a thread on Windows.  In fact, when
  // you "set" the name, you are only firing a one-time message to a debugger
  // which it interprets as a program setting its threads' name.  We may be
  // able to get fancy by creating a TLS entry when someone calls
  // set_thread_name so that subsequent calls to get_thread_name return this
  // value.
  Name.clear();
}

SetThreadPriorityResult llvm::set_thread_priority(ThreadPriority Priority) {
  // https://docs.microsoft.com/en-us/windows/desktop/api/processthreadsapi/nf-processthreadsapi-setthreadpriority
  // Begin background processing mode. The system lowers the resource scheduling
  // priorities of the thread so that it can perform background work without
  // significantly affecting activity in the foreground.
  // End background processing mode. The system restores the resource scheduling
  // priorities of the thread as they were before the thread entered background
  // processing mode.
  return SetThreadPriority(GetCurrentThread(),
                           Priority == ThreadPriority::Background
                               ? THREAD_MODE_BACKGROUND_BEGIN
                               : THREAD_MODE_BACKGROUND_END)
             ? SetThreadPriorityResult::SUCCESS
             : SetThreadPriorityResult::FAILURE;
}
