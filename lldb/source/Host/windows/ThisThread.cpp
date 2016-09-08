//===-- ThisThread.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Error.h"

#include "lldb/Host/ThisThread.h"
#include "lldb/Host/windows/windows.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace lldb;
using namespace lldb_private;

#if defined(_MSC_VER) && !defined(__clang__)

namespace {
static const DWORD MS_VC_EXCEPTION = 0x406D1388;

#pragma pack(push, 8)
struct THREADNAME_INFO {
  DWORD dwType;     // Must be 0x1000.
  LPCSTR szName;    // Pointer to thread name
  DWORD dwThreadId; // Thread ID (-1 == current thread)
  DWORD dwFlags;    // Reserved.  Do not use.
};
#pragma pack(pop)
}

#endif

void ThisThread::SetName(llvm::StringRef name) {
// Other compilers don't yet support SEH, so we can only set the thread if
// compiling with MSVC.
// TODO(zturner): Once clang-cl supports SEH, relax this conditional.
#if defined(_MSC_VER) && !defined(__clang__)
  THREADNAME_INFO info;
  info.dwType = 0x1000;
  info.szName = name.data();
  info.dwThreadId = ::GetCurrentThreadId();
  info.dwFlags = 0;

  __try {
    ::RaiseException(MS_VC_EXCEPTION, 0, sizeof(info) / sizeof(ULONG_PTR),
                     (ULONG_PTR *)&info);
  } __except (EXCEPTION_EXECUTE_HANDLER) {
  }
#endif
}

void ThisThread::GetName(llvm::SmallVectorImpl<char> &name) {
  // Getting the thread name is not supported on Windows.
  // TODO(zturner): In SetName(), make a TLS entry that contains the thread's
  // name, and in this function
  // try to extract that TLS entry.
  name.clear();
}
