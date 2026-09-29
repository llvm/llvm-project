//===-- HostThreadWindows.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/windows/LazyImport.h"
#include "lldb/Utility/Status.h"

#include "lldb/Host/windows/HostThreadWindows.h"
#include "lldb/Host/windows/windows.h"

#include "llvm/ADT/STLExtras.h"

using namespace lldb;
using namespace lldb_private;

static void __stdcall ExitThreadProxy(ULONG_PTR dwExitCode) {
  ::ExitThread(dwExitCode);
}

HostThreadWindows::HostThreadWindows()
    : HostNativeThreadBase(), m_owns_handle(true) {}

HostThreadWindows::HostThreadWindows(lldb::thread_t thread)
    : HostNativeThreadBase(thread), m_owns_handle(true) {}

HostThreadWindows::~HostThreadWindows() { Reset(); }

void HostThreadWindows::SetOwnsHandle(bool owns) { m_owns_handle = owns; }

Status HostThreadWindows::Join(lldb::thread_result_t *result) {
  if (!IsJoinable())
    return Status(ERROR_INVALID_HANDLE, eErrorTypeWin32);

  Status error;
  DWORD wait_result = ::WaitForSingleObject(m_thread, INFINITE);
  if (wait_result == WAIT_OBJECT_0) {
    if (result) {
      DWORD exit_code = 0;
      if (::GetExitCodeThread(m_thread, &exit_code))
        *result = exit_code;
      else
        *result = 0;
    }
  } else {
    error = Status(::GetLastError(), eErrorTypeWin32);
  }

  Reset();
  return error;
}

Status HostThreadWindows::Cancel() {
  if (!::QueueUserAPC(&ExitThreadProxy, m_thread, 0))
    return Status(::GetLastError(), eErrorTypeWin32);
  return Status();
}

lldb::tid_t HostThreadWindows::GetThreadId() const {
  return ::GetThreadId(m_thread);
}

void HostThreadWindows::Reset() {
  if (m_owns_handle && m_thread != LLDB_INVALID_HOST_THREAD)
    ::CloseHandle(m_thread);

  HostNativeThreadBase::Reset();
}

bool HostThreadWindows::EqualsThread(lldb::thread_t thread) const {
  return GetThreadId() == ::GetThreadId(thread);
}

StructuredData::ObjectSP HostThreadWindows::GetExtendedInfo() const {
  struct ThreadBasicInformation {
    LONG ExitStatus;
    PVOID TebBaseAddress;
    struct {
      HANDLE UniqueProcess;
      HANDLE UniqueThread;
    } ClientId;
    ULONG_PTR AffinityMask;
    LONG Priority;
    LONG BasePriority;
  };
  using NtQueryInformationThreadFn =
      LONG(WINAPI *)(HANDLE ThreadHandle, ULONG ThreadInformationClass,
                     PVOID ThreadInformation, ULONG ThreadInformationLength,
                     PULONG ReturnLength);

  static LazyImport<NtQueryInformationThreadFn> s_query_information_thread{
      L"Kernel32.dll", "NtQueryInformationThread"};
  if (!s_query_information_thread)
    return StructuredData::ObjectSP();
  auto NtQueryInformationThread = *s_query_information_thread;

  HANDLE handle = GetSystemHandle();
  if (!handle || handle == INVALID_HANDLE_VALUE)
    return StructuredData::ObjectSP();

  ThreadBasicInformation info = {};
  // ThreadInformationClass 0 is ThreadBasicInformation.
  if (NtQueryInformationThread(handle, 0, &info, sizeof(info), nullptr) < 0)
    return StructuredData::ObjectSP();
  if (!info.TebBaseAddress)
    return StructuredData::ObjectSP();

  auto dict = std::make_shared<StructuredData::Dictionary>();
  dict->AddIntegerItem("teb_address",
                       reinterpret_cast<uint64_t>(info.TebBaseAddress));
  return dict;
}
