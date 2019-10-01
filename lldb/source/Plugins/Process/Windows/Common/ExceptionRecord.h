//===-- ExceptionRecord.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Plugins_Process_Windows_ExceptionRecord_H_
#define liblldb_Plugins_Process_Windows_ExceptionRecord_H_

#include "lldb/Host/windows/windows.h"
#include "lldb/lldb-forward.h"
#include <dbghelp.h>

#include <memory>
#include <vector>

namespace lldb_private {

// ExceptionRecord
//
// ExceptionRecord defines an interface which allows implementors to receive
// notification of events that happen in a debugged process.
class ExceptionRecord {
public:
  ExceptionRecord(const EXCEPTION_RECORD &record, lldb::tid_t thread_id) {
    m_code = record.ExceptionCode;
    m_continuable = (record.ExceptionFlags == 0);
    if (record.ExceptionRecord)
      m_next_exception.reset(
          new ExceptionRecord(*record.ExceptionRecord, thread_id));
    m_exception_addr = reinterpret_cast<lldb::addr_t>(record.ExceptionAddress);
    m_thread_id = thread_id;
    m_arguments.assign(record.ExceptionInformation,
                       record.ExceptionInformation + record.NumberParameters);
  }

  // MINIDUMP_EXCEPTIONs are almost identical to EXCEPTION_RECORDs.
  ExceptionRecord(const MINIDUMP_EXCEPTION &record, lldb::tid_t thread_id)
      : m_code(record.ExceptionCode), m_continuable(record.ExceptionFlags == 0),
        m_next_exception(nullptr),
        m_exception_addr(static_cast<lldb::addr_t>(record.ExceptionAddress)),
        m_thread_id(thread_id),
        m_arguments(record.ExceptionInformation,
                    record.ExceptionInformation + record.NumberParameters) {
    // Set up link to nested exception.
    if (record.ExceptionRecord) {
      m_next_exception.reset(new ExceptionRecord(
          *reinterpret_cast<const MINIDUMP_EXCEPTION *>(record.ExceptionRecord),
          thread_id));
    }
  }

  virtual ~ExceptionRecord() {}

  DWORD
  GetExceptionCode() const { return m_code; }
  bool IsContinuable() const { return m_continuable; }
  const ExceptionRecord *GetNextException() const {
    return m_next_exception.get();
  }
  lldb::addr_t GetExceptionAddress() const { return m_exception_addr; }

  lldb::tid_t GetThreadID() const { return m_thread_id; }

  const std::vector<ULONG_PTR>& GetExceptionArguments() const { return m_arguments; }

private:
  DWORD m_code;
  bool m_continuable;
  std::shared_ptr<ExceptionRecord> m_next_exception;
  lldb::addr_t m_exception_addr;
  lldb::tid_t m_thread_id;
  std::vector<ULONG_PTR> m_arguments;
};
}

#endif
