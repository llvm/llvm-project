//===-- ExceptionRecord.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Plugins_Process_Windows_ExceptionRecord_H_
#define liblldb_Plugins_Process_Windows_ExceptionRecord_H_

#include "lldb/lldb-forward.h"
#include "lldb/lldb-types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <vector>

struct _EXCEPTION_RECORD;
typedef struct _EXCEPTION_RECORD EXCEPTION_RECORD;

struct _MINIDUMP_EXCEPTION;
typedef struct _MINIDUMP_EXCEPTION MINIDUMP_EXCEPTION;

namespace lldb_private {

// ExceptionRecord
//
// ExceptionRecord defines an interface which allows implementors to receive
// notification of events that happen in a debugged process.
class ExceptionRecord {
public:
  ExceptionRecord(const EXCEPTION_RECORD &record, lldb::tid_t thread_id);

  // MINIDUMP_EXCEPTIONs are almost identical to EXCEPTION_RECORDs.
  ExceptionRecord(const MINIDUMP_EXCEPTION &record, lldb::tid_t thread_id);

  virtual ~ExceptionRecord() = default;

  unsigned long GetExceptionValue() const { return m_code; }
  bool IsContinuable() const { return m_continuable; }
  lldb::addr_t GetExceptionAddress() const { return m_exception_addr; }

  lldb::tid_t GetThreadID() const { return m_thread_id; }

  llvm::ArrayRef<uint64_t> GetExceptionArguments() const { return m_arguments; }

  void Dump(llvm::raw_ostream &stream) const;

private:
  unsigned long m_code;
  bool m_continuable;
  lldb::addr_t m_exception_addr;
  lldb::tid_t m_thread_id;
  std::vector<uint64_t> m_arguments;
};
}

#endif
