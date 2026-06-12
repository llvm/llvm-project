//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExceptionRecord.h"
#include "lldb/Host/windows/windows.h"
#include "lldb/lldb-forward.h"
#include "lldb/lldb-types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <dbghelp.h>
#include <memory>
#include <vector>

using namespace lldb_private;

ExceptionRecord::ExceptionRecord(const EXCEPTION_RECORD &record,
                                 lldb::tid_t thread_id) {
  // Notes about the `record.ExceptionRecord` field:
  // In the past, some code tried to parse the nested exception with it, but
  // in practice, that code just causes Access Violation. I suspect
  // `ExceptionRecord` here actually points to the address space of the
  // debuggee process. However, I did not manage to find any official or
  // unofficial reference that clarifies this point. If anyone would like to
  // reimplement this, please also keep in mind to check how this behaves when
  // debugging a WOW64 process. I suspect you may have to use the explicit
  // `EXCEPTION_RECORD32` and `EXCEPTION_RECORD64` structs.
  m_code = record.ExceptionCode;
  m_continuable = (record.ExceptionFlags == 0);
  m_exception_addr = reinterpret_cast<lldb::addr_t>(record.ExceptionAddress);
  m_thread_id = thread_id;
  m_arguments.assign(record.ExceptionInformation,
                     record.ExceptionInformation + record.NumberParameters);
}

lldb_private::ExceptionRecord::ExceptionRecord(const MINIDUMP_EXCEPTION &record,
                                               lldb::tid_t thread_id)
    : m_code(record.ExceptionCode), m_continuable(record.ExceptionFlags == 0),
      m_exception_addr(static_cast<lldb::addr_t>(record.ExceptionAddress)),
      m_thread_id(thread_id),
      m_arguments(record.ExceptionInformation,
                  record.ExceptionInformation + record.NumberParameters) {}

void lldb_private::ExceptionRecord::Dump(llvm::raw_ostream &stream) const {
  // Decode additional exception information for specific exception types based
  // on
  // https://docs.microsoft.com/en-us/windows/desktop/api/winnt/ns-winnt-_exception_record
  // Mirrors ProcessWindows::DumpAdditionalExceptionInformation so that
  // lldb-server stop descriptions match the in-process debugger.

  const int addr_min_width = 2 + 8; // "0x" + 4 address bytes

  const llvm::ArrayRef<unsigned long long> args = GetExceptionArguments();
  switch (GetExceptionValue()) {
  case EXCEPTION_ACCESS_VIOLATION: {
    if (args.size() < 2)
      break;

    stream << ": ";
    const int access_violation_code = args[0];
    const lldb::addr_t access_violation_address = args[1];
    switch (access_violation_code) {
    case 0:
      stream << "Access violation reading";
      break;
    case 1:
      stream << "Access violation writing";
      break;
    case 8:
      stream << "User-mode data execution prevention (DEP) violation at";
      break;
    default:
      stream << "Unknown access violation (code " << access_violation_code
             << ") at";
      break;
    }
    stream << " location "
           << llvm::format_hex(access_violation_address, addr_min_width);
    break;
  }
  case EXCEPTION_IN_PAGE_ERROR: {
    if (args.size() < 3)
      break;

    stream << ": ";
    const int page_load_error_code = args[0];
    const lldb::addr_t page_load_error_address = args[1];
    const DWORD underlying_code = args[2];
    switch (page_load_error_code) {
    case 0:
      stream << "In page error reading";
      break;
    case 1:
      stream << "In page error writing";
      break;
    case 8:
      stream << "User-mode data execution prevention (DEP) violation at";
      break;
    default:
      stream << "Unknown page loading error (code " << page_load_error_code
             << ") at";
      break;
    }
    stream << " location "
           << llvm::format_hex(page_load_error_address, addr_min_width)
           << " (status code " << llvm::format_hex(underlying_code, 8) << ")";
    break;
  }
  }
}
