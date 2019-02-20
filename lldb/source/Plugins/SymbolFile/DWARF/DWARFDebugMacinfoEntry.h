//===-- DWARFDebugMacinfoEntry.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARF_DWARFDebugMacinfoEntry_h_
#define SymbolFileDWARF_DWARFDebugMacinfoEntry_h_

#include "SymbolFileDWARF.h"

class DWARFDebugMacinfoEntry {
public:
  DWARFDebugMacinfoEntry();

  ~DWARFDebugMacinfoEntry();

  uint8_t TypeCode() const { return m_type_code; }

  uint8_t GetLineNumber() const { return m_line; }

  void Dump(lldb_private::Stream *s) const;

  const char *GetCString() const;

  bool Extract(const lldb_private::DWARFDataExtractor &mac_info_data,
               lldb::offset_t *offset_ptr);

protected:
private:
  uint8_t m_type_code;
  dw_uleb128_t m_line;
  union {
    dw_uleb128_t file_idx;
    const char *cstr;
  } m_op2;
};

#endif // SymbolFileDWARF_DWARFDebugMacinfoEntry_h_
