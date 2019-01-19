//===-- DWARFDebugMacinfo.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARF_DWARFDebugMacinfo_h_
#define SymbolFileDWARF_DWARFDebugMacinfo_h_

#include "SymbolFileDWARF.h"

class DWARFDebugMacinfo {
public:
  DWARFDebugMacinfo();

  ~DWARFDebugMacinfo();

  static void Dump(lldb_private::Stream *s,
                   const lldb_private::DWARFDataExtractor &macinfo_data,
                   lldb::offset_t offset = LLDB_INVALID_OFFSET);
};

#endif // SymbolFileDWARF_DWARFDebugMacinfo_h_
