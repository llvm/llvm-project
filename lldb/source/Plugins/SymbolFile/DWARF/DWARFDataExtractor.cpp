//===-- DWARFDataExtractor.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DWARFDataExtractor.h"

namespace lldb_private {

uint64_t
DWARFDataExtractor::GetDWARFInitialLength(lldb::offset_t *offset_ptr) const {
  uint64_t length = GetU32(offset_ptr);
  m_is_dwarf64 = (length == UINT32_MAX);
  if (m_is_dwarf64)
    length = GetU64(offset_ptr);
  return length;
}

dw_offset_t
DWARFDataExtractor::GetDWARFOffset(lldb::offset_t *offset_ptr) const {
  return GetMaxU64(offset_ptr, GetDWARFSizeOfOffset());
}
}
