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
  return GetU32(offset_ptr);
}

dw_offset_t
DWARFDataExtractor::GetDWARFOffset(lldb::offset_t *offset_ptr) const {
  return GetMaxU64(offset_ptr, GetDWARFSizeOfOffset());
}
}
