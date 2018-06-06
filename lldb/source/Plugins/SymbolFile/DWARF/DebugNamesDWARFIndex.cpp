//===-- DebugNamesDWARFIndex.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Plugins/SymbolFile/DWARF/DebugNamesDWARFIndex.h"

using namespace lldb_private;
using namespace lldb;

static llvm::DWARFDataExtractor ToLLVM(const DWARFDataExtractor &data) {
  return llvm::DWARFDataExtractor(
      llvm::StringRef(reinterpret_cast<const char *>(data.GetDataStart()),
                      data.GetByteSize()),
      data.GetByteOrder() == eByteOrderLittle, data.GetAddressByteSize());
}

llvm::Expected<std::unique_ptr<DebugNamesDWARFIndex>>
DebugNamesDWARFIndex::Create(Module &module, DWARFDataExtractor debug_names,
                             DWARFDataExtractor debug_str,
                             DWARFDebugInfo *debug_info) {
  auto index_up = llvm::make_unique<llvm::DWARFDebugNames>(ToLLVM(debug_names),
                                                           ToLLVM(debug_str));
  if (llvm::Error E = index_up->extract())
    return std::move(E);

  return std::unique_ptr<DebugNamesDWARFIndex>(new DebugNamesDWARFIndex(
      module, std::move(index_up), debug_names, debug_str, debug_info));
}
