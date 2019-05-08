//===-- DWARFContext.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_SYMBOLFILE_DWARF_DWARFCONTEXT_H
#define LLDB_PLUGINS_SYMBOLFILE_DWARF_DWARFCONTEXT_H

#include "DWARFDataExtractor.h"
#include "lldb/Core/Module.h"
#include "llvm/ADT/Optional.h"
#include <memory>

namespace lldb_private {
class DWARFContext {
private:
  Module &m_module;
  llvm::Optional<DWARFDataExtractor> m_data_debug_aranges;

public:
  explicit DWARFContext(Module &module);

  const DWARFDataExtractor *getOrLoadArangesData();
};
} // namespace lldb_private

#endif
