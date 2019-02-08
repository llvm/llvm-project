//===-- CodeViewRegisterMapping.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Plugins_SymbolFile_PDB_CodeViewRegisterMapping_h_
#define lldb_Plugins_SymbolFile_PDB_CodeViewRegisterMapping_h_

#include "llvm/ADT/Triple.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"

namespace lldb_private {
namespace npdb {

uint32_t GetLLDBRegisterNumber(llvm::Triple::ArchType arch_type,
                               llvm::codeview::RegisterId register_id);

} // namespace npdb
} // namespace lldb_private

#endif
