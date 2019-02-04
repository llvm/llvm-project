//===-- CodeViewRegisterMapping.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
