//===-- PDBFPOProgramToDWARFExpression.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Plugins_SymbolFile_PDB_PDBFPOProgramToDWARFExpression_h_
#define lldb_Plugins_SymbolFile_PDB_PDBFPOProgramToDWARFExpression_h_

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"

namespace lldb_private {
class Stream;

namespace npdb {
  
bool TranslateFPOProgramToDWARFExpression(llvm::StringRef program,
                                          llvm::StringRef register_name,
                                          llvm::Triple::ArchType arch_type,
                                          lldb_private::Stream &stream);

} // namespace npdb
} // namespace lldb_private

#endif
