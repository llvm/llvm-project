//===-- PDBFPOProgramToDWARFExpression.h ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
