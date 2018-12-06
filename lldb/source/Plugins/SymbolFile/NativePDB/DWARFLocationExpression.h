//===-- DWARFLocationExpression.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_SYMBOLFILE_NATIVEPDB_DWARFLOCATIONEXPRESSION_H
#define LLDB_PLUGINS_SYMBOLFILE_NATIVEPDB_DWARFLOCATIONEXPRESSION_H

#include "lldb/lldb-forward.h"

namespace llvm {
class APSInt;
namespace codeview {
class TypeIndex;
}
namespace pdb {
class TpiStream;
}
} // namespace llvm
namespace lldb_private {
namespace npdb {
DWARFExpression MakeGlobalLocationExpression(uint16_t section, uint32_t offset,
                                             lldb::ModuleSP module);
DWARFExpression MakeConstantLocationExpression(
    llvm::codeview::TypeIndex underlying_ti, llvm::pdb::TpiStream &tpi,
    const llvm::APSInt &constant, lldb::ModuleSP module);
} // namespace npdb
} // namespace lldb_private

#endif
