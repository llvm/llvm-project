//===- CVSymbolVisitor.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_CVSYMBOLVISITOR_H
#define LLVM_DEBUGINFO_CODEVIEW_CVSYMBOLVISITOR_H

#include "llvm/DebugInfo/CodeView/CVRecord.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace codeview {
class SymbolVisitorCallbacks;

class CVSymbolVisitor {
public:
  struct FilterOptions {
    std::optional<uint32_t> SymbolOffset;
    std::optional<uint32_t> ParentRecursiveDepth;
    std::optional<uint32_t> ChildRecursiveDepth;
  };

  LLVM_ABI CVSymbolVisitor(SymbolVisitorCallbacks &Callbacks);

  LLVM_ABI Error visitSymbolRecord(CVSymbol &Record);
  LLVM_ABI Error visitSymbolRecord(CVSymbol &Record, uint32_t Offset);
  LLVM_ABI Error visitSymbolStream(const CVSymbolArray &Symbols);
  LLVM_ABI Error visitSymbolStream(const CVSymbolArray &Symbols,
                                   uint32_t InitialOffset);
  LLVM_ABI Error visitSymbolStreamFiltered(const CVSymbolArray &Symbols,
                                           const FilterOptions &Filter);

private:
  SymbolVisitorCallbacks &Callbacks;
};

} // end namespace codeview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_CVSYMBOLVISITOR_H
