//===- llvm/CodeGen/DwarfStringPool.h - Dwarf Debug Framework ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_DWARFSTRINGPOOL_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_DWARFSTRINGPOOL_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/DwarfStringPoolEntry.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class AsmPrinter;
class MCSection;
class MCSymbol;

// Collection of strings for this unit and assorted symbols.
// A String->Symbol mapping of strings used by indirect
// references.
class DwarfStringPool {
  using EntryTy = DwarfStringPoolEntry;

  StringMap<EntryTy, BumpPtrAllocator &> Pool;
  StringRef Prefix;
  uint64_t NumBytes = 0;
  unsigned NumIndexedStrings = 0;
  bool ShouldCreateSymbols;

  StringMapEntry<EntryTy> &getEntryImpl(AsmPrinter &Asm, StringRef Str);

public:
  using EntryRef = DwarfStringPoolEntryRef;

  LLVM_ABI_FOR_TEST DwarfStringPool(BumpPtrAllocator &A, AsmPrinter &Asm,
                                    StringRef Prefix);

  LLVM_ABI_FOR_TEST void emitStringOffsetsTableHeader(AsmPrinter &Asm,
                                                      MCSection *OffsetSection,
                                                      MCSymbol *StartSym);

  LLVM_ABI_FOR_TEST void emit(AsmPrinter &Asm, MCSection *StrSection,
                              MCSection *OffsetSection = nullptr,
                              bool UseRelativeOffsets = false);

  bool empty() const { return Pool.empty(); }

  unsigned size() const { return Pool.size(); }

  unsigned getNumIndexedStrings() const { return NumIndexedStrings; }

  /// Get a reference to an entry in the string pool.
  LLVM_ABI_FOR_TEST EntryRef getEntry(AsmPrinter &Asm, StringRef Str);

  /// Same as getEntry, except that you can use EntryRef::getIndex to obtain a
  /// unique ID of this entry (e.g., for use in indexed forms like
  /// DW_FORM_strx).
  LLVM_ABI_FOR_TEST EntryRef getIndexedEntry(AsmPrinter &Asm, StringRef Str);
};

} // end namespace llvm

#endif // LLVM_LIB_CODEGEN_ASMPRINTER_DWARFSTRINGPOOL_H
