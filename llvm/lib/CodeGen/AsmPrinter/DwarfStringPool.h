//===- llvm/CodeGen/DwarfStringPool.h - Dwarf Debug Framework ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_DWARFSTRINGPOOL_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_DWARFSTRINGPOOL_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/DwarfStringPoolEntry.h"
#include "llvm/Support/Allocator.h"

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
  unsigned NumBytes = 0;
  bool ShouldCreateSymbols;

public:
  using EntryRef = DwarfStringPoolEntryRef;

  DwarfStringPool(BumpPtrAllocator &A, AsmPrinter &Asm, StringRef Prefix);

  void emitStringOffsetsTableHeader(AsmPrinter &Asm, MCSection *OffsetSection,
                                    MCSymbol *StartSym);

  void emit(AsmPrinter &Asm, MCSection *StrSection,
            MCSection *OffsetSection = nullptr,
            bool UseRelativeOffsets = false);

  bool empty() const { return Pool.empty(); }

  unsigned size() const { return Pool.size(); }

  /// Get a reference to an entry in the string pool.
  EntryRef getEntry(AsmPrinter &Asm, StringRef Str);
};

} // end namespace llvm

#endif // LLVM_LIB_CODEGEN_ASMPRINTER_DWARFSTRINGPOOL_H
