//===- llvm/CodeGen/DwarfFile.h - Dwarf Debug Framework ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_DWARFFILE_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_DWARFFILE_H

#include "DwarfStringPool.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/DIE.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Allocator.h"
#include <map>
#include <memory>
#include <utility>

namespace llvm {

class AsmPrinter;
class DbgVariable;
class DwarfCompileUnit;
class DwarfUnit;
class LexicalScope;
class MCSection;

class DwarfFile {
  // Target of Dwarf emission, used for sizing of abbreviations.
  AsmPrinter *Asm;

  BumpPtrAllocator AbbrevAllocator;

  // Used to uniquely define abbreviations.
  DIEAbbrevSet Abbrevs;

  // A pointer to all units in the section.
  SmallVector<std::unique_ptr<DwarfCompileUnit>, 1> CUs;

  DwarfStringPool StrPool;

  /// DWARF v5: The symbol that designates the start of the contribution to
  /// the string offsets table. The contribution is shared by all units.
  MCSymbol *StringOffsetsStartSym = nullptr;

  /// The variables of a lexical scope.
  struct ScopeVars {
    /// We need to sort Args by ArgNo and check for duplicates. This could also
    /// be implemented as a list or vector + std::lower_bound().
    std::map<unsigned, DbgVariable *> Args;
    SmallVector<DbgVariable *, 8> Locals;
  };
  /// Collection of DbgVariables of each lexical scope.
  DenseMap<LexicalScope *, ScopeVars> ScopeVariables;

  // Collection of abstract subprogram DIEs.
  DenseMap<const MDNode *, DIE *> AbstractSPDies;
  DenseMap<const MDNode *, std::unique_ptr<DbgVariable>> AbstractVariables;

  /// Maps MDNodes for type system with the corresponding DIEs. These DIEs can
  /// be shared across CUs, that is why we keep the map here instead
  /// of in DwarfCompileUnit.
  DenseMap<const MDNode *, DIE *> DITypeNodeToDieMap;

public:
  DwarfFile(AsmPrinter *AP, StringRef Pref, BumpPtrAllocator &DA);

  const SmallVectorImpl<std::unique_ptr<DwarfCompileUnit>> &getUnits() {
    return CUs;
  }

  /// \brief Compute the size and offset of a DIE given an incoming Offset.
  unsigned computeSizeAndOffset(DIE &Die, unsigned Offset);

  /// \brief Compute the size and offset of all the DIEs.
  void computeSizeAndOffsets();

  /// \brief Compute the size and offset of all the DIEs in the given unit.
  /// \returns The size of the root DIE.
  unsigned computeSizeAndOffsetsForUnit(DwarfUnit *TheU);

  /// \brief Add a unit to the list of CUs.
  void addUnit(std::unique_ptr<DwarfCompileUnit> U);

  /// Emit the string table offsets header.
  void emitStringOffsetsTableHeader(MCSection *Section);

  /// \brief Emit all of the units to the section listed with the given
  /// abbreviation section.
  void emitUnits(bool UseOffsets);

  /// \brief Emit the given unit to its section.
  void emitUnit(DwarfUnit *U, bool UseOffsets);

  /// \brief Emit a set of abbreviations to the specific section.
  void emitAbbrevs(MCSection *);

  /// Emit all of the strings to the section given. If OffsetSection is
  /// non-null, emit a table of string offsets to it. If UseRelativeOffsets
  /// is false, emit absolute offsets to the strings. Otherwise, emit
  /// relocatable references to the strings if they are supported by the target.
  void emitStrings(MCSection *StrSection, MCSection *OffsetSection = nullptr,
                   bool UseRelativeOffsets = false);

  /// \brief Returns the string pool.
  DwarfStringPool &getStringPool() { return StrPool; }

  MCSymbol *getStringOffsetsStartSym() const { return StringOffsetsStartSym; }

  void setStringOffsetsStartSym(MCSymbol *Sym) { StringOffsetsStartSym = Sym; }

  /// \returns false if the variable was merged with a previous one.
  bool addScopeVariable(LexicalScope *LS, DbgVariable *Var);

  DenseMap<LexicalScope *, ScopeVars> &getScopeVariables() {
    return ScopeVariables;
  }

  DenseMap<const MDNode *, DIE *> &getAbstractSPDies() {
    return AbstractSPDies;
  }

  DenseMap<const MDNode *, std::unique_ptr<DbgVariable>> &getAbstractVariables() {
    return AbstractVariables;
  }

  void insertDIE(const MDNode *TypeMD, DIE *Die) {
    DITypeNodeToDieMap.insert(std::make_pair(TypeMD, Die));
  }

  DIE *getDIE(const MDNode *TypeMD) {
    return DITypeNodeToDieMap.lookup(TypeMD);
  }
};

} // end namespace llvm

#endif // LLVM_LIB_CODEGEN_ASMPRINTER_DWARFFILE_H
