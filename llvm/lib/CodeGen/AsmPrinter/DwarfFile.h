//===- llvm/CodeGen/DwarfFile.h - Dwarf Debug Framework ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_DWARFFILE_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_DWARFFILE_H

#include "DwarfStringPool.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/DIE.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Support/Allocator.h"
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <utility>

namespace llvm {

class AsmPrinter;
class DbgEntity;
class DbgVariable;
class DbgLabel;
class DwarfCompileUnit;
class DwarfUnit;
class LexicalScope;
class MCSection;
class MDNode;

// Data structure to hold a range for range lists.
struct RangeSpan {
  const MCSymbol *Begin;
  const MCSymbol *End;

  bool operator==(const RangeSpan &Other) const {
    return Begin == Other.Begin && End == Other.End;
  }
};

struct RangeSpanList {
  // Index for locating within the debug_range section this particular span.
  MCSymbol *Label;
  const DwarfCompileUnit *CU;
  // List of ranges.
  SmallVector<RangeSpan, 2> Ranges;
};

/// Tracks abstract and concrete DIEs for debug info entities of a certain type.
template <typename DINodeT, typename DbgEntityT> class DINodeInfoHolder {
public:
  using AbstractMapT = DenseMap<const DINodeT *, DIE *>;
  using ConcreteMapT =
      DenseMap<const DINodeT *, SmallDenseMap<const DbgEntityT *, DIE *, 2>>;

private:
  AbstractMapT AbstractMap;
  ConcreteMapT ConcreteMap;

public:
  void insertAbstractDIE(const DINodeT *N, DIE *D) {
    auto [_, Inserted] = AbstractMap.try_emplace(N, D);
    assert(Inserted && "Duplicate abstract DIE for debug info node");
  }

  void insertConcreteDIE(const DINodeT *N, const DbgEntityT *E, DIE *D) {
    auto [_, Inserted] = ConcreteMap[N].try_emplace(E, D);
    assert(Inserted && "Duplicate concrete DIE for debug info node");
  }

  void insertDIE(const DINodeT *N, const DbgEntityT *E, DIE *D, bool Abstract) {
    if (Abstract)
      insertAbstractDIE(N, D);
    else
      insertConcreteDIE(N, E, D);
  }

  DIE *getAbstractDIE(const DINodeT *N) const { return AbstractMap.lookup(N); }

  std::optional<
      std::reference_wrapper<const typename ConcreteMapT::mapped_type>>
  getConcreteDIEs(const DINodeT *N) const {
    if (auto I = ConcreteMap.find(N); I != ConcreteMap.end())
      return std::make_optional(std::ref(I->second));
    return std::nullopt;
  }

  DIE *getConcreteDIE(const DINodeT *N, const DbgEntityT *E) const {
    if (auto I = getConcreteDIEs(N))
      return I->get().lookup(E);
    return nullptr;
  }

  DIE *getAnyConcreteDIE(const DINodeT *N) const {
    if (auto I = getConcreteDIEs(N))
      return I->get().empty() ? nullptr : I->get().begin()->second;
    return nullptr;
  }

  /// Returns abstract DIE for the entity.
  /// If no abstract DIE was created, returns any concrete DIE for the entity.
  DIE *getDIE(const DINodeT *N) const {
    if (DIE *D = getAbstractDIE(N))
      return D;

    return getAnyConcreteDIE(N);
  }

  AbstractMapT &getAbstractDIEs() { return AbstractMap; }
};

/// Tracks DIEs for debug info entites.
/// These DIEs can be shared across CUs, that is why we keep the map here
/// instead of in DwarfCompileUnit.
class DwarfInfoHolder {
  /// DIEs of local DbgVariables.
  DINodeInfoHolder<DILocalVariable, DbgVariable> LVHolder;
  /// DIEs of labels.
  DINodeInfoHolder<DILabel, DbgLabel> LabelHolder;
  DenseMap<const DINode *, std::unique_ptr<DbgEntity>> AbstractEntities;
  // List of abstract local scopes (either DISubprogram or DILexicalBlock).
  DenseMap<const DILocalScope *, DIE *> AbstractLocalScopeDIEs;
  /// Keeps track of abstract subprograms to populate them only once.
  // FIXME: merge creation and population of abstract scopes.
  SmallPtrSet<const DISubprogram *, 8> FinalizedAbstractSubprograms;

  /// Other DINodes with the corresponding DIEs.
  DenseMap<const DINode *, DIE *> MDNodeToDieMap;

public:
  void insertDIE(const DINode *N, DIE *Die) {
    assert((!isa<DILabel>(N) && !isa<DILocalVariable>(N)) &&
           "Use getLabels().insertDIE() for labels or getLVs().insertDIE() for "
           "local variables");
    auto [_, Inserted] = MDNodeToDieMap.try_emplace(N, Die);
    assert((Inserted || isa<DISubprogram>(N) || isa<DIType>(N)) &&
           "DIE for this DINode has already been added");
  }

  void insertDIE(DIE *D) { MDNodeToDieMap.try_emplace(nullptr, D); }

  DIE *getDIE(const DINode *N) const {
    DIE *D = MDNodeToDieMap.lookup(N);
    assert((!D || (!isa<DILabel>(N) && !isa<DILocalVariable>(N))) &&
           "Use getLabels().getDIE() for labels or getLVs().getDIE() for "
           "local variables");
    return D;
  }

  auto &getLVs() { return LVHolder; }
  auto &getLVs() const { return LVHolder; }

  auto &getLabels() { return LabelHolder; }
  auto &getLabels() const { return LabelHolder; }

  /// For a global variable, returns DIE of the variable.
  ///
  /// For a local variable, returns abstract DIE of the variable.
  /// If no abstract DIE was created, returns any concrete DIE of the variable.
  DIE *getVariableDIE(const DIVariable *V) const {
    if (auto *LV = dyn_cast<DILocalVariable>(V))
      if (DIE *D = getLVs().getDIE(LV))
        return D;
    return getDIE(V);
  }

  DenseMap<const DILocalScope *, DIE *> &getAbstractScopeDIEs() {
    return AbstractLocalScopeDIEs;
  }

  DenseMap<const DINode *, std::unique_ptr<DbgEntity>> &getAbstractEntities() {
    return AbstractEntities;
  }

  auto &getFinalizedAbstractSubprograms() {
    return FinalizedAbstractSubprograms;
  }
};

class DwarfFile {
  // Target of Dwarf emission, used for sizing of abbreviations.
  AsmPrinter *Asm;

  BumpPtrAllocator AbbrevAllocator;

  // Used to uniquely define abbreviations.
  DIEAbbrevSet Abbrevs;

  // A pointer to all units in the section.
  SmallVector<std::unique_ptr<DwarfCompileUnit>, 1> CUs;

  DwarfStringPool StrPool;

  // List of range lists for a given compile unit, separate from the ranges for
  // the CU itself.
  SmallVector<RangeSpanList, 1> CURangeLists;

  /// DWARF v5: The symbol that designates the start of the contribution to
  /// the string offsets table. The contribution is shared by all units.
  MCSymbol *StringOffsetsStartSym = nullptr;

  /// DWARF v5: The symbol that designates the base of the range list table.
  /// The table is shared by all units.
  MCSymbol *RnglistsTableBaseSym = nullptr;

  /// The variables of a lexical scope.
  struct ScopeVars {
    /// We need to sort Args by ArgNo and check for duplicates. This could also
    /// be implemented as a list or vector + std::lower_bound().
    std::map<unsigned, DbgVariable *> Args;
    SmallVector<DbgVariable *, 8> Locals;
  };
  /// Collection of DbgVariables of each lexical scope.
  DenseMap<LexicalScope *, ScopeVars> ScopeVariables;

  /// Collection of DbgLabels of each lexical scope.
  using LabelList = SmallVector<DbgLabel *, 4>;
  DenseMap<LexicalScope *, LabelList> ScopeLabels;

  DwarfInfoHolder InfoHolder;

public:
  DwarfFile(AsmPrinter *AP, StringRef Pref, BumpPtrAllocator &DA);

  const SmallVectorImpl<std::unique_ptr<DwarfCompileUnit>> &getUnits() {
    return CUs;
  }

  std::pair<uint32_t, RangeSpanList *> addRange(const DwarfCompileUnit &CU,
                                                SmallVector<RangeSpan, 2> R);

  /// getRangeLists - Get the vector of range lists.
  const SmallVectorImpl<RangeSpanList> &getRangeLists() const {
    return CURangeLists;
  }

  /// Compute the size and offset of a DIE given an incoming Offset.
  unsigned computeSizeAndOffset(DIE &Die, unsigned Offset);

  /// Compute the size and offset of all the DIEs.
  void computeSizeAndOffsets();

  /// Compute the size and offset of all the DIEs in the given unit.
  /// \returns The size of the root DIE.
  unsigned computeSizeAndOffsetsForUnit(DwarfUnit *TheU);

  /// Add a unit to the list of CUs.
  void addUnit(std::unique_ptr<DwarfCompileUnit> U);

  /// Emit all of the units to the section listed with the given
  /// abbreviation section.
  void emitUnits(bool UseOffsets);

  /// Emit the given unit to its section.
  void emitUnit(DwarfUnit *TheU, bool UseOffsets);

  /// Emit a set of abbreviations to the specific section.
  void emitAbbrevs(MCSection *);

  /// Emit all of the strings to the section given. If OffsetSection is
  /// non-null, emit a table of string offsets to it. If UseRelativeOffsets
  /// is false, emit absolute offsets to the strings. Otherwise, emit
  /// relocatable references to the strings if they are supported by the target.
  void emitStrings(MCSection *StrSection, MCSection *OffsetSection = nullptr,
                   bool UseRelativeOffsets = false);

  /// Returns the string pool.
  DwarfStringPool &getStringPool() { return StrPool; }

  MCSymbol *getStringOffsetsStartSym() const { return StringOffsetsStartSym; }
  void setStringOffsetsStartSym(MCSymbol *Sym) { StringOffsetsStartSym = Sym; }

  MCSymbol *getRnglistsTableBaseSym() const { return RnglistsTableBaseSym; }
  void setRnglistsTableBaseSym(MCSymbol *Sym) { RnglistsTableBaseSym = Sym; }

  void addScopeVariable(LexicalScope *LS, DbgVariable *Var);

  void addScopeLabel(LexicalScope *LS, DbgLabel *Label);

  DenseMap<LexicalScope *, ScopeVars> &getScopeVariables() {
    return ScopeVariables;
  }

  DenseMap<LexicalScope *, LabelList> &getScopeLabels() {
    return ScopeLabels;
  }

  DwarfInfoHolder &getDIEs() { return InfoHolder; }
};

} // end namespace llvm

#endif // LLVM_LIB_CODEGEN_ASMPRINTER_DWARFFILE_H
