//===- llvm/CodeGen/DwarfCompileUnit.h - Dwarf Compile Unit -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing dwarf compile unit.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_DWARFCOMPILEUNIT_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_DWARFCOMPILEUNIT_H

#include "DwarfDebug.h"
#include "DwarfUnit.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/CodeGen/DbgEntityHistoryCalculator.h"
#include "llvm/CodeGen/LexicalScopes.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>

namespace llvm {

class AsmPrinter;
class DIE;
class DIELoc;
class DIEValueList;
class DwarfFile;
class GlobalVariable;
class MCExpr;
class MCSymbol;
class MDNode;

enum class UnitKind { Skeleton, Full };

class DwarfCompileUnit final : public DwarfUnit {
  /// A numeric ID unique among all CUs in the module
  unsigned UniqueID;
  bool HasRangeLists = false;

  /// The start of the unit line section, this is also
  /// reused in appyStmtList.
  MCSymbol *LineTableStartSym;

  /// Skeleton unit associated with this unit.
  DwarfCompileUnit *Skeleton = nullptr;

  /// The start of the unit within its section.
  MCSymbol *LabelBegin;

  /// The start of the unit macro info within macro section.
  MCSymbol *MacroLabelBegin;

  using ImportedEntityList = SmallVector<const MDNode *, 8>;
  using ImportedEntityMap = DenseMap<const MDNode *, ImportedEntityList>;

  ImportedEntityMap ImportedEntities;

  /// GlobalNames - A map of globally visible named entities for this unit.
  StringMap<const DIE *> GlobalNames;

  /// GlobalTypes - A map of globally visible types for this unit.
  StringMap<const DIE *> GlobalTypes;

  // List of ranges for a given compile unit.
  SmallVector<RangeSpan, 2> CURanges;

  // The base address of this unit, if any. Used for relative references in
  // ranges/locs.
  const MCSymbol *BaseAddress = nullptr;

  DenseMap<const MDNode *, DIE *> AbstractSPDies;
  DenseMap<const DINode *, std::unique_ptr<DbgEntity>> AbstractEntities;

  /// DWO ID for correlating skeleton and split units.
  uint64_t DWOId = 0;

  /// Construct a DIE for the given DbgVariable without initializing the
  /// DbgVariable's DIE reference.
  DIE *constructVariableDIEImpl(const DbgVariable &DV, bool Abstract);

  bool isDwoUnit() const override;

  DenseMap<const MDNode *, DIE *> &getAbstractSPDies() {
    if (isDwoUnit() && !DD->shareAcrossDWOCUs())
      return AbstractSPDies;
    return DU->getAbstractSPDies();
  }

  DenseMap<const DINode *, std::unique_ptr<DbgEntity>> &getAbstractEntities() {
    if (isDwoUnit() && !DD->shareAcrossDWOCUs())
      return AbstractEntities;
    return DU->getAbstractEntities();
  }

  void finishNonUnitTypeDIE(DIE& D, const DICompositeType *CTy) override;

public:
  DwarfCompileUnit(unsigned UID, const DICompileUnit *Node, AsmPrinter *A,
                   DwarfDebug *DW, DwarfFile *DWU,
                   UnitKind Kind = UnitKind::Full);

  bool hasRangeLists() const { return HasRangeLists; }
  unsigned getUniqueID() const { return UniqueID; }

  DwarfCompileUnit *getSkeleton() const {
    return Skeleton;
  }

  bool includeMinimalInlineScopes() const;

  void initStmtList();

  /// Apply the DW_AT_stmt_list from this compile unit to the specified DIE.
  void applyStmtList(DIE &D);

  /// Get line table start symbol for this unit.
  MCSymbol *getLineTableStartSym() const { return LineTableStartSym; }

  /// A pair of GlobalVariable and DIExpression.
  struct GlobalExpr {
    const GlobalVariable *Var;
    const DIExpression *Expr;
  };

  struct BaseTypeRef {
    BaseTypeRef(unsigned BitSize, dwarf::TypeKind Encoding) :
      BitSize(BitSize), Encoding(Encoding) {}
    unsigned BitSize;
    dwarf::TypeKind Encoding;
    DIE *Die = nullptr;
  };

  std::vector<BaseTypeRef> ExprRefedBaseTypes;

  /// Get or create global variable DIE.
  DIE *
  getOrCreateGlobalVariableDIE(const DIGlobalVariable *GV,
                               ArrayRef<GlobalExpr> GlobalExprs);

  DIE *getOrCreateCommonBlock(const DICommonBlock *CB,
                              ArrayRef<GlobalExpr> GlobalExprs);

  void addLocationAttribute(DIE *ToDIE, const DIGlobalVariable *GV,
                            ArrayRef<GlobalExpr> GlobalExprs);

  /// addLabelAddress - Add a dwarf label attribute data and value using
  /// either DW_FORM_addr or DW_FORM_GNU_addr_index.
  void addLabelAddress(DIE &Die, dwarf::Attribute Attribute,
                       const MCSymbol *Label);

  /// addLocalLabelAddress - Add a dwarf label attribute data and value using
  /// DW_FORM_addr only.
  void addLocalLabelAddress(DIE &Die, dwarf::Attribute Attribute,
                            const MCSymbol *Label);

  DwarfCompileUnit &getCU() override { return *this; }

  unsigned getOrCreateSourceID(const DIFile *File) override;

  void addImportedEntity(const DIImportedEntity* IE) {
    DIScope *Scope = IE->getScope();
    assert(Scope && "Invalid Scope encoding!");
    if (!isa<DILocalScope>(Scope))
      // No need to add imported enities that are not local declaration.
      return;

    auto *LocalScope = cast<DILocalScope>(Scope)->getNonLexicalBlockFileScope();
    ImportedEntities[LocalScope].push_back(IE);
  }

  /// addRange - Add an address range to the list of ranges for this unit.
  void addRange(RangeSpan Range);

  void attachLowHighPC(DIE &D, const MCSymbol *Begin, const MCSymbol *End);

  /// Find DIE for the given subprogram and attach appropriate
  /// DW_AT_low_pc and DW_AT_high_pc attributes. If there are global
  /// variables in this scope then create and insert DIEs for these
  /// variables.
  DIE &updateSubprogramScopeDIE(const DISubprogram *SP);

  void constructScopeDIE(LexicalScope *Scope,
                         SmallVectorImpl<DIE *> &FinalChildren);

  /// A helper function to construct a RangeSpanList for a given
  /// lexical scope.
  void addScopeRangeList(DIE &ScopeDIE, SmallVector<RangeSpan, 2> Range);

  void attachRangesOrLowHighPC(DIE &D, SmallVector<RangeSpan, 2> Ranges);

  void attachRangesOrLowHighPC(DIE &D,
                               const SmallVectorImpl<InsnRange> &Ranges);

  /// This scope represents inlined body of a function. Construct
  /// DIE to represent this concrete inlined copy of the function.
  DIE *constructInlinedScopeDIE(LexicalScope *Scope);

  /// Construct new DW_TAG_lexical_block for this scope and
  /// attach DW_AT_low_pc/DW_AT_high_pc labels.
  DIE *constructLexicalScopeDIE(LexicalScope *Scope);

  /// constructVariableDIE - Construct a DIE for the given DbgVariable.
  DIE *constructVariableDIE(DbgVariable &DV, bool Abstract = false);

  DIE *constructVariableDIE(DbgVariable &DV, const LexicalScope &Scope,
                            DIE *&ObjectPointer);

  /// Construct a DIE for the given DbgLabel.
  DIE *constructLabelDIE(DbgLabel &DL, const LexicalScope &Scope);

  /// A helper function to create children of a Scope DIE.
  DIE *createScopeChildrenDIE(LexicalScope *Scope,
                              SmallVectorImpl<DIE *> &Children,
                              bool *HasNonScopeChildren = nullptr);

  void createBaseTypeDIEs();

  /// Construct a DIE for this subprogram scope.
  DIE &constructSubprogramScopeDIE(const DISubprogram *Sub,
                                   LexicalScope *Scope);

  DIE *createAndAddScopeChildren(LexicalScope *Scope, DIE &ScopeDIE);

  void constructAbstractSubprogramScopeDIE(LexicalScope *Scope);

  /// Whether to use the GNU analog for a DWARF5 tag, attribute, or location
  /// atom. Only applicable when emitting otherwise DWARF4-compliant debug info.
  bool useGNUAnalogForDwarf5Feature() const;

  /// This takes a DWARF 5 tag and returns it or a GNU analog.
  dwarf::Tag getDwarf5OrGNUTag(dwarf::Tag Tag) const;

  /// This takes a DWARF 5 attribute and returns it or a GNU analog.
  dwarf::Attribute getDwarf5OrGNUAttr(dwarf::Attribute Attr) const;

  /// This takes a DWARF 5 location atom and either returns it or a GNU analog.
  dwarf::LocationAtom getDwarf5OrGNULocationAtom(dwarf::LocationAtom Loc) const;

  /// Construct a call site entry DIE describing a call within \p Scope to a
  /// callee described by \p CalleeDIE.
  /// \p CalleeDIE is a declaration or definition subprogram DIE for the callee.
  /// For indirect calls \p CalleeDIE is set to nullptr.
  /// \p IsTail specifies whether the call is a tail call.
  /// \p PCAddr points to the PC value after the call instruction.
  /// \p CallAddr points to the PC value at the call instruction (or is null).
  /// \p CallReg is a register location for an indirect call. For direct calls
  /// the \p CallReg is set to 0.
  DIE &constructCallSiteEntryDIE(DIE &ScopeDIE, DIE *CalleeDIE, bool IsTail,
                                 const MCSymbol *PCAddr,
                                 const MCSymbol *CallAddr, unsigned CallReg);
  /// Construct call site parameter DIEs for the \p CallSiteDIE. The \p Params
  /// were collected by the \ref collectCallSiteParameters.
  /// Note: The order of parameters does not matter, since debuggers recognize
  ///       call site parameters by the DW_AT_location attribute.
  void constructCallSiteParmEntryDIEs(DIE &CallSiteDIE,
                                      SmallVector<DbgCallSiteParam, 4> &Params);

  /// Construct import_module DIE.
  DIE *constructImportedEntityDIE(const DIImportedEntity *Module);

  void finishSubprogramDefinition(const DISubprogram *SP);
  void finishEntityDefinition(const DbgEntity *Entity);

  /// Find abstract variable associated with Var.
  using InlinedEntity = DbgValueHistoryMap::InlinedEntity;
  DbgEntity *getExistingAbstractEntity(const DINode *Node);
  void createAbstractEntity(const DINode *Node, LexicalScope *Scope);

  /// Set the skeleton unit associated with this unit.
  void setSkeleton(DwarfCompileUnit &Skel) { Skeleton = &Skel; }

  unsigned getHeaderSize() const override {
    // DWARF v5 added the DWO ID to the header for split/skeleton units.
    unsigned DWOIdSize =
        DD->getDwarfVersion() >= 5 && DD->useSplitDwarf() ? sizeof(uint64_t)
                                                          : 0;
    return DwarfUnit::getHeaderSize() + DWOIdSize;
  }
  unsigned getLength() {
    return sizeof(uint32_t) + // Length field
        getHeaderSize() + getUnitDie().getSize();
  }

  void emitHeader(bool UseOffsets) override;

  /// Add the DW_AT_addr_base attribute to the unit DIE.
  void addAddrTableBase();

  MCSymbol *getLabelBegin() const {
    assert(getSection());
    return LabelBegin;
  }

  MCSymbol *getMacroLabelBegin() const {
    return MacroLabelBegin;
  }

  /// Add a new global name to the compile unit.
  void addGlobalName(StringRef Name, const DIE &Die,
                     const DIScope *Context) override;

  /// Add a new global name present in a type unit to this compile unit.
  void addGlobalNameForTypeUnit(StringRef Name, const DIScope *Context);

  /// Add a new global type to the compile unit.
  void addGlobalType(const DIType *Ty, const DIE &Die,
                     const DIScope *Context) override;

  /// Add a new global type present in a type unit to this compile unit.
  void addGlobalTypeUnitType(const DIType *Ty, const DIScope *Context);

  const StringMap<const DIE *> &getGlobalNames() const { return GlobalNames; }
  const StringMap<const DIE *> &getGlobalTypes() const { return GlobalTypes; }

  /// Add DW_AT_location attribute for a DbgVariable based on provided
  /// MachineLocation.
  void addVariableAddress(const DbgVariable &DV, DIE &Die,
                          MachineLocation Location);
  /// Add an address attribute to a die based on the location provided.
  void addAddress(DIE &Die, dwarf::Attribute Attribute,
                  const MachineLocation &Location);

  /// Start with the address based on the location provided, and generate the
  /// DWARF information necessary to find the actual variable (navigating the
  /// extra location information encoded in the type) based on the starting
  /// location.  Add the DWARF information to the die.
  void addComplexAddress(const DbgVariable &DV, DIE &Die,
                         dwarf::Attribute Attribute,
                         const MachineLocation &Location);

  /// Add a Dwarf loclistptr attribute data and value.
  void addLocationList(DIE &Die, dwarf::Attribute Attribute, unsigned Index);
  void applyVariableAttributes(const DbgVariable &Var, DIE &VariableDie);

  /// Add a Dwarf expression attribute data and value.
  void addExpr(DIELoc &Die, dwarf::Form Form, const MCExpr *Expr);

  void applySubprogramAttributesToDefinition(const DISubprogram *SP,
                                             DIE &SPDie);

  void applyLabelAttributes(const DbgLabel &Label, DIE &LabelDie);

  /// getRanges - Get the list of ranges for this unit.
  const SmallVectorImpl<RangeSpan> &getRanges() const { return CURanges; }
  SmallVector<RangeSpan, 2> takeRanges() { return std::move(CURanges); }

  void setBaseAddress(const MCSymbol *Base) { BaseAddress = Base; }
  const MCSymbol *getBaseAddress() const { return BaseAddress; }

  uint64_t getDWOId() const { return DWOId; }
  void setDWOId(uint64_t DwoId) { DWOId = DwoId; }

  bool hasDwarfPubSections() const;

  void addBaseTypeRef(DIEValueList &Die, int64_t Idx);
};

} // end namespace llvm

#endif // LLVM_LIB_CODEGEN_ASMPRINTER_DWARFCOMPILEUNIT_H
