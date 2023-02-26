//===- DWARFLinkerCompileUnit.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DWARFLinker/DWARFLinkerCompileUnit.h"
#include "llvm/DWARFLinker/DWARFLinkerDeclContext.h"
#include "llvm/Support/FormatVariadic.h"

namespace llvm {

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void CompileUnit::DIEInfo::dump() {
  llvm::errs() << "{\n";
  llvm::errs() << "  AddrAdjust: " << AddrAdjust << '\n';
  llvm::errs() << "  Ctxt: " << formatv("{0:x}", Ctxt) << '\n';
  llvm::errs() << "  Clone: " << formatv("{0:x}", Clone) << '\n';
  llvm::errs() << "  ParentIdx: " << ParentIdx << '\n';
  llvm::errs() << "  Keep: " << Keep << '\n';
  llvm::errs() << "  InDebugMap: " << InDebugMap << '\n';
  llvm::errs() << "  Prune: " << Prune << '\n';
  llvm::errs() << "  Incomplete: " << Incomplete << '\n';
  llvm::errs() << "  InModuleScope: " << InModuleScope << '\n';
  llvm::errs() << "  ODRMarkingDone: " << ODRMarkingDone << '\n';
  llvm::errs() << "  UnclonedReference: " << UnclonedReference << '\n';
  llvm::errs() << "}\n";
}
#endif // if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)

/// Check if the DIE at \p Idx is in the scope of a function.
static bool inFunctionScope(CompileUnit &U, unsigned Idx) {
  while (Idx) {
    if (U.getOrigUnit().getDIEAtIndex(Idx).getTag() == dwarf::DW_TAG_subprogram)
      return true;
    Idx = U.getInfo(Idx).ParentIdx;
  }
  return false;
}

uint16_t CompileUnit::getLanguage() {
  if (!Language) {
    DWARFDie CU = getOrigUnit().getUnitDIE();
    Language = dwarf::toUnsigned(CU.find(dwarf::DW_AT_language), 0);
  }
  return Language;
}

StringRef CompileUnit::getSysRoot() {
  if (SysRoot.empty()) {
    DWARFDie CU = getOrigUnit().getUnitDIE();
    SysRoot = dwarf::toStringRef(CU.find(dwarf::DW_AT_LLVM_sysroot)).str();
  }
  return SysRoot;
}

void CompileUnit::markEverythingAsKept() {
  unsigned Idx = 0;

  for (auto &I : Info) {
    // Mark everything that wasn't explicit marked for pruning.
    I.Keep = !I.Prune;
    auto DIE = OrigUnit.getDIEAtIndex(Idx++);

    // Try to guess which DIEs must go to the accelerator tables. We do that
    // just for variables, because functions will be handled depending on
    // whether they carry a DW_AT_low_pc attribute or not.
    if (DIE.getTag() != dwarf::DW_TAG_variable &&
        DIE.getTag() != dwarf::DW_TAG_constant)
      continue;

    std::optional<DWARFFormValue> Value;
    if (!(Value = DIE.find(dwarf::DW_AT_location))) {
      if ((Value = DIE.find(dwarf::DW_AT_const_value)) &&
          !inFunctionScope(*this, I.ParentIdx))
        I.InDebugMap = true;
      continue;
    }
    if (auto Block = Value->getAsBlock()) {
      if (Block->size() > OrigUnit.getAddressByteSize() &&
          (*Block)[0] == dwarf::DW_OP_addr)
        I.InDebugMap = true;
    }
  }
}

uint64_t CompileUnit::computeNextUnitOffset(uint16_t DwarfVersion) {
  NextUnitOffset = StartOffset;
  if (NewUnit) {
    NextUnitOffset += (DwarfVersion >= 5) ? 12 : 11; // Header size
    NextUnitOffset += NewUnit->getUnitDie().getSize();
  }
  return NextUnitOffset;
}

/// Keep track of a forward cross-cu reference from this unit
/// to \p Die that lives in \p RefUnit.
void CompileUnit::noteForwardReference(DIE *Die, const CompileUnit *RefUnit,
                                       DeclContext *Ctxt, PatchLocation Attr) {
  ForwardDIEReferences.emplace_back(Die, RefUnit, Ctxt, Attr);
}

void CompileUnit::fixupForwardReferences() {
  for (const auto &Ref : ForwardDIEReferences) {
    DIE *RefDie;
    const CompileUnit *RefUnit;
    PatchLocation Attr;
    DeclContext *Ctxt;
    std::tie(RefDie, RefUnit, Ctxt, Attr) = Ref;
    if (Ctxt && Ctxt->hasCanonicalDIE()) {
      assert(Ctxt->getCanonicalDIEOffset() &&
             "Canonical die offset is not set");
      Attr.set(Ctxt->getCanonicalDIEOffset());
    } else {
      assert(RefDie->getOffset() && "Referenced die offset is not set");
      Attr.set(RefDie->getOffset() + RefUnit->getStartOffset());
    }
  }
}

void CompileUnit::addLabelLowPc(uint64_t LabelLowPc, int64_t PcOffset) {
  Labels.insert({LabelLowPc, PcOffset});
}

void CompileUnit::addFunctionRange(uint64_t FuncLowPc, uint64_t FuncHighPc,
                                   int64_t PcOffset) {
  Ranges.insert({FuncLowPc, FuncHighPc}, PcOffset);
  if (LowPc)
    LowPc = std::min(*LowPc, FuncLowPc + PcOffset);
  else
    LowPc = FuncLowPc + PcOffset;
  this->HighPc = std::max(HighPc, FuncHighPc + PcOffset);
}

void CompileUnit::noteRangeAttribute(const DIE &Die, PatchLocation Attr) {
  if (Die.getTag() == dwarf::DW_TAG_compile_unit) {
    UnitRangeAttribute = Attr;
    return;
  }

  RangeAttributes.emplace_back(Attr);
}

void CompileUnit::noteLocationAttribute(PatchLocation Attr, int64_t PcOffset) {
  LocationAttributes.emplace_back(Attr, PcOffset);
}

void CompileUnit::addNamespaceAccelerator(const DIE *Die,
                                          DwarfStringPoolEntryRef Name) {
  Namespaces.emplace_back(Name, Die);
}

void CompileUnit::addObjCAccelerator(const DIE *Die,
                                     DwarfStringPoolEntryRef Name,
                                     bool SkipPubSection) {
  ObjC.emplace_back(Name, Die, SkipPubSection);
}

void CompileUnit::addNameAccelerator(const DIE *Die,
                                     DwarfStringPoolEntryRef Name,
                                     bool SkipPubSection) {
  Pubnames.emplace_back(Name, Die, SkipPubSection);
}

void CompileUnit::addTypeAccelerator(const DIE *Die,
                                     DwarfStringPoolEntryRef Name,
                                     bool ObjcClassImplementation,
                                     uint32_t QualifiedNameHash) {
  Pubtypes.emplace_back(Name, Die, QualifiedNameHash, ObjcClassImplementation);
}

} // namespace llvm
