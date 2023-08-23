//===- DIEAttributeCloner.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DWARFLINKERPARALLEL_DIEATTRIBUTECLONER_H
#define LLVM_LIB_DWARFLINKERPARALLEL_DIEATTRIBUTECLONER_H

#include "ArrayList.h"
#include "DIEGenerator.h"
#include "DWARFLinkerCompileUnit.h"
#include "DWARFLinkerGlobalData.h"

namespace llvm {
namespace dwarflinker_parallel {

/// This class creates clones of input DIE attributes.
/// It enumerates attributes of input DIE, creates clone for each
/// attribute, adds cloned attribute to the output DIE.
class DIEAttributeCloner {
public:
  DIEAttributeCloner(DIE *OutDIE, CompileUnit &CU,
                     const DWARFDebugInfoEntry *InputDieEntry,
                     DIEGenerator &Generator,
                     std::optional<int64_t> FuncAddressAdjustment,
                     std::optional<int64_t> VarAddressAdjustment,
                     bool HasLocationExpressionAddress)
      : OutDIE(OutDIE), CU(CU),
        DebugInfoOutputSection(
            CU.getOrCreateSectionDescriptor(DebugSectionKind::DebugInfo)),
        InputDieEntry(InputDieEntry), Generator(Generator),
        FuncAddressAdjustment(FuncAddressAdjustment),
        VarAddressAdjustment(VarAddressAdjustment),
        HasLocationExpressionAddress(HasLocationExpressionAddress) {
    InputDIEIdx = CU.getDIEIndex(InputDieEntry);
  }

  /// Clone attributes of input DIE.
  void clone();

  /// Create abbreviations for the output DIE after all attributes are cloned.
  unsigned finalizeAbbreviations(bool HasChildrenToClone);

  /// Information gathered and exchanged between the various
  /// clone*Attr helpers about the attributes of a particular DIE.
  ///
  /// Cannot be used concurrently.
  struct AttributesInfo {
    /// Names.
    StringEntry *Name = nullptr;
    StringEntry *MangledName = nullptr;
    StringEntry *NameWithoutTemplate = nullptr;

    /// Does the DIE have a low_pc attribute?
    bool HasLowPc = false;

    /// Is this DIE only a declaration?
    bool IsDeclaration = false;

    /// Does the DIE have a ranges attribute?
    bool HasRanges = false;

    /// Does the DIE have a string offset attribute?
    bool HasStringOffsetBaseAttr = false;
  };

  AttributesInfo AttrInfo;

protected:
  /// Clone string attribute.
  size_t
  cloneStringAttr(const DWARFFormValue &Val,
                  const DWARFAbbreviationDeclaration::AttributeSpec &AttrSpec);

  /// Clone attribute referencing another DIE.
  size_t
  cloneDieRefAttr(const DWARFFormValue &Val,
                  const DWARFAbbreviationDeclaration::AttributeSpec &AttrSpec);

  /// Clone scalar attribute.
  size_t
  cloneScalarAttr(const DWARFFormValue &Val,
                  const DWARFAbbreviationDeclaration::AttributeSpec &AttrSpec);

  /// Clone block or exprloc attribute.
  size_t
  cloneBlockAttr(const DWARFFormValue &Val,
                 const DWARFAbbreviationDeclaration::AttributeSpec &AttrSpec);

  /// Clone address attribute.
  size_t
  cloneAddressAttr(const DWARFFormValue &Val,
                   const DWARFAbbreviationDeclaration::AttributeSpec &AttrSpec);

  /// Returns true if attribute should be skipped.
  bool
  shouldSkipAttribute(DWARFAbbreviationDeclaration::AttributeSpec AttrSpec);

  /// Update patches offsets with the size of abbreviation number.
  void
  updatePatchesWithSizeOfAbbreviationNumber(unsigned SizeOfAbbreviationNumber) {
    for (uint64_t *OffsetPtr : PatchesOffsets)
      *OffsetPtr += SizeOfAbbreviationNumber;
  }

  /// Output DIE.
  DIE *OutDIE = nullptr;

  /// Compile unit for the output DIE.
  CompileUnit &CU;

  /// .debug_info section descriptor.
  SectionDescriptor &DebugInfoOutputSection;

  /// Input DIE entry.
  const DWARFDebugInfoEntry *InputDieEntry = nullptr;

  /// Input DIE index.
  uint32_t InputDIEIdx = 0;

  /// Output DIE generator.
  DIEGenerator &Generator;

  /// Relocation adjustment for the function address ranges.
  std::optional<int64_t> FuncAddressAdjustment;

  /// Relocation adjustment for the variable locations.
  std::optional<int64_t> VarAddressAdjustment;

  /// Indicates whether InputDieEntry has an location attribute
  /// containg address expression.
  bool HasLocationExpressionAddress = false;

  /// Output offset after all attributes.
  unsigned AttrOutOffset = 0;

  /// Patches for the cloned attributes.
  OffsetsPtrVector PatchesOffsets;
};

} // end of namespace dwarflinker_parallel
} // end namespace llvm

#endif // LLVM_LIB_DWARFLINKERPARALLEL_DIEATTRIBUTECLONER_H
