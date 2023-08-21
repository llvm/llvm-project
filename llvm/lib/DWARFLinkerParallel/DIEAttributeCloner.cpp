//=== DIEAttributeCloner.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DIEAttributeCloner.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugMacro.h"

namespace llvm {
namespace dwarflinker_parallel {

void DIEAttributeCloner::clone() {
  DWARFUnit &U = CU.getOrigUnit();

  // Extract and clone every attribute.
  DWARFDataExtractor Data = U.getDebugInfoExtractor();

  uint64_t Offset = InputDieEntry->getOffset();
  // Point to the next DIE (generally there is always at least a NULL
  // entry after the current one). If this is a lone
  // DW_TAG_compile_unit without any children, point to the next unit.
  uint64_t NextOffset = (InputDIEIdx + 1 < U.getNumDIEs())
                            ? U.getDIEAtIndex(InputDIEIdx + 1).getOffset()
                            : U.getNextUnitOffset();

  // We could copy the data only if we need to apply a relocation to it. After
  // testing, it seems there is no performance downside to doing the copy
  // unconditionally, and it makes the code simpler.
  SmallString<40> DIECopy(Data.getData().substr(Offset, NextOffset - Offset));
  Data =
      DWARFDataExtractor(DIECopy, Data.isLittleEndian(), Data.getAddressSize());

  // Modify the copy with relocated addresses.
  CU.getContaingFile().Addresses->applyValidRelocs(DIECopy, Offset,
                                                   Data.isLittleEndian());

  // Reset the Offset to 0 as we will be working on the local copy of
  // the data.
  Offset = 0;

  const auto *Abbrev = InputDieEntry->getAbbreviationDeclarationPtr();
  Offset += getULEB128Size(Abbrev->getCode());

  // Set current output offset.
  AttrOutOffset = OutDIE->getOffset();
  for (const auto &AttrSpec : Abbrev->attributes()) {
    // Check whether current attribute should be skipped.
    if (shouldSkipAttribute(AttrSpec)) {
      DWARFFormValue::skipValue(AttrSpec.Form, Data, &Offset,
                                U.getFormParams());
      continue;
    }

    DWARFFormValue Val = AttrSpec.getFormValue();
    Val.extractValue(Data, &Offset, U.getFormParams(), &U);

    // Clone current attribute.
    switch (AttrSpec.Form) {
    case dwarf::DW_FORM_strp:
    case dwarf::DW_FORM_line_strp:
    case dwarf::DW_FORM_string:
    case dwarf::DW_FORM_strx:
    case dwarf::DW_FORM_strx1:
    case dwarf::DW_FORM_strx2:
    case dwarf::DW_FORM_strx3:
    case dwarf::DW_FORM_strx4:
      AttrOutOffset += cloneStringAttr(Val, AttrSpec);
      break;
    case dwarf::DW_FORM_ref_addr:
    case dwarf::DW_FORM_ref1:
    case dwarf::DW_FORM_ref2:
    case dwarf::DW_FORM_ref4:
    case dwarf::DW_FORM_ref8:
    case dwarf::DW_FORM_ref_udata:
      AttrOutOffset += cloneDieRefAttr(Val, AttrSpec);
      break;
    case dwarf::DW_FORM_data1:
    case dwarf::DW_FORM_data2:
    case dwarf::DW_FORM_data4:
    case dwarf::DW_FORM_data8:
    case dwarf::DW_FORM_udata:
    case dwarf::DW_FORM_sdata:
    case dwarf::DW_FORM_sec_offset:
    case dwarf::DW_FORM_flag:
    case dwarf::DW_FORM_flag_present:
    case dwarf::DW_FORM_rnglistx:
    case dwarf::DW_FORM_loclistx:
    case dwarf::DW_FORM_implicit_const:
      AttrOutOffset += cloneScalarAttr(Val, AttrSpec);
      break;
    case dwarf::DW_FORM_block:
    case dwarf::DW_FORM_block1:
    case dwarf::DW_FORM_block2:
    case dwarf::DW_FORM_block4:
    case dwarf::DW_FORM_exprloc:
      AttrOutOffset += cloneBlockAttr(Val, AttrSpec);
      break;
    case dwarf::DW_FORM_addr:
    case dwarf::DW_FORM_addrx:
    case dwarf::DW_FORM_addrx1:
    case dwarf::DW_FORM_addrx2:
    case dwarf::DW_FORM_addrx3:
    case dwarf::DW_FORM_addrx4:
      AttrOutOffset += cloneAddressAttr(Val, AttrSpec);
      break;
    default:
      CU.warn("unsupported attribute form " +
                  dwarf::FormEncodingString(AttrSpec.Form) +
                  " in DieAttributeCloner::clone(). Dropping.",
              InputDieEntry);
    }
  }

  // We convert source strings into the indexed form for DWARFv5.
  // Check if original compile unit already has DW_AT_str_offsets_base
  // attribute.
  if (InputDieEntry->getTag() == dwarf::DW_TAG_compile_unit &&
      CU.getVersion() >= 5 && !AttrInfo.HasStringOffsetBaseAttr) {
    DebugInfoOutputSection.notePatchWithOffsetUpdate(
        DebugOffsetPatch{
            AttrOutOffset,
            &CU.getOrCreateSectionDescriptor(DebugSectionKind::DebugStrOffsets),
            true},
        PatchesOffsets);

    AttrOutOffset += Generator
                         .addScalarAttribute(dwarf::DW_AT_str_offsets_base,
                                             dwarf::DW_FORM_sec_offset,
                                             CU.getDebugStrOffsetsHeaderSize())
                         .second;
  }
}

bool DIEAttributeCloner::shouldSkipAttribute(
    DWARFAbbreviationDeclaration::AttributeSpec AttrSpec) {
  switch (AttrSpec.Attr) {
  default:
    return false;
  case dwarf::DW_AT_low_pc:
  case dwarf::DW_AT_high_pc:
  case dwarf::DW_AT_ranges:
    if (CU.getGlobalData().getOptions().UpdateIndexTablesOnly)
      return false;

    // Skip address attribute if we are in function scope and function does not
    // reference live address.
    return CU.getDIEInfo(InputDIEIdx).getIsInFunctionScope() &&
           !FuncAddressAdjustment.has_value();
  case dwarf::DW_AT_rnglists_base:
    // In case !Update the .debug_addr table is not generated/preserved.
    // Thus instead of DW_FORM_rnglistx the DW_FORM_sec_offset is used.
    // Since DW_AT_rnglists_base is used for only DW_FORM_rnglistx the
    // DW_AT_rnglists_base is removed.
    return !CU.getGlobalData().getOptions().UpdateIndexTablesOnly;
  case dwarf::DW_AT_loclists_base:
    // In case !Update the .debug_addr table is not generated/preserved.
    // Thus instead of DW_FORM_loclistx the DW_FORM_sec_offset is used.
    // Since DW_AT_loclists_base is used for only DW_FORM_loclistx the
    // DW_AT_loclists_base is removed.
    return !CU.getGlobalData().getOptions().UpdateIndexTablesOnly;
  case dwarf::DW_AT_location:
  case dwarf::DW_AT_frame_base:
    if (CU.getGlobalData().getOptions().UpdateIndexTablesOnly)
      return false;

    // When location expression contains an address: skip this attribute
    // if it does not reference live address.
    if (HasLocationExpressionAddress)
      return !VarAddressAdjustment.has_value();

    // Skip location attribute if we are in function scope and function does not
    // reference live address.
    return CU.getDIEInfo(InputDIEIdx).getIsInFunctionScope() &&
           !FuncAddressAdjustment.has_value();
  }
}

size_t DIEAttributeCloner::cloneStringAttr(
    const DWARFFormValue &Val,
    const DWARFAbbreviationDeclaration::AttributeSpec &AttrSpec) {
  std::optional<const char *> String = dwarf::toString(Val);
  if (!String) {
    CU.warn("cann't read string attribute.");
    return 0;
  }

  StringEntry *StringInPool =
      CU.getGlobalData().getStringPool().insert(*String).first;
  if (AttrSpec.Form == dwarf::DW_FORM_line_strp) {
    DebugInfoOutputSection.notePatchWithOffsetUpdate(
        DebugLineStrPatch{{AttrOutOffset}, StringInPool}, PatchesOffsets);
    return Generator
        .addStringPlaceholderAttribute(AttrSpec.Attr, dwarf::DW_FORM_line_strp)
        .second;
  }

  // Update attributes info.
  if (AttrSpec.Attr == dwarf::DW_AT_name)
    AttrInfo.Name = StringInPool;
  else if (AttrSpec.Attr == dwarf::DW_AT_MIPS_linkage_name ||
           AttrSpec.Attr == dwarf::DW_AT_linkage_name)
    AttrInfo.MangledName = StringInPool;

  if (CU.getVersion() < 5) {
    DebugInfoOutputSection.notePatchWithOffsetUpdate(
        DebugStrPatch{{AttrOutOffset}, StringInPool}, PatchesOffsets);

    return Generator
        .addStringPlaceholderAttribute(AttrSpec.Attr, dwarf::DW_FORM_strp)
        .second;
  }

  return Generator
      .addIndexedStringAttribute(AttrSpec.Attr, dwarf::DW_FORM_strx,
                                 CU.getDebugStrIndex(StringInPool))
      .second;
}

size_t DIEAttributeCloner::cloneDieRefAttr(
    const DWARFFormValue &Val,
    const DWARFAbbreviationDeclaration::AttributeSpec &AttrSpec) {
  if (AttrSpec.Attr == dwarf::DW_AT_sibling)
    return 0;

  std::optional<std::pair<CompileUnit *, uint32_t>> RefDiePair =
      CU.resolveDIEReference(Val);
  if (!RefDiePair) {
    // If the referenced DIE is not found,  drop the attribute.
    CU.warn("cann't find referenced DIE.", InputDieEntry);
    return 0;
  }
  assert(RefDiePair->first->getStage() >= CompileUnit::Stage::Loaded);
  assert(RefDiePair->second != 0);

  // Get output offset for referenced DIE.
  uint64_t OutDieOffset =
      RefDiePair->first->getDieOutOffset(RefDiePair->second);

  // Examine whether referenced DIE is in current compile unit.
  bool IsLocal = CU.getUniqueID() == RefDiePair->first->getUniqueID();

  // Set attribute form basing on the kind of referenced DIE(local or not?).
  dwarf::Form NewForm = IsLocal ? dwarf::DW_FORM_ref4 : dwarf::DW_FORM_ref_addr;

  // Check whether current attribute references already cloned DIE inside
  // the same compilation unit. If true - write the already known offset value.
  if (IsLocal && (OutDieOffset != 0))
    return Generator.addScalarAttribute(AttrSpec.Attr, NewForm, OutDieOffset)
        .second;

  // If offset value is not known at this point then create patch for the
  // reference value and write dummy value into the attribute.
  DebugInfoOutputSection.notePatchWithOffsetUpdate(
      DebugDieRefPatch{AttrOutOffset, &CU, RefDiePair->first,
                       RefDiePair->second},
      PatchesOffsets);
  return Generator.addScalarAttribute(AttrSpec.Attr, NewForm, 0xBADDEF).second;
}

size_t DIEAttributeCloner::cloneScalarAttr(
    const DWARFFormValue &Val,
    const DWARFAbbreviationDeclaration::AttributeSpec &AttrSpec) {

  // Create patches for attribute referencing other non invariant section.
  // Invariant section could not be updated here as this section and
  // reference to it do not change value in case --update.
  if (AttrSpec.Attr == dwarf::DW_AT_macro_info) {
    if (std::optional<uint64_t> Offset = Val.getAsSectionOffset()) {
      const DWARFDebugMacro *Macro =
          CU.getContaingFile().Dwarf->getDebugMacinfo();
      if (Macro == nullptr || !Macro->hasEntryForOffset(*Offset))
        return 0;

      DebugInfoOutputSection.notePatchWithOffsetUpdate(
          DebugOffsetPatch{AttrOutOffset, &CU.getOrCreateSectionDescriptor(
                                              DebugSectionKind::DebugMacinfo)},
          PatchesOffsets);
    }
  } else if (AttrSpec.Attr == dwarf::DW_AT_macros) {
    if (std::optional<uint64_t> Offset = Val.getAsSectionOffset()) {
      const DWARFDebugMacro *Macro =
          CU.getContaingFile().Dwarf->getDebugMacro();
      if (Macro == nullptr || !Macro->hasEntryForOffset(*Offset))
        return 0;

      DebugInfoOutputSection.notePatchWithOffsetUpdate(
          DebugOffsetPatch{AttrOutOffset, &CU.getOrCreateSectionDescriptor(
                                              DebugSectionKind::DebugMacro)},
          PatchesOffsets);
    }
  } else if (AttrSpec.Attr == dwarf::DW_AT_stmt_list) {
    DebugInfoOutputSection.notePatchWithOffsetUpdate(
        DebugOffsetPatch{AttrOutOffset, &CU.getOrCreateSectionDescriptor(
                                            DebugSectionKind::DebugLine)},
        PatchesOffsets);
  } else if (AttrSpec.Attr == dwarf::DW_AT_str_offsets_base) {
    DebugInfoOutputSection.notePatchWithOffsetUpdate(
        DebugOffsetPatch{
            AttrOutOffset,
            &CU.getOrCreateSectionDescriptor(DebugSectionKind::DebugStrOffsets),
            true},
        PatchesOffsets);

    // Use size of .debug_str_offsets header as attribute value. The offset
    // to .debug_str_offsets would be added later while patching.
    AttrInfo.HasStringOffsetBaseAttr = true;
    return Generator
        .addScalarAttribute(AttrSpec.Attr, AttrSpec.Form,
                            CU.getDebugStrOffsetsHeaderSize())
        .second;
  }

  uint64_t Value;
  if (CU.getGlobalData().getOptions().UpdateIndexTablesOnly) {
    if (auto OptionalValue = Val.getAsUnsignedConstant())
      Value = *OptionalValue;
    else if (auto OptionalValue = Val.getAsSignedConstant())
      Value = *OptionalValue;
    else if (auto OptionalValue = Val.getAsSectionOffset())
      Value = *OptionalValue;
    else {
      CU.warn("unsupported scalar attribute form. Dropping attribute.",
              InputDieEntry);
      return 0;
    }

    if (AttrSpec.Attr == dwarf::DW_AT_declaration && Value)
      AttrInfo.IsDeclaration = true;

    if (AttrSpec.Form == dwarf::DW_FORM_loclistx)
      return Generator.addLocListAttribute(AttrSpec.Attr, AttrSpec.Form, Value)
          .second;

    return Generator.addScalarAttribute(AttrSpec.Attr, AttrSpec.Form, Value)
        .second;
  }

  dwarf::Form ResultingForm = AttrSpec.Form;
  if (AttrSpec.Form == dwarf::DW_FORM_rnglistx) {
    // DWARFLinker does not generate .debug_addr table. Thus we need to change
    // all "addrx" related forms to "addr" version. Change DW_FORM_rnglistx
    // to DW_FORM_sec_offset here.
    std::optional<uint64_t> Index = Val.getAsSectionOffset();
    if (!Index) {
      CU.warn("cann't read the attribute. Dropping.", InputDieEntry);
      return 0;
    }
    std::optional<uint64_t> Offset = CU.getOrigUnit().getRnglistOffset(*Index);
    if (!Offset) {
      CU.warn("cann't read the attribute. Dropping.", InputDieEntry);
      return 0;
    }

    Value = *Offset;
    ResultingForm = dwarf::DW_FORM_sec_offset;
  } else if (AttrSpec.Form == dwarf::DW_FORM_loclistx) {
    // DWARFLinker does not generate .debug_addr table. Thus we need to change
    // all "addrx" related forms to "addr" version. Change DW_FORM_loclistx
    // to DW_FORM_sec_offset here.
    std::optional<uint64_t> Index = Val.getAsSectionOffset();
    if (!Index) {
      CU.warn("cann't read the attribute. Dropping.", InputDieEntry);
      return 0;
    }
    std::optional<uint64_t> Offset = CU.getOrigUnit().getLoclistOffset(*Index);
    if (!Offset) {
      CU.warn("cann't read the attribute. Dropping.", InputDieEntry);
      return 0;
    }

    Value = *Offset;
    ResultingForm = dwarf::DW_FORM_sec_offset;
  } else if (AttrSpec.Attr == dwarf::DW_AT_high_pc &&
             InputDieEntry->getTag() == dwarf::DW_TAG_compile_unit) {
    std::optional<uint64_t> LowPC = CU.getLowPc();
    if (!LowPC)
      return 0;
    // Dwarf >= 4 high_pc is an size, not an address.
    Value = CU.getHighPc() - *LowPC;
  } else if (AttrSpec.Form == dwarf::DW_FORM_sec_offset)
    Value = *Val.getAsSectionOffset();
  else if (AttrSpec.Form == dwarf::DW_FORM_sdata)
    Value = *Val.getAsSignedConstant();
  else if (auto OptionalValue = Val.getAsUnsignedConstant())
    Value = *OptionalValue;
  else {
    CU.warn("unsupported scalar attribute form. Dropping attribute.",
            InputDieEntry);
    return 0;
  }

  if (AttrSpec.Attr == dwarf::DW_AT_ranges ||
      AttrSpec.Attr == dwarf::DW_AT_start_scope) {
    // Create patch for the range offset value.
    DebugInfoOutputSection.notePatchWithOffsetUpdate(
        DebugRangePatch{{AttrOutOffset},
                        InputDieEntry->getTag() == dwarf::DW_TAG_compile_unit},
        PatchesOffsets);
    AttrInfo.HasRanges = true;
  } else if (DWARFAttribute::mayHaveLocationList(AttrSpec.Attr) &&
             dwarf::doesFormBelongToClass(AttrSpec.Form,
                                          DWARFFormValue::FC_SectionOffset,
                                          CU.getOrigUnit().getVersion())) {
    int64_t AddrAdjustmentValue = 0;
    if (VarAddressAdjustment)
      AddrAdjustmentValue = *VarAddressAdjustment;
    else if (FuncAddressAdjustment)
      AddrAdjustmentValue = *FuncAddressAdjustment;

    // Create patch for the location offset value.
    DebugInfoOutputSection.notePatchWithOffsetUpdate(
        DebugLocPatch{{AttrOutOffset}, AddrAdjustmentValue}, PatchesOffsets);
  } else if (AttrSpec.Attr == dwarf::DW_AT_addr_base) {
    DebugInfoOutputSection.notePatchWithOffsetUpdate(
        DebugOffsetPatch{
            AttrOutOffset,
            &CU.getOrCreateSectionDescriptor(DebugSectionKind::DebugAddr),
            true},
        PatchesOffsets);

    // Use size of .debug_addr header as attribute value. The offset to
    // .debug_addr would be added later while patching.
    return Generator
        .addScalarAttribute(AttrSpec.Attr, AttrSpec.Form,
                            CU.getDebugAddrHeaderSize())
        .second;
  } else if (AttrSpec.Attr == dwarf::DW_AT_declaration && Value)
    AttrInfo.IsDeclaration = true;

  return Generator.addScalarAttribute(AttrSpec.Attr, ResultingForm, Value)
      .second;
}

size_t DIEAttributeCloner::cloneBlockAttr(
    const DWARFFormValue &Val,
    const DWARFAbbreviationDeclaration::AttributeSpec &AttrSpec) {

  size_t NumberOfPatchesAtStart = PatchesOffsets.size();

  // If the block is a DWARF Expression, clone it into the temporary
  // buffer using cloneExpression(), otherwise copy the data directly.
  SmallVector<uint8_t, 32> Buffer;
  ArrayRef<uint8_t> Bytes = *Val.getAsBlock();
  if (DWARFAttribute::mayHaveLocationExpr(AttrSpec.Attr) &&
      (Val.isFormClass(DWARFFormValue::FC_Block) ||
       Val.isFormClass(DWARFFormValue::FC_Exprloc))) {
    DWARFUnit &OrigUnit = CU.getOrigUnit();
    DataExtractor Data(StringRef((const char *)Bytes.data(), Bytes.size()),
                       OrigUnit.isLittleEndian(),
                       OrigUnit.getAddressByteSize());
    DWARFExpression Expr(Data, OrigUnit.getAddressByteSize(),
                         OrigUnit.getFormParams().Format);

    CU.cloneDieAttrExpression(Expr, Buffer, DebugInfoOutputSection,
                              VarAddressAdjustment, PatchesOffsets);
    Bytes = Buffer;
  }

  // The expression location data might be updated and exceed the original size.
  // Check whether the new data fits into the original form.
  dwarf::Form ResultForm = AttrSpec.Form;
  if ((ResultForm == dwarf::DW_FORM_block1 && Bytes.size() > UINT8_MAX) ||
      (ResultForm == dwarf::DW_FORM_block2 && Bytes.size() > UINT16_MAX) ||
      (ResultForm == dwarf::DW_FORM_block4 && Bytes.size() > UINT32_MAX))
    ResultForm = dwarf::DW_FORM_block;

  size_t FinalAttributeSize;
  if (AttrSpec.Form == dwarf::DW_FORM_exprloc)
    FinalAttributeSize =
        Generator.addLocationAttribute(AttrSpec.Attr, ResultForm, Bytes).second;
  else
    FinalAttributeSize =
        Generator.addBlockAttribute(AttrSpec.Attr, ResultForm, Bytes).second;

  // Update patches offsets with the size of length field for Bytes.
  for (size_t Idx = NumberOfPatchesAtStart; Idx < PatchesOffsets.size();
       Idx++) {
    assert(FinalAttributeSize > Bytes.size());
    *PatchesOffsets[Idx] +=
        (AttrOutOffset + (FinalAttributeSize - Bytes.size()));
  }

  return FinalAttributeSize;
}

size_t DIEAttributeCloner::cloneAddressAttr(
    const DWARFFormValue &Val,
    const DWARFAbbreviationDeclaration::AttributeSpec &AttrSpec) {
  if (AttrSpec.Attr == dwarf::DW_AT_low_pc)
    AttrInfo.HasLowPc = true;

  if (CU.getGlobalData().getOptions().UpdateIndexTablesOnly)
    return Generator
        .addScalarAttribute(AttrSpec.Attr, AttrSpec.Form, Val.getRawUValue())
        .second;

  // Cloned Die may have address attributes relocated to a
  // totally unrelated value. This can happen:
  //   - If high_pc is an address (Dwarf version == 2), then it might have been
  //     relocated to a totally unrelated value (because the end address in the
  //     object file might be start address of another function which got moved
  //     independently by the linker).
  //   - If address relocated in an inline_subprogram that happens at the
  //     beginning of its inlining function.
  //  To avoid above cases and to not apply relocation twice (in
  //  applyValidRelocs and here), read address attribute from InputDIE and apply
  //  Info.PCOffset here.

  std::optional<DWARFFormValue> AddrAttribute =
      CU.find(InputDieEntry, AttrSpec.Attr);
  if (!AddrAttribute)
    llvm_unreachable("Cann't find attribute");

  std::optional<uint64_t> Addr = AddrAttribute->getAsAddress();
  if (!Addr) {
    CU.warn("cann't read address attribute value.");
    return 0;
  }

  if (InputDieEntry->getTag() == dwarf::DW_TAG_compile_unit &&
      AttrSpec.Attr == dwarf::DW_AT_low_pc) {
    if (std::optional<uint64_t> LowPC = CU.getLowPc())
      Addr = *LowPC;
    else
      return 0;
  } else if (InputDieEntry->getTag() == dwarf::DW_TAG_compile_unit &&
             AttrSpec.Attr == dwarf::DW_AT_high_pc) {
    if (uint64_t HighPc = CU.getHighPc())
      Addr = HighPc;
    else
      return 0;
  } else {
    if (VarAddressAdjustment)
      *Addr += *VarAddressAdjustment;
    else if (FuncAddressAdjustment)
      *Addr += *FuncAddressAdjustment;
  }

  if (AttrSpec.Form == dwarf::DW_FORM_addr) {
    return Generator.addScalarAttribute(AttrSpec.Attr, AttrSpec.Form, *Addr)
        .second;
  }

  return Generator
      .addScalarAttribute(AttrSpec.Attr, dwarf::Form::DW_FORM_addrx,
                          CU.getDebugAddrIndex(*Addr))
      .second;
}

unsigned DIEAttributeCloner::finalizeAbbreviations(bool HasChildrenToClone) {
  size_t SizeOfAbbreviationNumber =
      Generator.finalizeAbbreviations(HasChildrenToClone);

  // We need to update patches offsets after we know the size of the
  // abbreviation number.
  updatePatchesWithSizeOfAbbreviationNumber(SizeOfAbbreviationNumber);

  // Add the size of the abbreviation number to the output offset.
  AttrOutOffset += SizeOfAbbreviationNumber;

  return AttrOutOffset;
}

} // end of namespace dwarflinker_parallel
} // namespace llvm
