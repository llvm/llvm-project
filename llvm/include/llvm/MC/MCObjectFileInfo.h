//===-- llvm/MC/MCObjectFileInfo.h - Object File Info -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes common object file formats.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCBJECTFILEINFO_H
#define LLVM_MC_MCBJECTFILEINFO_H

#include "llvm/Support/CodeGen.h"

namespace llvm {
  class MCContext;
  class MCSection;
  class StringRef;
  class Triple;

class MCObjectFileInfo {
protected:
  /// CommDirectiveSupportsAlignment - True if .comm supports alignment.  This
  /// is a hack for as long as we support 10.4 Tiger, whose assembler doesn't
  /// support alignment on comm.
  bool CommDirectiveSupportsAlignment;

  /// SupportsWeakEmptyEHFrame - True if target object file supports a
  /// weak_definition of constant 0 for an omitted EH frame.
  bool SupportsWeakOmittedEHFrame;

  /// IsFunctionEHFrameSymbolPrivate - This flag is set to true if the
  /// "EH_frame" symbol for EH information should be an assembler temporary (aka
  /// private linkage, aka an L or .L label) or false if it should be a normal
  /// non-.globl label.  This defaults to true.
  bool IsFunctionEHFrameSymbolPrivate;

  /// PersonalityEncoding, LSDAEncoding, FDEEncoding, TTypeEncoding - Some
  /// encoding values for EH.
  unsigned PersonalityEncoding;
  unsigned LSDAEncoding;
  unsigned FDEEncoding;
  unsigned FDECFIEncoding;
  unsigned TTypeEncoding;
  // Section flags for eh_frame
  unsigned EHSectionType;
  unsigned EHSectionFlags;

  /// TextSection - Section directive for standard text.
  ///
  const MCSection *TextSection;

  /// DataSection - Section directive for standard data.
  ///
  const MCSection *DataSection;

  /// BSSSection - Section that is default initialized to zero.
  const MCSection *BSSSection;

  /// ReadOnlySection - Section that is readonly and can contain arbitrary
  /// initialized data.  Targets are not required to have a readonly section.
  /// If they don't, various bits of code will fall back to using the data
  /// section for constants.
  const MCSection *ReadOnlySection;

  /// StaticCtorSection - This section contains the static constructor pointer
  /// list.
  const MCSection *StaticCtorSection;

  /// StaticDtorSection - This section contains the static destructor pointer
  /// list.
  const MCSection *StaticDtorSection;

  /// LSDASection - If exception handling is supported by the target, this is
  /// the section the Language Specific Data Area information is emitted to.
  const MCSection *LSDASection;

  /// CompactUnwindSection - If exception handling is supported by the target
  /// and the target can support a compact representation of the CIE and FDE,
  /// this is the section to emit them into.
  const MCSection *CompactUnwindSection;

  /// DwarfAccelNamesSection, DwarfAccelObjCSection,
  /// DwarfAccelNamespaceSection, DwarfAccelTypesSection -
  /// If we use the DWARF accelerated hash tables then we want toe emit these
  /// sections.
  const MCSection *DwarfAccelNamesSection;
  const MCSection *DwarfAccelObjCSection;
  const MCSection *DwarfAccelNamespaceSection;
  const MCSection *DwarfAccelTypesSection;

  // Dwarf sections for debug info.  If a target supports debug info, these must
  // be set.
  const MCSection *DwarfAbbrevSection;
  const MCSection *DwarfInfoSection;
  const MCSection *DwarfLineSection;
  const MCSection *DwarfFrameSection;
  const MCSection *DwarfPubTypesSection;
  const MCSection *DwarfDebugInlineSection;
  const MCSection *DwarfStrSection;
  const MCSection *DwarfLocSection;
  const MCSection *DwarfARangesSection;
  const MCSection *DwarfRangesSection;
  const MCSection *DwarfMacroInfoSection;

  // Extra TLS Variable Data section.  If the target needs to put additional
  // information for a TLS variable, it'll go here.
  const MCSection *TLSExtraDataSection;

  /// TLSDataSection - Section directive for Thread Local data.
  /// ELF, MachO and COFF.
  const MCSection *TLSDataSection;        // Defaults to ".tdata".

  /// TLSBSSSection - Section directive for Thread Local uninitialized data.
  /// Null if this target doesn't support a BSS section.
  /// ELF and MachO only.
  const MCSection *TLSBSSSection;         // Defaults to ".tbss".


  /// EHFrameSection - EH frame section. It is initialized on demand so it
  /// can be overwritten (with uniquing).
  const MCSection *EHFrameSection;

  /// ELF specific sections.
  ///
  const MCSection *DataRelSection;
  const MCSection *DataRelLocalSection;
  const MCSection *DataRelROSection;
  const MCSection *DataRelROLocalSection;
  const MCSection *MergeableConst4Section;
  const MCSection *MergeableConst8Section;
  const MCSection *MergeableConst16Section;

  /// MachO specific sections.
  ///

  /// TLSTLVSection - Section for thread local structure information.
  /// Contains the source code name of the variable, visibility and a pointer
  /// to the initial value (.tdata or .tbss).
  const MCSection *TLSTLVSection;         // Defaults to ".tlv".

  /// TLSThreadInitSection - Section for thread local data initialization
  /// functions.
  const MCSection *TLSThreadInitSection;  // Defaults to ".thread_init_func".

  const MCSection *CStringSection;
  const MCSection *UStringSection;
  const MCSection *TextCoalSection;
  const MCSection *ConstTextCoalSection;
  const MCSection *ConstDataSection;
  const MCSection *DataCoalSection;
  const MCSection *DataCommonSection;
  const MCSection *DataBSSSection;
  const MCSection *FourByteConstantSection;
  const MCSection *EightByteConstantSection;
  const MCSection *SixteenByteConstantSection;
  const MCSection *LazySymbolPointerSection;
  const MCSection *NonLazySymbolPointerSection;

  /// COFF specific sections.
  ///
  const MCSection *DrectveSection;
  const MCSection *PDataSection;
  const MCSection *XDataSection;

public:
  void InitMCObjectFileInfo(StringRef TT, Reloc::Model RM, CodeModel::Model CM,
                            MCContext &ctx);

  bool isFunctionEHFrameSymbolPrivate() const {
    return IsFunctionEHFrameSymbolPrivate;
  }
  bool getSupportsWeakOmittedEHFrame() const {
    return SupportsWeakOmittedEHFrame;
  }
  bool getCommDirectiveSupportsAlignment() const {
    return CommDirectiveSupportsAlignment;
  }

  unsigned getPersonalityEncoding() const { return PersonalityEncoding; }
  unsigned getLSDAEncoding() const { return LSDAEncoding; }
  unsigned getFDEEncoding(bool CFI) const {
    return CFI ? FDECFIEncoding : FDEEncoding;
  }
  unsigned getTTypeEncoding() const { return TTypeEncoding; }

  const MCSection *getTextSection() const { return TextSection; }
  const MCSection *getDataSection() const { return DataSection; }
  const MCSection *getBSSSection() const { return BSSSection; }
  const MCSection *getLSDASection() const { return LSDASection; }
  const MCSection *getCompactUnwindSection() const{
    return CompactUnwindSection;
  }
  const MCSection *getDwarfAccelNamesSection() const {
    return DwarfAccelNamesSection;
  }
  const MCSection *getDwarfAccelObjCSection() const {
    return DwarfAccelObjCSection;
  }
  const MCSection *getDwarfAccelNamespaceSection() const {
    return DwarfAccelNamespaceSection;
  }
  const MCSection *getDwarfAccelTypesSection() const {
    return DwarfAccelTypesSection;
  }
  const MCSection *getDwarfAbbrevSection() const { return DwarfAbbrevSection; }
  const MCSection *getDwarfInfoSection() const { return DwarfInfoSection; }
  const MCSection *getDwarfLineSection() const { return DwarfLineSection; }
  const MCSection *getDwarfFrameSection() const { return DwarfFrameSection; }
  const MCSection *getDwarfPubTypesSection() const{return DwarfPubTypesSection;}
  const MCSection *getDwarfDebugInlineSection() const {
    return DwarfDebugInlineSection;
  }
  const MCSection *getDwarfStrSection() const { return DwarfStrSection; }
  const MCSection *getDwarfLocSection() const { return DwarfLocSection; }
  const MCSection *getDwarfARangesSection() const { return DwarfARangesSection;}
  const MCSection *getDwarfRangesSection() const { return DwarfRangesSection; }
  const MCSection *getDwarfMacroInfoSection() const {
    return DwarfMacroInfoSection;
  }
  const MCSection *getTLSExtraDataSection() const {
    return TLSExtraDataSection;
  }
  const MCSection *getTLSDataSection() const { return TLSDataSection; }
  const MCSection *getTLSBSSSection() const { return TLSBSSSection; }

  /// ELF specific sections.
  ///
  const MCSection *getDataRelSection() const { return DataRelSection; }
  const MCSection *getDataRelLocalSection() const {
    return DataRelLocalSection;
  }
  const MCSection *getDataRelROSection() const { return DataRelROSection; }
  const MCSection *getDataRelROLocalSection() const {
    return DataRelROLocalSection;
  }
  const MCSection *getMergeableConst4Section() const {
    return MergeableConst4Section;
  }
  const MCSection *getMergeableConst8Section() const {
    return MergeableConst8Section;
  }
  const MCSection *getMergeableConst16Section() const {
    return MergeableConst16Section;
  }

  /// MachO specific sections.
  ///
  const MCSection *getTLSTLVSection() const { return TLSTLVSection; }
  const MCSection *getTLSThreadInitSection() const {
    return TLSThreadInitSection;
  }
  const MCSection *getCStringSection() const { return CStringSection; }
  const MCSection *getUStringSection() const { return UStringSection; }
  const MCSection *getTextCoalSection() const { return TextCoalSection; }
  const MCSection *getConstTextCoalSection() const {
    return ConstTextCoalSection;
  }
  const MCSection *getConstDataSection() const { return ConstDataSection; }
  const MCSection *getDataCoalSection() const { return DataCoalSection; }
  const MCSection *getDataCommonSection() const { return DataCommonSection; }
  const MCSection *getDataBSSSection() const { return DataBSSSection; }
  const MCSection *getFourByteConstantSection() const {
    return FourByteConstantSection;
  }
  const MCSection *getEightByteConstantSection() const {
    return EightByteConstantSection;
  }
  const MCSection *getSixteenByteConstantSection() const {
    return SixteenByteConstantSection;
  }
  const MCSection *getLazySymbolPointerSection() const {
    return LazySymbolPointerSection;
  }
  const MCSection *getNonLazySymbolPointerSection() const {
    return NonLazySymbolPointerSection;
  }

  /// COFF specific sections.
  ///
  const MCSection *getDrectveSection() const { return DrectveSection; }
  const MCSection *getPDataSection() const { return PDataSection; }
  const MCSection *getXDataSection() const { return XDataSection; }

  const MCSection *getEHFrameSection() {
    if (!EHFrameSection)
      InitEHFrameSection();
    return EHFrameSection;
  }

private:
  enum Environment { IsMachO, IsELF, IsCOFF };
  Environment Env;
  Reloc::Model RelocM;
  CodeModel::Model CMModel;
  MCContext *Ctx;

  void InitMachOMCObjectFileInfo(Triple T);
  void InitELFMCObjectFileInfo(Triple T);
  void InitCOFFMCObjectFileInfo(Triple T);

  /// InitEHFrameSection - Initialize EHFrameSection on demand.
  ///
  void InitEHFrameSection();
};

} // end namespace llvm

#endif
