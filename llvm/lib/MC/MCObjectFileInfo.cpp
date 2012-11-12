//===-- MObjectFileInfo.cpp - Object File Information ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSectionCOFF.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/ADT/Triple.h"
using namespace llvm;

void MCObjectFileInfo::InitMachOMCObjectFileInfo(Triple T) {
  // MachO
  IsFunctionEHFrameSymbolPrivate = false;
  SupportsWeakOmittedEHFrame = false;

  PersonalityEncoding = dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel
    | dwarf::DW_EH_PE_sdata4;
  LSDAEncoding = FDEEncoding = FDECFIEncoding = dwarf::DW_EH_PE_pcrel;
  TTypeEncoding = dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel |
    dwarf::DW_EH_PE_sdata4;

  // .comm doesn't support alignment before Leopard.
  if (T.isMacOSX() && T.isMacOSXVersionLT(10, 5))
    CommDirectiveSupportsAlignment = false;

  TextSection // .text
    = Ctx->getMachOSection("__TEXT", "__text",
                           MCSectionMachO::S_ATTR_PURE_INSTRUCTIONS,
                           SectionKind::getText());
  DataSection // .data
    = Ctx->getMachOSection("__DATA", "__data", 0,
                           SectionKind::getDataRel());

  TLSDataSection // .tdata
    = Ctx->getMachOSection("__DATA", "__thread_data",
                           MCSectionMachO::S_THREAD_LOCAL_REGULAR,
                           SectionKind::getDataRel());
  TLSBSSSection // .tbss
    = Ctx->getMachOSection("__DATA", "__thread_bss",
                           MCSectionMachO::S_THREAD_LOCAL_ZEROFILL,
                           SectionKind::getThreadBSS());

  // TODO: Verify datarel below.
  TLSTLVSection // .tlv
    = Ctx->getMachOSection("__DATA", "__thread_vars",
                           MCSectionMachO::S_THREAD_LOCAL_VARIABLES,
                           SectionKind::getDataRel());

  TLSThreadInitSection
    = Ctx->getMachOSection("__DATA", "__thread_init",
                          MCSectionMachO::S_THREAD_LOCAL_INIT_FUNCTION_POINTERS,
                          SectionKind::getDataRel());

  CStringSection // .cstring
    = Ctx->getMachOSection("__TEXT", "__cstring",
                           MCSectionMachO::S_CSTRING_LITERALS,
                           SectionKind::getMergeable1ByteCString());
  UStringSection
    = Ctx->getMachOSection("__TEXT","__ustring", 0,
                           SectionKind::getMergeable2ByteCString());
  FourByteConstantSection // .literal4
    = Ctx->getMachOSection("__TEXT", "__literal4",
                           MCSectionMachO::S_4BYTE_LITERALS,
                           SectionKind::getMergeableConst4());
  EightByteConstantSection // .literal8
    = Ctx->getMachOSection("__TEXT", "__literal8",
                           MCSectionMachO::S_8BYTE_LITERALS,
                           SectionKind::getMergeableConst8());

  // ld_classic doesn't support .literal16 in 32-bit mode, and ld64 falls back
  // to using it in -static mode.
  SixteenByteConstantSection = 0;
  if (RelocM != Reloc::Static &&
      T.getArch() != Triple::x86_64 && T.getArch() != Triple::ppc64)
    SixteenByteConstantSection =   // .literal16
      Ctx->getMachOSection("__TEXT", "__literal16",
                           MCSectionMachO::S_16BYTE_LITERALS,
                           SectionKind::getMergeableConst16());

  ReadOnlySection  // .const
    = Ctx->getMachOSection("__TEXT", "__const", 0,
                           SectionKind::getReadOnly());

  TextCoalSection
    = Ctx->getMachOSection("__TEXT", "__textcoal_nt",
                           MCSectionMachO::S_COALESCED |
                           MCSectionMachO::S_ATTR_PURE_INSTRUCTIONS,
                           SectionKind::getText());
  ConstTextCoalSection
    = Ctx->getMachOSection("__TEXT", "__const_coal",
                           MCSectionMachO::S_COALESCED,
                           SectionKind::getReadOnly());
  ConstDataSection  // .const_data
    = Ctx->getMachOSection("__DATA", "__const", 0,
                           SectionKind::getReadOnlyWithRel());
  DataCoalSection
    = Ctx->getMachOSection("__DATA","__datacoal_nt",
                           MCSectionMachO::S_COALESCED,
                           SectionKind::getDataRel());
  DataCommonSection
    = Ctx->getMachOSection("__DATA","__common",
                           MCSectionMachO::S_ZEROFILL,
                           SectionKind::getBSS());
  DataBSSSection
    = Ctx->getMachOSection("__DATA","__bss", MCSectionMachO::S_ZEROFILL,
                           SectionKind::getBSS());


  LazySymbolPointerSection
    = Ctx->getMachOSection("__DATA", "__la_symbol_ptr",
                           MCSectionMachO::S_LAZY_SYMBOL_POINTERS,
                           SectionKind::getMetadata());
  NonLazySymbolPointerSection
    = Ctx->getMachOSection("__DATA", "__nl_symbol_ptr",
                           MCSectionMachO::S_NON_LAZY_SYMBOL_POINTERS,
                           SectionKind::getMetadata());

  if (RelocM == Reloc::Static) {
    StaticCtorSection
      = Ctx->getMachOSection("__TEXT", "__constructor", 0,
                             SectionKind::getDataRel());
    StaticDtorSection
      = Ctx->getMachOSection("__TEXT", "__destructor", 0,
                             SectionKind::getDataRel());
  } else {
    StaticCtorSection
      = Ctx->getMachOSection("__DATA", "__mod_init_func",
                             MCSectionMachO::S_MOD_INIT_FUNC_POINTERS,
                             SectionKind::getDataRel());
    StaticDtorSection
      = Ctx->getMachOSection("__DATA", "__mod_term_func",
                             MCSectionMachO::S_MOD_TERM_FUNC_POINTERS,
                             SectionKind::getDataRel());
  }

  // Exception Handling.
  LSDASection = Ctx->getMachOSection("__TEXT", "__gcc_except_tab", 0,
                                     SectionKind::getReadOnlyWithRel());

  if (T.isMacOSX() && !T.isMacOSXVersionLT(10, 6))
    CompactUnwindSection =
      Ctx->getMachOSection("__LD", "__compact_unwind",
                           MCSectionMachO::S_ATTR_DEBUG,
                           SectionKind::getReadOnly());

  // Debug Information.
  DwarfAccelNamesSection =
    Ctx->getMachOSection("__DWARF", "__apple_names",
                         MCSectionMachO::S_ATTR_DEBUG,
                         SectionKind::getMetadata());
  DwarfAccelObjCSection =
    Ctx->getMachOSection("__DWARF", "__apple_objc",
                         MCSectionMachO::S_ATTR_DEBUG,
                         SectionKind::getMetadata());
  // 16 character section limit...
  DwarfAccelNamespaceSection =
    Ctx->getMachOSection("__DWARF", "__apple_namespac",
                         MCSectionMachO::S_ATTR_DEBUG,
                         SectionKind::getMetadata());
  DwarfAccelTypesSection =
    Ctx->getMachOSection("__DWARF", "__apple_types",
                         MCSectionMachO::S_ATTR_DEBUG,
                         SectionKind::getMetadata());

  DwarfAbbrevSection =
    Ctx->getMachOSection("__DWARF", "__debug_abbrev",
                         MCSectionMachO::S_ATTR_DEBUG,
                         SectionKind::getMetadata());
  DwarfInfoSection =
    Ctx->getMachOSection("__DWARF", "__debug_info",
                         MCSectionMachO::S_ATTR_DEBUG,
                         SectionKind::getMetadata());
  DwarfLineSection =
    Ctx->getMachOSection("__DWARF", "__debug_line",
                         MCSectionMachO::S_ATTR_DEBUG,
                         SectionKind::getMetadata());
  DwarfFrameSection =
    Ctx->getMachOSection("__DWARF", "__debug_frame",
                         MCSectionMachO::S_ATTR_DEBUG,
                         SectionKind::getMetadata());
  DwarfPubTypesSection =
    Ctx->getMachOSection("__DWARF", "__debug_pubtypes",
                         MCSectionMachO::S_ATTR_DEBUG,
                         SectionKind::getMetadata());
  DwarfStrSection =
    Ctx->getMachOSection("__DWARF", "__debug_str",
                         MCSectionMachO::S_ATTR_DEBUG,
                         SectionKind::getMetadata());
  DwarfLocSection =
    Ctx->getMachOSection("__DWARF", "__debug_loc",
                         MCSectionMachO::S_ATTR_DEBUG,
                         SectionKind::getMetadata());
  DwarfARangesSection =
    Ctx->getMachOSection("__DWARF", "__debug_aranges",
                         MCSectionMachO::S_ATTR_DEBUG,
                         SectionKind::getMetadata());
  DwarfRangesSection =
    Ctx->getMachOSection("__DWARF", "__debug_ranges",
                         MCSectionMachO::S_ATTR_DEBUG,
                         SectionKind::getMetadata());
  DwarfMacroInfoSection =
    Ctx->getMachOSection("__DWARF", "__debug_macinfo",
                         MCSectionMachO::S_ATTR_DEBUG,
                         SectionKind::getMetadata());
  DwarfDebugInlineSection =
    Ctx->getMachOSection("__DWARF", "__debug_inlined",
                         MCSectionMachO::S_ATTR_DEBUG,
                         SectionKind::getMetadata());

  TLSExtraDataSection = TLSTLVSection;
}

void MCObjectFileInfo::InitELFMCObjectFileInfo(Triple T) {
  if (T.getArch() == Triple::x86) {
    PersonalityEncoding = (RelocM == Reloc::PIC_)
     ? dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4
     : dwarf::DW_EH_PE_absptr;
    LSDAEncoding = (RelocM == Reloc::PIC_)
      ? dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4
      : dwarf::DW_EH_PE_absptr;
    FDEEncoding = FDECFIEncoding = (RelocM == Reloc::PIC_)
      ? dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4
      : dwarf::DW_EH_PE_absptr;
    TTypeEncoding = (RelocM == Reloc::PIC_)
     ? dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4
     : dwarf::DW_EH_PE_absptr;
  } else if (T.getArch() == Triple::x86_64) {
    FDECFIEncoding = dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4;

    if (RelocM == Reloc::PIC_) {
      PersonalityEncoding = dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel |
        ((CMModel == CodeModel::Small || CMModel == CodeModel::Medium)
         ? dwarf::DW_EH_PE_sdata4 : dwarf::DW_EH_PE_sdata8);
      LSDAEncoding = dwarf::DW_EH_PE_pcrel |
        (CMModel == CodeModel::Small
         ? dwarf::DW_EH_PE_sdata4 : dwarf::DW_EH_PE_sdata8);
      FDEEncoding = dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4;
      TTypeEncoding = dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel |
        ((CMModel == CodeModel::Small || CMModel == CodeModel::Medium)
         ? dwarf::DW_EH_PE_sdata4 : dwarf::DW_EH_PE_sdata8);
    } else {
      PersonalityEncoding =
        (CMModel == CodeModel::Small || CMModel == CodeModel::Medium)
        ? dwarf::DW_EH_PE_udata4 : dwarf::DW_EH_PE_absptr;
      LSDAEncoding = (CMModel == CodeModel::Small)
        ? dwarf::DW_EH_PE_udata4 : dwarf::DW_EH_PE_absptr;
      FDEEncoding = dwarf::DW_EH_PE_udata4;
      TTypeEncoding = (CMModel == CodeModel::Small)
        ? dwarf::DW_EH_PE_udata4 : dwarf::DW_EH_PE_absptr;
    }
  }

  // Solaris requires different flags for .eh_frame to seemingly every other
  // platform.
  EHSectionType = ELF::SHT_PROGBITS;
  EHSectionFlags = ELF::SHF_ALLOC;
  if (T.getOS() == Triple::Solaris) {
    if (T.getArch() == Triple::x86_64)
      EHSectionType = ELF::SHT_X86_64_UNWIND;
    else
      EHSectionFlags |= ELF::SHF_WRITE;
  }


  // ELF
  BSSSection =
    Ctx->getELFSection(".bss", ELF::SHT_NOBITS,
                       ELF::SHF_WRITE | ELF::SHF_ALLOC,
                       SectionKind::getBSS());

  TextSection =
    Ctx->getELFSection(".text", ELF::SHT_PROGBITS,
                       ELF::SHF_EXECINSTR |
                       ELF::SHF_ALLOC,
                       SectionKind::getText());

  DataSection =
    Ctx->getELFSection(".data", ELF::SHT_PROGBITS,
                       ELF::SHF_WRITE |ELF::SHF_ALLOC,
                       SectionKind::getDataRel());

  ReadOnlySection =
    Ctx->getELFSection(".rodata", ELF::SHT_PROGBITS,
                       ELF::SHF_ALLOC,
                       SectionKind::getReadOnly());

  TLSDataSection =
    Ctx->getELFSection(".tdata", ELF::SHT_PROGBITS,
                       ELF::SHF_ALLOC | ELF::SHF_TLS |
                       ELF::SHF_WRITE,
                       SectionKind::getThreadData());

  TLSBSSSection =
    Ctx->getELFSection(".tbss", ELF::SHT_NOBITS,
                       ELF::SHF_ALLOC | ELF::SHF_TLS |
                       ELF::SHF_WRITE,
                       SectionKind::getThreadBSS());

  DataRelSection =
    Ctx->getELFSection(".data.rel", ELF::SHT_PROGBITS,
                       ELF::SHF_ALLOC |ELF::SHF_WRITE,
                       SectionKind::getDataRel());

  DataRelLocalSection =
    Ctx->getELFSection(".data.rel.local", ELF::SHT_PROGBITS,
                       ELF::SHF_ALLOC |ELF::SHF_WRITE,
                       SectionKind::getDataRelLocal());

  DataRelROSection =
    Ctx->getELFSection(".data.rel.ro", ELF::SHT_PROGBITS,
                       ELF::SHF_ALLOC |ELF::SHF_WRITE,
                       SectionKind::getReadOnlyWithRel());

  DataRelROLocalSection =
    Ctx->getELFSection(".data.rel.ro.local", ELF::SHT_PROGBITS,
                       ELF::SHF_ALLOC |ELF::SHF_WRITE,
                       SectionKind::getReadOnlyWithRelLocal());

  MergeableConst4Section =
    Ctx->getELFSection(".rodata.cst4", ELF::SHT_PROGBITS,
                       ELF::SHF_ALLOC |ELF::SHF_MERGE,
                       SectionKind::getMergeableConst4());

  MergeableConst8Section =
    Ctx->getELFSection(".rodata.cst8", ELF::SHT_PROGBITS,
                       ELF::SHF_ALLOC |ELF::SHF_MERGE,
                       SectionKind::getMergeableConst8());

  MergeableConst16Section =
    Ctx->getELFSection(".rodata.cst16", ELF::SHT_PROGBITS,
                       ELF::SHF_ALLOC |ELF::SHF_MERGE,
                       SectionKind::getMergeableConst16());

  StaticCtorSection =
    Ctx->getELFSection(".ctors", ELF::SHT_PROGBITS,
                       ELF::SHF_ALLOC |ELF::SHF_WRITE,
                       SectionKind::getDataRel());

  StaticDtorSection =
    Ctx->getELFSection(".dtors", ELF::SHT_PROGBITS,
                       ELF::SHF_ALLOC |ELF::SHF_WRITE,
                       SectionKind::getDataRel());

  // Exception Handling Sections.

  // FIXME: We're emitting LSDA info into a readonly section on ELF, even though
  // it contains relocatable pointers.  In PIC mode, this is probably a big
  // runtime hit for C++ apps.  Either the contents of the LSDA need to be
  // adjusted or this should be a data section.
  LSDASection =
    Ctx->getELFSection(".gcc_except_table", ELF::SHT_PROGBITS,
                       ELF::SHF_ALLOC,
                       SectionKind::getReadOnly());

  // Debug Info Sections.
  DwarfAbbrevSection =
    Ctx->getELFSection(".debug_abbrev", ELF::SHT_PROGBITS, 0,
                       SectionKind::getMetadata());
  DwarfInfoSection =
    Ctx->getELFSection(".debug_info", ELF::SHT_PROGBITS, 0,
                       SectionKind::getMetadata());
  DwarfLineSection =
    Ctx->getELFSection(".debug_line", ELF::SHT_PROGBITS, 0,
                       SectionKind::getMetadata());
  DwarfFrameSection =
    Ctx->getELFSection(".debug_frame", ELF::SHT_PROGBITS, 0,
                       SectionKind::getMetadata());
  DwarfPubTypesSection =
    Ctx->getELFSection(".debug_pubtypes", ELF::SHT_PROGBITS, 0,
                       SectionKind::getMetadata());
  DwarfStrSection =
    Ctx->getELFSection(".debug_str", ELF::SHT_PROGBITS,
                       ELF::SHF_MERGE | ELF::SHF_STRINGS,
                       SectionKind::getMergeable1ByteCString());
  DwarfLocSection =
    Ctx->getELFSection(".debug_loc", ELF::SHT_PROGBITS, 0,
                       SectionKind::getMetadata());
  DwarfARangesSection =
    Ctx->getELFSection(".debug_aranges", ELF::SHT_PROGBITS, 0,
                       SectionKind::getMetadata());
  DwarfRangesSection =
    Ctx->getELFSection(".debug_ranges", ELF::SHT_PROGBITS, 0,
                       SectionKind::getMetadata());
  DwarfMacroInfoSection =
    Ctx->getELFSection(".debug_macinfo", ELF::SHT_PROGBITS, 0,
                       SectionKind::getMetadata());
  DwarfAccelNamesSection =
    Ctx->getELFSection(".apple_names", ELF::SHT_PROGBITS, 0,
                       SectionKind::getMetadata());
  DwarfAccelObjCSection =
    Ctx->getELFSection(".apple_objc", ELF::SHT_PROGBITS, 0,
                       SectionKind::getMetadata());
  DwarfAccelNamespaceSection =
    Ctx->getELFSection(".apple_namespaces", ELF::SHT_PROGBITS, 0,
                       SectionKind::getMetadata());
  DwarfAccelTypesSection =
    Ctx->getELFSection(".apple_types", ELF::SHT_PROGBITS, 0,
                       SectionKind::getMetadata());
}


void MCObjectFileInfo::InitCOFFMCObjectFileInfo(Triple T) {
  // COFF
  TextSection =
    Ctx->getCOFFSection(".text",
                        COFF::IMAGE_SCN_CNT_CODE |
                        COFF::IMAGE_SCN_MEM_EXECUTE |
                        COFF::IMAGE_SCN_MEM_READ,
                        SectionKind::getText());
  DataSection =
    Ctx->getCOFFSection(".data",
                        COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                        COFF::IMAGE_SCN_MEM_READ |
                        COFF::IMAGE_SCN_MEM_WRITE,
                        SectionKind::getDataRel());
  ReadOnlySection =
    Ctx->getCOFFSection(".rdata",
                        COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                        COFF::IMAGE_SCN_MEM_READ,
                        SectionKind::getReadOnly());
  if (T.getOS() == Triple::Win32) {
    StaticCtorSection =
      Ctx->getCOFFSection(".CRT$XCU",
                          COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                          COFF::IMAGE_SCN_MEM_READ,
                          SectionKind::getReadOnly());
  } else {
    StaticCtorSection =
      Ctx->getCOFFSection(".ctors",
                          COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                          COFF::IMAGE_SCN_MEM_READ |
                          COFF::IMAGE_SCN_MEM_WRITE,
                          SectionKind::getDataRel());
  }


  if (T.getOS() == Triple::Win32) {
    StaticDtorSection =
      Ctx->getCOFFSection(".CRT$XTX",
                          COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                          COFF::IMAGE_SCN_MEM_READ,
                          SectionKind::getReadOnly());
  } else {
    StaticDtorSection =
      Ctx->getCOFFSection(".dtors",
                          COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                          COFF::IMAGE_SCN_MEM_READ |
                          COFF::IMAGE_SCN_MEM_WRITE,
                          SectionKind::getDataRel());
  }

  // FIXME: We're emitting LSDA info into a readonly section on COFF, even
  // though it contains relocatable pointers.  In PIC mode, this is probably a
  // big runtime hit for C++ apps.  Either the contents of the LSDA need to be
  // adjusted or this should be a data section.
  LSDASection =
    Ctx->getCOFFSection(".gcc_except_table",
                        COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                        COFF::IMAGE_SCN_MEM_READ,
                        SectionKind::getReadOnly());

  // Debug info.
  DwarfAbbrevSection =
    Ctx->getCOFFSection(".debug_abbrev",
                        COFF::IMAGE_SCN_MEM_DISCARDABLE |
                        COFF::IMAGE_SCN_MEM_READ,
                        SectionKind::getMetadata());
  DwarfInfoSection =
    Ctx->getCOFFSection(".debug_info",
                        COFF::IMAGE_SCN_MEM_DISCARDABLE |
                        COFF::IMAGE_SCN_MEM_READ,
                        SectionKind::getMetadata());
  DwarfLineSection =
    Ctx->getCOFFSection(".debug_line",
                        COFF::IMAGE_SCN_MEM_DISCARDABLE |
                        COFF::IMAGE_SCN_MEM_READ,
                        SectionKind::getMetadata());
  DwarfFrameSection =
    Ctx->getCOFFSection(".debug_frame",
                        COFF::IMAGE_SCN_MEM_DISCARDABLE |
                        COFF::IMAGE_SCN_MEM_READ,
                        SectionKind::getMetadata());
  DwarfPubTypesSection =
    Ctx->getCOFFSection(".debug_pubtypes",
                        COFF::IMAGE_SCN_MEM_DISCARDABLE |
                        COFF::IMAGE_SCN_MEM_READ,
                        SectionKind::getMetadata());
  DwarfStrSection =
    Ctx->getCOFFSection(".debug_str",
                        COFF::IMAGE_SCN_MEM_DISCARDABLE |
                        COFF::IMAGE_SCN_MEM_READ,
                        SectionKind::getMetadata());
  DwarfLocSection =
    Ctx->getCOFFSection(".debug_loc",
                        COFF::IMAGE_SCN_MEM_DISCARDABLE |
                        COFF::IMAGE_SCN_MEM_READ,
                        SectionKind::getMetadata());
  DwarfARangesSection =
    Ctx->getCOFFSection(".debug_aranges",
                        COFF::IMAGE_SCN_MEM_DISCARDABLE |
                        COFF::IMAGE_SCN_MEM_READ,
                        SectionKind::getMetadata());
  DwarfRangesSection =
    Ctx->getCOFFSection(".debug_ranges",
                        COFF::IMAGE_SCN_MEM_DISCARDABLE |
                        COFF::IMAGE_SCN_MEM_READ,
                        SectionKind::getMetadata());
  DwarfMacroInfoSection =
    Ctx->getCOFFSection(".debug_macinfo",
                        COFF::IMAGE_SCN_MEM_DISCARDABLE |
                        COFF::IMAGE_SCN_MEM_READ,
                        SectionKind::getMetadata());

  DrectveSection =
    Ctx->getCOFFSection(".drectve",
                        COFF::IMAGE_SCN_LNK_INFO,
                        SectionKind::getMetadata());

  PDataSection =
    Ctx->getCOFFSection(".pdata",
                        COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                        COFF::IMAGE_SCN_MEM_READ,
                        SectionKind::getDataRel());

  XDataSection =
    Ctx->getCOFFSection(".xdata",
                        COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                        COFF::IMAGE_SCN_MEM_READ,
                        SectionKind::getDataRel());
  TLSDataSection =
    Ctx->getCOFFSection(".tls$",
                        COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                        COFF::IMAGE_SCN_MEM_READ |
                        COFF::IMAGE_SCN_MEM_WRITE,
                        SectionKind::getDataRel());
}

void MCObjectFileInfo::InitMCObjectFileInfo(StringRef TT, Reloc::Model relocm,
                                            CodeModel::Model cm,
                                            MCContext &ctx) {
  RelocM = relocm;
  CMModel = cm;
  Ctx = &ctx;

  // Common.
  CommDirectiveSupportsAlignment = true;
  SupportsWeakOmittedEHFrame = true;
  IsFunctionEHFrameSymbolPrivate = true;

  PersonalityEncoding = LSDAEncoding = FDEEncoding = FDECFIEncoding =
    TTypeEncoding = dwarf::DW_EH_PE_absptr;

  EHFrameSection = 0;             // Created on demand.
  CompactUnwindSection = 0;       // Used only by selected targets.
  DwarfAccelNamesSection = 0;     // Used only by selected targets.
  DwarfAccelObjCSection = 0;      // Used only by selected targets.
  DwarfAccelNamespaceSection = 0; // Used only by selected targets.
  DwarfAccelTypesSection = 0;     // Used only by selected targets.

  Triple T(TT);
  Triple::ArchType Arch = T.getArch();
  // FIXME: Checking for Arch here to filter out bogus triples such as
  // cellspu-apple-darwin. Perhaps we should fix in Triple?
  if ((Arch == Triple::x86 || Arch == Triple::x86_64 ||
       Arch == Triple::arm || Arch == Triple::thumb ||
       Arch == Triple::ppc || Arch == Triple::ppc64 ||
       Arch == Triple::UnknownArch) &&
      (T.isOSDarwin() || T.getEnvironment() == Triple::MachO)) {
    Env = IsMachO;
    InitMachOMCObjectFileInfo(T);
  } else if ((Arch == Triple::x86 || Arch == Triple::x86_64) &&
             (T.getEnvironment() != Triple::ELF) &&
             (T.getOS() == Triple::MinGW32 || T.getOS() == Triple::Cygwin ||
              T.getOS() == Triple::Win32)) {
    Env = IsCOFF;
    InitCOFFMCObjectFileInfo(T);
  } else {
    Env = IsELF;
    InitELFMCObjectFileInfo(T);
  }
}

void MCObjectFileInfo::InitEHFrameSection() {
  if (Env == IsMachO)
    EHFrameSection =
      Ctx->getMachOSection("__TEXT", "__eh_frame",
                           MCSectionMachO::S_COALESCED |
                           MCSectionMachO::S_ATTR_NO_TOC |
                           MCSectionMachO::S_ATTR_STRIP_STATIC_SYMS |
                           MCSectionMachO::S_ATTR_LIVE_SUPPORT,
                           SectionKind::getReadOnly());
  else if (Env == IsELF)
    EHFrameSection =
      Ctx->getELFSection(".eh_frame", EHSectionType,
                         EHSectionFlags,
                         SectionKind::getDataRel());
  else
    EHFrameSection =
      Ctx->getCOFFSection(".eh_frame",
                          COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                          COFF::IMAGE_SCN_MEM_READ |
                          COFF::IMAGE_SCN_MEM_WRITE,
                          SectionKind::getDataRel());
}
