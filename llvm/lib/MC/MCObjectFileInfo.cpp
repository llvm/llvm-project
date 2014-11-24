//===-- MObjectFileInfo.cpp - Object File Information ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSectionCOFF.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSectionMachO.h"
using namespace llvm;

static bool useCompactUnwind(const Triple &T) {
  // Only on darwin.
  if (!T.isOSDarwin())
    return false;

  // aarch64 always has it.
  if (T.getArch() == Triple::arm64 || T.getArch() == Triple::aarch64)
    return true;

  // Use it on newer version of OS X.
  if (T.isMacOSX() && !T.isMacOSXVersionLT(10, 6))
    return true;

  // And the iOS simulator.
  if (T.isiOS() &&
      (T.getArch() == Triple::x86_64 || T.getArch() == Triple::x86))
    return true;

  return false;
}

void MCObjectFileInfo::InitMachOMCObjectFileInfo(Triple T) {
  // MachO
  SupportsWeakOmittedEHFrame = false;

  if (T.isOSDarwin() &&
      (T.getArch() == Triple::arm64 || T.getArch() == Triple::aarch64))
    SupportsCompactUnwindWithoutEHFrame = true;

  PersonalityEncoding = dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel
    | dwarf::DW_EH_PE_sdata4;
  LSDAEncoding = FDECFIEncoding = dwarf::DW_EH_PE_pcrel;
  TTypeEncoding = dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel |
    dwarf::DW_EH_PE_sdata4;

  // .comm doesn't support alignment before Leopard.
  if (T.isMacOSX() && T.isMacOSXVersionLT(10, 5))
    CommDirectiveSupportsAlignment = false;

  TextSection // .text
    = Ctx->getMachOSection("__TEXT", "__text",
                           MachO::S_ATTR_PURE_INSTRUCTIONS,
                           SectionKind::getText());
  DataSection // .data
    = Ctx->getMachOSection("__DATA", "__data", 0,
                           SectionKind::getDataRel());

  // BSSSection might not be expected initialized on msvc.
  BSSSection = nullptr;

  TLSDataSection // .tdata
    = Ctx->getMachOSection("__DATA", "__thread_data",
                           MachO::S_THREAD_LOCAL_REGULAR,
                           SectionKind::getDataRel());
  TLSBSSSection // .tbss
    = Ctx->getMachOSection("__DATA", "__thread_bss",
                           MachO::S_THREAD_LOCAL_ZEROFILL,
                           SectionKind::getThreadBSS());

  // TODO: Verify datarel below.
  TLSTLVSection // .tlv
    = Ctx->getMachOSection("__DATA", "__thread_vars",
                           MachO::S_THREAD_LOCAL_VARIABLES,
                           SectionKind::getDataRel());

  TLSThreadInitSection
    = Ctx->getMachOSection("__DATA", "__thread_init",
                          MachO::S_THREAD_LOCAL_INIT_FUNCTION_POINTERS,
                          SectionKind::getDataRel());

  CStringSection // .cstring
    = Ctx->getMachOSection("__TEXT", "__cstring",
                           MachO::S_CSTRING_LITERALS,
                           SectionKind::getMergeable1ByteCString());
  UStringSection
    = Ctx->getMachOSection("__TEXT","__ustring", 0,
                           SectionKind::getMergeable2ByteCString());
  FourByteConstantSection // .literal4
    = Ctx->getMachOSection("__TEXT", "__literal4",
                           MachO::S_4BYTE_LITERALS,
                           SectionKind::getMergeableConst4());
  EightByteConstantSection // .literal8
    = Ctx->getMachOSection("__TEXT", "__literal8",
                           MachO::S_8BYTE_LITERALS,
                           SectionKind::getMergeableConst8());

  SixteenByteConstantSection // .literal16
      = Ctx->getMachOSection("__TEXT", "__literal16",
                             MachO::S_16BYTE_LITERALS,
                             SectionKind::getMergeableConst16());

  ReadOnlySection  // .const
    = Ctx->getMachOSection("__TEXT", "__const", 0,
                           SectionKind::getReadOnly());

  TextCoalSection
    = Ctx->getMachOSection("__TEXT", "__textcoal_nt",
                           MachO::S_COALESCED |
                           MachO::S_ATTR_PURE_INSTRUCTIONS,
                           SectionKind::getText());
  ConstTextCoalSection
    = Ctx->getMachOSection("__TEXT", "__const_coal",
                           MachO::S_COALESCED,
                           SectionKind::getReadOnly());
  ConstDataSection  // .const_data
    = Ctx->getMachOSection("__DATA", "__const", 0,
                           SectionKind::getReadOnlyWithRel());
  DataCoalSection
    = Ctx->getMachOSection("__DATA","__datacoal_nt",
                           MachO::S_COALESCED,
                           SectionKind::getDataRel());
  DataCommonSection
    = Ctx->getMachOSection("__DATA","__common",
                           MachO::S_ZEROFILL,
                           SectionKind::getBSS());
  DataBSSSection
    = Ctx->getMachOSection("__DATA","__bss", MachO::S_ZEROFILL,
                           SectionKind::getBSS());


  LazySymbolPointerSection
    = Ctx->getMachOSection("__DATA", "__la_symbol_ptr",
                           MachO::S_LAZY_SYMBOL_POINTERS,
                           SectionKind::getMetadata());
  NonLazySymbolPointerSection
    = Ctx->getMachOSection("__DATA", "__nl_symbol_ptr",
                           MachO::S_NON_LAZY_SYMBOL_POINTERS,
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
                             MachO::S_MOD_INIT_FUNC_POINTERS,
                             SectionKind::getDataRel());
    StaticDtorSection
      = Ctx->getMachOSection("__DATA", "__mod_term_func",
                             MachO::S_MOD_TERM_FUNC_POINTERS,
                             SectionKind::getDataRel());
  }

  // Exception Handling.
  LSDASection = Ctx->getMachOSection("__TEXT", "__gcc_except_tab", 0,
                                     SectionKind::getReadOnlyWithRel());

  COFFDebugSymbolsSection = nullptr;

  if (useCompactUnwind(T)) {
    CompactUnwindSection =
        Ctx->getMachOSection("__LD", "__compact_unwind", MachO::S_ATTR_DEBUG,
                             SectionKind::getReadOnly());

    if (T.getArch() == Triple::x86_64 || T.getArch() == Triple::x86)
      CompactUnwindDwarfEHFrameOnly = 0x04000000;
    else if (T.getArch() == Triple::arm64 || T.getArch() == Triple::aarch64)
      CompactUnwindDwarfEHFrameOnly = 0x03000000;
  }

  // Debug Information.
  DwarfAccelNamesSection =
    Ctx->getMachOSection("__DWARF", "__apple_names",
                         MachO::S_ATTR_DEBUG,
                         SectionKind::getMetadata());
  DwarfAccelObjCSection =
    Ctx->getMachOSection("__DWARF", "__apple_objc",
                         MachO::S_ATTR_DEBUG,
                         SectionKind::getMetadata());
  // 16 character section limit...
  DwarfAccelNamespaceSection =
    Ctx->getMachOSection("__DWARF", "__apple_namespac",
                         MachO::S_ATTR_DEBUG,
                         SectionKind::getMetadata());
  DwarfAccelTypesSection =
    Ctx->getMachOSection("__DWARF", "__apple_types",
                         MachO::S_ATTR_DEBUG,
                         SectionKind::getMetadata());

  DwarfAbbrevSection =
    Ctx->getMachOSection("__DWARF", "__debug_abbrev",
                         MachO::S_ATTR_DEBUG,
                         SectionKind::getMetadata());
  DwarfInfoSection =
    Ctx->getMachOSection("__DWARF", "__debug_info",
                         MachO::S_ATTR_DEBUG,
                         SectionKind::getMetadata());
  DwarfLineSection =
    Ctx->getMachOSection("__DWARF", "__debug_line",
                         MachO::S_ATTR_DEBUG,
                         SectionKind::getMetadata());
  DwarfFrameSection =
    Ctx->getMachOSection("__DWARF", "__debug_frame",
                         MachO::S_ATTR_DEBUG,
                         SectionKind::getMetadata());
  DwarfPubNamesSection =
    Ctx->getMachOSection("__DWARF", "__debug_pubnames",
                         MachO::S_ATTR_DEBUG,
                         SectionKind::getMetadata());
  DwarfPubTypesSection =
    Ctx->getMachOSection("__DWARF", "__debug_pubtypes",
                         MachO::S_ATTR_DEBUG,
                         SectionKind::getMetadata());
  DwarfGnuPubNamesSection =
    Ctx->getMachOSection("__DWARF", "__debug_gnu_pubn",
                         MachO::S_ATTR_DEBUG,
                         SectionKind::getMetadata());
  DwarfGnuPubTypesSection =
    Ctx->getMachOSection("__DWARF", "__debug_gnu_pubt",
                         MachO::S_ATTR_DEBUG,
                         SectionKind::getMetadata());
  DwarfStrSection =
    Ctx->getMachOSection("__DWARF", "__debug_str",
                         MachO::S_ATTR_DEBUG,
                         SectionKind::getMetadata());
  DwarfLocSection =
    Ctx->getMachOSection("__DWARF", "__debug_loc",
                         MachO::S_ATTR_DEBUG,
                         SectionKind::getMetadata());
  DwarfARangesSection =
    Ctx->getMachOSection("__DWARF", "__debug_aranges",
                         MachO::S_ATTR_DEBUG,
                         SectionKind::getMetadata());
  DwarfRangesSection =
    Ctx->getMachOSection("__DWARF", "__debug_ranges",
                         MachO::S_ATTR_DEBUG,
                         SectionKind::getMetadata());
  DwarfMacroInfoSection =
    Ctx->getMachOSection("__DWARF", "__debug_macinfo",
                         MachO::S_ATTR_DEBUG,
                         SectionKind::getMetadata());
  DwarfDebugInlineSection =
    Ctx->getMachOSection("__DWARF", "__debug_inlined",
                         MachO::S_ATTR_DEBUG,
                         SectionKind::getMetadata());
  StackMapSection =
    Ctx->getMachOSection("__LLVM_STACKMAPS", "__llvm_stackmaps", 0,
                         SectionKind::getMetadata());

  TLSExtraDataSection = TLSTLVSection;
}

void MCObjectFileInfo::InitELFMCObjectFileInfo(Triple T) {
  switch (T.getArch()) {
  case Triple::mips:
  case Triple::mipsel:
    FDECFIEncoding = dwarf::DW_EH_PE_sdata4;
    break;
  case Triple::mips64:
  case Triple::mips64el:
    FDECFIEncoding = dwarf::DW_EH_PE_sdata8;
    break;
  default:
    FDECFIEncoding = dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4;
    break;
  }

  switch (T.getArch()) {
  case Triple::arm:
  case Triple::armeb:
  case Triple::thumb:
  case Triple::thumbeb:
    if (Ctx->getAsmInfo()->getExceptionHandlingType() == ExceptionHandling::ARM)
      break;
    // Fallthrough if not using EHABI
  case Triple::x86:
    PersonalityEncoding = (RelocM == Reloc::PIC_)
     ? dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4
     : dwarf::DW_EH_PE_absptr;
    LSDAEncoding = (RelocM == Reloc::PIC_)
      ? dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4
      : dwarf::DW_EH_PE_absptr;
    TTypeEncoding = (RelocM == Reloc::PIC_)
     ? dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4
     : dwarf::DW_EH_PE_absptr;
    break;
  case Triple::x86_64:
    if (RelocM == Reloc::PIC_) {
      PersonalityEncoding = dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel |
        ((CMModel == CodeModel::Small || CMModel == CodeModel::Medium)
         ? dwarf::DW_EH_PE_sdata4 : dwarf::DW_EH_PE_sdata8);
      LSDAEncoding = dwarf::DW_EH_PE_pcrel |
        (CMModel == CodeModel::Small
         ? dwarf::DW_EH_PE_sdata4 : dwarf::DW_EH_PE_sdata8);
      TTypeEncoding = dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel |
        ((CMModel == CodeModel::Small || CMModel == CodeModel::Medium)
         ? dwarf::DW_EH_PE_sdata4 : dwarf::DW_EH_PE_sdata8);
    } else {
      PersonalityEncoding =
        (CMModel == CodeModel::Small || CMModel == CodeModel::Medium)
        ? dwarf::DW_EH_PE_udata4 : dwarf::DW_EH_PE_absptr;
      LSDAEncoding = (CMModel == CodeModel::Small)
        ? dwarf::DW_EH_PE_udata4 : dwarf::DW_EH_PE_absptr;
      TTypeEncoding = (CMModel == CodeModel::Small)
        ? dwarf::DW_EH_PE_udata4 : dwarf::DW_EH_PE_absptr;
    }
    break;
  case Triple::aarch64:
  case Triple::aarch64_be:
  case Triple::arm64:
  case Triple::arm64_be:
    // The small model guarantees static code/data size < 4GB, but not where it
    // will be in memory. Most of these could end up >2GB away so even a signed
    // pc-relative 32-bit address is insufficient, theoretically.
    if (RelocM == Reloc::PIC_) {
      PersonalityEncoding = dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel |
        dwarf::DW_EH_PE_sdata8;
      LSDAEncoding = dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata8;
      TTypeEncoding = dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel |
        dwarf::DW_EH_PE_sdata8;
    } else {
      PersonalityEncoding = dwarf::DW_EH_PE_absptr;
      LSDAEncoding = dwarf::DW_EH_PE_absptr;
      TTypeEncoding = dwarf::DW_EH_PE_absptr;
    }
    break;
  case Triple::mips:
  case Triple::mipsel:
  case Triple::mips64:
  case Triple::mips64el:
    // MIPS uses indirect pointer to refer personality functions, so that the
    // eh_frame section can be read-only.  DW.ref.personality will be generated
    // for relocation.
    PersonalityEncoding = dwarf::DW_EH_PE_indirect;
    break;
  case Triple::ppc64:
  case Triple::ppc64le:
    PersonalityEncoding = dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel |
      dwarf::DW_EH_PE_udata8;
    LSDAEncoding = dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_udata8;
    TTypeEncoding = dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel |
      dwarf::DW_EH_PE_udata8;
    break;
  case Triple::sparc:
    if (RelocM == Reloc::PIC_) {
      LSDAEncoding = dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4;
      PersonalityEncoding = dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel |
        dwarf::DW_EH_PE_sdata4;
      TTypeEncoding = dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel |
        dwarf::DW_EH_PE_sdata4;
    } else {
      LSDAEncoding = dwarf::DW_EH_PE_absptr;
      PersonalityEncoding = dwarf::DW_EH_PE_absptr;
      TTypeEncoding = dwarf::DW_EH_PE_absptr;
    }
    break;
  case Triple::sparcv9:
    LSDAEncoding = dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4;
    if (RelocM == Reloc::PIC_) {
      PersonalityEncoding = dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel |
        dwarf::DW_EH_PE_sdata4;
      TTypeEncoding = dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel |
        dwarf::DW_EH_PE_sdata4;
    } else {
      PersonalityEncoding = dwarf::DW_EH_PE_absptr;
      TTypeEncoding = dwarf::DW_EH_PE_absptr;
    }
    break;
  case Triple::systemz:
    // All currently-defined code models guarantee that 4-byte PC-relative
    // values will be in range.
    if (RelocM == Reloc::PIC_) {
      PersonalityEncoding = dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel |
        dwarf::DW_EH_PE_sdata4;
      LSDAEncoding = dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4;
      TTypeEncoding = dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel |
        dwarf::DW_EH_PE_sdata4;
    } else {
      PersonalityEncoding = dwarf::DW_EH_PE_absptr;
      LSDAEncoding = dwarf::DW_EH_PE_absptr;
      TTypeEncoding = dwarf::DW_EH_PE_absptr;
    }
    break;
  default:
    break;
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

  COFFDebugSymbolsSection = nullptr;

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
  DwarfPubNamesSection =
    Ctx->getELFSection(".debug_pubnames", ELF::SHT_PROGBITS, 0,
                       SectionKind::getMetadata());
  DwarfPubTypesSection =
    Ctx->getELFSection(".debug_pubtypes", ELF::SHT_PROGBITS, 0,
                       SectionKind::getMetadata());
  DwarfGnuPubNamesSection =
    Ctx->getELFSection(".debug_gnu_pubnames", ELF::SHT_PROGBITS, 0,
                       SectionKind::getMetadata());
  DwarfGnuPubTypesSection =
    Ctx->getELFSection(".debug_gnu_pubtypes", ELF::SHT_PROGBITS, 0,
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

  // DWARF5 Experimental Debug Info

  // Accelerator Tables
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

  // Fission Sections
  DwarfInfoDWOSection =
    Ctx->getELFSection(".debug_info.dwo", ELF::SHT_PROGBITS, 0,
                       SectionKind::getMetadata());
  DwarfAbbrevDWOSection =
    Ctx->getELFSection(".debug_abbrev.dwo", ELF::SHT_PROGBITS, 0,
                       SectionKind::getMetadata());
  DwarfStrDWOSection =
    Ctx->getELFSection(".debug_str.dwo", ELF::SHT_PROGBITS,
                       ELF::SHF_MERGE | ELF::SHF_STRINGS,
                       SectionKind::getMergeable1ByteCString());
  DwarfLineDWOSection =
    Ctx->getELFSection(".debug_line.dwo", ELF::SHT_PROGBITS, 0,
                       SectionKind::getMetadata());
  DwarfLocDWOSection =
    Ctx->getELFSection(".debug_loc.dwo", ELF::SHT_PROGBITS, 0,
                       SectionKind::getMetadata());
  DwarfStrOffDWOSection =
    Ctx->getELFSection(".debug_str_offsets.dwo", ELF::SHT_PROGBITS, 0,
                       SectionKind::getMetadata());
  DwarfAddrSection =
    Ctx->getELFSection(".debug_addr", ELF::SHT_PROGBITS, 0,
                       SectionKind::getMetadata());
}


void MCObjectFileInfo::InitCOFFMCObjectFileInfo(Triple T) {
  bool IsWoA = T.getArch() == Triple::arm || T.getArch() == Triple::thumb;

  // The object file format cannot represent common symbols with explicit
  // alignments.
  CommDirectiveSupportsAlignment = false;

  // COFF
  BSSSection =
    Ctx->getCOFFSection(".bss",
                        COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA |
                        COFF::IMAGE_SCN_MEM_READ |
                        COFF::IMAGE_SCN_MEM_WRITE,
                        SectionKind::getBSS());
  TextSection =
    Ctx->getCOFFSection(".text",
                        (IsWoA ? COFF::IMAGE_SCN_MEM_16BIT
                               : (COFF::SectionCharacteristics)0) |
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

  if (T.isKnownWindowsMSVCEnvironment() || T.isWindowsItaniumEnvironment()) {
    StaticCtorSection =
      Ctx->getCOFFSection(".CRT$XCU",
                          COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                          COFF::IMAGE_SCN_MEM_READ,
                          SectionKind::getReadOnly());
    StaticDtorSection =
      Ctx->getCOFFSection(".CRT$XTX",
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
  assert(T.isOSWindows() && "Windows is the only supported COFF target");
  if (T.getArch() == Triple::x86_64) {
    // On Windows 64 with SEH, the LSDA is emitted into the .xdata section
    LSDASection = 0;
  } else {
    LSDASection = Ctx->getCOFFSection(".gcc_except_table",
                                      COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                                          COFF::IMAGE_SCN_MEM_READ,
                                      SectionKind::getReadOnly());
  }

  // Debug info.
  COFFDebugSymbolsSection =
    Ctx->getCOFFSection(".debug$S",
                        COFF::IMAGE_SCN_MEM_DISCARDABLE |
                        COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                        COFF::IMAGE_SCN_MEM_READ,
                        SectionKind::getMetadata());

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
  DwarfPubNamesSection =
    Ctx->getCOFFSection(".debug_pubnames",
                        COFF::IMAGE_SCN_MEM_DISCARDABLE |
                        COFF::IMAGE_SCN_MEM_READ,
                        SectionKind::getMetadata());
  DwarfPubTypesSection =
    Ctx->getCOFFSection(".debug_pubtypes",
                        COFF::IMAGE_SCN_MEM_DISCARDABLE |
                        COFF::IMAGE_SCN_MEM_READ,
                        SectionKind::getMetadata());
  DwarfGnuPubNamesSection =
    Ctx->getCOFFSection(".debug_gnu_pubnames",
                        COFF::IMAGE_SCN_MEM_DISCARDABLE |
                        COFF::IMAGE_SCN_MEM_READ,
                        SectionKind::getMetadata());
  DwarfGnuPubTypesSection =
    Ctx->getCOFFSection(".debug_gnu_pubtypes",
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
  DwarfInfoDWOSection =
      Ctx->getCOFFSection(".debug_info.dwo",
                          COFF::IMAGE_SCN_MEM_DISCARDABLE |
                          COFF::IMAGE_SCN_MEM_READ,
                          SectionKind::getMetadata());
  DwarfAbbrevDWOSection =
      Ctx->getCOFFSection(".debug_abbrev.dwo",
                          COFF::IMAGE_SCN_MEM_DISCARDABLE |
                          COFF::IMAGE_SCN_MEM_READ,
                          SectionKind::getMetadata());
  DwarfStrDWOSection =
      Ctx->getCOFFSection(".debug_str.dwo",
                          COFF::IMAGE_SCN_MEM_DISCARDABLE |
                          COFF::IMAGE_SCN_MEM_READ,
                          SectionKind::getMetadata());
  DwarfLineDWOSection =
      Ctx->getCOFFSection(".debug_line.dwo",
                          COFF::IMAGE_SCN_MEM_DISCARDABLE |
                          COFF::IMAGE_SCN_MEM_READ,
                          SectionKind::getMetadata());
  DwarfLocDWOSection =
      Ctx->getCOFFSection(".debug_loc.dwo",
                          COFF::IMAGE_SCN_MEM_DISCARDABLE |
                          COFF::IMAGE_SCN_MEM_READ,
                          SectionKind::getMetadata());
  DwarfStrOffDWOSection =
      Ctx->getCOFFSection(".debug_str_offsets.dwo",
                          COFF::IMAGE_SCN_MEM_DISCARDABLE |
                          COFF::IMAGE_SCN_MEM_READ,
                          SectionKind::getMetadata());

  DwarfAddrSection =
    Ctx->getCOFFSection(".debug_addr",
                        COFF::IMAGE_SCN_MEM_DISCARDABLE |
                        COFF::IMAGE_SCN_MEM_READ,
                        SectionKind::getMetadata());

  DrectveSection =
    Ctx->getCOFFSection(".drectve",
                        COFF::IMAGE_SCN_LNK_INFO |
                        COFF::IMAGE_SCN_LNK_REMOVE,
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

void MCObjectFileInfo::InitMCObjectFileInfo(StringRef T, Reloc::Model relocm,
                                            CodeModel::Model cm,
                                            MCContext &ctx) {
  RelocM = relocm;
  CMModel = cm;
  Ctx = &ctx;

  // Common.
  CommDirectiveSupportsAlignment = true;
  SupportsWeakOmittedEHFrame = true;
  SupportsCompactUnwindWithoutEHFrame = false;

  PersonalityEncoding = LSDAEncoding = FDECFIEncoding = TTypeEncoding =
      dwarf::DW_EH_PE_absptr;

  CompactUnwindDwarfEHFrameOnly = 0;

  EHFrameSection = nullptr;             // Created on demand.
  CompactUnwindSection = nullptr;       // Used only by selected targets.
  DwarfAccelNamesSection = nullptr;     // Used only by selected targets.
  DwarfAccelObjCSection = nullptr;      // Used only by selected targets.
  DwarfAccelNamespaceSection = nullptr; // Used only by selected targets.
  DwarfAccelTypesSection = nullptr;     // Used only by selected targets.

  TT = Triple(T);

  Triple::ArchType Arch = TT.getArch();
  // FIXME: Checking for Arch here to filter out bogus triples such as
  // cellspu-apple-darwin. Perhaps we should fix in Triple?
  if ((Arch == Triple::x86 || Arch == Triple::x86_64 ||
       Arch == Triple::arm || Arch == Triple::thumb ||
       Arch == Triple::arm64 || Arch == Triple::aarch64 ||
       Arch == Triple::ppc || Arch == Triple::ppc64 ||
       Arch == Triple::UnknownArch) &&
      (TT.isOSDarwin() || TT.isOSBinFormatMachO())) {
    Env = IsMachO;
    InitMachOMCObjectFileInfo(TT);
  } else if ((Arch == Triple::x86 || Arch == Triple::x86_64 ||
              Arch == Triple::arm || Arch == Triple::thumb) &&
             (TT.isOSWindows() && TT.getObjectFormat() == Triple::COFF)) {
    Env = IsCOFF;
    InitCOFFMCObjectFileInfo(TT);
  } else {
    Env = IsELF;
    InitELFMCObjectFileInfo(TT);
  }
}

const MCSection *MCObjectFileInfo::getDwarfTypesSection(uint64_t Hash) const {
  return Ctx->getELFSection(".debug_types", ELF::SHT_PROGBITS, ELF::SHF_GROUP,
                            SectionKind::getMetadata(), 0, utostr(Hash));
}

const MCSection *
MCObjectFileInfo::getDwarfTypesDWOSection(uint64_t Hash) const {
  return Ctx->getELFSection(".debug_types.dwo", ELF::SHT_PROGBITS,
                            ELF::SHF_GROUP, SectionKind::getMetadata(), 0,
                            utostr(Hash));
}

void MCObjectFileInfo::InitEHFrameSection() {
  if (Env == IsMachO)
    EHFrameSection =
      Ctx->getMachOSection("__TEXT", "__eh_frame",
                           MachO::S_COALESCED |
                           MachO::S_ATTR_NO_TOC |
                           MachO::S_ATTR_STRIP_STATIC_SYMS |
                           MachO::S_ATTR_LIVE_SUPPORT,
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
