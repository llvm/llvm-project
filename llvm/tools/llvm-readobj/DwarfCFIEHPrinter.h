//===--- DwarfCFIEHPrinter.h - DWARF-based Unwind Information Printer -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_READOBJ_DWARFCFIEHPRINTER_H
#define LLVM_TOOLS_LLVM_READOBJ_DWARFCFIEHPRINTER_H

#include "Error.h"
#include "llvm-readobj.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/Debug.h"
#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/type_traits.h"

namespace llvm {
namespace DwarfCFIEH {

template <typename ELFT>
class PrinterContext {
  ScopedPrinter &W;
  const object::ELFObjectFile<ELFT> *ObjF;

  void printEHFrameHdr(uint64_t Offset, uint64_t Address, uint64_t Size) const;

  void printEHFrame(const typename ELFT::Shdr *EHFrameShdr) const;

public:
  PrinterContext(ScopedPrinter &W, const object::ELFObjectFile<ELFT> *ObjF)
      : W(W), ObjF(ObjF) {}

  void printUnwindInformation() const;
};

template <class ELFT>
static const typename object::ELFObjectFile<ELFT>::Elf_Shdr *
findSectionByAddress(const object::ELFObjectFile<ELFT> *ObjF, uint64_t Addr) {
  auto Sections = ObjF->getELFFile()->sections();
  if (Error E = Sections.takeError())
    reportError(std::move(E), ObjF->getFileName());

  for (const auto &Shdr : *Sections)
    if (Shdr.sh_addr == Addr)
      return &Shdr;
  return nullptr;
}

template <typename ELFT>
void PrinterContext<ELFT>::printUnwindInformation() const {
  const object::ELFFile<ELFT> *Obj = ObjF->getELFFile();
  const typename ELFT::Phdr *EHFramePhdr = nullptr;

  auto PHs = Obj->program_headers();
  if (Error E = PHs.takeError())
    reportError(std::move(E), ObjF->getFileName());

  for (const auto &Phdr : *PHs) {
    if (Phdr.p_type == ELF::PT_GNU_EH_FRAME) {
      EHFramePhdr = &Phdr;
      if (Phdr.p_memsz != Phdr.p_filesz)
        reportError(object::createError(
                        "p_memsz does not match p_filesz for GNU_EH_FRAME"),
                    ObjF->getFileName());
      break;
    }
  }

  if (EHFramePhdr)
    printEHFrameHdr(EHFramePhdr->p_offset, EHFramePhdr->p_vaddr,
                    EHFramePhdr->p_memsz);

  auto Sections = Obj->sections();
  if (Error E = Sections.takeError())
    reportError(std::move(E), ObjF->getFileName());

  for (const auto &Shdr : *Sections) {
    auto SectionName = Obj->getSectionName(&Shdr);
    if (Error E = SectionName.takeError())
      reportError(std::move(E), ObjF->getFileName());

    if (*SectionName == ".eh_frame")
      printEHFrame(&Shdr);
  }
}

template <typename ELFT>
void PrinterContext<ELFT>::printEHFrameHdr(uint64_t EHFrameHdrOffset,
                                           uint64_t EHFrameHdrAddress,
                                           uint64_t EHFrameHdrSize) const {
  DictScope L(W, "EHFrameHeader");
  W.startLine() << format("Address: 0x%" PRIx64 "\n", EHFrameHdrAddress);
  W.startLine() << format("Offset: 0x%" PRIx64 "\n", EHFrameHdrOffset);
  W.startLine() << format("Size: 0x%" PRIx64 "\n", EHFrameHdrSize);

  const object::ELFFile<ELFT> *Obj = ObjF->getELFFile();
  const auto *EHFrameHdrShdr = findSectionByAddress(ObjF, EHFrameHdrAddress);
  if (EHFrameHdrShdr) {
    auto SectionName = Obj->getSectionName(EHFrameHdrShdr);
    if (Error E = SectionName.takeError())
      reportError(std::move(E), ObjF->getFileName());

    W.printString("Corresponding Section", *SectionName);
  }

  DataExtractor DE(makeArrayRef(Obj->base() + EHFrameHdrOffset, EHFrameHdrSize),
                   ELFT::TargetEndianness == support::endianness::little,
                   ELFT::Is64Bits ? 8 : 4);

  DictScope D(W, "Header");
  uint64_t Offset = 0;

  auto Version = DE.getU8(&Offset);
  W.printNumber("version", Version);
  if (Version != 1)
    reportError(
        object::createError("only version 1 of .eh_frame_hdr is supported"),
        ObjF->getFileName());

  uint64_t EHFramePtrEnc = DE.getU8(&Offset);
  W.startLine() << format("eh_frame_ptr_enc: 0x%" PRIx64 "\n", EHFramePtrEnc);
  if (EHFramePtrEnc != (dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4))
    reportError(object::createError("unexpected encoding eh_frame_ptr_enc"),
                ObjF->getFileName());

  uint64_t FDECountEnc = DE.getU8(&Offset);
  W.startLine() << format("fde_count_enc: 0x%" PRIx64 "\n", FDECountEnc);
  if (FDECountEnc != dwarf::DW_EH_PE_udata4)
    reportError(object::createError("unexpected encoding fde_count_enc"),
                ObjF->getFileName());

  uint64_t TableEnc = DE.getU8(&Offset);
  W.startLine() << format("table_enc: 0x%" PRIx64 "\n", TableEnc);
  if (TableEnc != (dwarf::DW_EH_PE_datarel | dwarf::DW_EH_PE_sdata4))
    reportError(object::createError("unexpected encoding table_enc"),
                ObjF->getFileName());

  auto EHFramePtr = DE.getSigned(&Offset, 4) + EHFrameHdrAddress + 4;
  W.startLine() << format("eh_frame_ptr: 0x%" PRIx64 "\n", EHFramePtr);

  auto FDECount = DE.getUnsigned(&Offset, 4);
  W.printNumber("fde_count", FDECount);

  unsigned NumEntries = 0;
  uint64_t PrevPC = 0;
  while (Offset + 8 <= EHFrameHdrSize && NumEntries < FDECount) {
    DictScope D(W, std::string("entry ")  + std::to_string(NumEntries));

    auto InitialPC = DE.getSigned(&Offset, 4) + EHFrameHdrAddress;
    W.startLine() << format("initial_location: 0x%" PRIx64 "\n", InitialPC);
    auto Address = DE.getSigned(&Offset, 4) + EHFrameHdrAddress;
    W.startLine() << format("address: 0x%" PRIx64 "\n", Address);

    if (InitialPC < PrevPC)
      reportError(object::createError("initial_location is out of order"),
                  ObjF->getFileName());

    PrevPC = InitialPC;
    ++NumEntries;
  }
}

template <typename ELFT>
void PrinterContext<ELFT>::printEHFrame(
    const typename ELFT::Shdr *EHFrameShdr) const {
  uint64_t Address = EHFrameShdr->sh_addr;
  uint64_t ShOffset = EHFrameShdr->sh_offset;
  W.startLine() << format(".eh_frame section at offset 0x%" PRIx64
                          " address 0x%" PRIx64 ":\n",
                          ShOffset, Address);
  W.indent();

  const object::ELFFile<ELFT> *Obj = ObjF->getELFFile();
  auto Result = Obj->getSectionContents(EHFrameShdr);
  if (Error E = Result.takeError())
    reportError(std::move(E), ObjF->getFileName());

  DWARFDataExtractor DE(*Result,
                        ELFT::TargetEndianness == support::endianness::little,
                        ELFT::Is64Bits ? 8 : 4);
  DWARFDebugFrame EHFrame(Triple::ArchType(ObjF->getArch()), /*IsEH=*/true,
                          /*EHFrameAddress=*/Address);
  if (Error E = EHFrame.parse(DE))
    reportError(std::move(E), ObjF->getFileName());

  for (const auto &Entry : EHFrame) {
    if (const auto *CIE = dyn_cast<dwarf::CIE>(&Entry)) {
      W.startLine() << format("[0x%" PRIx64 "] CIE length=%" PRIu64 "\n",
                              Address + CIE->getOffset(),
                              CIE->getLength());
      W.indent();

      W.printNumber("version", CIE->getVersion());
      W.printString("augmentation", CIE->getAugmentationString());
      W.printNumber("code_alignment_factor", CIE->getCodeAlignmentFactor());
      W.printNumber("data_alignment_factor", CIE->getDataAlignmentFactor());
      W.printNumber("return_address_register", CIE->getReturnAddressRegister());

      W.getOStream() << "\n";
      W.startLine() << "Program:\n";
      W.indent();
      CIE->cfis().dump(W.getOStream(), nullptr, W.getIndentLevel());
      W.unindent();

      W.unindent();
      W.getOStream() << "\n";

    } else if (const auto *FDE = dyn_cast<dwarf::FDE>(&Entry)) {
      W.startLine() << format("[0x%" PRIx64 "] FDE length=%" PRIu64
                              " cie=[0x%" PRIx64 "]\n",
                              Address + FDE->getOffset(),
                              FDE->getLength(),
                              Address + FDE->getLinkedCIE()->getOffset());
      W.indent();

      W.startLine() << format("initial_location: 0x%" PRIx64 "\n",
                              FDE->getInitialLocation());
      W.startLine()
        << format("address_range: 0x%" PRIx64 " (end : 0x%" PRIx64 ")\n",
                  FDE->getAddressRange(),
                  FDE->getInitialLocation() + FDE->getAddressRange());

      W.getOStream() << "\n";
      W.startLine() << "Program:\n";
      W.indent();
      FDE->cfis().dump(W.getOStream(), nullptr, W.getIndentLevel());
      W.unindent();

      W.unindent();
      W.getOStream() << "\n";
    } else {
      llvm_unreachable("unexpected DWARF frame kind");
    }
  }

  W.unindent();
}

}
}

#endif
