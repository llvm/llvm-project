//===-- AMDGPUTargetStreamer.cpp - Mips Target Streamer Methods -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides AMDGPU specific target streamer methods.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUTargetStreamer.h"
#include "AMDGPU.h"
#include "SIDefines.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "Utils/AMDKernelCodeTUtils.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/Support/FormattedStream.h"

namespace llvm {
#include "AMDGPUPTNote.h"
}

using namespace llvm;
using namespace llvm::AMDGPU;

//===----------------------------------------------------------------------===//
// AMDGPUTargetStreamer
//===----------------------------------------------------------------------===//

unsigned AMDGPUTargetStreamer::getMACH(StringRef GPU) const {
  return llvm::StringSwitch<unsigned>(GPU)
      // Radeon HD 2000/3000 Series (R600).
      .Case("r600", ELF::EF_AMDGPU_MACH_R600_R600)
      .Case("r630", ELF::EF_AMDGPU_MACH_R600_R630)
      .Case("rs880", ELF::EF_AMDGPU_MACH_R600_RS880)
      .Case("rv670", ELF::EF_AMDGPU_MACH_R600_RV670)
      // Radeon HD 4000 Series (R700).
      .Case("rv710", ELF::EF_AMDGPU_MACH_R600_RV710)
      .Case("rv730", ELF::EF_AMDGPU_MACH_R600_RV730)
      .Case("rv770", ELF::EF_AMDGPU_MACH_R600_RV770)
      // Radeon HD 5000 Series (Evergreen).
      .Case("cedar", ELF::EF_AMDGPU_MACH_R600_CEDAR)
      .Case("cypress", ELF::EF_AMDGPU_MACH_R600_CYPRESS)
      .Case("juniper", ELF::EF_AMDGPU_MACH_R600_JUNIPER)
      .Case("redwood", ELF::EF_AMDGPU_MACH_R600_REDWOOD)
      .Case("sumo", ELF::EF_AMDGPU_MACH_R600_SUMO)
      // Radeon HD 6000 Series (Northern Islands).
      .Case("barts", ELF::EF_AMDGPU_MACH_R600_BARTS)
      .Case("caicos", ELF::EF_AMDGPU_MACH_R600_CAICOS)
      .Case("cayman", ELF::EF_AMDGPU_MACH_R600_CAYMAN)
      .Case("turks", ELF::EF_AMDGPU_MACH_R600_TURKS)
      // AMDGCN GFX6.
      .Case("gfx600", ELF::EF_AMDGPU_MACH_AMDGCN_GFX600)
      .Case("tahiti", ELF::EF_AMDGPU_MACH_AMDGCN_GFX600)
      .Case("gfx601", ELF::EF_AMDGPU_MACH_AMDGCN_GFX601)
      .Case("hainan", ELF::EF_AMDGPU_MACH_AMDGCN_GFX601)
      .Case("oland", ELF::EF_AMDGPU_MACH_AMDGCN_GFX601)
      .Case("pitcairn", ELF::EF_AMDGPU_MACH_AMDGCN_GFX601)
      .Case("verde", ELF::EF_AMDGPU_MACH_AMDGCN_GFX601)
      // AMDGCN GFX7.
      .Case("gfx700", ELF::EF_AMDGPU_MACH_AMDGCN_GFX700)
      .Case("kaveri", ELF::EF_AMDGPU_MACH_AMDGCN_GFX700)
      .Case("gfx701", ELF::EF_AMDGPU_MACH_AMDGCN_GFX701)
      .Case("hawaii", ELF::EF_AMDGPU_MACH_AMDGCN_GFX701)
      .Case("gfx702", ELF::EF_AMDGPU_MACH_AMDGCN_GFX702)
      .Case("gfx703", ELF::EF_AMDGPU_MACH_AMDGCN_GFX703)
      .Case("kabini", ELF::EF_AMDGPU_MACH_AMDGCN_GFX703)
      .Case("mullins", ELF::EF_AMDGPU_MACH_AMDGCN_GFX703)
      .Case("gfx704", ELF::EF_AMDGPU_MACH_AMDGCN_GFX704)
      .Case("bonaire", ELF::EF_AMDGPU_MACH_AMDGCN_GFX704)
      // AMDGCN GFX8.
      .Case("gfx801", ELF::EF_AMDGPU_MACH_AMDGCN_GFX801)
      .Case("carrizo", ELF::EF_AMDGPU_MACH_AMDGCN_GFX801)
      .Case("gfx802", ELF::EF_AMDGPU_MACH_AMDGCN_GFX802)
      .Case("iceland", ELF::EF_AMDGPU_MACH_AMDGCN_GFX802)
      .Case("tonga", ELF::EF_AMDGPU_MACH_AMDGCN_GFX802)
      .Case("gfx803", ELF::EF_AMDGPU_MACH_AMDGCN_GFX803)
      .Case("fiji", ELF::EF_AMDGPU_MACH_AMDGCN_GFX803)
      .Case("polaris10", ELF::EF_AMDGPU_MACH_AMDGCN_GFX803)
      .Case("polaris11", ELF::EF_AMDGPU_MACH_AMDGCN_GFX803)
      .Case("gfx810", ELF::EF_AMDGPU_MACH_AMDGCN_GFX810)
      .Case("stoney", ELF::EF_AMDGPU_MACH_AMDGCN_GFX810)
      // AMDGCN GFX9.
      .Case("gfx900", ELF::EF_AMDGPU_MACH_AMDGCN_GFX900)
      .Case("gfx902", ELF::EF_AMDGPU_MACH_AMDGCN_GFX902)
      // Not specified processor.
      .Default(ELF::EF_AMDGPU_MACH_NONE);
}

bool AMDGPUTargetStreamer::EmitHSAMetadata(StringRef HSAMetadataString) {
  HSAMD::Metadata HSAMetadata;
  if (HSAMD::fromString(HSAMetadataString, HSAMetadata))
    return false;

  return EmitHSAMetadata(HSAMetadata);
}

//===----------------------------------------------------------------------===//
// AMDGPUTargetAsmStreamer
//===----------------------------------------------------------------------===//

AMDGPUTargetAsmStreamer::AMDGPUTargetAsmStreamer(MCStreamer &S,
                                                 formatted_raw_ostream &OS)
    : AMDGPUTargetStreamer(S), OS(OS) { }

void
AMDGPUTargetAsmStreamer::EmitDirectiveHSACodeObjectVersion(uint32_t Major,
                                                           uint32_t Minor) {
  OS << "\t.hsa_code_object_version " <<
        Twine(Major) << "," << Twine(Minor) << '\n';
}

void
AMDGPUTargetAsmStreamer::EmitDirectiveHSACodeObjectISA(uint32_t Major,
                                                       uint32_t Minor,
                                                       uint32_t Stepping,
                                                       StringRef VendorName,
                                                       StringRef ArchName) {
  OS << "\t.hsa_code_object_isa " <<
        Twine(Major) << "," << Twine(Minor) << "," << Twine(Stepping) <<
        ",\"" << VendorName << "\",\"" << ArchName << "\"\n";

}

void
AMDGPUTargetAsmStreamer::EmitAMDKernelCodeT(const amd_kernel_code_t &Header) {
  OS << "\t.amd_kernel_code_t\n";
  dumpAmdKernelCode(&Header, OS, "\t\t");
  OS << "\t.end_amd_kernel_code_t\n";
}

void AMDGPUTargetAsmStreamer::EmitAMDGPUSymbolType(StringRef SymbolName,
                                                   unsigned Type) {
  switch (Type) {
    default: llvm_unreachable("Invalid AMDGPU symbol type");
    case ELF::STT_AMDGPU_HSA_KERNEL:
      OS << "\t.amdgpu_hsa_kernel " << SymbolName << '\n' ;
      break;
  }
}

bool AMDGPUTargetAsmStreamer::EmitISAVersion(StringRef IsaVersionString) {
  OS << "\t.amd_amdgpu_isa \"" << IsaVersionString << "\"\n";
  return true;
}

bool AMDGPUTargetAsmStreamer::EmitHSAMetadata(
    const AMDGPU::HSAMD::Metadata &HSAMetadata) {
  std::string HSAMetadataString;
  if (HSAMD::toString(HSAMetadata, HSAMetadataString))
    return false;

  OS << '\t' << HSAMD::AssemblerDirectiveBegin << '\n';
  OS << HSAMetadataString << '\n';
  OS << '\t' << HSAMD::AssemblerDirectiveEnd << '\n';
  return true;
}

bool AMDGPUTargetAsmStreamer::EmitPALMetadata(
    const PALMD::Metadata &PALMetadata) {
  std::string PALMetadataString;
  if (PALMD::toString(PALMetadata, PALMetadataString))
    return false;

  OS << '\t' << PALMD::AssemblerDirective << PALMetadataString << '\n';
  return true;
}

//===----------------------------------------------------------------------===//
// AMDGPUTargetELFStreamer
//===----------------------------------------------------------------------===//

AMDGPUTargetELFStreamer::AMDGPUTargetELFStreamer(
    MCStreamer &S, const MCSubtargetInfo &STI)
    : AMDGPUTargetStreamer(S), Streamer(S) {
  MCAssembler &MCA = getStreamer().getAssembler();
  unsigned EFlags = MCA.getELFHeaderEFlags();

  EFlags &= ~ELF::EF_AMDGPU_MACH;
  EFlags |= getMACH(STI.getCPU());

  EFlags &= ~ELF::EF_AMDGPU_XNACK;
  if (AMDGPU::hasXNACK(STI))
    EFlags |= ELF::EF_AMDGPU_XNACK;

  MCA.setELFHeaderEFlags(EFlags);
}

MCELFStreamer &AMDGPUTargetELFStreamer::getStreamer() {
  return static_cast<MCELFStreamer &>(Streamer);
}

void AMDGPUTargetELFStreamer::EmitAMDGPUNote(
    const MCExpr *DescSZ, unsigned NoteType,
    function_ref<void(MCELFStreamer &)> EmitDesc) {
  auto &S = getStreamer();
  auto &Context = S.getContext();

  auto NameSZ = sizeof(ElfNote::NoteName);

  S.PushSection();
  S.SwitchSection(Context.getELFSection(
    ElfNote::SectionName, ELF::SHT_NOTE, ELF::SHF_ALLOC));
  S.EmitIntValue(NameSZ, 4);                                  // namesz
  S.EmitValue(DescSZ, 4);                                     // descz
  S.EmitIntValue(NoteType, 4);                                // type
  S.EmitBytes(StringRef(ElfNote::NoteName, NameSZ));          // name
  S.EmitValueToAlignment(4, 0, 1, 0);                         // padding 0
  EmitDesc(S);                                                // desc
  S.EmitValueToAlignment(4, 0, 1, 0);                         // padding 0
  S.PopSection();
}

void
AMDGPUTargetELFStreamer::EmitDirectiveHSACodeObjectVersion(uint32_t Major,
                                                           uint32_t Minor) {

  EmitAMDGPUNote(
    MCConstantExpr::create(8, getContext()),
    ElfNote::NT_AMDGPU_HSA_CODE_OBJECT_VERSION,
    [&](MCELFStreamer &OS){
      OS.EmitIntValue(Major, 4);
      OS.EmitIntValue(Minor, 4);
    }
  );
}

void
AMDGPUTargetELFStreamer::EmitDirectiveHSACodeObjectISA(uint32_t Major,
                                                       uint32_t Minor,
                                                       uint32_t Stepping,
                                                       StringRef VendorName,
                                                       StringRef ArchName) {
  uint16_t VendorNameSize = VendorName.size() + 1;
  uint16_t ArchNameSize = ArchName.size() + 1;

  unsigned DescSZ = sizeof(VendorNameSize) + sizeof(ArchNameSize) +
    sizeof(Major) + sizeof(Minor) + sizeof(Stepping) +
    VendorNameSize + ArchNameSize;

  EmitAMDGPUNote(
    MCConstantExpr::create(DescSZ, getContext()),
    ElfNote::NT_AMDGPU_HSA_ISA,
    [&](MCELFStreamer &OS) {
      OS.EmitIntValue(VendorNameSize, 2);
      OS.EmitIntValue(ArchNameSize, 2);
      OS.EmitIntValue(Major, 4);
      OS.EmitIntValue(Minor, 4);
      OS.EmitIntValue(Stepping, 4);
      OS.EmitBytes(VendorName);
      OS.EmitIntValue(0, 1); // NULL terminate VendorName
      OS.EmitBytes(ArchName);
      OS.EmitIntValue(0, 1); // NULL terminte ArchName
    }
  );
}

void
AMDGPUTargetELFStreamer::EmitAMDKernelCodeT(const amd_kernel_code_t &Header) {

  MCStreamer &OS = getStreamer();
  OS.PushSection();
  OS.EmitBytes(StringRef((const char*)&Header, sizeof(Header)));
  OS.PopSection();
}

void AMDGPUTargetELFStreamer::EmitAMDGPUSymbolType(StringRef SymbolName,
                                                   unsigned Type) {
  MCSymbolELF *Symbol = cast<MCSymbolELF>(
      getStreamer().getContext().getOrCreateSymbol(SymbolName));
  Symbol->setType(ELF::STT_AMDGPU_HSA_KERNEL);
}

bool AMDGPUTargetELFStreamer::EmitISAVersion(StringRef IsaVersionString) {
  // Create two labels to mark the beginning and end of the desc field
  // and a MCExpr to calculate the size of the desc field.
  auto &Context = getContext();
  auto *DescBegin = Context.createTempSymbol();
  auto *DescEnd = Context.createTempSymbol();
  auto *DescSZ = MCBinaryExpr::createSub(
    MCSymbolRefExpr::create(DescEnd, Context),
    MCSymbolRefExpr::create(DescBegin, Context), Context);

  EmitAMDGPUNote(
    DescSZ,
    ELF::NT_AMD_AMDGPU_ISA,
    [&](MCELFStreamer &OS) {
      OS.EmitLabel(DescBegin);
      OS.EmitBytes(IsaVersionString);
      OS.EmitLabel(DescEnd);
    }
  );
  return true;
}

bool AMDGPUTargetELFStreamer::EmitHSAMetadata(
    const AMDGPU::HSAMD::Metadata &HSAMetadata) {
  std::string HSAMetadataString;
  if (HSAMD::toString(HSAMetadata, HSAMetadataString))
    return false;

  // Create two labels to mark the beginning and end of the desc field
  // and a MCExpr to calculate the size of the desc field.
  auto &Context = getContext();
  auto *DescBegin = Context.createTempSymbol();
  auto *DescEnd = Context.createTempSymbol();
  auto *DescSZ = MCBinaryExpr::createSub(
    MCSymbolRefExpr::create(DescEnd, Context),
    MCSymbolRefExpr::create(DescBegin, Context), Context);

  EmitAMDGPUNote(
    DescSZ,
    ELF::NT_AMD_AMDGPU_HSA_METADATA,
    [&](MCELFStreamer &OS) {
      OS.EmitLabel(DescBegin);
      OS.EmitBytes(HSAMetadataString);
      OS.EmitLabel(DescEnd);
    }
  );
  return true;
}

bool AMDGPUTargetELFStreamer::EmitPALMetadata(
    const PALMD::Metadata &PALMetadata) {
  EmitAMDGPUNote(
    MCConstantExpr::create(PALMetadata.size() * sizeof(uint32_t), getContext()),
    ELF::NT_AMD_AMDGPU_PAL_METADATA,
    [&](MCELFStreamer &OS){
      for (auto I : PALMetadata)
        OS.EmitIntValue(I, sizeof(uint32_t));
    }
  );
  return true;
}
