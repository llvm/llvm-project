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

static const struct {
  const char *Name;
  unsigned Mach;
} MachTable[] = {
      // Radeon HD 2000/3000 Series (R600).
      { "r600", ELF::EF_AMDGPU_MACH_R600_R600 },
      { "r630", ELF::EF_AMDGPU_MACH_R600_R630 },
      { "rs880", ELF::EF_AMDGPU_MACH_R600_RS880 },
      { "rv670", ELF::EF_AMDGPU_MACH_R600_RV670 },
      // Radeon HD 4000 Series (R700).
      { "rv710", ELF::EF_AMDGPU_MACH_R600_RV710 },
      { "rv730", ELF::EF_AMDGPU_MACH_R600_RV730 },
      { "rv770", ELF::EF_AMDGPU_MACH_R600_RV770 },
      // Radeon HD 5000 Series (Evergreen).
      { "cedar", ELF::EF_AMDGPU_MACH_R600_CEDAR },
      { "cypress", ELF::EF_AMDGPU_MACH_R600_CYPRESS },
      { "juniper", ELF::EF_AMDGPU_MACH_R600_JUNIPER },
      { "redwood", ELF::EF_AMDGPU_MACH_R600_REDWOOD },
      { "sumo", ELF::EF_AMDGPU_MACH_R600_SUMO },
      // Radeon HD 6000 Series (Northern Islands).
      { "barts", ELF::EF_AMDGPU_MACH_R600_BARTS },
      { "caicos", ELF::EF_AMDGPU_MACH_R600_CAICOS },
      { "cayman", ELF::EF_AMDGPU_MACH_R600_CAYMAN },
      { "turks", ELF::EF_AMDGPU_MACH_R600_TURKS },
      // AMDGCN GFX6.
      { "gfx600", ELF::EF_AMDGPU_MACH_AMDGCN_GFX600 },
      { "tahiti", ELF::EF_AMDGPU_MACH_AMDGCN_GFX600 },
      { "gfx601", ELF::EF_AMDGPU_MACH_AMDGCN_GFX601 },
      { "hainan", ELF::EF_AMDGPU_MACH_AMDGCN_GFX601 },
      { "oland", ELF::EF_AMDGPU_MACH_AMDGCN_GFX601 },
      { "pitcairn", ELF::EF_AMDGPU_MACH_AMDGCN_GFX601 },
      { "verde", ELF::EF_AMDGPU_MACH_AMDGCN_GFX601 },
      // AMDGCN GFX7.
      { "gfx700", ELF::EF_AMDGPU_MACH_AMDGCN_GFX700 },
      { "kaveri", ELF::EF_AMDGPU_MACH_AMDGCN_GFX700 },
      { "gfx701", ELF::EF_AMDGPU_MACH_AMDGCN_GFX701 },
      { "hawaii", ELF::EF_AMDGPU_MACH_AMDGCN_GFX701 },
      { "gfx702", ELF::EF_AMDGPU_MACH_AMDGCN_GFX702 },
      { "gfx703", ELF::EF_AMDGPU_MACH_AMDGCN_GFX703 },
      { "kabini", ELF::EF_AMDGPU_MACH_AMDGCN_GFX703 },
      { "mullins", ELF::EF_AMDGPU_MACH_AMDGCN_GFX703 },
      { "gfx704", ELF::EF_AMDGPU_MACH_AMDGCN_GFX704 },
      { "bonaire", ELF::EF_AMDGPU_MACH_AMDGCN_GFX704 },
      // AMDGCN GFX8.
      { "gfx801", ELF::EF_AMDGPU_MACH_AMDGCN_GFX801 },
      { "carrizo", ELF::EF_AMDGPU_MACH_AMDGCN_GFX801 },
      { "gfx802", ELF::EF_AMDGPU_MACH_AMDGCN_GFX802 },
      { "iceland", ELF::EF_AMDGPU_MACH_AMDGCN_GFX802 },
      { "tonga", ELF::EF_AMDGPU_MACH_AMDGCN_GFX802 },
      { "gfx803", ELF::EF_AMDGPU_MACH_AMDGCN_GFX803 },
      { "fiji", ELF::EF_AMDGPU_MACH_AMDGCN_GFX803 },
      { "polaris10", ELF::EF_AMDGPU_MACH_AMDGCN_GFX803 },
      { "polaris11", ELF::EF_AMDGPU_MACH_AMDGCN_GFX803 },
      { "gfx810", ELF::EF_AMDGPU_MACH_AMDGCN_GFX810 },
      { "stoney", ELF::EF_AMDGPU_MACH_AMDGCN_GFX810 },
      // AMDGCN GFX9.
      { "gfx900", ELF::EF_AMDGPU_MACH_AMDGCN_GFX900 },
      { "gfx902", ELF::EF_AMDGPU_MACH_AMDGCN_GFX902 },
      { "gfx904", ELF::EF_AMDGPU_MACH_AMDGCN_GFX904 },
      { "gfx906", ELF::EF_AMDGPU_MACH_AMDGCN_GFX906 },
      // Not specified processor.
      { nullptr, ELF::EF_AMDGPU_MACH_NONE }
};

unsigned AMDGPUTargetStreamer::getMACH(StringRef GPU) const {
  auto Entry = MachTable;
  for (; Entry->Name && GPU != Entry->Name; ++Entry)
    ;
  return Entry->Mach;
}

const char *AMDGPUTargetStreamer::getMachName(unsigned Mach) {
  auto Entry = MachTable;
  for (; Entry->Name && Mach != Entry->Mach; ++Entry)
    ;
  return Entry->Name;
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
  Symbol->setType(Type);
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
