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

#include "AMDGPU.h"
#include "AMDGPUTargetStreamer.h"
#include "SIDefines.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "Utils/AMDKernelCodeTUtils.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/FormattedStream.h"
#include "AMDGPURuntimeMD.h"

namespace llvm {
#include "AMDGPUPTNote.h"
}

using namespace llvm;
using namespace llvm::AMDGPU;

AMDGPUTargetStreamer::AMDGPUTargetStreamer(MCStreamer &S)
    : MCTargetStreamer(S) {}

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

void AMDGPUTargetAsmStreamer::EmitAMDGPUHsaModuleScopeGlobal(
    StringRef GlobalName) {
  OS << "\t.amdgpu_hsa_module_global " << GlobalName << '\n';
}

void AMDGPUTargetAsmStreamer::EmitAMDGPUHsaProgramScopeGlobal(
    StringRef GlobalName) {
  OS << "\t.amdgpu_hsa_program_global " << GlobalName << '\n';
}

//===----------------------------------------------------------------------===//
// AMDGPUTargetELFStreamer
//===----------------------------------------------------------------------===//

AMDGPUTargetELFStreamer::AMDGPUTargetELFStreamer(MCStreamer &S)
    : AMDGPUTargetStreamer(S), Streamer(S) {}

MCELFStreamer &AMDGPUTargetELFStreamer::getStreamer() {
  return static_cast<MCELFStreamer &>(Streamer);
}

void
AMDGPUTargetELFStreamer::EmitDirectiveHSACodeObjectVersion(uint32_t Major,
                                                           uint32_t Minor) {
  MCStreamer &OS = getStreamer();
  MCSectionELF *Note =
      OS.getContext().getELFSection(PT_NOTE::SectionName, ELF::SHT_NOTE,
                                    ELF::SHF_ALLOC);

  auto NameSZ = sizeof(PT_NOTE::NoteName);
  OS.PushSection();
  OS.SwitchSection(Note);
  OS.EmitIntValue(NameSZ, 4);                                     // namesz
  OS.EmitIntValue(8, 4);                                          // descz
  OS.EmitIntValue(PT_NOTE::NT_AMDGPU_HSA_CODE_OBJECT_VERSION, 4); // type
  OS.EmitBytes(StringRef(PT_NOTE::NoteName, NameSZ));             // name
  OS.EmitValueToAlignment(4);
  OS.EmitIntValue(Major, 4);                                      // desc
  OS.EmitIntValue(Minor, 4);
  OS.EmitValueToAlignment(4);
  OS.PopSection();
}

void
AMDGPUTargetELFStreamer::EmitDirectiveHSACodeObjectISA(uint32_t Major,
                                                       uint32_t Minor,
                                                       uint32_t Stepping,
                                                       StringRef VendorName,
                                                       StringRef ArchName) {
  MCStreamer &OS = getStreamer();
  MCSectionELF *Note =
      OS.getContext().getELFSection(PT_NOTE::SectionName, ELF::SHT_NOTE,
                                    ELF::SHF_ALLOC);

  uint16_t VendorNameSize = VendorName.size() + 1;
  uint16_t ArchNameSize = ArchName.size() + 1;
  unsigned DescSZ = sizeof(VendorNameSize) + sizeof(ArchNameSize) +
                    sizeof(Major) + sizeof(Minor) + sizeof(Stepping) +
                    VendorNameSize + ArchNameSize;

  OS.PushSection();
  OS.SwitchSection(Note);
  auto NameSZ = sizeof(PT_NOTE::NoteName);
  OS.EmitIntValue(NameSZ, 4);                              // namesz
  OS.EmitIntValue(DescSZ, 4);                              // descsz
  OS.EmitIntValue(PT_NOTE::NT_AMDGPU_HSA_ISA, 4);          // type
  OS.EmitBytes(StringRef(PT_NOTE::NoteName, NameSZ));      // name
  OS.EmitValueToAlignment(4);
  OS.EmitIntValue(VendorNameSize, 2);                      // desc
  OS.EmitIntValue(ArchNameSize, 2);
  OS.EmitIntValue(Major, 4);
  OS.EmitIntValue(Minor, 4);
  OS.EmitIntValue(Stepping, 4);
  OS.EmitBytes(VendorName);
  OS.EmitIntValue(0, 1); // NULL terminate VendorName
  OS.EmitBytes(ArchName);
  OS.EmitIntValue(0, 1); // NULL terminte ArchName
  OS.EmitValueToAlignment(4);
  OS.PopSection();
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

void AMDGPUTargetELFStreamer::EmitAMDGPUHsaModuleScopeGlobal(
    StringRef GlobalName) {

  MCSymbolELF *Symbol = cast<MCSymbolELF>(
      getStreamer().getContext().getOrCreateSymbol(GlobalName));
  Symbol->setType(ELF::STT_OBJECT);
  Symbol->setBinding(ELF::STB_LOCAL);
}

void AMDGPUTargetELFStreamer::EmitAMDGPUHsaProgramScopeGlobal(
    StringRef GlobalName) {

  MCSymbolELF *Symbol = cast<MCSymbolELF>(
      getStreamer().getContext().getOrCreateSymbol(GlobalName));
  Symbol->setType(ELF::STT_OBJECT);
  Symbol->setBinding(ELF::STB_GLOBAL);
}

void AMDGPUTargetELFStreamer::emitRuntimeMetadata(Module &M) {
  auto &S = getStreamer();
  auto &Context = S.getContext();

  auto NameSZ = sizeof(PT_NOTE::NoteName); // Size of note name including trailing null.

  S.PushSection();
  S.SwitchSection(Context.getELFSection(
      PT_NOTE::SectionName, ELF::SHT_NOTE, ELF::SHF_ALLOC));

  // Create two labels to mark the beginning and end of the desc field
  // and a MCExpr to calculate the size of the desc field.
  auto *DescBegin = Context.createTempSymbol();
  auto *DescEnd = Context.createTempSymbol();
  auto *DescSZ = MCBinaryExpr::createSub(
      MCSymbolRefExpr::create(DescEnd, Context),
      MCSymbolRefExpr::create(DescBegin, Context), Context);

  // Emit the note element for runtime metadata.
  // Name and desc should be padded to 4 byte boundary but size of name and
  // desc should not include padding 0's.
  S.EmitIntValue(NameSZ, 4);                                  // namesz
  S.EmitValue(DescSZ, 4);                                     // descz
  S.EmitIntValue(PT_NOTE::NT_AMDGPU_HSA_RUNTIME_METADATA, 4); // type
  S.EmitBytes(StringRef(PT_NOTE::NoteName, NameSZ));          // name
  S.EmitValueToAlignment(4, 0, 1, 0);                         // padding 0
  S.EmitLabel(DescBegin);
  S.EmitBytes(getRuntimeMDYAMLString(M));                               // desc
  S.EmitLabel(DescEnd);
  S.EmitValueToAlignment(4, 0, 1, 0);                         // padding 0
  S.PopSection();
}
