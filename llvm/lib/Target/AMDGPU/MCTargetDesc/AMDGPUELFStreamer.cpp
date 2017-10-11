//===-------- AMDGPUELFStreamer.cpp - ELF Object Output -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUELFStreamer.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmBackend.h"

using namespace llvm;

AMDGPUELFStreamer::AMDGPUELFStreamer(const Triple &T, MCContext &Context,
                                     std::unique_ptr<MCAsmBackend> MAB,
                                     raw_pwrite_stream &OS,
                                     MCCodeEmitter *Emitter)
    : MCELFStreamer(Context, std::move(MAB), OS, Emitter) {
  unsigned Arch = ELF::EF_AMDGPU_ARCH_NONE;
  switch (T.getArch()) {
  case Triple::r600:
    Arch = ELF::EF_AMDGPU_ARCH_R600;
    break;
  case Triple::amdgcn:
    Arch = ELF::EF_AMDGPU_ARCH_GCN;
    break;
  default:
    break;
  }

  MCAssembler &MCA = getAssembler();
  unsigned EFlags = MCA.getELFHeaderEFlags();
  EFlags &= ~ELF::EF_AMDGPU_ARCH;
  EFlags |= Arch;
  MCA.setELFHeaderEFlags(EFlags);
}

MCELFStreamer *llvm::createAMDGPUELFStreamer(
    const Triple &T, MCContext &Context, std::unique_ptr<MCAsmBackend> MAB,
    raw_pwrite_stream &OS, MCCodeEmitter *Emitter, bool RelaxAll) {
  return new AMDGPUELFStreamer(T, Context, std::move(MAB), OS, Emitter);
}
