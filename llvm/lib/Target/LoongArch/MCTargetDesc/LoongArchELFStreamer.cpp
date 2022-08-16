//===-- LoongArchELFStreamer.cpp - LoongArch ELF Target Streamer Methods --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides LoongArch specific target streamer methods.
//
//===----------------------------------------------------------------------===//

#include "LoongArchELFStreamer.h"
#include "LoongArchAsmBackend.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCObjectWriter.h"

using namespace llvm;

// This part is for ELF object output.
LoongArchTargetELFStreamer::LoongArchTargetELFStreamer(
    MCStreamer &S, const MCSubtargetInfo &STI)
    : LoongArchTargetStreamer(S) {
  // FIXME: select appropriate ABI.
  setTargetABI(STI.getTargetTriple().isArch64Bit() ? LoongArchABI::ABI_LP64D
                                                   : LoongArchABI::ABI_ILP32D);
}

MCELFStreamer &LoongArchTargetELFStreamer::getStreamer() {
  return static_cast<MCELFStreamer &>(Streamer);
}

void LoongArchTargetELFStreamer::finish() {
  LoongArchTargetStreamer::finish();
  MCAssembler &MCA = getStreamer().getAssembler();
  LoongArchABI::ABI ABI = getTargetABI();

  // FIXME:
  // There are several PRs [1][2][3] that may affect the e_flags.
  // After they got closed or merged, we should update the implementation here
  // accordingly.
  //
  // [1] https://github.com/loongson/LoongArch-Documentation/pull/33
  // [2] https://github.com/loongson/LoongArch-Documentation/pull/47
  // [2] https://github.com/loongson/LoongArch-Documentation/pull/61
  unsigned EFlags = MCA.getELFHeaderEFlags();
  switch (ABI) {
  case LoongArchABI::ABI_ILP32S:
    EFlags |= ELF::EF_LOONGARCH_BASE_ABI_ILP32S;
    break;
  case LoongArchABI::ABI_ILP32F:
    EFlags |= ELF::EF_LOONGARCH_BASE_ABI_ILP32F;
    break;
  case LoongArchABI::ABI_ILP32D:
    EFlags |= ELF::EF_LOONGARCH_BASE_ABI_ILP32D;
    break;
  case LoongArchABI::ABI_LP64S:
    EFlags |= ELF::EF_LOONGARCH_BASE_ABI_LP64S;
    break;
  case LoongArchABI::ABI_LP64F:
    EFlags |= ELF::EF_LOONGARCH_BASE_ABI_LP64F;
    break;
  case LoongArchABI::ABI_LP64D:
    EFlags |= ELF::EF_LOONGARCH_BASE_ABI_LP64D;
    break;
  case LoongArchABI::ABI_Unknown:
    llvm_unreachable("Improperly initialized target ABI");
  }
  MCA.setELFHeaderEFlags(EFlags);
}

namespace {
class LoongArchELFStreamer : public MCELFStreamer {
public:
  LoongArchELFStreamer(MCContext &C, std::unique_ptr<MCAsmBackend> MAB,
                       std::unique_ptr<MCObjectWriter> MOW,
                       std::unique_ptr<MCCodeEmitter> MCE)
      : MCELFStreamer(C, std::move(MAB), std::move(MOW), std::move(MCE)) {}
};
} // end namespace

namespace llvm {
MCELFStreamer *createLoongArchELFStreamer(MCContext &C,
                                          std::unique_ptr<MCAsmBackend> MAB,
                                          std::unique_ptr<MCObjectWriter> MOW,
                                          std::unique_ptr<MCCodeEmitter> MCE,
                                          bool RelaxAll) {
  LoongArchELFStreamer *S = new LoongArchELFStreamer(
      C, std::move(MAB), std::move(MOW), std::move(MCE));
  S->getAssembler().setRelaxAll(RelaxAll);
  return S;
}
} // end namespace llvm
