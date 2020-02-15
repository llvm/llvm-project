//===-- MSP430ELFStreamer.cpp - MSP430 ELF Target Streamer Methods --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides MSP430 specific target streamer methods.
//
//===----------------------------------------------------------------------===//

#include "MSP430MCTargetDesc.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"

using namespace llvm;

namespace llvm {

class MSP430TargetELFStreamer : public MCTargetStreamer {
public:
  MCELFStreamer &getStreamer();
  MSP430TargetELFStreamer(MCStreamer &S, const MCSubtargetInfo &STI);
};

// This part is for ELF object output.
MSP430TargetELFStreamer::MSP430TargetELFStreamer(MCStreamer &S,
                                                 const MCSubtargetInfo &STI)
    : MCTargetStreamer(S) {
  MCAssembler &MCA = getStreamer().getAssembler();
  unsigned EFlags = MCA.getELFHeaderEFlags();
  MCA.setELFHeaderEFlags(EFlags);

  // Emit build attributes section according to
  // MSP430 EABI (slaa534.pdf, part 13).
  MCSection *AttributeSection = getStreamer().getContext().getELFSection(
      ".MSP430.attributes", ELF::SHT_MSP430_ATTRIBUTES, 0);
  Streamer.SwitchSection(AttributeSection);

  // Format version.
  Streamer.emitIntValue(0x41, 1);
  // Subsection length.
  Streamer.emitIntValue(22, 4);
  // Vendor name string, zero-terminated.
  Streamer.emitBytes("mspabi");
  Streamer.emitIntValue(0, 1);

  // Attribute vector scope tag. 1 stands for the entire file.
  Streamer.emitIntValue(1, 1);
  // Attribute vector length.
  Streamer.emitIntValue(11, 4);
  // OFBA_MSPABI_Tag_ISA(4) = 1, MSP430
  Streamer.emitIntValue(4, 1);
  Streamer.emitIntValue(1, 1);
  // OFBA_MSPABI_Tag_Code_Model(6) = 1, Small
  Streamer.emitIntValue(6, 1);
  Streamer.emitIntValue(1, 1);
  // OFBA_MSPABI_Tag_Data_Model(8) = 1, Small
  Streamer.emitIntValue(8, 1);
  Streamer.emitIntValue(1, 1);
}

MCELFStreamer &MSP430TargetELFStreamer::getStreamer() {
  return static_cast<MCELFStreamer &>(Streamer);
}

MCTargetStreamer *
createMSP430ObjectTargetStreamer(MCStreamer &S, const MCSubtargetInfo &STI) {
  const Triple &TT = STI.getTargetTriple();
  if (TT.isOSBinFormatELF())
    return new MSP430TargetELFStreamer(S, STI);
  return nullptr;
}

} // namespace llvm
