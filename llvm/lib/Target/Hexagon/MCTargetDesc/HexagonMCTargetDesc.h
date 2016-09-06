//===-- HexagonMCTargetDesc.h - Hexagon Target Descriptions -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Hexagon specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HEXAGON_MCTARGETDESC_HEXAGONMCTARGETDESC_H
#define LLVM_LIB_TARGET_HEXAGON_MCTARGETDESC_HEXAGONMCTARGETDESC_H

#include <cstdint>

#include "llvm/Support/CommandLine.h"

namespace llvm {
struct InstrItinerary;
struct InstrStage;
class MCAsmBackend;
class MCCodeEmitter;
class MCContext;
class MCInstrInfo;
class MCObjectWriter;
class MCRegisterInfo;
class MCSubtargetInfo;
class MCTargetOptions;
class Target;
class Triple;
class StringRef;
class raw_ostream;
class raw_pwrite_stream;

extern Target TheHexagonTarget;
extern cl::opt<bool> HexagonDisableCompound;
extern cl::opt<bool> HexagonDisableDuplex;
extern const InstrStage HexagonStages[];

MCInstrInfo *createHexagonMCInstrInfo();

MCCodeEmitter *createHexagonMCCodeEmitter(const MCInstrInfo &MCII,
                                          const MCRegisterInfo &MRI,
                                          MCContext &MCT);

MCAsmBackend *createHexagonAsmBackend(const Target &T,
                                      const MCRegisterInfo &MRI,
                                      const Triple &TT, StringRef CPU,
                                      const MCTargetOptions &Options);

MCObjectWriter *createHexagonELFObjectWriter(raw_pwrite_stream &OS,
                                             uint8_t OSABI, StringRef CPU);

namespace HEXAGON_MC {
  StringRef selectHexagonCPU(const Triple &TT, StringRef CPU);
}

} // End llvm namespace

// Define symbolic names for Hexagon registers.  This defines a mapping from
// register name to register number.
//
#define GET_REGINFO_ENUM
#include "HexagonGenRegisterInfo.inc"

// Defines symbolic names for the Hexagon instructions.
//
#define GET_INSTRINFO_ENUM
#include "HexagonGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "HexagonGenSubtargetInfo.inc"

#endif
