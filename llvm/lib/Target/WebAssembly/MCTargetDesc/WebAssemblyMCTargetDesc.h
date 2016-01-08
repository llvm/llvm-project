//==- WebAssemblyMCTargetDesc.h - WebAssembly Target Descriptions -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides WebAssembly-specific target descriptions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_WEBASSEMBLY_MCTARGETDESC_WEBASSEMBLYMCTARGETDESC_H
#define LLVM_LIB_TARGET_WEBASSEMBLY_MCTARGETDESC_WEBASSEMBLYMCTARGETDESC_H

#include "llvm/Support/DataTypes.h"

namespace llvm {

class MCAsmBackend;
class MCCodeEmitter;
class MCContext;
class MCInstrInfo;
class MCObjectWriter;
class MCSubtargetInfo;
class Target;
class Triple;
class raw_pwrite_stream;

extern Target TheWebAssemblyTarget32;
extern Target TheWebAssemblyTarget64;

MCCodeEmitter *createWebAssemblyMCCodeEmitter(const MCInstrInfo &MCII,
                                              MCContext &Ctx);

MCAsmBackend *createWebAssemblyAsmBackend(const Triple &TT);

MCObjectWriter *createWebAssemblyELFObjectWriter(raw_pwrite_stream &OS,
                                                 bool Is64Bit, uint8_t OSABI);

} // end namespace llvm

// Defines symbolic names for WebAssembly registers. This defines a mapping from
// register name to register number.
//
#define GET_REGINFO_ENUM
#include "WebAssemblyGenRegisterInfo.inc"

// Defines symbolic names for the WebAssembly instructions.
//
#define GET_INSTRINFO_ENUM
#include "WebAssemblyGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "WebAssemblyGenSubtargetInfo.inc"

#endif
