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
#include <string>

namespace llvm {

class formatted_raw_ostream;
class MCAsmBackend;
class MCCodeEmitter;
class MCContext;
class MCInstrInfo;
class MCRegisterInfo;
class MCObjectWriter;
class MCStreamer;
class MCSubtargetInfo;
class MCTargetStreamer;
class StringRef;
class Target;
class Triple;
class raw_ostream;

extern Target TheWebAssemblyTarget32;
extern Target TheWebAssemblyTarget64;

MCAsmBackend *createWebAssemblyAsmBackend(const Target &T,
                                          const MCRegisterInfo &MRI,
                                          StringRef TT, StringRef CPU);

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
