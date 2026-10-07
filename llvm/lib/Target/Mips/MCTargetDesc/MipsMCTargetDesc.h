//===-- MipsMCTargetDesc.h - Mips Target Descriptions -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides Mips specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MIPS_MCTARGETDESC_MIPSMCTARGETDESC_H
#define LLVM_LIB_TARGET_MIPS_MCTARGETDESC_MIPSMCTARGETDESC_H

#include "llvm/MC/MCRegister.h"
#include "llvm/Support/DataTypes.h"

#include <memory>

namespace llvm {
class MCAsmBackend;
class MCCodeEmitter;
class MCContext;
class MCInstrInfo;
class MCObjectTargetWriter;
class MCObjectWriter;
class MCRegister;
class MCRegisterInfo;
class MCStreamer;
class MCSubtargetInfo;
class MCTargetOptions;
class StringRef;
class Target;
class Triple;

MCCodeEmitter *createMipsMCCodeEmitterEB(const MCInstrInfo &MCII,
                                         MCContext &Ctx);
MCCodeEmitter *createMipsMCCodeEmitterEL(const MCInstrInfo &MCII,
                                         MCContext &Ctx);

MCAsmBackend *createMipsAsmBackend(const Target &T, const MCSubtargetInfo &STI,
                                   const MCRegisterInfo &MRI,
                                   const MCTargetOptions &Options);

/// Construct an MIPS Windows COFF machine code streamer which will generate
/// PE/COFF format object files.
///
/// Takes ownership of \p AB and \p CE.
MCStreamer *createMipsWinCOFFStreamer(MCContext &C,
                                      std::unique_ptr<MCAsmBackend> &&AB,
                                      std::unique_ptr<MCObjectWriter> &&OW,
                                      std::unique_ptr<MCCodeEmitter> &&CE);

/// Construct a Mips ELF object writer.
std::unique_ptr<MCObjectTargetWriter>
createMipsELFObjectWriter(const Triple &TT, bool IsN32);
/// Construct a Mips Win COFF object writer.
std::unique_ptr<MCObjectTargetWriter> createMipsWinCOFFObjectWriter();

namespace MIPS_MC {
void initLLVMToCVRegMapping(MCRegisterInfo *MRI);

StringRef selectMipsCPU(const Triple &TT, StringRef CPU);

/// Match a symbolic name in RegClassID, or return an invalid register.
MCRegister matchRegisterName(StringRef Name, const MCRegisterInfo &MRI,
                             unsigned RegClassID, unsigned AltIdx);

/// Return a GPR name's hardware index, or -1 if unknown.
int getCPURegisterIndex(StringRef Name, const MCRegisterInfo &MRI,
                        unsigned AltIdx, bool *IsDeprecated = nullptr);
}

} // End llvm namespace

// Defines symbolic names for Mips registers.  This defines a mapping from
// register name to register number.
#define GET_REGINFO_ENUM
#include "MipsGenRegisterInfo.inc"

namespace llvm {
namespace Mips {
// R2-R3 and R8-R15 have different names in the O32 (O), N32/N64 (N) and
// P32/P64 (P) ABIs. These aliases match the ones in MipsRegisterInfo.td; the
// unsuffixed O names are kept for compatibility. The remaining O32-role
// aliases (ZERO, AT, A0-A3, S0-S7, T8-T9, K0-K1, GP, SP, FP, RA) below are
// to be removed in favour of MipsABIInfo accessors or plain R<n> names.

// O32 names for R2-R3/R8-R15.
inline constexpr MCPhysReg V0 = R2;
inline constexpr MCPhysReg V1 = R3;
inline constexpr MCPhysReg T0 = R8;
inline constexpr MCPhysReg T1 = R9;
inline constexpr MCPhysReg T2 = R10;
inline constexpr MCPhysReg T3 = R11;
inline constexpr MCPhysReg T4 = R12;
inline constexpr MCPhysReg T5 = R13;
inline constexpr MCPhysReg T6 = R14;
inline constexpr MCPhysReg T7 = R15;
inline constexpr MCPhysReg V0_64 = R2_64;
inline constexpr MCPhysReg V1_64 = R3_64;
inline constexpr MCPhysReg T0_64 = R8_64;
inline constexpr MCPhysReg T1_64 = R9_64;
inline constexpr MCPhysReg T2_64 = R10_64;
inline constexpr MCPhysReg T3_64 = R11_64;
inline constexpr MCPhysReg T4_64 = R12_64;
inline constexpr MCPhysReg T5_64 = R13_64;
inline constexpr MCPhysReg T6_64 = R14_64;
inline constexpr MCPhysReg T7_64 = R15_64;

// N32/N64 names.
inline constexpr MCPhysReg V0_N = R2;
inline constexpr MCPhysReg V1_N = R3;
inline constexpr MCPhysReg A4_N = R8;
inline constexpr MCPhysReg A5_N = R9;
inline constexpr MCPhysReg A6_N = R10;
inline constexpr MCPhysReg A7_N = R11;
inline constexpr MCPhysReg T0_N = R12;
inline constexpr MCPhysReg T1_N = R13;
inline constexpr MCPhysReg T2_N = R14;
inline constexpr MCPhysReg T3_N = R15;
inline constexpr MCPhysReg V0_N_64 = R2_64;
inline constexpr MCPhysReg V1_N_64 = R3_64;
inline constexpr MCPhysReg A4_N_64 = R8_64;
inline constexpr MCPhysReg A5_N_64 = R9_64;
inline constexpr MCPhysReg A6_N_64 = R10_64;
inline constexpr MCPhysReg A7_N_64 = R11_64;
inline constexpr MCPhysReg T0_N_64 = R12_64;
inline constexpr MCPhysReg T1_N_64 = R13_64;
inline constexpr MCPhysReg T2_N_64 = R14_64;
inline constexpr MCPhysReg T3_N_64 = R15_64;

// P32/P64 names.
inline constexpr MCPhysReg T4_P = R2;
inline constexpr MCPhysReg T5_P = R3;
inline constexpr MCPhysReg A4_P = R8;
inline constexpr MCPhysReg A5_P = R9;
inline constexpr MCPhysReg A6_P = R10;
inline constexpr MCPhysReg A7_P = R11;
inline constexpr MCPhysReg T0_P = R12;
inline constexpr MCPhysReg T1_P = R13;
inline constexpr MCPhysReg T2_P = R14;
inline constexpr MCPhysReg T3_P = R15;
inline constexpr MCPhysReg T4_P_64 = R2_64;
inline constexpr MCPhysReg T5_P_64 = R3_64;
inline constexpr MCPhysReg A4_P_64 = R8_64;
inline constexpr MCPhysReg A5_P_64 = R9_64;
inline constexpr MCPhysReg A6_P_64 = R10_64;
inline constexpr MCPhysReg A7_P_64 = R11_64;
inline constexpr MCPhysReg T0_P_64 = R12_64;
inline constexpr MCPhysReg T1_P_64 = R13_64;
inline constexpr MCPhysReg T2_P_64 = R14_64;
inline constexpr MCPhysReg T3_P_64 = R15_64;

// O32-role names for the remaining registers.
inline constexpr MCPhysReg ZERO = R0;
inline constexpr MCPhysReg AT = R1;
inline constexpr MCPhysReg A0 = R4;
inline constexpr MCPhysReg A1 = R5;
inline constexpr MCPhysReg A2 = R6;
inline constexpr MCPhysReg A3 = R7;
inline constexpr MCPhysReg S0 = R16;
inline constexpr MCPhysReg S1 = R17;
inline constexpr MCPhysReg S2 = R18;
inline constexpr MCPhysReg S3 = R19;
inline constexpr MCPhysReg S4 = R20;
inline constexpr MCPhysReg S5 = R21;
inline constexpr MCPhysReg S6 = R22;
inline constexpr MCPhysReg S7 = R23;
inline constexpr MCPhysReg T8 = R24;
inline constexpr MCPhysReg T9 = R25;
inline constexpr MCPhysReg K0 = R26;
inline constexpr MCPhysReg K1 = R27;
inline constexpr MCPhysReg GP = R28;
inline constexpr MCPhysReg SP = R29;
inline constexpr MCPhysReg FP = R30;
inline constexpr MCPhysReg RA = R31;
inline constexpr MCPhysReg ZERO_64 = R0_64;
inline constexpr MCPhysReg AT_64 = R1_64;
inline constexpr MCPhysReg A0_64 = R4_64;
inline constexpr MCPhysReg A1_64 = R5_64;
inline constexpr MCPhysReg A2_64 = R6_64;
inline constexpr MCPhysReg A3_64 = R7_64;
inline constexpr MCPhysReg S0_64 = R16_64;
inline constexpr MCPhysReg S1_64 = R17_64;
inline constexpr MCPhysReg S2_64 = R18_64;
inline constexpr MCPhysReg S3_64 = R19_64;
inline constexpr MCPhysReg S4_64 = R20_64;
inline constexpr MCPhysReg S5_64 = R21_64;
inline constexpr MCPhysReg S6_64 = R22_64;
inline constexpr MCPhysReg S7_64 = R23_64;
inline constexpr MCPhysReg T8_64 = R24_64;
inline constexpr MCPhysReg T9_64 = R25_64;
inline constexpr MCPhysReg K0_64 = R26_64;
inline constexpr MCPhysReg K1_64 = R27_64;
inline constexpr MCPhysReg GP_64 = R28_64;
inline constexpr MCPhysReg SP_64 = R29_64;
inline constexpr MCPhysReg FP_64 = R30_64;
inline constexpr MCPhysReg RA_64 = R31_64;
} // namespace Mips
} // namespace llvm

// Defines symbolic names for the Mips instructions.
#define GET_INSTRINFO_ENUM
#define GET_INSTRINFO_MC_HELPER_DECLS
#include "MipsGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "MipsGenSubtargetInfo.inc"

#endif
