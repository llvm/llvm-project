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

#include "llvm/MC/MCInstrDesc.h"
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

MCCodeEmitter *createWebAssemblyMCCodeEmitter(const MCInstrInfo &MCII);

MCAsmBackend *createWebAssemblyAsmBackend(const Triple &TT);

MCObjectWriter *createWebAssemblyELFObjectWriter(raw_pwrite_stream &OS,
                                                 bool Is64Bit, uint8_t OSABI);

namespace WebAssembly {
enum OperandType {
  /// Basic block label in a branch construct.
  OPERAND_BASIC_BLOCK = MCOI::OPERAND_FIRST_TARGET,
  /// 32-bit floating-point immediates.
  OPERAND_FP32IMM,
  /// 64-bit floating-point immediates.
  OPERAND_FP64IMM,
  /// p2align immediate for load and store address alignment.
  OPERAND_P2ALIGN
};

/// WebAssembly-specific directive identifiers.
enum Directive {
  // FIXME: This is not the real binary encoding.
  DotParam = UINT64_MAX - 0,   ///< .param
  DotResult = UINT64_MAX - 1,  ///< .result
  DotLocal = UINT64_MAX - 2,   ///< .local
  DotEndFunc = UINT64_MAX - 3, ///< .endfunc
};

} // end namespace WebAssembly

namespace WebAssemblyII {
enum {
  // For variadic instructions, this flag indicates whether an operand
  // in the variable_ops range is an immediate value.
  VariableOpIsImmediate = (1 << 0),
  // For immediate values in the variable_ops range, this flag indicates
  // whether the value represents a control-flow label.
  VariableOpImmediateIsLabel = (1 << 1),
};
} // end namespace WebAssemblyII

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

namespace llvm {
namespace WebAssembly {

/// Return the default p2align value for a load or store with the given opcode.
inline unsigned GetDefaultP2Align(unsigned Opcode) {
  switch (Opcode) {
  case WebAssembly::LOAD8_S_I32:
  case WebAssembly::LOAD8_U_I32:
  case WebAssembly::LOAD8_S_I64:
  case WebAssembly::LOAD8_U_I64:
  case WebAssembly::STORE8_I32:
  case WebAssembly::STORE8_I64:
    return 0;
  case WebAssembly::LOAD16_S_I32:
  case WebAssembly::LOAD16_U_I32:
  case WebAssembly::LOAD16_S_I64:
  case WebAssembly::LOAD16_U_I64:
  case WebAssembly::STORE16_I32:
  case WebAssembly::STORE16_I64:
    return 1;
  case WebAssembly::LOAD_I32:
  case WebAssembly::LOAD_F32:
  case WebAssembly::STORE_I32:
  case WebAssembly::STORE_F32:
  case WebAssembly::LOAD32_S_I64:
  case WebAssembly::LOAD32_U_I64:
  case WebAssembly::STORE32_I64:
    return 2;
  case WebAssembly::LOAD_I64:
  case WebAssembly::LOAD_F64:
  case WebAssembly::STORE_I64:
  case WebAssembly::STORE_F64:
    return 3;
  default: llvm_unreachable("Only loads and stores have p2align values");
  }
}

/// The operand number of the load or store address in load/store instructions.
static const unsigned MemOpAddressOperandNo = 2;
/// The operand number of the stored value in a store instruction.
static const unsigned StoreValueOperandNo = 4;

} // end namespace WebAssembly
} // end namespace llvm

#endif
