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

#include "llvm/BinaryFormat/Wasm.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

class MCAsmBackend;
class MCCodeEmitter;
class MCContext;
class MCInstrInfo;
class MCObjectWriter;
class MCSubtargetInfo;
class MVT;
class Target;
class Triple;
class raw_pwrite_stream;

Target &getTheWebAssemblyTarget32();
Target &getTheWebAssemblyTarget64();

MCCodeEmitter *createWebAssemblyMCCodeEmitter(const MCInstrInfo &MCII);

MCAsmBackend *createWebAssemblyAsmBackend(const Triple &TT);

MCObjectWriter *createWebAssemblyELFObjectWriter(raw_pwrite_stream &OS,
                                                 bool Is64Bit, uint8_t OSABI);

MCObjectWriter *createWebAssemblyWasmObjectWriter(raw_pwrite_stream &OS,
                                                  bool Is64Bit);

namespace WebAssembly {
enum OperandType {
  /// Basic block label in a branch construct.
  OPERAND_BASIC_BLOCK = MCOI::OPERAND_FIRST_TARGET,
  /// Local index.
  OPERAND_LOCAL,
  /// Global index.
  OPERAND_GLOBAL,
  /// 32-bit integer immediates.
  OPERAND_I32IMM,
  /// 64-bit integer immediates.
  OPERAND_I64IMM,
  /// 32-bit floating-point immediates.
  OPERAND_F32IMM,
  /// 64-bit floating-point immediates.
  OPERAND_F64IMM,
  /// 32-bit unsigned function indices.
  OPERAND_FUNCTION32,
  /// 32-bit unsigned memory offsets.
  OPERAND_OFFSET32,
  /// p2align immediate for load and store address alignment.
  OPERAND_P2ALIGN,
  /// signature immediate for block/loop.
  OPERAND_SIGNATURE,
  /// type signature immediate for call_indirect.
  OPERAND_TYPEINDEX,
};
} // end namespace WebAssembly

namespace WebAssemblyII {
enum {
  // For variadic instructions, this flag indicates whether an operand
  // in the variable_ops range is an immediate value.
  VariableOpIsImmediate = (1 << 0),
  // For immediate values in the variable_ops range, this flag indicates
  // whether the value represents a control-flow label.
  VariableOpImmediateIsLabel = (1 << 1)
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
  default:
    llvm_unreachable("Only loads and stores have p2align values");
  }
}

/// The operand number of the load or store address in load/store instructions.
static const unsigned LoadAddressOperandNo = 3;
static const unsigned StoreAddressOperandNo = 2;

/// The operand number of the load or store p2align in load/store instructions.
static const unsigned LoadP2AlignOperandNo = 1;
static const unsigned StoreP2AlignOperandNo = 0;

/// This is used to indicate block signatures.
enum class ExprType {
  Void    = -0x40,
  I32     = -0x01,
  I64     = -0x02,
  F32     = -0x03,
  F64     = -0x04,
  I8x16   = -0x05,
  I16x8   = -0x06,
  I32x4   = -0x07,
  F32x4   = -0x08,
  B8x16   = -0x09,
  B16x8   = -0x0a,
  B32x4   = -0x0b
};

/// Instruction opcodes emitted via means other than CodeGen.
static const unsigned Nop = 0x01;
static const unsigned End = 0x0b;

wasm::ValType toValType(const MVT &Ty);

} // end namespace WebAssembly
} // end namespace llvm

#endif
