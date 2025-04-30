//===-- Next32BaseInfo.h - Next32 Helpers APIs ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Next32 helpers APIs and constants
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NEXT32_NEXT32CONSTANTS_H
#define LLVM_LIB_TARGET_NEXT32_NEXT32CONSTANTS_H

#include "MCTargetDesc/Next32FixupKinds.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {

enum AddressSpace {
  ADDRESS_SPACE_GENERIC = 0,
  // RW, per-thread scratchpad
  ADDRESS_SPACE_TLS = 273,
  // RW, process-global scratchpad
  ADDRESS_SPACE_GLOBAL = 274,
  // RO, process-global scratchpad
  ADDRESS_SPACE_CONST = 275,
  // RW, thread shared scratchpad
  ADDRESS_SPACE_LOCAL = 3
};

namespace Next32Constants {
const unsigned int ImplicitDefValue = 0X22112A19;

enum Next32LibCalls {
  TLS_BASE = 0,
};

enum Next32Attributes { ATT_NORETURN, ATT_INLINE, ATT_SIZE };

const unsigned int MemoryOperationSizeShift = 0;
const unsigned int MemoryOperationSizeMask = 3;
enum MemoryOperationSize : unsigned int {
  Size8 = 0x03,
  Size16 = 0x01,
  Size32 = 0x00,
};

const unsigned int MemoryOperationFlagShift = 2;
const unsigned int MemoryOperationFlagMask = 1;
enum MemoryOperationFlag : unsigned int {
  OnceBit = 0x01,
};

const unsigned int MemoryOperationShift = 4;
const unsigned int MemoryOperationMask = 3;
enum MemoryOperation : unsigned int {
  Read = 0x00,
  Write = 0x01,
  Commit = 0x02,
  Flush = 0x3
};

enum RRIAttribute : unsigned int {
  Default = 0x0,
  ReadOnly = 0x1,
  ReadNone = 0x2,
  WriterParallel = 0x3
};

enum InstructionSize : unsigned int {
  InstructionSize8,
  InstructionSize16,
  InstructionSize32,
  InstructionSize64,
  InstructionSize128,
  InstructionSize256,
  InstructionSize512,
  InstructionSize1024
};

enum InstCodeAddressSpace : unsigned int {
  GENERIC = 0,
  TLS = 1,
  GLOBAL = 2,
  CONST = 3,
  LOCAL = 4
};

enum CondCode : unsigned int {
  NoCondition = 0x0,
  E = 0x4,
  NE = 0x6,
  BE = 0x8,
  AE = 0x9,
  B = 0xA,
  A = 0xB,
  GE = 0xC,
  LE = 0xD,
  G = 0xE,
  L = 0xF
};
} // namespace Next32Constants

struct Next32Helpers {
  static StringRef GetRRIAttributeMnemonic(unsigned int AttributeValue);
  static Next32Constants::RRIAttribute
  GetFunctionAttribute(StringRef FuncName, const TargetLoweringBase *TLB);
  static StringRef GetFunctionAttrName(Next32Constants::Next32Attributes Attr);
  static StringRef GetParallelMnemonic();
  static StringRef GetWriterMnemonic();
  static StringRef GetFeederMnemonic();
  static StringRef GetCondCodeString(Next32Constants::CondCode Cond);
  static Next32Constants::CondCode GetCondCodeFromString(StringRef CondStr);
  static Next32Constants::CondCode ISDCCToNext32CC(ISD::CondCode ISDCC);
  static Next32Constants::CondCode
  GetReverseNext32CC(Next32Constants::CondCode Cond);

  static bool IsPseudoMemOpcode(unsigned int Next32ISDOpcode);
  static bool IsPseudoReadOpcode(unsigned int Next32ISDOpcode);
  static bool IsPseudoWriteOpcode(unsigned int Next32ISDOpcode);
  static bool IsPseudoAtomicOpcode(unsigned int Next32ISDOpcode);

  static unsigned GetNext32VariadicPosition();
  static unsigned BitsToSizeFieldValue(unsigned Bits);
  static unsigned SizeFieldValueToBits(unsigned SizeFieldValue);
  static unsigned Log2AlignValueToBytes(unsigned SizeAlignValue);
  static unsigned BytesToLog2AlignValue(unsigned Bytes);
  static unsigned CountToLog2VecElemFieldValue(unsigned Count);
  static unsigned Log2VecElemFieldValueToCount(unsigned VecElemFieldValue);
  static unsigned GetInstAddressSpace(unsigned AddrSpaceValue);
  static unsigned MemNodeTypeToMemOps(MemSDNode *Mem);

  static bool IsValidVectorTy(EVT VT);

  static const char *
  GetLibcallFunctionName(Next32Constants::Next32LibCalls Func);

  static MachineBasicBlock::iterator
  FindArgumentFeedersEnd(MachineBasicBlock &MBB, MachineBasicBlock::iterator I);
};

namespace Next32II {
enum TOF {
  // Table gen instruction target specific flags start
  IsWriterChain = 1 << 0,
  Is128BitRRRRInstruction = 1 << 1,
  Is128BitRRRRInstructionSrcReg1In = 1 << 2,
  Is128BitRRRRInstructionSrcReg1Out = 1 << 3,
  Is128BitRRRRInstructionSrcReg2In = 1 << 4,
  Is128BitRRRRInstructionSrcReg2Out = 1 << 5,
  is128BitRRRRInstructionWithCount = 1 << 6,
  is128BitRRRRInstructionWithAddrSpace = 1 << 7,

  // Table gen instruction target specific flags end
  MO_MEM_64HI = Next32::reloc_4byte_mem_high,
  MO_MEM_64LO = Next32::reloc_4byte_mem_low,
  MO_FUNC_64HI = Next32::reloc_4byte_func_high,
  MO_FUNC_64LO = Next32::reloc_4byte_func_low,
  MO_FUNCTION = Next32::reloc_4byte_sym_function,
  MO_BB = Next32::reloc_4byte_sym_bb_imm
};
} // namespace Next32II
} // namespace llvm

#endif
