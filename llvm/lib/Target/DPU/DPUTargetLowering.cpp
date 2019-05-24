//===-- DPUTargetLowering.h - BPF DAG Lowering Interface --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that DPU uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#include "DPUTargetLowering.h"
#include "DPUISelLowering.h"
#include "DPUTargetMachine.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include <iostream>
#include <llvm/MC/MCSymbol.h>

#define GET_REGINFO_ENUM

#include "DPUGenRegisterInfo.inc"

using namespace llvm;

// Calling Convention Implementation
#include "DPUGenCallingConv.inc"

#include "DPUCondCodes.h"
#include "MCTargetDesc/DPUAsmCondition.h"

#define DEBUG_TYPE "dpu-lower"

// A bunch of helper to ease code reading
#define VOLATILE true
#define NON_VOLATILE !VOLATILE
#define NON_TEMPORAL true
#define TEMPORAL !NON_TEMPORAL
#define INVARIANT true
#define TAIL_CALL true
#define NOT_TAIL_CALL !TAIL_CALL
#define ALWAYS_INLINE true
#define OPAQUE true
#define MUTABLE false
#define IMMUTABLE !MUTABLE

DPUTargetLowering::DPUTargetLowering(const TargetMachine &TM, DPUSubtarget &STI)
    : TargetLowering(TM), optLevel(TM.getOptLevel()) {
  // We need to be sure that the intrinsic will not generate instructions
  // that may imply things like stores of 64 bits immediate.
  // To do this, let's adjust the associated variables.
  MaxStoresPerMemset = MaxStoresPerMemcpy = MaxStoresPerMemmove = 4;
  MaxStoresPerMemsetOptSize = MaxStoresPerMemcpyOptSize =
      MaxStoresPerMemmoveOptSize = 4;
  // We cannot easily select values, since there is no status register.
  // However, the DPU IS is very good at checking and branching.
  PredictableSelectIsExpensive = true;
  setJumpIsExpensive(false);

  // Set up the register classes.
  addRegisterClass(MVT::i32, &DPU::GP_REGRegClass);
  addRegisterClass(MVT::i64, &DPU::GP64_REGRegClass);

  // Compute derived properties from the register classes
  computeRegisterProperties(STI.getRegisterInfo());

  setStackPointerRegisterToSaveRestore(R_STKP);

  setBooleanContents(BooleanContent::ZeroOrOneBooleanContent);

  setMinStackArgumentAlignment(4);

  // Global addresses require a special processing, achieved by LowerOperation.
  setOperationAction(ISD::GlobalTLSAddress, MVT::i32, Custom);
  setOperationAction(ISD::GlobalAddress, MVT::i32, Custom);
  setOperationAction(ISD::ExternalSymbol, MVT::i32, Custom);
  setOperationAction(ISD::BlockAddress, MVT::i32, Custom);
  setOperationAction(ISD::JumpTable, MVT::i32, Custom);
  setOperationAction(ISD::ConstantPool, MVT::i32, Custom);
  setOperationAction(ISD::BR_JT, MVT::Other, Expand);

  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::Other, Custom);
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::Other, Custom);
  setOperationAction(ISD::INTRINSIC_VOID, MVT::Other, Custom);

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v2i8, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v4i8, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v2i16, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v4i16, Expand);

  setOperationAction(ISD::SETCC, MVT::i1, Promote);
  setOperationAction(ISD::SETCC, MVT::i8, Promote);
  setOperationAction(ISD::SETCC, MVT::i16, Promote);
  setOperationAction(ISD::SETCC, MVT::i32, Custom);
  setOperationAction(ISD::SETCC, MVT::i64, Custom);

  setOperationAction(ISD::SELECT, MVT::i1, Promote);
  setOperationAction(ISD::SELECT, MVT::i8, Promote);
  setOperationAction(ISD::SELECT, MVT::i16, Promote);

  setOperationAction(ISD::SELECT_CC, MVT::i1, Promote);
  setOperationAction(ISD::SELECT_CC, MVT::i8, Promote);
  setOperationAction(ISD::SELECT_CC, MVT::i16, Promote);
  // todo do not expand select_cc and implement it in the .td
  setOperationAction(ISD::SELECT_CC, MVT::i32, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::i64, Expand);

  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i1, Promote);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i8, Promote);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i16, Promote);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i32, Custom);

  setOperationAction(ISD::STACKSAVE, MVT::i32, Custom);
  setOperationAction(ISD::STACKRESTORE, MVT::i32, Custom);

  // @todo MULHU and MULHS could work with 8 and 16 bits... need to implement it
  // for 8 bits... 16 is expansive.
  setOperationAction(ISD::MULHU, MVT::i1, Expand);
  setOperationAction(ISD::MULHU, MVT::i8, Expand);
  setOperationAction(ISD::MULHU, MVT::i16, Expand);
  setOperationAction(ISD::MULHU, MVT::i32, Expand);
  setOperationAction(ISD::MULHU, MVT::i64, Expand);

  setOperationAction(ISD::MULHS, MVT::i1, Expand);
  setOperationAction(ISD::MULHS, MVT::i8, Expand);
  setOperationAction(ISD::MULHS, MVT::i16, Expand);
  setOperationAction(ISD::MULHS, MVT::i32, Expand);
  setOperationAction(ISD::MULHS, MVT::i64, Expand);

  setOperationAction(ISD::UMUL_LOHI, MVT::i1, Expand);
  setOperationAction(ISD::UMUL_LOHI, MVT::i8, Expand);
  setOperationAction(ISD::UMUL_LOHI, MVT::i16, Expand);
  setOperationAction(ISD::UMUL_LOHI, MVT::i32, Expand);
  setOperationAction(ISD::UMUL_LOHI, MVT::i64, Expand);
  setOperationAction(ISD::SMUL_LOHI, MVT::i8, Expand);
  setOperationAction(ISD::SMUL_LOHI, MVT::i16, Expand);
  setOperationAction(ISD::SMUL_LOHI, MVT::i32, Expand);
  setOperationAction(ISD::SMUL_LOHI, MVT::i64, Expand);

  setOperationAction(ISD::MUL, MVT::i8, Expand);
  setOperationAction(ISD::MUL, MVT::i16, Expand);
  setOperationAction(ISD::MUL, MVT::i32, Custom);

  setOperationAction(ISD::UDIVREM, MVT::i8, Expand);
  setOperationAction(ISD::SDIVREM, MVT::i8, Expand);
  setOperationAction(ISD::UDIVREM, MVT::i16, Expand);
  setOperationAction(ISD::SDIVREM, MVT::i16, Expand);

  setOperationAction(ISD::UDIVREM, MVT::i32, Expand);
  setOperationAction(ISD::SDIVREM, MVT::i32, Expand);

  setOperationAction(ISD::CTLZ, MVT::i8, Promote);
  setOperationAction(ISD::CTLZ, MVT::i16, Promote);
  setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i8, Promote);
  setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i16, Promote);
  setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i32, Expand);
  setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i64, Expand);

  setOperationAction(ISD::CTTZ, MVT::i8, Promote);
  setOperationAction(ISD::CTTZ, MVT::i16, Promote);
  setOperationAction(ISD::CTTZ, MVT::i32, Expand);
  setOperationAction(ISD::CTTZ, MVT::i64, Expand);
  setOperationAction(ISD::CTTZ_ZERO_UNDEF, MVT::i8, Promote);
  setOperationAction(ISD::CTTZ_ZERO_UNDEF, MVT::i16, Promote);
  setOperationAction(ISD::CTTZ_ZERO_UNDEF, MVT::i32, Expand);
  setOperationAction(ISD::CTTZ_ZERO_UNDEF, MVT::i64, Expand);

  setOperationAction(ISD::CTPOP, MVT::i8, Promote);
  setOperationAction(ISD::CTPOP, MVT::i16, Promote);

  // 64x64 basic operations: use native library functions
  setOperationAction(ISD::MUL, MVT::i64, LibCall);
  setOperationAction(ISD::SDIV, MVT::i64, LibCall);
  setOperationAction(ISD::UDIV, MVT::i64, LibCall);
  setOperationAction(ISD::SREM, MVT::i64, LibCall);
  setOperationAction(ISD::UREM, MVT::i64, LibCall);
  setOperationAction(ISD::SDIVREM, MVT::i64, LibCall);
  setOperationAction(ISD::UDIVREM, MVT::i64, LibCall);

  setOperationAction(ISD::SHL_PARTS, MVT::i1, Expand);
  setOperationAction(ISD::SHL_PARTS, MVT::i8, Expand);
  setOperationAction(ISD::SHL_PARTS, MVT::i16, Expand);
  setOperationAction(ISD::SHL_PARTS, MVT::i32, Custom);

  setOperationAction(ISD::SRL_PARTS, MVT::i1, Expand);
  setOperationAction(ISD::SRL_PARTS, MVT::i8, Expand);
  setOperationAction(ISD::SRL_PARTS, MVT::i16, Expand);
  setOperationAction(ISD::SRL_PARTS, MVT::i32, Custom);

  setOperationAction(ISD::SRA_PARTS, MVT::i1, Expand);
  setOperationAction(ISD::SRA_PARTS, MVT::i8, Expand);
  setOperationAction(ISD::SRA_PARTS, MVT::i16, Expand);
  setOperationAction(ISD::SRA_PARTS, MVT::i32, Custom);

  setOperationAction(ISD::BRCOND, MVT::i64, Expand);

  setOperationAction(ISD::BR_CC, MVT::i1, Expand);
  setOperationAction(ISD::BR_CC, MVT::i8, Expand);
  setOperationAction(ISD::BR_CC, MVT::i16, Expand);
  setOperationAction(ISD::BR_CC, MVT::i32, Custom);
  setOperationAction(ISD::BR_CC, MVT::i64, Custom);

  setOperationAction(ISD::ADDC, MVT::i8, Expand);
  setOperationAction(ISD::ADDC, MVT::i16, Expand);
  setOperationAction(ISD::ADDC, MVT::i32, Expand);
  setOperationAction(ISD::ADDE, MVT::i8, Expand);
  setOperationAction(ISD::ADDE, MVT::i16, Expand);
  setOperationAction(ISD::ADDE, MVT::i32, Expand);

  setOperationAction(ISD::SUBC, MVT::i8, Expand);
  setOperationAction(ISD::SUBC, MVT::i16, Expand);
  setOperationAction(ISD::SUBC, MVT::i32, Expand);
  setOperationAction(ISD::SUBE, MVT::i8, Expand);
  setOperationAction(ISD::SUBE, MVT::i16, Expand);
  setOperationAction(ISD::SUBE, MVT::i32, Expand);

  // ID Register optimization
  setOperationAction(ISD::SHL, MVT::i32, Custom);

  // Trap load and store operations, for 64 bits only.
  // I tried to generalize the process with other lengths, the
  // main problem is that in this case llvm re-injects the
  // generated instructions to legalize the types (i.e. truncstore
  // and al.) and things become very complex for nothing
  // (see DAGTypeLegalizer::PromoteIntegerResult).
  setOperationAction(ISD::LOAD, MVT::i64, Custom);
  setOperationAction(ISD::STORE, MVT::i64, Custom);
  setOperationAction(ISD::STORE, MVT::i1, Custom);
  setOperationAction(ISD::STORE, MVT::i8, Custom);
  setOperationAction(ISD::STORE, MVT::i16, Custom);
  setOperationAction(ISD::STORE, MVT::i32, Custom);

  setOperationAction(ISD::VAARG, MVT::Other, Custom);
  setOperationAction(ISD::VASTART, MVT::Other, Custom);
  setOperationAction(ISD::VAEND, MVT::Other, Expand);
  setOperationAction(ISD::VACOPY, MVT::Other, Expand);

  setOperationAction(ISD::ADDRSPACECAST, MVT::i32, Custom);

  for (MVT VT : MVT::integer_valuetypes()) {
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::i1, Promote);
    setLoadExtAction(ISD::ZEXTLOAD, VT, MVT::i1, Promote);
    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::i1, Promote);
  }
}

static inline bool isALegalCallingConvention(CallingConv::ID Convention) {
  return Convention == CallingConv::C || Convention == CallingConv::Fast;
}

SDValue DPUTargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) const {
  // See this ctor for the list of specific operation actions.
  LLVM_DEBUG({
    dbgs() << "DPU/Lower - lower operation ";
    Op->dump(&DAG);
    dbgs() << "\n";
  });
  switch (Op.getOpcode()) {
  case ISD::ADDRSPACECAST:
    return LowerUnsupported(Op,DAG,
                            "Cast between addresses of different address space is not supported");

  case ISD::DYNAMIC_STACKALLOC:
  case ISD::STACKSAVE:
  case ISD::STACKRESTORE:
    return LowerUnsupported(Op, DAG,
                            "Dynamic allocation of stack is not supported");

  case ISD::MUL:
    return LowerMultiplication(Op, DAG);

  case ISD::GlobalAddress:
    return LowerGlobalAddress(Op, DAG);

  case ISD::GlobalTLSAddress:
    return LowerGlobalTLSAddress(Op, DAG);

  case ISD::JumpTable:
    return LowerJumpTable(Op, DAG);

  case ISD::SETCC:
    return LowerSetCc(Op, DAG);

  case ISD::BR_CC:
    return LowerBrCc(Op, DAG);

  case ISD::STORE:
    return LowerStore(Op, DAG);

  case ISD::LOAD:
    return LowerLoad(Op, DAG);

  case ISD::SHL:
    return LowerShift(Op, DAG, 0);

  case ISD::SRL:
    return LowerShift(Op, DAG, 1);

  case ISD::SRA:
    return LowerShift(Op, DAG, 2);

  case ISD::INTRINSIC_WO_CHAIN:
    return LowerIntrinsic(Op, DAG, ISD::INTRINSIC_WO_CHAIN);

  case ISD::INTRINSIC_W_CHAIN:
    return LowerIntrinsic(Op, DAG, ISD::INTRINSIC_W_CHAIN);

  case ISD::INTRINSIC_VOID:
    return LowerIntrinsic(Op, DAG, ISD::INTRINSIC_VOID);

  case ISD::VASTART:
    return LowerVASTART(Op, DAG);

  case ISD::VAARG:
    return LowerVAARG(Op, DAG);

  default: {
    const char *NodeName = getTargetNodeName(Op.getOpcode());
    LLVM_DEBUG({
      dbgs() << "FAIL: ";
      Op.dump(&DAG);
    });
    if (NodeName != nullptr) {
      LLVM_DEBUG(dbgs() << "\tnode name = " << NodeName << "\n");
    }
    for (unsigned eachOp = 0; eachOp < Op.getNumOperands(); eachOp++) {
      LLVM_DEBUG({
        dbgs() << "\toperand #" << std::to_string(eachOp) << " = ";
        Op.getOperand(eachOp).dump(&DAG);
      });
    }
  }
    report_fatal_error("NOT implemented: lowering of such a type of SDValue");
  }
}

const char *DPUTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  default:
    return nullptr;
  case DPUISD::RET_FLAG:
    return "DPUISD::RET_FLAG";
  case DPUISD::CALL:
    return "DPUISD::CALL";
  case DPUISD::INTRINSIC_CALL:
    return "DPUISD::INTRINSIC_CALL";
  case DPUISD::SetCC:
    return "DPUISD::SetCC";
  case DPUISD::BrCC:
    return "DPUISD::BrCC";
  case DPUISD::BrCCi:
    return "DPUISD::BrCCi";
  case DPUISD::BrCCZero:
    return "DPUISD::BrCCZero";
  case DPUISD::OrJCCZero:
    return "DPUISD::OrJCCZero";
  case DPUISD::AndJCCZero:
    return "DPUISD::AndJCCZero";
  case DPUISD::XorJCCZero:
    return "DPUISD::XorJCCZero";
  case DPUISD::AddJCCZero:
    return "DPUISD::AddJCCZero";
  case DPUISD::SubJCCZero:
    return "DPUISD::SubJCCZero";
  case DPUISD::Wrapper:
    return "DPUISD::Wrapper";
  case DPUISD::WRAM_STORE_64:
    return "DPUISD::WRAM_STORE_64";
  case DPUISD::MRAM_STORE_64:
    return "DPUISD::MRAM_STORE_64";
  case DPUISD::TRUNC64:
    return "DPUISD::TRUNC64";
  case DPUISD::LSL64_32:
    return "DPUISD::LSL64_32";
  case DPUISD::LSL64_LT32:
    return "DPUISD::LSL64_LT32";
  case DPUISD::LSL64_GT32:
    return "DPUISD::LSL64_GT32";
  case DPUISD::LSR64_32:
    return "DPUISD::LSR64_32";
  case DPUISD::LSR64_LT32:
    return "DPUISD::LSR64_LT32";
  case DPUISD::LSR64_GT32:
    return "DPUISD::LSR64_GT32";
  case DPUISD::ASR64_32:
    return "DPUISD::ASR64_32";
  case DPUISD::ASR64_LT32:
    return "DPUISD::ASR64_LT32";
  case DPUISD::ASR64_GT32:
    return "DPUISD::ASR64_GT32";
  case DPUISD::ROL64_32:
    return "DPUISD::ROL64_32";
  case DPUISD::ROL64_LT32:
    return "DPUISD::ROL64_LT32";
  case DPUISD::ROL64_GT32:
    return "DPUISD::ROL64_GT32";
  case DPUISD::ROR64_32:
    return "DPUISD::ROR64_32";
  case DPUISD::ROR64_LT32:
    return "DPUISD::ROR64_LT32";
  case DPUISD::ROR64_GT32:
    return "DPUISD::ROR64_GT32";
  case DPUISD::MUL8_UU:
    return "DPUISD::MUL8_UU";
  case DPUISD::MUL8_SU:
    return "DPUISD::MUL8_SU";
  case DPUISD::MUL8_SS:
    return "DPUISD::MUL8_SS";
  case DPUISD::MUL16_UU:
    return "DPUISD::MUL16_UU";
  case DPUISD::MUL16_SU:
    return "DPUISD::MUL16_SU";
  case DPUISD::MUL16_SS:
    return "DPUISD::MUL16_SS";
  case DPUISD::WRAM_STORE_8_IMM:
    return "DPUISD::WRAM_STORE_8_IMM";
  case DPUISD::WRAM_STORE_16_IMM:
    return "DPUISD::WRAM_STORE_16_IMM";
  case DPUISD::WRAM_STORE_32_IMM:
    return "DPUISD::WRAM_STORE_32_IMM";
  case DPUISD::WRAM_STORE_64_IMM:
    return "DPUISD::WRAM_STORE_64_IMM";
  case DPUISD::Addc:
    return "DPUISD::Addc";
  case DPUISD::Subc:
    return "DPUISD::Subc";
  case DPUISD::Rsubc:
    return "DPUISD::Rsubc";
  case DPUISD::Clo:
    return "DPUISD::Clo";
  case DPUISD::Cls:
    return "DPUISD::Cls";
  case DPUISD::Lslx:
    return "DPUISD::Lslx";
  case DPUISD::Lsl1:
    return "DPUISD::Lsl1";
  case DPUISD::Lsl1x:
    return "DPUISD::Lsl1x";
  case DPUISD::Lsrx:
    return "DPUISD::Lsrx";
  case DPUISD::Lsr1:
    return "DPUISD::Lsr1";
  case DPUISD::Lsr1x:
    return "DPUISD::Lsr1x";
  case DPUISD::AddJcc:
    return "DPUISD::AddJcc";
  case DPUISD::AddNullJcc:
    return "DPUISD::AddNullJcc";
  case DPUISD::AddcJcc:
    return "DPUISD::AddcJcc";
  case DPUISD::AddcNullJcc:
    return "DPUISD::AddcNullJcc";
  case DPUISD::AndJcc:
    return "DPUISD::AndJcc";
  case DPUISD::AndNullJcc:
    return "DPUISD::AndNullJcc";
  case DPUISD::OrJcc:
    return "DPUISD::OrJcc";
  case DPUISD::OrNullJcc:
    return "DPUISD::OrNullJcc";
  case DPUISD::XorJcc:
    return "DPUISD::XorJcc";
  case DPUISD::XorNullJcc:
    return "DPUISD::XorNullJcc";
  case DPUISD::NandJcc:
    return "DPUISD::NandJcc";
  case DPUISD::NandNullJcc:
    return "DPUISD::NandNullJcc";
  case DPUISD::NorJcc:
    return "DPUISD::NorJcc";
  case DPUISD::NorNullJcc:
    return "DPUISD::NorNullJcc";
  case DPUISD::NxorJcc:
    return "DPUISD::NxorJcc";
  case DPUISD::NxorNullJcc:
    return "DPUISD::NxorNullJcc";
  case DPUISD::AndnJcc:
    return "DPUISD::AndnJcc";
  case DPUISD::AndnNullJcc:
    return "DPUISD::AndnNullJcc";
  case DPUISD::OrnJcc:
    return "DPUISD::OrnJcc";
  case DPUISD::OrnNullJcc:
    return "DPUISD::OrnNullJcc";
  case DPUISD::LslJcc:
    return "DPUISD::LslJcc";
  case DPUISD::LslNullJcc:
    return "DPUISD::LslNullJcc";
  case DPUISD::LslxJcc:
    return "DPUISD::LslxJcc";
  case DPUISD::LslxNullJcc:
    return "DPUISD::LslxNullJcc";
  case DPUISD::Lsl1Jcc:
    return "DPUISD::Lsl1Jcc";
  case DPUISD::Lsl1NullJcc:
    return "DPUISD::Lsl1NullJcc";
  case DPUISD::Lsl1xJcc:
    return "DPUISD::Lsl1xJcc";
  case DPUISD::Lsl1xNullJcc:
    return "DPUISD::Lsl1xNullJcc";
  case DPUISD::LsrJcc:
    return "DPUISD::LsrJcc";
  case DPUISD::LsrNullJcc:
    return "DPUISD::LsrNullJcc";
  case DPUISD::LsrxJcc:
    return "DPUISD::LsrxJcc";
  case DPUISD::LsrxNullJcc:
    return "DPUISD::LsrxNullJcc";
  case DPUISD::Lsr1Jcc:
    return "DPUISD::Lsr1Jcc";
  case DPUISD::Lsr1NullJcc:
    return "DPUISD::Lsr1NullJcc";
  case DPUISD::Lsr1xJcc:
    return "DPUISD::Lsr1xJcc";
  case DPUISD::Lsr1xNullJcc:
    return "DPUISD::Lsr1xNullJcc";
  case DPUISD::AsrJcc:
    return "DPUISD::AsrJcc";
  case DPUISD::AsrNullJcc:
    return "DPUISD::AsrNullJcc";
  case DPUISD::RolJcc:
    return "DPUISD::RolJcc";
  case DPUISD::RolNullJcc:
    return "DPUISD::RolNullJcc";
  case DPUISD::RorJcc:
    return "DPUISD::RorJcc";
  case DPUISD::RorNullJcc:
    return "DPUISD::RorNullJcc";
  case DPUISD::MUL8_UUJcc:
    return "DPUISD::MUL8_UUJcc";
  case DPUISD::MUL8_UUNullJcc:
    return "DPUISD::MUL8_UUNullJcc";
  case DPUISD::MUL8_SUJcc:
    return "DPUISD::MUL8_SUJcc";
  case DPUISD::MUL8_SUNullJcc:
    return "DPUISD::MUL8_SUNullJcc";
  case DPUISD::MUL8_SSJcc:
    return "DPUISD::MUL8_SSJcc";
  case DPUISD::MUL8_SSNullJcc:
    return "DPUISD::MUL8_SSNullJcc";
  case DPUISD::SubJcc:
    return "DPUISD::SubJcc";
  case DPUISD::SubNullJcc:
    return "DPUISD::SubNullJcc";
  case DPUISD::RsubJcc:
    return "DPUISD::RsubJcc";
  case DPUISD::RsubNullJcc:
    return "DPUISD::RsubNullJcc";
  case DPUISD::SubcJcc:
    return "DPUISD::SubcJcc";
  case DPUISD::SubcNullJcc:
    return "DPUISD::SubcNullJcc";
  case DPUISD::RsubcJcc:
    return "DPUISD::RsubcJcc";
  case DPUISD::RsubcNullJcc:
    return "DPUISD::RsubcNullJcc";
  case DPUISD::CaoJcc:
    return "DPUISD::CaoJcc";
  case DPUISD::CaoNullJcc:
    return "DPUISD::CaoNullJcc";
  case DPUISD::ClzJcc:
    return "DPUISD::ClzJcc";
  case DPUISD::ClzNullJcc:
    return "DPUISD::ClzNullJcc";
  case DPUISD::CloJcc:
    return "DPUISD::CloJcc";
  case DPUISD::CloNullJcc:
    return "DPUISD::CloNullJcc";
  case DPUISD::ClsJcc:
    return "DPUISD::ClsJcc";
  case DPUISD::ClsNullJcc:
    return "DPUISD::ClsNullJcc";
  case DPUISD::MoveJcc:
    return "DPUISD::MoveJcc";
  case DPUISD::MoveNullJcc:
    return "DPUISD::MoveNullJcc";
  case DPUISD::RolAddJcc:
    return "DPUISD::RolAddJcc";
  case DPUISD::RolAddNullJcc:
    return "DPUISD::RolAddNullJcc";
  case DPUISD::LsrAddJcc:
    return "DPUISD::LsrAddJcc";
  case DPUISD::LsrAddNullJcc:
    return "DPUISD::LsrAddNullJcc";
  case DPUISD::LslAddJcc:
    return "DPUISD::LslAddJcc";
  case DPUISD::LslAddNullJcc:
    return "DPUISD::LslAddNullJcc";
  case DPUISD::LslSubJcc:
    return "DPUISD::LslSubJcc";
  case DPUISD::LslSubNullJcc:
    return "DPUISD::LslSubNullJcc";
  case DPUISD::TEST_NODE:
    return "DPUISD::TEST_NODE";
  }
}

SDValue DPUTargetLowering::LowerUnsupported(SDValue Op,
                                            SelectionDAG &DAG,
                                            StringRef Message) const {
  const Function & Func = DAG.getMachineFunction().getFunction();
  const DebugLoc &DL = Op.getDebugLoc();
  DiagnosticInfoUnsupported Diag(Func, Message, DL);
  Func.getContext().diagnose(Diag);
  if (DL.get() == NULL) {
    report_fatal_error(Message + "\n(add -g to cflags for more precise diagnostic)",
                       false);
  } else {
    report_fatal_error(Message, false);
  }
  return Op;
}

SDValue DPUTargetLowering::LowerVASTART(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue StackPtr = DAG.getCopyFromReg(Chain, DL, DPU::STKP,
                                        getPointerTy(DAG.getDataLayout()));
  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();

  // just store the stackptr (start of the variable argument list) in the
  // frameindex given in Op1
  return DAG.getStore(Chain, DL, StackPtr, Op.getOperand(1),
                      MachinePointerInfo(SV));
}

SDValue DPUTargetLowering::LowerVAARG(SDValue Op, SelectionDAG &DAG) const {
  SDNode *Node = Op.getNode();
  EVT VT = Node->getValueType(0);
  SDValue Chain = Node->getOperand(0);
  SDValue VAListPtr = Node->getOperand(1);
  unsigned ArgSize = Node->getConstantOperandVal(3);
  const Value *SV = cast<SrcValueSDNode>(Node->getOperand(2))->getValue();
  SDLoc DL(Node);

  // Load the vararg ptr to the last vararg loaded
  SDValue VAListLoad = DAG.getLoad(getPointerTy(DAG.getDataLayout()), DL, Chain,
                                   VAListPtr, MachinePointerInfo(SV));
  // Get to the next vararg by substracting the size of the arg (should be 4 or
  // 8)
  SDValue NextPtr = DAG.getNode(ISD::SUB, DL, VAListLoad.getValueType(),
                                VAListLoad, DAG.getIntPtrConstant(ArgSize, DL));
  // If the vararg is a 8 bytes variable, align the vararg ptr on 8 bytes.
  if (ArgSize == 8) {
    NextPtr = DAG.getNode(ISD::AND, DL, VAListLoad.getValueType(), NextPtr,
                          DAG.getIntPtrConstant(-8, DL));
  } else {
    assert(ArgSize == 4);
  }
  // Store the vararg ptr to be ready for next vararg
  Chain = DAG.getStore(VAListLoad.getValue(1), DL, NextPtr, VAListPtr,
                       MachinePointerInfo(SV));

  // Load vararg
  return DAG.getLoad(
      VT, DL, Chain, NextPtr, MachinePointerInfo(SV),
      std::max(VAListPtr.getValueType().getSizeInBits(), VT.getSizeInBits()) /
          8);
}

SDValue DPUTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  LLVM_DEBUG(dbgs() << "DPU/Lower - lower formal arguments\n");
  if (!isALegalCallingConvention(CallConv)) {
    DAG.getContext()->emitError("calling convention not supported");
  }

  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());
  CCInfo.AnalyzeFormalArguments(Ins, CC_DPU);

  LLVM_DEBUG({
    dbgs() << "DPU/Lower - lowering formal argument ";
    Chain->dump(&DAG);
  });
  int argIndex = 0;
  for (auto &VA : ArgLocs) {
    if (VA.isRegLoc()) {
      LLVM_DEBUG(dbgs() << "DPU/Lower - argument #" << std::to_string(argIndex)
                        << " is a register\n");
      EVT RegVT = VA.getLocVT();
      auto SimpleTy = RegVT.getSimpleVT().SimpleTy;
      if (SimpleTy == MVT::i32 || SimpleTy == MVT::i64) {
        // Transform the arguments in physical registers into virtual ones.
        const TargetRegisterClass *RG = (SimpleTy == MVT::i32)
                                            ? &DPU::GP_REGRegClass
                                            : &DPU::GP64_REGRegClass;
        unsigned VReg = RegInfo.createVirtualRegister(RG);
        RegInfo.addLiveIn(VA.getLocReg(), VReg);
        SDValue ArgValue = DAG.getCopyFromReg(Chain, DL, VReg, RegVT);

        // If this is an 8/16-bit value, it is really passed promoted to 32
        // bits. Insert an assert[sz]ext to capture this, then truncate to the
        // right size.
        if (VA.getLocInfo() == CCValAssign::SExt)
          ArgValue = DAG.getNode(ISD::AssertSext, DL, RegVT, ArgValue,
                                 DAG.getValueType(VA.getValVT()));
        else if (VA.getLocInfo() == CCValAssign::ZExt)
          ArgValue = DAG.getNode(ISD::AssertZext, DL, RegVT, ArgValue,
                                 DAG.getValueType(VA.getValVT()));

        if (VA.getLocInfo() != CCValAssign::Full)
          ArgValue = DAG.getNode(ISD::TRUNCATE, DL, VA.getValVT(), ArgValue);

        LLVM_DEBUG({
          dbgs() << "DPU/Lower - register argument pushed back as ";
          ArgValue.dump(&DAG);
        });
        InVals.push_back(ArgValue);
      } else {
        // Theoretically, we have declared in the calling conventions
        // that every basic type of argument should be promoted as i32.
        // If not the case => bug.
        DAG.getContext()->emitError(
            "unexpected register argument type - not i32");
      }
    } else {
      assert(VA.isMemLoc());
      LLVM_DEBUG(dbgs() << "DPU/Lower - argument #" << std::to_string(argIndex)
                        << " is in memory\n");

      SDValue InVal;
      ISD::ArgFlagsTy Flags = Ins[argIndex].Flags;

      if (Flags.isByVal()) {
        LLVM_DEBUG(dbgs() << "DPU/Lower - argument passed by value\n");
        int FI = MFI.CreateFixedObject(
            Flags.getByValSize(), -VA.getLocMemOffset() - Flags.getByValSize(),
            IMMUTABLE);
        LLVM_DEBUG(dbgs() << "DPU/Lower - added FI " << std::to_string(FI)
                          << "\n");
        InVal = DAG.getFrameIndex(FI, getPointerTy(DAG.getDataLayout()));
      } else {
        LLVM_DEBUG(
            dbgs()
            << "DPU/Lower - argument passed by reference - size in bits = "
            << std::to_string(VA.getLocVT().getSizeInBits()) << "\n");
        // Create the frame index object for this incoming parameter...
        int FI = MFI.CreateFixedObject(VA.getLocVT().getSizeInBits() / 8,
                                       -VA.getLocMemOffset() -
                                           VA.getLocVT().getSizeInBits() / 8,
                                       IMMUTABLE);
        LLVM_DEBUG(dbgs() << "DPU/Lower - added FI " << std::to_string(FI)
                          << "\n");

        // Create the SelectionDAG nodes corresponding to a load
        // from this parameter
        // NON_VOLATILE, TEMPORAL, !INVARIANT, /* Alignment */ 0
        SDValue FIN = DAG.getFrameIndex(FI, MVT::i32);

        InVal = DAG.getLoad(VA.getLocVT(), DL, Chain, FIN,
                            MachinePointerInfo::getFixedStack(
                                DAG.getMachineFunction(), FI /*, 0*/));
      }

      LLVM_DEBUG({
        dbgs() << "DPU/Lower - memory argument pushed back as ";
        InVal.dump(&DAG);
      });

      InVals.push_back(InVal);
    }
    argIndex++;
  }

  return Chain;
}

SDValue
DPUTargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                               bool IsVarArg,
                               const SmallVectorImpl<ISD::OutputArg> &Outs,
                               const SmallVectorImpl<SDValue> &OutVals,
                               const SDLoc &DL, SelectionDAG &DAG) const {
  // CCValAssign - represent the assignment of the return value to a location
  SmallVector<CCValAssign, 16> RVLocs;
  MachineFunction &MF = DAG.getMachineFunction();

  // CCState - Info about the registers and stack slot.
  CCState CCInfo(CallConv, IsVarArg, MF, RVLocs, *DAG.getContext());

  // Analyze return values.
  CCInfo.AnalyzeReturn(Outs, RetCC_DPU);

  SDValue Flag;
  SmallVector<SDValue, 4> RetOps(1, Chain);

  // Copy the result values into the output registers.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");

    Chain = DAG.getCopyToReg(Chain, DL, VA.getLocReg(), OutVals[i], Flag);

    // Guarantee that all emitted copies are stuck together,
    // avoiding something bad.
    Flag = Chain.getValue(1);
    RetOps.push_back(DAG.getRegister(VA.getLocReg(), VA.getLocVT()));
  }

  RetOps[0] = Chain; // Update chain.

  // Add the flag if we have it.
  if (Flag.getNode())
    RetOps.push_back(Flag);

  return DAG.getNode(DPUISD::RET_FLAG, DL, MVT::Other, RetOps);
}

static bool isAnIntrinsicInvocationSymbol(const char *FunctionName) {
  int i = strncmp(FunctionName, "__intrinsic__", strlen("__intrinsic__"));
  return (i == 0);
}

SDValue DPUTargetLowering::LowerCall(TargetLowering::CallLoweringInfo &CLI,
                                     SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG = CLI.DAG;
  auto &Outs = CLI.Outs;
  auto &OutVals = CLI.OutVals;
  auto &Ins = CLI.Ins;
  SDValue Chain = CLI.Chain;
  SDValue Callee = CLI.Callee;
  bool &IsTailCall = CLI.IsTailCall;
  CallingConv::ID CallConv = CLI.CallConv;
  MachineFunction &MF = DAG.getMachineFunction();
  bool isVarArg = CLI.IsVarArg;
  SDLoc &dl = CLI.DL;
  SmallVector<SDValue, 12> MemOpChains;
  bool isIntrinsicCall = false;
  MachineFrameInfo &MFI = DAG.getMachineFunction().getFrameInfo();

  // Does not support tail call optimization.
  IsTailCall = NOT_TAIL_CALL;

  switch (CallConv) {
  default:
    report_fatal_error("Unsupported calling convention");
  case CallingConv::Fast:
  case CallingConv::C:
    break;
  }

  // Analyze operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, MF, ArgLocs, *DAG.getContext());

  SmallVector<ISD::OutputArg, 8> FixedOuts;
  SmallVector<ISD::OutputArg, 8> VarArgOuts;

  // Extract VarArg from fixed Outs
  for (unsigned int eachOuts = 0; eachOuts < Outs.size(); eachOuts++) {
    ISD::OutputArg &currOut = Outs[eachOuts];
    if (currOut.IsFixed) {
      FixedOuts.push_back(currOut);
    } else {
      VarArgOuts.push_back(currOut);
    }
  }
  CCInfo.AnalyzeCallOperands(FixedOuts, CC_DPU);
  CCInfo.AnalyzeCallOperands(VarArgOuts, CC_DPU_VarArg);
  unsigned NumBytes = CCInfo.getNextStackOffset();
  Chain = DAG.getCALLSEQ_START(Chain, NumBytes, 0, CLI.DL);

  SmallVector<std::pair<unsigned, SDValue>, 5> RegsToPass;
  SDValue StackPtr;
  SmallVector<SDValue, 8> ByValArgs;

  LLVM_DEBUG(dbgs() << "DPU/Lower - walking through argument assignment n="
                    << std::to_string(ArgLocs.size()) << "\n");
  for (unsigned argIndex = 0, e = ArgLocs.size(); argIndex != e; ++argIndex) {
    CCValAssign &VA = ArgLocs[argIndex];
    SDValue Arg = OutVals[argIndex];
    ISD::ArgFlagsTy Flags = Outs[argIndex].Flags;

    if (!Flags.isByVal()) {
      continue;
    }

    // Promote the value if needed.
    switch (VA.getLocInfo()) {
    default:
      llvm_unreachable("Unknown loc info");
    case CCValAssign::Full:
      break;
    case CCValAssign::SExt:
      Arg = DAG.getNode(ISD::SIGN_EXTEND, CLI.DL, VA.getLocVT(), Arg);
      break;
    case CCValAssign::ZExt:
      Arg = DAG.getNode(ISD::ZERO_EXTEND, CLI.DL, VA.getLocVT(), Arg);
      break;
    case CCValAssign::AExt:
      Arg = DAG.getNode(ISD::ANY_EXTEND, CLI.DL, VA.getLocVT(), Arg);
      break;
    }

    unsigned Size = Flags.getByValSize();
    unsigned Align = Flags.getByValAlign();
    int FI = MFI.CreateStackObject(Size, Align, false);
    SDValue FIPtr = DAG.getFrameIndex(FI, getPointerTy(DAG.getDataLayout()));
    SDValue ArgSize = DAG.getConstant(Size, dl, MVT::i32);

    LLVM_DEBUG(dbgs() << "DPU/Lower - argument is pushed by value\n");
    Chain = DAG.getMemcpy(Chain, dl, FIPtr, Arg, ArgSize, Align, NON_VOLATILE,
                          ALWAYS_INLINE, NOT_TAIL_CALL, MachinePointerInfo(),
                          MachinePointerInfo());
    ByValArgs.push_back(FIPtr);
  }

  for (unsigned argIndex = 0, ByValIdx = 0, e = ArgLocs.size(); argIndex != e;
       ++argIndex) {
    CCValAssign &VA = ArgLocs[argIndex];
    SDValue Arg = OutVals[argIndex];
    ISD::ArgFlagsTy Flags = Outs[argIndex].Flags;

    if (Flags.isByVal())
      Arg = ByValArgs[ByValIdx++];

    if (VA.isRegLoc()) {
      // Push arguments into RegsToPass vector
      auto RegVal = std::make_pair(VA.getLocReg(), Arg);
      LLVM_DEBUG({
        dbgs() << "DPU/Lower - argument is a register, pushing back as pair "
               << std::to_string(RegVal.first) << " :: ";
        RegVal.second.dump(&DAG);
      });
      RegsToPass.push_back(RegVal);
    } else {
      assert(VA.isMemLoc());
      if (!StackPtr.getNode())
        StackPtr = DAG.getCopyFromReg(Chain, dl, DPU::STKP,
                                      getPointerTy(DAG.getDataLayout()));

      SDValue PtrOff = DAG.getNode(
          ISD::ADD, dl, getPointerTy(DAG.getDataLayout()), StackPtr,
          DAG.getIntPtrConstant(
              -VA.getLocMemOffset() - VA.getLocVT().getSizeInBits() / 8, dl));

      SDValue MemOp;

      LLVM_DEBUG(dbgs() << "DPU/Lower - argument is pushed by reference\n");
      // NON_VOLATILE, TEMPORAL, 0
      MemOp = DAG.getStore(Chain, dl, Arg, PtrOff, MachinePointerInfo());
      LLVM_DEBUG({
        dbgs() << "DPU/Lower - argument is on stack, pushing back as ";
        MemOp.dump(&DAG);
      });
      MemOpChains.push_back(MemOp);
    }
  }

  // Transform all store nodes into one single node because all store nodes are
  // independent of each other.
  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, MemOpChains);

  SDValue InFlag;

  // Build a sequence of copy-to-reg nodes chained together with token chain and
  // flag operands which copy the outgoing args into registers.  The InFlag in
  // necessary since all emitted instructions must be stuck together.
  for (auto &Reg : RegsToPass) {
    Chain = DAG.getCopyToReg(Chain, CLI.DL, Reg.first, Reg.second, InFlag);
    InFlag = Chain.getValue(1);
  }

  // If the callee is a GlobalAddress node (quite common, every direct call is)
  // turn it into a TargetGlobalAddress node so that legalize doesn't hack it.
  // Likewise ExternalSymbol -> TargetExternalSymbol.
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee))
    Callee = DAG.getTargetGlobalAddress(G->getGlobal(), CLI.DL,
                                        getPointerTy(DAG.getDataLayout()),
                                        G->getOffset(), 0);
  else if (ExternalSymbolSDNode *E = dyn_cast<ExternalSymbolSDNode>(Callee)) {
    const char *SymbolName = E->getSymbol();
    if (isAnIntrinsicInvocationSymbol(SymbolName)) {
      isIntrinsicCall = true;
      Callee =
          DAG.getTargetExternalSymbol(SymbolName + strlen("__intrinsic"),
                                      getPointerTy(DAG.getDataLayout()), 0);
    } else {
      Callee = DAG.getTargetExternalSymbol(
          E->getSymbol(), getPointerTy(DAG.getDataLayout()), 0);
    }
  }

  // Returns a chain & a flag for retval copy to use.
  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);
  SmallVector<SDValue, 8> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  // Add argument registers to the end of the list so that they are
  // known live into the call.
  for (auto &Reg : RegsToPass)
    Ops.push_back(DAG.getRegister(Reg.first, Reg.second.getValueType()));

  if (InFlag.getNode())
    Ops.push_back(InFlag);

  Chain = DAG.getNode(isIntrinsicCall ? DPUISD::INTRINSIC_CALL : DPUISD::CALL,
                      CLI.DL, NodeTys, Ops);
  InFlag = Chain.getValue(1);

  // Create the CALLSEQ_END node.
  Chain = DAG.getCALLSEQ_END(
      Chain,
      DAG.getConstant(NumBytes, CLI.DL, getPointerTy(DAG.getDataLayout()),
                      OPAQUE),
      DAG.getConstant(0, CLI.DL, getPointerTy(DAG.getDataLayout()), OPAQUE),
      InFlag, CLI.DL);
  InFlag = Chain.getValue(1);

  // Handle result values, copying them out of physregs into vregs that we
  // return.
  return LowerCallResult(Chain, InFlag, CallConv, isVarArg, Ins, CLI.DL, DAG,
                         InVals);
}

SDValue DPUTargetLowering::LowerCallResult(
    SDValue Chain, SDValue InFlag, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, SDLoc DL, SelectionDAG &DAG,
    SmallVectorImpl<SDValue> &InVals) const {

  MachineFunction &MF = DAG.getMachineFunction();
  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, RVLocs, *DAG.getContext());

  CCInfo.AnalyzeCallResult(Ins, RetCC_DPU);

  // Copy all of the result registers out of their specified physreg.
  for (auto &Val : RVLocs) {
    Chain =
        DAG.getCopyFromReg(Chain, DL, Val.getLocReg(), Val.getValVT(), InFlag)
            .getValue(1);
    InFlag = Chain.getValue(2);
    InVals.push_back(Chain.getValue(0));
  }

  return Chain;
}

std::pair<unsigned, const TargetRegisterClass *>
DPUTargetLowering::getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                                                StringRef Constraint,
                                                MVT VT) const {
  if (Constraint.size() == 1) {
    LLVM_DEBUG(dbgs() << "DPU/Lower - get reg for inline asm constraint ("
                      << Constraint << ")\n");
    return std::make_pair(0U, &DPU::GP_REGRegClass);
  }
  return TargetLowering::getRegForInlineAsmConstraint(TRI, Constraint, VT);
}

static bool canEncodeImmediateOnNBitsSigned(int64_t value, uint32_t bits) {
  int64_t threshold = (1L << (bits - 1)) - 1;

  return (value <= threshold) && (value >= ~threshold);
}

bool DPUTargetLowering::isLegalICmpImmediate(int64_t value) const {
  // Jcc handles 11-bit signed immediates
  return canEncodeImmediateOnNBitsSigned(value, 11);
}

bool DPUTargetLowering::isLegalAddressingMode(const DataLayout &DL,
                                              const AddrMode &AM, Type *Ty,
                                              unsigned AS,
                                              Instruction *I) const {
  LLVM_DEBUG({
    dbgs() << "DPU/Lower - is checking legal addressing mode\n";
    Ty->dump();
  });
  // Theoretically we support everything here.
  return true;
}

bool DPUTargetLowering::isOffsetFoldingLegal(
    const GlobalAddressSDNode *GA) const {
  return false;
}

static bool isArgumentExtended(SDValue Op, MVT::SimpleValueType initialType,
                               bool &isSigned, SDValue &realArgument) {
  bool argCouldWork = false;
  int opCode = Op.getOpcode();
  SDValue operand;

  switch (opCode) {
  case ISD::AssertSext:
    argCouldWork = true;
    isSigned = true;
    operand = Op.getOperand(1);
    break;
  case ISD::AssertZext:
    argCouldWork = true;
    isSigned = false;
    operand = Op.getOperand(1);
    break;
  case ISD::SIGN_EXTEND:
    argCouldWork = true;
    isSigned = false;
    operand = Op.getOperand(0);
    break;
  case ISD::ZERO_EXTEND:
    argCouldWork = true;
    isSigned = true;
    operand = Op.getOperand(0);
    break;
  case ISD::LOAD: {
    LoadSDNode *LoadOp = cast<LoadSDNode>(Op);

    if (LoadOp) {
      switch (LoadOp->getExtensionType()) {
      case ISD::LoadExtType::SEXTLOAD:
        argCouldWork = true;
        isSigned = true;
        break;
      case ISD::LoadExtType::ZEXTLOAD:
        argCouldWork = true;
        isSigned = false;
        break;
      default:
        break;
      }

      if (argCouldWork) {
        if (LoadOp->getMemoryVT().getSimpleVT().SimpleTy <= initialType) {
          realArgument = Op;
          return true;
        } else {
          return false;
        }
      }
    }

    break;
  }
  default:
    break;
  }

  if (argCouldWork) {
    if (const VTSDNode *N = dyn_cast<VTSDNode>(operand)) {
      if (N->getVT().isSimple() &&
          N->getVT().getSimpleVT().SimpleTy <= initialType) {
        realArgument = Op.getOperand(0);
        return true;
      }
    }
  }

  return false;
}

// Tries to convert a SDValue to a constant, if possible.
//
// Checks whether the provided SDValue represents a constant which can be used
// as an immediate for an operation, without checking the constant size (since
// the length of immediate operands depends on the opcodes). Returns NULL if the
// conversion is not applicable, the constant value otherwise.
static bool canFetchConstantTo(SDValue Value, uint64_t *pValue) {
  if (Value.getOpcode() == ISD::Constant) {
    ConstantSDNode *Constant = cast<ConstantSDNode>(Value);
    if ((!Constant->isOpaque()) && (Constant->getConstantIntValue() != NULL)) {
      *pValue = *(Constant->getConstantIntValue()->getValue().getRawData());
      return true;
    }
  }
  return false;
}

static bool canUseMulX(SDValue firstOperand, SDValue secondOperand,
                       MVT::SimpleValueType initialType,
                       MVT::SimpleValueType resultType, unsigned int mulSS,
                       unsigned int mulSU, unsigned int mulUU,
                       SelectionDAG &DAG, SDLoc dl, SDValue &MulXNode) {
  bool firstArgumentIsSigned, secondArgumentIsSigned;
  SDValue realFirstArgument, realSecondArgument;
  unsigned int opCode;
  SDValue Ops[2];

  if (isArgumentExtended(firstOperand, initialType, firstArgumentIsSigned,
                         realFirstArgument) &&
      isArgumentExtended(secondOperand, initialType, secondArgumentIsSigned,
                         realSecondArgument)) {
    Ops[0] = realFirstArgument;
    Ops[1] = realSecondArgument;

    if (firstArgumentIsSigned) {
      if (secondArgumentIsSigned) {
        opCode = mulSS;
      } else {
        opCode = mulSU;
      }
    } else {
      if (secondArgumentIsSigned) {
        // We do not have mulUS: let's swap the operands.
        opCode = mulSU;
        SDValue tmp = Ops[0];
        Ops[0] = Ops[1];
        Ops[1] = tmp;
      } else {
        opCode = mulUU;
      }
    }

    unsigned int assertOpCode =
        (firstArgumentIsSigned || secondArgumentIsSigned) ? ISD::AssertSext
                                                          : ISD::AssertZext;
    SDValue valueType = DAG.getValueType(EVT(resultType));
    SDVTList VTs = DAG.getVTList(MVT::i32);

    SDValue MulNode = DAG.getNode(opCode, dl, VTs, Ops);
    SDValue assertOps[] = {MulNode, valueType};

    MulXNode = DAG.getNode(assertOpCode, dl, VTs, assertOps);

    return true;
  }

  return false;
}

static bool canUseMulByConstant(SDValue firstOperand, SDValue secondOperand,
                                unsigned int &nrOfInstructions,
                                SelectionDAG &DAG, SDLoc dl, EVT VT,
                                SDValue &MulCstNode) {
  /*
   * We are using the "group method" (cf Hacker's Delight) to represent
   * multiplications by a constant. This is not always optimal... For example, y
   * = x * 45 (45 = 0b101101) will be translated to: y = lsl_add x x 2 y =
   * lsl_add y x 3 y = lsl_add y x 5
   *
   * This could be translated to:
   * y = lsl_add x x 2
   * y = lsl_add y y 3
   *
   * todo: try to optimize
   */
  ConstantSDNode *C = dyn_cast<ConstantSDNode>(secondOperand);

  if (!C) {
    return false;
  }

  bool isNegative = false;
  int64_t Cst = C->getSExtValue();
  uint64_t UCst = (uint64_t)Cst;

  if (Cst == 0) {
    nrOfInstructions = 0;
    MulCstNode = DAG.getConstant(0, dl, VT);
    return true;
  } else if (Cst == 1) {
    nrOfInstructions = 0;
    MulCstNode = firstOperand;
    return true;
  } else if (Cst == -1) {
    nrOfInstructions = 1;
    MulCstNode =
        DAG.getNode(ISD::SUB, dl, VT, DAG.getConstant(0, dl, VT), firstOperand);
    return true;
  }

  if (Cst < 0) {
    isNegative = true;
    UCst = (uint64_t)(-Cst);
  }

  bool isOdd = (UCst & 1) == 1;
  UCst = UCst & ~1;

  std::vector<int32_t> powersOfTwo = {};
  uint32_t currentSeriesLength = 0;

  for (int eachBit = 0; eachBit < 63; ++eachBit) {
    if ((UCst & (1L << eachBit)) != 0) {
      powersOfTwo.push_back(eachBit);
      currentSeriesLength++;
    } else {
      if (currentSeriesLength >= 3) {
        for (unsigned int eachSeriesBit = 0;
             eachSeriesBit < currentSeriesLength; ++eachSeriesBit) {
          powersOfTwo.pop_back();
        }

        powersOfTwo.push_back(eachBit);
        powersOfTwo.push_back(-(eachBit - currentSeriesLength));
      }

      currentSeriesLength = 0;
    }
  }

  SDValue currentNode;
  bool isFirst = true;
  unsigned int totalNrOfInstructions = 0;

  if (isOdd) {
    currentNode = firstOperand;
    isFirst = false;
  }

  for (int32_t eachPowerOfTwo : powersOfTwo) {
    if (isFirst) {
      currentNode = DAG.getNode(ISD::SHL, dl, VT, firstOperand,
                                DAG.getConstant(eachPowerOfTwo, dl, VT));
      isFirst = false;
    } else if (eachPowerOfTwo < 0) {
      currentNode =
          DAG.getNode(ISD::SUB, dl, VT, currentNode,
                      DAG.getNode(ISD::SHL, dl, VT, firstOperand,
                                  DAG.getConstant(-eachPowerOfTwo, dl, VT)));
    } else {
      currentNode =
          DAG.getNode(ISD::ADD, dl, VT, currentNode,
                      DAG.getNode(ISD::SHL, dl, VT, firstOperand,
                                  DAG.getConstant(eachPowerOfTwo, dl, VT)));
    }

    totalNrOfInstructions++;
  }

  if (isNegative) {
    currentNode =
        DAG.getNode(ISD::SUB, dl, VT, DAG.getConstant(0, dl, VT), currentNode);
    totalNrOfInstructions++;
  }

  nrOfInstructions = totalNrOfInstructions;
  MulCstNode = currentNode;
  return true;
}

/*
 * These constants do not describe the number of instructions executed, but the
 * number instructions needed in IRAM for one more multiplication.
 */
const static unsigned int nrOfInstructionsForMul8 = 1;
const static unsigned int nrOfInstructionsForMul16 = 7;
const static unsigned int nrOfInstructionsForMul32 = 7;

SDValue DPUTargetLowering::LowerMultiplication(SDValue Op,
                                               SelectionDAG &DAG) const {
  /*
   * Multiplications on DPU are generally slow.
   * We can still try to optimize small multiplications, and multiplications by
   * a constant.
   */
  SDValue firstOperand = Op.getOperand(0);
  SDValue secondOperand = Op.getOperand(1);
  SDValue LoweredMultiplication = Op;
  SDValue resultWithConstant;
  SDLoc dl(Op);
  EVT VT = Op->getValueType(0);

  unsigned int nrOfInstructions = nrOfInstructionsForMul32;
  unsigned int nrOfInstructionsWithConstant;
  bool canUseOptimizedMultiplication, canUseMultiplicationByConstant;

  canUseOptimizedMultiplication = canUseMulX(
      firstOperand, secondOperand, MVT::i8, MVT::i16, DPUISD::MUL8_SS,
      DPUISD::MUL8_SU, DPUISD::MUL8_UU, DAG, dl, LoweredMultiplication);

  if (canUseOptimizedMultiplication) {
    nrOfInstructions = nrOfInstructionsForMul8;
  } else {
    canUseOptimizedMultiplication = canUseMulX(
        firstOperand, secondOperand, MVT::i16, MVT::i32, DPUISD::MUL16_SS,
        DPUISD::MUL16_SU, DPUISD::MUL16_UU, DAG, dl, LoweredMultiplication);

    if (canUseOptimizedMultiplication) {
      nrOfInstructions = nrOfInstructionsForMul16;
    }
  }

  canUseMultiplicationByConstant = canUseMulByConstant(
      firstOperand, secondOperand, nrOfInstructionsWithConstant, DAG, dl, VT,
      resultWithConstant);

  // Check if it interesting to switch to the constant representation
  // ("nrOfInstructions + 1" because we suppose we need an additional MOVE
  // before the multiplication)
  if (canUseMultiplicationByConstant &&
      (nrOfInstructionsWithConstant < (nrOfInstructions + 1))) {
    LoweredMultiplication = resultWithConstant;
  }

  return LoweredMultiplication;
}

SDValue DPUTargetLowering::LowerGlobalAddress(SDValue Op,
                                              SelectionDAG &DAG) const {
  const GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
  int64_t Offset = cast<GlobalAddressSDNode>(Op)->getOffset();
  // Create the TargetGlobalAddress node, folding in the constant offset.
  SDValue Result = DAG.getTargetGlobalAddress(
      GV, SDLoc(Op), getPointerTy(DAG.getDataLayout()), Offset);
  return DAG.getNode(DPUISD::Wrapper, SDLoc(Op),
                     getPointerTy(DAG.getDataLayout()), Result);
}

SDValue DPUTargetLowering::LowerGlobalTLSAddress(SDValue Op,
                                                 SelectionDAG &DAG) const {
  GlobalAddressSDNode *GA = cast<GlobalAddressSDNode>(Op);
  if (DAG.getTarget().Options.EmulatedTLS)
    return LowerToTLSEmulatedModel(GA, DAG);

  SDLoc DL(GA);
  const GlobalValue *GV = GA->getGlobal();
  EVT PtrVT = getPointerTy(DAG.getDataLayout());

  TLSModel::Model model = getTargetMachine().getTLSModel(GV);

  switch (model) {
  case TLSModel::GeneralDynamic:
  case TLSModel::LocalDynamic:
    return Op;
  case TLSModel::InitialExec:
  case TLSModel::LocalExec:
    break;
  }

  int64_t Offset = GA->getOffset();
  // Create the TargetGlobalAddress node, folding in the constant offset.
  SDValue GlobalBase = DAG.getTargetGlobalAddress(GV, DL, PtrVT, 0);
  const SDValue &Wrapped = DAG.getNode(DPUISD::Wrapper, DL, PtrVT, GlobalBase);

  const uint64_t ValueAlignment = GV->getAlignment();
  const uint64_t ValueSize =
      DAG.getDataLayout().getTypeAllocSize(GV->getValueType());
  const uint64_t ValueContainerSize =
      (ValueAlignment > ValueSize) ? ValueAlignment : ValueSize;

  const SDValue &ThreadOffsetBase =
      DAG.getNode(ISD::MUL, DL, PtrVT, DAG.getRegister(DPU::ID, MVT::i32),
                  DAG.getConstant(ValueContainerSize, DL, MVT::i32));
  const SDValue &ThreadOffset =
      DAG.getNode(ISD::ADD, DL, PtrVT, ThreadOffsetBase,
                  DAG.getConstant(Offset, DL, MVT::i32));

  return DAG.getNode(ISD::ADD, DL, PtrVT, ThreadOffset, Wrapped);
}

SDValue DPUTargetLowering::LowerJumpTable(SDValue Op, SelectionDAG &DAG) const {
  JumpTableSDNode *JT = cast<JumpTableSDNode>(Op);
  SDValue Result =
      DAG.getTargetJumpTable(JT->getIndex(), getPointerTy(DAG.getDataLayout()));
  return DAG.getNode(DPUISD::Wrapper, SDLoc(JT),
                     getPointerTy(DAG.getDataLayout()), Result);
}

static uint64_t translateToBinarySetDPUAsmCondition(ISD::CondCode cond) {
  switch (cond) {
  default:
    llvm_unreachable("invalid condition");
  case ISD::SETOEQ:
  case ISD::SETUEQ:
  case ISD::SETEQ:
    return DPUAsmCondition::Condition::SetEqual;
  case ISD::SETONE:
  case ISD::SETUNE:
  case ISD::SETNE:
    return DPUAsmCondition::Condition::SetNotEqual;
  case ISD::SETOGT:
  case ISD::SETGT:
    return DPUAsmCondition::Condition::SetGreaterThanSigned;
  case ISD::SETOGE:
  case ISD::SETGE:
    return DPUAsmCondition::Condition::SetGreaterOrEqualSigned;
  case ISD::SETOLT:
  case ISD::SETLT:
    return DPUAsmCondition::Condition::SetLessThanSigned;
  case ISD::SETOLE:
  case ISD::SETLE:
    return DPUAsmCondition::Condition::SetLessOrEqualSigned;
  case ISD::SETUGT:
    return DPUAsmCondition::Condition::SetGreaterThanUnsigned;
  case ISD::SETUGE:
    return DPUAsmCondition::Condition::SetGreaterOrEqualUnsigned;
  case ISD::SETULT:
    return DPUAsmCondition::Condition::SetLessThanUnsigned;
  case ISD::SETULE:
    return DPUAsmCondition::Condition::SetLessOrEqualUnsigned;
  case ISD::SETTRUE:
  case ISD::SETTRUE2:
    return DPUAsmCondition::Condition::SetTrue;
  }
}

SDValue DPUTargetLowering::LowerSetCc(SDValue Op, SelectionDAG &DAG) const {
  SDValue leftOp = Op.getOperand(0);
  SDValue rightOp = Op.getOperand(1);
  SDValue Condition = Op.getOperand(2);
  ISD::CondCode CC = cast<CondCodeSDNode>(Condition)->get();
  SDLoc dl(Op);

  LLVM_DEBUG({
    EVT LhType = leftOp.getValueType();
    EVT RhType = rightOp.getValueType();
    dbgs() << "DPU/Lower - lowering SET_CC [src types=" << LhType.getEVTString()
           << "," << RhType.getEVTString() << "]"
           << " CondCode=" << CC << "\n";
  });

  uint64_t realCC = translateToBinarySetDPUAsmCondition(CC);

  SDVTList VTs = DAG.getVTList(Op.getValueType(), MVT::Glue);
  SDValue Ops[] = {DAG.getConstant(realCC, dl, MVT::i32), leftOp, rightOp};
  return DAG.getNode(DPUISD::SetCC, dl, VTs, Ops);
}

SDValue DPUTargetLowering::LowerBrCc(SDValue Op, SelectionDAG &DAG) const {
  SDValue Condition = Op.getOperand(1);
  ISD::CondCode CC = cast<CondCodeSDNode>(Condition)->get();
  SDValue leftOp = Op.getOperand(2);
  SDValue rightOp = Op.getOperand(3);
  SDValue Destination = Op.getOperand(4);

  LLVM_DEBUG({
    dbgs() << "DPU/Lower - lowering BR_CC ";
    Op.dump(&DAG);
  });

  // First, let's determine if there is a constant operand we can keep as
  // immediate.
  ConstantSDNode *C = dyn_cast<ConstantSDNode>(rightOp);

  // todo: handle 64bit compare with immediate
  if (!(C && isLegalICmpImmediate(C->getSExtValue())) ||
      (rightOp.getValueType().getSimpleVT().SimpleTy == MVT::i64)) {
    // No suitable constant found. We cannot do anything special.
    SDValue Chain = Op.getOperand(0);
    SDLoc dl(Op);
    SDVTList VTs = DAG.getVTList(Op.getValueType(), MVT::Glue);
    SDValue Ops[] = {Chain, DAG.getConstant(CC, dl, MVT::i32), leftOp, rightOp,
                     Destination};
    return DAG.getNode(DPUISD::BrCC, dl, VTs, Ops);
  }

  // Nothing fancy.
  SDValue Chain = Op.getOperand(0);
  SDLoc dl(Op);
  SDVTList VTs = DAG.getVTList(Op.getValueType(), MVT::Glue);
  SDValue Ops[] = {Chain, DAG.getConstant(CC, dl, MVT::i32), leftOp, rightOp,
                   Destination};
  return DAG.getNode(DPUISD::BrCCi, dl, VTs, Ops);
}

static bool canBeEncodedOn12Bits(uint64_t value) {
  return (value <= 0x7FFL) || (value >= 0xFFFFFFFFFFFFF800L);
}

static bool canBeEncodedOn16Bits(uint64_t value) {
  return (value <= 0x7FFFL) || (value >= 0xFFFFFFFFFFFF8000L);
}

SDValue DPUTargetLowering::LowerLoad(SDValue Op, SelectionDAG &DAG) const {
  SDLoc dl(Op);
  LLVM_DEBUG({
    dbgs() << "DPU/Lower - lowering load ";
    Op.dump(&DAG);
  });

  LoadSDNode *LoadOp = cast<LoadSDNode>(Op);

  if (LoadOp) {
    unsigned AddressSpace = LoadOp->getAddressSpace();
    MVT::SimpleValueType TargetType =
        LoadOp->getMemoryVT().getSimpleVT().SimpleTy;

    MachineMemOperand *MemOperand = LoadOp->getMemOperand();
    if (!MemOperand) {
      return Op;
    }

    if ((TargetType == MVT::i64) && (AddressSpace == DPUADDR_SPACE::WRAM) &&
        ((MemOperand->getAlignment() % 8) == 0)) {
      SDValue Chain = Op.getOperand(0);
      SDValue SourceOperand = Op.getOperand(1);
      SDValue DestinationOperand = Op.getOperand(2);

      SDValue Ops[] = {Chain, SourceOperand, DestinationOperand};
      SDVTList VTs = DAG.getVTList(Op.getValueType(), MVT::Glue);

      return DAG.getNode(DPUISD::WRAM_LOAD_64_ALIGNED, dl, VTs, Ops);
    } else {
      return Op;
    }

  } else {
    return Op;
  }
}

SDValue DPUTargetLowering::LowerStore(SDValue Op, SelectionDAG &DAG) const {
  SDLoc dl(Op);
  LLVM_DEBUG({
    dbgs() << "DPU/Lower - lowering store ";
    Op.dump(&DAG);
  });

  assert((Op.getNumOperands() == 4) &&
         "unexpected number of operands to store operation");
  SDValue Chain = Op.getOperand(0);
  SDValue SourceOperand = Op.getOperand(1);
  SDValue DestinationOperand = Op.getOperand(2);
  SDValue ArgOperand = Op.getOperand(3);
  bool usesImmediateOperand = false;
  bool canUseStoreImmediate = false;

  unsigned DPUOpcode;

  StoreSDNode *StoreOp = cast<StoreSDNode>(Op);
  unsigned AddressSpace = StoreOp->getAddressSpace();
  MVT::SimpleValueType TargetType =
      StoreOp->getMemoryVT().getSimpleVT().SimpleTy;

  if (StoreOp) {
    LLVM_DEBUG({
      unsigned Alignment = StoreOp->getAlignment();
      bool IsTruncating = StoreOp->isTruncatingStore();
      dbgs() << "DPU/Lower - interpreting store operation:"
             << " type = " << std::to_string(TargetType)
             << " align=" << std::to_string(Alignment)
             << " space=" << std::to_string(AddressSpace)
             << " trunc=" << std::to_string(IsTruncating) << "\n";
    });
    LLVM_DEBUG(
        dbgs() << "DPU/Lower - value type = "
               << std::to_string(Op.getValueType().getSimpleVT().SimpleTy)
               << "\n");
    LLVM_DEBUG(dbgs() << "DPU/Lower - store args =\n");
    for (unsigned eachArg = 0; eachArg < Op.getNumOperands(); eachArg++) {
      LLVM_DEBUG({
        dbgs() << "argument #" << std::to_string(eachArg) << " = ";
        Op.getOperand(eachArg).dump(&DAG);
      });
    }

    MachineMemOperand *MemOperand = StoreOp->getMemOperand();
    if (!MemOperand) {
      return Op;
    }

    // Because store immediate instructions only have an address offset on 12
    // bits, we need to be careful, and check if we can really use them
    unsigned int addressOpCode = DestinationOperand.getOpcode();
    uint64_t offset = 0x8000000000000000L;
    if (addressOpCode == ISD::ADD) {
      SDValue firstAddOperand = DestinationOperand.getOperand(0);
      SDValue secondAddOperand = DestinationOperand.getOperand(1);

      (void)(canFetchConstantTo(firstAddOperand, &offset) ||
             canFetchConstantTo(secondAddOperand, &offset));
    } else if (const FrameIndexSDNode *FIDN =
                   dyn_cast<FrameIndexSDNode>(DestinationOperand)) {
      offset = (uint64_t)FIDN->getIndex() * 4;
    } else {
      canFetchConstantTo(DestinationOperand, &offset);
    }

    // Source may be either an immediate or a register... The DPU only supports
    // a very restricted set of immediate stores, implying to force a copy to
    // register in most of the cases. 8 and 16 bits operands can be stored as
    // immediate. To avoid re-looping for truncstore stuff, we can immediately
    // promote the operand to its value and let the instruction info play around
    // with 32 bits values.
    uint64_t immediate;
    usesImmediateOperand = canFetchConstantTo(SourceOperand, &immediate);
    if (((addressOpCode == ISD::CopyFromReg) || canBeEncodedOn12Bits(offset)) &&
        usesImmediateOperand) {
      switch (SourceOperand.getSimpleValueType().SimpleTy) {
      case MVT::i1:
      case MVT::i8:
        LLVM_DEBUG({
          dbgs() << "DPU/Lower - immediate operand to store can be truncated "
                    "to 8 bits ";
          SourceOperand.dump(&DAG);
        });
        SourceOperand =
            DAG.getConstant(immediate, dl, EVT(MVT::i32), false /* isTarget */,
                            false /* isOpaque */);
        canUseStoreImmediate = true;
        break;

      case MVT::i16:
        LLVM_DEBUG({
          dbgs() << "DPU/Lower - immediate operand to store can be truncated "
                    "to 16 bits ";
          SourceOperand.dump(&DAG);
        });
        SourceOperand =
            DAG.getConstant(immediate, dl, EVT(MVT::i32), false /* isTarget */,
                            false /* isOpaque */);
        canUseStoreImmediate = true;
        break;
      case MVT::i32:
        if (canBeEncodedOn16Bits(immediate)) {
          LLVM_DEBUG({
            dbgs() << "DPU/Lower - immediate operand to store can be truncated "
                      "to 16 bits ";
            SourceOperand.dump(&DAG);
          });
          SourceOperand =
              DAG.getConstant(immediate, dl, EVT(MVT::i32),
                              false /* isTarget */, false /* isOpaque */);
          canUseStoreImmediate = true;
        } else {
          LLVM_DEBUG({
            dbgs() << "DPU/Lower - immediate operand will be fetched into "
                      "temporary register: ";
            SourceOperand.dump(&DAG);
          });
        }

        break;
      case MVT::i64:
        if (canBeEncodedOn16Bits(immediate)) {
          LLVM_DEBUG({
            dbgs() << "DPU/Lower - immediate operand to store can be truncated "
                      "to 16 bits ";
            SourceOperand.dump(&DAG);
          });
          SourceOperand =
              DAG.getConstant(immediate, dl, EVT(MVT::i64),
                              false /* isTarget */, false /* isOpaque */);
          canUseStoreImmediate = (MemOperand->getAlignment() % 8) == 0;
        } else {
          LLVM_DEBUG({
            dbgs() << "DPU/Lower - immediate operand will be fetched into "
                      "temporary register: ";
            SourceOperand.dump(&DAG);
          });
        }

        break;
      default:
        LLVM_DEBUG({
          dbgs() << "DPU/Lower - immediate operand will be fetched into "
                    "temporary register: ";
          SourceOperand.dump(&DAG);
        });
        break;
      }
    }

    if ((optLevel != CodeGenOpt::None) && canUseStoreImmediate &&
        (AddressSpace == DPUADDR_SPACE::WRAM)) {
      switch (TargetType) {
      case MVT::i1:
      case MVT::i8:
        DPUOpcode = DPUISD::WRAM_STORE_8_IMM;
        break;
      case MVT::i16:
        DPUOpcode = DPUISD::WRAM_STORE_16_IMM;
        break;
      case MVT::i32:
        DPUOpcode = DPUISD::WRAM_STORE_32_IMM;
        break;
      case MVT::i64:
        DPUOpcode = DPUISD::WRAM_STORE_64_IMM;
        break;
      default:
        return SDValue();
      }

      SDValue Ops[] = {Chain, SourceOperand, DestinationOperand, ArgOperand};

      SDVTList VTs = DAG.getVTList(Op.getValueType(), MVT::Glue);
      SDValue result = DAG.getNode(DPUOpcode, dl, VTs, Ops);
      return result;
    } else if (TargetType == MVT::i64) {
      // A security: don't know where exactly the actual address space
      // information is supplied. Making paranoid check sounds wiser right now.
      const Value *Operand = MemOperand->getValue();
      if (Operand) {
        auto *PT = dyn_cast<PointerType>(Operand->getType());
        if (PT) {
          assert((PT->getAddressSpace() == AddressSpace) &&
                 "address spaces are not consistent!!!");
        }
      }

      switch (AddressSpace) {
      default:
        assert(false && "unknown target address space!");

      case DPUADDR_SPACE::WRAM:
        DPUOpcode = ((MemOperand->getAlignment() % 8) == 0)
                        ? DPUISD::WRAM_STORE_64_ALIGNED
                        : DPUISD::WRAM_STORE_64;
        break;

      case DPUADDR_SPACE::MRAM:
        DPUOpcode = DPUISD::MRAM_STORE_64;
        break;
      }

      SDValue Ops[] = {Chain, SourceOperand, DestinationOperand, ArgOperand};

      SDVTList VTs = DAG.getVTList(Op.getValueType(), MVT::Glue);
      SDValue result = DAG.getNode(DPUOpcode, dl, VTs, Ops);
      return result;
    } else {
      return SDValue();
    }
  } else {
    return Op;
  }
}

SDValue DPUTargetLowering::LowerShift(SDValue Op, SelectionDAG &DAG,
                                      int LRA) const {
  SDLoc dl(Op);
  SDValue Operand1 = Op.getOperand(0);
  SDValue Operand2 = Op.getOperand(1);
  MVT::SimpleValueType opType = Op.getValueType().getSimpleVT().SimpleTy;
  const EVT &evt = Op.getValueType();
  uint64_t shiftConstant;

  if (opType == MVT::i32) {
    if (LRA == 0) {
      if ((Operand1.getOpcode() == ISD::CopyFromReg)) {
        RegisterSDNode *registerNode =
            cast<RegisterSDNode>(Operand1.getOperand(1));

        if (registerNode->getReg() == DPU::ID) {
          if (canFetchConstantTo(Operand2, &shiftConstant)) {
            bool canOptimizeIdShift = false;
            unsigned newIdRegister = 0;

            switch (shiftConstant) {
            case 1:
              canOptimizeIdShift = true;
              newIdRegister = DPU::ID2;
              break;
            case 2:
              canOptimizeIdShift = true;
              newIdRegister = DPU::ID4;
              break;
            case 3:
              canOptimizeIdShift = true;
              newIdRegister = DPU::ID8;
              break;
            default:
              break;
            }

            if (canOptimizeIdShift) {
              return DAG.getCopyFromReg(DAG.getEntryNode(), dl, newIdRegister,
                                        evt);
            }
          }
        }
      }
    }

    return Op;
  } else {
    return Op;
  }
}

SDValue DPUTargetLowering::LowerIntrinsic(SDValue Op, SelectionDAG &DAG,
                                          int IntrinsicType) const {
  SDLoc dl(Op);
  SDValue result = SDValue();
  const EVT &evt = Op.getValueType();

  switch (IntrinsicType) {
  default:
    break;
  case ISD::INTRINSIC_WO_CHAIN:
    switch (cast<ConstantSDNode>(Op->getOperand(0))->getZExtValue()) {
    default:
      break;
    case Intrinsic::dpu_tid: {
      result = DAG.getCopyFromReg(DAG.getEntryNode(), dl, DPU::ID, evt);
      break;
    }
    }
    break;
  case ISD::INTRINSIC_W_CHAIN:
    break;
  case ISD::INTRINSIC_VOID:
    break;
  }

  return result;
}

#define GET_INSTRINFO_ENUM
#include "DPUGenInstrInfo.inc"

#include "MCTargetDesc/DPUAsmCondition.h"

static MachineBasicBlock *EmitMul32WithCustomInserter(MachineInstr &MI,
                                                      MachineBasicBlock *BB) {
  // todo __mul32 should have an abstract representation

  const TargetInstrInfo &TII = *BB->getParent()->getSubtarget().getInstrInfo();
  DebugLoc dl = MI.getDebugLoc();
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction::iterator I = ++BB->getIterator();
  MachineFunction *F = BB->getParent();
  MachineBasicBlock *normalMBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *invertedMBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *callMBB = F->CreateMachineBasicBlock(LLVM_BB);
  F->insert(I, normalMBB);
  F->insert(I, invertedMBB);
  F->insert(I, callMBB);
  // Update machine-CFG edges by transferring all successors of the current
  // block to the new block which will contain the Phi node for the select.
  callMBB->splice(callMBB->begin(), BB,
                  std::next(MachineBasicBlock::iterator(MI)), BB->end());
  callMBB->transferSuccessorsAndUpdatePHIs(BB);
  BB->addSuccessor(normalMBB);
  BB->addSuccessor(invertedMBB);
  normalMBB->addSuccessor(callMBB);
  invertedMBB->addSuccessor(callMBB);

  unsigned int Dest = MI.getOperand(0).getReg();
  unsigned int Op1 = MI.getOperand(1).getReg();
  unsigned int Op2 = MI.getOperand(2).getReg();

  BuildMI(BB, dl, TII.get(DPU::JGTUrri))
      .addReg(Op1)
      .addReg(Op2)
      .addMBB(invertedMBB);

  BuildMI(normalMBB, dl, TII.get(DPU::MOVErr), DPU::R18).addReg(Op1);

  BuildMI(normalMBB, dl, TII.get(DPU::MOVErr), DPU::R17).addReg(Op2);

  BuildMI(normalMBB, dl, TII.get(DPU::JUMPi)).addMBB(callMBB);

  BuildMI(invertedMBB, dl, TII.get(DPU::MOVErr), DPU::R18).addReg(Op2);

  BuildMI(invertedMBB, dl, TII.get(DPU::MOVErr), DPU::R17).addReg(Op1);

  BuildMI(*callMBB, callMBB->begin(), dl, TII.get(DPU::CALLri), DPU::RADD)
      .addExternalSymbol("__mul32");

  BuildMI(*callMBB, ++callMBB->begin(), dl, TII.get(DPU::MOVErr), Dest)
      .addReg(DPU::R19);

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return callMBB;
}

static MachineBasicBlock *
EmitDivRem32WithCustomInserter(MachineInstr &MI, MachineBasicBlock *BB,
                               const char *funcName, bool selectRem,
                               bool Op1IsImm, bool Op2IsImm) {
  // todo __udiv32 and __sdiv32 should have an abstract representation
  const TargetInstrInfo &TII = *BB->getParent()->getSubtarget().getInstrInfo();
  DebugLoc dl = MI.getDebugLoc();

  unsigned int Dest = MI.getOperand(0).getReg();
  auto Op1 = MI.getOperand(1);
  auto Op2 = MI.getOperand(2);

  if (Op1IsImm) {
    BuildMI(*BB, MI, dl, TII.get(DPU::MOVEri), DPU::R19).addImm(Op1.getImm());
  } else {
    BuildMI(*BB, MI, dl, TII.get(DPU::MOVErr), DPU::R19).addReg(Op1.getReg());
  }

  if (Op2IsImm) {
    BuildMI(*BB, MI, dl, TII.get(DPU::MOVEri), DPU::R17).addImm(Op2.getImm());
  } else {
    BuildMI(*BB, MI, dl, TII.get(DPU::MOVErr), DPU::R17).addReg(Op2.getReg());
  }

  BuildMI(*BB, MI, dl, TII.get(DPU::CALLri), DPU::RADD)
      .addExternalSymbol(funcName);

  unsigned int ResultReg = selectRem ? DPU::R19 : DPU::R18;

  BuildMI(*BB, MI, dl, TII.get(DPU::MOVErr), Dest).addReg(ResultReg);

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return BB;
}

static MachineBasicBlock *
EmitMul16WithCustomInserter(MachineInstr &MI, MachineBasicBlock *BB,
                            unsigned MulLL, unsigned MulHL, unsigned MulHH) {
  const TargetInstrInfo &TII = *BB->getParent()->getSubtarget().getInstrInfo();
  DebugLoc dl = MI.getDebugLoc();
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction::iterator I = ++BB->getIterator();
  MachineFunction *F = BB->getParent();
  MachineBasicBlock *slowMBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *fastMBB = F->CreateMachineBasicBlock(LLVM_BB);
  F->insert(I, slowMBB);
  F->insert(I, fastMBB);
  // Update machine-CFG edges by transferring all successors of the current
  // block to the new block which will contain the Phi node for the select.
  fastMBB->splice(fastMBB->begin(), BB,
                  std::next(MachineBasicBlock::iterator(MI)), BB->end());
  fastMBB->transferSuccessorsAndUpdatePHIs(BB);
  // Next, add the true and fallthrough blocks as its successors.
  BB->addSuccessor(slowMBB);
  BB->addSuccessor(fastMBB);
  slowMBB->addSuccessor(fastMBB);

  unsigned int Dest = MI.getOperand(0).getReg();
  unsigned int Op1 = MI.getOperand(1).getReg();
  unsigned int Op2 = MI.getOperand(2).getReg();

  BuildMI(BB, dl, TII.get(MulLL), DPU::R17)
      .addReg(Op1)
      .addReg(Op2)
      .addImm(DPUAsmCondition::Small)
      .addMBB(fastMBB);

  BuildMI(slowMBB, dl, TII.get(MulHL), DPU::R18).addReg(Op1).addReg(Op2);

  BuildMI(slowMBB, dl, TII.get(DPU::LSL_ADDrrri), DPU::R17)
      .addReg(DPU::R17)
      .addReg(DPU::R18)
      .addImm(8);

  BuildMI(slowMBB, dl, TII.get(MulHL), DPU::R18).addReg(Op2).addReg(Op1);

  BuildMI(slowMBB, dl, TII.get(DPU::LSL_ADDrrri), DPU::R17)
      .addReg(DPU::R17)
      .addReg(DPU::R18)
      .addImm(8);

  BuildMI(slowMBB, dl, TII.get(MulHH), DPU::R18).addReg(Op1).addReg(Op2);

  BuildMI(slowMBB, dl, TII.get(DPU::LSL_ADDrrri), DPU::R17)
      .addReg(DPU::R17)
      .addReg(DPU::R18)
      .addImm(16);

  BuildMI(*fastMBB, fastMBB->begin(), dl, TII.get(TargetOpcode::COPY), Dest)
      .addReg(DPU::R17);

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return fastMBB;
}

static MachineBasicBlock *EmitSelectWithCustomInserter(MachineInstr &MI,
                                                       MachineBasicBlock *BB) {
  const TargetInstrInfo &TII = *BB->getParent()->getSubtarget().getInstrInfo();
  DebugLoc dl = MI.getDebugLoc();
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction::iterator I = ++BB->getIterator();
  MachineFunction *F = BB->getParent();
  MachineBasicBlock *trueMBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *endMBB = F->CreateMachineBasicBlock(LLVM_BB);
  F->insert(I, trueMBB);
  F->insert(I, endMBB);
  // Update machine-CFG edges by transferring all successors of the current
  // block to the new block which will contain the Phi node for the select.
  endMBB->splice(endMBB->begin(), BB,
                 std::next(MachineBasicBlock::iterator(MI)), BB->end());
  endMBB->transferSuccessorsAndUpdatePHIs(BB);
  // Next, add the true and fallthrough blocks as its successors.
  BB->addSuccessor(trueMBB);
  BB->addSuccessor(endMBB);
  trueMBB->addSuccessor(endMBB);

  unsigned int Dest = MI.getOperand(0).getReg();
  unsigned int CondReg = MI.getOperand(1).getReg();
  unsigned int TrueReg = MI.getOperand(2).getReg();
  unsigned int FalseReg = MI.getOperand(3).getReg();

  MachineRegisterInfo &RI = F->getRegInfo();
  unsigned FalseResultReg = RI.createVirtualRegister(&DPU::GP_REGRegClass);

  BuildMI(BB, dl, TII.get(DPU::ORrrr), FalseResultReg)
      .addReg(CondReg)
      .addReg(FalseReg);

  BuildMI(BB, dl, TII.get(DPU::TmpJcci))
      .addImm(ISD::CondCode::SETEQ)
      .addReg(CondReg)
      .addImm(0)
      .addReg(FalseResultReg)
      .addMBB(endMBB);

  BuildMI(*endMBB, endMBB->begin(), dl, TII.get(DPU::PHI), Dest)
      .addReg(TrueReg)
      .addMBB(trueMBB)
      .addReg(FalseResultReg)
      .addMBB(BB);

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return endMBB;
}

static MachineBasicBlock *
EmitSelect64WithCustomInserter(MachineInstr &MI, MachineBasicBlock *BB) {
  const TargetInstrInfo &TII = *BB->getParent()->getSubtarget().getInstrInfo();
  DebugLoc dl = MI.getDebugLoc();
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction::iterator I = ++BB->getIterator();
  MachineFunction *F = BB->getParent();
  MachineBasicBlock *trueMBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *endMBB = F->CreateMachineBasicBlock(LLVM_BB);
  F->insert(I, trueMBB);
  F->insert(I, endMBB);
  // Update machine-CFG edges by transferring all successors of the current
  // block to the new block which will contain the Phi node for the select.
  endMBB->splice(endMBB->begin(), BB,
                 std::next(MachineBasicBlock::iterator(MI)), BB->end());
  endMBB->transferSuccessorsAndUpdatePHIs(BB);
  // Next, add the true and fallthrough blocks as its successors.
  BB->addSuccessor(trueMBB);
  BB->addSuccessor(endMBB);

  unsigned int Dest = MI.getOperand(0).getReg();
  unsigned int CondReg = MI.getOperand(1).getReg();
  unsigned int TrueReg = MI.getOperand(2).getReg();
  unsigned int FalseReg = MI.getOperand(3).getReg();

  BuildMI(BB, dl, TII.get(DPU::Jcci))
      .addImm(ISD::CondCode::SETEQ)
      .addReg(CondReg)
      .addImm(0)
      .addMBB(endMBB);

  trueMBB->addSuccessor(endMBB);

  BuildMI(*endMBB, endMBB->begin(), dl, TII.get(DPU::PHI), Dest)
      .addReg(TrueReg)
      .addMBB(trueMBB)
      .addReg(FalseReg)
      .addMBB(BB);

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return endMBB;
}

static MachineBasicBlock *
EmitMramSubStoreWithCustomInserter(MachineInstr &MI, MachineBasicBlock *BB,
                                   unsigned int Mask, unsigned int Store,
                                   bool is64Bits) {
  const TargetInstrInfo &TII = *BB->getParent()->getSubtarget().getInstrInfo();
  DebugLoc dl = MI.getDebugLoc();
  MachineFunction *F = BB->getParent();

  MachineRegisterInfo &RI = F->getRegInfo();
  unsigned WramCacheAddrReg = RI.createVirtualRegister(&DPU::GP_REGRegClass);
  unsigned MramAddrReg = RI.createVirtualRegister(&DPU::GP_REGRegClass);
  unsigned MramAddrMSBReg = RI.createVirtualRegister(&DPU::GP_REGRegClass);
  unsigned ExactWramCacheAddrReg =
      RI.createVirtualRegister(&DPU::GP_REGRegClass);

  // todo __sw_cache_buffer should have abstract representation

  BuildMI(*BB, MI, dl, TII.get(DPU::ADDrri), WramCacheAddrReg)
      .addReg(DPU::ID8)
      .addExternalSymbol("__sw_cache_buffer");

  if (MI.getOperand(2).getImm() == 0) {
    BuildMI(*BB, MI, dl, TII.get(DPU::COPY), MramAddrReg).add(MI.getOperand(1));
  } else {
    BuildMI(*BB, MI, dl, TII.get(DPU::ADDrri), MramAddrReg)
        .add(MI.getOperand(1))
        .add(MI.getOperand(2));
  }

  BuildMI(*BB, MI, dl, TII.get(DPU::LDMArri))
      .addReg(WramCacheAddrReg)
      .addReg(MramAddrReg)
      .addImm(0);

  BuildMI(*BB, MI, dl, TII.get(DPU::ANDrri), MramAddrMSBReg)
      .addReg(MramAddrReg)
      .addImm(Mask);

  BuildMI(*BB, MI, dl, TII.get(DPU::ADDrrr), ExactWramCacheAddrReg)
      .addReg(WramCacheAddrReg)
      .addReg(MramAddrMSBReg);

  unsigned int storeRegister;
  if (is64Bits) {
    unsigned storeRegLsb = RI.createVirtualRegister(&DPU::GP_REGRegClass);

    BuildMI(*BB, MI, dl, TII.get(DPU::EXTRACT_SUBREG), storeRegLsb)
        .addReg(MI.getOperand(0).getReg())
        .addImm(DPU::sub_32bit);

    storeRegister = storeRegLsb;
  } else {
    storeRegister = MI.getOperand(0).getReg();
  }

  BuildMI(*BB, MI, dl, TII.get(Store))
      .addReg(ExactWramCacheAddrReg)
      .addImm(0)
      .addReg(storeRegister);

  BuildMI(*BB, MI, dl, TII.get(DPU::SDMArri))
      .addReg(ExactWramCacheAddrReg)
      .addReg(MramAddrReg)
      .addImm(0);

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return BB;
}

static MachineBasicBlock *
EmitMramStoreDoubleWithCustomInserter(MachineInstr &MI, MachineBasicBlock *BB) {

  const TargetInstrInfo &TII = *BB->getParent()->getSubtarget().getInstrInfo();
  DebugLoc dl = MI.getDebugLoc();
  MachineFunction *F = BB->getParent();

  MachineRegisterInfo &RI = F->getRegInfo();
  unsigned WramCacheAddrReg = RI.createVirtualRegister(&DPU::GP_REGRegClass);
  unsigned MramAddrReg = RI.createVirtualRegister(&DPU::GP_REGRegClass);

  // todo __sw_cache_buffer should have abstract representation

  BuildMI(*BB, MI, dl, TII.get(DPU::ADDrri), WramCacheAddrReg)
      .addReg(DPU::ID8)
      .addExternalSymbol("__sw_cache_buffer");

  if (MI.getOperand(2).getImm() == 0) {
    BuildMI(*BB, MI, dl, TII.get(DPU::COPY), MramAddrReg).add(MI.getOperand(1));
  } else {
    BuildMI(*BB, MI, dl, TII.get(DPU::ADDrri), MramAddrReg)
        .add(MI.getOperand(1))
        .add(MI.getOperand(2));
  }

  BuildMI(*BB, MI, dl, TII.get(DPU::SDrir))
      .addReg(WramCacheAddrReg)
      .addImm(0)
      .add(MI.getOperand(0));

  BuildMI(*BB, MI, dl, TII.get(DPU::SDMArri))
      .addReg(WramCacheAddrReg)
      .addReg(MramAddrReg)
      .addImm(0);

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return BB;
}

static MachineBasicBlock *
EmitMramSubLoadWithCustomInserter(MachineInstr &MI, MachineBasicBlock *BB,
                                  unsigned int Mask, unsigned int Load) {
  const TargetInstrInfo &TII = *BB->getParent()->getSubtarget().getInstrInfo();
  DebugLoc dl = MI.getDebugLoc();
  MachineFunction *F = BB->getParent();

  MachineRegisterInfo &RI = F->getRegInfo();
  unsigned WramCacheAddrReg = RI.createVirtualRegister(&DPU::GP_REGRegClass);
  unsigned MramAddrReg = RI.createVirtualRegister(&DPU::GP_REGRegClass);
  unsigned MramAddrMSBReg = RI.createVirtualRegister(&DPU::GP_REGRegClass);
  unsigned ExactWramCacheAddrReg =
      RI.createVirtualRegister(&DPU::GP_REGRegClass);

  // todo __sw_cache_buffer should have abstract representation

  BuildMI(*BB, MI, dl, TII.get(DPU::ADDrri), WramCacheAddrReg)
      .addReg(DPU::ID8)
      .addExternalSymbol("__sw_cache_buffer");

  if (MI.getOperand(2).getImm() == 0) {
    BuildMI(*BB, MI, dl, TII.get(DPU::COPY), MramAddrReg).add(MI.getOperand(1));
  } else {
    BuildMI(*BB, MI, dl, TII.get(DPU::ADDrri), MramAddrReg)
        .add(MI.getOperand(1))
        .add(MI.getOperand(2));
  }

  BuildMI(*BB, MI, dl, TII.get(DPU::LDMArri))
      .addReg(WramCacheAddrReg)
      .addReg(MramAddrReg)
      .addImm(0);

  BuildMI(*BB, MI, dl, TII.get(DPU::ANDrri), MramAddrMSBReg)
      .addReg(MramAddrReg)
      .addImm(Mask);

  BuildMI(*BB, MI, dl, TII.get(DPU::ADDrrr), ExactWramCacheAddrReg)
      .addReg(WramCacheAddrReg)
      .addReg(MramAddrMSBReg);

  BuildMI(*BB, MI, dl, TII.get(Load))
      .add(MI.getOperand(0))
      .addReg(ExactWramCacheAddrReg)
      .addImm(0);

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return BB;
}

static MachineBasicBlock *
EmitAlignedStoreWramDoubleRegisterWithCustomInserter(MachineInstr &MI,
                                                     MachineBasicBlock *BB) {
  const TargetInstrInfo &TII = *BB->getParent()->getSubtarget().getInstrInfo();
  DebugLoc dl = MI.getDebugLoc();

  BuildMI(*BB, MI, dl, TII.get(DPU::SDrir))
      .add(MI.getOperand(1))
      .add(MI.getOperand(2))
      .add(MI.getOperand(0));

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return BB;
}

static MachineBasicBlock *
EmitUnalignedStoreWramDoubleRegisterWithCustomInserter(MachineInstr &MI,
                                                       MachineBasicBlock *BB) {
  const TargetInstrInfo &TII = *BB->getParent()->getSubtarget().getInstrInfo();
  DebugLoc dl = MI.getDebugLoc();
  MachineFunction *F = BB->getParent();
  MachineRegisterInfo &RI = F->getRegInfo();
  int Alignment = -1;
  int offset = MI.getOperand(2).getImm();

  auto MemOp = MI.memoperands_begin();
  auto MemOpEnd = MI.memoperands_end();
  for (; MemOp!= MemOpEnd; MemOp++) {
    int MemOpAlignment = (*MemOp)->getAlignment();
    if ((Alignment == -1) || (Alignment > MemOpAlignment))
      Alignment = MemOpAlignment;
  }
  if (Alignment == 0)
    Alignment = 1;
  else if (Alignment == -1)
    Alignment = 4;

  if ((Alignment % 8) == 0 && (offset % 8) == 0) {
    BuildMI(*BB, MI, dl, TII.get(DPU::SDrir))
      .add(MI.getOperand(1))
      .add(MI.getOperand(2))
      .add(MI.getOperand(0));
  } else {
    if ((Alignment % 4) == 0 && (offset % 4) == 0) {

      BuildMI(*BB, MI, dl, TII.get(DPU::SWrir))
        .add(MI.getOperand(1))
        .add(MI.getOperand(2))
        .addReg(MI.getOperand(0).getReg(), 0, DPU::sub_32bit);

      BuildMI(*BB, MI, dl, TII.get(DPU::SWrir))
        .add(MI.getOperand(1))
        .addImm(MI.getOperand(2).getImm() + 4)
        .addReg(MI.getOperand(0).getReg(), 0, DPU::sub_32bit_hi);
    } else if ((Alignment % 2) == 0 && (offset % 2) == 0) {
      unsigned TmpReg1 = RI.createVirtualRegister(&DPU::GP_REGRegClass);
      unsigned TmpReg2 = RI.createVirtualRegister(&DPU::GP_REGRegClass);
      BuildMI(*BB, MI, dl, TII.get(DPU::SHrir))
        .add(MI.getOperand(1))
        .add(MI.getOperand(2))
        .addReg(MI.getOperand(0).getReg(), 0, DPU::sub_32bit);
      BuildMI(*BB, MI, dl, TII.get(DPU::LSRrri), TmpReg1)
        .addReg(MI.getOperand(0).getReg(), 0, DPU::sub_32bit)
        .addImm(16);
      BuildMI(*BB, MI, dl, TII.get(DPU::SHrir))
        .add(MI.getOperand(1))
        .addImm(MI.getOperand(2).getImm() + 2)
        .addReg(TmpReg1);

      BuildMI(*BB, MI, dl, TII.get(DPU::SHrir))
        .add(MI.getOperand(1))
        .addImm(MI.getOperand(2).getImm() + 4)
        .addReg(MI.getOperand(0).getReg(), 0, DPU::sub_32bit_hi);
      BuildMI(*BB, MI, dl, TII.get(DPU::LSRrri), TmpReg2)
        .addReg(MI.getOperand(0).getReg(), 0, DPU::sub_32bit_hi)
        .addImm(16);
      BuildMI(*BB, MI, dl, TII.get(DPU::SHrir))
        .add(MI.getOperand(1))
        .addImm(MI.getOperand(2).getImm() + 6)
        .addReg(TmpReg2);
    } else if (Alignment == 1) {
      unsigned TmpReg1 = RI.createVirtualRegister(&DPU::GP_REGRegClass);
      unsigned TmpReg2 = RI.createVirtualRegister(&DPU::GP_REGRegClass);
      unsigned TmpReg3 = RI.createVirtualRegister(&DPU::GP_REGRegClass);
      unsigned TmpReg4 = RI.createVirtualRegister(&DPU::GP_REGRegClass);
      unsigned TmpReg5 = RI.createVirtualRegister(&DPU::GP_REGRegClass);
      unsigned TmpReg6 = RI.createVirtualRegister(&DPU::GP_REGRegClass);
      BuildMI(*BB, MI, dl, TII.get(DPU::SBrir))
        .add(MI.getOperand(1))
        .add(MI.getOperand(2))
        .addReg(MI.getOperand(0).getReg(), 0, DPU::sub_32bit);
      BuildMI(*BB, MI, dl, TII.get(DPU::LSRrri), TmpReg1)
        .addReg(MI.getOperand(0).getReg(), 0, DPU::sub_32bit)
        .addImm(8);
      BuildMI(*BB, MI, dl, TII.get(DPU::SBrir))
        .add(MI.getOperand(1))
        .addImm(MI.getOperand(2).getImm() + 1)
        .addReg(TmpReg1);
      BuildMI(*BB, MI, dl, TII.get(DPU::LSRrri), TmpReg2)
        .addReg(TmpReg1)
        .addImm(8);
      BuildMI(*BB, MI, dl, TII.get(DPU::SBrir))
        .add(MI.getOperand(1))
        .addImm(MI.getOperand(2).getImm() + 2)
        .addReg(TmpReg2);
      BuildMI(*BB, MI, dl, TII.get(DPU::LSRrri), TmpReg3)
        .addReg(TmpReg2)
        .addImm(8);
      BuildMI(*BB, MI, dl, TII.get(DPU::SBrir))
        .add(MI.getOperand(1))
        .addImm(MI.getOperand(2).getImm() + 3)
        .addReg(TmpReg3);

      BuildMI(*BB, MI, dl, TII.get(DPU::SBrir))
        .add(MI.getOperand(1))
        .addImm(MI.getOperand(2).getImm() + 4)
        .addReg(MI.getOperand(0).getReg(), 0, DPU::sub_32bit_hi);
      BuildMI(*BB, MI, dl, TII.get(DPU::LSRrri), TmpReg4)
        .addReg(MI.getOperand(0).getReg(), 0, DPU::sub_32bit_hi)
        .addImm(8);
      BuildMI(*BB, MI, dl, TII.get(DPU::SBrir))
        .add(MI.getOperand(1))
        .addImm(MI.getOperand(2).getImm() + 5)
        .addReg(TmpReg4);
      BuildMI(*BB, MI, dl, TII.get(DPU::LSRrri), TmpReg5)
        .addReg(TmpReg4)
        .addImm(8);
      BuildMI(*BB, MI, dl, TII.get(DPU::SBrir))
        .add(MI.getOperand(1))
        .addImm(MI.getOperand(2).getImm() + 6)
        .addReg(TmpReg5);
      BuildMI(*BB, MI, dl, TII.get(DPU::LSRrri), TmpReg6)
        .addReg(TmpReg5)
        .addImm(8);
      BuildMI(*BB, MI, dl, TII.get(DPU::SBrir))
        .add(MI.getOperand(1))
        .addImm(MI.getOperand(2).getImm() + 7)
        .addReg(TmpReg6);
    } else {
      llvm_unreachable("Unexpected Alignment in EmitUnalignedStoreWramDoubleRegisterWithCustomInserter");
    }
  }

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return BB;
}

static MachineBasicBlock *
EmitAlignedStoreWramDoubleImmediateWithCustomInserter(MachineInstr &MI,
                                                      MachineBasicBlock *BB) {
  const TargetInstrInfo &TII = *BB->getParent()->getSubtarget().getInstrInfo();
  DebugLoc dl = MI.getDebugLoc();

  BuildMI(*BB, MI, dl, TII.get(DPU::SDrii))
      .add(MI.getOperand(1))
      .add(MI.getOperand(2))
      .add(MI.getOperand(0));

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return BB;
}

static MachineBasicBlock *
EmitUnalignedLoadWramDoubleRegisterWithCustomInserter(MachineInstr &MI,
                                                      MachineBasicBlock *BB) {
  const TargetInstrInfo &TII = *BB->getParent()->getSubtarget().getInstrInfo();
  DebugLoc dl = MI.getDebugLoc();
  int Alignment = -1;
  int offset = MI.getOperand(2).getImm();
  MachineFunction *F = BB->getParent();
  MachineRegisterInfo &RI = F->getRegInfo();

  auto MemOp = MI.memoperands_begin();
  auto MemOpEnd = MI.memoperands_end();
  for (; MemOp!= MemOpEnd; MemOp++) {
    int MemOpAlignment = (*MemOp)->getAlignment();
    if ((Alignment == -1) || (Alignment > MemOpAlignment))
      Alignment = MemOpAlignment;
  }
  if (Alignment == 0)
    Alignment = 1;
  else if (Alignment == -1)
    Alignment = 4;

  if ((Alignment % 8) == 0 && (offset % 8) == 0) {
      BuildMI(*BB, MI, dl, TII.get(DPU::LDrri))
      .add(MI.getOperand(0))
      .add(MI.getOperand(1))
      .add(MI.getOperand(2));
  } else {
    unsigned DestLsb = RI.createVirtualRegister(&DPU::GP_REGRegClass);
    unsigned DestMsb = RI.createVirtualRegister(&DPU::GP_REGRegClass);
    unsigned UndefReg = RI.createVirtualRegister(&DPU::GP64_REGRegClass);
    unsigned DestPart = RI.createVirtualRegister(&DPU::GP64_REGRegClass);

    if ((Alignment % 4) == 0 && (offset % 4) == 0) {
      BuildMI(*BB, MI, dl, TII.get(DPU::LWrri), DestLsb)
        .add(MI.getOperand(1))
        .add(MI.getOperand(2));

      BuildMI(*BB, MI, dl, TII.get(DPU::LWrri), DestMsb)
        .add(MI.getOperand(1))
        .addImm(MI.getOperand(2).getImm() + 4);
    } else if ((Alignment % 2) == 0 && (offset % 2) == 0) {
      unsigned TmpReg1 = RI.createVirtualRegister(&DPU::GP_REGRegClass);
      unsigned TmpReg2 = RI.createVirtualRegister(&DPU::GP_REGRegClass);
      unsigned TmpReg3 = RI.createVirtualRegister(&DPU::GP_REGRegClass);
      unsigned TmpReg4 = RI.createVirtualRegister(&DPU::GP_REGRegClass);
      BuildMI(*BB, MI, dl, TII.get(DPU::LHUrri), TmpReg1)
        .add(MI.getOperand(1))
        .add(MI.getOperand(2));
      BuildMI(*BB, MI, dl, TII.get(DPU::LHUrri), TmpReg2)
        .add(MI.getOperand(1))
        .addImm(MI.getOperand(2).getImm() + 2);
      BuildMI(*BB, MI, dl, TII.get(DPU::LSL_ADDrrri), DestLsb)
        .addReg(TmpReg1)
        .addReg(TmpReg2)
        .addImm(16);

      BuildMI(*BB, MI, dl, TII.get(DPU::LHUrri), TmpReg3)
        .add(MI.getOperand(1))
        .addImm(MI.getOperand(2).getImm() + 4);
      BuildMI(*BB, MI, dl, TII.get(DPU::LHUrri), TmpReg4)
        .add(MI.getOperand(1))
        .addImm(MI.getOperand(2).getImm() + 6);
      BuildMI(*BB, MI, dl, TII.get(DPU::LSL_ADDrrri), DestMsb)
        .addReg(TmpReg3)
        .addReg(TmpReg4)
        .addImm(16);
    } else if (Alignment == 1) {
      unsigned TmpReg1 = RI.createVirtualRegister(&DPU::GP_REGRegClass);
      unsigned TmpReg2 = RI.createVirtualRegister(&DPU::GP_REGRegClass);
      unsigned TmpReg3 = RI.createVirtualRegister(&DPU::GP_REGRegClass);
      unsigned TmpReg4 = RI.createVirtualRegister(&DPU::GP_REGRegClass);
      unsigned TmpReg5 = RI.createVirtualRegister(&DPU::GP_REGRegClass);
      unsigned TmpReg6 = RI.createVirtualRegister(&DPU::GP_REGRegClass);
      unsigned TmpReg7 = RI.createVirtualRegister(&DPU::GP_REGRegClass);
      unsigned TmpReg8 = RI.createVirtualRegister(&DPU::GP_REGRegClass);
      unsigned TmpReg9 = RI.createVirtualRegister(&DPU::GP_REGRegClass);
      unsigned TmpReg10 = RI.createVirtualRegister(&DPU::GP_REGRegClass);
      unsigned TmpReg11 = RI.createVirtualRegister(&DPU::GP_REGRegClass);
      unsigned TmpReg12 = RI.createVirtualRegister(&DPU::GP_REGRegClass);
      BuildMI(*BB, MI, dl, TII.get(DPU::LBUrri), TmpReg1)
        .add(MI.getOperand(1))
        .add(MI.getOperand(2));
      BuildMI(*BB, MI, dl, TII.get(DPU::LBUrri), TmpReg2)
        .add(MI.getOperand(1))
        .addImm(MI.getOperand(2).getImm() + 1);
      BuildMI(*BB, MI, dl, TII.get(DPU::LSL_ADDrrri), TmpReg3)
        .addReg(TmpReg1)
        .addReg(TmpReg2)
        .addImm(8);
      BuildMI(*BB, MI, dl, TII.get(DPU::LBUrri), TmpReg4)
        .add(MI.getOperand(1))
        .addImm(MI.getOperand(2).getImm() + 2);
      BuildMI(*BB, MI, dl, TII.get(DPU::LSL_ADDrrri), TmpReg5)
        .addReg(TmpReg3)
        .addReg(TmpReg4)
        .addImm(16);
      BuildMI(*BB, MI, dl, TII.get(DPU::LBUrri), TmpReg6)
        .add(MI.getOperand(1))
        .addImm(MI.getOperand(2).getImm() + 3);
      BuildMI(*BB, MI, dl, TII.get(DPU::LSL_ADDrrri), DestLsb)
        .addReg(TmpReg5)
        .addReg(TmpReg6)
        .addImm(24);

      BuildMI(*BB, MI, dl, TII.get(DPU::LBUrri), TmpReg7)
        .add(MI.getOperand(1))
        .addImm(MI.getOperand(2).getImm() + 4);
      BuildMI(*BB, MI, dl, TII.get(DPU::LBUrri), TmpReg8)
        .add(MI.getOperand(1))
        .addImm(MI.getOperand(2).getImm() + 5);
      BuildMI(*BB, MI, dl, TII.get(DPU::LSL_ADDrrri), TmpReg9)
        .addReg(TmpReg7)
        .addReg(TmpReg8)
        .addImm(8);
      BuildMI(*BB, MI, dl, TII.get(DPU::LBUrri), TmpReg10)
        .add(MI.getOperand(1))
        .addImm(MI.getOperand(2).getImm() + 6);
      BuildMI(*BB, MI, dl, TII.get(DPU::LSL_ADDrrri), TmpReg11)
        .addReg(TmpReg9)
        .addReg(TmpReg10)
        .addImm(16);
      BuildMI(*BB, MI, dl, TII.get(DPU::LBUrri), TmpReg12)
        .add(MI.getOperand(1))
        .addImm(MI.getOperand(2).getImm() + 7);
      BuildMI(*BB, MI, dl, TII.get(DPU::LSL_ADDrrri), DestMsb)
        .addReg(TmpReg11)
        .addReg(TmpReg12)
        .addImm(24);
    } else {
      llvm_unreachable("Unexpected Alignment in EmitUnalignedLoadWramDoubleRegisterWithCustomInserter");
    }

    BuildMI(*BB, MI, dl, TII.get(DPU::IMPLICIT_DEF), UndefReg);

    BuildMI(*BB, MI, dl, TII.get(DPU::INSERT_SUBREG), DestPart)
      .addReg(UndefReg)
      .addReg(DestLsb)
      .addImm(DPU::sub_32bit);

    BuildMI(*BB, MI, dl, TII.get(DPU::INSERT_SUBREG), MI.getOperand(0).getReg())
      .addReg(DestPart)
      .addReg(DestMsb)
      .addImm(DPU::sub_32bit_hi);
  }

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return BB;
}

static MachineBasicBlock *
EmitAlignedLoadWramDoubleRegisterWithCustomInserter(MachineInstr &MI,
                                                    MachineBasicBlock *BB) {
  const TargetInstrInfo &TII = *BB->getParent()->getSubtarget().getInstrInfo();
  DebugLoc dl = MI.getDebugLoc();

  BuildMI(*BB, MI, dl, TII.get(DPU::LDrri))
      .add(MI.getOperand(0))
      .add(MI.getOperand(1))
      .add(MI.getOperand(2));

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return BB;
}

static MachineBasicBlock *
EmitLsl64RegisterWithCustomInserter(MachineInstr &MI, MachineBasicBlock *BB) {
  /*
      What we want to generate (with dc.h != rb in that example):
      lslx       __R0, da.l, rb, ?sh32 @+4
      lsl        dc.h, da.h, rb
      or         dc.h, dc.h, __R0
      lsl        dc.l, da.l, rb, ?true @+3
      lsl        dc.h, da.l, rb
      move       dc.l, 0
   */
  const TargetInstrInfo &TII = *BB->getParent()->getSubtarget().getInstrInfo();
  DebugLoc dl = MI.getDebugLoc();
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction::iterator I = ++BB->getIterator();
  MachineFunction *F = BB->getParent();
  MachineBasicBlock *smallShiftMBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *bigShiftMBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *endMBB = F->CreateMachineBasicBlock(LLVM_BB);
  F->insert(I, smallShiftMBB);
  F->insert(I, bigShiftMBB);
  F->insert(I, endMBB);
  // Update machine-CFG edges by transferring all successors of the current
  // block to the new block which will contain the Phi node for the select.
  endMBB->splice(endMBB->begin(), BB,
                 std::next(MachineBasicBlock::iterator(MI)), BB->end());
  endMBB->transferSuccessorsAndUpdatePHIs(BB);

  unsigned int Dest = MI.getOperand(0).getReg();
  unsigned int Op1Reg = MI.getOperand(1).getReg();
  unsigned int ShiftReg = MI.getOperand(2).getReg();

  MachineRegisterInfo &RI = F->getRegInfo();
  unsigned LsbToMsbPartReg = RI.createVirtualRegister(&DPU::GP_REGRegClass);
  unsigned MsbToMsbPartReg = RI.createVirtualRegister(&DPU::GP_REGRegClass);
  unsigned LsbOp1Reg = RI.createVirtualRegister(&DPU::GP_REGRegClass);
  unsigned MsbOp1Reg = RI.createVirtualRegister(&DPU::GP_REGRegClass);

  unsigned BigShiftMsbReg = RI.createVirtualRegister(&DPU::GP_REGRegClass);
  unsigned BigShiftLsbReg = RI.createVirtualRegister(&DPU::GP_REGRegClass);

  unsigned SmallShiftMsbReg = RI.createVirtualRegister(&DPU::GP_REGRegClass);
  unsigned SmallShiftLsbReg = RI.createVirtualRegister(&DPU::GP_REGRegClass);

  unsigned BigShiftResultReg = RI.createVirtualRegister(&DPU::GP64_REGRegClass);
  unsigned SmallShiftResultReg =
      RI.createVirtualRegister(&DPU::GP64_REGRegClass);

  unsigned BigShiftResultPart0Reg =
      RI.createVirtualRegister(&DPU::GP64_REGRegClass);
  unsigned SmallShiftResultPart0Reg =
      RI.createVirtualRegister(&DPU::GP64_REGRegClass);
  unsigned UndefReg = RI.createVirtualRegister(&DPU::GP64_REGRegClass);
  unsigned Undef2Reg = RI.createVirtualRegister(&DPU::GP64_REGRegClass);

  BuildMI(BB, dl, TII.get(DPU::COPY), LsbOp1Reg)
      .addReg(Op1Reg, 0, DPU::sub_32bit);

  BuildMI(BB, dl, TII.get(DPU::LSLXrrrci), LsbToMsbPartReg)
      .addReg(LsbOp1Reg)
      .addReg(ShiftReg)
      .addImm(DPUAsmCondition::Condition::Shift32)
      .addMBB(bigShiftMBB);

  BuildMI(smallShiftMBB, dl, TII.get(DPU::COPY), MsbOp1Reg)
      .addReg(Op1Reg, 0, DPU::sub_32bit_hi);

  BuildMI(smallShiftMBB, dl, TII.get(DPU::LSLrrr), MsbToMsbPartReg)
      .addReg(MsbOp1Reg)
      .addReg(ShiftReg);

  BuildMI(smallShiftMBB, dl, TII.get(DPU::ORrrr), SmallShiftMsbReg)
      .addReg(MsbToMsbPartReg)
      .addReg(LsbToMsbPartReg);

  BuildMI(smallShiftMBB, dl, TII.get(DPU::LSLrrr), SmallShiftLsbReg)
      .addReg(LsbOp1Reg)
      .addReg(ShiftReg);

  BuildMI(smallShiftMBB, dl, TII.get(DPU::IMPLICIT_DEF), Undef2Reg);

  BuildMI(smallShiftMBB, dl, TII.get(DPU::INSERT_SUBREG),
          SmallShiftResultPart0Reg)
      .addReg(Undef2Reg)
      .addReg(SmallShiftLsbReg)
      .addImm(DPU::sub_32bit);

  BuildMI(smallShiftMBB, dl, TII.get(DPU::INSERT_SUBREG), SmallShiftResultReg)
      .addReg(SmallShiftResultPart0Reg)
      .addReg(SmallShiftMsbReg)
      .addImm(DPU::sub_32bit_hi);

  BuildMI(smallShiftMBB, dl, TII.get(DPU::JUMPi)).addMBB(endMBB);

  BuildMI(bigShiftMBB, dl, TII.get(DPU::LSLrrr), BigShiftMsbReg)
      .addReg(LsbOp1Reg)
      .addReg(ShiftReg);

  BuildMI(bigShiftMBB, dl, TII.get(DPU::MOVEri), BigShiftLsbReg).addImm(0);

  BuildMI(bigShiftMBB, dl, TII.get(DPU::IMPLICIT_DEF), UndefReg);

  BuildMI(bigShiftMBB, dl, TII.get(DPU::INSERT_SUBREG), BigShiftResultPart0Reg)
      .addReg(UndefReg)
      .addReg(BigShiftLsbReg)
      .addImm(DPU::sub_32bit);

  BuildMI(bigShiftMBB, dl, TII.get(DPU::INSERT_SUBREG), BigShiftResultReg)
      .addReg(BigShiftResultPart0Reg)
      .addReg(BigShiftMsbReg)
      .addImm(DPU::sub_32bit_hi);

  BB->addSuccessor(smallShiftMBB);
  BB->addSuccessor(bigShiftMBB);
  smallShiftMBB->addSuccessor(endMBB);
  bigShiftMBB->addSuccessor(endMBB);

  BuildMI(*endMBB, endMBB->begin(), dl, TII.get(DPU::PHI), Dest)
      .addReg(BigShiftResultReg)
      .addMBB(bigShiftMBB)
      .addReg(SmallShiftResultReg)
      .addMBB(smallShiftMBB);

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return endMBB;
}

static MachineBasicBlock *
EmitLsl64ImmediateWithCustomInserter(MachineInstr &MI, MachineBasicBlock *BB) {
  const TargetInstrInfo &TII = *BB->getParent()->getSubtarget().getInstrInfo();
  DebugLoc dl = MI.getDebugLoc();
  MachineFunction *F = BB->getParent();
  MachineRegisterInfo &RI = F->getRegInfo();

  unsigned int Dest = MI.getOperand(0).getReg();
  unsigned int Op1Reg = MI.getOperand(1).getReg();
  int64_t ShiftImm = MI.getOperand(2).getImm();

  if (ShiftImm < 32) {
    // ShiftImm < 32
    /*
          lslx    __R0 da.l ShiftImm
          lsl_add dc.h __R0 da.h ShiftImm
          lsl     dc.l da.l ShiftImm
     */
    unsigned Op1Lsb = RI.createVirtualRegister(&DPU::GP_REGRegClass);
    unsigned Op1Msb = RI.createVirtualRegister(&DPU::GP_REGRegClass);
    unsigned ResultLsb = RI.createVirtualRegister(&DPU::GP_REGRegClass);
    unsigned ResultMsbPart = RI.createVirtualRegister(&DPU::GP_REGRegClass);
    unsigned ResultMsb = RI.createVirtualRegister(&DPU::GP_REGRegClass);
    unsigned UndefReg = RI.createVirtualRegister(&DPU::GP64_REGRegClass);
    unsigned ResultPart = RI.createVirtualRegister(&DPU::GP64_REGRegClass);

    BuildMI(*BB, MI, dl, TII.get(DPU::COPY), Op1Lsb)
        .addReg(Op1Reg, 0, DPU::sub_32bit);
    BuildMI(*BB, MI, dl, TII.get(DPU::COPY), Op1Msb)
        .addReg(Op1Reg, 0, DPU::sub_32bit_hi);

    BuildMI(*BB, MI, dl, TII.get(DPU::LSLXrri), ResultMsbPart)
        .addReg(Op1Lsb)
        .addImm(ShiftImm);
    BuildMI(*BB, MI, dl, TII.get(DPU::LSL_ADDrrri), ResultMsb)
        .addReg(ResultMsbPart)
        .addReg(Op1Msb)
        .addImm(ShiftImm);
    BuildMI(*BB, MI, dl, TII.get(DPU::LSLrri), ResultLsb)
        .addReg(Op1Lsb)
        .addImm(ShiftImm);

    BuildMI(*BB, MI, dl, TII.get(DPU::IMPLICIT_DEF), UndefReg);

    BuildMI(*BB, MI, dl, TII.get(DPU::INSERT_SUBREG), ResultPart)
        .addReg(UndefReg)
        .addReg(ResultLsb)
        .addImm(DPU::sub_32bit);

    BuildMI(*BB, MI, dl, TII.get(DPU::INSERT_SUBREG), Dest)
        .addReg(ResultPart)
        .addReg(ResultMsb)
        .addImm(DPU::sub_32bit_hi);
  } else if (ShiftImm > 32) {
    if (ShiftImm >= 64) {
      // ShiftImm >= 64 (should not be generated, undef behavior) ==> Result = 0
      BuildMI(*BB, MI, dl, TII.get(DPU::MOVE64ri), Dest).addImm(0);
    } else {
      // 32 < ShiftImm < 64
      /*
          lsl dc.h da.l ${ShiftImm - 32}
          move dc.l 0
       */
      unsigned ResultLsb = RI.createVirtualRegister(&DPU::GP_REGRegClass);
      unsigned ResultMsb = RI.createVirtualRegister(&DPU::GP_REGRegClass);
      unsigned UndefReg = RI.createVirtualRegister(&DPU::GP64_REGRegClass);
      unsigned ResultPart = RI.createVirtualRegister(&DPU::GP64_REGRegClass);

      BuildMI(*BB, MI, dl, TII.get(DPU::MOVEri), ResultLsb).addImm(0);
      BuildMI(*BB, MI, dl, TII.get(DPU::LSLrri), ResultMsb)
          .addReg(Op1Reg, 0, DPU::sub_32bit)
          .addImm(ShiftImm - 32);

      BuildMI(*BB, MI, dl, TII.get(DPU::IMPLICIT_DEF), UndefReg);

      BuildMI(*BB, MI, dl, TII.get(DPU::INSERT_SUBREG), ResultPart)
          .addReg(UndefReg)
          .addReg(ResultLsb)
          .addImm(DPU::sub_32bit);

      BuildMI(*BB, MI, dl, TII.get(DPU::INSERT_SUBREG), Dest)
          .addReg(ResultPart)
          .addReg(ResultMsb)
          .addImm(DPU::sub_32bit_hi);
    }
  } else {
    // ShiftImm == 32
    /*
        move dc.h da.l
        move dc.l 0
     */
    unsigned ResultLsb = RI.createVirtualRegister(&DPU::GP_REGRegClass);
    unsigned ResultMsb = RI.createVirtualRegister(&DPU::GP_REGRegClass);
    unsigned UndefReg = RI.createVirtualRegister(&DPU::GP64_REGRegClass);
    unsigned ResultPart = RI.createVirtualRegister(&DPU::GP64_REGRegClass);

    BuildMI(*BB, MI, dl, TII.get(DPU::MOVEri), ResultLsb).addImm(0);
    BuildMI(*BB, MI, dl, TII.get(DPU::COPY), ResultMsb)
        .addReg(Op1Reg, 0, DPU::sub_32bit);

    BuildMI(*BB, MI, dl, TII.get(DPU::IMPLICIT_DEF), UndefReg);

    BuildMI(*BB, MI, dl, TII.get(DPU::INSERT_SUBREG), ResultPart)
        .addReg(UndefReg)
        .addReg(ResultLsb)
        .addImm(DPU::sub_32bit);

    BuildMI(*BB, MI, dl, TII.get(DPU::INSERT_SUBREG), Dest)
        .addReg(ResultPart)
        .addReg(ResultMsb)
        .addImm(DPU::sub_32bit_hi);
  }

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return BB;
}

static MachineBasicBlock *EmitShiftRight64RegisterWithCustomInserter(
    MachineInstr &MI, MachineBasicBlock *BB, unsigned int shiftRight,
    unsigned int shiftRightExtended) {
  /*
      What we want to generate (with dc.l != rb in that example):
      lsrx    __R0, da.h, rb, ?sh32 @+4
      lsr     dc.l, da.l, rb
      or      dc.l, dc.l, __R0
      lsr     dc.h, da.h, rb, ?true @+2       // asr     dc.h, da.h, rb, ?true
     @+2 lsr.u   dc, da.h, rb                    // asr.s   dc, da.h, rb
   */
  const TargetInstrInfo &TII = *BB->getParent()->getSubtarget().getInstrInfo();
  DebugLoc dl = MI.getDebugLoc();
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction::iterator I = ++BB->getIterator();
  MachineFunction *F = BB->getParent();
  MachineBasicBlock *smallShiftMBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *bigShiftMBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *endMBB = F->CreateMachineBasicBlock(LLVM_BB);
  F->insert(I, smallShiftMBB);
  F->insert(I, bigShiftMBB);
  F->insert(I, endMBB);
  // Update machine-CFG edges by transferring all successors of the current
  // block to the new block which will contain the Phi node for the select.
  endMBB->splice(endMBB->begin(), BB,
                 std::next(MachineBasicBlock::iterator(MI)), BB->end());
  endMBB->transferSuccessorsAndUpdatePHIs(BB);

  unsigned int Dest = MI.getOperand(0).getReg();
  unsigned int Op1Reg = MI.getOperand(1).getReg();
  unsigned int ShiftReg = MI.getOperand(2).getReg();

  MachineRegisterInfo &RI = F->getRegInfo();
  unsigned LsbOp1Reg = RI.createVirtualRegister(&DPU::GP_REGRegClass);
  unsigned MsbOp1Reg = RI.createVirtualRegister(&DPU::GP_REGRegClass);
  unsigned MsbToLsbPartReg = RI.createVirtualRegister(&DPU::GP_REGRegClass);
  unsigned LsbToLsbPartReg = RI.createVirtualRegister(&DPU::GP_REGRegClass);
  unsigned SmallShiftLsbReg = RI.createVirtualRegister(&DPU::GP_REGRegClass);
  unsigned SmallShiftMsbReg = RI.createVirtualRegister(&DPU::GP_REGRegClass);
  unsigned UndefReg = RI.createVirtualRegister(&DPU::GP64_REGRegClass);
  unsigned SmallShiftResultPart0Reg =
      RI.createVirtualRegister(&DPU::GP64_REGRegClass);
  unsigned SmallShiftResultReg =
      RI.createVirtualRegister(&DPU::GP64_REGRegClass);
  unsigned BigShiftResultReg = RI.createVirtualRegister(&DPU::GP64_REGRegClass);

  BuildMI(BB, dl, TII.get(DPU::COPY), MsbOp1Reg)
      .addReg(Op1Reg, 0, DPU::sub_32bit_hi);

  BuildMI(BB, dl, TII.get(DPU::LSRXrrrci), MsbToLsbPartReg)
      .addReg(MsbOp1Reg)
      .addReg(ShiftReg)
      .addImm(DPUAsmCondition::Condition::Shift32)
      .addMBB(bigShiftMBB);

  BuildMI(smallShiftMBB, dl, TII.get(DPU::COPY), LsbOp1Reg)
      .addReg(Op1Reg, 0, DPU::sub_32bit);

  BuildMI(smallShiftMBB, dl, TII.get(DPU::LSRrrr), LsbToLsbPartReg)
      .addReg(LsbOp1Reg)
      .addReg(ShiftReg);

  BuildMI(smallShiftMBB, dl, TII.get(DPU::ORrrr), SmallShiftLsbReg)
      .addReg(MsbToLsbPartReg)
      .addReg(LsbToLsbPartReg);

  BuildMI(smallShiftMBB, dl, TII.get(shiftRight), SmallShiftMsbReg)
      .addReg(MsbOp1Reg)
      .addReg(ShiftReg);

  BuildMI(smallShiftMBB, dl, TII.get(DPU::IMPLICIT_DEF), UndefReg);

  BuildMI(smallShiftMBB, dl, TII.get(DPU::INSERT_SUBREG),
          SmallShiftResultPart0Reg)
      .addReg(UndefReg)
      .addReg(SmallShiftLsbReg)
      .addImm(DPU::sub_32bit);

  BuildMI(smallShiftMBB, dl, TII.get(DPU::INSERT_SUBREG), SmallShiftResultReg)
      .addReg(SmallShiftResultPart0Reg)
      .addReg(SmallShiftMsbReg)
      .addImm(DPU::sub_32bit_hi);

  BuildMI(smallShiftMBB, dl, TII.get(DPU::JUMPi)).addMBB(endMBB);

  BuildMI(bigShiftMBB, dl, TII.get(shiftRightExtended), BigShiftResultReg)
      .addReg(MsbOp1Reg)
      .addReg(ShiftReg);

  BB->addSuccessor(smallShiftMBB);
  BB->addSuccessor(bigShiftMBB);
  smallShiftMBB->addSuccessor(endMBB);
  bigShiftMBB->addSuccessor(endMBB);

  BuildMI(*endMBB, endMBB->begin(), dl, TII.get(DPU::PHI), Dest)
      .addReg(BigShiftResultReg)
      .addMBB(bigShiftMBB)
      .addReg(SmallShiftResultReg)
      .addMBB(smallShiftMBB);

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return endMBB;
}

static MachineBasicBlock *EmitShiftRight64ImmediateWithCustomInserter(
    MachineInstr &MI, MachineBasicBlock *BB, unsigned int shiftRight,
    unsigned int shiftRightExtended, unsigned int moveExtended) {
  const TargetInstrInfo &TII = *BB->getParent()->getSubtarget().getInstrInfo();
  DebugLoc dl = MI.getDebugLoc();
  MachineFunction *F = BB->getParent();
  MachineRegisterInfo &RI = F->getRegInfo();

  unsigned int Dest = MI.getOperand(0).getReg();
  unsigned int Op1Reg = MI.getOperand(1).getReg();
  int64_t ShiftImm = MI.getOperand(2).getImm();

  if (ShiftImm < 32) {
    // ShiftImm < 32
    /*
          lsrx     __R0 da.h ShiftImm
          lsr_add  dc.l __R0 da.l ShiftImm
          lsr      dc.h da.h ShiftImm       // asr      dc.h da.h ShiftImm
     */
    unsigned Op1Lsb = RI.createVirtualRegister(&DPU::GP_REGRegClass);
    unsigned Op1Msb = RI.createVirtualRegister(&DPU::GP_REGRegClass);
    unsigned ResultLsb = RI.createVirtualRegister(&DPU::GP_REGRegClass);
    unsigned ResultLsbPart = RI.createVirtualRegister(&DPU::GP_REGRegClass);
    unsigned ResultMsb = RI.createVirtualRegister(&DPU::GP_REGRegClass);
    unsigned UndefReg = RI.createVirtualRegister(&DPU::GP64_REGRegClass);
    unsigned ResultPart = RI.createVirtualRegister(&DPU::GP64_REGRegClass);

    BuildMI(*BB, MI, dl, TII.get(DPU::COPY), Op1Lsb)
        .addReg(Op1Reg, 0, DPU::sub_32bit);
    BuildMI(*BB, MI, dl, TII.get(DPU::COPY), Op1Msb)
        .addReg(Op1Reg, 0, DPU::sub_32bit_hi);

    BuildMI(*BB, MI, dl, TII.get(DPU::LSRXrri), ResultLsbPart)
        .addReg(Op1Msb)
        .addImm(ShiftImm);
    BuildMI(*BB, MI, dl, TII.get(DPU::LSR_ADDrrri), ResultLsb)
        .addReg(ResultLsbPart)
        .addReg(Op1Lsb)
        .addImm(ShiftImm);
    BuildMI(*BB, MI, dl, TII.get(shiftRight), ResultMsb)
        .addReg(Op1Msb)
        .addImm(ShiftImm);

    BuildMI(*BB, MI, dl, TII.get(DPU::IMPLICIT_DEF), UndefReg);

    BuildMI(*BB, MI, dl, TII.get(DPU::INSERT_SUBREG), ResultPart)
        .addReg(UndefReg)
        .addReg(ResultLsb)
        .addImm(DPU::sub_32bit);

    BuildMI(*BB, MI, dl, TII.get(DPU::INSERT_SUBREG), Dest)
        .addReg(ResultPart)
        .addReg(ResultMsb)
        .addImm(DPU::sub_32bit_hi);
  } else if (ShiftImm > 32) {
    if (ShiftImm >= 64) {
      // ShiftImm >= 64 (should not be generated, undef behavior) ==> Result = 0
      BuildMI(*BB, MI, dl, TII.get(DPU::MOVE64ri), Dest).addImm(0);
    } else {
      // 32 < ShiftImm < 64
      /*
          lsr.u dc, da.h ${ShiftImm - 32}     // asr.s dc, da.h ${ShiftImm - 32}
       */
      BuildMI(*BB, MI, dl, TII.get(shiftRightExtended), Dest)
          .addReg(Op1Reg, 0, DPU::sub_32bit_hi)
          .addImm(ShiftImm - 32);
    }
  } else {
    // ShiftImm == 32
    /*
        move.u dc u, da.h       //  move.s dc u, da.h
     */
    BuildMI(*BB, MI, dl, TII.get(moveExtended), Dest)
        .addReg(Op1Reg, 0, DPU::sub_32bit_hi);
  }

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return BB;
}

static MachineBasicBlock *
EmitRot64RegisterWithCustomInserter(MachineInstr &MI, MachineBasicBlock *BB,
                                    unsigned int lsN, unsigned int lsNJump,
                                    unsigned int lsNx) {
  /*
      What we want to generate (N = l/r, ie left/right) (with dc.h != rb in that
     example): lsNx    __R0, da.l, rb lsNx    __R1, da.h, rb lsN     dc.h, da.h,
     rb lsN     __R2, da.l, rb  , ?sh32 @+3 or      dc.h, dc.h, __R0 or dc.l,
     __R2, __R1, ?true @+3 or      dc.l, dc.h, __R0 or      dc.h, __R2, __R1
   */
  const TargetInstrInfo &TII = *BB->getParent()->getSubtarget().getInstrInfo();
  DebugLoc dl = MI.getDebugLoc();
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction::iterator I = ++BB->getIterator();
  MachineFunction *F = BB->getParent();
  MachineBasicBlock *smallShiftMBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *bigShiftMBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *endMBB = F->CreateMachineBasicBlock(LLVM_BB);
  F->insert(I, smallShiftMBB);
  F->insert(I, bigShiftMBB);
  F->insert(I, endMBB);
  // Update machine-CFG edges by transferring all successors of the current
  // block to the new block which will contain the Phi node for the select.
  endMBB->splice(endMBB->begin(), BB,
                 std::next(MachineBasicBlock::iterator(MI)), BB->end());
  endMBB->transferSuccessorsAndUpdatePHIs(BB);

  unsigned int Dest = MI.getOperand(0).getReg();
  unsigned int Op1Reg = MI.getOperand(1).getReg();
  unsigned int ShiftReg = MI.getOperand(2).getReg();

  MachineRegisterInfo &RI = F->getRegInfo();
  unsigned Op1Lsb = RI.createVirtualRegister(&DPU::GP_REGRegClass);
  unsigned Op1Msb = RI.createVirtualRegister(&DPU::GP_REGRegClass);
  unsigned Op1LsbShiftX = RI.createVirtualRegister(&DPU::GP_REGRegClass);
  unsigned Op1MsbShiftX = RI.createVirtualRegister(&DPU::GP_REGRegClass);
  unsigned Op1LsbShift = RI.createVirtualRegister(&DPU::GP_REGRegClass);
  unsigned Op1MsbShift = RI.createVirtualRegister(&DPU::GP_REGRegClass);
  unsigned SmallShiftLsbResultReg =
      RI.createVirtualRegister(&DPU::GP_REGRegClass);
  unsigned SmallShiftMsbResultReg =
      RI.createVirtualRegister(&DPU::GP_REGRegClass);
  unsigned BigShiftLsbResultReg =
      RI.createVirtualRegister(&DPU::GP_REGRegClass);
  unsigned BigShiftMsbResultReg =
      RI.createVirtualRegister(&DPU::GP_REGRegClass);
  unsigned BigShiftResultReg = RI.createVirtualRegister(&DPU::GP64_REGRegClass);
  unsigned SmallShiftResultReg =
      RI.createVirtualRegister(&DPU::GP64_REGRegClass);

  unsigned UndefReg = RI.createVirtualRegister(&DPU::GP64_REGRegClass);
  unsigned UndefReg1 = RI.createVirtualRegister(&DPU::GP64_REGRegClass);
  unsigned SmallShiftResultPart0Reg =
      RI.createVirtualRegister(&DPU::GP64_REGRegClass);
  unsigned BigShiftResultPart0Reg =
      RI.createVirtualRegister(&DPU::GP64_REGRegClass);

  BuildMI(*BB, MI, dl, TII.get(DPU::COPY), Op1Lsb)
      .addReg(Op1Reg, 0, DPU::sub_32bit);
  BuildMI(*BB, MI, dl, TII.get(DPU::COPY), Op1Msb)
      .addReg(Op1Reg, 0, DPU::sub_32bit_hi);

  BuildMI(*BB, MI, dl, TII.get(lsNx), Op1MsbShiftX)
      .addReg(Op1Msb)
      .addReg(ShiftReg);
  BuildMI(*BB, MI, dl, TII.get(lsNx), Op1LsbShiftX)
      .addReg(Op1Lsb)
      .addReg(ShiftReg);

  BuildMI(*BB, MI, dl, TII.get(lsN), Op1MsbShift)
      .addReg(Op1Msb)
      .addReg(ShiftReg);
  BuildMI(*BB, MI, dl, TII.get(lsNJump), Op1LsbShift)
      .addReg(Op1Msb)
      .addReg(ShiftReg)
      .addImm(DPUAsmCondition::Condition::Shift32)
      .addMBB(bigShiftMBB);

  BuildMI(smallShiftMBB, dl, TII.get(DPU::ORrrr), SmallShiftMsbResultReg)
      .addReg(Op1MsbShift)
      .addReg(Op1LsbShiftX);
  BuildMI(smallShiftMBB, dl, TII.get(DPU::ORrrr), SmallShiftLsbResultReg)
      .addReg(Op1LsbShift)
      .addReg(Op1MsbShiftX);

  BuildMI(smallShiftMBB, dl, TII.get(DPU::IMPLICIT_DEF), UndefReg);

  BuildMI(smallShiftMBB, dl, TII.get(DPU::INSERT_SUBREG),
          SmallShiftResultPart0Reg)
      .addReg(UndefReg)
      .addReg(SmallShiftLsbResultReg)
      .addImm(DPU::sub_32bit);

  BuildMI(smallShiftMBB, dl, TII.get(DPU::INSERT_SUBREG), SmallShiftResultReg)
      .addReg(SmallShiftResultPart0Reg)
      .addReg(SmallShiftMsbResultReg)
      .addImm(DPU::sub_32bit_hi);

  BuildMI(smallShiftMBB, dl, TII.get(DPU::JUMPi)).addMBB(endMBB);

  BuildMI(bigShiftMBB, dl, TII.get(DPU::ORrrr), BigShiftLsbResultReg)
      .addReg(Op1MsbShift)
      .addReg(Op1LsbShiftX);
  BuildMI(bigShiftMBB, dl, TII.get(DPU::ORrrr), BigShiftMsbResultReg)
      .addReg(Op1LsbShift)
      .addReg(Op1MsbShiftX);

  BuildMI(bigShiftMBB, dl, TII.get(DPU::IMPLICIT_DEF), UndefReg1);

  BuildMI(bigShiftMBB, dl, TII.get(DPU::INSERT_SUBREG), BigShiftResultPart0Reg)
      .addReg(UndefReg1)
      .addReg(BigShiftLsbResultReg)
      .addImm(DPU::sub_32bit);

  BuildMI(bigShiftMBB, dl, TII.get(DPU::INSERT_SUBREG), BigShiftResultReg)
      .addReg(BigShiftResultPart0Reg)
      .addReg(BigShiftMsbResultReg)
      .addImm(DPU::sub_32bit_hi);

  BB->addSuccessor(smallShiftMBB);
  BB->addSuccessor(bigShiftMBB);
  smallShiftMBB->addSuccessor(endMBB);
  bigShiftMBB->addSuccessor(endMBB);

  BuildMI(*endMBB, endMBB->begin(), dl, TII.get(DPU::PHI), Dest)
      .addReg(BigShiftResultReg)
      .addMBB(bigShiftMBB)
      .addReg(SmallShiftResultReg)
      .addMBB(smallShiftMBB);

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return endMBB;
}

static MachineBasicBlock *
EmitRot64ImmediateWithCustomInserter(MachineInstr &MI, MachineBasicBlock *BB,
                                     unsigned int lsNx, unsigned int lsN_add) {
  const TargetInstrInfo &TII = *BB->getParent()->getSubtarget().getInstrInfo();
  DebugLoc dl = MI.getDebugLoc();
  MachineFunction *F = BB->getParent();
  MachineRegisterInfo &RI = F->getRegInfo();

  unsigned int Dest = MI.getOperand(0).getReg();
  unsigned int Op1Reg = MI.getOperand(1).getReg();
  int64_t ShiftImm = MI.getOperand(2).getImm();

  ShiftImm = ShiftImm % 64;

  if (ShiftImm < 32) {
    // ShiftImm < 32 (N = l/r, ie left/right)
    /*
        lsNx    __R0, da.l, imm
        lsNx    __R1, da.h, imm
        lsN_add dc.l, __R1, da.l, imm
        lsN_add dc.h, __R0, da.h, imm
     */
    unsigned Op1Lsb = RI.createVirtualRegister(&DPU::GP_REGRegClass);
    unsigned Op1Msb = RI.createVirtualRegister(&DPU::GP_REGRegClass);
    unsigned ResultLsbPart = RI.createVirtualRegister(&DPU::GP_REGRegClass);
    unsigned ResultLsb = RI.createVirtualRegister(&DPU::GP_REGRegClass);
    unsigned ResultMsbPart = RI.createVirtualRegister(&DPU::GP_REGRegClass);
    unsigned ResultMsb = RI.createVirtualRegister(&DPU::GP_REGRegClass);
    unsigned UndefReg = RI.createVirtualRegister(&DPU::GP64_REGRegClass);
    unsigned ResultPart = RI.createVirtualRegister(&DPU::GP64_REGRegClass);

    BuildMI(*BB, MI, dl, TII.get(DPU::COPY), Op1Lsb)
        .addReg(Op1Reg, 0, DPU::sub_32bit);
    BuildMI(*BB, MI, dl, TII.get(DPU::COPY), Op1Msb)
        .addReg(Op1Reg, 0, DPU::sub_32bit_hi);

    BuildMI(*BB, MI, dl, TII.get(lsNx), ResultLsbPart)
        .addReg(Op1Msb)
        .addImm(ShiftImm);
    BuildMI(*BB, MI, dl, TII.get(lsNx), ResultMsbPart)
        .addReg(Op1Lsb)
        .addImm(ShiftImm);
    BuildMI(*BB, MI, dl, TII.get(lsN_add), ResultLsb)
        .addReg(ResultLsbPart)
        .addReg(Op1Lsb)
        .addImm(ShiftImm);
    BuildMI(*BB, MI, dl, TII.get(lsN_add), ResultMsb)
        .addReg(ResultMsbPart)
        .addReg(Op1Msb)
        .addImm(ShiftImm);

    BuildMI(*BB, MI, dl, TII.get(DPU::IMPLICIT_DEF), UndefReg);

    BuildMI(*BB, MI, dl, TII.get(DPU::INSERT_SUBREG), ResultPart)
        .addReg(UndefReg)
        .addReg(ResultLsb)
        .addImm(DPU::sub_32bit);

    BuildMI(*BB, MI, dl, TII.get(DPU::INSERT_SUBREG), Dest)
        .addReg(ResultPart)
        .addReg(ResultMsb)
        .addImm(DPU::sub_32bit_hi);
  } else if (ShiftImm > 32) {
    // 32 < ShiftImm < 64 (N = l/r, ie left/right) (dc != da for that example)
    /*
        lsNx    __R0, da.l, ${ShiftImm - 32}
        lsNx    __R1, da.h, ${ShiftImm - 32}
        lsN_add dc.h, __R1, da.l, ${ShiftImm - 32}
        lsN_add dc.l, __R0, da.h, ${ShiftImm - 32}
     */
    unsigned Op1Lsb = RI.createVirtualRegister(&DPU::GP_REGRegClass);
    unsigned Op1Msb = RI.createVirtualRegister(&DPU::GP_REGRegClass);
    unsigned ResultLsbPart = RI.createVirtualRegister(&DPU::GP_REGRegClass);
    unsigned ResultLsb = RI.createVirtualRegister(&DPU::GP_REGRegClass);
    unsigned ResultMsbPart = RI.createVirtualRegister(&DPU::GP_REGRegClass);
    unsigned ResultMsb = RI.createVirtualRegister(&DPU::GP_REGRegClass);
    unsigned UndefReg = RI.createVirtualRegister(&DPU::GP64_REGRegClass);
    unsigned ResultPart = RI.createVirtualRegister(&DPU::GP64_REGRegClass);

    BuildMI(*BB, MI, dl, TII.get(DPU::COPY), Op1Lsb)
        .addReg(Op1Reg, 0, DPU::sub_32bit);
    BuildMI(*BB, MI, dl, TII.get(DPU::COPY), Op1Msb)
        .addReg(Op1Reg, 0, DPU::sub_32bit_hi);

    BuildMI(*BB, MI, dl, TII.get(lsNx), ResultLsbPart)
        .addReg(Op1Lsb)
        .addImm(ShiftImm - 32);
    BuildMI(*BB, MI, dl, TII.get(lsNx), ResultMsbPart)
        .addReg(Op1Msb)
        .addImm(ShiftImm - 32);
    BuildMI(*BB, MI, dl, TII.get(lsN_add), ResultLsb)
        .addReg(ResultLsbPart)
        .addReg(Op1Msb)
        .addImm(ShiftImm - 32);
    BuildMI(*BB, MI, dl, TII.get(lsN_add), ResultMsb)
        .addReg(ResultMsbPart)
        .addReg(Op1Lsb)
        .addImm(ShiftImm - 32);

    BuildMI(*BB, MI, dl, TII.get(DPU::IMPLICIT_DEF), UndefReg);

    BuildMI(*BB, MI, dl, TII.get(DPU::INSERT_SUBREG), ResultPart)
        .addReg(UndefReg)
        .addReg(ResultLsb)
        .addImm(DPU::sub_32bit);

    BuildMI(*BB, MI, dl, TII.get(DPU::INSERT_SUBREG), Dest)
        .addReg(ResultPart)
        .addReg(ResultMsb)
        .addImm(DPU::sub_32bit_hi);
  } else {
    // ShiftImm == 32
    /*
        swapd dc da
     */
    BuildMI(*BB, MI, dl, TII.get(DPU::SWAPDrr), Dest).addReg(Op1Reg);
  }

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return BB;
}

static MachineBasicBlock *EmitClz64WithCustomInserter(MachineInstr &MI,
                                                      MachineBasicBlock *BB) {
  /*
      What we want to generate (with dc != da in that example):
      clz.u dc, da.h ?nmax @+3
      clz dc.l da.l
      add dc.l dc.l 32
   */
  const TargetInstrInfo &TII = *BB->getParent()->getSubtarget().getInstrInfo();
  DebugLoc dl = MI.getDebugLoc();
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction::iterator I = ++BB->getIterator();
  MachineFunction *F = BB->getParent();
  MachineBasicBlock *msbAreZerosMBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *endMBB = F->CreateMachineBasicBlock(LLVM_BB);
  F->insert(I, msbAreZerosMBB);
  F->insert(I, endMBB);
  // Update machine-CFG edges by transferring all successors of the current
  // block to the new block which will contain the Phi node for the select.
  endMBB->splice(endMBB->begin(), BB,
                 std::next(MachineBasicBlock::iterator(MI)), BB->end());
  endMBB->transferSuccessorsAndUpdatePHIs(BB);

  BB->addSuccessor(msbAreZerosMBB);
  BB->addSuccessor(endMBB);
  msbAreZerosMBB->addSuccessor(endMBB);

  unsigned int Dest = MI.getOperand(0).getReg();
  unsigned int Op1Reg = MI.getOperand(1).getReg();

  MachineRegisterInfo &RI = F->getRegInfo();
  unsigned FastResultReg = RI.createVirtualRegister(&DPU::GP64_REGRegClass);
  unsigned SlowResultReg = RI.createVirtualRegister(&DPU::GP64_REGRegClass);
  unsigned UndefReg = RI.createVirtualRegister(&DPU::GP64_REGRegClass);
  unsigned SlowResultPart1Reg =
      RI.createVirtualRegister(&DPU::GP64_REGRegClass);
  unsigned SlowResultPartReg = RI.createVirtualRegister(&DPU::GP_REGRegClass);
  unsigned LsbClzReg = RI.createVirtualRegister(&DPU::GP_REGRegClass);

  BuildMI(BB, dl, TII.get(DPU::CLZ_Urrci), FastResultReg)
      .addReg(Op1Reg, 0, DPU::sub_32bit_hi)
      .addImm(DPUAsmCondition::Condition::NotMaximum)
      .addMBB(endMBB);

  BuildMI(msbAreZerosMBB, dl, TII.get(DPU::CLZrr), LsbClzReg)
      .addReg(Op1Reg, 0, DPU::sub_32bit);

  BuildMI(msbAreZerosMBB, dl, TII.get(DPU::ADDrri), SlowResultPartReg)
      .addReg(LsbClzReg)
      .addImm(32);

  BuildMI(msbAreZerosMBB, dl, TII.get(DPU::IMPLICIT_DEF), UndefReg);

  BuildMI(msbAreZerosMBB, dl, TII.get(DPU::INSERT_SUBREG), SlowResultPart1Reg)
      .addReg(UndefReg)
      .addReg(SlowResultPartReg)
      .addImm(DPU::sub_32bit);

  BuildMI(msbAreZerosMBB, dl, TII.get(DPU::INSERT_SUBREG), SlowResultReg)
      .addReg(SlowResultPart1Reg)
      .addReg(FastResultReg, 0, DPU::sub_32bit_hi)
      .addImm(DPU::sub_32bit_hi);

  BuildMI(*endMBB, endMBB->begin(), dl, TII.get(DPU::PHI), Dest)
      .addReg(FastResultReg)
      .addMBB(BB)
      .addReg(SlowResultReg)
      .addMBB(msbAreZerosMBB);

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return endMBB;
}

MachineBasicBlock *
DPUTargetLowering::EmitInstrWithCustomInserter(MachineInstr &MI,
                                               MachineBasicBlock *BB) const {
  switch (MI.getOpcode()) {
  default:
    llvm_unreachable("Unexpected instr type to insert");
  case DPU::MULrr:
    return EmitMul32WithCustomInserter(MI, BB);
  case DPU::Mul16UUrr:
    return EmitMul16WithCustomInserter(MI, BB, DPU::MUL_UL_ULrrrci,
                                       DPU::MUL_UH_ULrrr, DPU::MUL_UH_UHrrr);
  case DPU::Mul16SUrr:
    return EmitMul16WithCustomInserter(MI, BB, DPU::MUL_UL_ULrrrci,
                                       DPU::MUL_SH_ULrrr, DPU::MUL_SH_UHrrr);
  case DPU::Mul16SSrr:
    return EmitMul16WithCustomInserter(MI, BB, DPU::MUL_UL_ULrrrci,
                                       DPU::MUL_SH_ULrrr, DPU::MUL_SH_SHrrr);
  case DPU::UDIVir:
    return EmitDivRem32WithCustomInserter(MI, BB, "__udiv32", false, true,
                                          false);
  case DPU::UDIVri:
    return EmitDivRem32WithCustomInserter(MI, BB, "__udiv32", false, false,
                                          true);
  case DPU::UDIVrr:
    return EmitDivRem32WithCustomInserter(MI, BB, "__udiv32", false, false,
                                          false);
  case DPU::UREMir:
    return EmitDivRem32WithCustomInserter(MI, BB, "__udiv32", true, true,
                                          false);
  case DPU::UREMri:
    return EmitDivRem32WithCustomInserter(MI, BB, "__udiv32", true, false,
                                          true);
  case DPU::UREMrr:
    return EmitDivRem32WithCustomInserter(MI, BB, "__udiv32", true, false,
                                          false);
  case DPU::SDIVir:
    return EmitDivRem32WithCustomInserter(MI, BB, "__sdiv32", false, true,
                                          false);
  case DPU::SDIVri:
    return EmitDivRem32WithCustomInserter(MI, BB, "__sdiv32", false, false,
                                          true);
  case DPU::SDIVrr:
    return EmitDivRem32WithCustomInserter(MI, BB, "__sdiv32", false, false,
                                          false);
  case DPU::SREMir:
    return EmitDivRem32WithCustomInserter(MI, BB, "__sdiv32", true, true,
                                          false);
  case DPU::SREMri:
    return EmitDivRem32WithCustomInserter(MI, BB, "__sdiv32", true, false,
                                          true);
  case DPU::SREMrr:
    return EmitDivRem32WithCustomInserter(MI, BB, "__sdiv32", true, false,
                                          false);
  case DPU::SELECTrr:
    return EmitSelectWithCustomInserter(MI, BB);
  case DPU::SELECT64rr:
    return EmitSelect64WithCustomInserter(MI, BB);
  case DPU::MRAM_STORE_BYTErm:
    return EmitMramSubStoreWithCustomInserter(MI, BB, 7, DPU::SBrir, false);
  case DPU::MRAM_STORE64_BYTErm:
    return EmitMramSubStoreWithCustomInserter(MI, BB, 7, DPU::SBrir, true);
  case DPU::MRAM_STORE_HALFrm:
    return EmitMramSubStoreWithCustomInserter(MI, BB, 6, DPU::SHrir, false);
  case DPU::MRAM_STORE64_HALFrm:
    return EmitMramSubStoreWithCustomInserter(MI, BB, 6, DPU::SHrir, true);
  case DPU::MRAM_STORErm:
  case DPU::MRAM_STORE_WORDrm:
    return EmitMramSubStoreWithCustomInserter(MI, BB, 4, DPU::SWrir, false);
  case DPU::MRAM_STORE64_WORDrm:
    return EmitMramSubStoreWithCustomInserter(MI, BB, 4, DPU::SWrir, true);
  case DPU::MRAM_STORE_DOUBLErm:
    return EmitMramStoreDoubleWithCustomInserter(MI, BB);
  case DPU::MRAM_LOAD_U8mr:
    return EmitMramSubLoadWithCustomInserter(MI, BB, 7, DPU::LBUrri);
  case DPU::MRAM_LOAD_S8mr:
  case DPU::MRAM_LOAD_X8mr:
    return EmitMramSubLoadWithCustomInserter(MI, BB, 7, DPU::LBSrri);
  case DPU::MRAM_LOAD_U16mr:
    return EmitMramSubLoadWithCustomInserter(MI, BB, 6, DPU::LHUrri);
  case DPU::MRAM_LOAD_S16mr:
  case DPU::MRAM_LOAD_X16mr:
    return EmitMramSubLoadWithCustomInserter(MI, BB, 6, DPU::LHSrri);
  case DPU::MRAM_LOAD_S32mr:
  case DPU::MRAM_LOAD_U32mr:
  case DPU::MRAM_LOADmr:
    return EmitMramSubLoadWithCustomInserter(MI, BB, 4, DPU::LWrri);
  case DPU::WRAM_STORE_DOUBLErm:
    return EmitUnalignedStoreWramDoubleRegisterWithCustomInserter(MI, BB);
  case DPU::WRAM_STORE_DOUBLE_ALIGNEDrm:
    return EmitAlignedStoreWramDoubleRegisterWithCustomInserter(MI, BB);
  case DPU::WRAM_STORE_DOUBLEim:
    return EmitAlignedStoreWramDoubleImmediateWithCustomInserter(MI, BB);
  case DPU::WRAM_LOAD_DOUBLErm:
    return EmitUnalignedLoadWramDoubleRegisterWithCustomInserter(MI, BB);
  case DPU::WRAM_LOAD_DOUBLE_ALIGNEDrm:
    return EmitAlignedLoadWramDoubleRegisterWithCustomInserter(MI, BB);
  case DPU::LSL64rr:
    return EmitLsl64RegisterWithCustomInserter(MI, BB);
  case DPU::LSL64ri:
    return EmitLsl64ImmediateWithCustomInserter(MI, BB);
  case DPU::LSR64rr:
    return EmitShiftRight64RegisterWithCustomInserter(MI, BB, DPU::LSRrrr,
                                                      DPU::LSR_Urrr);
  case DPU::LSR64ri:
    return EmitShiftRight64ImmediateWithCustomInserter(
        MI, BB, DPU::LSRrri, DPU::LSR_Urri, DPU::MOVE_Urr);
  case DPU::ASR64rr:
    return EmitShiftRight64RegisterWithCustomInserter(MI, BB, DPU::ASRrrr,
                                                      DPU::ASR_Srrr);
  case DPU::ASR64ri:
    return EmitShiftRight64ImmediateWithCustomInserter(
        MI, BB, DPU::ASRrri, DPU::ASR_Srri, DPU::MOVE_Srr);
  case DPU::ROL64rr:
    return EmitRot64RegisterWithCustomInserter(MI, BB, DPU::LSLrrr,
                                               DPU::LSLrrrci, DPU::LSLXrrr);
  case DPU::ROR64rr:
    return EmitRot64RegisterWithCustomInserter(MI, BB, DPU::LSRrrr,
                                               DPU::LSRrrrci, DPU::LSRXrrr);
  case DPU::ROL64ri:
    return EmitRot64ImmediateWithCustomInserter(MI, BB, DPU::LSLXrri,
                                                DPU::LSL_ADDrrri);
  case DPU::ROR64ri:
    return EmitRot64ImmediateWithCustomInserter(MI, BB, DPU::LSRXrri,
                                                DPU::LSR_ADDrrri);
  case DPU::CLZ64r:
    return EmitClz64WithCustomInserter(MI, BB);
  }
}
