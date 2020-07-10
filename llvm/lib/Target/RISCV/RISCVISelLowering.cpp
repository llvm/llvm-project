//===-- RISCVISelLowering.cpp - RISCV DAG Lowering Implementation  --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that RISCV uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#include "RISCVISelLowering.h"
#include "RISCV.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVRegisterInfo.h"
#include "RISCVSubtarget.h"
#include "RISCVTargetMachine.h"
#include "Utils/RISCVMatInt.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-lower"

STATISTIC(NumTailCalls, "Number of tail calls");

RISCVTargetLowering::RISCVTargetLowering(const TargetMachine &TM,
                                         const RISCVSubtarget &STI)
    : TargetLowering(TM), Subtarget(STI) {

  if (Subtarget.isRV32E())
    report_fatal_error("Codegen not yet implemented for RV32E");

  RISCVABI::ABI ABI = Subtarget.getTargetABI();
  assert(ABI != RISCVABI::ABI_Unknown && "Improperly initialised target ABI");

  if ((ABI == RISCVABI::ABI_ILP32F || ABI == RISCVABI::ABI_LP64F) &&
      !Subtarget.hasStdExtF()) {
    errs() << "Hard-float 'f' ABI can't be used for a target that "
                "doesn't support the F instruction set extension (ignoring "
                          "target-abi)\n";
    ABI = Subtarget.is64Bit() ? RISCVABI::ABI_LP64 : RISCVABI::ABI_ILP32;
  } else if ((ABI == RISCVABI::ABI_ILP32D || ABI == RISCVABI::ABI_LP64D) &&
             !Subtarget.hasStdExtD()) {
    errs() << "Hard-float 'd' ABI can't be used for a target that "
              "doesn't support the D instruction set extension (ignoring "
              "target-abi)\n";
    ABI = Subtarget.is64Bit() ? RISCVABI::ABI_LP64 : RISCVABI::ABI_ILP32;
  }

  switch (ABI) {
  default:
    report_fatal_error("Don't know how to lower this ABI");
  case RISCVABI::ABI_ILP32:
  case RISCVABI::ABI_ILP32F:
  case RISCVABI::ABI_ILP32D:
  case RISCVABI::ABI_LP64:
  case RISCVABI::ABI_LP64F:
  case RISCVABI::ABI_LP64D:
    break;
  }

  MVT XLenVT = Subtarget.getXLenVT();

  // Set up the register classes.
  addRegisterClass(XLenVT, &RISCV::GPRRegClass);

  if (Subtarget.hasStdExtF())
    addRegisterClass(MVT::f32, &RISCV::FPR32RegClass);
  if (Subtarget.hasStdExtD())
    addRegisterClass(MVT::f64, &RISCV::FPR64RegClass);

  // Compute derived properties from the register classes.
  computeRegisterProperties(STI.getRegisterInfo());

  setStackPointerRegisterToSaveRestore(RISCV::X2);

  for (auto N : {ISD::EXTLOAD, ISD::SEXTLOAD, ISD::ZEXTLOAD})
    setLoadExtAction(N, XLenVT, MVT::i1, Promote);

  // TODO: add all necessary setOperationAction calls.
  setOperationAction(ISD::DYNAMIC_STACKALLOC, XLenVT, Expand);

  setOperationAction(ISD::BR_JT, MVT::Other, Expand);
  setOperationAction(ISD::BR_CC, XLenVT, Expand);
  setOperationAction(ISD::SELECT, XLenVT, Custom);
  setOperationAction(ISD::SELECT_CC, XLenVT, Expand);

  setOperationAction(ISD::STACKSAVE, MVT::Other, Expand);
  setOperationAction(ISD::STACKRESTORE, MVT::Other, Expand);

  setOperationAction(ISD::VASTART, MVT::Other, Custom);
  setOperationAction(ISD::VAARG, MVT::Other, Expand);
  setOperationAction(ISD::VACOPY, MVT::Other, Expand);
  setOperationAction(ISD::VAEND, MVT::Other, Expand);

  for (auto VT : {MVT::i1, MVT::i8, MVT::i16})
    setOperationAction(ISD::SIGN_EXTEND_INREG, VT, Expand);

  if (Subtarget.is64Bit()) {
    setOperationAction(ISD::ADD, MVT::i32, Custom);
    setOperationAction(ISD::SUB, MVT::i32, Custom);
    setOperationAction(ISD::SHL, MVT::i32, Custom);
    setOperationAction(ISD::SRA, MVT::i32, Custom);
    setOperationAction(ISD::SRL, MVT::i32, Custom);
  }

  if (!Subtarget.hasStdExtM()) {
    setOperationAction(ISD::MUL, XLenVT, Expand);
    setOperationAction(ISD::MULHS, XLenVT, Expand);
    setOperationAction(ISD::MULHU, XLenVT, Expand);
    setOperationAction(ISD::SDIV, XLenVT, Expand);
    setOperationAction(ISD::UDIV, XLenVT, Expand);
    setOperationAction(ISD::SREM, XLenVT, Expand);
    setOperationAction(ISD::UREM, XLenVT, Expand);
  }

  if (Subtarget.is64Bit() && Subtarget.hasStdExtM()) {
    setOperationAction(ISD::MUL, MVT::i32, Custom);
    setOperationAction(ISD::SDIV, MVT::i32, Custom);
    setOperationAction(ISD::UDIV, MVT::i32, Custom);
    setOperationAction(ISD::UREM, MVT::i32, Custom);
  }

  setOperationAction(ISD::SDIVREM, XLenVT, Expand);
  setOperationAction(ISD::UDIVREM, XLenVT, Expand);
  setOperationAction(ISD::SMUL_LOHI, XLenVT, Expand);
  setOperationAction(ISD::UMUL_LOHI, XLenVT, Expand);

  setOperationAction(ISD::SHL_PARTS, XLenVT, Custom);
  setOperationAction(ISD::SRL_PARTS, XLenVT, Custom);
  setOperationAction(ISD::SRA_PARTS, XLenVT, Custom);

  setOperationAction(ISD::ROTL, XLenVT, Expand);
  setOperationAction(ISD::ROTR, XLenVT, Expand);
  setOperationAction(ISD::BSWAP, XLenVT, Expand);
  setOperationAction(ISD::CTTZ, XLenVT, Expand);
  setOperationAction(ISD::CTLZ, XLenVT, Expand);
  setOperationAction(ISD::CTPOP, XLenVT, Expand);

  ISD::CondCode FPCCToExtend[] = {
      ISD::SETOGT, ISD::SETOGE, ISD::SETONE, ISD::SETUEQ, ISD::SETUGT,
      ISD::SETUGE, ISD::SETULT, ISD::SETULE, ISD::SETUNE, ISD::SETGT,
      ISD::SETGE,  ISD::SETNE};

  ISD::NodeType FPOpToExtend[] = {
      ISD::FSIN, ISD::FCOS, ISD::FSINCOS, ISD::FPOW, ISD::FREM, ISD::FP16_TO_FP,
      ISD::FP_TO_FP16};

  if (Subtarget.hasStdExtF()) {
    setOperationAction(ISD::FMINNUM, MVT::f32, Legal);
    setOperationAction(ISD::FMAXNUM, MVT::f32, Legal);
    for (auto CC : FPCCToExtend)
      setCondCodeAction(CC, MVT::f32, Expand);
    setOperationAction(ISD::SELECT_CC, MVT::f32, Expand);
    setOperationAction(ISD::SELECT, MVT::f32, Custom);
    setOperationAction(ISD::BR_CC, MVT::f32, Expand);
    for (auto Op : FPOpToExtend)
      setOperationAction(Op, MVT::f32, Expand);
    setLoadExtAction(ISD::EXTLOAD, MVT::f32, MVT::f16, Expand);
    setTruncStoreAction(MVT::f32, MVT::f16, Expand);
  }

  if (Subtarget.hasStdExtF() && Subtarget.is64Bit())
    setOperationAction(ISD::BITCAST, MVT::i32, Custom);

  if (Subtarget.hasStdExtD()) {
    setOperationAction(ISD::FMINNUM, MVT::f64, Legal);
    setOperationAction(ISD::FMAXNUM, MVT::f64, Legal);
    for (auto CC : FPCCToExtend)
      setCondCodeAction(CC, MVT::f64, Expand);
    setOperationAction(ISD::SELECT_CC, MVT::f64, Expand);
    setOperationAction(ISD::SELECT, MVT::f64, Custom);
    setOperationAction(ISD::BR_CC, MVT::f64, Expand);
    setLoadExtAction(ISD::EXTLOAD, MVT::f64, MVT::f32, Expand);
    setTruncStoreAction(MVT::f64, MVT::f32, Expand);
    for (auto Op : FPOpToExtend)
      setOperationAction(Op, MVT::f64, Expand);
    setLoadExtAction(ISD::EXTLOAD, MVT::f64, MVT::f16, Expand);
    setTruncStoreAction(MVT::f64, MVT::f16, Expand);
  }

  if (Subtarget.is64Bit() &&
      !(Subtarget.hasStdExtD() || Subtarget.hasStdExtF())) {
    setOperationAction(ISD::FP_TO_UINT, MVT::i32, Custom);
    setOperationAction(ISD::FP_TO_SINT, MVT::i32, Custom);
    setOperationAction(ISD::STRICT_FP_TO_UINT, MVT::i32, Custom);
    setOperationAction(ISD::STRICT_FP_TO_SINT, MVT::i32, Custom);
  }

  setOperationAction(ISD::GlobalAddress, XLenVT, Custom);
  setOperationAction(ISD::BlockAddress, XLenVT, Custom);
  setOperationAction(ISD::ConstantPool, XLenVT, Custom);

  setOperationAction(ISD::GlobalTLSAddress, XLenVT, Custom);

  // TODO: On M-mode only targets, the cycle[h] CSR may not be present.
  // Unfortunately this can't be determined just from the ISA naming string.
  setOperationAction(ISD::READCYCLECOUNTER, MVT::i64,
                     Subtarget.is64Bit() ? Legal : Custom);

  setOperationAction(ISD::TRAP, MVT::Other, Legal);
  setOperationAction(ISD::DEBUGTRAP, MVT::Other, Legal);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::Other, Custom);

  if (Subtarget.hasStdExtA()) {
    setMaxAtomicSizeInBitsSupported(Subtarget.getXLen());
    setMinCmpXchgSizeInBits(32);
  } else {
    setMaxAtomicSizeInBitsSupported(0);
  }

  setBooleanContents(ZeroOrOneBooleanContent);

  // Function alignments.
  const Align FunctionAlignment(Subtarget.hasStdExtC() ? 2 : 4);
  setMinFunctionAlignment(FunctionAlignment);
  setPrefFunctionAlignment(FunctionAlignment);

  // Effectively disable jump table generation.
  setMinimumJumpTableEntries(INT_MAX);

  // Jumps are expensive, compared to logic
  setJumpIsExpensive();

  // We can use any register for comparisons
  setHasMultipleConditionRegisters();
}

EVT RISCVTargetLowering::getSetCCResultType(const DataLayout &DL, LLVMContext &,
                                            EVT VT) const {
  if (!VT.isVector())
    return getPointerTy(DL);
  return VT.changeVectorElementTypeToInteger();
}

bool RISCVTargetLowering::getTgtMemIntrinsic(IntrinsicInfo &Info,
                                             const CallInst &I,
                                             MachineFunction &MF,
                                             unsigned Intrinsic) const {
  switch (Intrinsic) {
  default:
    return false;
  case Intrinsic::riscv_masked_atomicrmw_xchg_i32:
  case Intrinsic::riscv_masked_atomicrmw_add_i32:
  case Intrinsic::riscv_masked_atomicrmw_sub_i32:
  case Intrinsic::riscv_masked_atomicrmw_nand_i32:
  case Intrinsic::riscv_masked_atomicrmw_max_i32:
  case Intrinsic::riscv_masked_atomicrmw_min_i32:
  case Intrinsic::riscv_masked_atomicrmw_umax_i32:
  case Intrinsic::riscv_masked_atomicrmw_umin_i32:
  case Intrinsic::riscv_masked_cmpxchg_i32:
    PointerType *PtrTy = cast<PointerType>(I.getArgOperand(0)->getType());
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::getVT(PtrTy->getElementType());
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.align = Align(4);
    Info.flags = MachineMemOperand::MOLoad | MachineMemOperand::MOStore |
                 MachineMemOperand::MOVolatile;
    return true;
  }
}

bool RISCVTargetLowering::isLegalAddressingMode(const DataLayout &DL,
                                                const AddrMode &AM, Type *Ty,
                                                unsigned AS,
                                                Instruction *I) const {
  // No global is ever allowed as a base.
  if (AM.BaseGV)
    return false;

  // Require a 12-bit signed offset.
  if (!isInt<12>(AM.BaseOffs))
    return false;

  switch (AM.Scale) {
  case 0: // "r+i" or just "i", depending on HasBaseReg.
    break;
  case 1:
    if (!AM.HasBaseReg) // allow "r+i".
      break;
    return false; // disallow "r+r" or "r+r+i".
  default:
    return false;
  }

  return true;
}

bool RISCVTargetLowering::isLegalICmpImmediate(int64_t Imm) const {
  return isInt<12>(Imm);
}

bool RISCVTargetLowering::isLegalAddImmediate(int64_t Imm) const {
  return isInt<12>(Imm);
}

// On RV32, 64-bit integers are split into their high and low parts and held
// in two different registers, so the trunc is free since the low register can
// just be used.
bool RISCVTargetLowering::isTruncateFree(Type *SrcTy, Type *DstTy) const {
  if (Subtarget.is64Bit() || !SrcTy->isIntegerTy() || !DstTy->isIntegerTy())
    return false;
  unsigned SrcBits = SrcTy->getPrimitiveSizeInBits();
  unsigned DestBits = DstTy->getPrimitiveSizeInBits();
  return (SrcBits == 64 && DestBits == 32);
}

bool RISCVTargetLowering::isTruncateFree(EVT SrcVT, EVT DstVT) const {
  if (Subtarget.is64Bit() || SrcVT.isVector() || DstVT.isVector() ||
      !SrcVT.isInteger() || !DstVT.isInteger())
    return false;
  unsigned SrcBits = SrcVT.getSizeInBits();
  unsigned DestBits = DstVT.getSizeInBits();
  return (SrcBits == 64 && DestBits == 32);
}

bool RISCVTargetLowering::isZExtFree(SDValue Val, EVT VT2) const {
  // Zexts are free if they can be combined with a load.
  if (auto *LD = dyn_cast<LoadSDNode>(Val)) {
    EVT MemVT = LD->getMemoryVT();
    if ((MemVT == MVT::i8 || MemVT == MVT::i16 ||
         (Subtarget.is64Bit() && MemVT == MVT::i32)) &&
        (LD->getExtensionType() == ISD::NON_EXTLOAD ||
         LD->getExtensionType() == ISD::ZEXTLOAD))
      return true;
  }

  return TargetLowering::isZExtFree(Val, VT2);
}

bool RISCVTargetLowering::isSExtCheaperThanZExt(EVT SrcVT, EVT DstVT) const {
  return Subtarget.is64Bit() && SrcVT == MVT::i32 && DstVT == MVT::i64;
}

bool RISCVTargetLowering::isFPImmLegal(const APFloat &Imm, EVT VT,
                                       bool ForCodeSize) const {
  if (VT == MVT::f32 && !Subtarget.hasStdExtF())
    return false;
  if (VT == MVT::f64 && !Subtarget.hasStdExtD())
    return false;
  if (Imm.isNegZero())
    return false;
  return Imm.isZero();
}

bool RISCVTargetLowering::hasBitPreservingFPLogic(EVT VT) const {
  return (VT == MVT::f32 && Subtarget.hasStdExtF()) ||
         (VT == MVT::f64 && Subtarget.hasStdExtD());
}

// Changes the condition code and swaps operands if necessary, so the SetCC
// operation matches one of the comparisons supported directly in the RISC-V
// ISA.
static void normaliseSetCC(SDValue &LHS, SDValue &RHS, ISD::CondCode &CC) {
  switch (CC) {
  default:
    break;
  case ISD::SETGT:
  case ISD::SETLE:
  case ISD::SETUGT:
  case ISD::SETULE:
    CC = ISD::getSetCCSwappedOperands(CC);
    std::swap(LHS, RHS);
    break;
  }
}

// Return the RISC-V branch opcode that matches the given DAG integer
// condition code. The CondCode must be one of those supported by the RISC-V
// ISA (see normaliseSetCC).
static unsigned getBranchOpcodeForIntCondCode(ISD::CondCode CC) {
  switch (CC) {
  default:
    llvm_unreachable("Unsupported CondCode");
  case ISD::SETEQ:
    return RISCV::BEQ;
  case ISD::SETNE:
    return RISCV::BNE;
  case ISD::SETLT:
    return RISCV::BLT;
  case ISD::SETGE:
    return RISCV::BGE;
  case ISD::SETULT:
    return RISCV::BLTU;
  case ISD::SETUGE:
    return RISCV::BGEU;
  }
}

SDValue RISCVTargetLowering::LowerOperation(SDValue Op,
                                            SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  default:
    report_fatal_error("unimplemented operand");
  case ISD::GlobalAddress:
    return lowerGlobalAddress(Op, DAG);
  case ISD::BlockAddress:
    return lowerBlockAddress(Op, DAG);
  case ISD::ConstantPool:
    return lowerConstantPool(Op, DAG);
  case ISD::GlobalTLSAddress:
    return lowerGlobalTLSAddress(Op, DAG);
  case ISD::SELECT:
    return lowerSELECT(Op, DAG);
  case ISD::VASTART:
    return lowerVASTART(Op, DAG);
  case ISD::FRAMEADDR:
    return lowerFRAMEADDR(Op, DAG);
  case ISD::RETURNADDR:
    return lowerRETURNADDR(Op, DAG);
  case ISD::SHL_PARTS:
    return lowerShiftLeftParts(Op, DAG);
  case ISD::SRA_PARTS:
    return lowerShiftRightParts(Op, DAG, true);
  case ISD::SRL_PARTS:
    return lowerShiftRightParts(Op, DAG, false);
  case ISD::BITCAST: {
    assert(Subtarget.is64Bit() && Subtarget.hasStdExtF() &&
           "Unexpected custom legalisation");
    SDLoc DL(Op);
    SDValue Op0 = Op.getOperand(0);
    if (Op.getValueType() != MVT::f32 || Op0.getValueType() != MVT::i32)
      return SDValue();
    SDValue NewOp0 = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, Op0);
    SDValue FPConv = DAG.getNode(RISCVISD::FMV_W_X_RV64, DL, MVT::f32, NewOp0);
    return FPConv;
  }
  case ISD::INTRINSIC_WO_CHAIN:
    return LowerINTRINSIC_WO_CHAIN(Op, DAG);
  }
}

static SDValue getTargetNode(GlobalAddressSDNode *N, SDLoc DL, EVT Ty,
                             SelectionDAG &DAG, unsigned Flags) {
  return DAG.getTargetGlobalAddress(N->getGlobal(), DL, Ty, 0, Flags);
}

static SDValue getTargetNode(BlockAddressSDNode *N, SDLoc DL, EVT Ty,
                             SelectionDAG &DAG, unsigned Flags) {
  return DAG.getTargetBlockAddress(N->getBlockAddress(), Ty, N->getOffset(),
                                   Flags);
}

static SDValue getTargetNode(ConstantPoolSDNode *N, SDLoc DL, EVT Ty,
                             SelectionDAG &DAG, unsigned Flags) {
  return DAG.getTargetConstantPool(N->getConstVal(), Ty, N->getAlign(),
                                   N->getOffset(), Flags);
}

template <class NodeTy>
SDValue RISCVTargetLowering::getAddr(NodeTy *N, SelectionDAG &DAG,
                                     bool IsLocal) const {
  SDLoc DL(N);
  EVT Ty = getPointerTy(DAG.getDataLayout());

  if (isPositionIndependent()) {
    SDValue Addr = getTargetNode(N, DL, Ty, DAG, 0);
    if (IsLocal)
      // Use PC-relative addressing to access the symbol. This generates the
      // pattern (PseudoLLA sym), which expands to (addi (auipc %pcrel_hi(sym))
      // %pcrel_lo(auipc)).
      return SDValue(DAG.getMachineNode(RISCV::PseudoLLA, DL, Ty, Addr), 0);

    // Use PC-relative addressing to access the GOT for this symbol, then load
    // the address from the GOT. This generates the pattern (PseudoLA sym),
    // which expands to (ld (addi (auipc %got_pcrel_hi(sym)) %pcrel_lo(auipc))).
    return SDValue(DAG.getMachineNode(RISCV::PseudoLA, DL, Ty, Addr), 0);
  }

  switch (getTargetMachine().getCodeModel()) {
  default:
    report_fatal_error("Unsupported code model for lowering");
  case CodeModel::Small: {
    // Generate a sequence for accessing addresses within the first 2 GiB of
    // address space. This generates the pattern (addi (lui %hi(sym)) %lo(sym)).
    SDValue AddrHi = getTargetNode(N, DL, Ty, DAG, RISCVII::MO_HI);
    SDValue AddrLo = getTargetNode(N, DL, Ty, DAG, RISCVII::MO_LO);
    SDValue MNHi = SDValue(DAG.getMachineNode(RISCV::LUI, DL, Ty, AddrHi), 0);
    return SDValue(DAG.getMachineNode(RISCV::ADDI, DL, Ty, MNHi, AddrLo), 0);
  }
  case CodeModel::Medium: {
    // Generate a sequence for accessing addresses within any 2GiB range within
    // the address space. This generates the pattern (PseudoLLA sym), which
    // expands to (addi (auipc %pcrel_hi(sym)) %pcrel_lo(auipc)).
    SDValue Addr = getTargetNode(N, DL, Ty, DAG, 0);
    return SDValue(DAG.getMachineNode(RISCV::PseudoLLA, DL, Ty, Addr), 0);
  }
  }
}

SDValue RISCVTargetLowering::lowerGlobalAddress(SDValue Op,
                                                SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT Ty = Op.getValueType();
  GlobalAddressSDNode *N = cast<GlobalAddressSDNode>(Op);
  int64_t Offset = N->getOffset();
  MVT XLenVT = Subtarget.getXLenVT();

  const GlobalValue *GV = N->getGlobal();
  bool IsLocal = getTargetMachine().shouldAssumeDSOLocal(*GV->getParent(), GV);
  SDValue Addr = getAddr(N, DAG, IsLocal);

  // In order to maximise the opportunity for common subexpression elimination,
  // emit a separate ADD node for the global address offset instead of folding
  // it in the global address node. Later peephole optimisations may choose to
  // fold it back in when profitable.
  if (Offset != 0)
    return DAG.getNode(ISD::ADD, DL, Ty, Addr,
                       DAG.getConstant(Offset, DL, XLenVT));
  return Addr;
}

SDValue RISCVTargetLowering::lowerBlockAddress(SDValue Op,
                                               SelectionDAG &DAG) const {
  BlockAddressSDNode *N = cast<BlockAddressSDNode>(Op);

  return getAddr(N, DAG);
}

SDValue RISCVTargetLowering::lowerConstantPool(SDValue Op,
                                               SelectionDAG &DAG) const {
  ConstantPoolSDNode *N = cast<ConstantPoolSDNode>(Op);

  return getAddr(N, DAG);
}

SDValue RISCVTargetLowering::getStaticTLSAddr(GlobalAddressSDNode *N,
                                              SelectionDAG &DAG,
                                              bool UseGOT) const {
  SDLoc DL(N);
  EVT Ty = getPointerTy(DAG.getDataLayout());
  const GlobalValue *GV = N->getGlobal();
  MVT XLenVT = Subtarget.getXLenVT();

  if (UseGOT) {
    // Use PC-relative addressing to access the GOT for this TLS symbol, then
    // load the address from the GOT and add the thread pointer. This generates
    // the pattern (PseudoLA_TLS_IE sym), which expands to
    // (ld (auipc %tls_ie_pcrel_hi(sym)) %pcrel_lo(auipc)).
    SDValue Addr = DAG.getTargetGlobalAddress(GV, DL, Ty, 0, 0);
    SDValue Load =
        SDValue(DAG.getMachineNode(RISCV::PseudoLA_TLS_IE, DL, Ty, Addr), 0);

    // Add the thread pointer.
    SDValue TPReg = DAG.getRegister(RISCV::X4, XLenVT);
    return DAG.getNode(ISD::ADD, DL, Ty, Load, TPReg);
  }

  // Generate a sequence for accessing the address relative to the thread
  // pointer, with the appropriate adjustment for the thread pointer offset.
  // This generates the pattern
  // (add (add_tprel (lui %tprel_hi(sym)) tp %tprel_add(sym)) %tprel_lo(sym))
  SDValue AddrHi =
      DAG.getTargetGlobalAddress(GV, DL, Ty, 0, RISCVII::MO_TPREL_HI);
  SDValue AddrAdd =
      DAG.getTargetGlobalAddress(GV, DL, Ty, 0, RISCVII::MO_TPREL_ADD);
  SDValue AddrLo =
      DAG.getTargetGlobalAddress(GV, DL, Ty, 0, RISCVII::MO_TPREL_LO);

  SDValue MNHi = SDValue(DAG.getMachineNode(RISCV::LUI, DL, Ty, AddrHi), 0);
  SDValue TPReg = DAG.getRegister(RISCV::X4, XLenVT);
  SDValue MNAdd = SDValue(
      DAG.getMachineNode(RISCV::PseudoAddTPRel, DL, Ty, MNHi, TPReg, AddrAdd),
      0);
  return SDValue(DAG.getMachineNode(RISCV::ADDI, DL, Ty, MNAdd, AddrLo), 0);
}

SDValue RISCVTargetLowering::getDynamicTLSAddr(GlobalAddressSDNode *N,
                                               SelectionDAG &DAG) const {
  SDLoc DL(N);
  EVT Ty = getPointerTy(DAG.getDataLayout());
  IntegerType *CallTy = Type::getIntNTy(*DAG.getContext(), Ty.getSizeInBits());
  const GlobalValue *GV = N->getGlobal();

  // Use a PC-relative addressing mode to access the global dynamic GOT address.
  // This generates the pattern (PseudoLA_TLS_GD sym), which expands to
  // (addi (auipc %tls_gd_pcrel_hi(sym)) %pcrel_lo(auipc)).
  SDValue Addr = DAG.getTargetGlobalAddress(GV, DL, Ty, 0, 0);
  SDValue Load =
      SDValue(DAG.getMachineNode(RISCV::PseudoLA_TLS_GD, DL, Ty, Addr), 0);

  // Prepare argument list to generate call.
  ArgListTy Args;
  ArgListEntry Entry;
  Entry.Node = Load;
  Entry.Ty = CallTy;
  Args.push_back(Entry);

  // Setup call to __tls_get_addr.
  TargetLowering::CallLoweringInfo CLI(DAG);
  CLI.setDebugLoc(DL)
      .setChain(DAG.getEntryNode())
      .setLibCallee(CallingConv::C, CallTy,
                    DAG.getExternalSymbol("__tls_get_addr", Ty),
                    std::move(Args));

  return LowerCallTo(CLI).first;
}

SDValue RISCVTargetLowering::lowerGlobalTLSAddress(SDValue Op,
                                                   SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT Ty = Op.getValueType();
  GlobalAddressSDNode *N = cast<GlobalAddressSDNode>(Op);
  int64_t Offset = N->getOffset();
  MVT XLenVT = Subtarget.getXLenVT();

  TLSModel::Model Model = getTargetMachine().getTLSModel(N->getGlobal());

  SDValue Addr;
  switch (Model) {
  case TLSModel::LocalExec:
    Addr = getStaticTLSAddr(N, DAG, /*UseGOT=*/false);
    break;
  case TLSModel::InitialExec:
    Addr = getStaticTLSAddr(N, DAG, /*UseGOT=*/true);
    break;
  case TLSModel::LocalDynamic:
  case TLSModel::GeneralDynamic:
    Addr = getDynamicTLSAddr(N, DAG);
    break;
  }

  // In order to maximise the opportunity for common subexpression elimination,
  // emit a separate ADD node for the global address offset instead of folding
  // it in the global address node. Later peephole optimisations may choose to
  // fold it back in when profitable.
  if (Offset != 0)
    return DAG.getNode(ISD::ADD, DL, Ty, Addr,
                       DAG.getConstant(Offset, DL, XLenVT));
  return Addr;
}

SDValue RISCVTargetLowering::lowerSELECT(SDValue Op, SelectionDAG &DAG) const {
  SDValue CondV = Op.getOperand(0);
  SDValue TrueV = Op.getOperand(1);
  SDValue FalseV = Op.getOperand(2);
  SDLoc DL(Op);
  MVT XLenVT = Subtarget.getXLenVT();

  // If the result type is XLenVT and CondV is the output of a SETCC node
  // which also operated on XLenVT inputs, then merge the SETCC node into the
  // lowered RISCVISD::SELECT_CC to take advantage of the integer
  // compare+branch instructions. i.e.:
  // (select (setcc lhs, rhs, cc), truev, falsev)
  // -> (riscvisd::select_cc lhs, rhs, cc, truev, falsev)
  if (Op.getSimpleValueType() == XLenVT && CondV.getOpcode() == ISD::SETCC &&
      CondV.getOperand(0).getSimpleValueType() == XLenVT) {
    SDValue LHS = CondV.getOperand(0);
    SDValue RHS = CondV.getOperand(1);
    auto CC = cast<CondCodeSDNode>(CondV.getOperand(2));
    ISD::CondCode CCVal = CC->get();

    normaliseSetCC(LHS, RHS, CCVal);

    SDValue TargetCC = DAG.getConstant(CCVal, DL, XLenVT);
    SDVTList VTs = DAG.getVTList(Op.getValueType(), MVT::Glue);
    SDValue Ops[] = {LHS, RHS, TargetCC, TrueV, FalseV};
    return DAG.getNode(RISCVISD::SELECT_CC, DL, VTs, Ops);
  }

  // Otherwise:
  // (select condv, truev, falsev)
  // -> (riscvisd::select_cc condv, zero, setne, truev, falsev)
  SDValue Zero = DAG.getConstant(0, DL, XLenVT);
  SDValue SetNE = DAG.getConstant(ISD::SETNE, DL, XLenVT);

  SDVTList VTs = DAG.getVTList(Op.getValueType(), MVT::Glue);
  SDValue Ops[] = {CondV, Zero, SetNE, TrueV, FalseV};

  return DAG.getNode(RISCVISD::SELECT_CC, DL, VTs, Ops);
}

SDValue RISCVTargetLowering::lowerVASTART(SDValue Op, SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  RISCVMachineFunctionInfo *FuncInfo = MF.getInfo<RISCVMachineFunctionInfo>();

  SDLoc DL(Op);
  SDValue FI = DAG.getFrameIndex(FuncInfo->getVarArgsFrameIndex(),
                                 getPointerTy(MF.getDataLayout()));

  // vastart just stores the address of the VarArgsFrameIndex slot into the
  // memory location argument.
  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
  return DAG.getStore(Op.getOperand(0), DL, FI, Op.getOperand(1),
                      MachinePointerInfo(SV));
}

SDValue RISCVTargetLowering::lowerFRAMEADDR(SDValue Op,
                                            SelectionDAG &DAG) const {
  const RISCVRegisterInfo &RI = *Subtarget.getRegisterInfo();
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  MFI.setFrameAddressIsTaken(true);
  Register FrameReg = RI.getFrameRegister(MF);
  int XLenInBytes = Subtarget.getXLen() / 8;

  EVT VT = Op.getValueType();
  SDLoc DL(Op);
  SDValue FrameAddr = DAG.getCopyFromReg(DAG.getEntryNode(), DL, FrameReg, VT);
  unsigned Depth = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  while (Depth--) {
    int Offset = -(XLenInBytes * 2);
    SDValue Ptr = DAG.getNode(ISD::ADD, DL, VT, FrameAddr,
                              DAG.getIntPtrConstant(Offset, DL));
    FrameAddr =
        DAG.getLoad(VT, DL, DAG.getEntryNode(), Ptr, MachinePointerInfo());
  }
  return FrameAddr;
}

SDValue RISCVTargetLowering::lowerRETURNADDR(SDValue Op,
                                             SelectionDAG &DAG) const {
  const RISCVRegisterInfo &RI = *Subtarget.getRegisterInfo();
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  MFI.setReturnAddressIsTaken(true);
  MVT XLenVT = Subtarget.getXLenVT();
  int XLenInBytes = Subtarget.getXLen() / 8;

  if (verifyReturnAddressArgumentIsConstant(Op, DAG))
    return SDValue();

  EVT VT = Op.getValueType();
  SDLoc DL(Op);
  unsigned Depth = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  if (Depth) {
    int Off = -XLenInBytes;
    SDValue FrameAddr = lowerFRAMEADDR(Op, DAG);
    SDValue Offset = DAG.getConstant(Off, DL, VT);
    return DAG.getLoad(VT, DL, DAG.getEntryNode(),
                       DAG.getNode(ISD::ADD, DL, VT, FrameAddr, Offset),
                       MachinePointerInfo());
  }

  // Return the value of the return address register, marking it an implicit
  // live-in.
  Register Reg = MF.addLiveIn(RI.getRARegister(), getRegClassFor(XLenVT));
  return DAG.getCopyFromReg(DAG.getEntryNode(), DL, Reg, XLenVT);
}

SDValue RISCVTargetLowering::lowerShiftLeftParts(SDValue Op,
                                                 SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue Lo = Op.getOperand(0);
  SDValue Hi = Op.getOperand(1);
  SDValue Shamt = Op.getOperand(2);
  EVT VT = Lo.getValueType();

  // if Shamt-XLEN < 0: // Shamt < XLEN
  //   Lo = Lo << Shamt
  //   Hi = (Hi << Shamt) | ((Lo >>u 1) >>u (XLEN-1 - Shamt))
  // else:
  //   Lo = 0
  //   Hi = Lo << (Shamt-XLEN)

  SDValue Zero = DAG.getConstant(0, DL, VT);
  SDValue One = DAG.getConstant(1, DL, VT);
  SDValue MinusXLen = DAG.getConstant(-(int)Subtarget.getXLen(), DL, VT);
  SDValue XLenMinus1 = DAG.getConstant(Subtarget.getXLen() - 1, DL, VT);
  SDValue ShamtMinusXLen = DAG.getNode(ISD::ADD, DL, VT, Shamt, MinusXLen);
  SDValue XLenMinus1Shamt = DAG.getNode(ISD::SUB, DL, VT, XLenMinus1, Shamt);

  SDValue LoTrue = DAG.getNode(ISD::SHL, DL, VT, Lo, Shamt);
  SDValue ShiftRight1Lo = DAG.getNode(ISD::SRL, DL, VT, Lo, One);
  SDValue ShiftRightLo =
      DAG.getNode(ISD::SRL, DL, VT, ShiftRight1Lo, XLenMinus1Shamt);
  SDValue ShiftLeftHi = DAG.getNode(ISD::SHL, DL, VT, Hi, Shamt);
  SDValue HiTrue = DAG.getNode(ISD::OR, DL, VT, ShiftLeftHi, ShiftRightLo);
  SDValue HiFalse = DAG.getNode(ISD::SHL, DL, VT, Lo, ShamtMinusXLen);

  SDValue CC = DAG.getSetCC(DL, VT, ShamtMinusXLen, Zero, ISD::SETLT);

  Lo = DAG.getNode(ISD::SELECT, DL, VT, CC, LoTrue, Zero);
  Hi = DAG.getNode(ISD::SELECT, DL, VT, CC, HiTrue, HiFalse);

  SDValue Parts[2] = {Lo, Hi};
  return DAG.getMergeValues(Parts, DL);
}

SDValue RISCVTargetLowering::lowerShiftRightParts(SDValue Op, SelectionDAG &DAG,
                                                  bool IsSRA) const {
  SDLoc DL(Op);
  SDValue Lo = Op.getOperand(0);
  SDValue Hi = Op.getOperand(1);
  SDValue Shamt = Op.getOperand(2);
  EVT VT = Lo.getValueType();

  // SRA expansion:
  //   if Shamt-XLEN < 0: // Shamt < XLEN
  //     Lo = (Lo >>u Shamt) | ((Hi << 1) << (XLEN-1 - Shamt))
  //     Hi = Hi >>s Shamt
  //   else:
  //     Lo = Hi >>s (Shamt-XLEN);
  //     Hi = Hi >>s (XLEN-1)
  //
  // SRL expansion:
  //   if Shamt-XLEN < 0: // Shamt < XLEN
  //     Lo = (Lo >>u Shamt) | ((Hi << 1) << (XLEN-1 - Shamt))
  //     Hi = Hi >>u Shamt
  //   else:
  //     Lo = Hi >>u (Shamt-XLEN);
  //     Hi = 0;

  unsigned ShiftRightOp = IsSRA ? ISD::SRA : ISD::SRL;

  SDValue Zero = DAG.getConstant(0, DL, VT);
  SDValue One = DAG.getConstant(1, DL, VT);
  SDValue MinusXLen = DAG.getConstant(-(int)Subtarget.getXLen(), DL, VT);
  SDValue XLenMinus1 = DAG.getConstant(Subtarget.getXLen() - 1, DL, VT);
  SDValue ShamtMinusXLen = DAG.getNode(ISD::ADD, DL, VT, Shamt, MinusXLen);
  SDValue XLenMinus1Shamt = DAG.getNode(ISD::SUB, DL, VT, XLenMinus1, Shamt);

  SDValue ShiftRightLo = DAG.getNode(ISD::SRL, DL, VT, Lo, Shamt);
  SDValue ShiftLeftHi1 = DAG.getNode(ISD::SHL, DL, VT, Hi, One);
  SDValue ShiftLeftHi =
      DAG.getNode(ISD::SHL, DL, VT, ShiftLeftHi1, XLenMinus1Shamt);
  SDValue LoTrue = DAG.getNode(ISD::OR, DL, VT, ShiftRightLo, ShiftLeftHi);
  SDValue HiTrue = DAG.getNode(ShiftRightOp, DL, VT, Hi, Shamt);
  SDValue LoFalse = DAG.getNode(ShiftRightOp, DL, VT, Hi, ShamtMinusXLen);
  SDValue HiFalse =
      IsSRA ? DAG.getNode(ISD::SRA, DL, VT, Hi, XLenMinus1) : Zero;

  SDValue CC = DAG.getSetCC(DL, VT, ShamtMinusXLen, Zero, ISD::SETLT);

  Lo = DAG.getNode(ISD::SELECT, DL, VT, CC, LoTrue, LoFalse);
  Hi = DAG.getNode(ISD::SELECT, DL, VT, CC, HiTrue, HiFalse);

  SDValue Parts[2] = {Lo, Hi};
  return DAG.getMergeValues(Parts, DL);
}

SDValue RISCVTargetLowering::LowerINTRINSIC_WO_CHAIN(SDValue Op,
                                                     SelectionDAG &DAG) const {
  unsigned IntNo = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  SDLoc DL(Op);
  switch (IntNo) {
  default:
    return SDValue();    // Don't custom lower most intrinsics.
  case Intrinsic::thread_pointer: {
    EVT PtrVT = getPointerTy(DAG.getDataLayout());
    return DAG.getRegister(RISCV::X4, PtrVT);
  }
  }
}

// Returns the opcode of the target-specific SDNode that implements the 32-bit
// form of the given Opcode.
static RISCVISD::NodeType getRISCVWOpcode(unsigned Opcode) {
  switch (Opcode) {
  default:
    llvm_unreachable("Unexpected opcode");
  case ISD::SHL:
    return RISCVISD::SLLW;
  case ISD::SRA:
    return RISCVISD::SRAW;
  case ISD::SRL:
    return RISCVISD::SRLW;
  case ISD::SDIV:
    return RISCVISD::DIVW;
  case ISD::UDIV:
    return RISCVISD::DIVUW;
  case ISD::UREM:
    return RISCVISD::REMUW;
  }
}

// Converts the given 32-bit operation to a target-specific SelectionDAG node.
// Because i32 isn't a legal type for RV64, these operations would otherwise
// be promoted to i64, making it difficult to select the SLLW/DIVUW/.../*W
// later one because the fact the operation was originally of type i32 is
// lost.
static SDValue customLegalizeToWOp(SDNode *N, SelectionDAG &DAG) {
  SDLoc DL(N);
  RISCVISD::NodeType WOpcode = getRISCVWOpcode(N->getOpcode());
  SDValue NewOp0 = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(0));
  SDValue NewOp1 = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(1));
  SDValue NewRes = DAG.getNode(WOpcode, DL, MVT::i64, NewOp0, NewOp1);
  // ReplaceNodeResults requires we maintain the same type for the return value.
  return DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, NewRes);
}

// Converts the given 32-bit operation to a i64 operation with signed extension
// semantic to reduce the signed extension instructions.
static SDValue customLegalizeToWOpWithSExt(SDNode *N, SelectionDAG &DAG) {
  SDLoc DL(N);
  SDValue NewOp0 = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(0));
  SDValue NewOp1 = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(1));
  SDValue NewWOp = DAG.getNode(N->getOpcode(), DL, MVT::i64, NewOp0, NewOp1);
  SDValue NewRes = DAG.getNode(ISD::SIGN_EXTEND_INREG, DL, MVT::i64, NewWOp,
                               DAG.getValueType(MVT::i32));
  return DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, NewRes);
}

void RISCVTargetLowering::ReplaceNodeResults(SDNode *N,
                                             SmallVectorImpl<SDValue> &Results,
                                             SelectionDAG &DAG) const {
  SDLoc DL(N);
  switch (N->getOpcode()) {
  default:
    llvm_unreachable("Don't know how to custom type legalize this operation!");
  case ISD::STRICT_FP_TO_SINT:
  case ISD::STRICT_FP_TO_UINT:
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT: {
    bool IsStrict = N->isStrictFPOpcode();
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
    SDValue Op0 = IsStrict ? N->getOperand(1) : N->getOperand(0);
    RTLIB::Libcall LC;
    if (N->getOpcode() == ISD::FP_TO_SINT ||
        N->getOpcode() == ISD::STRICT_FP_TO_SINT)
      LC = RTLIB::getFPTOSINT(Op0.getValueType(), N->getValueType(0));
    else
      LC = RTLIB::getFPTOUINT(Op0.getValueType(), N->getValueType(0));
    MakeLibCallOptions CallOptions;
    EVT OpVT = Op0.getValueType();
    CallOptions.setTypeListBeforeSoften(OpVT, N->getValueType(0), true);
    SDValue Chain = IsStrict ? N->getOperand(0) : SDValue();
    SDValue Result;
    std::tie(Result, Chain) =
        makeLibCall(DAG, LC, N->getValueType(0), Op0, CallOptions, DL, Chain);
    Results.push_back(Result);
    if (IsStrict)
      Results.push_back(Chain);
    break;
  }
  case ISD::READCYCLECOUNTER: {
    assert(!Subtarget.is64Bit() &&
           "READCYCLECOUNTER only has custom type legalization on riscv32");

    SDVTList VTs = DAG.getVTList(MVT::i32, MVT::i32, MVT::Other);
    SDValue RCW =
        DAG.getNode(RISCVISD::READ_CYCLE_WIDE, DL, VTs, N->getOperand(0));

    Results.push_back(
        DAG.getNode(ISD::BUILD_PAIR, DL, MVT::i64, RCW, RCW.getValue(1)));
    Results.push_back(RCW.getValue(2));
    break;
  }
  case ISD::ADD:
  case ISD::SUB:
  case ISD::MUL:
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
    if (N->getOperand(1).getOpcode() == ISD::Constant)
      return;
    Results.push_back(customLegalizeToWOpWithSExt(N, DAG));
    break;
  case ISD::SHL:
  case ISD::SRA:
  case ISD::SRL:
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
    if (N->getOperand(1).getOpcode() == ISD::Constant)
      return;
    Results.push_back(customLegalizeToWOp(N, DAG));
    break;
  case ISD::SDIV:
  case ISD::UDIV:
  case ISD::UREM:
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           Subtarget.hasStdExtM() && "Unexpected custom legalisation");
    if (N->getOperand(0).getOpcode() == ISD::Constant ||
        N->getOperand(1).getOpcode() == ISD::Constant)
      return;
    Results.push_back(customLegalizeToWOp(N, DAG));
    break;
  case ISD::BITCAST: {
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           Subtarget.hasStdExtF() && "Unexpected custom legalisation");
    SDLoc DL(N);
    SDValue Op0 = N->getOperand(0);
    if (Op0.getValueType() != MVT::f32)
      return;
    SDValue FPConv =
        DAG.getNode(RISCVISD::FMV_X_ANYEXTW_RV64, DL, MVT::i64, Op0);
    Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, FPConv));
    break;
  }
  }
}

SDValue RISCVTargetLowering::PerformDAGCombine(SDNode *N,
                                               DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;

  switch (N->getOpcode()) {
  default:
    break;
  case RISCVISD::SplitF64: {
    SDValue Op0 = N->getOperand(0);
    // If the input to SplitF64 is just BuildPairF64 then the operation is
    // redundant. Instead, use BuildPairF64's operands directly.
    if (Op0->getOpcode() == RISCVISD::BuildPairF64)
      return DCI.CombineTo(N, Op0.getOperand(0), Op0.getOperand(1));

    SDLoc DL(N);

    // It's cheaper to materialise two 32-bit integers than to load a double
    // from the constant pool and transfer it to integer registers through the
    // stack.
    if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(Op0)) {
      APInt V = C->getValueAPF().bitcastToAPInt();
      SDValue Lo = DAG.getConstant(V.trunc(32), DL, MVT::i32);
      SDValue Hi = DAG.getConstant(V.lshr(32).trunc(32), DL, MVT::i32);
      return DCI.CombineTo(N, Lo, Hi);
    }

    // This is a target-specific version of a DAGCombine performed in
    // DAGCombiner::visitBITCAST. It performs the equivalent of:
    // fold (bitconvert (fneg x)) -> (xor (bitconvert x), signbit)
    // fold (bitconvert (fabs x)) -> (and (bitconvert x), (not signbit))
    if (!(Op0.getOpcode() == ISD::FNEG || Op0.getOpcode() == ISD::FABS) ||
        !Op0.getNode()->hasOneUse())
      break;
    SDValue NewSplitF64 =
        DAG.getNode(RISCVISD::SplitF64, DL, DAG.getVTList(MVT::i32, MVT::i32),
                    Op0.getOperand(0));
    SDValue Lo = NewSplitF64.getValue(0);
    SDValue Hi = NewSplitF64.getValue(1);
    APInt SignBit = APInt::getSignMask(32);
    if (Op0.getOpcode() == ISD::FNEG) {
      SDValue NewHi = DAG.getNode(ISD::XOR, DL, MVT::i32, Hi,
                                  DAG.getConstant(SignBit, DL, MVT::i32));
      return DCI.CombineTo(N, Lo, NewHi);
    }
    assert(Op0.getOpcode() == ISD::FABS);
    SDValue NewHi = DAG.getNode(ISD::AND, DL, MVT::i32, Hi,
                                DAG.getConstant(~SignBit, DL, MVT::i32));
    return DCI.CombineTo(N, Lo, NewHi);
  }
  case RISCVISD::SLLW:
  case RISCVISD::SRAW:
  case RISCVISD::SRLW: {
    // Only the lower 32 bits of LHS and lower 5 bits of RHS are read.
    SDValue LHS = N->getOperand(0);
    SDValue RHS = N->getOperand(1);
    APInt LHSMask = APInt::getLowBitsSet(LHS.getValueSizeInBits(), 32);
    APInt RHSMask = APInt::getLowBitsSet(RHS.getValueSizeInBits(), 5);
    if ((SimplifyDemandedBits(N->getOperand(0), LHSMask, DCI)) ||
        (SimplifyDemandedBits(N->getOperand(1), RHSMask, DCI)))
      return SDValue();
    break;
  }
  case RISCVISD::FMV_X_ANYEXTW_RV64: {
    SDLoc DL(N);
    SDValue Op0 = N->getOperand(0);
    // If the input to FMV_X_ANYEXTW_RV64 is just FMV_W_X_RV64 then the
    // conversion is unnecessary and can be replaced with an ANY_EXTEND
    // of the FMV_W_X_RV64 operand.
    if (Op0->getOpcode() == RISCVISD::FMV_W_X_RV64) {
      SDValue AExtOp =
          DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, Op0.getOperand(0));
      return DCI.CombineTo(N, AExtOp);
    }

    // This is a target-specific version of a DAGCombine performed in
    // DAGCombiner::visitBITCAST. It performs the equivalent of:
    // fold (bitconvert (fneg x)) -> (xor (bitconvert x), signbit)
    // fold (bitconvert (fabs x)) -> (and (bitconvert x), (not signbit))
    if (!(Op0.getOpcode() == ISD::FNEG || Op0.getOpcode() == ISD::FABS) ||
        !Op0.getNode()->hasOneUse())
      break;
    SDValue NewFMV = DAG.getNode(RISCVISD::FMV_X_ANYEXTW_RV64, DL, MVT::i64,
                                 Op0.getOperand(0));
    APInt SignBit = APInt::getSignMask(32).sext(64);
    if (Op0.getOpcode() == ISD::FNEG) {
      return DCI.CombineTo(N,
                           DAG.getNode(ISD::XOR, DL, MVT::i64, NewFMV,
                                       DAG.getConstant(SignBit, DL, MVT::i64)));
    }
    assert(Op0.getOpcode() == ISD::FABS);
    return DCI.CombineTo(N,
                         DAG.getNode(ISD::AND, DL, MVT::i64, NewFMV,
                                     DAG.getConstant(~SignBit, DL, MVT::i64)));
  }
  }

  return SDValue();
}

bool RISCVTargetLowering::isDesirableToCommuteWithShift(
    const SDNode *N, CombineLevel Level) const {
  // The following folds are only desirable if `(OP _, c1 << c2)` can be
  // materialised in fewer instructions than `(OP _, c1)`:
  //
  //   (shl (add x, c1), c2) -> (add (shl x, c2), c1 << c2)
  //   (shl (or x, c1), c2) -> (or (shl x, c2), c1 << c2)
  SDValue N0 = N->getOperand(0);
  EVT Ty = N0.getValueType();
  if (Ty.isScalarInteger() &&
      (N0.getOpcode() == ISD::ADD || N0.getOpcode() == ISD::OR)) {
    auto *C1 = dyn_cast<ConstantSDNode>(N0->getOperand(1));
    auto *C2 = dyn_cast<ConstantSDNode>(N->getOperand(1));
    if (C1 && C2) {
      APInt C1Int = C1->getAPIntValue();
      APInt ShiftedC1Int = C1Int << C2->getAPIntValue();

      // We can materialise `c1 << c2` into an add immediate, so it's "free",
      // and the combine should happen, to potentially allow further combines
      // later.
      if (ShiftedC1Int.getMinSignedBits() <= 64 &&
          isLegalAddImmediate(ShiftedC1Int.getSExtValue()))
        return true;

      // We can materialise `c1` in an add immediate, so it's "free", and the
      // combine should be prevented.
      if (C1Int.getMinSignedBits() <= 64 &&
          isLegalAddImmediate(C1Int.getSExtValue()))
        return false;

      // Neither constant will fit into an immediate, so find materialisation
      // costs.
      int C1Cost = RISCVMatInt::getIntMatCost(C1Int, Ty.getSizeInBits(),
                                              Subtarget.is64Bit());
      int ShiftedC1Cost = RISCVMatInt::getIntMatCost(
          ShiftedC1Int, Ty.getSizeInBits(), Subtarget.is64Bit());

      // Materialising `c1` is cheaper than materialising `c1 << c2`, so the
      // combine should be prevented.
      if (C1Cost < ShiftedC1Cost)
        return false;
    }
  }
  return true;
}

unsigned RISCVTargetLowering::ComputeNumSignBitsForTargetNode(
    SDValue Op, const APInt &DemandedElts, const SelectionDAG &DAG,
    unsigned Depth) const {
  switch (Op.getOpcode()) {
  default:
    break;
  case RISCVISD::SLLW:
  case RISCVISD::SRAW:
  case RISCVISD::SRLW:
  case RISCVISD::DIVW:
  case RISCVISD::DIVUW:
  case RISCVISD::REMUW:
    // TODO: As the result is sign-extended, this is conservatively correct. A
    // more precise answer could be calculated for SRAW depending on known
    // bits in the shift amount.
    return 33;
  }

  return 1;
}

static MachineBasicBlock *emitReadCycleWidePseudo(MachineInstr &MI,
                                                  MachineBasicBlock *BB) {
  assert(MI.getOpcode() == RISCV::ReadCycleWide && "Unexpected instruction");

  // To read the 64-bit cycle CSR on a 32-bit target, we read the two halves.
  // Should the count have wrapped while it was being read, we need to try
  // again.
  // ...
  // read:
  // rdcycleh x3 # load high word of cycle
  // rdcycle  x2 # load low word of cycle
  // rdcycleh x4 # load high word of cycle
  // bne x3, x4, read # check if high word reads match, otherwise try again
  // ...

  MachineFunction &MF = *BB->getParent();
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction::iterator It = ++BB->getIterator();

  MachineBasicBlock *LoopMBB = MF.CreateMachineBasicBlock(LLVM_BB);
  MF.insert(It, LoopMBB);

  MachineBasicBlock *DoneMBB = MF.CreateMachineBasicBlock(LLVM_BB);
  MF.insert(It, DoneMBB);

  // Transfer the remainder of BB and its successor edges to DoneMBB.
  DoneMBB->splice(DoneMBB->begin(), BB,
                  std::next(MachineBasicBlock::iterator(MI)), BB->end());
  DoneMBB->transferSuccessorsAndUpdatePHIs(BB);

  BB->addSuccessor(LoopMBB);

  MachineRegisterInfo &RegInfo = MF.getRegInfo();
  Register ReadAgainReg = RegInfo.createVirtualRegister(&RISCV::GPRRegClass);
  Register LoReg = MI.getOperand(0).getReg();
  Register HiReg = MI.getOperand(1).getReg();
  DebugLoc DL = MI.getDebugLoc();

  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  BuildMI(LoopMBB, DL, TII->get(RISCV::CSRRS), HiReg)
      .addImm(RISCVSysReg::lookupSysRegByName("CYCLEH")->Encoding)
      .addReg(RISCV::X0);
  BuildMI(LoopMBB, DL, TII->get(RISCV::CSRRS), LoReg)
      .addImm(RISCVSysReg::lookupSysRegByName("CYCLE")->Encoding)
      .addReg(RISCV::X0);
  BuildMI(LoopMBB, DL, TII->get(RISCV::CSRRS), ReadAgainReg)
      .addImm(RISCVSysReg::lookupSysRegByName("CYCLEH")->Encoding)
      .addReg(RISCV::X0);

  BuildMI(LoopMBB, DL, TII->get(RISCV::BNE))
      .addReg(HiReg)
      .addReg(ReadAgainReg)
      .addMBB(LoopMBB);

  LoopMBB->addSuccessor(LoopMBB);
  LoopMBB->addSuccessor(DoneMBB);

  MI.eraseFromParent();

  return DoneMBB;
}

static MachineBasicBlock *emitSplitF64Pseudo(MachineInstr &MI,
                                             MachineBasicBlock *BB) {
  assert(MI.getOpcode() == RISCV::SplitF64Pseudo && "Unexpected instruction");

  MachineFunction &MF = *BB->getParent();
  DebugLoc DL = MI.getDebugLoc();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  const TargetRegisterInfo *RI = MF.getSubtarget().getRegisterInfo();
  Register LoReg = MI.getOperand(0).getReg();
  Register HiReg = MI.getOperand(1).getReg();
  Register SrcReg = MI.getOperand(2).getReg();
  const TargetRegisterClass *SrcRC = &RISCV::FPR64RegClass;
  int FI = MF.getInfo<RISCVMachineFunctionInfo>()->getMoveF64FrameIndex(MF);

  TII.storeRegToStackSlot(*BB, MI, SrcReg, MI.getOperand(2).isKill(), FI, SrcRC,
                          RI);
  MachineMemOperand *MMO =
      MF.getMachineMemOperand(MachinePointerInfo::getFixedStack(MF, FI),
                              MachineMemOperand::MOLoad, 8, Align(8));
  BuildMI(*BB, MI, DL, TII.get(RISCV::LW), LoReg)
      .addFrameIndex(FI)
      .addImm(0)
      .addMemOperand(MMO);
  BuildMI(*BB, MI, DL, TII.get(RISCV::LW), HiReg)
      .addFrameIndex(FI)
      .addImm(4)
      .addMemOperand(MMO);
  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return BB;
}

static MachineBasicBlock *emitBuildPairF64Pseudo(MachineInstr &MI,
                                                 MachineBasicBlock *BB) {
  assert(MI.getOpcode() == RISCV::BuildPairF64Pseudo &&
         "Unexpected instruction");

  MachineFunction &MF = *BB->getParent();
  DebugLoc DL = MI.getDebugLoc();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  const TargetRegisterInfo *RI = MF.getSubtarget().getRegisterInfo();
  Register DstReg = MI.getOperand(0).getReg();
  Register LoReg = MI.getOperand(1).getReg();
  Register HiReg = MI.getOperand(2).getReg();
  const TargetRegisterClass *DstRC = &RISCV::FPR64RegClass;
  int FI = MF.getInfo<RISCVMachineFunctionInfo>()->getMoveF64FrameIndex(MF);

  MachineMemOperand *MMO =
      MF.getMachineMemOperand(MachinePointerInfo::getFixedStack(MF, FI),
                              MachineMemOperand::MOStore, 8, Align(8));
  BuildMI(*BB, MI, DL, TII.get(RISCV::SW))
      .addReg(LoReg, getKillRegState(MI.getOperand(1).isKill()))
      .addFrameIndex(FI)
      .addImm(0)
      .addMemOperand(MMO);
  BuildMI(*BB, MI, DL, TII.get(RISCV::SW))
      .addReg(HiReg, getKillRegState(MI.getOperand(2).isKill()))
      .addFrameIndex(FI)
      .addImm(4)
      .addMemOperand(MMO);
  TII.loadRegFromStackSlot(*BB, MI, DstReg, FI, DstRC, RI);
  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return BB;
}

static bool isSelectPseudo(MachineInstr &MI) {
  switch (MI.getOpcode()) {
  default:
    return false;
  case RISCV::Select_GPR_Using_CC_GPR:
  case RISCV::Select_FPR32_Using_CC_GPR:
  case RISCV::Select_FPR64_Using_CC_GPR:
    return true;
  }
}

static MachineBasicBlock *emitSelectPseudo(MachineInstr &MI,
                                           MachineBasicBlock *BB) {
  // To "insert" Select_* instructions, we actually have to insert the triangle
  // control-flow pattern.  The incoming instructions know the destination vreg
  // to set, the condition code register to branch on, the true/false values to
  // select between, and the condcode to use to select the appropriate branch.
  //
  // We produce the following control flow:
  //     HeadMBB
  //     |  \
  //     |  IfFalseMBB
  //     | /
  //    TailMBB
  //
  // When we find a sequence of selects we attempt to optimize their emission
  // by sharing the control flow. Currently we only handle cases where we have
  // multiple selects with the exact same condition (same LHS, RHS and CC).
  // The selects may be interleaved with other instructions if the other
  // instructions meet some requirements we deem safe:
  // - They are debug instructions. Otherwise,
  // - They do not have side-effects, do not access memory and their inputs do
  //   not depend on the results of the select pseudo-instructions.
  // The TrueV/FalseV operands of the selects cannot depend on the result of
  // previous selects in the sequence.
  // These conditions could be further relaxed. See the X86 target for a
  // related approach and more information.
  Register LHS = MI.getOperand(1).getReg();
  Register RHS = MI.getOperand(2).getReg();
  auto CC = static_cast<ISD::CondCode>(MI.getOperand(3).getImm());

  SmallVector<MachineInstr *, 4> SelectDebugValues;
  SmallSet<Register, 4> SelectDests;
  SelectDests.insert(MI.getOperand(0).getReg());

  MachineInstr *LastSelectPseudo = &MI;

  for (auto E = BB->end(), SequenceMBBI = MachineBasicBlock::iterator(MI);
       SequenceMBBI != E; ++SequenceMBBI) {
    if (SequenceMBBI->isDebugInstr())
      continue;
    else if (isSelectPseudo(*SequenceMBBI)) {
      if (SequenceMBBI->getOperand(1).getReg() != LHS ||
          SequenceMBBI->getOperand(2).getReg() != RHS ||
          SequenceMBBI->getOperand(3).getImm() != CC ||
          SelectDests.count(SequenceMBBI->getOperand(4).getReg()) ||
          SelectDests.count(SequenceMBBI->getOperand(5).getReg()))
        break;
      LastSelectPseudo = &*SequenceMBBI;
      SequenceMBBI->collectDebugValues(SelectDebugValues);
      SelectDests.insert(SequenceMBBI->getOperand(0).getReg());
    } else {
      if (SequenceMBBI->hasUnmodeledSideEffects() ||
          SequenceMBBI->mayLoadOrStore())
        break;
      if (llvm::any_of(SequenceMBBI->operands(), [&](MachineOperand &MO) {
            return MO.isReg() && MO.isUse() && SelectDests.count(MO.getReg());
          }))
        break;
    }
  }

  const TargetInstrInfo &TII = *BB->getParent()->getSubtarget().getInstrInfo();
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  DebugLoc DL = MI.getDebugLoc();
  MachineFunction::iterator I = ++BB->getIterator();

  MachineBasicBlock *HeadMBB = BB;
  MachineFunction *F = BB->getParent();
  MachineBasicBlock *TailMBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *IfFalseMBB = F->CreateMachineBasicBlock(LLVM_BB);

  F->insert(I, IfFalseMBB);
  F->insert(I, TailMBB);

  // Transfer debug instructions associated with the selects to TailMBB.
  for (MachineInstr *DebugInstr : SelectDebugValues) {
    TailMBB->push_back(DebugInstr->removeFromParent());
  }

  // Move all instructions after the sequence to TailMBB.
  TailMBB->splice(TailMBB->end(), HeadMBB,
                  std::next(LastSelectPseudo->getIterator()), HeadMBB->end());
  // Update machine-CFG edges by transferring all successors of the current
  // block to the new block which will contain the Phi nodes for the selects.
  TailMBB->transferSuccessorsAndUpdatePHIs(HeadMBB);
  // Set the successors for HeadMBB.
  HeadMBB->addSuccessor(IfFalseMBB);
  HeadMBB->addSuccessor(TailMBB);

  // Insert appropriate branch.
  unsigned Opcode = getBranchOpcodeForIntCondCode(CC);

  BuildMI(HeadMBB, DL, TII.get(Opcode))
    .addReg(LHS)
    .addReg(RHS)
    .addMBB(TailMBB);

  // IfFalseMBB just falls through to TailMBB.
  IfFalseMBB->addSuccessor(TailMBB);

  // Create PHIs for all of the select pseudo-instructions.
  auto SelectMBBI = MI.getIterator();
  auto SelectEnd = std::next(LastSelectPseudo->getIterator());
  auto InsertionPoint = TailMBB->begin();
  while (SelectMBBI != SelectEnd) {
    auto Next = std::next(SelectMBBI);
    if (isSelectPseudo(*SelectMBBI)) {
      // %Result = phi [ %TrueValue, HeadMBB ], [ %FalseValue, IfFalseMBB ]
      BuildMI(*TailMBB, InsertionPoint, SelectMBBI->getDebugLoc(),
              TII.get(RISCV::PHI), SelectMBBI->getOperand(0).getReg())
          .addReg(SelectMBBI->getOperand(4).getReg())
          .addMBB(HeadMBB)
          .addReg(SelectMBBI->getOperand(5).getReg())
          .addMBB(IfFalseMBB);
      SelectMBBI->eraseFromParent();
    }
    SelectMBBI = Next;
  }

  F->getProperties().reset(MachineFunctionProperties::Property::NoPHIs);
  return TailMBB;
}

MachineBasicBlock *
RISCVTargetLowering::EmitInstrWithCustomInserter(MachineInstr &MI,
                                                 MachineBasicBlock *BB) const {
  switch (MI.getOpcode()) {
  default:
    llvm_unreachable("Unexpected instr type to insert");
  case RISCV::ReadCycleWide:
    assert(!Subtarget.is64Bit() &&
           "ReadCycleWrite is only to be used on riscv32");
    return emitReadCycleWidePseudo(MI, BB);
  case RISCV::Select_GPR_Using_CC_GPR:
  case RISCV::Select_FPR32_Using_CC_GPR:
  case RISCV::Select_FPR64_Using_CC_GPR:
    return emitSelectPseudo(MI, BB);
  case RISCV::BuildPairF64Pseudo:
    return emitBuildPairF64Pseudo(MI, BB);
  case RISCV::SplitF64Pseudo:
    return emitSplitF64Pseudo(MI, BB);
  }
}

// Calling Convention Implementation.
// The expectations for frontend ABI lowering vary from target to target.
// Ideally, an LLVM frontend would be able to avoid worrying about many ABI
// details, but this is a longer term goal. For now, we simply try to keep the
// role of the frontend as simple and well-defined as possible. The rules can
// be summarised as:
// * Never split up large scalar arguments. We handle them here.
// * If a hardfloat calling convention is being used, and the struct may be
// passed in a pair of registers (fp+fp, int+fp), and both registers are
// available, then pass as two separate arguments. If either the GPRs or FPRs
// are exhausted, then pass according to the rule below.
// * If a struct could never be passed in registers or directly in a stack
// slot (as it is larger than 2*XLEN and the floating point rules don't
// apply), then pass it using a pointer with the byval attribute.
// * If a struct is less than 2*XLEN, then coerce to either a two-element
// word-sized array or a 2*XLEN scalar (depending on alignment).
// * The frontend can determine whether a struct is returned by reference or
// not based on its size and fields. If it will be returned by reference, the
// frontend must modify the prototype so a pointer with the sret annotation is
// passed as the first argument. This is not necessary for large scalar
// returns.
// * Struct return values and varargs should be coerced to structs containing
// register-size fields in the same situations they would be for fixed
// arguments.

static const MCPhysReg ArgGPRs[] = {
  RISCV::X10, RISCV::X11, RISCV::X12, RISCV::X13,
  RISCV::X14, RISCV::X15, RISCV::X16, RISCV::X17
};
static const MCPhysReg ArgFPR32s[] = {
  RISCV::F10_F, RISCV::F11_F, RISCV::F12_F, RISCV::F13_F,
  RISCV::F14_F, RISCV::F15_F, RISCV::F16_F, RISCV::F17_F
};
static const MCPhysReg ArgFPR64s[] = {
  RISCV::F10_D, RISCV::F11_D, RISCV::F12_D, RISCV::F13_D,
  RISCV::F14_D, RISCV::F15_D, RISCV::F16_D, RISCV::F17_D
};

// Pass a 2*XLEN argument that has been split into two XLEN values through
// registers or the stack as necessary.
static bool CC_RISCVAssign2XLen(unsigned XLen, CCState &State, CCValAssign VA1,
                                ISD::ArgFlagsTy ArgFlags1, unsigned ValNo2,
                                MVT ValVT2, MVT LocVT2,
                                ISD::ArgFlagsTy ArgFlags2) {
  unsigned XLenInBytes = XLen / 8;
  if (Register Reg = State.AllocateReg(ArgGPRs)) {
    // At least one half can be passed via register.
    State.addLoc(CCValAssign::getReg(VA1.getValNo(), VA1.getValVT(), Reg,
                                     VA1.getLocVT(), CCValAssign::Full));
  } else {
    // Both halves must be passed on the stack, with proper alignment.
    Align StackAlign =
        std::max(Align(XLenInBytes), ArgFlags1.getNonZeroOrigAlign());
    State.addLoc(
        CCValAssign::getMem(VA1.getValNo(), VA1.getValVT(),
                            State.AllocateStack(XLenInBytes, StackAlign),
                            VA1.getLocVT(), CCValAssign::Full));
    State.addLoc(CCValAssign::getMem(
        ValNo2, ValVT2, State.AllocateStack(XLenInBytes, Align(XLenInBytes)),
        LocVT2, CCValAssign::Full));
    return false;
  }

  if (Register Reg = State.AllocateReg(ArgGPRs)) {
    // The second half can also be passed via register.
    State.addLoc(
        CCValAssign::getReg(ValNo2, ValVT2, Reg, LocVT2, CCValAssign::Full));
  } else {
    // The second half is passed via the stack, without additional alignment.
    State.addLoc(CCValAssign::getMem(
        ValNo2, ValVT2, State.AllocateStack(XLenInBytes, Align(XLenInBytes)),
        LocVT2, CCValAssign::Full));
  }

  return false;
}

// Implements the RISC-V calling convention. Returns true upon failure.
static bool CC_RISCV(const DataLayout &DL, RISCVABI::ABI ABI, unsigned ValNo,
                     MVT ValVT, MVT LocVT, CCValAssign::LocInfo LocInfo,
                     ISD::ArgFlagsTy ArgFlags, CCState &State, bool IsFixed,
                     bool IsRet, Type *OrigTy) {
  unsigned XLen = DL.getLargestLegalIntTypeSizeInBits();
  assert(XLen == 32 || XLen == 64);
  MVT XLenVT = XLen == 32 ? MVT::i32 : MVT::i64;

  // Any return value split in to more than two values can't be returned
  // directly.
  if (IsRet && ValNo > 1)
    return true;

  // UseGPRForF32 if targeting one of the soft-float ABIs, if passing a
  // variadic argument, or if no F32 argument registers are available.
  bool UseGPRForF32 = true;
  // UseGPRForF64 if targeting soft-float ABIs or an FLEN=32 ABI, if passing a
  // variadic argument, or if no F64 argument registers are available.
  bool UseGPRForF64 = true;

  switch (ABI) {
  default:
    llvm_unreachable("Unexpected ABI");
  case RISCVABI::ABI_ILP32:
  case RISCVABI::ABI_LP64:
    break;
  case RISCVABI::ABI_ILP32F:
  case RISCVABI::ABI_LP64F:
    UseGPRForF32 = !IsFixed;
    break;
  case RISCVABI::ABI_ILP32D:
  case RISCVABI::ABI_LP64D:
    UseGPRForF32 = !IsFixed;
    UseGPRForF64 = !IsFixed;
    break;
  }

  if (State.getFirstUnallocated(ArgFPR32s) == array_lengthof(ArgFPR32s))
    UseGPRForF32 = true;
  if (State.getFirstUnallocated(ArgFPR64s) == array_lengthof(ArgFPR64s))
    UseGPRForF64 = true;

  // From this point on, rely on UseGPRForF32, UseGPRForF64 and similar local
  // variables rather than directly checking against the target ABI.

  if (UseGPRForF32 && ValVT == MVT::f32) {
    LocVT = XLenVT;
    LocInfo = CCValAssign::BCvt;
  } else if (UseGPRForF64 && XLen == 64 && ValVT == MVT::f64) {
    LocVT = MVT::i64;
    LocInfo = CCValAssign::BCvt;
  }

  // If this is a variadic argument, the RISC-V calling convention requires
  // that it is assigned an 'even' or 'aligned' register if it has 8-byte
  // alignment (RV32) or 16-byte alignment (RV64). An aligned register should
  // be used regardless of whether the original argument was split during
  // legalisation or not. The argument will not be passed by registers if the
  // original type is larger than 2*XLEN, so the register alignment rule does
  // not apply.
  unsigned TwoXLenInBytes = (2 * XLen) / 8;
  if (!IsFixed && ArgFlags.getNonZeroOrigAlign() == TwoXLenInBytes &&
      DL.getTypeAllocSize(OrigTy) == TwoXLenInBytes) {
    unsigned RegIdx = State.getFirstUnallocated(ArgGPRs);
    // Skip 'odd' register if necessary.
    if (RegIdx != array_lengthof(ArgGPRs) && RegIdx % 2 == 1)
      State.AllocateReg(ArgGPRs);
  }

  SmallVectorImpl<CCValAssign> &PendingLocs = State.getPendingLocs();
  SmallVectorImpl<ISD::ArgFlagsTy> &PendingArgFlags =
      State.getPendingArgFlags();

  assert(PendingLocs.size() == PendingArgFlags.size() &&
         "PendingLocs and PendingArgFlags out of sync");

  // Handle passing f64 on RV32D with a soft float ABI or when floating point
  // registers are exhausted.
  if (UseGPRForF64 && XLen == 32 && ValVT == MVT::f64) {
    assert(!ArgFlags.isSplit() && PendingLocs.empty() &&
           "Can't lower f64 if it is split");
    // Depending on available argument GPRS, f64 may be passed in a pair of
    // GPRs, split between a GPR and the stack, or passed completely on the
    // stack. LowerCall/LowerFormalArguments/LowerReturn must recognise these
    // cases.
    Register Reg = State.AllocateReg(ArgGPRs);
    LocVT = MVT::i32;
    if (!Reg) {
      unsigned StackOffset = State.AllocateStack(8, Align(8));
      State.addLoc(
          CCValAssign::getMem(ValNo, ValVT, StackOffset, LocVT, LocInfo));
      return false;
    }
    if (!State.AllocateReg(ArgGPRs))
      State.AllocateStack(4, Align(4));
    State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
    return false;
  }

  // Split arguments might be passed indirectly, so keep track of the pending
  // values.
  if (ArgFlags.isSplit() || !PendingLocs.empty()) {
    LocVT = XLenVT;
    LocInfo = CCValAssign::Indirect;
    PendingLocs.push_back(
        CCValAssign::getPending(ValNo, ValVT, LocVT, LocInfo));
    PendingArgFlags.push_back(ArgFlags);
    if (!ArgFlags.isSplitEnd()) {
      return false;
    }
  }

  // If the split argument only had two elements, it should be passed directly
  // in registers or on the stack.
  if (ArgFlags.isSplitEnd() && PendingLocs.size() <= 2) {
    assert(PendingLocs.size() == 2 && "Unexpected PendingLocs.size()");
    // Apply the normal calling convention rules to the first half of the
    // split argument.
    CCValAssign VA = PendingLocs[0];
    ISD::ArgFlagsTy AF = PendingArgFlags[0];
    PendingLocs.clear();
    PendingArgFlags.clear();
    return CC_RISCVAssign2XLen(XLen, State, VA, AF, ValNo, ValVT, LocVT,
                               ArgFlags);
  }

  // Allocate to a register if possible, or else a stack slot.
  Register Reg;
  if (ValVT == MVT::f32 && !UseGPRForF32)
    Reg = State.AllocateReg(ArgFPR32s, ArgFPR64s);
  else if (ValVT == MVT::f64 && !UseGPRForF64)
    Reg = State.AllocateReg(ArgFPR64s, ArgFPR32s);
  else
    Reg = State.AllocateReg(ArgGPRs);
  unsigned StackOffset =
      Reg ? 0 : State.AllocateStack(XLen / 8, Align(XLen / 8));

  // If we reach this point and PendingLocs is non-empty, we must be at the
  // end of a split argument that must be passed indirectly.
  if (!PendingLocs.empty()) {
    assert(ArgFlags.isSplitEnd() && "Expected ArgFlags.isSplitEnd()");
    assert(PendingLocs.size() > 2 && "Unexpected PendingLocs.size()");

    for (auto &It : PendingLocs) {
      if (Reg)
        It.convertToReg(Reg);
      else
        It.convertToMem(StackOffset);
      State.addLoc(It);
    }
    PendingLocs.clear();
    PendingArgFlags.clear();
    return false;
  }

  assert((!UseGPRForF32 || !UseGPRForF64 || LocVT == XLenVT) &&
         "Expected an XLenVT at this stage");

  if (Reg) {
    State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
    return false;
  }

  // When an f32 or f64 is passed on the stack, no bit-conversion is needed.
  if (ValVT == MVT::f32 || ValVT == MVT::f64) {
    LocVT = ValVT;
    LocInfo = CCValAssign::Full;
  }
  State.addLoc(CCValAssign::getMem(ValNo, ValVT, StackOffset, LocVT, LocInfo));
  return false;
}

void RISCVTargetLowering::analyzeInputArgs(
    MachineFunction &MF, CCState &CCInfo,
    const SmallVectorImpl<ISD::InputArg> &Ins, bool IsRet) const {
  unsigned NumArgs = Ins.size();
  FunctionType *FType = MF.getFunction().getFunctionType();

  for (unsigned i = 0; i != NumArgs; ++i) {
    MVT ArgVT = Ins[i].VT;
    ISD::ArgFlagsTy ArgFlags = Ins[i].Flags;

    Type *ArgTy = nullptr;
    if (IsRet)
      ArgTy = FType->getReturnType();
    else if (Ins[i].isOrigArg())
      ArgTy = FType->getParamType(Ins[i].getOrigArgIndex());

    RISCVABI::ABI ABI = MF.getSubtarget<RISCVSubtarget>().getTargetABI();
    if (CC_RISCV(MF.getDataLayout(), ABI, i, ArgVT, ArgVT, CCValAssign::Full,
                 ArgFlags, CCInfo, /*IsFixed=*/true, IsRet, ArgTy)) {
      LLVM_DEBUG(dbgs() << "InputArg #" << i << " has unhandled type "
                        << EVT(ArgVT).getEVTString() << '\n');
      llvm_unreachable(nullptr);
    }
  }
}

void RISCVTargetLowering::analyzeOutputArgs(
    MachineFunction &MF, CCState &CCInfo,
    const SmallVectorImpl<ISD::OutputArg> &Outs, bool IsRet,
    CallLoweringInfo *CLI) const {
  unsigned NumArgs = Outs.size();

  for (unsigned i = 0; i != NumArgs; i++) {
    MVT ArgVT = Outs[i].VT;
    ISD::ArgFlagsTy ArgFlags = Outs[i].Flags;
    Type *OrigTy = CLI ? CLI->getArgs()[Outs[i].OrigArgIndex].Ty : nullptr;

    RISCVABI::ABI ABI = MF.getSubtarget<RISCVSubtarget>().getTargetABI();
    if (CC_RISCV(MF.getDataLayout(), ABI, i, ArgVT, ArgVT, CCValAssign::Full,
                 ArgFlags, CCInfo, Outs[i].IsFixed, IsRet, OrigTy)) {
      LLVM_DEBUG(dbgs() << "OutputArg #" << i << " has unhandled type "
                        << EVT(ArgVT).getEVTString() << "\n");
      llvm_unreachable(nullptr);
    }
  }
}

// Convert Val to a ValVT. Should not be called for CCValAssign::Indirect
// values.
static SDValue convertLocVTToValVT(SelectionDAG &DAG, SDValue Val,
                                   const CCValAssign &VA, const SDLoc &DL) {
  switch (VA.getLocInfo()) {
  default:
    llvm_unreachable("Unexpected CCValAssign::LocInfo");
  case CCValAssign::Full:
    break;
  case CCValAssign::BCvt:
    if (VA.getLocVT() == MVT::i64 && VA.getValVT() == MVT::f32) {
      Val = DAG.getNode(RISCVISD::FMV_W_X_RV64, DL, MVT::f32, Val);
      break;
    }
    Val = DAG.getNode(ISD::BITCAST, DL, VA.getValVT(), Val);
    break;
  }
  return Val;
}

// The caller is responsible for loading the full value if the argument is
// passed with CCValAssign::Indirect.
static SDValue unpackFromRegLoc(SelectionDAG &DAG, SDValue Chain,
                                const CCValAssign &VA, const SDLoc &DL) {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();
  EVT LocVT = VA.getLocVT();
  SDValue Val;
  const TargetRegisterClass *RC;

  switch (LocVT.getSimpleVT().SimpleTy) {
  default:
    llvm_unreachable("Unexpected register type");
  case MVT::i32:
  case MVT::i64:
    RC = &RISCV::GPRRegClass;
    break;
  case MVT::f32:
    RC = &RISCV::FPR32RegClass;
    break;
  case MVT::f64:
    RC = &RISCV::FPR64RegClass;
    break;
  }

  Register VReg = RegInfo.createVirtualRegister(RC);
  RegInfo.addLiveIn(VA.getLocReg(), VReg);
  Val = DAG.getCopyFromReg(Chain, DL, VReg, LocVT);

  if (VA.getLocInfo() == CCValAssign::Indirect)
    return Val;

  return convertLocVTToValVT(DAG, Val, VA, DL);
}

static SDValue convertValVTToLocVT(SelectionDAG &DAG, SDValue Val,
                                   const CCValAssign &VA, const SDLoc &DL) {
  EVT LocVT = VA.getLocVT();

  switch (VA.getLocInfo()) {
  default:
    llvm_unreachable("Unexpected CCValAssign::LocInfo");
  case CCValAssign::Full:
    break;
  case CCValAssign::BCvt:
    if (VA.getLocVT() == MVT::i64 && VA.getValVT() == MVT::f32) {
      Val = DAG.getNode(RISCVISD::FMV_X_ANYEXTW_RV64, DL, MVT::i64, Val);
      break;
    }
    Val = DAG.getNode(ISD::BITCAST, DL, LocVT, Val);
    break;
  }
  return Val;
}

// The caller is responsible for loading the full value if the argument is
// passed with CCValAssign::Indirect.
static SDValue unpackFromMemLoc(SelectionDAG &DAG, SDValue Chain,
                                const CCValAssign &VA, const SDLoc &DL) {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  EVT LocVT = VA.getLocVT();
  EVT ValVT = VA.getValVT();
  EVT PtrVT = MVT::getIntegerVT(DAG.getDataLayout().getPointerSizeInBits(0));
  int FI = MFI.CreateFixedObject(ValVT.getSizeInBits() / 8,
                                 VA.getLocMemOffset(), /*Immutable=*/true);
  SDValue FIN = DAG.getFrameIndex(FI, PtrVT);
  SDValue Val;

  ISD::LoadExtType ExtType;
  switch (VA.getLocInfo()) {
  default:
    llvm_unreachable("Unexpected CCValAssign::LocInfo");
  case CCValAssign::Full:
  case CCValAssign::Indirect:
  case CCValAssign::BCvt:
    ExtType = ISD::NON_EXTLOAD;
    break;
  }
  Val = DAG.getExtLoad(
      ExtType, DL, LocVT, Chain, FIN,
      MachinePointerInfo::getFixedStack(DAG.getMachineFunction(), FI), ValVT);
  return Val;
}

static SDValue unpackF64OnRV32DSoftABI(SelectionDAG &DAG, SDValue Chain,
                                       const CCValAssign &VA, const SDLoc &DL) {
  assert(VA.getLocVT() == MVT::i32 && VA.getValVT() == MVT::f64 &&
         "Unexpected VA");
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();

  if (VA.isMemLoc()) {
    // f64 is passed on the stack.
    int FI = MFI.CreateFixedObject(8, VA.getLocMemOffset(), /*Immutable=*/true);
    SDValue FIN = DAG.getFrameIndex(FI, MVT::i32);
    return DAG.getLoad(MVT::f64, DL, Chain, FIN,
                       MachinePointerInfo::getFixedStack(MF, FI));
  }

  assert(VA.isRegLoc() && "Expected register VA assignment");

  Register LoVReg = RegInfo.createVirtualRegister(&RISCV::GPRRegClass);
  RegInfo.addLiveIn(VA.getLocReg(), LoVReg);
  SDValue Lo = DAG.getCopyFromReg(Chain, DL, LoVReg, MVT::i32);
  SDValue Hi;
  if (VA.getLocReg() == RISCV::X17) {
    // Second half of f64 is passed on the stack.
    int FI = MFI.CreateFixedObject(4, 0, /*Immutable=*/true);
    SDValue FIN = DAG.getFrameIndex(FI, MVT::i32);
    Hi = DAG.getLoad(MVT::i32, DL, Chain, FIN,
                     MachinePointerInfo::getFixedStack(MF, FI));
  } else {
    // Second half of f64 is passed in another GPR.
    Register HiVReg = RegInfo.createVirtualRegister(&RISCV::GPRRegClass);
    RegInfo.addLiveIn(VA.getLocReg() + 1, HiVReg);
    Hi = DAG.getCopyFromReg(Chain, DL, HiVReg, MVT::i32);
  }
  return DAG.getNode(RISCVISD::BuildPairF64, DL, MVT::f64, Lo, Hi);
}

// FastCC has less than 1% performance improvement for some particular
// benchmark. But theoretically, it may has benenfit for some cases.
static bool CC_RISCV_FastCC(unsigned ValNo, MVT ValVT, MVT LocVT,
                            CCValAssign::LocInfo LocInfo,
                            ISD::ArgFlagsTy ArgFlags, CCState &State) {

  if (LocVT == MVT::i32 || LocVT == MVT::i64) {
    // X5 and X6 might be used for save-restore libcall.
    static const MCPhysReg GPRList[] = {
        RISCV::X10, RISCV::X11, RISCV::X12, RISCV::X13, RISCV::X14,
        RISCV::X15, RISCV::X16, RISCV::X17, RISCV::X7,  RISCV::X28,
        RISCV::X29, RISCV::X30, RISCV::X31};
    if (unsigned Reg = State.AllocateReg(GPRList)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (LocVT == MVT::f32) {
    static const MCPhysReg FPR32List[] = {
        RISCV::F10_F, RISCV::F11_F, RISCV::F12_F, RISCV::F13_F, RISCV::F14_F,
        RISCV::F15_F, RISCV::F16_F, RISCV::F17_F, RISCV::F0_F,  RISCV::F1_F,
        RISCV::F2_F,  RISCV::F3_F,  RISCV::F4_F,  RISCV::F5_F,  RISCV::F6_F,
        RISCV::F7_F,  RISCV::F28_F, RISCV::F29_F, RISCV::F30_F, RISCV::F31_F};
    if (unsigned Reg = State.AllocateReg(FPR32List)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (LocVT == MVT::f64) {
    static const MCPhysReg FPR64List[] = {
        RISCV::F10_D, RISCV::F11_D, RISCV::F12_D, RISCV::F13_D, RISCV::F14_D,
        RISCV::F15_D, RISCV::F16_D, RISCV::F17_D, RISCV::F0_D,  RISCV::F1_D,
        RISCV::F2_D,  RISCV::F3_D,  RISCV::F4_D,  RISCV::F5_D,  RISCV::F6_D,
        RISCV::F7_D,  RISCV::F28_D, RISCV::F29_D, RISCV::F30_D, RISCV::F31_D};
    if (unsigned Reg = State.AllocateReg(FPR64List)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (LocVT == MVT::i32 || LocVT == MVT::f32) {
    unsigned Offset4 = State.AllocateStack(4, Align(4));
    State.addLoc(CCValAssign::getMem(ValNo, ValVT, Offset4, LocVT, LocInfo));
    return false;
  }

  if (LocVT == MVT::i64 || LocVT == MVT::f64) {
    unsigned Offset5 = State.AllocateStack(8, Align(8));
    State.addLoc(CCValAssign::getMem(ValNo, ValVT, Offset5, LocVT, LocInfo));
    return false;
  }

  return true; // CC didn't match.
}

// Transform physical registers into virtual registers.
SDValue RISCVTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {

  switch (CallConv) {
  default:
    report_fatal_error("Unsupported calling convention");
  case CallingConv::C:
  case CallingConv::Fast:
    break;
  }

  MachineFunction &MF = DAG.getMachineFunction();

  const Function &Func = MF.getFunction();
  if (Func.hasFnAttribute("interrupt")) {
    if (!Func.arg_empty())
      report_fatal_error(
        "Functions with the interrupt attribute cannot have arguments!");

    StringRef Kind =
      MF.getFunction().getFnAttribute("interrupt").getValueAsString();

    if (!(Kind == "user" || Kind == "supervisor" || Kind == "machine"))
      report_fatal_error(
        "Function interrupt attribute argument not supported!");
  }

  EVT PtrVT = getPointerTy(DAG.getDataLayout());
  MVT XLenVT = Subtarget.getXLenVT();
  unsigned XLenInBytes = Subtarget.getXLen() / 8;
  // Used with vargs to acumulate store chains.
  std::vector<SDValue> OutChains;

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());

  if (CallConv == CallingConv::Fast)
    CCInfo.AnalyzeFormalArguments(Ins, CC_RISCV_FastCC);
  else
    analyzeInputArgs(MF, CCInfo, Ins, /*IsRet=*/false);

  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    SDValue ArgValue;
    // Passing f64 on RV32D with a soft float ABI must be handled as a special
    // case.
    if (VA.getLocVT() == MVT::i32 && VA.getValVT() == MVT::f64)
      ArgValue = unpackF64OnRV32DSoftABI(DAG, Chain, VA, DL);
    else if (VA.isRegLoc())
      ArgValue = unpackFromRegLoc(DAG, Chain, VA, DL);
    else
      ArgValue = unpackFromMemLoc(DAG, Chain, VA, DL);

    if (VA.getLocInfo() == CCValAssign::Indirect) {
      // If the original argument was split and passed by reference (e.g. i128
      // on RV32), we need to load all parts of it here (using the same
      // address).
      InVals.push_back(DAG.getLoad(VA.getValVT(), DL, Chain, ArgValue,
                                   MachinePointerInfo()));
      unsigned ArgIndex = Ins[i].OrigArgIndex;
      assert(Ins[i].PartOffset == 0);
      while (i + 1 != e && Ins[i + 1].OrigArgIndex == ArgIndex) {
        CCValAssign &PartVA = ArgLocs[i + 1];
        unsigned PartOffset = Ins[i + 1].PartOffset;
        SDValue Address = DAG.getNode(ISD::ADD, DL, PtrVT, ArgValue,
                                      DAG.getIntPtrConstant(PartOffset, DL));
        InVals.push_back(DAG.getLoad(PartVA.getValVT(), DL, Chain, Address,
                                     MachinePointerInfo()));
        ++i;
      }
      continue;
    }
    InVals.push_back(ArgValue);
  }

  if (IsVarArg) {
    ArrayRef<MCPhysReg> ArgRegs = makeArrayRef(ArgGPRs);
    unsigned Idx = CCInfo.getFirstUnallocated(ArgRegs);
    const TargetRegisterClass *RC = &RISCV::GPRRegClass;
    MachineFrameInfo &MFI = MF.getFrameInfo();
    MachineRegisterInfo &RegInfo = MF.getRegInfo();
    RISCVMachineFunctionInfo *RVFI = MF.getInfo<RISCVMachineFunctionInfo>();

    // Offset of the first variable argument from stack pointer, and size of
    // the vararg save area. For now, the varargs save area is either zero or
    // large enough to hold a0-a7.
    int VaArgOffset, VarArgsSaveSize;

    // If all registers are allocated, then all varargs must be passed on the
    // stack and we don't need to save any argregs.
    if (ArgRegs.size() == Idx) {
      VaArgOffset = CCInfo.getNextStackOffset();
      VarArgsSaveSize = 0;
    } else {
      VarArgsSaveSize = XLenInBytes * (ArgRegs.size() - Idx);
      VaArgOffset = -VarArgsSaveSize;
    }

    // Record the frame index of the first variable argument
    // which is a value necessary to VASTART.
    int FI = MFI.CreateFixedObject(XLenInBytes, VaArgOffset, true);
    RVFI->setVarArgsFrameIndex(FI);

    // If saving an odd number of registers then create an extra stack slot to
    // ensure that the frame pointer is 2*XLEN-aligned, which in turn ensures
    // offsets to even-numbered registered remain 2*XLEN-aligned.
    if (Idx % 2) {
      MFI.CreateFixedObject(XLenInBytes, VaArgOffset - (int)XLenInBytes, true);
      VarArgsSaveSize += XLenInBytes;
    }

    // Copy the integer registers that may have been used for passing varargs
    // to the vararg save area.
    for (unsigned I = Idx; I < ArgRegs.size();
         ++I, VaArgOffset += XLenInBytes) {
      const Register Reg = RegInfo.createVirtualRegister(RC);
      RegInfo.addLiveIn(ArgRegs[I], Reg);
      SDValue ArgValue = DAG.getCopyFromReg(Chain, DL, Reg, XLenVT);
      FI = MFI.CreateFixedObject(XLenInBytes, VaArgOffset, true);
      SDValue PtrOff = DAG.getFrameIndex(FI, getPointerTy(DAG.getDataLayout()));
      SDValue Store = DAG.getStore(Chain, DL, ArgValue, PtrOff,
                                   MachinePointerInfo::getFixedStack(MF, FI));
      cast<StoreSDNode>(Store.getNode())
          ->getMemOperand()
          ->setValue((Value *)nullptr);
      OutChains.push_back(Store);
    }
    RVFI->setVarArgsSaveSize(VarArgsSaveSize);
  }

  // All stores are grouped in one node to allow the matching between
  // the size of Ins and InVals. This only happens for vararg functions.
  if (!OutChains.empty()) {
    OutChains.push_back(Chain);
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, OutChains);
  }

  return Chain;
}

/// isEligibleForTailCallOptimization - Check whether the call is eligible
/// for tail call optimization.
/// Note: This is modelled after ARM's IsEligibleForTailCallOptimization.
bool RISCVTargetLowering::isEligibleForTailCallOptimization(
    CCState &CCInfo, CallLoweringInfo &CLI, MachineFunction &MF,
    const SmallVector<CCValAssign, 16> &ArgLocs) const {

  auto &Callee = CLI.Callee;
  auto CalleeCC = CLI.CallConv;
  auto &Outs = CLI.Outs;
  auto &Caller = MF.getFunction();
  auto CallerCC = Caller.getCallingConv();

  // Exception-handling functions need a special set of instructions to
  // indicate a return to the hardware. Tail-calling another function would
  // probably break this.
  // TODO: The "interrupt" attribute isn't currently defined by RISC-V. This
  // should be expanded as new function attributes are introduced.
  if (Caller.hasFnAttribute("interrupt"))
    return false;

  // Do not tail call opt if the stack is used to pass parameters.
  if (CCInfo.getNextStackOffset() != 0)
    return false;

  // Do not tail call opt if any parameters need to be passed indirectly.
  // Since long doubles (fp128) and i128 are larger than 2*XLEN, they are
  // passed indirectly. So the address of the value will be passed in a
  // register, or if not available, then the address is put on the stack. In
  // order to pass indirectly, space on the stack often needs to be allocated
  // in order to store the value. In this case the CCInfo.getNextStackOffset()
  // != 0 check is not enough and we need to check if any CCValAssign ArgsLocs
  // are passed CCValAssign::Indirect.
  for (auto &VA : ArgLocs)
    if (VA.getLocInfo() == CCValAssign::Indirect)
      return false;

  // Do not tail call opt if either caller or callee uses struct return
  // semantics.
  auto IsCallerStructRet = Caller.hasStructRetAttr();
  auto IsCalleeStructRet = Outs.empty() ? false : Outs[0].Flags.isSRet();
  if (IsCallerStructRet || IsCalleeStructRet)
    return false;

  // Externally-defined functions with weak linkage should not be
  // tail-called. The behaviour of branch instructions in this situation (as
  // used for tail calls) is implementation-defined, so we cannot rely on the
  // linker replacing the tail call with a return.
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
    const GlobalValue *GV = G->getGlobal();
    if (GV->hasExternalWeakLinkage())
      return false;
  }

  // The callee has to preserve all registers the caller needs to preserve.
  const RISCVRegisterInfo *TRI = Subtarget.getRegisterInfo();
  const uint32_t *CallerPreserved = TRI->getCallPreservedMask(MF, CallerCC);
  if (CalleeCC != CallerCC) {
    const uint32_t *CalleePreserved = TRI->getCallPreservedMask(MF, CalleeCC);
    if (!TRI->regmaskSubsetEqual(CallerPreserved, CalleePreserved))
      return false;
  }

  // Byval parameters hand the function a pointer directly into the stack area
  // we want to reuse during a tail call. Working around this *is* possible
  // but less efficient and uglier in LowerCall.
  for (auto &Arg : Outs)
    if (Arg.Flags.isByVal())
      return false;

  return true;
}

// Lower a call to a callseq_start + CALL + callseq_end chain, and add input
// and output parameter nodes.
SDValue RISCVTargetLowering::LowerCall(CallLoweringInfo &CLI,
                                       SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG = CLI.DAG;
  SDLoc &DL = CLI.DL;
  SmallVectorImpl<ISD::OutputArg> &Outs = CLI.Outs;
  SmallVectorImpl<SDValue> &OutVals = CLI.OutVals;
  SmallVectorImpl<ISD::InputArg> &Ins = CLI.Ins;
  SDValue Chain = CLI.Chain;
  SDValue Callee = CLI.Callee;
  bool &IsTailCall = CLI.IsTailCall;
  CallingConv::ID CallConv = CLI.CallConv;
  bool IsVarArg = CLI.IsVarArg;
  EVT PtrVT = getPointerTy(DAG.getDataLayout());
  MVT XLenVT = Subtarget.getXLenVT();

  MachineFunction &MF = DAG.getMachineFunction();

  // Analyze the operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState ArgCCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());

  if (CallConv == CallingConv::Fast)
    ArgCCInfo.AnalyzeCallOperands(Outs, CC_RISCV_FastCC);
  else
    analyzeOutputArgs(MF, ArgCCInfo, Outs, /*IsRet=*/false, &CLI);

  // Check if it's really possible to do a tail call.
  if (IsTailCall)
    IsTailCall = isEligibleForTailCallOptimization(ArgCCInfo, CLI, MF, ArgLocs);

  if (IsTailCall)
    ++NumTailCalls;
  else if (CLI.CB && CLI.CB->isMustTailCall())
    report_fatal_error("failed to perform tail call elimination on a call "
                       "site marked musttail");

  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NumBytes = ArgCCInfo.getNextStackOffset();

  // Create local copies for byval args
  SmallVector<SDValue, 8> ByValArgs;
  for (unsigned i = 0, e = Outs.size(); i != e; ++i) {
    ISD::ArgFlagsTy Flags = Outs[i].Flags;
    if (!Flags.isByVal())
      continue;

    SDValue Arg = OutVals[i];
    unsigned Size = Flags.getByValSize();
    Align Alignment = Flags.getNonZeroByValAlign();

    int FI =
        MF.getFrameInfo().CreateStackObject(Size, Alignment, /*isSS=*/false);
    SDValue FIPtr = DAG.getFrameIndex(FI, getPointerTy(DAG.getDataLayout()));
    SDValue SizeNode = DAG.getConstant(Size, DL, XLenVT);

    Chain = DAG.getMemcpy(Chain, DL, FIPtr, Arg, SizeNode, Alignment,
                          /*IsVolatile=*/false,
                          /*AlwaysInline=*/false, IsTailCall,
                          MachinePointerInfo(), MachinePointerInfo());
    ByValArgs.push_back(FIPtr);
  }

  if (!IsTailCall)
    Chain = DAG.getCALLSEQ_START(Chain, NumBytes, 0, CLI.DL);

  // Copy argument values to their designated locations.
  SmallVector<std::pair<Register, SDValue>, 8> RegsToPass;
  SmallVector<SDValue, 8> MemOpChains;
  SDValue StackPtr;
  for (unsigned i = 0, j = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    SDValue ArgValue = OutVals[i];
    ISD::ArgFlagsTy Flags = Outs[i].Flags;

    // Handle passing f64 on RV32D with a soft float ABI as a special case.
    bool IsF64OnRV32DSoftABI =
        VA.getLocVT() == MVT::i32 && VA.getValVT() == MVT::f64;
    if (IsF64OnRV32DSoftABI && VA.isRegLoc()) {
      SDValue SplitF64 = DAG.getNode(
          RISCVISD::SplitF64, DL, DAG.getVTList(MVT::i32, MVT::i32), ArgValue);
      SDValue Lo = SplitF64.getValue(0);
      SDValue Hi = SplitF64.getValue(1);

      Register RegLo = VA.getLocReg();
      RegsToPass.push_back(std::make_pair(RegLo, Lo));

      if (RegLo == RISCV::X17) {
        // Second half of f64 is passed on the stack.
        // Work out the address of the stack slot.
        if (!StackPtr.getNode())
          StackPtr = DAG.getCopyFromReg(Chain, DL, RISCV::X2, PtrVT);
        // Emit the store.
        MemOpChains.push_back(
            DAG.getStore(Chain, DL, Hi, StackPtr, MachinePointerInfo()));
      } else {
        // Second half of f64 is passed in another GPR.
        assert(RegLo < RISCV::X31 && "Invalid register pair");
        Register RegHigh = RegLo + 1;
        RegsToPass.push_back(std::make_pair(RegHigh, Hi));
      }
      continue;
    }

    // IsF64OnRV32DSoftABI && VA.isMemLoc() is handled below in the same way
    // as any other MemLoc.

    // Promote the value if needed.
    // For now, only handle fully promoted and indirect arguments.
    if (VA.getLocInfo() == CCValAssign::Indirect) {
      // Store the argument in a stack slot and pass its address.
      SDValue SpillSlot = DAG.CreateStackTemporary(Outs[i].ArgVT);
      int FI = cast<FrameIndexSDNode>(SpillSlot)->getIndex();
      MemOpChains.push_back(
          DAG.getStore(Chain, DL, ArgValue, SpillSlot,
                       MachinePointerInfo::getFixedStack(MF, FI)));
      // If the original argument was split (e.g. i128), we need
      // to store all parts of it here (and pass just one address).
      unsigned ArgIndex = Outs[i].OrigArgIndex;
      assert(Outs[i].PartOffset == 0);
      while (i + 1 != e && Outs[i + 1].OrigArgIndex == ArgIndex) {
        SDValue PartValue = OutVals[i + 1];
        unsigned PartOffset = Outs[i + 1].PartOffset;
        SDValue Address = DAG.getNode(ISD::ADD, DL, PtrVT, SpillSlot,
                                      DAG.getIntPtrConstant(PartOffset, DL));
        MemOpChains.push_back(
            DAG.getStore(Chain, DL, PartValue, Address,
                         MachinePointerInfo::getFixedStack(MF, FI)));
        ++i;
      }
      ArgValue = SpillSlot;
    } else {
      ArgValue = convertValVTToLocVT(DAG, ArgValue, VA, DL);
    }

    // Use local copy if it is a byval arg.
    if (Flags.isByVal())
      ArgValue = ByValArgs[j++];

    if (VA.isRegLoc()) {
      // Queue up the argument copies and emit them at the end.
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), ArgValue));
    } else {
      assert(VA.isMemLoc() && "Argument not register or memory");
      assert(!IsTailCall && "Tail call not allowed if stack is used "
                            "for passing parameters");

      // Work out the address of the stack slot.
      if (!StackPtr.getNode())
        StackPtr = DAG.getCopyFromReg(Chain, DL, RISCV::X2, PtrVT);
      SDValue Address =
          DAG.getNode(ISD::ADD, DL, PtrVT, StackPtr,
                      DAG.getIntPtrConstant(VA.getLocMemOffset(), DL));

      // Emit the store.
      MemOpChains.push_back(
          DAG.getStore(Chain, DL, ArgValue, Address, MachinePointerInfo()));
    }
  }

  // Join the stores, which are independent of one another.
  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, MemOpChains);

  SDValue Glue;

  // Build a sequence of copy-to-reg nodes, chained and glued together.
  for (auto &Reg : RegsToPass) {
    Chain = DAG.getCopyToReg(Chain, DL, Reg.first, Reg.second, Glue);
    Glue = Chain.getValue(1);
  }

  // Validate that none of the argument registers have been marked as
  // reserved, if so report an error. Do the same for the return address if this
  // is not a tailcall.
  validateCCReservedRegs(RegsToPass, MF);
  if (!IsTailCall &&
      MF.getSubtarget<RISCVSubtarget>().isRegisterReservedByUser(RISCV::X1))
    MF.getFunction().getContext().diagnose(DiagnosticInfoUnsupported{
        MF.getFunction(),
        "Return address register required, but has been reserved."});

  // If the callee is a GlobalAddress/ExternalSymbol node, turn it into a
  // TargetGlobalAddress/TargetExternalSymbol node so that legalize won't
  // split it and then direct call can be matched by PseudoCALL.
  if (GlobalAddressSDNode *S = dyn_cast<GlobalAddressSDNode>(Callee)) {
    const GlobalValue *GV = S->getGlobal();

    unsigned OpFlags = RISCVII::MO_CALL;
    if (!getTargetMachine().shouldAssumeDSOLocal(*GV->getParent(), GV))
      OpFlags = RISCVII::MO_PLT;

    Callee = DAG.getTargetGlobalAddress(GV, DL, PtrVT, 0, OpFlags);
  } else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Callee)) {
    unsigned OpFlags = RISCVII::MO_CALL;

    if (!getTargetMachine().shouldAssumeDSOLocal(*MF.getFunction().getParent(),
                                                 nullptr))
      OpFlags = RISCVII::MO_PLT;

    Callee = DAG.getTargetExternalSymbol(S->getSymbol(), PtrVT, OpFlags);
  }

  // The first call operand is the chain and the second is the target address.
  SmallVector<SDValue, 8> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  // Add argument registers to the end of the list so that they are
  // known live into the call.
  for (auto &Reg : RegsToPass)
    Ops.push_back(DAG.getRegister(Reg.first, Reg.second.getValueType()));

  if (!IsTailCall) {
    // Add a register mask operand representing the call-preserved registers.
    const TargetRegisterInfo *TRI = Subtarget.getRegisterInfo();
    const uint32_t *Mask = TRI->getCallPreservedMask(MF, CallConv);
    assert(Mask && "Missing call preserved mask for calling convention");
    Ops.push_back(DAG.getRegisterMask(Mask));
  }

  // Glue the call to the argument copies, if any.
  if (Glue.getNode())
    Ops.push_back(Glue);

  // Emit the call.
  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);

  if (IsTailCall) {
    MF.getFrameInfo().setHasTailCall();
    return DAG.getNode(RISCVISD::TAIL, DL, NodeTys, Ops);
  }

  Chain = DAG.getNode(RISCVISD::CALL, DL, NodeTys, Ops);
  DAG.addNoMergeSiteInfo(Chain.getNode(), CLI.NoMerge);
  Glue = Chain.getValue(1);

  // Mark the end of the call, which is glued to the call itself.
  Chain = DAG.getCALLSEQ_END(Chain,
                             DAG.getConstant(NumBytes, DL, PtrVT, true),
                             DAG.getConstant(0, DL, PtrVT, true),
                             Glue, DL);
  Glue = Chain.getValue(1);

  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RVLocs;
  CCState RetCCInfo(CallConv, IsVarArg, MF, RVLocs, *DAG.getContext());
  analyzeInputArgs(MF, RetCCInfo, Ins, /*IsRet=*/true);

  // Copy all of the result registers out of their specified physreg.
  for (auto &VA : RVLocs) {
    // Copy the value out
    SDValue RetValue =
        DAG.getCopyFromReg(Chain, DL, VA.getLocReg(), VA.getLocVT(), Glue);
    // Glue the RetValue to the end of the call sequence
    Chain = RetValue.getValue(1);
    Glue = RetValue.getValue(2);

    if (VA.getLocVT() == MVT::i32 && VA.getValVT() == MVT::f64) {
      assert(VA.getLocReg() == ArgGPRs[0] && "Unexpected reg assignment");
      SDValue RetValue2 =
          DAG.getCopyFromReg(Chain, DL, ArgGPRs[1], MVT::i32, Glue);
      Chain = RetValue2.getValue(1);
      Glue = RetValue2.getValue(2);
      RetValue = DAG.getNode(RISCVISD::BuildPairF64, DL, MVT::f64, RetValue,
                             RetValue2);
    }

    RetValue = convertLocVTToValVT(DAG, RetValue, VA, DL);

    InVals.push_back(RetValue);
  }

  return Chain;
}

bool RISCVTargetLowering::CanLowerReturn(
    CallingConv::ID CallConv, MachineFunction &MF, bool IsVarArg,
    const SmallVectorImpl<ISD::OutputArg> &Outs, LLVMContext &Context) const {
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, RVLocs, Context);
  for (unsigned i = 0, e = Outs.size(); i != e; ++i) {
    MVT VT = Outs[i].VT;
    ISD::ArgFlagsTy ArgFlags = Outs[i].Flags;
    RISCVABI::ABI ABI = MF.getSubtarget<RISCVSubtarget>().getTargetABI();
    if (CC_RISCV(MF.getDataLayout(), ABI, i, VT, VT, CCValAssign::Full,
                 ArgFlags, CCInfo, /*IsFixed=*/true, /*IsRet=*/true, nullptr))
      return false;
  }
  return true;
}

SDValue
RISCVTargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                                 bool IsVarArg,
                                 const SmallVectorImpl<ISD::OutputArg> &Outs,
                                 const SmallVectorImpl<SDValue> &OutVals,
                                 const SDLoc &DL, SelectionDAG &DAG) const {
  const MachineFunction &MF = DAG.getMachineFunction();
  const RISCVSubtarget &STI = MF.getSubtarget<RISCVSubtarget>();

  // Stores the assignment of the return value to a location.
  SmallVector<CCValAssign, 16> RVLocs;

  // Info about the registers and stack slot.
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), RVLocs,
                 *DAG.getContext());

  analyzeOutputArgs(DAG.getMachineFunction(), CCInfo, Outs, /*IsRet=*/true,
                    nullptr);

  SDValue Glue;
  SmallVector<SDValue, 4> RetOps(1, Chain);

  // Copy the result values into the output registers.
  for (unsigned i = 0, e = RVLocs.size(); i < e; ++i) {
    SDValue Val = OutVals[i];
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");

    if (VA.getLocVT() == MVT::i32 && VA.getValVT() == MVT::f64) {
      // Handle returning f64 on RV32D with a soft float ABI.
      assert(VA.isRegLoc() && "Expected return via registers");
      SDValue SplitF64 = DAG.getNode(RISCVISD::SplitF64, DL,
                                     DAG.getVTList(MVT::i32, MVT::i32), Val);
      SDValue Lo = SplitF64.getValue(0);
      SDValue Hi = SplitF64.getValue(1);
      Register RegLo = VA.getLocReg();
      assert(RegLo < RISCV::X31 && "Invalid register pair");
      Register RegHi = RegLo + 1;

      if (STI.isRegisterReservedByUser(RegLo) ||
          STI.isRegisterReservedByUser(RegHi))
        MF.getFunction().getContext().diagnose(DiagnosticInfoUnsupported{
            MF.getFunction(),
            "Return value register required, but has been reserved."});

      Chain = DAG.getCopyToReg(Chain, DL, RegLo, Lo, Glue);
      Glue = Chain.getValue(1);
      RetOps.push_back(DAG.getRegister(RegLo, MVT::i32));
      Chain = DAG.getCopyToReg(Chain, DL, RegHi, Hi, Glue);
      Glue = Chain.getValue(1);
      RetOps.push_back(DAG.getRegister(RegHi, MVT::i32));
    } else {
      // Handle a 'normal' return.
      Val = convertValVTToLocVT(DAG, Val, VA, DL);
      Chain = DAG.getCopyToReg(Chain, DL, VA.getLocReg(), Val, Glue);

      if (STI.isRegisterReservedByUser(VA.getLocReg()))
        MF.getFunction().getContext().diagnose(DiagnosticInfoUnsupported{
            MF.getFunction(),
            "Return value register required, but has been reserved."});

      // Guarantee that all emitted copies are stuck together.
      Glue = Chain.getValue(1);
      RetOps.push_back(DAG.getRegister(VA.getLocReg(), VA.getLocVT()));
    }
  }

  RetOps[0] = Chain; // Update chain.

  // Add the glue node if we have it.
  if (Glue.getNode()) {
    RetOps.push_back(Glue);
  }

  // Interrupt service routines use different return instructions.
  const Function &Func = DAG.getMachineFunction().getFunction();
  if (Func.hasFnAttribute("interrupt")) {
    if (!Func.getReturnType()->isVoidTy())
      report_fatal_error(
          "Functions with the interrupt attribute must have void return type!");

    MachineFunction &MF = DAG.getMachineFunction();
    StringRef Kind =
      MF.getFunction().getFnAttribute("interrupt").getValueAsString();

    unsigned RetOpc;
    if (Kind == "user")
      RetOpc = RISCVISD::URET_FLAG;
    else if (Kind == "supervisor")
      RetOpc = RISCVISD::SRET_FLAG;
    else
      RetOpc = RISCVISD::MRET_FLAG;

    return DAG.getNode(RetOpc, DL, MVT::Other, RetOps);
  }

  return DAG.getNode(RISCVISD::RET_FLAG, DL, MVT::Other, RetOps);
}

void RISCVTargetLowering::validateCCReservedRegs(
    const SmallVectorImpl<std::pair<llvm::Register, llvm::SDValue>> &Regs,
    MachineFunction &MF) const {
  const Function &F = MF.getFunction();
  const RISCVSubtarget &STI = MF.getSubtarget<RISCVSubtarget>();

  if (std::any_of(std::begin(Regs), std::end(Regs), [&STI](auto Reg) {
        return STI.isRegisterReservedByUser(Reg.first);
      }))
    F.getContext().diagnose(DiagnosticInfoUnsupported{
        F, "Argument register required, but has been reserved."});
}

bool RISCVTargetLowering::mayBeEmittedAsTailCall(const CallInst *CI) const {
  return CI->isTailCall();
}

const char *RISCVTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch ((RISCVISD::NodeType)Opcode) {
  case RISCVISD::FIRST_NUMBER:
    break;
  case RISCVISD::RET_FLAG:
    return "RISCVISD::RET_FLAG";
  case RISCVISD::URET_FLAG:
    return "RISCVISD::URET_FLAG";
  case RISCVISD::SRET_FLAG:
    return "RISCVISD::SRET_FLAG";
  case RISCVISD::MRET_FLAG:
    return "RISCVISD::MRET_FLAG";
  case RISCVISD::CALL:
    return "RISCVISD::CALL";
  case RISCVISD::SELECT_CC:
    return "RISCVISD::SELECT_CC";
  case RISCVISD::BuildPairF64:
    return "RISCVISD::BuildPairF64";
  case RISCVISD::SplitF64:
    return "RISCVISD::SplitF64";
  case RISCVISD::TAIL:
    return "RISCVISD::TAIL";
  case RISCVISD::SLLW:
    return "RISCVISD::SLLW";
  case RISCVISD::SRAW:
    return "RISCVISD::SRAW";
  case RISCVISD::SRLW:
    return "RISCVISD::SRLW";
  case RISCVISD::DIVW:
    return "RISCVISD::DIVW";
  case RISCVISD::DIVUW:
    return "RISCVISD::DIVUW";
  case RISCVISD::REMUW:
    return "RISCVISD::REMUW";
  case RISCVISD::FMV_W_X_RV64:
    return "RISCVISD::FMV_W_X_RV64";
  case RISCVISD::FMV_X_ANYEXTW_RV64:
    return "RISCVISD::FMV_X_ANYEXTW_RV64";
  case RISCVISD::READ_CYCLE_WIDE:
    return "RISCVISD::READ_CYCLE_WIDE";
  }
  return nullptr;
}

/// getConstraintType - Given a constraint letter, return the type of
/// constraint it is for this target.
RISCVTargetLowering::ConstraintType
RISCVTargetLowering::getConstraintType(StringRef Constraint) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    default:
      break;
    case 'f':
      return C_RegisterClass;
    case 'I':
    case 'J':
    case 'K':
      return C_Immediate;
    case 'A':
      return C_Memory;
    }
  }
  return TargetLowering::getConstraintType(Constraint);
}

std::pair<unsigned, const TargetRegisterClass *>
RISCVTargetLowering::getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                                                  StringRef Constraint,
                                                  MVT VT) const {
  // First, see if this is a constraint that directly corresponds to a
  // RISCV register class.
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    case 'r':
      return std::make_pair(0U, &RISCV::GPRRegClass);
    case 'f':
      if (Subtarget.hasStdExtF() && VT == MVT::f32)
        return std::make_pair(0U, &RISCV::FPR32RegClass);
      if (Subtarget.hasStdExtD() && VT == MVT::f64)
        return std::make_pair(0U, &RISCV::FPR64RegClass);
      break;
    default:
      break;
    }
  }

  // Clang will correctly decode the usage of register name aliases into their
  // official names. However, other frontends like `rustc` do not. This allows
  // users of these frontends to use the ABI names for registers in LLVM-style
  // register constraints.
  Register XRegFromAlias = StringSwitch<Register>(Constraint.lower())
                               .Case("{zero}", RISCV::X0)
                               .Case("{ra}", RISCV::X1)
                               .Case("{sp}", RISCV::X2)
                               .Case("{gp}", RISCV::X3)
                               .Case("{tp}", RISCV::X4)
                               .Case("{t0}", RISCV::X5)
                               .Case("{t1}", RISCV::X6)
                               .Case("{t2}", RISCV::X7)
                               .Cases("{s0}", "{fp}", RISCV::X8)
                               .Case("{s1}", RISCV::X9)
                               .Case("{a0}", RISCV::X10)
                               .Case("{a1}", RISCV::X11)
                               .Case("{a2}", RISCV::X12)
                               .Case("{a3}", RISCV::X13)
                               .Case("{a4}", RISCV::X14)
                               .Case("{a5}", RISCV::X15)
                               .Case("{a6}", RISCV::X16)
                               .Case("{a7}", RISCV::X17)
                               .Case("{s2}", RISCV::X18)
                               .Case("{s3}", RISCV::X19)
                               .Case("{s4}", RISCV::X20)
                               .Case("{s5}", RISCV::X21)
                               .Case("{s6}", RISCV::X22)
                               .Case("{s7}", RISCV::X23)
                               .Case("{s8}", RISCV::X24)
                               .Case("{s9}", RISCV::X25)
                               .Case("{s10}", RISCV::X26)
                               .Case("{s11}", RISCV::X27)
                               .Case("{t3}", RISCV::X28)
                               .Case("{t4}", RISCV::X29)
                               .Case("{t5}", RISCV::X30)
                               .Case("{t6}", RISCV::X31)
                               .Default(RISCV::NoRegister);
  if (XRegFromAlias != RISCV::NoRegister)
    return std::make_pair(XRegFromAlias, &RISCV::GPRRegClass);

  // Since TargetLowering::getRegForInlineAsmConstraint uses the name of the
  // TableGen record rather than the AsmName to choose registers for InlineAsm
  // constraints, plus we want to match those names to the widest floating point
  // register type available, manually select floating point registers here.
  //
  // The second case is the ABI name of the register, so that frontends can also
  // use the ABI names in register constraint lists.
  if (Subtarget.hasStdExtF() || Subtarget.hasStdExtD()) {
    std::pair<Register, Register> FReg =
        StringSwitch<std::pair<Register, Register>>(Constraint.lower())
            .Cases("{f0}", "{ft0}", {RISCV::F0_F, RISCV::F0_D})
            .Cases("{f1}", "{ft1}", {RISCV::F1_F, RISCV::F1_D})
            .Cases("{f2}", "{ft2}", {RISCV::F2_F, RISCV::F2_D})
            .Cases("{f3}", "{ft3}", {RISCV::F3_F, RISCV::F3_D})
            .Cases("{f4}", "{ft4}", {RISCV::F4_F, RISCV::F4_D})
            .Cases("{f5}", "{ft5}", {RISCV::F5_F, RISCV::F5_D})
            .Cases("{f6}", "{ft6}", {RISCV::F6_F, RISCV::F6_D})
            .Cases("{f7}", "{ft7}", {RISCV::F7_F, RISCV::F7_D})
            .Cases("{f8}", "{fs0}", {RISCV::F8_F, RISCV::F8_D})
            .Cases("{f9}", "{fs1}", {RISCV::F9_F, RISCV::F9_D})
            .Cases("{f10}", "{fa0}", {RISCV::F10_F, RISCV::F10_D})
            .Cases("{f11}", "{fa1}", {RISCV::F11_F, RISCV::F11_D})
            .Cases("{f12}", "{fa2}", {RISCV::F12_F, RISCV::F12_D})
            .Cases("{f13}", "{fa3}", {RISCV::F13_F, RISCV::F13_D})
            .Cases("{f14}", "{fa4}", {RISCV::F14_F, RISCV::F14_D})
            .Cases("{f15}", "{fa5}", {RISCV::F15_F, RISCV::F15_D})
            .Cases("{f16}", "{fa6}", {RISCV::F16_F, RISCV::F16_D})
            .Cases("{f17}", "{fa7}", {RISCV::F17_F, RISCV::F17_D})
            .Cases("{f18}", "{fs2}", {RISCV::F18_F, RISCV::F18_D})
            .Cases("{f19}", "{fs3}", {RISCV::F19_F, RISCV::F19_D})
            .Cases("{f20}", "{fs4}", {RISCV::F20_F, RISCV::F20_D})
            .Cases("{f21}", "{fs5}", {RISCV::F21_F, RISCV::F21_D})
            .Cases("{f22}", "{fs6}", {RISCV::F22_F, RISCV::F22_D})
            .Cases("{f23}", "{fs7}", {RISCV::F23_F, RISCV::F23_D})
            .Cases("{f24}", "{fs8}", {RISCV::F24_F, RISCV::F24_D})
            .Cases("{f25}", "{fs9}", {RISCV::F25_F, RISCV::F25_D})
            .Cases("{f26}", "{fs10}", {RISCV::F26_F, RISCV::F26_D})
            .Cases("{f27}", "{fs11}", {RISCV::F27_F, RISCV::F27_D})
            .Cases("{f28}", "{ft8}", {RISCV::F28_F, RISCV::F28_D})
            .Cases("{f29}", "{ft9}", {RISCV::F29_F, RISCV::F29_D})
            .Cases("{f30}", "{ft10}", {RISCV::F30_F, RISCV::F30_D})
            .Cases("{f31}", "{ft11}", {RISCV::F31_F, RISCV::F31_D})
            .Default({RISCV::NoRegister, RISCV::NoRegister});
    if (FReg.first != RISCV::NoRegister)
      return Subtarget.hasStdExtD()
                 ? std::make_pair(FReg.second, &RISCV::FPR64RegClass)
                 : std::make_pair(FReg.first, &RISCV::FPR32RegClass);
  }

  return TargetLowering::getRegForInlineAsmConstraint(TRI, Constraint, VT);
}

unsigned
RISCVTargetLowering::getInlineAsmMemConstraint(StringRef ConstraintCode) const {
  // Currently only support length 1 constraints.
  if (ConstraintCode.size() == 1) {
    switch (ConstraintCode[0]) {
    case 'A':
      return InlineAsm::Constraint_A;
    default:
      break;
    }
  }

  return TargetLowering::getInlineAsmMemConstraint(ConstraintCode);
}

void RISCVTargetLowering::LowerAsmOperandForConstraint(
    SDValue Op, std::string &Constraint, std::vector<SDValue> &Ops,
    SelectionDAG &DAG) const {
  // Currently only support length 1 constraints.
  if (Constraint.length() == 1) {
    switch (Constraint[0]) {
    case 'I':
      // Validate & create a 12-bit signed immediate operand.
      if (auto *C = dyn_cast<ConstantSDNode>(Op)) {
        uint64_t CVal = C->getSExtValue();
        if (isInt<12>(CVal))
          Ops.push_back(
              DAG.getTargetConstant(CVal, SDLoc(Op), Subtarget.getXLenVT()));
      }
      return;
    case 'J':
      // Validate & create an integer zero operand.
      if (auto *C = dyn_cast<ConstantSDNode>(Op))
        if (C->getZExtValue() == 0)
          Ops.push_back(
              DAG.getTargetConstant(0, SDLoc(Op), Subtarget.getXLenVT()));
      return;
    case 'K':
      // Validate & create a 5-bit unsigned immediate operand.
      if (auto *C = dyn_cast<ConstantSDNode>(Op)) {
        uint64_t CVal = C->getZExtValue();
        if (isUInt<5>(CVal))
          Ops.push_back(
              DAG.getTargetConstant(CVal, SDLoc(Op), Subtarget.getXLenVT()));
      }
      return;
    default:
      break;
    }
  }
  TargetLowering::LowerAsmOperandForConstraint(Op, Constraint, Ops, DAG);
}

Instruction *RISCVTargetLowering::emitLeadingFence(IRBuilder<> &Builder,
                                                   Instruction *Inst,
                                                   AtomicOrdering Ord) const {
  if (isa<LoadInst>(Inst) && Ord == AtomicOrdering::SequentiallyConsistent)
    return Builder.CreateFence(Ord);
  if (isa<StoreInst>(Inst) && isReleaseOrStronger(Ord))
    return Builder.CreateFence(AtomicOrdering::Release);
  return nullptr;
}

Instruction *RISCVTargetLowering::emitTrailingFence(IRBuilder<> &Builder,
                                                    Instruction *Inst,
                                                    AtomicOrdering Ord) const {
  if (isa<LoadInst>(Inst) && isAcquireOrStronger(Ord))
    return Builder.CreateFence(AtomicOrdering::Acquire);
  return nullptr;
}

TargetLowering::AtomicExpansionKind
RISCVTargetLowering::shouldExpandAtomicRMWInIR(AtomicRMWInst *AI) const {
  // atomicrmw {fadd,fsub} must be expanded to use compare-exchange, as floating
  // point operations can't be used in an lr/sc sequence without breaking the
  // forward-progress guarantee.
  if (AI->isFloatingPointOperation())
    return AtomicExpansionKind::CmpXChg;

  unsigned Size = AI->getType()->getPrimitiveSizeInBits();
  if (Size == 8 || Size == 16)
    return AtomicExpansionKind::MaskedIntrinsic;
  return AtomicExpansionKind::None;
}

static Intrinsic::ID
getIntrinsicForMaskedAtomicRMWBinOp(unsigned XLen, AtomicRMWInst::BinOp BinOp) {
  if (XLen == 32) {
    switch (BinOp) {
    default:
      llvm_unreachable("Unexpected AtomicRMW BinOp");
    case AtomicRMWInst::Xchg:
      return Intrinsic::riscv_masked_atomicrmw_xchg_i32;
    case AtomicRMWInst::Add:
      return Intrinsic::riscv_masked_atomicrmw_add_i32;
    case AtomicRMWInst::Sub:
      return Intrinsic::riscv_masked_atomicrmw_sub_i32;
    case AtomicRMWInst::Nand:
      return Intrinsic::riscv_masked_atomicrmw_nand_i32;
    case AtomicRMWInst::Max:
      return Intrinsic::riscv_masked_atomicrmw_max_i32;
    case AtomicRMWInst::Min:
      return Intrinsic::riscv_masked_atomicrmw_min_i32;
    case AtomicRMWInst::UMax:
      return Intrinsic::riscv_masked_atomicrmw_umax_i32;
    case AtomicRMWInst::UMin:
      return Intrinsic::riscv_masked_atomicrmw_umin_i32;
    }
  }

  if (XLen == 64) {
    switch (BinOp) {
    default:
      llvm_unreachable("Unexpected AtomicRMW BinOp");
    case AtomicRMWInst::Xchg:
      return Intrinsic::riscv_masked_atomicrmw_xchg_i64;
    case AtomicRMWInst::Add:
      return Intrinsic::riscv_masked_atomicrmw_add_i64;
    case AtomicRMWInst::Sub:
      return Intrinsic::riscv_masked_atomicrmw_sub_i64;
    case AtomicRMWInst::Nand:
      return Intrinsic::riscv_masked_atomicrmw_nand_i64;
    case AtomicRMWInst::Max:
      return Intrinsic::riscv_masked_atomicrmw_max_i64;
    case AtomicRMWInst::Min:
      return Intrinsic::riscv_masked_atomicrmw_min_i64;
    case AtomicRMWInst::UMax:
      return Intrinsic::riscv_masked_atomicrmw_umax_i64;
    case AtomicRMWInst::UMin:
      return Intrinsic::riscv_masked_atomicrmw_umin_i64;
    }
  }

  llvm_unreachable("Unexpected XLen\n");
}

Value *RISCVTargetLowering::emitMaskedAtomicRMWIntrinsic(
    IRBuilder<> &Builder, AtomicRMWInst *AI, Value *AlignedAddr, Value *Incr,
    Value *Mask, Value *ShiftAmt, AtomicOrdering Ord) const {
  unsigned XLen = Subtarget.getXLen();
  Value *Ordering =
      Builder.getIntN(XLen, static_cast<uint64_t>(AI->getOrdering()));
  Type *Tys[] = {AlignedAddr->getType()};
  Function *LrwOpScwLoop = Intrinsic::getDeclaration(
      AI->getModule(),
      getIntrinsicForMaskedAtomicRMWBinOp(XLen, AI->getOperation()), Tys);

  if (XLen == 64) {
    Incr = Builder.CreateSExt(Incr, Builder.getInt64Ty());
    Mask = Builder.CreateSExt(Mask, Builder.getInt64Ty());
    ShiftAmt = Builder.CreateSExt(ShiftAmt, Builder.getInt64Ty());
  }

  Value *Result;

  // Must pass the shift amount needed to sign extend the loaded value prior
  // to performing a signed comparison for min/max. ShiftAmt is the number of
  // bits to shift the value into position. Pass XLen-ShiftAmt-ValWidth, which
  // is the number of bits to left+right shift the value in order to
  // sign-extend.
  if (AI->getOperation() == AtomicRMWInst::Min ||
      AI->getOperation() == AtomicRMWInst::Max) {
    const DataLayout &DL = AI->getModule()->getDataLayout();
    unsigned ValWidth =
        DL.getTypeStoreSizeInBits(AI->getValOperand()->getType());
    Value *SextShamt =
        Builder.CreateSub(Builder.getIntN(XLen, XLen - ValWidth), ShiftAmt);
    Result = Builder.CreateCall(LrwOpScwLoop,
                                {AlignedAddr, Incr, Mask, SextShamt, Ordering});
  } else {
    Result =
        Builder.CreateCall(LrwOpScwLoop, {AlignedAddr, Incr, Mask, Ordering});
  }

  if (XLen == 64)
    Result = Builder.CreateTrunc(Result, Builder.getInt32Ty());
  return Result;
}

TargetLowering::AtomicExpansionKind
RISCVTargetLowering::shouldExpandAtomicCmpXchgInIR(
    AtomicCmpXchgInst *CI) const {
  unsigned Size = CI->getCompareOperand()->getType()->getPrimitiveSizeInBits();
  if (Size == 8 || Size == 16)
    return AtomicExpansionKind::MaskedIntrinsic;
  return AtomicExpansionKind::None;
}

Value *RISCVTargetLowering::emitMaskedAtomicCmpXchgIntrinsic(
    IRBuilder<> &Builder, AtomicCmpXchgInst *CI, Value *AlignedAddr,
    Value *CmpVal, Value *NewVal, Value *Mask, AtomicOrdering Ord) const {
  unsigned XLen = Subtarget.getXLen();
  Value *Ordering = Builder.getIntN(XLen, static_cast<uint64_t>(Ord));
  Intrinsic::ID CmpXchgIntrID = Intrinsic::riscv_masked_cmpxchg_i32;
  if (XLen == 64) {
    CmpVal = Builder.CreateSExt(CmpVal, Builder.getInt64Ty());
    NewVal = Builder.CreateSExt(NewVal, Builder.getInt64Ty());
    Mask = Builder.CreateSExt(Mask, Builder.getInt64Ty());
    CmpXchgIntrID = Intrinsic::riscv_masked_cmpxchg_i64;
  }
  Type *Tys[] = {AlignedAddr->getType()};
  Function *MaskedCmpXchg =
      Intrinsic::getDeclaration(CI->getModule(), CmpXchgIntrID, Tys);
  Value *Result = Builder.CreateCall(
      MaskedCmpXchg, {AlignedAddr, CmpVal, NewVal, Mask, Ordering});
  if (XLen == 64)
    Result = Builder.CreateTrunc(Result, Builder.getInt32Ty());
  return Result;
}

Register RISCVTargetLowering::getExceptionPointerRegister(
    const Constant *PersonalityFn) const {
  return RISCV::X10;
}

Register RISCVTargetLowering::getExceptionSelectorRegister(
    const Constant *PersonalityFn) const {
  return RISCV::X11;
}

bool RISCVTargetLowering::shouldExtendTypeInLibCall(EVT Type) const {
  // Return false to suppress the unnecessary extensions if the LibCall
  // arguments or return value is f32 type for LP64 ABI.
  RISCVABI::ABI ABI = Subtarget.getTargetABI();
  if (ABI == RISCVABI::ABI_LP64 && (Type == MVT::f32))
    return false;

  return true;
}

bool RISCVTargetLowering::decomposeMulByConstant(LLVMContext &Context, EVT VT,
                                                 SDValue C) const {
  // Check integral scalar types.
  if (VT.isScalarInteger()) {
    // Do not perform the transformation on riscv32 with the M extension.
    if (!Subtarget.is64Bit() && Subtarget.hasStdExtM())
      return false;
    if (auto *ConstNode = dyn_cast<ConstantSDNode>(C.getNode())) {
      if (ConstNode->getAPIntValue().getBitWidth() > 8 * sizeof(int64_t))
        return false;
      int64_t Imm = ConstNode->getSExtValue();
      if (isPowerOf2_64(Imm + 1) || isPowerOf2_64(Imm - 1) ||
          isPowerOf2_64(1 - Imm) || isPowerOf2_64(-1 - Imm))
        return true;
    }
  }

  return false;
}

#define GET_REGISTER_MATCHER
#include "RISCVGenAsmMatcher.inc"

Register
RISCVTargetLowering::getRegisterByName(const char *RegName, LLT VT,
                                       const MachineFunction &MF) const {
  Register Reg = MatchRegisterAltName(RegName);
  if (Reg == RISCV::NoRegister)
    Reg = MatchRegisterName(RegName);
  if (Reg == RISCV::NoRegister)
    report_fatal_error(
        Twine("Invalid register name \"" + StringRef(RegName) + "\"."));
  BitVector ReservedRegs = Subtarget.getRegisterInfo()->getReservedRegs(MF);
  if (!ReservedRegs.test(Reg) && !Subtarget.isRegisterReservedByUser(Reg))
    report_fatal_error(Twine("Trying to obtain non-reserved register \"" +
                             StringRef(RegName) + "\"."));
  return Reg;
}
