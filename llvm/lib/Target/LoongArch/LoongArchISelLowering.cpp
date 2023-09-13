//=- LoongArchISelLowering.cpp - LoongArch DAG Lowering Implementation  ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that LoongArch uses to lower LLVM code into
// a selection DAG.
//
//===----------------------------------------------------------------------===//

#include "LoongArchISelLowering.h"
#include "LoongArch.h"
#include "LoongArchMachineFunctionInfo.h"
#include "LoongArchRegisterInfo.h"
#include "LoongArchSubtarget.h"
#include "LoongArchTargetMachine.h"
#include "MCTargetDesc/LoongArchBaseInfo.h"
#include "MCTargetDesc/LoongArchMCTargetDesc.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/RuntimeLibcalls.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsLoongArch.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

#define DEBUG_TYPE "loongarch-isel-lowering"

STATISTIC(NumTailCalls, "Number of tail calls");

static cl::opt<bool> ZeroDivCheck("loongarch-check-zero-division", cl::Hidden,
                                  cl::desc("Trap on integer division by zero."),
                                  cl::init(false));

LoongArchTargetLowering::LoongArchTargetLowering(const TargetMachine &TM,
                                                 const LoongArchSubtarget &STI)
    : TargetLowering(TM), Subtarget(STI) {

  MVT GRLenVT = Subtarget.getGRLenVT();

  // Set up the register classes.

  addRegisterClass(GRLenVT, &LoongArch::GPRRegClass);
  if (Subtarget.hasBasicF())
    addRegisterClass(MVT::f32, &LoongArch::FPR32RegClass);
  if (Subtarget.hasBasicD())
    addRegisterClass(MVT::f64, &LoongArch::FPR64RegClass);

  static const MVT::SimpleValueType LSXVTs[] = {
      MVT::v16i8, MVT::v8i16, MVT::v4i32, MVT::v2i64, MVT::v4f32, MVT::v2f64};
  static const MVT::SimpleValueType LASXVTs[] = {
      MVT::v32i8, MVT::v16i16, MVT::v8i32, MVT::v4i64, MVT::v8f32, MVT::v4f64};

  if (Subtarget.hasExtLSX())
    for (MVT VT : LSXVTs)
      addRegisterClass(VT, &LoongArch::LSX128RegClass);

  if (Subtarget.hasExtLASX())
    for (MVT VT : LASXVTs)
      addRegisterClass(VT, &LoongArch::LASX256RegClass);

  // Set operations for LA32 and LA64.

  setLoadExtAction({ISD::EXTLOAD, ISD::SEXTLOAD, ISD::ZEXTLOAD}, GRLenVT,
                   MVT::i1, Promote);

  setOperationAction(ISD::SHL_PARTS, GRLenVT, Custom);
  setOperationAction(ISD::SRA_PARTS, GRLenVT, Custom);
  setOperationAction(ISD::SRL_PARTS, GRLenVT, Custom);
  setOperationAction(ISD::FP_TO_SINT, GRLenVT, Custom);
  setOperationAction(ISD::ROTL, GRLenVT, Expand);
  setOperationAction(ISD::CTPOP, GRLenVT, Expand);

  setOperationAction({ISD::GlobalAddress, ISD::BlockAddress, ISD::ConstantPool,
                      ISD::JumpTable, ISD::GlobalTLSAddress},
                     GRLenVT, Custom);

  setOperationAction(ISD::EH_DWARF_CFA, GRLenVT, Custom);

  setOperationAction(ISD::DYNAMIC_STACKALLOC, GRLenVT, Expand);
  setOperationAction({ISD::STACKSAVE, ISD::STACKRESTORE}, MVT::Other, Expand);
  setOperationAction(ISD::VASTART, MVT::Other, Custom);
  setOperationAction({ISD::VAARG, ISD::VACOPY, ISD::VAEND}, MVT::Other, Expand);

  setOperationAction(ISD::DEBUGTRAP, MVT::Other, Legal);
  setOperationAction(ISD::TRAP, MVT::Other, Legal);

  setOperationAction(ISD::INTRINSIC_VOID, MVT::Other, Custom);
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::Other, Custom);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::Other, Custom);

  // Expand bitreverse.i16 with native-width bitrev and shift for now, before
  // we get to know which of sll and revb.2h is faster.
  setOperationAction(ISD::BITREVERSE, MVT::i8, Custom);
  setOperationAction(ISD::BITREVERSE, GRLenVT, Legal);

  // LA32 does not have REVB.2W and REVB.D due to the 64-bit operands, and
  // the narrower REVB.W does not exist. But LA32 does have REVB.2H, so i16
  // and i32 could still be byte-swapped relatively cheaply.
  setOperationAction(ISD::BSWAP, MVT::i16, Custom);

  setOperationAction(ISD::BR_JT, MVT::Other, Expand);
  setOperationAction(ISD::BR_CC, GRLenVT, Expand);
  setOperationAction(ISD::SELECT_CC, GRLenVT, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);
  setOperationAction({ISD::SMUL_LOHI, ISD::UMUL_LOHI}, GRLenVT, Expand);

  setOperationAction(ISD::FP_TO_UINT, GRLenVT, Custom);
  setOperationAction(ISD::UINT_TO_FP, GRLenVT, Expand);

  // Set operations for LA64 only.

  if (Subtarget.is64Bit()) {
    setOperationAction(ISD::SHL, MVT::i32, Custom);
    setOperationAction(ISD::SRA, MVT::i32, Custom);
    setOperationAction(ISD::SRL, MVT::i32, Custom);
    setOperationAction(ISD::FP_TO_SINT, MVT::i32, Custom);
    setOperationAction(ISD::BITCAST, MVT::i32, Custom);
    setOperationAction(ISD::ROTR, MVT::i32, Custom);
    setOperationAction(ISD::ROTL, MVT::i32, Custom);
    setOperationAction(ISD::CTTZ, MVT::i32, Custom);
    setOperationAction(ISD::CTLZ, MVT::i32, Custom);
    setOperationAction(ISD::EH_DWARF_CFA, MVT::i32, Custom);
    setOperationAction(ISD::READ_REGISTER, MVT::i32, Custom);
    setOperationAction(ISD::WRITE_REGISTER, MVT::i32, Custom);
    setOperationAction(ISD::INTRINSIC_VOID, MVT::i32, Custom);
    setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::i32, Custom);
    setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::i32, Custom);

    setOperationAction(ISD::BITREVERSE, MVT::i32, Custom);
    setOperationAction(ISD::BSWAP, MVT::i32, Custom);
  }

  // Set operations for LA32 only.

  if (!Subtarget.is64Bit()) {
    setOperationAction(ISD::READ_REGISTER, MVT::i64, Custom);
    setOperationAction(ISD::WRITE_REGISTER, MVT::i64, Custom);
    setOperationAction(ISD::INTRINSIC_VOID, MVT::i64, Custom);
    setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::i64, Custom);
    setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::i64, Custom);

    // Set libcalls.
    setLibcallName(RTLIB::MUL_I128, nullptr);
  }

  static const ISD::CondCode FPCCToExpand[] = {
      ISD::SETOGT, ISD::SETOGE, ISD::SETUGT, ISD::SETUGE,
      ISD::SETGE,  ISD::SETNE,  ISD::SETGT};

  // Set operations for 'F' feature.

  if (Subtarget.hasBasicF()) {
    setCondCodeAction(FPCCToExpand, MVT::f32, Expand);

    setOperationAction(ISD::SELECT_CC, MVT::f32, Expand);
    setOperationAction(ISD::BR_CC, MVT::f32, Expand);
    setOperationAction(ISD::FMA, MVT::f32, Legal);
    setOperationAction(ISD::FMINNUM_IEEE, MVT::f32, Legal);
    setOperationAction(ISD::FMAXNUM_IEEE, MVT::f32, Legal);
    setOperationAction(ISD::STRICT_FSETCCS, MVT::f32, Legal);
    setOperationAction(ISD::STRICT_FSETCC, MVT::f32, Legal);
    setOperationAction(ISD::FSIN, MVT::f32, Expand);
    setOperationAction(ISD::FCOS, MVT::f32, Expand);
    setOperationAction(ISD::FSINCOS, MVT::f32, Expand);
    setOperationAction(ISD::FPOW, MVT::f32, Expand);
    setOperationAction(ISD::FREM, MVT::f32, Expand);

    if (Subtarget.is64Bit())
      setOperationAction(ISD::FRINT, MVT::f32, Legal);

    if (!Subtarget.hasBasicD()) {
      setOperationAction(ISD::FP_TO_UINT, MVT::i32, Custom);
      if (Subtarget.is64Bit()) {
        setOperationAction(ISD::SINT_TO_FP, MVT::i64, Custom);
        setOperationAction(ISD::UINT_TO_FP, MVT::i64, Custom);
      }
    }
  }

  // Set operations for 'D' feature.

  if (Subtarget.hasBasicD()) {
    setLoadExtAction(ISD::EXTLOAD, MVT::f64, MVT::f32, Expand);
    setTruncStoreAction(MVT::f64, MVT::f32, Expand);
    setCondCodeAction(FPCCToExpand, MVT::f64, Expand);

    setOperationAction(ISD::SELECT_CC, MVT::f64, Expand);
    setOperationAction(ISD::BR_CC, MVT::f64, Expand);
    setOperationAction(ISD::STRICT_FSETCCS, MVT::f64, Legal);
    setOperationAction(ISD::STRICT_FSETCC, MVT::f64, Legal);
    setOperationAction(ISD::FMA, MVT::f64, Legal);
    setOperationAction(ISD::FMINNUM_IEEE, MVT::f64, Legal);
    setOperationAction(ISD::FMAXNUM_IEEE, MVT::f64, Legal);
    setOperationAction(ISD::FSIN, MVT::f64, Expand);
    setOperationAction(ISD::FCOS, MVT::f64, Expand);
    setOperationAction(ISD::FSINCOS, MVT::f64, Expand);
    setOperationAction(ISD::FPOW, MVT::f64, Expand);
    setOperationAction(ISD::FREM, MVT::f64, Expand);

    if (Subtarget.is64Bit())
      setOperationAction(ISD::FRINT, MVT::f64, Legal);
  }

  // Set operations for 'LSX' feature.

  if (Subtarget.hasExtLSX())
    setOperationAction({ISD::UMAX, ISD::UMIN, ISD::SMAX, ISD::SMIN},
                       {MVT::v2i64, MVT::v4i32, MVT::v8i16, MVT::v16i8}, Legal);

  // Set operations for 'LASX' feature.

  if (Subtarget.hasExtLASX())
    setOperationAction({ISD::UMAX, ISD::UMIN, ISD::SMAX, ISD::SMIN},
                       {MVT::v4i64, MVT::v8i32, MVT::v16i16, MVT::v32i8},
                       Legal);

  // Set DAG combine for LA32 and LA64.

  setTargetDAGCombine(ISD::AND);
  setTargetDAGCombine(ISD::OR);
  setTargetDAGCombine(ISD::SRL);

  // Set DAG combine for 'LSX' feature.

  if (Subtarget.hasExtLSX())
    setTargetDAGCombine(ISD::INTRINSIC_WO_CHAIN);

  // Compute derived properties from the register classes.
  computeRegisterProperties(Subtarget.getRegisterInfo());

  setStackPointerRegisterToSaveRestore(LoongArch::R3);

  setBooleanContents(ZeroOrOneBooleanContent);

  setMaxAtomicSizeInBitsSupported(Subtarget.getGRLen());

  setMinCmpXchgSizeInBits(32);

  // Function alignments.
  setMinFunctionAlignment(Align(4));
  // Set preferred alignments.
  setPrefFunctionAlignment(Subtarget.getPrefFunctionAlignment());
  setPrefLoopAlignment(Subtarget.getPrefLoopAlignment());
  setMaxBytesForAlignment(Subtarget.getMaxBytesForAlignment());
}

bool LoongArchTargetLowering::isOffsetFoldingLegal(
    const GlobalAddressSDNode *GA) const {
  // In order to maximise the opportunity for common subexpression elimination,
  // keep a separate ADD node for the global address offset instead of folding
  // it in the global address node. Later peephole optimisations may choose to
  // fold it back in when profitable.
  return false;
}

SDValue LoongArchTargetLowering::LowerOperation(SDValue Op,
                                                SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  case ISD::EH_DWARF_CFA:
    return lowerEH_DWARF_CFA(Op, DAG);
  case ISD::GlobalAddress:
    return lowerGlobalAddress(Op, DAG);
  case ISD::GlobalTLSAddress:
    return lowerGlobalTLSAddress(Op, DAG);
  case ISD::INTRINSIC_WO_CHAIN:
    return lowerINTRINSIC_WO_CHAIN(Op, DAG);
  case ISD::INTRINSIC_W_CHAIN:
    return lowerINTRINSIC_W_CHAIN(Op, DAG);
  case ISD::INTRINSIC_VOID:
    return lowerINTRINSIC_VOID(Op, DAG);
  case ISD::BlockAddress:
    return lowerBlockAddress(Op, DAG);
  case ISD::JumpTable:
    return lowerJumpTable(Op, DAG);
  case ISD::SHL_PARTS:
    return lowerShiftLeftParts(Op, DAG);
  case ISD::SRA_PARTS:
    return lowerShiftRightParts(Op, DAG, true);
  case ISD::SRL_PARTS:
    return lowerShiftRightParts(Op, DAG, false);
  case ISD::ConstantPool:
    return lowerConstantPool(Op, DAG);
  case ISD::FP_TO_SINT:
    return lowerFP_TO_SINT(Op, DAG);
  case ISD::BITCAST:
    return lowerBITCAST(Op, DAG);
  case ISD::UINT_TO_FP:
    return lowerUINT_TO_FP(Op, DAG);
  case ISD::SINT_TO_FP:
    return lowerSINT_TO_FP(Op, DAG);
  case ISD::VASTART:
    return lowerVASTART(Op, DAG);
  case ISD::FRAMEADDR:
    return lowerFRAMEADDR(Op, DAG);
  case ISD::RETURNADDR:
    return lowerRETURNADDR(Op, DAG);
  case ISD::WRITE_REGISTER:
    return lowerWRITE_REGISTER(Op, DAG);
  }
  return SDValue();
}

SDValue LoongArchTargetLowering::lowerWRITE_REGISTER(SDValue Op,
                                                     SelectionDAG &DAG) const {

  if (Subtarget.is64Bit() && Op.getOperand(2).getValueType() == MVT::i32) {
    DAG.getContext()->emitError(
        "On LA64, only 64-bit registers can be written.");
    return Op.getOperand(0);
  }

  if (!Subtarget.is64Bit() && Op.getOperand(2).getValueType() == MVT::i64) {
    DAG.getContext()->emitError(
        "On LA32, only 32-bit registers can be written.");
    return Op.getOperand(0);
  }

  return Op;
}

SDValue LoongArchTargetLowering::lowerFRAMEADDR(SDValue Op,
                                                SelectionDAG &DAG) const {
  if (!isa<ConstantSDNode>(Op.getOperand(0))) {
    DAG.getContext()->emitError("argument to '__builtin_frame_address' must "
                                "be a constant integer");
    return SDValue();
  }

  MachineFunction &MF = DAG.getMachineFunction();
  MF.getFrameInfo().setFrameAddressIsTaken(true);
  Register FrameReg = Subtarget.getRegisterInfo()->getFrameRegister(MF);
  EVT VT = Op.getValueType();
  SDLoc DL(Op);
  SDValue FrameAddr = DAG.getCopyFromReg(DAG.getEntryNode(), DL, FrameReg, VT);
  unsigned Depth = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  int GRLenInBytes = Subtarget.getGRLen() / 8;

  while (Depth--) {
    int Offset = -(GRLenInBytes * 2);
    SDValue Ptr = DAG.getNode(ISD::ADD, DL, VT, FrameAddr,
                              DAG.getIntPtrConstant(Offset, DL));
    FrameAddr =
        DAG.getLoad(VT, DL, DAG.getEntryNode(), Ptr, MachinePointerInfo());
  }
  return FrameAddr;
}

SDValue LoongArchTargetLowering::lowerRETURNADDR(SDValue Op,
                                                 SelectionDAG &DAG) const {
  if (verifyReturnAddressArgumentIsConstant(Op, DAG))
    return SDValue();

  // Currently only support lowering return address for current frame.
  if (cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue() != 0) {
    DAG.getContext()->emitError(
        "return address can only be determined for the current frame");
    return SDValue();
  }

  MachineFunction &MF = DAG.getMachineFunction();
  MF.getFrameInfo().setReturnAddressIsTaken(true);
  MVT GRLenVT = Subtarget.getGRLenVT();

  // Return the value of the return address register, marking it an implicit
  // live-in.
  Register Reg = MF.addLiveIn(Subtarget.getRegisterInfo()->getRARegister(),
                              getRegClassFor(GRLenVT));
  return DAG.getCopyFromReg(DAG.getEntryNode(), SDLoc(Op), Reg, GRLenVT);
}

SDValue LoongArchTargetLowering::lowerEH_DWARF_CFA(SDValue Op,
                                                   SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  auto Size = Subtarget.getGRLen() / 8;
  auto FI = MF.getFrameInfo().CreateFixedObject(Size, 0, false);
  return DAG.getFrameIndex(FI, getPointerTy(DAG.getDataLayout()));
}

SDValue LoongArchTargetLowering::lowerVASTART(SDValue Op,
                                              SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  auto *FuncInfo = MF.getInfo<LoongArchMachineFunctionInfo>();

  SDLoc DL(Op);
  SDValue FI = DAG.getFrameIndex(FuncInfo->getVarArgsFrameIndex(),
                                 getPointerTy(MF.getDataLayout()));

  // vastart just stores the address of the VarArgsFrameIndex slot into the
  // memory location argument.
  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
  return DAG.getStore(Op.getOperand(0), DL, FI, Op.getOperand(1),
                      MachinePointerInfo(SV));
}

SDValue LoongArchTargetLowering::lowerUINT_TO_FP(SDValue Op,
                                                 SelectionDAG &DAG) const {
  assert(Subtarget.is64Bit() && Subtarget.hasBasicF() &&
         !Subtarget.hasBasicD() && "unexpected target features");

  SDLoc DL(Op);
  SDValue Op0 = Op.getOperand(0);
  if (Op0->getOpcode() == ISD::AND) {
    auto *C = dyn_cast<ConstantSDNode>(Op0.getOperand(1));
    if (C && C->getZExtValue() < UINT64_C(0xFFFFFFFF))
      return Op;
  }

  if (Op0->getOpcode() == LoongArchISD::BSTRPICK &&
      Op0.getConstantOperandVal(1) < UINT64_C(0X1F) &&
      Op0.getConstantOperandVal(2) == UINT64_C(0))
    return Op;

  if (Op0.getOpcode() == ISD::AssertZext &&
      dyn_cast<VTSDNode>(Op0.getOperand(1))->getVT().bitsLT(MVT::i32))
    return Op;

  EVT OpVT = Op0.getValueType();
  EVT RetVT = Op.getValueType();
  RTLIB::Libcall LC = RTLIB::getUINTTOFP(OpVT, RetVT);
  MakeLibCallOptions CallOptions;
  CallOptions.setTypeListBeforeSoften(OpVT, RetVT, true);
  SDValue Chain = SDValue();
  SDValue Result;
  std::tie(Result, Chain) =
      makeLibCall(DAG, LC, Op.getValueType(), Op0, CallOptions, DL, Chain);
  return Result;
}

SDValue LoongArchTargetLowering::lowerSINT_TO_FP(SDValue Op,
                                                 SelectionDAG &DAG) const {
  assert(Subtarget.is64Bit() && Subtarget.hasBasicF() &&
         !Subtarget.hasBasicD() && "unexpected target features");

  SDLoc DL(Op);
  SDValue Op0 = Op.getOperand(0);

  if ((Op0.getOpcode() == ISD::AssertSext ||
       Op0.getOpcode() == ISD::SIGN_EXTEND_INREG) &&
      dyn_cast<VTSDNode>(Op0.getOperand(1))->getVT().bitsLE(MVT::i32))
    return Op;

  EVT OpVT = Op0.getValueType();
  EVT RetVT = Op.getValueType();
  RTLIB::Libcall LC = RTLIB::getSINTTOFP(OpVT, RetVT);
  MakeLibCallOptions CallOptions;
  CallOptions.setTypeListBeforeSoften(OpVT, RetVT, true);
  SDValue Chain = SDValue();
  SDValue Result;
  std::tie(Result, Chain) =
      makeLibCall(DAG, LC, Op.getValueType(), Op0, CallOptions, DL, Chain);
  return Result;
}

SDValue LoongArchTargetLowering::lowerBITCAST(SDValue Op,
                                              SelectionDAG &DAG) const {

  SDLoc DL(Op);
  SDValue Op0 = Op.getOperand(0);

  if (Op.getValueType() == MVT::f32 && Op0.getValueType() == MVT::i32 &&
      Subtarget.is64Bit() && Subtarget.hasBasicF()) {
    SDValue NewOp0 = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, Op0);
    return DAG.getNode(LoongArchISD::MOVGR2FR_W_LA64, DL, MVT::f32, NewOp0);
  }
  return Op;
}

SDValue LoongArchTargetLowering::lowerFP_TO_SINT(SDValue Op,
                                                 SelectionDAG &DAG) const {

  SDLoc DL(Op);

  if (Op.getValueSizeInBits() > 32 && Subtarget.hasBasicF() &&
      !Subtarget.hasBasicD()) {
    SDValue Dst =
        DAG.getNode(LoongArchISD::FTINT, DL, MVT::f32, Op.getOperand(0));
    return DAG.getNode(LoongArchISD::MOVFR2GR_S_LA64, DL, MVT::i64, Dst);
  }

  EVT FPTy = EVT::getFloatingPointVT(Op.getValueSizeInBits());
  SDValue Trunc = DAG.getNode(LoongArchISD::FTINT, DL, FPTy, Op.getOperand(0));
  return DAG.getNode(ISD::BITCAST, DL, Op.getValueType(), Trunc);
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

static SDValue getTargetNode(JumpTableSDNode *N, SDLoc DL, EVT Ty,
                             SelectionDAG &DAG, unsigned Flags) {
  return DAG.getTargetJumpTable(N->getIndex(), Ty, Flags);
}

template <class NodeTy>
SDValue LoongArchTargetLowering::getAddr(NodeTy *N, SelectionDAG &DAG,
                                         bool IsLocal) const {
  SDLoc DL(N);
  EVT Ty = getPointerTy(DAG.getDataLayout());
  SDValue Addr = getTargetNode(N, DL, Ty, DAG, 0);

  switch (DAG.getTarget().getCodeModel()) {
  default:
    report_fatal_error("Unsupported code model");

  case CodeModel::Large: {
    assert(Subtarget.is64Bit() && "Large code model requires LA64");

    // This is not actually used, but is necessary for successfully matching
    // the PseudoLA_*_LARGE nodes.
    SDValue Tmp = DAG.getConstant(0, DL, Ty);
    if (IsLocal)
      // This generates the pattern (PseudoLA_PCREL_LARGE tmp sym), that
      // eventually becomes the desired 5-insn code sequence.
      return SDValue(DAG.getMachineNode(LoongArch::PseudoLA_PCREL_LARGE, DL, Ty,
                                        Tmp, Addr),
                     0);

    // This generates the pattern (PseudoLA_GOT_LARGE tmp sym), that eventually
    // becomes the desired 5-insn code sequence.
    return SDValue(
        DAG.getMachineNode(LoongArch::PseudoLA_GOT_LARGE, DL, Ty, Tmp, Addr),
        0);
  }

  case CodeModel::Small:
  case CodeModel::Medium:
    if (IsLocal)
      // This generates the pattern (PseudoLA_PCREL sym), which expands to
      // (addi.w/d (pcalau12i %pc_hi20(sym)) %pc_lo12(sym)).
      return SDValue(
          DAG.getMachineNode(LoongArch::PseudoLA_PCREL, DL, Ty, Addr), 0);

    // This generates the pattern (PseudoLA_GOT sym), which expands to (ld.w/d
    // (pcalau12i %got_pc_hi20(sym)) %got_pc_lo12(sym)).
    return SDValue(DAG.getMachineNode(LoongArch::PseudoLA_GOT, DL, Ty, Addr),
                   0);
  }
}

SDValue LoongArchTargetLowering::lowerBlockAddress(SDValue Op,
                                                   SelectionDAG &DAG) const {
  return getAddr(cast<BlockAddressSDNode>(Op), DAG);
}

SDValue LoongArchTargetLowering::lowerJumpTable(SDValue Op,
                                                SelectionDAG &DAG) const {
  return getAddr(cast<JumpTableSDNode>(Op), DAG);
}

SDValue LoongArchTargetLowering::lowerConstantPool(SDValue Op,
                                                   SelectionDAG &DAG) const {
  return getAddr(cast<ConstantPoolSDNode>(Op), DAG);
}

SDValue LoongArchTargetLowering::lowerGlobalAddress(SDValue Op,
                                                    SelectionDAG &DAG) const {
  GlobalAddressSDNode *N = cast<GlobalAddressSDNode>(Op);
  assert(N->getOffset() == 0 && "unexpected offset in global node");
  return getAddr(N, DAG, N->getGlobal()->isDSOLocal());
}

SDValue LoongArchTargetLowering::getStaticTLSAddr(GlobalAddressSDNode *N,
                                                  SelectionDAG &DAG,
                                                  unsigned Opc,
                                                  bool Large) const {
  SDLoc DL(N);
  EVT Ty = getPointerTy(DAG.getDataLayout());
  MVT GRLenVT = Subtarget.getGRLenVT();

  // This is not actually used, but is necessary for successfully matching the
  // PseudoLA_*_LARGE nodes.
  SDValue Tmp = DAG.getConstant(0, DL, Ty);
  SDValue Addr = DAG.getTargetGlobalAddress(N->getGlobal(), DL, Ty, 0, 0);
  SDValue Offset = Large
                       ? SDValue(DAG.getMachineNode(Opc, DL, Ty, Tmp, Addr), 0)
                       : SDValue(DAG.getMachineNode(Opc, DL, Ty, Addr), 0);

  // Add the thread pointer.
  return DAG.getNode(ISD::ADD, DL, Ty, Offset,
                     DAG.getRegister(LoongArch::R2, GRLenVT));
}

SDValue LoongArchTargetLowering::getDynamicTLSAddr(GlobalAddressSDNode *N,
                                                   SelectionDAG &DAG,
                                                   unsigned Opc,
                                                   bool Large) const {
  SDLoc DL(N);
  EVT Ty = getPointerTy(DAG.getDataLayout());
  IntegerType *CallTy = Type::getIntNTy(*DAG.getContext(), Ty.getSizeInBits());

  // This is not actually used, but is necessary for successfully matching the
  // PseudoLA_*_LARGE nodes.
  SDValue Tmp = DAG.getConstant(0, DL, Ty);

  // Use a PC-relative addressing mode to access the dynamic GOT address.
  SDValue Addr = DAG.getTargetGlobalAddress(N->getGlobal(), DL, Ty, 0, 0);
  SDValue Load = Large ? SDValue(DAG.getMachineNode(Opc, DL, Ty, Tmp, Addr), 0)
                       : SDValue(DAG.getMachineNode(Opc, DL, Ty, Addr), 0);

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

SDValue
LoongArchTargetLowering::lowerGlobalTLSAddress(SDValue Op,
                                               SelectionDAG &DAG) const {
  if (DAG.getMachineFunction().getFunction().getCallingConv() ==
      CallingConv::GHC)
    report_fatal_error("In GHC calling convention TLS is not supported");

  bool Large = DAG.getTarget().getCodeModel() == CodeModel::Large;
  assert((!Large || Subtarget.is64Bit()) && "Large code model requires LA64");

  GlobalAddressSDNode *N = cast<GlobalAddressSDNode>(Op);
  assert(N->getOffset() == 0 && "unexpected offset in global node");

  SDValue Addr;
  switch (getTargetMachine().getTLSModel(N->getGlobal())) {
  case TLSModel::GeneralDynamic:
    // In this model, application code calls the dynamic linker function
    // __tls_get_addr to locate TLS offsets into the dynamic thread vector at
    // runtime.
    Addr = getDynamicTLSAddr(N, DAG,
                             Large ? LoongArch::PseudoLA_TLS_GD_LARGE
                                   : LoongArch::PseudoLA_TLS_GD,
                             Large);
    break;
  case TLSModel::LocalDynamic:
    // Same as GeneralDynamic, except for assembly modifiers and relocation
    // records.
    Addr = getDynamicTLSAddr(N, DAG,
                             Large ? LoongArch::PseudoLA_TLS_LD_LARGE
                                   : LoongArch::PseudoLA_TLS_LD,
                             Large);
    break;
  case TLSModel::InitialExec:
    // This model uses the GOT to resolve TLS offsets.
    Addr = getStaticTLSAddr(N, DAG,
                            Large ? LoongArch::PseudoLA_TLS_IE_LARGE
                                  : LoongArch::PseudoLA_TLS_IE,
                            Large);
    break;
  case TLSModel::LocalExec:
    // This model is used when static linking as the TLS offsets are resolved
    // during program linking.
    //
    // This node doesn't need an extra argument for the large code model.
    Addr = getStaticTLSAddr(N, DAG, LoongArch::PseudoLA_TLS_LE);
    break;
  }

  return Addr;
}

template <unsigned N>
static SDValue checkIntrinsicImmArg(SDValue Op, unsigned ImmOp,
                                    SelectionDAG &DAG, bool IsSigned = false) {
  auto *CImm = cast<ConstantSDNode>(Op->getOperand(ImmOp));
  // Check the ImmArg.
  if ((IsSigned && !isInt<N>(CImm->getSExtValue())) ||
      (!IsSigned && !isUInt<N>(CImm->getZExtValue()))) {
    DAG.getContext()->emitError(Op->getOperationName(0) +
                                ": argument out of range.");
    return DAG.getNode(ISD::UNDEF, SDLoc(Op), Op.getValueType());
  }
  return SDValue();
}

SDValue
LoongArchTargetLowering::lowerINTRINSIC_WO_CHAIN(SDValue Op,
                                                 SelectionDAG &DAG) const {
  SDLoc DL(Op);
  switch (Op.getConstantOperandVal(0)) {
  default:
    return SDValue(); // Don't custom lower most intrinsics.
  case Intrinsic::thread_pointer: {
    EVT PtrVT = getPointerTy(DAG.getDataLayout());
    return DAG.getRegister(LoongArch::R2, PtrVT);
  }
  case Intrinsic::loongarch_lsx_vpickve2gr_d:
  case Intrinsic::loongarch_lsx_vpickve2gr_du:
  case Intrinsic::loongarch_lsx_vreplvei_d:
  case Intrinsic::loongarch_lasx_xvrepl128vei_d:
    return checkIntrinsicImmArg<1>(Op, 2, DAG);
  case Intrinsic::loongarch_lsx_vreplvei_w:
  case Intrinsic::loongarch_lasx_xvrepl128vei_w:
  case Intrinsic::loongarch_lasx_xvpickve2gr_d:
  case Intrinsic::loongarch_lasx_xvpickve2gr_du:
  case Intrinsic::loongarch_lasx_xvpickve_d:
  case Intrinsic::loongarch_lasx_xvpickve_d_f:
    return checkIntrinsicImmArg<2>(Op, 2, DAG);
  case Intrinsic::loongarch_lasx_xvinsve0_d:
    return checkIntrinsicImmArg<2>(Op, 3, DAG);
  case Intrinsic::loongarch_lsx_vsat_b:
  case Intrinsic::loongarch_lsx_vsat_bu:
  case Intrinsic::loongarch_lsx_vrotri_b:
  case Intrinsic::loongarch_lsx_vsllwil_h_b:
  case Intrinsic::loongarch_lsx_vsllwil_hu_bu:
  case Intrinsic::loongarch_lsx_vsrlri_b:
  case Intrinsic::loongarch_lsx_vsrari_b:
  case Intrinsic::loongarch_lsx_vreplvei_h:
  case Intrinsic::loongarch_lasx_xvsat_b:
  case Intrinsic::loongarch_lasx_xvsat_bu:
  case Intrinsic::loongarch_lasx_xvrotri_b:
  case Intrinsic::loongarch_lasx_xvsllwil_h_b:
  case Intrinsic::loongarch_lasx_xvsllwil_hu_bu:
  case Intrinsic::loongarch_lasx_xvsrlri_b:
  case Intrinsic::loongarch_lasx_xvsrari_b:
  case Intrinsic::loongarch_lasx_xvrepl128vei_h:
  case Intrinsic::loongarch_lasx_xvpickve_w:
  case Intrinsic::loongarch_lasx_xvpickve_w_f:
    return checkIntrinsicImmArg<3>(Op, 2, DAG);
  case Intrinsic::loongarch_lasx_xvinsve0_w:
    return checkIntrinsicImmArg<3>(Op, 3, DAG);
  case Intrinsic::loongarch_lsx_vsat_h:
  case Intrinsic::loongarch_lsx_vsat_hu:
  case Intrinsic::loongarch_lsx_vrotri_h:
  case Intrinsic::loongarch_lsx_vsllwil_w_h:
  case Intrinsic::loongarch_lsx_vsllwil_wu_hu:
  case Intrinsic::loongarch_lsx_vsrlri_h:
  case Intrinsic::loongarch_lsx_vsrari_h:
  case Intrinsic::loongarch_lsx_vreplvei_b:
  case Intrinsic::loongarch_lasx_xvsat_h:
  case Intrinsic::loongarch_lasx_xvsat_hu:
  case Intrinsic::loongarch_lasx_xvrotri_h:
  case Intrinsic::loongarch_lasx_xvsllwil_w_h:
  case Intrinsic::loongarch_lasx_xvsllwil_wu_hu:
  case Intrinsic::loongarch_lasx_xvsrlri_h:
  case Intrinsic::loongarch_lasx_xvsrari_h:
  case Intrinsic::loongarch_lasx_xvrepl128vei_b:
    return checkIntrinsicImmArg<4>(Op, 2, DAG);
  case Intrinsic::loongarch_lsx_vsrlni_b_h:
  case Intrinsic::loongarch_lsx_vsrani_b_h:
  case Intrinsic::loongarch_lsx_vsrlrni_b_h:
  case Intrinsic::loongarch_lsx_vsrarni_b_h:
  case Intrinsic::loongarch_lsx_vssrlni_b_h:
  case Intrinsic::loongarch_lsx_vssrani_b_h:
  case Intrinsic::loongarch_lsx_vssrlni_bu_h:
  case Intrinsic::loongarch_lsx_vssrani_bu_h:
  case Intrinsic::loongarch_lsx_vssrlrni_b_h:
  case Intrinsic::loongarch_lsx_vssrarni_b_h:
  case Intrinsic::loongarch_lsx_vssrlrni_bu_h:
  case Intrinsic::loongarch_lsx_vssrarni_bu_h:
  case Intrinsic::loongarch_lasx_xvsrlni_b_h:
  case Intrinsic::loongarch_lasx_xvsrani_b_h:
  case Intrinsic::loongarch_lasx_xvsrlrni_b_h:
  case Intrinsic::loongarch_lasx_xvsrarni_b_h:
  case Intrinsic::loongarch_lasx_xvssrlni_b_h:
  case Intrinsic::loongarch_lasx_xvssrani_b_h:
  case Intrinsic::loongarch_lasx_xvssrlni_bu_h:
  case Intrinsic::loongarch_lasx_xvssrani_bu_h:
  case Intrinsic::loongarch_lasx_xvssrlrni_b_h:
  case Intrinsic::loongarch_lasx_xvssrarni_b_h:
  case Intrinsic::loongarch_lasx_xvssrlrni_bu_h:
  case Intrinsic::loongarch_lasx_xvssrarni_bu_h:
    return checkIntrinsicImmArg<4>(Op, 3, DAG);
  case Intrinsic::loongarch_lsx_vsat_w:
  case Intrinsic::loongarch_lsx_vsat_wu:
  case Intrinsic::loongarch_lsx_vrotri_w:
  case Intrinsic::loongarch_lsx_vsllwil_d_w:
  case Intrinsic::loongarch_lsx_vsllwil_du_wu:
  case Intrinsic::loongarch_lsx_vsrlri_w:
  case Intrinsic::loongarch_lsx_vsrari_w:
  case Intrinsic::loongarch_lsx_vslei_bu:
  case Intrinsic::loongarch_lsx_vslei_hu:
  case Intrinsic::loongarch_lsx_vslei_wu:
  case Intrinsic::loongarch_lsx_vslei_du:
  case Intrinsic::loongarch_lsx_vslti_bu:
  case Intrinsic::loongarch_lsx_vslti_hu:
  case Intrinsic::loongarch_lsx_vslti_wu:
  case Intrinsic::loongarch_lsx_vslti_du:
  case Intrinsic::loongarch_lsx_vbsll_v:
  case Intrinsic::loongarch_lsx_vbsrl_v:
  case Intrinsic::loongarch_lasx_xvsat_w:
  case Intrinsic::loongarch_lasx_xvsat_wu:
  case Intrinsic::loongarch_lasx_xvrotri_w:
  case Intrinsic::loongarch_lasx_xvsllwil_d_w:
  case Intrinsic::loongarch_lasx_xvsllwil_du_wu:
  case Intrinsic::loongarch_lasx_xvsrlri_w:
  case Intrinsic::loongarch_lasx_xvsrari_w:
  case Intrinsic::loongarch_lasx_xvslei_bu:
  case Intrinsic::loongarch_lasx_xvslei_hu:
  case Intrinsic::loongarch_lasx_xvslei_wu:
  case Intrinsic::loongarch_lasx_xvslei_du:
  case Intrinsic::loongarch_lasx_xvslti_bu:
  case Intrinsic::loongarch_lasx_xvslti_hu:
  case Intrinsic::loongarch_lasx_xvslti_wu:
  case Intrinsic::loongarch_lasx_xvslti_du:
  case Intrinsic::loongarch_lasx_xvbsll_v:
  case Intrinsic::loongarch_lasx_xvbsrl_v:
    return checkIntrinsicImmArg<5>(Op, 2, DAG);
  case Intrinsic::loongarch_lsx_vseqi_b:
  case Intrinsic::loongarch_lsx_vseqi_h:
  case Intrinsic::loongarch_lsx_vseqi_w:
  case Intrinsic::loongarch_lsx_vseqi_d:
  case Intrinsic::loongarch_lsx_vslei_b:
  case Intrinsic::loongarch_lsx_vslei_h:
  case Intrinsic::loongarch_lsx_vslei_w:
  case Intrinsic::loongarch_lsx_vslei_d:
  case Intrinsic::loongarch_lsx_vslti_b:
  case Intrinsic::loongarch_lsx_vslti_h:
  case Intrinsic::loongarch_lsx_vslti_w:
  case Intrinsic::loongarch_lsx_vslti_d:
  case Intrinsic::loongarch_lasx_xvseqi_b:
  case Intrinsic::loongarch_lasx_xvseqi_h:
  case Intrinsic::loongarch_lasx_xvseqi_w:
  case Intrinsic::loongarch_lasx_xvseqi_d:
  case Intrinsic::loongarch_lasx_xvslei_b:
  case Intrinsic::loongarch_lasx_xvslei_h:
  case Intrinsic::loongarch_lasx_xvslei_w:
  case Intrinsic::loongarch_lasx_xvslei_d:
  case Intrinsic::loongarch_lasx_xvslti_b:
  case Intrinsic::loongarch_lasx_xvslti_h:
  case Intrinsic::loongarch_lasx_xvslti_w:
  case Intrinsic::loongarch_lasx_xvslti_d:
    return checkIntrinsicImmArg<5>(Op, 2, DAG, /*IsSigned=*/true);
  case Intrinsic::loongarch_lsx_vsrlni_h_w:
  case Intrinsic::loongarch_lsx_vsrani_h_w:
  case Intrinsic::loongarch_lsx_vsrlrni_h_w:
  case Intrinsic::loongarch_lsx_vsrarni_h_w:
  case Intrinsic::loongarch_lsx_vssrlni_h_w:
  case Intrinsic::loongarch_lsx_vssrani_h_w:
  case Intrinsic::loongarch_lsx_vssrlni_hu_w:
  case Intrinsic::loongarch_lsx_vssrani_hu_w:
  case Intrinsic::loongarch_lsx_vssrlrni_h_w:
  case Intrinsic::loongarch_lsx_vssrarni_h_w:
  case Intrinsic::loongarch_lsx_vssrlrni_hu_w:
  case Intrinsic::loongarch_lsx_vssrarni_hu_w:
  case Intrinsic::loongarch_lsx_vfrstpi_b:
  case Intrinsic::loongarch_lsx_vfrstpi_h:
  case Intrinsic::loongarch_lasx_xvsrlni_h_w:
  case Intrinsic::loongarch_lasx_xvsrani_h_w:
  case Intrinsic::loongarch_lasx_xvsrlrni_h_w:
  case Intrinsic::loongarch_lasx_xvsrarni_h_w:
  case Intrinsic::loongarch_lasx_xvssrlni_h_w:
  case Intrinsic::loongarch_lasx_xvssrani_h_w:
  case Intrinsic::loongarch_lasx_xvssrlni_hu_w:
  case Intrinsic::loongarch_lasx_xvssrani_hu_w:
  case Intrinsic::loongarch_lasx_xvssrlrni_h_w:
  case Intrinsic::loongarch_lasx_xvssrarni_h_w:
  case Intrinsic::loongarch_lasx_xvssrlrni_hu_w:
  case Intrinsic::loongarch_lasx_xvssrarni_hu_w:
  case Intrinsic::loongarch_lasx_xvfrstpi_b:
  case Intrinsic::loongarch_lasx_xvfrstpi_h:
    return checkIntrinsicImmArg<5>(Op, 3, DAG);
  case Intrinsic::loongarch_lsx_vsat_d:
  case Intrinsic::loongarch_lsx_vsat_du:
  case Intrinsic::loongarch_lsx_vrotri_d:
  case Intrinsic::loongarch_lsx_vsrlri_d:
  case Intrinsic::loongarch_lsx_vsrari_d:
  case Intrinsic::loongarch_lasx_xvsat_d:
  case Intrinsic::loongarch_lasx_xvsat_du:
  case Intrinsic::loongarch_lasx_xvrotri_d:
  case Intrinsic::loongarch_lasx_xvsrlri_d:
  case Intrinsic::loongarch_lasx_xvsrari_d:
    return checkIntrinsicImmArg<6>(Op, 2, DAG);
  case Intrinsic::loongarch_lsx_vsrlni_w_d:
  case Intrinsic::loongarch_lsx_vsrani_w_d:
  case Intrinsic::loongarch_lsx_vsrlrni_w_d:
  case Intrinsic::loongarch_lsx_vsrarni_w_d:
  case Intrinsic::loongarch_lsx_vssrlni_w_d:
  case Intrinsic::loongarch_lsx_vssrani_w_d:
  case Intrinsic::loongarch_lsx_vssrlni_wu_d:
  case Intrinsic::loongarch_lsx_vssrani_wu_d:
  case Intrinsic::loongarch_lsx_vssrlrni_w_d:
  case Intrinsic::loongarch_lsx_vssrarni_w_d:
  case Intrinsic::loongarch_lsx_vssrlrni_wu_d:
  case Intrinsic::loongarch_lsx_vssrarni_wu_d:
  case Intrinsic::loongarch_lasx_xvsrlni_w_d:
  case Intrinsic::loongarch_lasx_xvsrani_w_d:
  case Intrinsic::loongarch_lasx_xvsrlrni_w_d:
  case Intrinsic::loongarch_lasx_xvsrarni_w_d:
  case Intrinsic::loongarch_lasx_xvssrlni_w_d:
  case Intrinsic::loongarch_lasx_xvssrani_w_d:
  case Intrinsic::loongarch_lasx_xvssrlni_wu_d:
  case Intrinsic::loongarch_lasx_xvssrani_wu_d:
  case Intrinsic::loongarch_lasx_xvssrlrni_w_d:
  case Intrinsic::loongarch_lasx_xvssrarni_w_d:
  case Intrinsic::loongarch_lasx_xvssrlrni_wu_d:
  case Intrinsic::loongarch_lasx_xvssrarni_wu_d:
    return checkIntrinsicImmArg<6>(Op, 3, DAG);
  case Intrinsic::loongarch_lsx_vsrlni_d_q:
  case Intrinsic::loongarch_lsx_vsrani_d_q:
  case Intrinsic::loongarch_lsx_vsrlrni_d_q:
  case Intrinsic::loongarch_lsx_vsrarni_d_q:
  case Intrinsic::loongarch_lsx_vssrlni_d_q:
  case Intrinsic::loongarch_lsx_vssrani_d_q:
  case Intrinsic::loongarch_lsx_vssrlni_du_q:
  case Intrinsic::loongarch_lsx_vssrani_du_q:
  case Intrinsic::loongarch_lsx_vssrlrni_d_q:
  case Intrinsic::loongarch_lsx_vssrarni_d_q:
  case Intrinsic::loongarch_lsx_vssrlrni_du_q:
  case Intrinsic::loongarch_lsx_vssrarni_du_q:
  case Intrinsic::loongarch_lasx_xvsrlni_d_q:
  case Intrinsic::loongarch_lasx_xvsrani_d_q:
  case Intrinsic::loongarch_lasx_xvsrlrni_d_q:
  case Intrinsic::loongarch_lasx_xvsrarni_d_q:
  case Intrinsic::loongarch_lasx_xvssrlni_d_q:
  case Intrinsic::loongarch_lasx_xvssrani_d_q:
  case Intrinsic::loongarch_lasx_xvssrlni_du_q:
  case Intrinsic::loongarch_lasx_xvssrani_du_q:
  case Intrinsic::loongarch_lasx_xvssrlrni_d_q:
  case Intrinsic::loongarch_lasx_xvssrarni_d_q:
  case Intrinsic::loongarch_lasx_xvssrlrni_du_q:
  case Intrinsic::loongarch_lasx_xvssrarni_du_q:
    return checkIntrinsicImmArg<7>(Op, 3, DAG);
  case Intrinsic::loongarch_lsx_vnori_b:
  case Intrinsic::loongarch_lsx_vshuf4i_b:
  case Intrinsic::loongarch_lsx_vshuf4i_h:
  case Intrinsic::loongarch_lsx_vshuf4i_w:
  case Intrinsic::loongarch_lasx_xvnori_b:
  case Intrinsic::loongarch_lasx_xvshuf4i_b:
  case Intrinsic::loongarch_lasx_xvshuf4i_h:
  case Intrinsic::loongarch_lasx_xvshuf4i_w:
  case Intrinsic::loongarch_lasx_xvpermi_d:
    return checkIntrinsicImmArg<8>(Op, 2, DAG);
  case Intrinsic::loongarch_lsx_vshuf4i_d:
  case Intrinsic::loongarch_lsx_vpermi_w:
  case Intrinsic::loongarch_lsx_vbitseli_b:
  case Intrinsic::loongarch_lsx_vextrins_b:
  case Intrinsic::loongarch_lsx_vextrins_h:
  case Intrinsic::loongarch_lsx_vextrins_w:
  case Intrinsic::loongarch_lsx_vextrins_d:
  case Intrinsic::loongarch_lasx_xvshuf4i_d:
  case Intrinsic::loongarch_lasx_xvpermi_w:
  case Intrinsic::loongarch_lasx_xvpermi_q:
  case Intrinsic::loongarch_lasx_xvbitseli_b:
  case Intrinsic::loongarch_lasx_xvextrins_b:
  case Intrinsic::loongarch_lasx_xvextrins_h:
  case Intrinsic::loongarch_lasx_xvextrins_w:
  case Intrinsic::loongarch_lasx_xvextrins_d:
    return checkIntrinsicImmArg<8>(Op, 3, DAG);
  case Intrinsic::loongarch_lsx_vrepli_b:
  case Intrinsic::loongarch_lsx_vrepli_h:
  case Intrinsic::loongarch_lsx_vrepli_w:
  case Intrinsic::loongarch_lsx_vrepli_d:
  case Intrinsic::loongarch_lasx_xvrepli_b:
  case Intrinsic::loongarch_lasx_xvrepli_h:
  case Intrinsic::loongarch_lasx_xvrepli_w:
  case Intrinsic::loongarch_lasx_xvrepli_d:
    return checkIntrinsicImmArg<10>(Op, 1, DAG, /*IsSigned=*/true);
  case Intrinsic::loongarch_lsx_vldi:
  case Intrinsic::loongarch_lasx_xvldi:
    return checkIntrinsicImmArg<13>(Op, 1, DAG, /*IsSigned=*/true);
  }
}

// Helper function that emits error message for intrinsics with chain and return
// merge values of a UNDEF and the chain.
static SDValue emitIntrinsicWithChainErrorMessage(SDValue Op,
                                                  StringRef ErrorMsg,
                                                  SelectionDAG &DAG) {
  DAG.getContext()->emitError(Op->getOperationName(0) + ": " + ErrorMsg + ".");
  return DAG.getMergeValues({DAG.getUNDEF(Op.getValueType()), Op.getOperand(0)},
                            SDLoc(Op));
}

SDValue
LoongArchTargetLowering::lowerINTRINSIC_W_CHAIN(SDValue Op,
                                                SelectionDAG &DAG) const {
  SDLoc DL(Op);
  MVT GRLenVT = Subtarget.getGRLenVT();
  EVT VT = Op.getValueType();
  SDValue Chain = Op.getOperand(0);
  const StringRef ErrorMsgOOR = "argument out of range";
  const StringRef ErrorMsgReqLA64 = "requires loongarch64";
  const StringRef ErrorMsgReqF = "requires basic 'f' target feature";

  switch (Op.getConstantOperandVal(1)) {
  default:
    return Op;
  case Intrinsic::loongarch_crc_w_b_w:
  case Intrinsic::loongarch_crc_w_h_w:
  case Intrinsic::loongarch_crc_w_w_w:
  case Intrinsic::loongarch_crc_w_d_w:
  case Intrinsic::loongarch_crcc_w_b_w:
  case Intrinsic::loongarch_crcc_w_h_w:
  case Intrinsic::loongarch_crcc_w_w_w:
  case Intrinsic::loongarch_crcc_w_d_w:
    return emitIntrinsicWithChainErrorMessage(Op, ErrorMsgReqLA64, DAG);
  case Intrinsic::loongarch_csrrd_w:
  case Intrinsic::loongarch_csrrd_d: {
    unsigned Imm = cast<ConstantSDNode>(Op.getOperand(2))->getZExtValue();
    return !isUInt<14>(Imm)
               ? emitIntrinsicWithChainErrorMessage(Op, ErrorMsgOOR, DAG)
               : DAG.getNode(LoongArchISD::CSRRD, DL, {GRLenVT, MVT::Other},
                             {Chain, DAG.getConstant(Imm, DL, GRLenVT)});
  }
  case Intrinsic::loongarch_csrwr_w:
  case Intrinsic::loongarch_csrwr_d: {
    unsigned Imm = cast<ConstantSDNode>(Op.getOperand(3))->getZExtValue();
    return !isUInt<14>(Imm)
               ? emitIntrinsicWithChainErrorMessage(Op, ErrorMsgOOR, DAG)
               : DAG.getNode(LoongArchISD::CSRWR, DL, {GRLenVT, MVT::Other},
                             {Chain, Op.getOperand(2),
                              DAG.getConstant(Imm, DL, GRLenVT)});
  }
  case Intrinsic::loongarch_csrxchg_w:
  case Intrinsic::loongarch_csrxchg_d: {
    unsigned Imm = cast<ConstantSDNode>(Op.getOperand(4))->getZExtValue();
    return !isUInt<14>(Imm)
               ? emitIntrinsicWithChainErrorMessage(Op, ErrorMsgOOR, DAG)
               : DAG.getNode(LoongArchISD::CSRXCHG, DL, {GRLenVT, MVT::Other},
                             {Chain, Op.getOperand(2), Op.getOperand(3),
                              DAG.getConstant(Imm, DL, GRLenVT)});
  }
  case Intrinsic::loongarch_iocsrrd_d: {
    return DAG.getNode(
        LoongArchISD::IOCSRRD_D, DL, {GRLenVT, MVT::Other},
        {Chain, DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, Op.getOperand(2))});
  }
#define IOCSRRD_CASE(NAME, NODE)                                               \
  case Intrinsic::loongarch_##NAME: {                                          \
    return DAG.getNode(LoongArchISD::NODE, DL, {GRLenVT, MVT::Other},          \
                       {Chain, Op.getOperand(2)});                             \
  }
    IOCSRRD_CASE(iocsrrd_b, IOCSRRD_B);
    IOCSRRD_CASE(iocsrrd_h, IOCSRRD_H);
    IOCSRRD_CASE(iocsrrd_w, IOCSRRD_W);
#undef IOCSRRD_CASE
  case Intrinsic::loongarch_cpucfg: {
    return DAG.getNode(LoongArchISD::CPUCFG, DL, {GRLenVT, MVT::Other},
                       {Chain, Op.getOperand(2)});
  }
  case Intrinsic::loongarch_lddir_d: {
    unsigned Imm = cast<ConstantSDNode>(Op.getOperand(3))->getZExtValue();
    return !isUInt<8>(Imm)
               ? emitIntrinsicWithChainErrorMessage(Op, ErrorMsgOOR, DAG)
               : Op;
  }
  case Intrinsic::loongarch_movfcsr2gr: {
    if (!Subtarget.hasBasicF())
      return emitIntrinsicWithChainErrorMessage(Op, ErrorMsgReqF, DAG);
    unsigned Imm = cast<ConstantSDNode>(Op.getOperand(2))->getZExtValue();
    return !isUInt<2>(Imm)
               ? emitIntrinsicWithChainErrorMessage(Op, ErrorMsgOOR, DAG)
               : DAG.getNode(LoongArchISD::MOVFCSR2GR, DL, {VT, MVT::Other},
                             {Chain, DAG.getConstant(Imm, DL, GRLenVT)});
  }
  case Intrinsic::loongarch_lsx_vld:
  case Intrinsic::loongarch_lsx_vldrepl_b:
  case Intrinsic::loongarch_lasx_xvld:
  case Intrinsic::loongarch_lasx_xvldrepl_b:
    return !isInt<12>(cast<ConstantSDNode>(Op.getOperand(3))->getSExtValue())
               ? emitIntrinsicWithChainErrorMessage(Op, ErrorMsgOOR, DAG)
               : SDValue();
  case Intrinsic::loongarch_lsx_vldrepl_h:
  case Intrinsic::loongarch_lasx_xvldrepl_h:
    return !isShiftedInt<11, 1>(
               cast<ConstantSDNode>(Op.getOperand(3))->getSExtValue())
               ? emitIntrinsicWithChainErrorMessage(
                     Op, "argument out of range or not a multiple of 2", DAG)
               : SDValue();
  case Intrinsic::loongarch_lsx_vldrepl_w:
  case Intrinsic::loongarch_lasx_xvldrepl_w:
    return !isShiftedInt<10, 2>(
               cast<ConstantSDNode>(Op.getOperand(3))->getSExtValue())
               ? emitIntrinsicWithChainErrorMessage(
                     Op, "argument out of range or not a multiple of 4", DAG)
               : SDValue();
  case Intrinsic::loongarch_lsx_vldrepl_d:
  case Intrinsic::loongarch_lasx_xvldrepl_d:
    return !isShiftedInt<9, 3>(
               cast<ConstantSDNode>(Op.getOperand(3))->getSExtValue())
               ? emitIntrinsicWithChainErrorMessage(
                     Op, "argument out of range or not a multiple of 8", DAG)
               : SDValue();
  }
}

// Helper function that emits error message for intrinsics with void return
// value and return the chain.
static SDValue emitIntrinsicErrorMessage(SDValue Op, StringRef ErrorMsg,
                                         SelectionDAG &DAG) {

  DAG.getContext()->emitError(Op->getOperationName(0) + ": " + ErrorMsg + ".");
  return Op.getOperand(0);
}

SDValue LoongArchTargetLowering::lowerINTRINSIC_VOID(SDValue Op,
                                                     SelectionDAG &DAG) const {
  SDLoc DL(Op);
  MVT GRLenVT = Subtarget.getGRLenVT();
  SDValue Chain = Op.getOperand(0);
  uint64_t IntrinsicEnum = Op.getConstantOperandVal(1);
  SDValue Op2 = Op.getOperand(2);
  const StringRef ErrorMsgOOR = "argument out of range";
  const StringRef ErrorMsgReqLA64 = "requires loongarch64";
  const StringRef ErrorMsgReqLA32 = "requires loongarch32";
  const StringRef ErrorMsgReqF = "requires basic 'f' target feature";

  switch (IntrinsicEnum) {
  default:
    // TODO: Add more Intrinsics.
    return SDValue();
  case Intrinsic::loongarch_cacop_d:
  case Intrinsic::loongarch_cacop_w: {
    if (IntrinsicEnum == Intrinsic::loongarch_cacop_d && !Subtarget.is64Bit())
      return emitIntrinsicErrorMessage(Op, ErrorMsgReqLA64, DAG);
    if (IntrinsicEnum == Intrinsic::loongarch_cacop_w && Subtarget.is64Bit())
      return emitIntrinsicErrorMessage(Op, ErrorMsgReqLA32, DAG);
    // call void @llvm.loongarch.cacop.[d/w](uimm5, rj, simm12)
    unsigned Imm1 = cast<ConstantSDNode>(Op2)->getZExtValue();
    int Imm2 = cast<ConstantSDNode>(Op.getOperand(4))->getSExtValue();
    if (!isUInt<5>(Imm1) || !isInt<12>(Imm2))
      return emitIntrinsicErrorMessage(Op, ErrorMsgOOR, DAG);
    return Op;
  }
  case Intrinsic::loongarch_dbar: {
    unsigned Imm = cast<ConstantSDNode>(Op2)->getZExtValue();
    return !isUInt<15>(Imm)
               ? emitIntrinsicErrorMessage(Op, ErrorMsgOOR, DAG)
               : DAG.getNode(LoongArchISD::DBAR, DL, MVT::Other, Chain,
                             DAG.getConstant(Imm, DL, GRLenVT));
  }
  case Intrinsic::loongarch_ibar: {
    unsigned Imm = cast<ConstantSDNode>(Op2)->getZExtValue();
    return !isUInt<15>(Imm)
               ? emitIntrinsicErrorMessage(Op, ErrorMsgOOR, DAG)
               : DAG.getNode(LoongArchISD::IBAR, DL, MVT::Other, Chain,
                             DAG.getConstant(Imm, DL, GRLenVT));
  }
  case Intrinsic::loongarch_break: {
    unsigned Imm = cast<ConstantSDNode>(Op2)->getZExtValue();
    return !isUInt<15>(Imm)
               ? emitIntrinsicErrorMessage(Op, ErrorMsgOOR, DAG)
               : DAG.getNode(LoongArchISD::BREAK, DL, MVT::Other, Chain,
                             DAG.getConstant(Imm, DL, GRLenVT));
  }
  case Intrinsic::loongarch_movgr2fcsr: {
    if (!Subtarget.hasBasicF())
      return emitIntrinsicErrorMessage(Op, ErrorMsgReqF, DAG);
    unsigned Imm = cast<ConstantSDNode>(Op2)->getZExtValue();
    return !isUInt<2>(Imm)
               ? emitIntrinsicErrorMessage(Op, ErrorMsgOOR, DAG)
               : DAG.getNode(LoongArchISD::MOVGR2FCSR, DL, MVT::Other, Chain,
                             DAG.getConstant(Imm, DL, GRLenVT),
                             DAG.getNode(ISD::ANY_EXTEND, DL, GRLenVT,
                                         Op.getOperand(3)));
  }
  case Intrinsic::loongarch_syscall: {
    unsigned Imm = cast<ConstantSDNode>(Op2)->getZExtValue();
    return !isUInt<15>(Imm)
               ? emitIntrinsicErrorMessage(Op, ErrorMsgOOR, DAG)
               : DAG.getNode(LoongArchISD::SYSCALL, DL, MVT::Other, Chain,
                             DAG.getConstant(Imm, DL, GRLenVT));
  }
#define IOCSRWR_CASE(NAME, NODE)                                               \
  case Intrinsic::loongarch_##NAME: {                                          \
    SDValue Op3 = Op.getOperand(3);                                            \
    return Subtarget.is64Bit()                                                 \
               ? DAG.getNode(LoongArchISD::NODE, DL, MVT::Other, Chain,        \
                             DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, Op2),  \
                             DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, Op3))  \
               : DAG.getNode(LoongArchISD::NODE, DL, MVT::Other, Chain, Op2,   \
                             Op3);                                             \
  }
    IOCSRWR_CASE(iocsrwr_b, IOCSRWR_B);
    IOCSRWR_CASE(iocsrwr_h, IOCSRWR_H);
    IOCSRWR_CASE(iocsrwr_w, IOCSRWR_W);
#undef IOCSRWR_CASE
  case Intrinsic::loongarch_iocsrwr_d: {
    return !Subtarget.is64Bit()
               ? emitIntrinsicErrorMessage(Op, ErrorMsgReqLA64, DAG)
               : DAG.getNode(LoongArchISD::IOCSRWR_D, DL, MVT::Other, Chain,
                             Op2,
                             DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64,
                                         Op.getOperand(3)));
  }
#define ASRT_LE_GT_CASE(NAME)                                                  \
  case Intrinsic::loongarch_##NAME: {                                          \
    return !Subtarget.is64Bit()                                                \
               ? emitIntrinsicErrorMessage(Op, ErrorMsgReqLA64, DAG)           \
               : Op;                                                           \
  }
    ASRT_LE_GT_CASE(asrtle_d)
    ASRT_LE_GT_CASE(asrtgt_d)
#undef ASRT_LE_GT_CASE
  case Intrinsic::loongarch_ldpte_d: {
    unsigned Imm = cast<ConstantSDNode>(Op.getOperand(3))->getZExtValue();
    return !Subtarget.is64Bit()
               ? emitIntrinsicErrorMessage(Op, ErrorMsgReqLA64, DAG)
           : !isUInt<8>(Imm) ? emitIntrinsicErrorMessage(Op, ErrorMsgOOR, DAG)
                             : Op;
  }
  case Intrinsic::loongarch_lsx_vst:
  case Intrinsic::loongarch_lasx_xvst:
    return !isInt<12>(cast<ConstantSDNode>(Op.getOperand(4))->getSExtValue())
               ? emitIntrinsicErrorMessage(Op, ErrorMsgOOR, DAG)
               : SDValue();
  case Intrinsic::loongarch_lasx_xvstelm_b:
    return (!isInt<8>(cast<ConstantSDNode>(Op.getOperand(4))->getSExtValue()) ||
            !isUInt<5>(cast<ConstantSDNode>(Op.getOperand(5))->getZExtValue()))
               ? emitIntrinsicErrorMessage(Op, ErrorMsgOOR, DAG)
               : SDValue();
  case Intrinsic::loongarch_lsx_vstelm_b:
    return (!isInt<8>(cast<ConstantSDNode>(Op.getOperand(4))->getSExtValue()) ||
            !isUInt<4>(cast<ConstantSDNode>(Op.getOperand(5))->getZExtValue()))
               ? emitIntrinsicErrorMessage(Op, ErrorMsgOOR, DAG)
               : SDValue();
  case Intrinsic::loongarch_lasx_xvstelm_h:
    return (!isShiftedInt<8, 1>(
                cast<ConstantSDNode>(Op.getOperand(4))->getSExtValue()) ||
            !isUInt<4>(cast<ConstantSDNode>(Op.getOperand(5))->getZExtValue()))
               ? emitIntrinsicErrorMessage(
                     Op, "argument out of range or not a multiple of 2", DAG)
               : SDValue();
  case Intrinsic::loongarch_lsx_vstelm_h:
    return (!isShiftedInt<8, 1>(
                cast<ConstantSDNode>(Op.getOperand(4))->getSExtValue()) ||
            !isUInt<3>(cast<ConstantSDNode>(Op.getOperand(5))->getZExtValue()))
               ? emitIntrinsicErrorMessage(
                     Op, "argument out of range or not a multiple of 2", DAG)
               : SDValue();
  case Intrinsic::loongarch_lasx_xvstelm_w:
    return (!isShiftedInt<8, 2>(
                cast<ConstantSDNode>(Op.getOperand(4))->getSExtValue()) ||
            !isUInt<3>(cast<ConstantSDNode>(Op.getOperand(5))->getZExtValue()))
               ? emitIntrinsicErrorMessage(
                     Op, "argument out of range or not a multiple of 4", DAG)
               : SDValue();
  case Intrinsic::loongarch_lsx_vstelm_w:
    return (!isShiftedInt<8, 2>(
                cast<ConstantSDNode>(Op.getOperand(4))->getSExtValue()) ||
            !isUInt<2>(cast<ConstantSDNode>(Op.getOperand(5))->getZExtValue()))
               ? emitIntrinsicErrorMessage(
                     Op, "argument out of range or not a multiple of 4", DAG)
               : SDValue();
  case Intrinsic::loongarch_lasx_xvstelm_d:
    return (!isShiftedInt<8, 3>(
                cast<ConstantSDNode>(Op.getOperand(4))->getSExtValue()) ||
            !isUInt<2>(cast<ConstantSDNode>(Op.getOperand(5))->getZExtValue()))
               ? emitIntrinsicErrorMessage(
                     Op, "argument out of range or not a multiple of 8", DAG)
               : SDValue();
  case Intrinsic::loongarch_lsx_vstelm_d:
    return (!isShiftedInt<8, 3>(
                cast<ConstantSDNode>(Op.getOperand(4))->getSExtValue()) ||
            !isUInt<1>(cast<ConstantSDNode>(Op.getOperand(5))->getZExtValue()))
               ? emitIntrinsicErrorMessage(
                     Op, "argument out of range or not a multiple of 8", DAG)
               : SDValue();
  }
}

SDValue LoongArchTargetLowering::lowerShiftLeftParts(SDValue Op,
                                                     SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue Lo = Op.getOperand(0);
  SDValue Hi = Op.getOperand(1);
  SDValue Shamt = Op.getOperand(2);
  EVT VT = Lo.getValueType();

  // if Shamt-GRLen < 0: // Shamt < GRLen
  //   Lo = Lo << Shamt
  //   Hi = (Hi << Shamt) | ((Lo >>u 1) >>u (GRLen-1 ^ Shamt))
  // else:
  //   Lo = 0
  //   Hi = Lo << (Shamt-GRLen)

  SDValue Zero = DAG.getConstant(0, DL, VT);
  SDValue One = DAG.getConstant(1, DL, VT);
  SDValue MinusGRLen = DAG.getConstant(-(int)Subtarget.getGRLen(), DL, VT);
  SDValue GRLenMinus1 = DAG.getConstant(Subtarget.getGRLen() - 1, DL, VT);
  SDValue ShamtMinusGRLen = DAG.getNode(ISD::ADD, DL, VT, Shamt, MinusGRLen);
  SDValue GRLenMinus1Shamt = DAG.getNode(ISD::XOR, DL, VT, Shamt, GRLenMinus1);

  SDValue LoTrue = DAG.getNode(ISD::SHL, DL, VT, Lo, Shamt);
  SDValue ShiftRight1Lo = DAG.getNode(ISD::SRL, DL, VT, Lo, One);
  SDValue ShiftRightLo =
      DAG.getNode(ISD::SRL, DL, VT, ShiftRight1Lo, GRLenMinus1Shamt);
  SDValue ShiftLeftHi = DAG.getNode(ISD::SHL, DL, VT, Hi, Shamt);
  SDValue HiTrue = DAG.getNode(ISD::OR, DL, VT, ShiftLeftHi, ShiftRightLo);
  SDValue HiFalse = DAG.getNode(ISD::SHL, DL, VT, Lo, ShamtMinusGRLen);

  SDValue CC = DAG.getSetCC(DL, VT, ShamtMinusGRLen, Zero, ISD::SETLT);

  Lo = DAG.getNode(ISD::SELECT, DL, VT, CC, LoTrue, Zero);
  Hi = DAG.getNode(ISD::SELECT, DL, VT, CC, HiTrue, HiFalse);

  SDValue Parts[2] = {Lo, Hi};
  return DAG.getMergeValues(Parts, DL);
}

SDValue LoongArchTargetLowering::lowerShiftRightParts(SDValue Op,
                                                      SelectionDAG &DAG,
                                                      bool IsSRA) const {
  SDLoc DL(Op);
  SDValue Lo = Op.getOperand(0);
  SDValue Hi = Op.getOperand(1);
  SDValue Shamt = Op.getOperand(2);
  EVT VT = Lo.getValueType();

  // SRA expansion:
  //   if Shamt-GRLen < 0: // Shamt < GRLen
  //     Lo = (Lo >>u Shamt) | ((Hi << 1) << (ShAmt ^ GRLen-1))
  //     Hi = Hi >>s Shamt
  //   else:
  //     Lo = Hi >>s (Shamt-GRLen);
  //     Hi = Hi >>s (GRLen-1)
  //
  // SRL expansion:
  //   if Shamt-GRLen < 0: // Shamt < GRLen
  //     Lo = (Lo >>u Shamt) | ((Hi << 1) << (ShAmt ^ GRLen-1))
  //     Hi = Hi >>u Shamt
  //   else:
  //     Lo = Hi >>u (Shamt-GRLen);
  //     Hi = 0;

  unsigned ShiftRightOp = IsSRA ? ISD::SRA : ISD::SRL;

  SDValue Zero = DAG.getConstant(0, DL, VT);
  SDValue One = DAG.getConstant(1, DL, VT);
  SDValue MinusGRLen = DAG.getConstant(-(int)Subtarget.getGRLen(), DL, VT);
  SDValue GRLenMinus1 = DAG.getConstant(Subtarget.getGRLen() - 1, DL, VT);
  SDValue ShamtMinusGRLen = DAG.getNode(ISD::ADD, DL, VT, Shamt, MinusGRLen);
  SDValue GRLenMinus1Shamt = DAG.getNode(ISD::XOR, DL, VT, Shamt, GRLenMinus1);

  SDValue ShiftRightLo = DAG.getNode(ISD::SRL, DL, VT, Lo, Shamt);
  SDValue ShiftLeftHi1 = DAG.getNode(ISD::SHL, DL, VT, Hi, One);
  SDValue ShiftLeftHi =
      DAG.getNode(ISD::SHL, DL, VT, ShiftLeftHi1, GRLenMinus1Shamt);
  SDValue LoTrue = DAG.getNode(ISD::OR, DL, VT, ShiftRightLo, ShiftLeftHi);
  SDValue HiTrue = DAG.getNode(ShiftRightOp, DL, VT, Hi, Shamt);
  SDValue LoFalse = DAG.getNode(ShiftRightOp, DL, VT, Hi, ShamtMinusGRLen);
  SDValue HiFalse =
      IsSRA ? DAG.getNode(ISD::SRA, DL, VT, Hi, GRLenMinus1) : Zero;

  SDValue CC = DAG.getSetCC(DL, VT, ShamtMinusGRLen, Zero, ISD::SETLT);

  Lo = DAG.getNode(ISD::SELECT, DL, VT, CC, LoTrue, LoFalse);
  Hi = DAG.getNode(ISD::SELECT, DL, VT, CC, HiTrue, HiFalse);

  SDValue Parts[2] = {Lo, Hi};
  return DAG.getMergeValues(Parts, DL);
}

// Returns the opcode of the target-specific SDNode that implements the 32-bit
// form of the given Opcode.
static LoongArchISD::NodeType getLoongArchWOpcode(unsigned Opcode) {
  switch (Opcode) {
  default:
    llvm_unreachable("Unexpected opcode");
  case ISD::SHL:
    return LoongArchISD::SLL_W;
  case ISD::SRA:
    return LoongArchISD::SRA_W;
  case ISD::SRL:
    return LoongArchISD::SRL_W;
  case ISD::ROTR:
    return LoongArchISD::ROTR_W;
  case ISD::ROTL:
    return LoongArchISD::ROTL_W;
  case ISD::CTTZ:
    return LoongArchISD::CTZ_W;
  case ISD::CTLZ:
    return LoongArchISD::CLZ_W;
  }
}

// Converts the given i8/i16/i32 operation to a target-specific SelectionDAG
// node. Because i8/i16/i32 isn't a legal type for LA64, these operations would
// otherwise be promoted to i64, making it difficult to select the
// SLL_W/.../*W later one because the fact the operation was originally of
// type i8/i16/i32 is lost.
static SDValue customLegalizeToWOp(SDNode *N, SelectionDAG &DAG, int NumOp,
                                   unsigned ExtOpc = ISD::ANY_EXTEND) {
  SDLoc DL(N);
  LoongArchISD::NodeType WOpcode = getLoongArchWOpcode(N->getOpcode());
  SDValue NewOp0, NewRes;

  switch (NumOp) {
  default:
    llvm_unreachable("Unexpected NumOp");
  case 1: {
    NewOp0 = DAG.getNode(ExtOpc, DL, MVT::i64, N->getOperand(0));
    NewRes = DAG.getNode(WOpcode, DL, MVT::i64, NewOp0);
    break;
  }
  case 2: {
    NewOp0 = DAG.getNode(ExtOpc, DL, MVT::i64, N->getOperand(0));
    SDValue NewOp1 = DAG.getNode(ExtOpc, DL, MVT::i64, N->getOperand(1));
    NewRes = DAG.getNode(WOpcode, DL, MVT::i64, NewOp0, NewOp1);
    break;
  }
    // TODO:Handle more NumOp.
  }

  // ReplaceNodeResults requires we maintain the same type for the return
  // value.
  return DAG.getNode(ISD::TRUNCATE, DL, N->getValueType(0), NewRes);
}

// Helper function that emits error message for intrinsics with/without chain
// and return a UNDEF or and the chain as the results.
static void emitErrorAndReplaceIntrinsicResults(
    SDNode *N, SmallVectorImpl<SDValue> &Results, SelectionDAG &DAG,
    StringRef ErrorMsg, bool WithChain = true) {
  DAG.getContext()->emitError(N->getOperationName(0) + ": " + ErrorMsg + ".");
  Results.push_back(DAG.getUNDEF(N->getValueType(0)));
  if (!WithChain)
    return;
  Results.push_back(N->getOperand(0));
}

template <unsigned N>
static void
replaceVPICKVE2GRResults(SDNode *Node, SmallVectorImpl<SDValue> &Results,
                         SelectionDAG &DAG, const LoongArchSubtarget &Subtarget,
                         unsigned ResOp) {
  const StringRef ErrorMsgOOR = "argument out of range";
  unsigned Imm = cast<ConstantSDNode>(Node->getOperand(2))->getZExtValue();
  if (!isUInt<N>(Imm)) {
    emitErrorAndReplaceIntrinsicResults(Node, Results, DAG, ErrorMsgOOR,
                                        /*WithChain=*/false);
    return;
  }
  SDLoc DL(Node);
  SDValue Vec = Node->getOperand(1);

  SDValue PickElt =
      DAG.getNode(ResOp, DL, Subtarget.getGRLenVT(), Vec,
                  DAG.getConstant(Imm, DL, Subtarget.getGRLenVT()),
                  DAG.getValueType(Vec.getValueType().getVectorElementType()));
  Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, Node->getValueType(0),
                                PickElt.getValue(0)));
}

static void replaceVecCondBranchResults(SDNode *N,
                                        SmallVectorImpl<SDValue> &Results,
                                        SelectionDAG &DAG,
                                        const LoongArchSubtarget &Subtarget,
                                        unsigned ResOp) {
  SDLoc DL(N);
  SDValue Vec = N->getOperand(1);

  SDValue CB = DAG.getNode(ResOp, DL, Subtarget.getGRLenVT(), Vec);
  Results.push_back(
      DAG.getNode(ISD::TRUNCATE, DL, N->getValueType(0), CB.getValue(0)));
}

static void
replaceINTRINSIC_WO_CHAINResults(SDNode *N, SmallVectorImpl<SDValue> &Results,
                                 SelectionDAG &DAG,
                                 const LoongArchSubtarget &Subtarget) {
  switch (N->getConstantOperandVal(0)) {
  default:
    llvm_unreachable("Unexpected Intrinsic.");
  case Intrinsic::loongarch_lsx_vpickve2gr_b:
    replaceVPICKVE2GRResults<4>(N, Results, DAG, Subtarget,
                                LoongArchISD::VPICK_SEXT_ELT);
    break;
  case Intrinsic::loongarch_lsx_vpickve2gr_h:
  case Intrinsic::loongarch_lasx_xvpickve2gr_w:
    replaceVPICKVE2GRResults<3>(N, Results, DAG, Subtarget,
                                LoongArchISD::VPICK_SEXT_ELT);
    break;
  case Intrinsic::loongarch_lsx_vpickve2gr_w:
    replaceVPICKVE2GRResults<2>(N, Results, DAG, Subtarget,
                                LoongArchISD::VPICK_SEXT_ELT);
    break;
  case Intrinsic::loongarch_lsx_vpickve2gr_bu:
    replaceVPICKVE2GRResults<4>(N, Results, DAG, Subtarget,
                                LoongArchISD::VPICK_ZEXT_ELT);
    break;
  case Intrinsic::loongarch_lsx_vpickve2gr_hu:
  case Intrinsic::loongarch_lasx_xvpickve2gr_wu:
    replaceVPICKVE2GRResults<3>(N, Results, DAG, Subtarget,
                                LoongArchISD::VPICK_ZEXT_ELT);
    break;
  case Intrinsic::loongarch_lsx_vpickve2gr_wu:
    replaceVPICKVE2GRResults<2>(N, Results, DAG, Subtarget,
                                LoongArchISD::VPICK_ZEXT_ELT);
    break;
  case Intrinsic::loongarch_lsx_bz_b:
  case Intrinsic::loongarch_lsx_bz_h:
  case Intrinsic::loongarch_lsx_bz_w:
  case Intrinsic::loongarch_lsx_bz_d:
  case Intrinsic::loongarch_lasx_xbz_b:
  case Intrinsic::loongarch_lasx_xbz_h:
  case Intrinsic::loongarch_lasx_xbz_w:
  case Intrinsic::loongarch_lasx_xbz_d:
    replaceVecCondBranchResults(N, Results, DAG, Subtarget,
                                LoongArchISD::VALL_ZERO);
    break;
  case Intrinsic::loongarch_lsx_bz_v:
  case Intrinsic::loongarch_lasx_xbz_v:
    replaceVecCondBranchResults(N, Results, DAG, Subtarget,
                                LoongArchISD::VANY_ZERO);
    break;
  case Intrinsic::loongarch_lsx_bnz_b:
  case Intrinsic::loongarch_lsx_bnz_h:
  case Intrinsic::loongarch_lsx_bnz_w:
  case Intrinsic::loongarch_lsx_bnz_d:
  case Intrinsic::loongarch_lasx_xbnz_b:
  case Intrinsic::loongarch_lasx_xbnz_h:
  case Intrinsic::loongarch_lasx_xbnz_w:
  case Intrinsic::loongarch_lasx_xbnz_d:
    replaceVecCondBranchResults(N, Results, DAG, Subtarget,
                                LoongArchISD::VALL_NONZERO);
    break;
  case Intrinsic::loongarch_lsx_bnz_v:
  case Intrinsic::loongarch_lasx_xbnz_v:
    replaceVecCondBranchResults(N, Results, DAG, Subtarget,
                                LoongArchISD::VANY_NONZERO);
    break;
  }
}

void LoongArchTargetLowering::ReplaceNodeResults(
    SDNode *N, SmallVectorImpl<SDValue> &Results, SelectionDAG &DAG) const {
  SDLoc DL(N);
  EVT VT = N->getValueType(0);
  switch (N->getOpcode()) {
  default:
    llvm_unreachable("Don't know how to legalize this operation");
  case ISD::SHL:
  case ISD::SRA:
  case ISD::SRL:
  case ISD::ROTR:
    assert(VT == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
    if (N->getOperand(1).getOpcode() != ISD::Constant) {
      Results.push_back(customLegalizeToWOp(N, DAG, 2));
      break;
    }
    break;
  case ISD::ROTL:
    ConstantSDNode *CN;
    if ((CN = dyn_cast<ConstantSDNode>(N->getOperand(1)))) {
      Results.push_back(customLegalizeToWOp(N, DAG, 2));
      break;
    }
    break;
  case ISD::FP_TO_SINT: {
    assert(VT == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
    SDValue Src = N->getOperand(0);
    EVT FVT = EVT::getFloatingPointVT(N->getValueSizeInBits(0));
    if (getTypeAction(*DAG.getContext(), Src.getValueType()) !=
        TargetLowering::TypeSoftenFloat) {
      SDValue Dst = DAG.getNode(LoongArchISD::FTINT, DL, FVT, Src);
      Results.push_back(DAG.getNode(ISD::BITCAST, DL, VT, Dst));
      return;
    }
    // If the FP type needs to be softened, emit a library call using the 'si'
    // version. If we left it to default legalization we'd end up with 'di'.
    RTLIB::Libcall LC;
    LC = RTLIB::getFPTOSINT(Src.getValueType(), VT);
    MakeLibCallOptions CallOptions;
    EVT OpVT = Src.getValueType();
    CallOptions.setTypeListBeforeSoften(OpVT, VT, true);
    SDValue Chain = SDValue();
    SDValue Result;
    std::tie(Result, Chain) =
        makeLibCall(DAG, LC, VT, Src, CallOptions, DL, Chain);
    Results.push_back(Result);
    break;
  }
  case ISD::BITCAST: {
    SDValue Src = N->getOperand(0);
    EVT SrcVT = Src.getValueType();
    if (VT == MVT::i32 && SrcVT == MVT::f32 && Subtarget.is64Bit() &&
        Subtarget.hasBasicF()) {
      SDValue Dst =
          DAG.getNode(LoongArchISD::MOVFR2GR_S_LA64, DL, MVT::i64, Src);
      Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, Dst));
    }
    break;
  }
  case ISD::FP_TO_UINT: {
    assert(VT == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
    auto &TLI = DAG.getTargetLoweringInfo();
    SDValue Tmp1, Tmp2;
    TLI.expandFP_TO_UINT(N, Tmp1, Tmp2, DAG);
    Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, Tmp1));
    break;
  }
  case ISD::BSWAP: {
    SDValue Src = N->getOperand(0);
    assert((VT == MVT::i16 || VT == MVT::i32) &&
           "Unexpected custom legalization");
    MVT GRLenVT = Subtarget.getGRLenVT();
    SDValue NewSrc = DAG.getNode(ISD::ANY_EXTEND, DL, GRLenVT, Src);
    SDValue Tmp;
    switch (VT.getSizeInBits()) {
    default:
      llvm_unreachable("Unexpected operand width");
    case 16:
      Tmp = DAG.getNode(LoongArchISD::REVB_2H, DL, GRLenVT, NewSrc);
      break;
    case 32:
      // Only LA64 will get to here due to the size mismatch between VT and
      // GRLenVT, LA32 lowering is directly defined in LoongArchInstrInfo.
      Tmp = DAG.getNode(LoongArchISD::REVB_2W, DL, GRLenVT, NewSrc);
      break;
    }
    Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, VT, Tmp));
    break;
  }
  case ISD::BITREVERSE: {
    SDValue Src = N->getOperand(0);
    assert((VT == MVT::i8 || (VT == MVT::i32 && Subtarget.is64Bit())) &&
           "Unexpected custom legalization");
    MVT GRLenVT = Subtarget.getGRLenVT();
    SDValue NewSrc = DAG.getNode(ISD::ANY_EXTEND, DL, GRLenVT, Src);
    SDValue Tmp;
    switch (VT.getSizeInBits()) {
    default:
      llvm_unreachable("Unexpected operand width");
    case 8:
      Tmp = DAG.getNode(LoongArchISD::BITREV_4B, DL, GRLenVT, NewSrc);
      break;
    case 32:
      Tmp = DAG.getNode(LoongArchISD::BITREV_W, DL, GRLenVT, NewSrc);
      break;
    }
    Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, VT, Tmp));
    break;
  }
  case ISD::CTLZ:
  case ISD::CTTZ: {
    assert(VT == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
    Results.push_back(customLegalizeToWOp(N, DAG, 1));
    break;
  }
  case ISD::INTRINSIC_W_CHAIN: {
    SDValue Chain = N->getOperand(0);
    SDValue Op2 = N->getOperand(2);
    MVT GRLenVT = Subtarget.getGRLenVT();
    const StringRef ErrorMsgOOR = "argument out of range";
    const StringRef ErrorMsgReqLA64 = "requires loongarch64";
    const StringRef ErrorMsgReqF = "requires basic 'f' target feature";

    switch (N->getConstantOperandVal(1)) {
    default:
      llvm_unreachable("Unexpected Intrinsic.");
    case Intrinsic::loongarch_movfcsr2gr: {
      if (!Subtarget.hasBasicF()) {
        emitErrorAndReplaceIntrinsicResults(N, Results, DAG, ErrorMsgReqF);
        return;
      }
      unsigned Imm = cast<ConstantSDNode>(Op2)->getZExtValue();
      if (!isUInt<2>(Imm)) {
        emitErrorAndReplaceIntrinsicResults(N, Results, DAG, ErrorMsgOOR);
        return;
      }
      SDValue MOVFCSR2GRResults = DAG.getNode(
          LoongArchISD::MOVFCSR2GR, SDLoc(N), {MVT::i64, MVT::Other},
          {Chain, DAG.getConstant(Imm, DL, GRLenVT)});
      Results.push_back(
          DAG.getNode(ISD::TRUNCATE, DL, VT, MOVFCSR2GRResults.getValue(0)));
      Results.push_back(MOVFCSR2GRResults.getValue(1));
      break;
    }
#define CRC_CASE_EXT_BINARYOP(NAME, NODE)                                      \
  case Intrinsic::loongarch_##NAME: {                                          \
    SDValue NODE = DAG.getNode(                                                \
        LoongArchISD::NODE, DL, {MVT::i64, MVT::Other},                        \
        {Chain, DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, Op2),               \
         DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(3))});       \
    Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, VT, NODE.getValue(0)));   \
    Results.push_back(NODE.getValue(1));                                       \
    break;                                                                     \
  }
      CRC_CASE_EXT_BINARYOP(crc_w_b_w, CRC_W_B_W)
      CRC_CASE_EXT_BINARYOP(crc_w_h_w, CRC_W_H_W)
      CRC_CASE_EXT_BINARYOP(crc_w_w_w, CRC_W_W_W)
      CRC_CASE_EXT_BINARYOP(crcc_w_b_w, CRCC_W_B_W)
      CRC_CASE_EXT_BINARYOP(crcc_w_h_w, CRCC_W_H_W)
      CRC_CASE_EXT_BINARYOP(crcc_w_w_w, CRCC_W_W_W)
#undef CRC_CASE_EXT_BINARYOP

#define CRC_CASE_EXT_UNARYOP(NAME, NODE)                                       \
  case Intrinsic::loongarch_##NAME: {                                          \
    SDValue NODE = DAG.getNode(                                                \
        LoongArchISD::NODE, DL, {MVT::i64, MVT::Other},                        \
        {Chain, Op2,                                                           \
         DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(3))});       \
    Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, VT, NODE.getValue(0)));   \
    Results.push_back(NODE.getValue(1));                                       \
    break;                                                                     \
  }
      CRC_CASE_EXT_UNARYOP(crc_w_d_w, CRC_W_D_W)
      CRC_CASE_EXT_UNARYOP(crcc_w_d_w, CRCC_W_D_W)
#undef CRC_CASE_EXT_UNARYOP
#define CSR_CASE(ID)                                                           \
  case Intrinsic::loongarch_##ID: {                                            \
    if (!Subtarget.is64Bit())                                                  \
      emitErrorAndReplaceIntrinsicResults(N, Results, DAG, ErrorMsgReqLA64);   \
    break;                                                                     \
  }
      CSR_CASE(csrrd_d);
      CSR_CASE(csrwr_d);
      CSR_CASE(csrxchg_d);
      CSR_CASE(iocsrrd_d);
#undef CSR_CASE
    case Intrinsic::loongarch_csrrd_w: {
      unsigned Imm = cast<ConstantSDNode>(Op2)->getZExtValue();
      if (!isUInt<14>(Imm)) {
        emitErrorAndReplaceIntrinsicResults(N, Results, DAG, ErrorMsgOOR);
        return;
      }
      SDValue CSRRDResults =
          DAG.getNode(LoongArchISD::CSRRD, DL, {GRLenVT, MVT::Other},
                      {Chain, DAG.getConstant(Imm, DL, GRLenVT)});
      Results.push_back(
          DAG.getNode(ISD::TRUNCATE, DL, VT, CSRRDResults.getValue(0)));
      Results.push_back(CSRRDResults.getValue(1));
      break;
    }
    case Intrinsic::loongarch_csrwr_w: {
      unsigned Imm = cast<ConstantSDNode>(N->getOperand(3))->getZExtValue();
      if (!isUInt<14>(Imm)) {
        emitErrorAndReplaceIntrinsicResults(N, Results, DAG, ErrorMsgOOR);
        return;
      }
      SDValue CSRWRResults =
          DAG.getNode(LoongArchISD::CSRWR, DL, {GRLenVT, MVT::Other},
                      {Chain, DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, Op2),
                       DAG.getConstant(Imm, DL, GRLenVT)});
      Results.push_back(
          DAG.getNode(ISD::TRUNCATE, DL, VT, CSRWRResults.getValue(0)));
      Results.push_back(CSRWRResults.getValue(1));
      break;
    }
    case Intrinsic::loongarch_csrxchg_w: {
      unsigned Imm = cast<ConstantSDNode>(N->getOperand(4))->getZExtValue();
      if (!isUInt<14>(Imm)) {
        emitErrorAndReplaceIntrinsicResults(N, Results, DAG, ErrorMsgOOR);
        return;
      }
      SDValue CSRXCHGResults = DAG.getNode(
          LoongArchISD::CSRXCHG, DL, {GRLenVT, MVT::Other},
          {Chain, DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, Op2),
           DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(3)),
           DAG.getConstant(Imm, DL, GRLenVT)});
      Results.push_back(
          DAG.getNode(ISD::TRUNCATE, DL, VT, CSRXCHGResults.getValue(0)));
      Results.push_back(CSRXCHGResults.getValue(1));
      break;
    }
#define IOCSRRD_CASE(NAME, NODE)                                               \
  case Intrinsic::loongarch_##NAME: {                                          \
    SDValue IOCSRRDResults =                                                   \
        DAG.getNode(LoongArchISD::NODE, DL, {MVT::i64, MVT::Other},            \
                    {Chain, DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, Op2)}); \
    Results.push_back(                                                         \
        DAG.getNode(ISD::TRUNCATE, DL, VT, IOCSRRDResults.getValue(0)));       \
    Results.push_back(IOCSRRDResults.getValue(1));                             \
    break;                                                                     \
  }
      IOCSRRD_CASE(iocsrrd_b, IOCSRRD_B);
      IOCSRRD_CASE(iocsrrd_h, IOCSRRD_H);
      IOCSRRD_CASE(iocsrrd_w, IOCSRRD_W);
#undef IOCSRRD_CASE
    case Intrinsic::loongarch_cpucfg: {
      SDValue CPUCFGResults =
          DAG.getNode(LoongArchISD::CPUCFG, DL, {GRLenVT, MVT::Other},
                      {Chain, DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, Op2)});
      Results.push_back(
          DAG.getNode(ISD::TRUNCATE, DL, VT, CPUCFGResults.getValue(0)));
      Results.push_back(CPUCFGResults.getValue(1));
      break;
    }
    case Intrinsic::loongarch_lddir_d: {
      if (!Subtarget.is64Bit()) {
        emitErrorAndReplaceIntrinsicResults(N, Results, DAG, ErrorMsgReqLA64);
        return;
      }
      break;
    }
    }
    break;
  }
  case ISD::READ_REGISTER: {
    if (Subtarget.is64Bit())
      DAG.getContext()->emitError(
          "On LA64, only 64-bit registers can be read.");
    else
      DAG.getContext()->emitError(
          "On LA32, only 32-bit registers can be read.");
    Results.push_back(DAG.getUNDEF(VT));
    Results.push_back(N->getOperand(0));
    break;
  }
  case ISD::INTRINSIC_WO_CHAIN: {
    replaceINTRINSIC_WO_CHAINResults(N, Results, DAG, Subtarget);
    break;
  }
  }
}

static SDValue performANDCombine(SDNode *N, SelectionDAG &DAG,
                                 TargetLowering::DAGCombinerInfo &DCI,
                                 const LoongArchSubtarget &Subtarget) {
  if (DCI.isBeforeLegalizeOps())
    return SDValue();

  SDValue FirstOperand = N->getOperand(0);
  SDValue SecondOperand = N->getOperand(1);
  unsigned FirstOperandOpc = FirstOperand.getOpcode();
  EVT ValTy = N->getValueType(0);
  SDLoc DL(N);
  uint64_t lsb, msb;
  unsigned SMIdx, SMLen;
  ConstantSDNode *CN;
  SDValue NewOperand;
  MVT GRLenVT = Subtarget.getGRLenVT();

  // Op's second operand must be a shifted mask.
  if (!(CN = dyn_cast<ConstantSDNode>(SecondOperand)) ||
      !isShiftedMask_64(CN->getZExtValue(), SMIdx, SMLen))
    return SDValue();

  if (FirstOperandOpc == ISD::SRA || FirstOperandOpc == ISD::SRL) {
    // Pattern match BSTRPICK.
    //  $dst = and ((sra or srl) $src , lsb), (2**len - 1)
    //  => BSTRPICK $dst, $src, msb, lsb
    //  where msb = lsb + len - 1

    // The second operand of the shift must be an immediate.
    if (!(CN = dyn_cast<ConstantSDNode>(FirstOperand.getOperand(1))))
      return SDValue();

    lsb = CN->getZExtValue();

    // Return if the shifted mask does not start at bit 0 or the sum of its
    // length and lsb exceeds the word's size.
    if (SMIdx != 0 || lsb + SMLen > ValTy.getSizeInBits())
      return SDValue();

    NewOperand = FirstOperand.getOperand(0);
  } else {
    // Pattern match BSTRPICK.
    //  $dst = and $src, (2**len- 1) , if len > 12
    //  => BSTRPICK $dst, $src, msb, lsb
    //  where lsb = 0 and msb = len - 1

    // If the mask is <= 0xfff, andi can be used instead.
    if (CN->getZExtValue() <= 0xfff)
      return SDValue();

    // Return if the MSB exceeds.
    if (SMIdx + SMLen > ValTy.getSizeInBits())
      return SDValue();

    if (SMIdx > 0) {
      // Omit if the constant has more than 2 uses. This a conservative
      // decision. Whether it is a win depends on the HW microarchitecture.
      // However it should always be better for 1 and 2 uses.
      if (CN->use_size() > 2)
        return SDValue();
      // Return if the constant can be composed by a single LU12I.W.
      if ((CN->getZExtValue() & 0xfff) == 0)
        return SDValue();
      // Return if the constand can be composed by a single ADDI with
      // the zero register.
      if (CN->getSExtValue() >= -2048 && CN->getSExtValue() < 0)
        return SDValue();
    }

    lsb = SMIdx;
    NewOperand = FirstOperand;
  }

  msb = lsb + SMLen - 1;
  SDValue NR0 = DAG.getNode(LoongArchISD::BSTRPICK, DL, ValTy, NewOperand,
                            DAG.getConstant(msb, DL, GRLenVT),
                            DAG.getConstant(lsb, DL, GRLenVT));
  if (FirstOperandOpc == ISD::SRA || FirstOperandOpc == ISD::SRL || lsb == 0)
    return NR0;
  // Try to optimize to
  //   bstrpick $Rd, $Rs, msb, lsb
  //   slli     $Rd, $Rd, lsb
  return DAG.getNode(ISD::SHL, DL, ValTy, NR0,
                     DAG.getConstant(lsb, DL, GRLenVT));
}

static SDValue performSRLCombine(SDNode *N, SelectionDAG &DAG,
                                 TargetLowering::DAGCombinerInfo &DCI,
                                 const LoongArchSubtarget &Subtarget) {
  if (DCI.isBeforeLegalizeOps())
    return SDValue();

  // $dst = srl (and $src, Mask), Shamt
  // =>
  // BSTRPICK $dst, $src, MaskIdx+MaskLen-1, Shamt
  // when Mask is a shifted mask, and MaskIdx <= Shamt <= MaskIdx+MaskLen-1
  //

  SDValue FirstOperand = N->getOperand(0);
  ConstantSDNode *CN;
  EVT ValTy = N->getValueType(0);
  SDLoc DL(N);
  MVT GRLenVT = Subtarget.getGRLenVT();
  unsigned MaskIdx, MaskLen;
  uint64_t Shamt;

  // The first operand must be an AND and the second operand of the AND must be
  // a shifted mask.
  if (FirstOperand.getOpcode() != ISD::AND ||
      !(CN = dyn_cast<ConstantSDNode>(FirstOperand.getOperand(1))) ||
      !isShiftedMask_64(CN->getZExtValue(), MaskIdx, MaskLen))
    return SDValue();

  // The second operand (shift amount) must be an immediate.
  if (!(CN = dyn_cast<ConstantSDNode>(N->getOperand(1))))
    return SDValue();

  Shamt = CN->getZExtValue();
  if (MaskIdx <= Shamt && Shamt <= MaskIdx + MaskLen - 1)
    return DAG.getNode(LoongArchISD::BSTRPICK, DL, ValTy,
                       FirstOperand->getOperand(0),
                       DAG.getConstant(MaskIdx + MaskLen - 1, DL, GRLenVT),
                       DAG.getConstant(Shamt, DL, GRLenVT));

  return SDValue();
}

static SDValue performORCombine(SDNode *N, SelectionDAG &DAG,
                                TargetLowering::DAGCombinerInfo &DCI,
                                const LoongArchSubtarget &Subtarget) {
  MVT GRLenVT = Subtarget.getGRLenVT();
  EVT ValTy = N->getValueType(0);
  SDValue N0 = N->getOperand(0), N1 = N->getOperand(1);
  ConstantSDNode *CN0, *CN1;
  SDLoc DL(N);
  unsigned ValBits = ValTy.getSizeInBits();
  unsigned MaskIdx0, MaskLen0, MaskIdx1, MaskLen1;
  unsigned Shamt;
  bool SwapAndRetried = false;

  if (DCI.isBeforeLegalizeOps())
    return SDValue();

  if (ValBits != 32 && ValBits != 64)
    return SDValue();

Retry:
  // 1st pattern to match BSTRINS:
  //  R = or (and X, mask0), (and (shl Y, lsb), mask1)
  //  where mask1 = (2**size - 1) << lsb, mask0 = ~mask1
  //  =>
  //  R = BSTRINS X, Y, msb, lsb (where msb = lsb + size - 1)
  if (N0.getOpcode() == ISD::AND &&
      (CN0 = dyn_cast<ConstantSDNode>(N0.getOperand(1))) &&
      isShiftedMask_64(~CN0->getSExtValue(), MaskIdx0, MaskLen0) &&
      N1.getOpcode() == ISD::AND && N1.getOperand(0).getOpcode() == ISD::SHL &&
      (CN1 = dyn_cast<ConstantSDNode>(N1.getOperand(1))) &&
      isShiftedMask_64(CN1->getZExtValue(), MaskIdx1, MaskLen1) &&
      MaskIdx0 == MaskIdx1 && MaskLen0 == MaskLen1 &&
      (CN1 = dyn_cast<ConstantSDNode>(N1.getOperand(0).getOperand(1))) &&
      (Shamt = CN1->getZExtValue()) == MaskIdx0 &&
      (MaskIdx0 + MaskLen0 <= ValBits)) {
    LLVM_DEBUG(dbgs() << "Perform OR combine: match pattern 1\n");
    return DAG.getNode(LoongArchISD::BSTRINS, DL, ValTy, N0.getOperand(0),
                       N1.getOperand(0).getOperand(0),
                       DAG.getConstant((MaskIdx0 + MaskLen0 - 1), DL, GRLenVT),
                       DAG.getConstant(MaskIdx0, DL, GRLenVT));
  }

  // 2nd pattern to match BSTRINS:
  //  R = or (and X, mask0), (shl (and Y, mask1), lsb)
  //  where mask1 = (2**size - 1), mask0 = ~(mask1 << lsb)
  //  =>
  //  R = BSTRINS X, Y, msb, lsb (where msb = lsb + size - 1)
  if (N0.getOpcode() == ISD::AND &&
      (CN0 = dyn_cast<ConstantSDNode>(N0.getOperand(1))) &&
      isShiftedMask_64(~CN0->getSExtValue(), MaskIdx0, MaskLen0) &&
      N1.getOpcode() == ISD::SHL && N1.getOperand(0).getOpcode() == ISD::AND &&
      (CN1 = dyn_cast<ConstantSDNode>(N1.getOperand(1))) &&
      (Shamt = CN1->getZExtValue()) == MaskIdx0 &&
      (CN1 = dyn_cast<ConstantSDNode>(N1.getOperand(0).getOperand(1))) &&
      isShiftedMask_64(CN1->getZExtValue(), MaskIdx1, MaskLen1) &&
      MaskLen0 == MaskLen1 && MaskIdx1 == 0 &&
      (MaskIdx0 + MaskLen0 <= ValBits)) {
    LLVM_DEBUG(dbgs() << "Perform OR combine: match pattern 2\n");
    return DAG.getNode(LoongArchISD::BSTRINS, DL, ValTy, N0.getOperand(0),
                       N1.getOperand(0).getOperand(0),
                       DAG.getConstant((MaskIdx0 + MaskLen0 - 1), DL, GRLenVT),
                       DAG.getConstant(MaskIdx0, DL, GRLenVT));
  }

  // 3rd pattern to match BSTRINS:
  //  R = or (and X, mask0), (and Y, mask1)
  //  where ~mask0 = (2**size - 1) << lsb, mask0 & mask1 = 0
  //  =>
  //  R = BSTRINS X, (shr (and Y, mask1), lsb), msb, lsb
  //  where msb = lsb + size - 1
  if (N0.getOpcode() == ISD::AND && N1.getOpcode() == ISD::AND &&
      (CN0 = dyn_cast<ConstantSDNode>(N0.getOperand(1))) &&
      isShiftedMask_64(~CN0->getSExtValue(), MaskIdx0, MaskLen0) &&
      (MaskIdx0 + MaskLen0 <= 64) &&
      (CN1 = dyn_cast<ConstantSDNode>(N1->getOperand(1))) &&
      (CN1->getSExtValue() & CN0->getSExtValue()) == 0) {
    LLVM_DEBUG(dbgs() << "Perform OR combine: match pattern 3\n");
    return DAG.getNode(LoongArchISD::BSTRINS, DL, ValTy, N0.getOperand(0),
                       DAG.getNode(ISD::SRL, DL, N1->getValueType(0), N1,
                                   DAG.getConstant(MaskIdx0, DL, GRLenVT)),
                       DAG.getConstant(ValBits == 32
                                           ? (MaskIdx0 + (MaskLen0 & 31) - 1)
                                           : (MaskIdx0 + MaskLen0 - 1),
                                       DL, GRLenVT),
                       DAG.getConstant(MaskIdx0, DL, GRLenVT));
  }

  // 4th pattern to match BSTRINS:
  //  R = or (and X, mask), (shl Y, shamt)
  //  where mask = (2**shamt - 1)
  //  =>
  //  R = BSTRINS X, Y, ValBits - 1, shamt
  //  where ValBits = 32 or 64
  if (N0.getOpcode() == ISD::AND && N1.getOpcode() == ISD::SHL &&
      (CN0 = dyn_cast<ConstantSDNode>(N0.getOperand(1))) &&
      isShiftedMask_64(CN0->getZExtValue(), MaskIdx0, MaskLen0) &&
      MaskIdx0 == 0 && (CN1 = dyn_cast<ConstantSDNode>(N1.getOperand(1))) &&
      (Shamt = CN1->getZExtValue()) == MaskLen0 &&
      (MaskIdx0 + MaskLen0 <= ValBits)) {
    LLVM_DEBUG(dbgs() << "Perform OR combine: match pattern 4\n");
    return DAG.getNode(LoongArchISD::BSTRINS, DL, ValTy, N0.getOperand(0),
                       N1.getOperand(0),
                       DAG.getConstant((ValBits - 1), DL, GRLenVT),
                       DAG.getConstant(Shamt, DL, GRLenVT));
  }

  // 5th pattern to match BSTRINS:
  //  R = or (and X, mask), const
  //  where ~mask = (2**size - 1) << lsb, mask & const = 0
  //  =>
  //  R = BSTRINS X, (const >> lsb), msb, lsb
  //  where msb = lsb + size - 1
  if (N0.getOpcode() == ISD::AND &&
      (CN0 = dyn_cast<ConstantSDNode>(N0.getOperand(1))) &&
      isShiftedMask_64(~CN0->getSExtValue(), MaskIdx0, MaskLen0) &&
      (CN1 = dyn_cast<ConstantSDNode>(N1)) &&
      (CN1->getSExtValue() & CN0->getSExtValue()) == 0) {
    LLVM_DEBUG(dbgs() << "Perform OR combine: match pattern 5\n");
    return DAG.getNode(
        LoongArchISD::BSTRINS, DL, ValTy, N0.getOperand(0),
        DAG.getConstant(CN1->getSExtValue() >> MaskIdx0, DL, ValTy),
        DAG.getConstant((MaskIdx0 + MaskLen0 - 1), DL, GRLenVT),
        DAG.getConstant(MaskIdx0, DL, GRLenVT));
  }

  // 6th pattern.
  // a = b | ((c & mask) << shamt), where all positions in b to be overwritten
  // by the incoming bits are known to be zero.
  // =>
  // a = BSTRINS b, c, shamt + MaskLen - 1, shamt
  //
  // Note that the 1st pattern is a special situation of the 6th, i.e. the 6th
  // pattern is more common than the 1st. So we put the 1st before the 6th in
  // order to match as many nodes as possible.
  ConstantSDNode *CNMask, *CNShamt;
  unsigned MaskIdx, MaskLen;
  if (N1.getOpcode() == ISD::SHL && N1.getOperand(0).getOpcode() == ISD::AND &&
      (CNMask = dyn_cast<ConstantSDNode>(N1.getOperand(0).getOperand(1))) &&
      isShiftedMask_64(CNMask->getZExtValue(), MaskIdx, MaskLen) &&
      MaskIdx == 0 && (CNShamt = dyn_cast<ConstantSDNode>(N1.getOperand(1))) &&
      CNShamt->getZExtValue() + MaskLen <= ValBits) {
    Shamt = CNShamt->getZExtValue();
    APInt ShMask(ValBits, CNMask->getZExtValue() << Shamt);
    if (ShMask.isSubsetOf(DAG.computeKnownBits(N0).Zero)) {
      LLVM_DEBUG(dbgs() << "Perform OR combine: match pattern 6\n");
      return DAG.getNode(LoongArchISD::BSTRINS, DL, ValTy, N0,
                         N1.getOperand(0).getOperand(0),
                         DAG.getConstant(Shamt + MaskLen - 1, DL, GRLenVT),
                         DAG.getConstant(Shamt, DL, GRLenVT));
    }
  }

  // 7th pattern.
  // a = b | ((c << shamt) & shifted_mask), where all positions in b to be
  // overwritten by the incoming bits are known to be zero.
  // =>
  // a = BSTRINS b, c, MaskIdx + MaskLen - 1, MaskIdx
  //
  // Similarly, the 7th pattern is more common than the 2nd. So we put the 2nd
  // before the 7th in order to match as many nodes as possible.
  if (N1.getOpcode() == ISD::AND &&
      (CNMask = dyn_cast<ConstantSDNode>(N1.getOperand(1))) &&
      isShiftedMask_64(CNMask->getZExtValue(), MaskIdx, MaskLen) &&
      N1.getOperand(0).getOpcode() == ISD::SHL &&
      (CNShamt = dyn_cast<ConstantSDNode>(N1.getOperand(0).getOperand(1))) &&
      CNShamt->getZExtValue() == MaskIdx) {
    APInt ShMask(ValBits, CNMask->getZExtValue());
    if (ShMask.isSubsetOf(DAG.computeKnownBits(N0).Zero)) {
      LLVM_DEBUG(dbgs() << "Perform OR combine: match pattern 7\n");
      return DAG.getNode(LoongArchISD::BSTRINS, DL, ValTy, N0,
                         N1.getOperand(0).getOperand(0),
                         DAG.getConstant(MaskIdx + MaskLen - 1, DL, GRLenVT),
                         DAG.getConstant(MaskIdx, DL, GRLenVT));
    }
  }

  // (or a, b) and (or b, a) are equivalent, so swap the operands and retry.
  if (!SwapAndRetried) {
    std::swap(N0, N1);
    SwapAndRetried = true;
    goto Retry;
  }

  SwapAndRetried = false;
Retry2:
  // 8th pattern.
  // a = b | (c & shifted_mask), where all positions in b to be overwritten by
  // the incoming bits are known to be zero.
  // =>
  // a = BSTRINS b, c >> MaskIdx, MaskIdx + MaskLen - 1, MaskIdx
  //
  // Similarly, the 8th pattern is more common than the 4th and 5th patterns. So
  // we put it here in order to match as many nodes as possible or generate less
  // instructions.
  if (N1.getOpcode() == ISD::AND &&
      (CNMask = dyn_cast<ConstantSDNode>(N1.getOperand(1))) &&
      isShiftedMask_64(CNMask->getZExtValue(), MaskIdx, MaskLen)) {
    APInt ShMask(ValBits, CNMask->getZExtValue());
    if (ShMask.isSubsetOf(DAG.computeKnownBits(N0).Zero)) {
      LLVM_DEBUG(dbgs() << "Perform OR combine: match pattern 8\n");
      return DAG.getNode(LoongArchISD::BSTRINS, DL, ValTy, N0,
                         DAG.getNode(ISD::SRL, DL, N1->getValueType(0),
                                     N1->getOperand(0),
                                     DAG.getConstant(MaskIdx, DL, GRLenVT)),
                         DAG.getConstant(MaskIdx + MaskLen - 1, DL, GRLenVT),
                         DAG.getConstant(MaskIdx, DL, GRLenVT));
    }
  }
  // Swap N0/N1 and retry.
  if (!SwapAndRetried) {
    std::swap(N0, N1);
    SwapAndRetried = true;
    goto Retry2;
  }

  return SDValue();
}

// Combine (loongarch_bitrev_w (loongarch_revb_2w X)) to loongarch_bitrev_4b.
static SDValue performBITREV_WCombine(SDNode *N, SelectionDAG &DAG,
                                      TargetLowering::DAGCombinerInfo &DCI,
                                      const LoongArchSubtarget &Subtarget) {
  if (DCI.isBeforeLegalizeOps())
    return SDValue();

  SDValue Src = N->getOperand(0);
  if (Src.getOpcode() != LoongArchISD::REVB_2W)
    return SDValue();

  return DAG.getNode(LoongArchISD::BITREV_4B, SDLoc(N), N->getValueType(0),
                     Src.getOperand(0));
}

template <unsigned N>
static SDValue legalizeIntrinsicImmArg(SDNode *Node, unsigned ImmOp,
                                       SelectionDAG &DAG,
                                       const LoongArchSubtarget &Subtarget,
                                       bool IsSigned = false) {
  SDLoc DL(Node);
  auto *CImm = cast<ConstantSDNode>(Node->getOperand(ImmOp));
  // Check the ImmArg.
  if ((IsSigned && !isInt<N>(CImm->getSExtValue())) ||
      (!IsSigned && !isUInt<N>(CImm->getZExtValue()))) {
    DAG.getContext()->emitError(Node->getOperationName(0) +
                                ": argument out of range.");
    return DAG.getNode(ISD::UNDEF, DL, Subtarget.getGRLenVT());
  }
  return DAG.getConstant(CImm->getZExtValue(), DL, Subtarget.getGRLenVT());
}

template <unsigned N>
static SDValue lowerVectorSplatImm(SDNode *Node, unsigned ImmOp,
                                   SelectionDAG &DAG, bool IsSigned = false) {
  SDLoc DL(Node);
  EVT ResTy = Node->getValueType(0);
  auto *CImm = cast<ConstantSDNode>(Node->getOperand(ImmOp));

  // Check the ImmArg.
  if ((IsSigned && !isInt<N>(CImm->getSExtValue())) ||
      (!IsSigned && !isUInt<N>(CImm->getZExtValue()))) {
    DAG.getContext()->emitError(Node->getOperationName(0) +
                                ": argument out of range.");
    return DAG.getNode(ISD::UNDEF, DL, ResTy);
  }
  return DAG.getConstant(
      APInt(ResTy.getScalarType().getSizeInBits(),
            IsSigned ? CImm->getSExtValue() : CImm->getZExtValue(), IsSigned),
      DL, ResTy);
}

static SDValue truncateVecElts(SDNode *Node, SelectionDAG &DAG) {
  SDLoc DL(Node);
  EVT ResTy = Node->getValueType(0);
  SDValue Vec = Node->getOperand(2);
  SDValue Mask = DAG.getConstant(Vec.getScalarValueSizeInBits() - 1, DL, ResTy);
  return DAG.getNode(ISD::AND, DL, ResTy, Vec, Mask);
}

static SDValue lowerVectorBitClear(SDNode *Node, SelectionDAG &DAG) {
  SDLoc DL(Node);
  EVT ResTy = Node->getValueType(0);
  SDValue One = DAG.getConstant(1, DL, ResTy);
  SDValue Bit =
      DAG.getNode(ISD::SHL, DL, ResTy, One, truncateVecElts(Node, DAG));

  return DAG.getNode(ISD::AND, DL, ResTy, Node->getOperand(1),
                     DAG.getNOT(DL, Bit, ResTy));
}

template <unsigned N>
static SDValue lowerVectorBitClearImm(SDNode *Node, SelectionDAG &DAG) {
  SDLoc DL(Node);
  EVT ResTy = Node->getValueType(0);
  auto *CImm = cast<ConstantSDNode>(Node->getOperand(2));
  // Check the unsigned ImmArg.
  if (!isUInt<N>(CImm->getZExtValue())) {
    DAG.getContext()->emitError(Node->getOperationName(0) +
                                ": argument out of range.");
    return DAG.getNode(ISD::UNDEF, DL, ResTy);
  }

  APInt BitImm = APInt(ResTy.getScalarSizeInBits(), 1) << CImm->getAPIntValue();
  SDValue Mask = DAG.getConstant(~BitImm, DL, ResTy);

  return DAG.getNode(ISD::AND, DL, ResTy, Node->getOperand(1), Mask);
}

template <unsigned N>
static SDValue lowerVectorBitSetImm(SDNode *Node, SelectionDAG &DAG) {
  SDLoc DL(Node);
  EVT ResTy = Node->getValueType(0);
  auto *CImm = cast<ConstantSDNode>(Node->getOperand(2));
  // Check the unsigned ImmArg.
  if (!isUInt<N>(CImm->getZExtValue())) {
    DAG.getContext()->emitError(Node->getOperationName(0) +
                                ": argument out of range.");
    return DAG.getNode(ISD::UNDEF, DL, ResTy);
  }

  APInt Imm = APInt(ResTy.getScalarSizeInBits(), 1) << CImm->getAPIntValue();
  SDValue BitImm = DAG.getConstant(Imm, DL, ResTy);
  return DAG.getNode(ISD::OR, DL, ResTy, Node->getOperand(1), BitImm);
}

template <unsigned N>
static SDValue lowerVectorBitRevImm(SDNode *Node, SelectionDAG &DAG) {
  SDLoc DL(Node);
  EVT ResTy = Node->getValueType(0);
  auto *CImm = cast<ConstantSDNode>(Node->getOperand(2));
  // Check the unsigned ImmArg.
  if (!isUInt<N>(CImm->getZExtValue())) {
    DAG.getContext()->emitError(Node->getOperationName(0) +
                                ": argument out of range.");
    return DAG.getNode(ISD::UNDEF, DL, ResTy);
  }

  APInt Imm = APInt(ResTy.getScalarSizeInBits(), 1) << CImm->getAPIntValue();
  SDValue BitImm = DAG.getConstant(Imm, DL, ResTy);
  return DAG.getNode(ISD::XOR, DL, ResTy, Node->getOperand(1), BitImm);
}

static SDValue
performINTRINSIC_WO_CHAINCombine(SDNode *N, SelectionDAG &DAG,
                                 TargetLowering::DAGCombinerInfo &DCI,
                                 const LoongArchSubtarget &Subtarget) {
  SDLoc DL(N);
  switch (N->getConstantOperandVal(0)) {
  default:
    break;
  case Intrinsic::loongarch_lsx_vadd_b:
  case Intrinsic::loongarch_lsx_vadd_h:
  case Intrinsic::loongarch_lsx_vadd_w:
  case Intrinsic::loongarch_lsx_vadd_d:
  case Intrinsic::loongarch_lasx_xvadd_b:
  case Intrinsic::loongarch_lasx_xvadd_h:
  case Intrinsic::loongarch_lasx_xvadd_w:
  case Intrinsic::loongarch_lasx_xvadd_d:
    return DAG.getNode(ISD::ADD, DL, N->getValueType(0), N->getOperand(1),
                       N->getOperand(2));
  case Intrinsic::loongarch_lsx_vaddi_bu:
  case Intrinsic::loongarch_lsx_vaddi_hu:
  case Intrinsic::loongarch_lsx_vaddi_wu:
  case Intrinsic::loongarch_lsx_vaddi_du:
  case Intrinsic::loongarch_lasx_xvaddi_bu:
  case Intrinsic::loongarch_lasx_xvaddi_hu:
  case Intrinsic::loongarch_lasx_xvaddi_wu:
  case Intrinsic::loongarch_lasx_xvaddi_du:
    return DAG.getNode(ISD::ADD, DL, N->getValueType(0), N->getOperand(1),
                       lowerVectorSplatImm<5>(N, 2, DAG));
  case Intrinsic::loongarch_lsx_vsub_b:
  case Intrinsic::loongarch_lsx_vsub_h:
  case Intrinsic::loongarch_lsx_vsub_w:
  case Intrinsic::loongarch_lsx_vsub_d:
  case Intrinsic::loongarch_lasx_xvsub_b:
  case Intrinsic::loongarch_lasx_xvsub_h:
  case Intrinsic::loongarch_lasx_xvsub_w:
  case Intrinsic::loongarch_lasx_xvsub_d:
    return DAG.getNode(ISD::SUB, DL, N->getValueType(0), N->getOperand(1),
                       N->getOperand(2));
  case Intrinsic::loongarch_lsx_vsubi_bu:
  case Intrinsic::loongarch_lsx_vsubi_hu:
  case Intrinsic::loongarch_lsx_vsubi_wu:
  case Intrinsic::loongarch_lsx_vsubi_du:
  case Intrinsic::loongarch_lasx_xvsubi_bu:
  case Intrinsic::loongarch_lasx_xvsubi_hu:
  case Intrinsic::loongarch_lasx_xvsubi_wu:
  case Intrinsic::loongarch_lasx_xvsubi_du:
    return DAG.getNode(ISD::SUB, DL, N->getValueType(0), N->getOperand(1),
                       lowerVectorSplatImm<5>(N, 2, DAG));
  case Intrinsic::loongarch_lsx_vneg_b:
  case Intrinsic::loongarch_lsx_vneg_h:
  case Intrinsic::loongarch_lsx_vneg_w:
  case Intrinsic::loongarch_lsx_vneg_d:
  case Intrinsic::loongarch_lasx_xvneg_b:
  case Intrinsic::loongarch_lasx_xvneg_h:
  case Intrinsic::loongarch_lasx_xvneg_w:
  case Intrinsic::loongarch_lasx_xvneg_d:
    return DAG.getNode(
        ISD::SUB, DL, N->getValueType(0),
        DAG.getConstant(
            APInt(N->getValueType(0).getScalarType().getSizeInBits(), 0,
                  /*isSigned=*/true),
            SDLoc(N), N->getValueType(0)),
        N->getOperand(1));
  case Intrinsic::loongarch_lsx_vmax_b:
  case Intrinsic::loongarch_lsx_vmax_h:
  case Intrinsic::loongarch_lsx_vmax_w:
  case Intrinsic::loongarch_lsx_vmax_d:
  case Intrinsic::loongarch_lasx_xvmax_b:
  case Intrinsic::loongarch_lasx_xvmax_h:
  case Intrinsic::loongarch_lasx_xvmax_w:
  case Intrinsic::loongarch_lasx_xvmax_d:
    return DAG.getNode(ISD::SMAX, DL, N->getValueType(0), N->getOperand(1),
                       N->getOperand(2));
  case Intrinsic::loongarch_lsx_vmax_bu:
  case Intrinsic::loongarch_lsx_vmax_hu:
  case Intrinsic::loongarch_lsx_vmax_wu:
  case Intrinsic::loongarch_lsx_vmax_du:
  case Intrinsic::loongarch_lasx_xvmax_bu:
  case Intrinsic::loongarch_lasx_xvmax_hu:
  case Intrinsic::loongarch_lasx_xvmax_wu:
  case Intrinsic::loongarch_lasx_xvmax_du:
    return DAG.getNode(ISD::UMAX, DL, N->getValueType(0), N->getOperand(1),
                       N->getOperand(2));
  case Intrinsic::loongarch_lsx_vmaxi_b:
  case Intrinsic::loongarch_lsx_vmaxi_h:
  case Intrinsic::loongarch_lsx_vmaxi_w:
  case Intrinsic::loongarch_lsx_vmaxi_d:
  case Intrinsic::loongarch_lasx_xvmaxi_b:
  case Intrinsic::loongarch_lasx_xvmaxi_h:
  case Intrinsic::loongarch_lasx_xvmaxi_w:
  case Intrinsic::loongarch_lasx_xvmaxi_d:
    return DAG.getNode(ISD::SMAX, DL, N->getValueType(0), N->getOperand(1),
                       lowerVectorSplatImm<5>(N, 2, DAG, /*IsSigned=*/true));
  case Intrinsic::loongarch_lsx_vmaxi_bu:
  case Intrinsic::loongarch_lsx_vmaxi_hu:
  case Intrinsic::loongarch_lsx_vmaxi_wu:
  case Intrinsic::loongarch_lsx_vmaxi_du:
  case Intrinsic::loongarch_lasx_xvmaxi_bu:
  case Intrinsic::loongarch_lasx_xvmaxi_hu:
  case Intrinsic::loongarch_lasx_xvmaxi_wu:
  case Intrinsic::loongarch_lasx_xvmaxi_du:
    return DAG.getNode(ISD::UMAX, DL, N->getValueType(0), N->getOperand(1),
                       lowerVectorSplatImm<5>(N, 2, DAG));
  case Intrinsic::loongarch_lsx_vmin_b:
  case Intrinsic::loongarch_lsx_vmin_h:
  case Intrinsic::loongarch_lsx_vmin_w:
  case Intrinsic::loongarch_lsx_vmin_d:
  case Intrinsic::loongarch_lasx_xvmin_b:
  case Intrinsic::loongarch_lasx_xvmin_h:
  case Intrinsic::loongarch_lasx_xvmin_w:
  case Intrinsic::loongarch_lasx_xvmin_d:
    return DAG.getNode(ISD::SMIN, DL, N->getValueType(0), N->getOperand(1),
                       N->getOperand(2));
  case Intrinsic::loongarch_lsx_vmin_bu:
  case Intrinsic::loongarch_lsx_vmin_hu:
  case Intrinsic::loongarch_lsx_vmin_wu:
  case Intrinsic::loongarch_lsx_vmin_du:
  case Intrinsic::loongarch_lasx_xvmin_bu:
  case Intrinsic::loongarch_lasx_xvmin_hu:
  case Intrinsic::loongarch_lasx_xvmin_wu:
  case Intrinsic::loongarch_lasx_xvmin_du:
    return DAG.getNode(ISD::UMIN, DL, N->getValueType(0), N->getOperand(1),
                       N->getOperand(2));
  case Intrinsic::loongarch_lsx_vmini_b:
  case Intrinsic::loongarch_lsx_vmini_h:
  case Intrinsic::loongarch_lsx_vmini_w:
  case Intrinsic::loongarch_lsx_vmini_d:
  case Intrinsic::loongarch_lasx_xvmini_b:
  case Intrinsic::loongarch_lasx_xvmini_h:
  case Intrinsic::loongarch_lasx_xvmini_w:
  case Intrinsic::loongarch_lasx_xvmini_d:
    return DAG.getNode(ISD::SMIN, DL, N->getValueType(0), N->getOperand(1),
                       lowerVectorSplatImm<5>(N, 2, DAG, /*IsSigned=*/true));
  case Intrinsic::loongarch_lsx_vmini_bu:
  case Intrinsic::loongarch_lsx_vmini_hu:
  case Intrinsic::loongarch_lsx_vmini_wu:
  case Intrinsic::loongarch_lsx_vmini_du:
  case Intrinsic::loongarch_lasx_xvmini_bu:
  case Intrinsic::loongarch_lasx_xvmini_hu:
  case Intrinsic::loongarch_lasx_xvmini_wu:
  case Intrinsic::loongarch_lasx_xvmini_du:
    return DAG.getNode(ISD::UMIN, DL, N->getValueType(0), N->getOperand(1),
                       lowerVectorSplatImm<5>(N, 2, DAG));
  case Intrinsic::loongarch_lsx_vmul_b:
  case Intrinsic::loongarch_lsx_vmul_h:
  case Intrinsic::loongarch_lsx_vmul_w:
  case Intrinsic::loongarch_lsx_vmul_d:
  case Intrinsic::loongarch_lasx_xvmul_b:
  case Intrinsic::loongarch_lasx_xvmul_h:
  case Intrinsic::loongarch_lasx_xvmul_w:
  case Intrinsic::loongarch_lasx_xvmul_d:
    return DAG.getNode(ISD::MUL, DL, N->getValueType(0), N->getOperand(1),
                       N->getOperand(2));
  case Intrinsic::loongarch_lsx_vmadd_b:
  case Intrinsic::loongarch_lsx_vmadd_h:
  case Intrinsic::loongarch_lsx_vmadd_w:
  case Intrinsic::loongarch_lsx_vmadd_d:
  case Intrinsic::loongarch_lasx_xvmadd_b:
  case Intrinsic::loongarch_lasx_xvmadd_h:
  case Intrinsic::loongarch_lasx_xvmadd_w:
  case Intrinsic::loongarch_lasx_xvmadd_d: {
    EVT ResTy = N->getValueType(0);
    return DAG.getNode(ISD::ADD, SDLoc(N), ResTy, N->getOperand(1),
                       DAG.getNode(ISD::MUL, SDLoc(N), ResTy, N->getOperand(2),
                                   N->getOperand(3)));
  }
  case Intrinsic::loongarch_lsx_vmsub_b:
  case Intrinsic::loongarch_lsx_vmsub_h:
  case Intrinsic::loongarch_lsx_vmsub_w:
  case Intrinsic::loongarch_lsx_vmsub_d:
  case Intrinsic::loongarch_lasx_xvmsub_b:
  case Intrinsic::loongarch_lasx_xvmsub_h:
  case Intrinsic::loongarch_lasx_xvmsub_w:
  case Intrinsic::loongarch_lasx_xvmsub_d: {
    EVT ResTy = N->getValueType(0);
    return DAG.getNode(ISD::SUB, SDLoc(N), ResTy, N->getOperand(1),
                       DAG.getNode(ISD::MUL, SDLoc(N), ResTy, N->getOperand(2),
                                   N->getOperand(3)));
  }
  case Intrinsic::loongarch_lsx_vdiv_b:
  case Intrinsic::loongarch_lsx_vdiv_h:
  case Intrinsic::loongarch_lsx_vdiv_w:
  case Intrinsic::loongarch_lsx_vdiv_d:
  case Intrinsic::loongarch_lasx_xvdiv_b:
  case Intrinsic::loongarch_lasx_xvdiv_h:
  case Intrinsic::loongarch_lasx_xvdiv_w:
  case Intrinsic::loongarch_lasx_xvdiv_d:
    return DAG.getNode(ISD::SDIV, DL, N->getValueType(0), N->getOperand(1),
                       N->getOperand(2));
  case Intrinsic::loongarch_lsx_vdiv_bu:
  case Intrinsic::loongarch_lsx_vdiv_hu:
  case Intrinsic::loongarch_lsx_vdiv_wu:
  case Intrinsic::loongarch_lsx_vdiv_du:
  case Intrinsic::loongarch_lasx_xvdiv_bu:
  case Intrinsic::loongarch_lasx_xvdiv_hu:
  case Intrinsic::loongarch_lasx_xvdiv_wu:
  case Intrinsic::loongarch_lasx_xvdiv_du:
    return DAG.getNode(ISD::UDIV, DL, N->getValueType(0), N->getOperand(1),
                       N->getOperand(2));
  case Intrinsic::loongarch_lsx_vmod_b:
  case Intrinsic::loongarch_lsx_vmod_h:
  case Intrinsic::loongarch_lsx_vmod_w:
  case Intrinsic::loongarch_lsx_vmod_d:
  case Intrinsic::loongarch_lasx_xvmod_b:
  case Intrinsic::loongarch_lasx_xvmod_h:
  case Intrinsic::loongarch_lasx_xvmod_w:
  case Intrinsic::loongarch_lasx_xvmod_d:
    return DAG.getNode(ISD::SREM, DL, N->getValueType(0), N->getOperand(1),
                       N->getOperand(2));
  case Intrinsic::loongarch_lsx_vmod_bu:
  case Intrinsic::loongarch_lsx_vmod_hu:
  case Intrinsic::loongarch_lsx_vmod_wu:
  case Intrinsic::loongarch_lsx_vmod_du:
  case Intrinsic::loongarch_lasx_xvmod_bu:
  case Intrinsic::loongarch_lasx_xvmod_hu:
  case Intrinsic::loongarch_lasx_xvmod_wu:
  case Intrinsic::loongarch_lasx_xvmod_du:
    return DAG.getNode(ISD::UREM, DL, N->getValueType(0), N->getOperand(1),
                       N->getOperand(2));
  case Intrinsic::loongarch_lsx_vand_v:
  case Intrinsic::loongarch_lasx_xvand_v:
    return DAG.getNode(ISD::AND, DL, N->getValueType(0), N->getOperand(1),
                       N->getOperand(2));
  case Intrinsic::loongarch_lsx_vor_v:
  case Intrinsic::loongarch_lasx_xvor_v:
    return DAG.getNode(ISD::OR, DL, N->getValueType(0), N->getOperand(1),
                       N->getOperand(2));
  case Intrinsic::loongarch_lsx_vxor_v:
  case Intrinsic::loongarch_lasx_xvxor_v:
    return DAG.getNode(ISD::XOR, DL, N->getValueType(0), N->getOperand(1),
                       N->getOperand(2));
  case Intrinsic::loongarch_lsx_vnor_v:
  case Intrinsic::loongarch_lasx_xvnor_v: {
    SDValue Res = DAG.getNode(ISD::OR, DL, N->getValueType(0), N->getOperand(1),
                              N->getOperand(2));
    return DAG.getNOT(DL, Res, Res->getValueType(0));
  }
  case Intrinsic::loongarch_lsx_vandi_b:
  case Intrinsic::loongarch_lasx_xvandi_b:
    return DAG.getNode(ISD::AND, DL, N->getValueType(0), N->getOperand(1),
                       lowerVectorSplatImm<8>(N, 2, DAG));
  case Intrinsic::loongarch_lsx_vori_b:
  case Intrinsic::loongarch_lasx_xvori_b:
    return DAG.getNode(ISD::OR, DL, N->getValueType(0), N->getOperand(1),
                       lowerVectorSplatImm<8>(N, 2, DAG));
  case Intrinsic::loongarch_lsx_vxori_b:
  case Intrinsic::loongarch_lasx_xvxori_b:
    return DAG.getNode(ISD::XOR, DL, N->getValueType(0), N->getOperand(1),
                       lowerVectorSplatImm<8>(N, 2, DAG));
  case Intrinsic::loongarch_lsx_vsll_b:
  case Intrinsic::loongarch_lsx_vsll_h:
  case Intrinsic::loongarch_lsx_vsll_w:
  case Intrinsic::loongarch_lsx_vsll_d:
  case Intrinsic::loongarch_lasx_xvsll_b:
  case Intrinsic::loongarch_lasx_xvsll_h:
  case Intrinsic::loongarch_lasx_xvsll_w:
  case Intrinsic::loongarch_lasx_xvsll_d:
    return DAG.getNode(ISD::SHL, DL, N->getValueType(0), N->getOperand(1),
                       truncateVecElts(N, DAG));
  case Intrinsic::loongarch_lsx_vslli_b:
  case Intrinsic::loongarch_lasx_xvslli_b:
    return DAG.getNode(ISD::SHL, DL, N->getValueType(0), N->getOperand(1),
                       lowerVectorSplatImm<3>(N, 2, DAG));
  case Intrinsic::loongarch_lsx_vslli_h:
  case Intrinsic::loongarch_lasx_xvslli_h:
    return DAG.getNode(ISD::SHL, DL, N->getValueType(0), N->getOperand(1),
                       lowerVectorSplatImm<4>(N, 2, DAG));
  case Intrinsic::loongarch_lsx_vslli_w:
  case Intrinsic::loongarch_lasx_xvslli_w:
    return DAG.getNode(ISD::SHL, DL, N->getValueType(0), N->getOperand(1),
                       lowerVectorSplatImm<5>(N, 2, DAG));
  case Intrinsic::loongarch_lsx_vslli_d:
  case Intrinsic::loongarch_lasx_xvslli_d:
    return DAG.getNode(ISD::SHL, DL, N->getValueType(0), N->getOperand(1),
                       lowerVectorSplatImm<6>(N, 2, DAG));
  case Intrinsic::loongarch_lsx_vsrl_b:
  case Intrinsic::loongarch_lsx_vsrl_h:
  case Intrinsic::loongarch_lsx_vsrl_w:
  case Intrinsic::loongarch_lsx_vsrl_d:
  case Intrinsic::loongarch_lasx_xvsrl_b:
  case Intrinsic::loongarch_lasx_xvsrl_h:
  case Intrinsic::loongarch_lasx_xvsrl_w:
  case Intrinsic::loongarch_lasx_xvsrl_d:
    return DAG.getNode(ISD::SRL, DL, N->getValueType(0), N->getOperand(1),
                       truncateVecElts(N, DAG));
  case Intrinsic::loongarch_lsx_vsrli_b:
  case Intrinsic::loongarch_lasx_xvsrli_b:
    return DAG.getNode(ISD::SRL, DL, N->getValueType(0), N->getOperand(1),
                       lowerVectorSplatImm<3>(N, 2, DAG));
  case Intrinsic::loongarch_lsx_vsrli_h:
  case Intrinsic::loongarch_lasx_xvsrli_h:
    return DAG.getNode(ISD::SRL, DL, N->getValueType(0), N->getOperand(1),
                       lowerVectorSplatImm<4>(N, 2, DAG));
  case Intrinsic::loongarch_lsx_vsrli_w:
  case Intrinsic::loongarch_lasx_xvsrli_w:
    return DAG.getNode(ISD::SRL, DL, N->getValueType(0), N->getOperand(1),
                       lowerVectorSplatImm<5>(N, 2, DAG));
  case Intrinsic::loongarch_lsx_vsrli_d:
  case Intrinsic::loongarch_lasx_xvsrli_d:
    return DAG.getNode(ISD::SRL, DL, N->getValueType(0), N->getOperand(1),
                       lowerVectorSplatImm<6>(N, 2, DAG));
  case Intrinsic::loongarch_lsx_vsra_b:
  case Intrinsic::loongarch_lsx_vsra_h:
  case Intrinsic::loongarch_lsx_vsra_w:
  case Intrinsic::loongarch_lsx_vsra_d:
  case Intrinsic::loongarch_lasx_xvsra_b:
  case Intrinsic::loongarch_lasx_xvsra_h:
  case Intrinsic::loongarch_lasx_xvsra_w:
  case Intrinsic::loongarch_lasx_xvsra_d:
    return DAG.getNode(ISD::SRA, DL, N->getValueType(0), N->getOperand(1),
                       truncateVecElts(N, DAG));
  case Intrinsic::loongarch_lsx_vsrai_b:
  case Intrinsic::loongarch_lasx_xvsrai_b:
    return DAG.getNode(ISD::SRA, DL, N->getValueType(0), N->getOperand(1),
                       lowerVectorSplatImm<3>(N, 2, DAG));
  case Intrinsic::loongarch_lsx_vsrai_h:
  case Intrinsic::loongarch_lasx_xvsrai_h:
    return DAG.getNode(ISD::SRA, DL, N->getValueType(0), N->getOperand(1),
                       lowerVectorSplatImm<4>(N, 2, DAG));
  case Intrinsic::loongarch_lsx_vsrai_w:
  case Intrinsic::loongarch_lasx_xvsrai_w:
    return DAG.getNode(ISD::SRA, DL, N->getValueType(0), N->getOperand(1),
                       lowerVectorSplatImm<5>(N, 2, DAG));
  case Intrinsic::loongarch_lsx_vsrai_d:
  case Intrinsic::loongarch_lasx_xvsrai_d:
    return DAG.getNode(ISD::SRA, DL, N->getValueType(0), N->getOperand(1),
                       lowerVectorSplatImm<6>(N, 2, DAG));
  case Intrinsic::loongarch_lsx_vpcnt_b:
  case Intrinsic::loongarch_lsx_vpcnt_h:
  case Intrinsic::loongarch_lsx_vpcnt_w:
  case Intrinsic::loongarch_lsx_vpcnt_d:
  case Intrinsic::loongarch_lasx_xvpcnt_b:
  case Intrinsic::loongarch_lasx_xvpcnt_h:
  case Intrinsic::loongarch_lasx_xvpcnt_w:
  case Intrinsic::loongarch_lasx_xvpcnt_d:
    return DAG.getNode(ISD::CTPOP, DL, N->getValueType(0), N->getOperand(1));
  case Intrinsic::loongarch_lsx_vbitclr_b:
  case Intrinsic::loongarch_lsx_vbitclr_h:
  case Intrinsic::loongarch_lsx_vbitclr_w:
  case Intrinsic::loongarch_lsx_vbitclr_d:
  case Intrinsic::loongarch_lasx_xvbitclr_b:
  case Intrinsic::loongarch_lasx_xvbitclr_h:
  case Intrinsic::loongarch_lasx_xvbitclr_w:
  case Intrinsic::loongarch_lasx_xvbitclr_d:
    return lowerVectorBitClear(N, DAG);
  case Intrinsic::loongarch_lsx_vbitclri_b:
  case Intrinsic::loongarch_lasx_xvbitclri_b:
    return lowerVectorBitClearImm<3>(N, DAG);
  case Intrinsic::loongarch_lsx_vbitclri_h:
  case Intrinsic::loongarch_lasx_xvbitclri_h:
    return lowerVectorBitClearImm<4>(N, DAG);
  case Intrinsic::loongarch_lsx_vbitclri_w:
  case Intrinsic::loongarch_lasx_xvbitclri_w:
    return lowerVectorBitClearImm<5>(N, DAG);
  case Intrinsic::loongarch_lsx_vbitclri_d:
  case Intrinsic::loongarch_lasx_xvbitclri_d:
    return lowerVectorBitClearImm<6>(N, DAG);
  case Intrinsic::loongarch_lsx_vbitset_b:
  case Intrinsic::loongarch_lsx_vbitset_h:
  case Intrinsic::loongarch_lsx_vbitset_w:
  case Intrinsic::loongarch_lsx_vbitset_d:
  case Intrinsic::loongarch_lasx_xvbitset_b:
  case Intrinsic::loongarch_lasx_xvbitset_h:
  case Intrinsic::loongarch_lasx_xvbitset_w:
  case Intrinsic::loongarch_lasx_xvbitset_d: {
    EVT VecTy = N->getValueType(0);
    SDValue One = DAG.getConstant(1, DL, VecTy);
    return DAG.getNode(
        ISD::OR, DL, VecTy, N->getOperand(1),
        DAG.getNode(ISD::SHL, DL, VecTy, One, truncateVecElts(N, DAG)));
  }
  case Intrinsic::loongarch_lsx_vbitseti_b:
  case Intrinsic::loongarch_lasx_xvbitseti_b:
    return lowerVectorBitSetImm<3>(N, DAG);
  case Intrinsic::loongarch_lsx_vbitseti_h:
  case Intrinsic::loongarch_lasx_xvbitseti_h:
    return lowerVectorBitSetImm<4>(N, DAG);
  case Intrinsic::loongarch_lsx_vbitseti_w:
  case Intrinsic::loongarch_lasx_xvbitseti_w:
    return lowerVectorBitSetImm<5>(N, DAG);
  case Intrinsic::loongarch_lsx_vbitseti_d:
  case Intrinsic::loongarch_lasx_xvbitseti_d:
    return lowerVectorBitSetImm<6>(N, DAG);
  case Intrinsic::loongarch_lsx_vbitrev_b:
  case Intrinsic::loongarch_lsx_vbitrev_h:
  case Intrinsic::loongarch_lsx_vbitrev_w:
  case Intrinsic::loongarch_lsx_vbitrev_d:
  case Intrinsic::loongarch_lasx_xvbitrev_b:
  case Intrinsic::loongarch_lasx_xvbitrev_h:
  case Intrinsic::loongarch_lasx_xvbitrev_w:
  case Intrinsic::loongarch_lasx_xvbitrev_d: {
    EVT VecTy = N->getValueType(0);
    SDValue One = DAG.getConstant(1, DL, VecTy);
    return DAG.getNode(
        ISD::XOR, DL, VecTy, N->getOperand(1),
        DAG.getNode(ISD::SHL, DL, VecTy, One, truncateVecElts(N, DAG)));
  }
  case Intrinsic::loongarch_lsx_vbitrevi_b:
  case Intrinsic::loongarch_lasx_xvbitrevi_b:
    return lowerVectorBitRevImm<3>(N, DAG);
  case Intrinsic::loongarch_lsx_vbitrevi_h:
  case Intrinsic::loongarch_lasx_xvbitrevi_h:
    return lowerVectorBitRevImm<4>(N, DAG);
  case Intrinsic::loongarch_lsx_vbitrevi_w:
  case Intrinsic::loongarch_lasx_xvbitrevi_w:
    return lowerVectorBitRevImm<5>(N, DAG);
  case Intrinsic::loongarch_lsx_vbitrevi_d:
  case Intrinsic::loongarch_lasx_xvbitrevi_d:
    return lowerVectorBitRevImm<6>(N, DAG);
  case Intrinsic::loongarch_lsx_vfadd_s:
  case Intrinsic::loongarch_lsx_vfadd_d:
  case Intrinsic::loongarch_lasx_xvfadd_s:
  case Intrinsic::loongarch_lasx_xvfadd_d:
    return DAG.getNode(ISD::FADD, DL, N->getValueType(0), N->getOperand(1),
                       N->getOperand(2));
  case Intrinsic::loongarch_lsx_vfsub_s:
  case Intrinsic::loongarch_lsx_vfsub_d:
  case Intrinsic::loongarch_lasx_xvfsub_s:
  case Intrinsic::loongarch_lasx_xvfsub_d:
    return DAG.getNode(ISD::FSUB, DL, N->getValueType(0), N->getOperand(1),
                       N->getOperand(2));
  case Intrinsic::loongarch_lsx_vfmul_s:
  case Intrinsic::loongarch_lsx_vfmul_d:
  case Intrinsic::loongarch_lasx_xvfmul_s:
  case Intrinsic::loongarch_lasx_xvfmul_d:
    return DAG.getNode(ISD::FMUL, DL, N->getValueType(0), N->getOperand(1),
                       N->getOperand(2));
  case Intrinsic::loongarch_lsx_vfdiv_s:
  case Intrinsic::loongarch_lsx_vfdiv_d:
  case Intrinsic::loongarch_lasx_xvfdiv_s:
  case Intrinsic::loongarch_lasx_xvfdiv_d:
    return DAG.getNode(ISD::FDIV, DL, N->getValueType(0), N->getOperand(1),
                       N->getOperand(2));
  case Intrinsic::loongarch_lsx_vfmadd_s:
  case Intrinsic::loongarch_lsx_vfmadd_d:
  case Intrinsic::loongarch_lasx_xvfmadd_s:
  case Intrinsic::loongarch_lasx_xvfmadd_d:
    return DAG.getNode(ISD::FMA, DL, N->getValueType(0), N->getOperand(1),
                       N->getOperand(2), N->getOperand(3));
  case Intrinsic::loongarch_lsx_vinsgr2vr_b:
    return DAG.getNode(ISD::INSERT_VECTOR_ELT, SDLoc(N), N->getValueType(0),
                       N->getOperand(1), N->getOperand(2),
                       legalizeIntrinsicImmArg<4>(N, 3, DAG, Subtarget));
  case Intrinsic::loongarch_lsx_vinsgr2vr_h:
  case Intrinsic::loongarch_lasx_xvinsgr2vr_w:
    return DAG.getNode(ISD::INSERT_VECTOR_ELT, SDLoc(N), N->getValueType(0),
                       N->getOperand(1), N->getOperand(2),
                       legalizeIntrinsicImmArg<3>(N, 3, DAG, Subtarget));
  case Intrinsic::loongarch_lsx_vinsgr2vr_w:
  case Intrinsic::loongarch_lasx_xvinsgr2vr_d:
    return DAG.getNode(ISD::INSERT_VECTOR_ELT, SDLoc(N), N->getValueType(0),
                       N->getOperand(1), N->getOperand(2),
                       legalizeIntrinsicImmArg<2>(N, 3, DAG, Subtarget));
  case Intrinsic::loongarch_lsx_vinsgr2vr_d:
    return DAG.getNode(ISD::INSERT_VECTOR_ELT, SDLoc(N), N->getValueType(0),
                       N->getOperand(1), N->getOperand(2),
                       legalizeIntrinsicImmArg<1>(N, 3, DAG, Subtarget));
  case Intrinsic::loongarch_lsx_vreplgr2vr_b:
  case Intrinsic::loongarch_lsx_vreplgr2vr_h:
  case Intrinsic::loongarch_lsx_vreplgr2vr_w:
  case Intrinsic::loongarch_lsx_vreplgr2vr_d:
  case Intrinsic::loongarch_lasx_xvreplgr2vr_b:
  case Intrinsic::loongarch_lasx_xvreplgr2vr_h:
  case Intrinsic::loongarch_lasx_xvreplgr2vr_w:
  case Intrinsic::loongarch_lasx_xvreplgr2vr_d: {
    EVT ResTy = N->getValueType(0);
    SmallVector<SDValue> Ops(ResTy.getVectorNumElements(), N->getOperand(1));
    return DAG.getBuildVector(ResTy, DL, Ops);
  }
  case Intrinsic::loongarch_lsx_vreplve_b:
  case Intrinsic::loongarch_lsx_vreplve_h:
  case Intrinsic::loongarch_lsx_vreplve_w:
  case Intrinsic::loongarch_lsx_vreplve_d:
  case Intrinsic::loongarch_lasx_xvreplve_b:
  case Intrinsic::loongarch_lasx_xvreplve_h:
  case Intrinsic::loongarch_lasx_xvreplve_w:
  case Intrinsic::loongarch_lasx_xvreplve_d:
    return DAG.getNode(LoongArchISD::VREPLVE, DL, N->getValueType(0),
                       N->getOperand(1),
                       DAG.getNode(ISD::ANY_EXTEND, DL, Subtarget.getGRLenVT(),
                                   N->getOperand(2)));
  }
  return SDValue();
}

SDValue LoongArchTargetLowering::PerformDAGCombine(SDNode *N,
                                                   DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  switch (N->getOpcode()) {
  default:
    break;
  case ISD::AND:
    return performANDCombine(N, DAG, DCI, Subtarget);
  case ISD::OR:
    return performORCombine(N, DAG, DCI, Subtarget);
  case ISD::SRL:
    return performSRLCombine(N, DAG, DCI, Subtarget);
  case LoongArchISD::BITREV_W:
    return performBITREV_WCombine(N, DAG, DCI, Subtarget);
  case ISD::INTRINSIC_WO_CHAIN:
    return performINTRINSIC_WO_CHAINCombine(N, DAG, DCI, Subtarget);
  }
  return SDValue();
}

static MachineBasicBlock *insertDivByZeroTrap(MachineInstr &MI,
                                              MachineBasicBlock *MBB) {
  if (!ZeroDivCheck)
    return MBB;

  // Build instructions:
  // MBB:
  //   div(or mod)   $dst, $dividend, $divisor
  //   bnez          $divisor, SinkMBB
  // BreakMBB:
  //   break         7 // BRK_DIVZERO
  // SinkMBB:
  //   fallthrough
  const BasicBlock *LLVM_BB = MBB->getBasicBlock();
  MachineFunction::iterator It = ++MBB->getIterator();
  MachineFunction *MF = MBB->getParent();
  auto BreakMBB = MF->CreateMachineBasicBlock(LLVM_BB);
  auto SinkMBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MF->insert(It, BreakMBB);
  MF->insert(It, SinkMBB);

  // Transfer the remainder of MBB and its successor edges to SinkMBB.
  SinkMBB->splice(SinkMBB->end(), MBB, std::next(MI.getIterator()), MBB->end());
  SinkMBB->transferSuccessorsAndUpdatePHIs(MBB);

  const TargetInstrInfo &TII = *MF->getSubtarget().getInstrInfo();
  DebugLoc DL = MI.getDebugLoc();
  MachineOperand &Divisor = MI.getOperand(2);
  Register DivisorReg = Divisor.getReg();

  // MBB:
  BuildMI(MBB, DL, TII.get(LoongArch::BNEZ))
      .addReg(DivisorReg, getKillRegState(Divisor.isKill()))
      .addMBB(SinkMBB);
  MBB->addSuccessor(BreakMBB);
  MBB->addSuccessor(SinkMBB);

  // BreakMBB:
  // See linux header file arch/loongarch/include/uapi/asm/break.h for the
  // definition of BRK_DIVZERO.
  BuildMI(BreakMBB, DL, TII.get(LoongArch::BREAK)).addImm(7 /*BRK_DIVZERO*/);
  BreakMBB->addSuccessor(SinkMBB);

  // Clear Divisor's kill flag.
  Divisor.setIsKill(false);

  return SinkMBB;
}

static MachineBasicBlock *
emitVecCondBranchPseudo(MachineInstr &MI, MachineBasicBlock *BB,
                        const LoongArchSubtarget &Subtarget) {
  unsigned CondOpc;
  switch (MI.getOpcode()) {
  default:
    llvm_unreachable("Unexpected opcode");
  case LoongArch::PseudoVBZ:
    CondOpc = LoongArch::VSETEQZ_V;
    break;
  case LoongArch::PseudoVBZ_B:
    CondOpc = LoongArch::VSETANYEQZ_B;
    break;
  case LoongArch::PseudoVBZ_H:
    CondOpc = LoongArch::VSETANYEQZ_H;
    break;
  case LoongArch::PseudoVBZ_W:
    CondOpc = LoongArch::VSETANYEQZ_W;
    break;
  case LoongArch::PseudoVBZ_D:
    CondOpc = LoongArch::VSETANYEQZ_D;
    break;
  case LoongArch::PseudoVBNZ:
    CondOpc = LoongArch::VSETNEZ_V;
    break;
  case LoongArch::PseudoVBNZ_B:
    CondOpc = LoongArch::VSETALLNEZ_B;
    break;
  case LoongArch::PseudoVBNZ_H:
    CondOpc = LoongArch::VSETALLNEZ_H;
    break;
  case LoongArch::PseudoVBNZ_W:
    CondOpc = LoongArch::VSETALLNEZ_W;
    break;
  case LoongArch::PseudoVBNZ_D:
    CondOpc = LoongArch::VSETALLNEZ_D;
    break;
  case LoongArch::PseudoXVBZ:
    CondOpc = LoongArch::XVSETEQZ_V;
    break;
  case LoongArch::PseudoXVBZ_B:
    CondOpc = LoongArch::XVSETANYEQZ_B;
    break;
  case LoongArch::PseudoXVBZ_H:
    CondOpc = LoongArch::XVSETANYEQZ_H;
    break;
  case LoongArch::PseudoXVBZ_W:
    CondOpc = LoongArch::XVSETANYEQZ_W;
    break;
  case LoongArch::PseudoXVBZ_D:
    CondOpc = LoongArch::XVSETANYEQZ_D;
    break;
  case LoongArch::PseudoXVBNZ:
    CondOpc = LoongArch::XVSETNEZ_V;
    break;
  case LoongArch::PseudoXVBNZ_B:
    CondOpc = LoongArch::XVSETALLNEZ_B;
    break;
  case LoongArch::PseudoXVBNZ_H:
    CondOpc = LoongArch::XVSETALLNEZ_H;
    break;
  case LoongArch::PseudoXVBNZ_W:
    CondOpc = LoongArch::XVSETALLNEZ_W;
    break;
  case LoongArch::PseudoXVBNZ_D:
    CondOpc = LoongArch::XVSETALLNEZ_D;
    break;
  }

  const TargetInstrInfo *TII = Subtarget.getInstrInfo();
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  DebugLoc DL = MI.getDebugLoc();
  MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();
  MachineFunction::iterator It = ++BB->getIterator();

  MachineFunction *F = BB->getParent();
  MachineBasicBlock *FalseBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *TrueBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *SinkBB = F->CreateMachineBasicBlock(LLVM_BB);

  F->insert(It, FalseBB);
  F->insert(It, TrueBB);
  F->insert(It, SinkBB);

  // Transfer the remainder of MBB and its successor edges to Sink.
  SinkBB->splice(SinkBB->end(), BB, std::next(MI.getIterator()), BB->end());
  SinkBB->transferSuccessorsAndUpdatePHIs(BB);

  // Insert the real instruction to BB.
  Register FCC = MRI.createVirtualRegister(&LoongArch::CFRRegClass);
  BuildMI(BB, DL, TII->get(CondOpc), FCC).addReg(MI.getOperand(1).getReg());

  // Insert branch.
  BuildMI(BB, DL, TII->get(LoongArch::BCNEZ)).addReg(FCC).addMBB(TrueBB);
  BB->addSuccessor(FalseBB);
  BB->addSuccessor(TrueBB);

  // FalseBB.
  Register RD1 = MRI.createVirtualRegister(&LoongArch::GPRRegClass);
  BuildMI(FalseBB, DL, TII->get(LoongArch::ADDI_W), RD1)
      .addReg(LoongArch::R0)
      .addImm(0);
  BuildMI(FalseBB, DL, TII->get(LoongArch::PseudoBR)).addMBB(SinkBB);
  FalseBB->addSuccessor(SinkBB);

  // TrueBB.
  Register RD2 = MRI.createVirtualRegister(&LoongArch::GPRRegClass);
  BuildMI(TrueBB, DL, TII->get(LoongArch::ADDI_W), RD2)
      .addReg(LoongArch::R0)
      .addImm(1);
  TrueBB->addSuccessor(SinkBB);

  // SinkBB: merge the results.
  BuildMI(*SinkBB, SinkBB->begin(), DL, TII->get(LoongArch::PHI),
          MI.getOperand(0).getReg())
      .addReg(RD1)
      .addMBB(FalseBB)
      .addReg(RD2)
      .addMBB(TrueBB);

  // The pseudo instruction is gone now.
  MI.eraseFromParent();
  return SinkBB;
}

MachineBasicBlock *LoongArchTargetLowering::EmitInstrWithCustomInserter(
    MachineInstr &MI, MachineBasicBlock *BB) const {
  const TargetInstrInfo *TII = Subtarget.getInstrInfo();
  DebugLoc DL = MI.getDebugLoc();

  switch (MI.getOpcode()) {
  default:
    llvm_unreachable("Unexpected instr type to insert");
  case LoongArch::DIV_W:
  case LoongArch::DIV_WU:
  case LoongArch::MOD_W:
  case LoongArch::MOD_WU:
  case LoongArch::DIV_D:
  case LoongArch::DIV_DU:
  case LoongArch::MOD_D:
  case LoongArch::MOD_DU:
    return insertDivByZeroTrap(MI, BB);
    break;
  case LoongArch::WRFCSR: {
    BuildMI(*BB, MI, DL, TII->get(LoongArch::MOVGR2FCSR),
            LoongArch::FCSR0 + MI.getOperand(0).getImm())
        .addReg(MI.getOperand(1).getReg());
    MI.eraseFromParent();
    return BB;
  }
  case LoongArch::RDFCSR: {
    MachineInstr *ReadFCSR =
        BuildMI(*BB, MI, DL, TII->get(LoongArch::MOVFCSR2GR),
                MI.getOperand(0).getReg())
            .addReg(LoongArch::FCSR0 + MI.getOperand(1).getImm());
    ReadFCSR->getOperand(1).setIsUndef();
    MI.eraseFromParent();
    return BB;
  }
  case LoongArch::PseudoVBZ:
  case LoongArch::PseudoVBZ_B:
  case LoongArch::PseudoVBZ_H:
  case LoongArch::PseudoVBZ_W:
  case LoongArch::PseudoVBZ_D:
  case LoongArch::PseudoVBNZ:
  case LoongArch::PseudoVBNZ_B:
  case LoongArch::PseudoVBNZ_H:
  case LoongArch::PseudoVBNZ_W:
  case LoongArch::PseudoVBNZ_D:
  case LoongArch::PseudoXVBZ:
  case LoongArch::PseudoXVBZ_B:
  case LoongArch::PseudoXVBZ_H:
  case LoongArch::PseudoXVBZ_W:
  case LoongArch::PseudoXVBZ_D:
  case LoongArch::PseudoXVBNZ:
  case LoongArch::PseudoXVBNZ_B:
  case LoongArch::PseudoXVBNZ_H:
  case LoongArch::PseudoXVBNZ_W:
  case LoongArch::PseudoXVBNZ_D:
    return emitVecCondBranchPseudo(MI, BB, Subtarget);
  }
}

bool LoongArchTargetLowering::allowsMisalignedMemoryAccesses(
    EVT VT, unsigned AddrSpace, Align Alignment, MachineMemOperand::Flags Flags,
    unsigned *Fast) const {
  if (!Subtarget.hasUAL())
    return false;

  // TODO: set reasonable speed number.
  if (Fast)
    *Fast = 1;
  return true;
}

const char *LoongArchTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch ((LoongArchISD::NodeType)Opcode) {
  case LoongArchISD::FIRST_NUMBER:
    break;

#define NODE_NAME_CASE(node)                                                   \
  case LoongArchISD::node:                                                     \
    return "LoongArchISD::" #node;

    // TODO: Add more target-dependent nodes later.
    NODE_NAME_CASE(CALL)
    NODE_NAME_CASE(RET)
    NODE_NAME_CASE(TAIL)
    NODE_NAME_CASE(SLL_W)
    NODE_NAME_CASE(SRA_W)
    NODE_NAME_CASE(SRL_W)
    NODE_NAME_CASE(BSTRINS)
    NODE_NAME_CASE(BSTRPICK)
    NODE_NAME_CASE(MOVGR2FR_W_LA64)
    NODE_NAME_CASE(MOVFR2GR_S_LA64)
    NODE_NAME_CASE(FTINT)
    NODE_NAME_CASE(REVB_2H)
    NODE_NAME_CASE(REVB_2W)
    NODE_NAME_CASE(BITREV_4B)
    NODE_NAME_CASE(BITREV_W)
    NODE_NAME_CASE(ROTR_W)
    NODE_NAME_CASE(ROTL_W)
    NODE_NAME_CASE(CLZ_W)
    NODE_NAME_CASE(CTZ_W)
    NODE_NAME_CASE(DBAR)
    NODE_NAME_CASE(IBAR)
    NODE_NAME_CASE(BREAK)
    NODE_NAME_CASE(SYSCALL)
    NODE_NAME_CASE(CRC_W_B_W)
    NODE_NAME_CASE(CRC_W_H_W)
    NODE_NAME_CASE(CRC_W_W_W)
    NODE_NAME_CASE(CRC_W_D_W)
    NODE_NAME_CASE(CRCC_W_B_W)
    NODE_NAME_CASE(CRCC_W_H_W)
    NODE_NAME_CASE(CRCC_W_W_W)
    NODE_NAME_CASE(CRCC_W_D_W)
    NODE_NAME_CASE(CSRRD)
    NODE_NAME_CASE(CSRWR)
    NODE_NAME_CASE(CSRXCHG)
    NODE_NAME_CASE(IOCSRRD_B)
    NODE_NAME_CASE(IOCSRRD_H)
    NODE_NAME_CASE(IOCSRRD_W)
    NODE_NAME_CASE(IOCSRRD_D)
    NODE_NAME_CASE(IOCSRWR_B)
    NODE_NAME_CASE(IOCSRWR_H)
    NODE_NAME_CASE(IOCSRWR_W)
    NODE_NAME_CASE(IOCSRWR_D)
    NODE_NAME_CASE(CPUCFG)
    NODE_NAME_CASE(MOVGR2FCSR)
    NODE_NAME_CASE(MOVFCSR2GR)
    NODE_NAME_CASE(CACOP_D)
    NODE_NAME_CASE(CACOP_W)
    NODE_NAME_CASE(VPICK_SEXT_ELT)
    NODE_NAME_CASE(VPICK_ZEXT_ELT)
    NODE_NAME_CASE(VREPLVE)
    NODE_NAME_CASE(VALL_ZERO)
    NODE_NAME_CASE(VANY_ZERO)
    NODE_NAME_CASE(VALL_NONZERO)
    NODE_NAME_CASE(VANY_NONZERO)
  }
#undef NODE_NAME_CASE
  return nullptr;
}

//===----------------------------------------------------------------------===//
//                     Calling Convention Implementation
//===----------------------------------------------------------------------===//

// Eight general-purpose registers a0-a7 used for passing integer arguments,
// with a0-a1 reused to return values. Generally, the GPRs are used to pass
// fixed-point arguments, and floating-point arguments when no FPR is available
// or with soft float ABI.
const MCPhysReg ArgGPRs[] = {LoongArch::R4,  LoongArch::R5, LoongArch::R6,
                             LoongArch::R7,  LoongArch::R8, LoongArch::R9,
                             LoongArch::R10, LoongArch::R11};
// Eight floating-point registers fa0-fa7 used for passing floating-point
// arguments, and fa0-fa1 are also used to return values.
const MCPhysReg ArgFPR32s[] = {LoongArch::F0, LoongArch::F1, LoongArch::F2,
                               LoongArch::F3, LoongArch::F4, LoongArch::F5,
                               LoongArch::F6, LoongArch::F7};
// FPR32 and FPR64 alias each other.
const MCPhysReg ArgFPR64s[] = {
    LoongArch::F0_64, LoongArch::F1_64, LoongArch::F2_64, LoongArch::F3_64,
    LoongArch::F4_64, LoongArch::F5_64, LoongArch::F6_64, LoongArch::F7_64};

const MCPhysReg ArgVRs[] = {LoongArch::VR0, LoongArch::VR1, LoongArch::VR2,
                            LoongArch::VR3, LoongArch::VR4, LoongArch::VR5,
                            LoongArch::VR6, LoongArch::VR7};

const MCPhysReg ArgXRs[] = {LoongArch::XR0, LoongArch::XR1, LoongArch::XR2,
                            LoongArch::XR3, LoongArch::XR4, LoongArch::XR5,
                            LoongArch::XR6, LoongArch::XR7};

// Pass a 2*GRLen argument that has been split into two GRLen values through
// registers or the stack as necessary.
static bool CC_LoongArchAssign2GRLen(unsigned GRLen, CCState &State,
                                     CCValAssign VA1, ISD::ArgFlagsTy ArgFlags1,
                                     unsigned ValNo2, MVT ValVT2, MVT LocVT2,
                                     ISD::ArgFlagsTy ArgFlags2) {
  unsigned GRLenInBytes = GRLen / 8;
  if (Register Reg = State.AllocateReg(ArgGPRs)) {
    // At least one half can be passed via register.
    State.addLoc(CCValAssign::getReg(VA1.getValNo(), VA1.getValVT(), Reg,
                                     VA1.getLocVT(), CCValAssign::Full));
  } else {
    // Both halves must be passed on the stack, with proper alignment.
    Align StackAlign =
        std::max(Align(GRLenInBytes), ArgFlags1.getNonZeroOrigAlign());
    State.addLoc(
        CCValAssign::getMem(VA1.getValNo(), VA1.getValVT(),
                            State.AllocateStack(GRLenInBytes, StackAlign),
                            VA1.getLocVT(), CCValAssign::Full));
    State.addLoc(CCValAssign::getMem(
        ValNo2, ValVT2, State.AllocateStack(GRLenInBytes, Align(GRLenInBytes)),
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
        ValNo2, ValVT2, State.AllocateStack(GRLenInBytes, Align(GRLenInBytes)),
        LocVT2, CCValAssign::Full));
  }
  return false;
}

// Implements the LoongArch calling convention. Returns true upon failure.
static bool CC_LoongArch(const DataLayout &DL, LoongArchABI::ABI ABI,
                         unsigned ValNo, MVT ValVT,
                         CCValAssign::LocInfo LocInfo, ISD::ArgFlagsTy ArgFlags,
                         CCState &State, bool IsFixed, bool IsRet,
                         Type *OrigTy) {
  unsigned GRLen = DL.getLargestLegalIntTypeSizeInBits();
  assert((GRLen == 32 || GRLen == 64) && "Unspport GRLen");
  MVT GRLenVT = GRLen == 32 ? MVT::i32 : MVT::i64;
  MVT LocVT = ValVT;

  // Any return value split into more than two values can't be returned
  // directly.
  if (IsRet && ValNo > 1)
    return true;

  // If passing a variadic argument, or if no FPR is available.
  bool UseGPRForFloat = true;

  switch (ABI) {
  default:
    llvm_unreachable("Unexpected ABI");
  case LoongArchABI::ABI_ILP32S:
  case LoongArchABI::ABI_ILP32F:
  case LoongArchABI::ABI_LP64F:
    report_fatal_error("Unimplemented ABI");
    break;
  case LoongArchABI::ABI_ILP32D:
  case LoongArchABI::ABI_LP64D:
    UseGPRForFloat = !IsFixed;
    break;
  case LoongArchABI::ABI_LP64S:
    break;
  }

  // FPR32 and FPR64 alias each other.
  if (State.getFirstUnallocated(ArgFPR32s) == std::size(ArgFPR32s))
    UseGPRForFloat = true;

  if (UseGPRForFloat && ValVT == MVT::f32) {
    LocVT = GRLenVT;
    LocInfo = CCValAssign::BCvt;
  } else if (UseGPRForFloat && GRLen == 64 && ValVT == MVT::f64) {
    LocVT = MVT::i64;
    LocInfo = CCValAssign::BCvt;
  } else if (UseGPRForFloat && GRLen == 32 && ValVT == MVT::f64) {
    // TODO: Handle passing f64 on LA32 with D feature.
    report_fatal_error("Passing f64 with GPR on LA32 is undefined");
  }

  // If this is a variadic argument, the LoongArch calling convention requires
  // that it is assigned an 'even' or 'aligned' register if it has (2*GRLen)/8
  // byte alignment. An aligned register should be used regardless of whether
  // the original argument was split during legalisation or not. The argument
  // will not be passed by registers if the original type is larger than
  // 2*GRLen, so the register alignment rule does not apply.
  unsigned TwoGRLenInBytes = (2 * GRLen) / 8;
  if (!IsFixed && ArgFlags.getNonZeroOrigAlign() == TwoGRLenInBytes &&
      DL.getTypeAllocSize(OrigTy) == TwoGRLenInBytes) {
    unsigned RegIdx = State.getFirstUnallocated(ArgGPRs);
    // Skip 'odd' register if necessary.
    if (RegIdx != std::size(ArgGPRs) && RegIdx % 2 == 1)
      State.AllocateReg(ArgGPRs);
  }

  SmallVectorImpl<CCValAssign> &PendingLocs = State.getPendingLocs();
  SmallVectorImpl<ISD::ArgFlagsTy> &PendingArgFlags =
      State.getPendingArgFlags();

  assert(PendingLocs.size() == PendingArgFlags.size() &&
         "PendingLocs and PendingArgFlags out of sync");

  // Split arguments might be passed indirectly, so keep track of the pending
  // values.
  if (ValVT.isScalarInteger() && (ArgFlags.isSplit() || !PendingLocs.empty())) {
    LocVT = GRLenVT;
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
  if (ValVT.isScalarInteger() && ArgFlags.isSplitEnd() &&
      PendingLocs.size() <= 2) {
    assert(PendingLocs.size() == 2 && "Unexpected PendingLocs.size()");
    // Apply the normal calling convention rules to the first half of the
    // split argument.
    CCValAssign VA = PendingLocs[0];
    ISD::ArgFlagsTy AF = PendingArgFlags[0];
    PendingLocs.clear();
    PendingArgFlags.clear();
    return CC_LoongArchAssign2GRLen(GRLen, State, VA, AF, ValNo, ValVT, LocVT,
                                    ArgFlags);
  }

  // Allocate to a register if possible, or else a stack slot.
  Register Reg;
  unsigned StoreSizeBytes = GRLen / 8;
  Align StackAlign = Align(GRLen / 8);

  if (ValVT == MVT::f32 && !UseGPRForFloat)
    Reg = State.AllocateReg(ArgFPR32s);
  else if (ValVT == MVT::f64 && !UseGPRForFloat)
    Reg = State.AllocateReg(ArgFPR64s);
  else if (ValVT.is128BitVector())
    Reg = State.AllocateReg(ArgVRs);
  else if (ValVT.is256BitVector())
    Reg = State.AllocateReg(ArgXRs);
  else
    Reg = State.AllocateReg(ArgGPRs);

  unsigned StackOffset =
      Reg ? 0 : State.AllocateStack(StoreSizeBytes, StackAlign);

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
  assert((!UseGPRForFloat || LocVT == GRLenVT) &&
         "Expected an GRLenVT at this stage");

  if (Reg) {
    State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
    return false;
  }

  // When a floating-point value is passed on the stack, no bit-cast is needed.
  if (ValVT.isFloatingPoint()) {
    LocVT = ValVT;
    LocInfo = CCValAssign::Full;
  }

  State.addLoc(CCValAssign::getMem(ValNo, ValVT, StackOffset, LocVT, LocInfo));
  return false;
}

void LoongArchTargetLowering::analyzeInputArgs(
    MachineFunction &MF, CCState &CCInfo,
    const SmallVectorImpl<ISD::InputArg> &Ins, bool IsRet,
    LoongArchCCAssignFn Fn) const {
  FunctionType *FType = MF.getFunction().getFunctionType();
  for (unsigned i = 0, e = Ins.size(); i != e; ++i) {
    MVT ArgVT = Ins[i].VT;
    Type *ArgTy = nullptr;
    if (IsRet)
      ArgTy = FType->getReturnType();
    else if (Ins[i].isOrigArg())
      ArgTy = FType->getParamType(Ins[i].getOrigArgIndex());
    LoongArchABI::ABI ABI =
        MF.getSubtarget<LoongArchSubtarget>().getTargetABI();
    if (Fn(MF.getDataLayout(), ABI, i, ArgVT, CCValAssign::Full, Ins[i].Flags,
           CCInfo, /*IsFixed=*/true, IsRet, ArgTy)) {
      LLVM_DEBUG(dbgs() << "InputArg #" << i << " has unhandled type " << ArgVT
                        << '\n');
      llvm_unreachable("");
    }
  }
}

void LoongArchTargetLowering::analyzeOutputArgs(
    MachineFunction &MF, CCState &CCInfo,
    const SmallVectorImpl<ISD::OutputArg> &Outs, bool IsRet,
    CallLoweringInfo *CLI, LoongArchCCAssignFn Fn) const {
  for (unsigned i = 0, e = Outs.size(); i != e; ++i) {
    MVT ArgVT = Outs[i].VT;
    Type *OrigTy = CLI ? CLI->getArgs()[Outs[i].OrigArgIndex].Ty : nullptr;
    LoongArchABI::ABI ABI =
        MF.getSubtarget<LoongArchSubtarget>().getTargetABI();
    if (Fn(MF.getDataLayout(), ABI, i, ArgVT, CCValAssign::Full, Outs[i].Flags,
           CCInfo, Outs[i].IsFixed, IsRet, OrigTy)) {
      LLVM_DEBUG(dbgs() << "OutputArg #" << i << " has unhandled type " << ArgVT
                        << "\n");
      llvm_unreachable("");
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
  case CCValAssign::Indirect:
    break;
  case CCValAssign::BCvt:
    if (VA.getLocVT() == MVT::i64 && VA.getValVT() == MVT::f32)
      Val = DAG.getNode(LoongArchISD::MOVGR2FR_W_LA64, DL, MVT::f32, Val);
    else
      Val = DAG.getNode(ISD::BITCAST, DL, VA.getValVT(), Val);
    break;
  }
  return Val;
}

static SDValue unpackFromRegLoc(SelectionDAG &DAG, SDValue Chain,
                                const CCValAssign &VA, const SDLoc &DL,
                                const LoongArchTargetLowering &TLI) {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();
  EVT LocVT = VA.getLocVT();
  SDValue Val;
  const TargetRegisterClass *RC = TLI.getRegClassFor(LocVT.getSimpleVT());
  Register VReg = RegInfo.createVirtualRegister(RC);
  RegInfo.addLiveIn(VA.getLocReg(), VReg);
  Val = DAG.getCopyFromReg(Chain, DL, VReg, LocVT);

  return convertLocVTToValVT(DAG, Val, VA, DL);
}

// The caller is responsible for loading the full value if the argument is
// passed with CCValAssign::Indirect.
static SDValue unpackFromMemLoc(SelectionDAG &DAG, SDValue Chain,
                                const CCValAssign &VA, const SDLoc &DL) {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  EVT ValVT = VA.getValVT();
  int FI = MFI.CreateFixedObject(ValVT.getStoreSize(), VA.getLocMemOffset(),
                                 /*IsImmutable=*/true);
  SDValue FIN = DAG.getFrameIndex(
      FI, MVT::getIntegerVT(DAG.getDataLayout().getPointerSizeInBits(0)));

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
  return DAG.getExtLoad(
      ExtType, DL, VA.getLocVT(), Chain, FIN,
      MachinePointerInfo::getFixedStack(DAG.getMachineFunction(), FI), ValVT);
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
    if (VA.getLocVT() == MVT::i64 && VA.getValVT() == MVT::f32)
      Val = DAG.getNode(LoongArchISD::MOVFR2GR_S_LA64, DL, MVT::i64, Val);
    else
      Val = DAG.getNode(ISD::BITCAST, DL, LocVT, Val);
    break;
  }
  return Val;
}

static bool CC_LoongArch_GHC(unsigned ValNo, MVT ValVT, MVT LocVT,
                             CCValAssign::LocInfo LocInfo,
                             ISD::ArgFlagsTy ArgFlags, CCState &State) {
  if (LocVT == MVT::i32 || LocVT == MVT::i64) {
    // Pass in STG registers: Base, Sp, Hp, R1, R2, R3, R4, R5, SpLim
    //                        s0    s1  s2  s3  s4  s5  s6  s7  s8
    static const MCPhysReg GPRList[] = {
        LoongArch::R23, LoongArch::R24, LoongArch::R25,
        LoongArch::R26, LoongArch::R27, LoongArch::R28,
        LoongArch::R29, LoongArch::R30, LoongArch::R31};
    if (unsigned Reg = State.AllocateReg(GPRList)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (LocVT == MVT::f32) {
    // Pass in STG registers: F1, F2, F3, F4
    //                        fs0,fs1,fs2,fs3
    static const MCPhysReg FPR32List[] = {LoongArch::F24, LoongArch::F25,
                                          LoongArch::F26, LoongArch::F27};
    if (unsigned Reg = State.AllocateReg(FPR32List)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (LocVT == MVT::f64) {
    // Pass in STG registers: D1, D2, D3, D4
    //                        fs4,fs5,fs6,fs7
    static const MCPhysReg FPR64List[] = {LoongArch::F28_64, LoongArch::F29_64,
                                          LoongArch::F30_64, LoongArch::F31_64};
    if (unsigned Reg = State.AllocateReg(FPR64List)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  report_fatal_error("No registers left in GHC calling convention");
  return true;
}

// Transform physical registers into virtual registers.
SDValue LoongArchTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {

  MachineFunction &MF = DAG.getMachineFunction();

  switch (CallConv) {
  default:
    llvm_unreachable("Unsupported calling convention");
  case CallingConv::C:
  case CallingConv::Fast:
    break;
  case CallingConv::GHC:
    if (!MF.getSubtarget().hasFeature(LoongArch::FeatureBasicF) ||
        !MF.getSubtarget().hasFeature(LoongArch::FeatureBasicD))
      report_fatal_error(
          "GHC calling convention requires the F and D extensions");
  }

  EVT PtrVT = getPointerTy(DAG.getDataLayout());
  MVT GRLenVT = Subtarget.getGRLenVT();
  unsigned GRLenInBytes = Subtarget.getGRLen() / 8;
  // Used with varargs to acumulate store chains.
  std::vector<SDValue> OutChains;

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());

  if (CallConv == CallingConv::GHC)
    CCInfo.AnalyzeFormalArguments(Ins, CC_LoongArch_GHC);
  else
    analyzeInputArgs(MF, CCInfo, Ins, /*IsRet=*/false, CC_LoongArch);

  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    SDValue ArgValue;
    if (VA.isRegLoc())
      ArgValue = unpackFromRegLoc(DAG, Chain, VA, DL, *this);
    else
      ArgValue = unpackFromMemLoc(DAG, Chain, VA, DL);
    if (VA.getLocInfo() == CCValAssign::Indirect) {
      // If the original argument was split and passed by reference, we need to
      // load all parts of it here (using the same address).
      InVals.push_back(DAG.getLoad(VA.getValVT(), DL, Chain, ArgValue,
                                   MachinePointerInfo()));
      unsigned ArgIndex = Ins[i].OrigArgIndex;
      unsigned ArgPartOffset = Ins[i].PartOffset;
      assert(ArgPartOffset == 0);
      while (i + 1 != e && Ins[i + 1].OrigArgIndex == ArgIndex) {
        CCValAssign &PartVA = ArgLocs[i + 1];
        unsigned PartOffset = Ins[i + 1].PartOffset - ArgPartOffset;
        SDValue Offset = DAG.getIntPtrConstant(PartOffset, DL);
        SDValue Address = DAG.getNode(ISD::ADD, DL, PtrVT, ArgValue, Offset);
        InVals.push_back(DAG.getLoad(PartVA.getValVT(), DL, Chain, Address,
                                     MachinePointerInfo()));
        ++i;
      }
      continue;
    }
    InVals.push_back(ArgValue);
  }

  if (IsVarArg) {
    ArrayRef<MCPhysReg> ArgRegs = ArrayRef(ArgGPRs);
    unsigned Idx = CCInfo.getFirstUnallocated(ArgRegs);
    const TargetRegisterClass *RC = &LoongArch::GPRRegClass;
    MachineFrameInfo &MFI = MF.getFrameInfo();
    MachineRegisterInfo &RegInfo = MF.getRegInfo();
    auto *LoongArchFI = MF.getInfo<LoongArchMachineFunctionInfo>();

    // Offset of the first variable argument from stack pointer, and size of
    // the vararg save area. For now, the varargs save area is either zero or
    // large enough to hold a0-a7.
    int VaArgOffset, VarArgsSaveSize;

    // If all registers are allocated, then all varargs must be passed on the
    // stack and we don't need to save any argregs.
    if (ArgRegs.size() == Idx) {
      VaArgOffset = CCInfo.getStackSize();
      VarArgsSaveSize = 0;
    } else {
      VarArgsSaveSize = GRLenInBytes * (ArgRegs.size() - Idx);
      VaArgOffset = -VarArgsSaveSize;
    }

    // Record the frame index of the first variable argument
    // which is a value necessary to VASTART.
    int FI = MFI.CreateFixedObject(GRLenInBytes, VaArgOffset, true);
    LoongArchFI->setVarArgsFrameIndex(FI);

    // If saving an odd number of registers then create an extra stack slot to
    // ensure that the frame pointer is 2*GRLen-aligned, which in turn ensures
    // offsets to even-numbered registered remain 2*GRLen-aligned.
    if (Idx % 2) {
      MFI.CreateFixedObject(GRLenInBytes, VaArgOffset - (int)GRLenInBytes,
                            true);
      VarArgsSaveSize += GRLenInBytes;
    }

    // Copy the integer registers that may have been used for passing varargs
    // to the vararg save area.
    for (unsigned I = Idx; I < ArgRegs.size();
         ++I, VaArgOffset += GRLenInBytes) {
      const Register Reg = RegInfo.createVirtualRegister(RC);
      RegInfo.addLiveIn(ArgRegs[I], Reg);
      SDValue ArgValue = DAG.getCopyFromReg(Chain, DL, Reg, GRLenVT);
      FI = MFI.CreateFixedObject(GRLenInBytes, VaArgOffset, true);
      SDValue PtrOff = DAG.getFrameIndex(FI, getPointerTy(DAG.getDataLayout()));
      SDValue Store = DAG.getStore(Chain, DL, ArgValue, PtrOff,
                                   MachinePointerInfo::getFixedStack(MF, FI));
      cast<StoreSDNode>(Store.getNode())
          ->getMemOperand()
          ->setValue((Value *)nullptr);
      OutChains.push_back(Store);
    }
    LoongArchFI->setVarArgsSaveSize(VarArgsSaveSize);
  }

  // All stores are grouped in one node to allow the matching between
  // the size of Ins and InVals. This only happens for vararg functions.
  if (!OutChains.empty()) {
    OutChains.push_back(Chain);
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, OutChains);
  }

  return Chain;
}

bool LoongArchTargetLowering::mayBeEmittedAsTailCall(const CallInst *CI) const {
  return CI->isTailCall();
}

// Check if the return value is used as only a return value, as otherwise
// we can't perform a tail-call.
bool LoongArchTargetLowering::isUsedByReturnOnly(SDNode *N,
                                                 SDValue &Chain) const {
  if (N->getNumValues() != 1)
    return false;
  if (!N->hasNUsesOfValue(1, 0))
    return false;

  SDNode *Copy = *N->use_begin();
  if (Copy->getOpcode() != ISD::CopyToReg)
    return false;

  // If the ISD::CopyToReg has a glue operand, we conservatively assume it
  // isn't safe to perform a tail call.
  if (Copy->getGluedNode())
    return false;

  // The copy must be used by a LoongArchISD::RET, and nothing else.
  bool HasRet = false;
  for (SDNode *Node : Copy->uses()) {
    if (Node->getOpcode() != LoongArchISD::RET)
      return false;
    HasRet = true;
  }

  if (!HasRet)
    return false;

  Chain = Copy->getOperand(0);
  return true;
}

// Check whether the call is eligible for tail call optimization.
bool LoongArchTargetLowering::isEligibleForTailCallOptimization(
    CCState &CCInfo, CallLoweringInfo &CLI, MachineFunction &MF,
    const SmallVectorImpl<CCValAssign> &ArgLocs) const {

  auto CalleeCC = CLI.CallConv;
  auto &Outs = CLI.Outs;
  auto &Caller = MF.getFunction();
  auto CallerCC = Caller.getCallingConv();

  // Do not tail call opt if the stack is used to pass parameters.
  if (CCInfo.getStackSize() != 0)
    return false;

  // Do not tail call opt if any parameters need to be passed indirectly.
  for (auto &VA : ArgLocs)
    if (VA.getLocInfo() == CCValAssign::Indirect)
      return false;

  // Do not tail call opt if either caller or callee uses struct return
  // semantics.
  auto IsCallerStructRet = Caller.hasStructRetAttr();
  auto IsCalleeStructRet = Outs.empty() ? false : Outs[0].Flags.isSRet();
  if (IsCallerStructRet || IsCalleeStructRet)
    return false;

  // Do not tail call opt if either the callee or caller has a byval argument.
  for (auto &Arg : Outs)
    if (Arg.Flags.isByVal())
      return false;

  // The callee has to preserve all registers the caller needs to preserve.
  const LoongArchRegisterInfo *TRI = Subtarget.getRegisterInfo();
  const uint32_t *CallerPreserved = TRI->getCallPreservedMask(MF, CallerCC);
  if (CalleeCC != CallerCC) {
    const uint32_t *CalleePreserved = TRI->getCallPreservedMask(MF, CalleeCC);
    if (!TRI->regmaskSubsetEqual(CallerPreserved, CalleePreserved))
      return false;
  }
  return true;
}

static Align getPrefTypeAlign(EVT VT, SelectionDAG &DAG) {
  return DAG.getDataLayout().getPrefTypeAlign(
      VT.getTypeForEVT(*DAG.getContext()));
}

// Lower a call to a callseq_start + CALL + callseq_end chain, and add input
// and output parameter nodes.
SDValue
LoongArchTargetLowering::LowerCall(CallLoweringInfo &CLI,
                                   SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG = CLI.DAG;
  SDLoc &DL = CLI.DL;
  SmallVectorImpl<ISD::OutputArg> &Outs = CLI.Outs;
  SmallVectorImpl<SDValue> &OutVals = CLI.OutVals;
  SmallVectorImpl<ISD::InputArg> &Ins = CLI.Ins;
  SDValue Chain = CLI.Chain;
  SDValue Callee = CLI.Callee;
  CallingConv::ID CallConv = CLI.CallConv;
  bool IsVarArg = CLI.IsVarArg;
  EVT PtrVT = getPointerTy(DAG.getDataLayout());
  MVT GRLenVT = Subtarget.getGRLenVT();
  bool &IsTailCall = CLI.IsTailCall;

  MachineFunction &MF = DAG.getMachineFunction();

  // Analyze the operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign> ArgLocs;
  CCState ArgCCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());

  if (CallConv == CallingConv::GHC)
    ArgCCInfo.AnalyzeCallOperands(Outs, CC_LoongArch_GHC);
  else
    analyzeOutputArgs(MF, ArgCCInfo, Outs, /*IsRet=*/false, &CLI, CC_LoongArch);

  // Check if it's really possible to do a tail call.
  if (IsTailCall)
    IsTailCall = isEligibleForTailCallOptimization(ArgCCInfo, CLI, MF, ArgLocs);

  if (IsTailCall)
    ++NumTailCalls;
  else if (CLI.CB && CLI.CB->isMustTailCall())
    report_fatal_error("failed to perform tail call elimination on a call "
                       "site marked musttail");

  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NumBytes = ArgCCInfo.getStackSize();

  // Create local copies for byval args.
  SmallVector<SDValue> ByValArgs;
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
    SDValue SizeNode = DAG.getConstant(Size, DL, GRLenVT);

    Chain = DAG.getMemcpy(Chain, DL, FIPtr, Arg, SizeNode, Alignment,
                          /*IsVolatile=*/false,
                          /*AlwaysInline=*/false, /*isTailCall=*/IsTailCall,
                          MachinePointerInfo(), MachinePointerInfo());
    ByValArgs.push_back(FIPtr);
  }

  if (!IsTailCall)
    Chain = DAG.getCALLSEQ_START(Chain, NumBytes, 0, CLI.DL);

  // Copy argument values to their designated locations.
  SmallVector<std::pair<Register, SDValue>> RegsToPass;
  SmallVector<SDValue> MemOpChains;
  SDValue StackPtr;
  for (unsigned i = 0, j = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    SDValue ArgValue = OutVals[i];
    ISD::ArgFlagsTy Flags = Outs[i].Flags;

    // Promote the value if needed.
    // For now, only handle fully promoted and indirect arguments.
    if (VA.getLocInfo() == CCValAssign::Indirect) {
      // Store the argument in a stack slot and pass its address.
      Align StackAlign =
          std::max(getPrefTypeAlign(Outs[i].ArgVT, DAG),
                   getPrefTypeAlign(ArgValue.getValueType(), DAG));
      TypeSize StoredSize = ArgValue.getValueType().getStoreSize();
      // If the original argument was split and passed by reference, we need to
      // store the required parts of it here (and pass just one address).
      unsigned ArgIndex = Outs[i].OrigArgIndex;
      unsigned ArgPartOffset = Outs[i].PartOffset;
      assert(ArgPartOffset == 0);
      // Calculate the total size to store. We don't have access to what we're
      // actually storing other than performing the loop and collecting the
      // info.
      SmallVector<std::pair<SDValue, SDValue>> Parts;
      while (i + 1 != e && Outs[i + 1].OrigArgIndex == ArgIndex) {
        SDValue PartValue = OutVals[i + 1];
        unsigned PartOffset = Outs[i + 1].PartOffset - ArgPartOffset;
        SDValue Offset = DAG.getIntPtrConstant(PartOffset, DL);
        EVT PartVT = PartValue.getValueType();

        StoredSize += PartVT.getStoreSize();
        StackAlign = std::max(StackAlign, getPrefTypeAlign(PartVT, DAG));
        Parts.push_back(std::make_pair(PartValue, Offset));
        ++i;
      }
      SDValue SpillSlot = DAG.CreateStackTemporary(StoredSize, StackAlign);
      int FI = cast<FrameIndexSDNode>(SpillSlot)->getIndex();
      MemOpChains.push_back(
          DAG.getStore(Chain, DL, ArgValue, SpillSlot,
                       MachinePointerInfo::getFixedStack(MF, FI)));
      for (const auto &Part : Parts) {
        SDValue PartValue = Part.first;
        SDValue PartOffset = Part.second;
        SDValue Address =
            DAG.getNode(ISD::ADD, DL, PtrVT, SpillSlot, PartOffset);
        MemOpChains.push_back(
            DAG.getStore(Chain, DL, PartValue, Address,
                         MachinePointerInfo::getFixedStack(MF, FI)));
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
        StackPtr = DAG.getCopyFromReg(Chain, DL, LoongArch::R3, PtrVT);
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

  // If the callee is a GlobalAddress/ExternalSymbol node, turn it into a
  // TargetGlobalAddress/TargetExternalSymbol node so that legalize won't
  // split it and then direct call can be matched by PseudoCALL.
  if (GlobalAddressSDNode *S = dyn_cast<GlobalAddressSDNode>(Callee)) {
    const GlobalValue *GV = S->getGlobal();
    unsigned OpFlags =
        getTargetMachine().shouldAssumeDSOLocal(*GV->getParent(), GV)
            ? LoongArchII::MO_CALL
            : LoongArchII::MO_CALL_PLT;
    Callee = DAG.getTargetGlobalAddress(S->getGlobal(), DL, PtrVT, 0, OpFlags);
  } else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Callee)) {
    unsigned OpFlags = getTargetMachine().shouldAssumeDSOLocal(
                           *MF.getFunction().getParent(), nullptr)
                           ? LoongArchII::MO_CALL
                           : LoongArchII::MO_CALL_PLT;
    Callee = DAG.getTargetExternalSymbol(S->getSymbol(), PtrVT, OpFlags);
  }

  // The first call operand is the chain and the second is the target address.
  SmallVector<SDValue> Ops;
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
    SDValue Ret = DAG.getNode(LoongArchISD::TAIL, DL, NodeTys, Ops);
    DAG.addNoMergeSiteInfo(Ret.getNode(), CLI.NoMerge);
    return Ret;
  }

  Chain = DAG.getNode(LoongArchISD::CALL, DL, NodeTys, Ops);
  DAG.addNoMergeSiteInfo(Chain.getNode(), CLI.NoMerge);
  Glue = Chain.getValue(1);

  // Mark the end of the call, which is glued to the call itself.
  Chain = DAG.getCALLSEQ_END(Chain, NumBytes, 0, Glue, DL);
  Glue = Chain.getValue(1);

  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign> RVLocs;
  CCState RetCCInfo(CallConv, IsVarArg, MF, RVLocs, *DAG.getContext());
  analyzeInputArgs(MF, RetCCInfo, Ins, /*IsRet=*/true, CC_LoongArch);

  // Copy all of the result registers out of their specified physreg.
  for (auto &VA : RVLocs) {
    // Copy the value out.
    SDValue RetValue =
        DAG.getCopyFromReg(Chain, DL, VA.getLocReg(), VA.getLocVT(), Glue);
    // Glue the RetValue to the end of the call sequence.
    Chain = RetValue.getValue(1);
    Glue = RetValue.getValue(2);

    RetValue = convertLocVTToValVT(DAG, RetValue, VA, DL);

    InVals.push_back(RetValue);
  }

  return Chain;
}

bool LoongArchTargetLowering::CanLowerReturn(
    CallingConv::ID CallConv, MachineFunction &MF, bool IsVarArg,
    const SmallVectorImpl<ISD::OutputArg> &Outs, LLVMContext &Context) const {
  SmallVector<CCValAssign> RVLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, RVLocs, Context);

  for (unsigned i = 0, e = Outs.size(); i != e; ++i) {
    LoongArchABI::ABI ABI =
        MF.getSubtarget<LoongArchSubtarget>().getTargetABI();
    if (CC_LoongArch(MF.getDataLayout(), ABI, i, Outs[i].VT, CCValAssign::Full,
                     Outs[i].Flags, CCInfo, /*IsFixed=*/true, /*IsRet=*/true,
                     nullptr))
      return false;
  }
  return true;
}

SDValue LoongArchTargetLowering::LowerReturn(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::OutputArg> &Outs,
    const SmallVectorImpl<SDValue> &OutVals, const SDLoc &DL,
    SelectionDAG &DAG) const {
  // Stores the assignment of the return value to a location.
  SmallVector<CCValAssign> RVLocs;

  // Info about the registers and stack slot.
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), RVLocs,
                 *DAG.getContext());

  analyzeOutputArgs(DAG.getMachineFunction(), CCInfo, Outs, /*IsRet=*/true,
                    nullptr, CC_LoongArch);
  if (CallConv == CallingConv::GHC && !RVLocs.empty())
    report_fatal_error("GHC functions return void only");
  SDValue Glue;
  SmallVector<SDValue, 4> RetOps(1, Chain);

  // Copy the result values into the output registers.
  for (unsigned i = 0, e = RVLocs.size(); i < e; ++i) {
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");

    // Handle a 'normal' return.
    SDValue Val = convertValVTToLocVT(DAG, OutVals[i], VA, DL);
    Chain = DAG.getCopyToReg(Chain, DL, VA.getLocReg(), Val, Glue);

    // Guarantee that all emitted copies are stuck together.
    Glue = Chain.getValue(1);
    RetOps.push_back(DAG.getRegister(VA.getLocReg(), VA.getLocVT()));
  }

  RetOps[0] = Chain; // Update chain.

  // Add the glue node if we have it.
  if (Glue.getNode())
    RetOps.push_back(Glue);

  return DAG.getNode(LoongArchISD::RET, DL, MVT::Other, RetOps);
}

bool LoongArchTargetLowering::isFPImmLegal(const APFloat &Imm, EVT VT,
                                           bool ForCodeSize) const {
  // TODO: Maybe need more checks here after vector extension is supported.
  if (VT == MVT::f32 && !Subtarget.hasBasicF())
    return false;
  if (VT == MVT::f64 && !Subtarget.hasBasicD())
    return false;
  return (Imm.isZero() || Imm.isExactlyValue(+1.0));
}

bool LoongArchTargetLowering::isCheapToSpeculateCttz(Type *) const {
  return true;
}

bool LoongArchTargetLowering::isCheapToSpeculateCtlz(Type *) const {
  return true;
}

bool LoongArchTargetLowering::shouldInsertFencesForAtomic(
    const Instruction *I) const {
  if (!Subtarget.is64Bit())
    return isa<LoadInst>(I) || isa<StoreInst>(I);

  if (isa<LoadInst>(I))
    return true;

  // On LA64, atomic store operations with IntegerBitWidth of 32 and 64 do not
  // require fences beacuse we can use amswap_db.[w/d].
  if (isa<StoreInst>(I)) {
    unsigned Size = I->getOperand(0)->getType()->getIntegerBitWidth();
    return (Size == 8 || Size == 16);
  }

  return false;
}

EVT LoongArchTargetLowering::getSetCCResultType(const DataLayout &DL,
                                                LLVMContext &Context,
                                                EVT VT) const {
  if (!VT.isVector())
    return getPointerTy(DL);
  return VT.changeVectorElementTypeToInteger();
}

bool LoongArchTargetLowering::hasAndNot(SDValue Y) const {
  // TODO: Support vectors.
  return Y.getValueType().isScalarInteger() && !isa<ConstantSDNode>(Y);
}

bool LoongArchTargetLowering::getTgtMemIntrinsic(IntrinsicInfo &Info,
                                                 const CallInst &I,
                                                 MachineFunction &MF,
                                                 unsigned Intrinsic) const {
  switch (Intrinsic) {
  default:
    return false;
  case Intrinsic::loongarch_masked_atomicrmw_xchg_i32:
  case Intrinsic::loongarch_masked_atomicrmw_add_i32:
  case Intrinsic::loongarch_masked_atomicrmw_sub_i32:
  case Intrinsic::loongarch_masked_atomicrmw_nand_i32:
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::i32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.align = Align(4);
    Info.flags = MachineMemOperand::MOLoad | MachineMemOperand::MOStore |
                 MachineMemOperand::MOVolatile;
    return true;
    // TODO: Add more Intrinsics later.
  }
}

TargetLowering::AtomicExpansionKind
LoongArchTargetLowering::shouldExpandAtomicRMWInIR(AtomicRMWInst *AI) const {
  // TODO: Add more AtomicRMWInst that needs to be extended.

  // Since floating-point operation requires a non-trivial set of data
  // operations, use CmpXChg to expand.
  if (AI->isFloatingPointOperation() ||
      AI->getOperation() == AtomicRMWInst::UIncWrap ||
      AI->getOperation() == AtomicRMWInst::UDecWrap)
    return AtomicExpansionKind::CmpXChg;

  unsigned Size = AI->getType()->getPrimitiveSizeInBits();
  if (Size == 8 || Size == 16)
    return AtomicExpansionKind::MaskedIntrinsic;
  return AtomicExpansionKind::None;
}

static Intrinsic::ID
getIntrinsicForMaskedAtomicRMWBinOp(unsigned GRLen,
                                    AtomicRMWInst::BinOp BinOp) {
  if (GRLen == 64) {
    switch (BinOp) {
    default:
      llvm_unreachable("Unexpected AtomicRMW BinOp");
    case AtomicRMWInst::Xchg:
      return Intrinsic::loongarch_masked_atomicrmw_xchg_i64;
    case AtomicRMWInst::Add:
      return Intrinsic::loongarch_masked_atomicrmw_add_i64;
    case AtomicRMWInst::Sub:
      return Intrinsic::loongarch_masked_atomicrmw_sub_i64;
    case AtomicRMWInst::Nand:
      return Intrinsic::loongarch_masked_atomicrmw_nand_i64;
    case AtomicRMWInst::UMax:
      return Intrinsic::loongarch_masked_atomicrmw_umax_i64;
    case AtomicRMWInst::UMin:
      return Intrinsic::loongarch_masked_atomicrmw_umin_i64;
    case AtomicRMWInst::Max:
      return Intrinsic::loongarch_masked_atomicrmw_max_i64;
    case AtomicRMWInst::Min:
      return Intrinsic::loongarch_masked_atomicrmw_min_i64;
      // TODO: support other AtomicRMWInst.
    }
  }

  if (GRLen == 32) {
    switch (BinOp) {
    default:
      llvm_unreachable("Unexpected AtomicRMW BinOp");
    case AtomicRMWInst::Xchg:
      return Intrinsic::loongarch_masked_atomicrmw_xchg_i32;
    case AtomicRMWInst::Add:
      return Intrinsic::loongarch_masked_atomicrmw_add_i32;
    case AtomicRMWInst::Sub:
      return Intrinsic::loongarch_masked_atomicrmw_sub_i32;
    case AtomicRMWInst::Nand:
      return Intrinsic::loongarch_masked_atomicrmw_nand_i32;
      // TODO: support other AtomicRMWInst.
    }
  }

  llvm_unreachable("Unexpected GRLen\n");
}

TargetLowering::AtomicExpansionKind
LoongArchTargetLowering::shouldExpandAtomicCmpXchgInIR(
    AtomicCmpXchgInst *CI) const {
  unsigned Size = CI->getCompareOperand()->getType()->getPrimitiveSizeInBits();
  if (Size == 8 || Size == 16)
    return AtomicExpansionKind::MaskedIntrinsic;
  return AtomicExpansionKind::None;
}

Value *LoongArchTargetLowering::emitMaskedAtomicCmpXchgIntrinsic(
    IRBuilderBase &Builder, AtomicCmpXchgInst *CI, Value *AlignedAddr,
    Value *CmpVal, Value *NewVal, Value *Mask, AtomicOrdering Ord) const {
  Value *Ordering =
      Builder.getIntN(Subtarget.getGRLen(), static_cast<uint64_t>(Ord));

  // TODO: Support cmpxchg on LA32.
  Intrinsic::ID CmpXchgIntrID = Intrinsic::loongarch_masked_cmpxchg_i64;
  CmpVal = Builder.CreateSExt(CmpVal, Builder.getInt64Ty());
  NewVal = Builder.CreateSExt(NewVal, Builder.getInt64Ty());
  Mask = Builder.CreateSExt(Mask, Builder.getInt64Ty());
  Type *Tys[] = {AlignedAddr->getType()};
  Function *MaskedCmpXchg =
      Intrinsic::getDeclaration(CI->getModule(), CmpXchgIntrID, Tys);
  Value *Result = Builder.CreateCall(
      MaskedCmpXchg, {AlignedAddr, CmpVal, NewVal, Mask, Ordering});
  Result = Builder.CreateTrunc(Result, Builder.getInt32Ty());
  return Result;
}

Value *LoongArchTargetLowering::emitMaskedAtomicRMWIntrinsic(
    IRBuilderBase &Builder, AtomicRMWInst *AI, Value *AlignedAddr, Value *Incr,
    Value *Mask, Value *ShiftAmt, AtomicOrdering Ord) const {
  unsigned GRLen = Subtarget.getGRLen();
  Value *Ordering =
      Builder.getIntN(GRLen, static_cast<uint64_t>(AI->getOrdering()));
  Type *Tys[] = {AlignedAddr->getType()};
  Function *LlwOpScwLoop = Intrinsic::getDeclaration(
      AI->getModule(),
      getIntrinsicForMaskedAtomicRMWBinOp(GRLen, AI->getOperation()), Tys);

  if (GRLen == 64) {
    Incr = Builder.CreateSExt(Incr, Builder.getInt64Ty());
    Mask = Builder.CreateSExt(Mask, Builder.getInt64Ty());
    ShiftAmt = Builder.CreateSExt(ShiftAmt, Builder.getInt64Ty());
  }

  Value *Result;

  // Must pass the shift amount needed to sign extend the loaded value prior
  // to performing a signed comparison for min/max. ShiftAmt is the number of
  // bits to shift the value into position. Pass GRLen-ShiftAmt-ValWidth, which
  // is the number of bits to left+right shift the value in order to
  // sign-extend.
  if (AI->getOperation() == AtomicRMWInst::Min ||
      AI->getOperation() == AtomicRMWInst::Max) {
    const DataLayout &DL = AI->getModule()->getDataLayout();
    unsigned ValWidth =
        DL.getTypeStoreSizeInBits(AI->getValOperand()->getType());
    Value *SextShamt =
        Builder.CreateSub(Builder.getIntN(GRLen, GRLen - ValWidth), ShiftAmt);
    Result = Builder.CreateCall(LlwOpScwLoop,
                                {AlignedAddr, Incr, Mask, SextShamt, Ordering});
  } else {
    Result =
        Builder.CreateCall(LlwOpScwLoop, {AlignedAddr, Incr, Mask, Ordering});
  }

  if (GRLen == 64)
    Result = Builder.CreateTrunc(Result, Builder.getInt32Ty());
  return Result;
}

bool LoongArchTargetLowering::isFMAFasterThanFMulAndFAdd(
    const MachineFunction &MF, EVT VT) const {
  VT = VT.getScalarType();

  if (!VT.isSimple())
    return false;

  switch (VT.getSimpleVT().SimpleTy) {
  case MVT::f32:
  case MVT::f64:
    return true;
  default:
    break;
  }

  return false;
}

Register LoongArchTargetLowering::getExceptionPointerRegister(
    const Constant *PersonalityFn) const {
  return LoongArch::R4;
}

Register LoongArchTargetLowering::getExceptionSelectorRegister(
    const Constant *PersonalityFn) const {
  return LoongArch::R5;
}

//===----------------------------------------------------------------------===//
//                           LoongArch Inline Assembly Support
//===----------------------------------------------------------------------===//

LoongArchTargetLowering::ConstraintType
LoongArchTargetLowering::getConstraintType(StringRef Constraint) const {
  // LoongArch specific constraints in GCC: config/loongarch/constraints.md
  //
  // 'f':  A floating-point register (if available).
  // 'k':  A memory operand whose address is formed by a base register and
  //       (optionally scaled) index register.
  // 'l':  A signed 16-bit constant.
  // 'm':  A memory operand whose address is formed by a base register and
  //       offset that is suitable for use in instructions with the same
  //       addressing mode as st.w and ld.w.
  // 'I':  A signed 12-bit constant (for arithmetic instructions).
  // 'J':  Integer zero.
  // 'K':  An unsigned 12-bit constant (for logic instructions).
  // "ZB": An address that is held in a general-purpose register. The offset is
  //       zero.
  // "ZC": A memory operand whose address is formed by a base register and
  //       offset that is suitable for use in instructions with the same
  //       addressing mode as ll.w and sc.w.
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    default:
      break;
    case 'f':
      return C_RegisterClass;
    case 'l':
    case 'I':
    case 'J':
    case 'K':
      return C_Immediate;
    case 'k':
      return C_Memory;
    }
  }

  if (Constraint == "ZC" || Constraint == "ZB")
    return C_Memory;

  // 'm' is handled here.
  return TargetLowering::getConstraintType(Constraint);
}

InlineAsm::ConstraintCode LoongArchTargetLowering::getInlineAsmMemConstraint(
    StringRef ConstraintCode) const {
  return StringSwitch<InlineAsm::ConstraintCode>(ConstraintCode)
      .Case("k", InlineAsm::ConstraintCode::k)
      .Case("ZB", InlineAsm::ConstraintCode::ZB)
      .Case("ZC", InlineAsm::ConstraintCode::ZC)
      .Default(TargetLowering::getInlineAsmMemConstraint(ConstraintCode));
}

std::pair<unsigned, const TargetRegisterClass *>
LoongArchTargetLowering::getRegForInlineAsmConstraint(
    const TargetRegisterInfo *TRI, StringRef Constraint, MVT VT) const {
  // First, see if this is a constraint that directly corresponds to a LoongArch
  // register class.
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    case 'r':
      // TODO: Support fixed vectors up to GRLen?
      if (VT.isVector())
        break;
      return std::make_pair(0U, &LoongArch::GPRRegClass);
    case 'f':
      if (Subtarget.hasBasicF() && VT == MVT::f32)
        return std::make_pair(0U, &LoongArch::FPR32RegClass);
      if (Subtarget.hasBasicD() && VT == MVT::f64)
        return std::make_pair(0U, &LoongArch::FPR64RegClass);
      if (Subtarget.hasExtLSX() &&
          TRI->isTypeLegalForClass(LoongArch::LSX128RegClass, VT))
        return std::make_pair(0U, &LoongArch::LSX128RegClass);
      if (Subtarget.hasExtLASX() &&
          TRI->isTypeLegalForClass(LoongArch::LASX256RegClass, VT))
        return std::make_pair(0U, &LoongArch::LASX256RegClass);
      break;
    default:
      break;
    }
  }

  // TargetLowering::getRegForInlineAsmConstraint uses the name of the TableGen
  // record (e.g. the "R0" in `def R0`) to choose registers for InlineAsm
  // constraints while the official register name is prefixed with a '$'. So we
  // clip the '$' from the original constraint string (e.g. {$r0} to {r0}.)
  // before it being parsed. And TargetLowering::getRegForInlineAsmConstraint is
  // case insensitive, so no need to convert the constraint to upper case here.
  //
  // For now, no need to support ABI names (e.g. `$a0`) as clang will correctly
  // decode the usage of register name aliases into their official names. And
  // AFAIK, the not yet upstreamed `rustc` for LoongArch will always use
  // official register names.
  if (Constraint.startswith("{$r") || Constraint.startswith("{$f") ||
      Constraint.startswith("{$vr") || Constraint.startswith("{$xr")) {
    bool IsFP = Constraint[2] == 'f';
    std::pair<StringRef, StringRef> Temp = Constraint.split('$');
    std::pair<unsigned, const TargetRegisterClass *> R;
    R = TargetLowering::getRegForInlineAsmConstraint(
        TRI, join_items("", Temp.first, Temp.second), VT);
    // Match those names to the widest floating point register type available.
    if (IsFP) {
      unsigned RegNo = R.first;
      if (LoongArch::F0 <= RegNo && RegNo <= LoongArch::F31) {
        if (Subtarget.hasBasicD() && (VT == MVT::f64 || VT == MVT::Other)) {
          unsigned DReg = RegNo - LoongArch::F0 + LoongArch::F0_64;
          return std::make_pair(DReg, &LoongArch::FPR64RegClass);
        }
      }
    }
    return R;
  }

  return TargetLowering::getRegForInlineAsmConstraint(TRI, Constraint, VT);
}

void LoongArchTargetLowering::LowerAsmOperandForConstraint(
    SDValue Op, std::string &Constraint, std::vector<SDValue> &Ops,
    SelectionDAG &DAG) const {
  // Currently only support length 1 constraints.
  if (Constraint.length() == 1) {
    switch (Constraint[0]) {
    case 'l':
      // Validate & create a 16-bit signed immediate operand.
      if (auto *C = dyn_cast<ConstantSDNode>(Op)) {
        uint64_t CVal = C->getSExtValue();
        if (isInt<16>(CVal))
          Ops.push_back(
              DAG.getTargetConstant(CVal, SDLoc(Op), Subtarget.getGRLenVT()));
      }
      return;
    case 'I':
      // Validate & create a 12-bit signed immediate operand.
      if (auto *C = dyn_cast<ConstantSDNode>(Op)) {
        uint64_t CVal = C->getSExtValue();
        if (isInt<12>(CVal))
          Ops.push_back(
              DAG.getTargetConstant(CVal, SDLoc(Op), Subtarget.getGRLenVT()));
      }
      return;
    case 'J':
      // Validate & create an integer zero operand.
      if (auto *C = dyn_cast<ConstantSDNode>(Op))
        if (C->getZExtValue() == 0)
          Ops.push_back(
              DAG.getTargetConstant(0, SDLoc(Op), Subtarget.getGRLenVT()));
      return;
    case 'K':
      // Validate & create a 12-bit unsigned immediate operand.
      if (auto *C = dyn_cast<ConstantSDNode>(Op)) {
        uint64_t CVal = C->getZExtValue();
        if (isUInt<12>(CVal))
          Ops.push_back(
              DAG.getTargetConstant(CVal, SDLoc(Op), Subtarget.getGRLenVT()));
      }
      return;
    default:
      break;
    }
  }
  TargetLowering::LowerAsmOperandForConstraint(Op, Constraint, Ops, DAG);
}

#define GET_REGISTER_MATCHER
#include "LoongArchGenAsmMatcher.inc"

Register
LoongArchTargetLowering::getRegisterByName(const char *RegName, LLT VT,
                                           const MachineFunction &MF) const {
  std::pair<StringRef, StringRef> Name = StringRef(RegName).split('$');
  std::string NewRegName = Name.second.str();
  Register Reg = MatchRegisterAltName(NewRegName);
  if (Reg == LoongArch::NoRegister)
    Reg = MatchRegisterName(NewRegName);
  if (Reg == LoongArch::NoRegister)
    report_fatal_error(
        Twine("Invalid register name \"" + StringRef(RegName) + "\"."));
  BitVector ReservedRegs = Subtarget.getRegisterInfo()->getReservedRegs(MF);
  if (!ReservedRegs.test(Reg))
    report_fatal_error(Twine("Trying to obtain non-reserved register \"" +
                             StringRef(RegName) + "\"."));
  return Reg;
}

bool LoongArchTargetLowering::decomposeMulByConstant(LLVMContext &Context,
                                                     EVT VT, SDValue C) const {
  // TODO: Support vectors.
  if (!VT.isScalarInteger())
    return false;

  // Omit the optimization if the data size exceeds GRLen.
  if (VT.getSizeInBits() > Subtarget.getGRLen())
    return false;

  if (auto *ConstNode = dyn_cast<ConstantSDNode>(C.getNode())) {
    const APInt &Imm = ConstNode->getAPIntValue();
    // Break MUL into (SLLI + ADD/SUB) or ALSL.
    if ((Imm + 1).isPowerOf2() || (Imm - 1).isPowerOf2() ||
        (1 - Imm).isPowerOf2() || (-1 - Imm).isPowerOf2())
      return true;
    // Break MUL into (ALSL x, (SLLI x, imm0), imm1).
    if (ConstNode->hasOneUse() &&
        ((Imm - 2).isPowerOf2() || (Imm - 4).isPowerOf2() ||
         (Imm - 8).isPowerOf2() || (Imm - 16).isPowerOf2()))
      return true;
    // Break (MUL x, imm) into (ADD (SLLI x, s0), (SLLI x, s1)),
    // in which the immediate has two set bits. Or Break (MUL x, imm)
    // into (SUB (SLLI x, s0), (SLLI x, s1)), in which the immediate
    // equals to (1 << s0) - (1 << s1).
    if (ConstNode->hasOneUse() && !(Imm.sge(-2048) && Imm.sle(4095))) {
      unsigned Shifts = Imm.countr_zero();
      // Reject immediates which can be composed via a single LUI.
      if (Shifts >= 12)
        return false;
      // Reject multiplications can be optimized to
      // (SLLI (ALSL x, x, 1/2/3/4), s).
      APInt ImmPop = Imm.ashr(Shifts);
      if (ImmPop == 3 || ImmPop == 5 || ImmPop == 9 || ImmPop == 17)
        return false;
      // We do not consider the case `(-Imm - ImmSmall).isPowerOf2()`,
      // since it needs one more instruction than other 3 cases.
      APInt ImmSmall = APInt(Imm.getBitWidth(), 1ULL << Shifts, true);
      if ((Imm - ImmSmall).isPowerOf2() || (Imm + ImmSmall).isPowerOf2() ||
          (ImmSmall - Imm).isPowerOf2())
        return true;
    }
  }

  return false;
}

bool LoongArchTargetLowering::isLegalAddressingMode(const DataLayout &DL,
                                                    const AddrMode &AM,
                                                    Type *Ty, unsigned AS,
                                                    Instruction *I) const {
  // LoongArch has four basic addressing modes:
  //  1. reg
  //  2. reg + 12-bit signed offset
  //  3. reg + 14-bit signed offset left-shifted by 2
  //  4. reg1 + reg2
  // TODO: Add more checks after support vector extension.

  // No global is ever allowed as a base.
  if (AM.BaseGV)
    return false;

  // Require a 12 or 14 bit signed offset.
  if (!isInt<12>(AM.BaseOffs) || !isShiftedInt<14, 2>(AM.BaseOffs))
    return false;

  switch (AM.Scale) {
  case 0:
    // "i" is not allowed.
    if (!AM.HasBaseReg)
      return false;
    // Otherwise we have "r+i".
    break;
  case 1:
    // "r+r+i" is not allowed.
    if (AM.HasBaseReg && AM.BaseOffs != 0)
      return false;
    // Otherwise we have "r+r" or "r+i".
    break;
  case 2:
    // "2*r+r" or "2*r+i" is not allowed.
    if (AM.HasBaseReg || AM.BaseOffs)
      return false;
    // Otherwise we have "r+r".
    break;
  default:
    return false;
  }

  return true;
}

bool LoongArchTargetLowering::isLegalICmpImmediate(int64_t Imm) const {
  return isInt<12>(Imm);
}

bool LoongArchTargetLowering::isLegalAddImmediate(int64_t Imm) const {
  return isInt<12>(Imm);
}

bool LoongArchTargetLowering::isZExtFree(SDValue Val, EVT VT2) const {
  // Zexts are free if they can be combined with a load.
  // Don't advertise i32->i64 zextload as being free for LA64. It interacts
  // poorly with type legalization of compares preferring sext.
  if (auto *LD = dyn_cast<LoadSDNode>(Val)) {
    EVT MemVT = LD->getMemoryVT();
    if ((MemVT == MVT::i8 || MemVT == MVT::i16) &&
        (LD->getExtensionType() == ISD::NON_EXTLOAD ||
         LD->getExtensionType() == ISD::ZEXTLOAD))
      return true;
  }

  return TargetLowering::isZExtFree(Val, VT2);
}

bool LoongArchTargetLowering::isSExtCheaperThanZExt(EVT SrcVT, EVT DstVT) const {
  return Subtarget.is64Bit() && SrcVT == MVT::i32 && DstVT == MVT::i64;
}

bool LoongArchTargetLowering::hasAndNotCompare(SDValue Y) const {
  // TODO: Support vectors.
  if (Y.getValueType().isVector())
    return false;

  return !isa<ConstantSDNode>(Y);
}
