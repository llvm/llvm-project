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
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/RuntimeLibcalls.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsLoongArch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/KnownBits.h"

using namespace llvm;

#define DEBUG_TYPE "loongarch-isel-lowering"

STATISTIC(NumTailCalls, "Number of tail calls");

static cl::opt<bool> ZeroDivCheck(
    "loongarch-check-zero-division", cl::Hidden,
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

  setLoadExtAction({ISD::EXTLOAD, ISD::SEXTLOAD, ISD::ZEXTLOAD}, GRLenVT,
                   MVT::i1, Promote);

  // TODO: add necessary setOperationAction calls later.
  setOperationAction(ISD::SHL_PARTS, GRLenVT, Custom);
  setOperationAction(ISD::SRA_PARTS, GRLenVT, Custom);
  setOperationAction(ISD::SRL_PARTS, GRLenVT, Custom);
  setOperationAction(ISD::FP_TO_SINT, GRLenVT, Custom);
  setOperationAction(ISD::ROTL, GRLenVT, Expand);
  setOperationAction(ISD::CTPOP, GRLenVT, Expand);
  setOperationAction(ISD::DEBUGTRAP, MVT::Other, Legal);
  setOperationAction(ISD::TRAP, MVT::Other, Legal);
  setOperationAction(ISD::INTRINSIC_VOID, MVT::Other, Custom);

  setOperationAction({ISD::GlobalAddress, ISD::BlockAddress, ISD::ConstantPool,
                      ISD::JumpTable},
                     GRLenVT, Custom);

  setOperationAction(ISD::GlobalTLSAddress, GRLenVT, Custom);

  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::Other, Custom);

  setOperationAction(ISD::EH_DWARF_CFA, MVT::i32, Custom);
  if (Subtarget.is64Bit())
    setOperationAction(ISD::EH_DWARF_CFA, MVT::i64, Custom);

  setOperationAction(ISD::DYNAMIC_STACKALLOC, GRLenVT, Expand);
  setOperationAction({ISD::STACKSAVE, ISD::STACKRESTORE}, MVT::Other, Expand);
  setOperationAction(ISD::VASTART, MVT::Other, Custom);
  setOperationAction({ISD::VAARG, ISD::VACOPY, ISD::VAEND}, MVT::Other, Expand);

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
    setOperationAction(ISD::INTRINSIC_VOID, MVT::i32, Custom);
    setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::i32, Custom);
    if (Subtarget.hasBasicF() && !Subtarget.hasBasicD())
      setOperationAction(ISD::FP_TO_UINT, MVT::i32, Custom);
    if (Subtarget.hasBasicF())
      setOperationAction(ISD::FRINT, MVT::f32, Legal);
    if (Subtarget.hasBasicD())
      setOperationAction(ISD::FRINT, MVT::f64, Legal);
  }

  // LA32 does not have REVB.2W and REVB.D due to the 64-bit operands, and
  // the narrower REVB.W does not exist. But LA32 does have REVB.2H, so i16
  // and i32 could still be byte-swapped relatively cheaply.
  setOperationAction(ISD::BSWAP, MVT::i16, Custom);
  if (Subtarget.is64Bit()) {
    setOperationAction(ISD::BSWAP, MVT::i32, Custom);
  }

  // Expand bitreverse.i16 with native-width bitrev and shift for now, before
  // we get to know which of sll and revb.2h is faster.
  setOperationAction(ISD::BITREVERSE, MVT::i8, Custom);
  if (Subtarget.is64Bit()) {
    setOperationAction(ISD::BITREVERSE, MVT::i32, Custom);
    setOperationAction(ISD::BITREVERSE, MVT::i64, Legal);
  } else {
    setOperationAction(ISD::BITREVERSE, MVT::i32, Legal);
    setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::i64, Custom);
  }

  static const ISD::CondCode FPCCToExpand[] = {
      ISD::SETOGT, ISD::SETOGE, ISD::SETUGT, ISD::SETUGE,
      ISD::SETGE,  ISD::SETNE,  ISD::SETGT};

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
  }
  if (Subtarget.hasBasicD()) {
    setCondCodeAction(FPCCToExpand, MVT::f64, Expand);
    setOperationAction(ISD::SELECT_CC, MVT::f64, Expand);
    setOperationAction(ISD::BR_CC, MVT::f64, Expand);
    setOperationAction(ISD::STRICT_FSETCCS, MVT::f64, Legal);
    setOperationAction(ISD::STRICT_FSETCC, MVT::f64, Legal);
    setLoadExtAction(ISD::EXTLOAD, MVT::f64, MVT::f32, Expand);
    setOperationAction(ISD::FMA, MVT::f64, Legal);
    setOperationAction(ISD::FMINNUM_IEEE, MVT::f64, Legal);
    setOperationAction(ISD::FMAXNUM_IEEE, MVT::f64, Legal);
    setOperationAction(ISD::FSIN, MVT::f64, Expand);
    setOperationAction(ISD::FCOS, MVT::f64, Expand);
    setOperationAction(ISD::FSINCOS, MVT::f64, Expand);
    setOperationAction(ISD::FPOW, MVT::f64, Expand);
    setOperationAction(ISD::FREM, MVT::f64, Expand);
    setTruncStoreAction(MVT::f64, MVT::f32, Expand);
  }

  setOperationAction(ISD::BR_JT, MVT::Other, Expand);

  setOperationAction(ISD::BR_CC, GRLenVT, Expand);
  setOperationAction(ISD::SELECT_CC, GRLenVT, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);
  setOperationAction({ISD::SMUL_LOHI, ISD::UMUL_LOHI}, GRLenVT, Expand);
  if (!Subtarget.is64Bit())
    setLibcallName(RTLIB::MUL_I128, nullptr);

  setOperationAction(ISD::FP_TO_UINT, GRLenVT, Custom);
  setOperationAction(ISD::UINT_TO_FP, GRLenVT, Expand);
  if ((Subtarget.is64Bit() && Subtarget.hasBasicF() &&
       !Subtarget.hasBasicD())) {
    setOperationAction(ISD::SINT_TO_FP, GRLenVT, Custom);
    setOperationAction(ISD::UINT_TO_FP, GRLenVT, Custom);
  }

  // Compute derived properties from the register classes.
  computeRegisterProperties(STI.getRegisterInfo());

  setStackPointerRegisterToSaveRestore(LoongArch::R3);

  setBooleanContents(ZeroOrOneBooleanContent);

  setMaxAtomicSizeInBitsSupported(Subtarget.getGRLen());

  setMinCmpXchgSizeInBits(32);

  // Function alignments.
  const Align FunctionAlignment(4);
  setMinFunctionAlignment(FunctionAlignment);

  setTargetDAGCombine(ISD::AND);
  setTargetDAGCombine(ISD::OR);
  setTargetDAGCombine(ISD::SRL);
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
  }
  return SDValue();
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
  // TODO: Check CodeModel.
  if (IsLocal)
    // This generates the pattern (PseudoLA_PCREL sym), which expands to
    // (addi.w/d (pcalau12i %pc_hi20(sym)) %pc_lo12(sym)).
    return SDValue(DAG.getMachineNode(LoongArch::PseudoLA_PCREL, DL, Ty, Addr),
                   0);

  // This generates the pattern (PseudoLA_GOT sym), which expands to (ld.w/d
  // (pcalau12i %got_pc_hi20(sym)) %got_pc_lo12(sym)).
  return SDValue(DAG.getMachineNode(LoongArch::PseudoLA_GOT, DL, Ty, Addr), 0);
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
                                                  unsigned Opc) const {
  SDLoc DL(N);
  EVT Ty = getPointerTy(DAG.getDataLayout());
  MVT GRLenVT = Subtarget.getGRLenVT();

  SDValue Addr = DAG.getTargetGlobalAddress(N->getGlobal(), DL, Ty, 0, 0);
  SDValue Offset = SDValue(DAG.getMachineNode(Opc, DL, Ty, Addr), 0);

  // Add the thread pointer.
  return DAG.getNode(ISD::ADD, DL, Ty, Offset,
                     DAG.getRegister(LoongArch::R2, GRLenVT));
}

SDValue LoongArchTargetLowering::getDynamicTLSAddr(GlobalAddressSDNode *N,
                                                   SelectionDAG &DAG,
                                                   unsigned Opc) const {
  SDLoc DL(N);
  EVT Ty = getPointerTy(DAG.getDataLayout());
  IntegerType *CallTy = Type::getIntNTy(*DAG.getContext(), Ty.getSizeInBits());

  // Use a PC-relative addressing mode to access the dynamic GOT address.
  SDValue Addr = DAG.getTargetGlobalAddress(N->getGlobal(), DL, Ty, 0, 0);
  SDValue Load = SDValue(DAG.getMachineNode(Opc, DL, Ty, Addr), 0);

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
  GlobalAddressSDNode *N = cast<GlobalAddressSDNode>(Op);
  assert(N->getOffset() == 0 && "unexpected offset in global node");

  SDValue Addr;
  TLSModel::Model Model = getTargetMachine().getTLSModel(N->getGlobal());

  switch (Model) {
  case TLSModel::GeneralDynamic:
    // In this model, application code calls the dynamic linker function
    // __tls_get_addr to locate TLS offsets into the dynamic thread vector at
    // runtime.
    Addr = getDynamicTLSAddr(N, DAG, LoongArch::PseudoLA_TLS_GD);
    break;
  case TLSModel::LocalDynamic:
    // Same as GeneralDynamic, except for assembly modifiers and relocation
    // records.
    Addr = getDynamicTLSAddr(N, DAG, LoongArch::PseudoLA_TLS_LD);
    break;
  case TLSModel::InitialExec:
    // This model uses the GOT to resolve TLS offsets.
    Addr = getStaticTLSAddr(N, DAG, LoongArch::PseudoLA_TLS_IE);
    break;
  case TLSModel::LocalExec:
    // This model is used when static linking as the TLS offsets are resolved
    // during program linking.
    Addr = getStaticTLSAddr(N, DAG, LoongArch::PseudoLA_TLS_LE);
    break;
  }

  return Addr;
}

SDValue
LoongArchTargetLowering::lowerINTRINSIC_WO_CHAIN(SDValue Op,
                                                 SelectionDAG &DAG) const {
  switch (Op.getConstantOperandVal(0)) {
  default:
    return SDValue(); // Don't custom lower most intrinsics.
  case Intrinsic::thread_pointer: {
    EVT PtrVT = getPointerTy(DAG.getDataLayout());
    return DAG.getRegister(LoongArch::R2, PtrVT);
  }
  }
}

SDValue
LoongArchTargetLowering::lowerINTRINSIC_W_CHAIN(SDValue Op,
                                                SelectionDAG &DAG) const {

  switch (Op.getConstantOperandVal(1)) {
  default:
    return Op;
  case Intrinsic::loongarch_crc_w_d_w: {
    DAG.getContext()->emitError(
        "llvm.loongarch.crc.w.d.w requires target: loongarch64");
    return DAG.getMergeValues(
        {DAG.getUNDEF(Op.getValueType()), Op.getOperand(0)}, SDLoc(Op));
  }
  }
}

// Helper function that emits error message for intrinsics with void return
// value.
static SDValue emitIntrinsicErrorMessage(SDValue Op, StringRef ErrorMsg,
                                         SelectionDAG &DAG) {

  DAG.getContext()->emitError("argument to '" + Op->getOperationName(0) + "' " +
                              ErrorMsg);
  return Op.getOperand(0);
}

SDValue LoongArchTargetLowering::lowerINTRINSIC_VOID(SDValue Op,
                                                     SelectionDAG &DAG) const {
  SDLoc DL(Op);
  MVT GRLenVT = Subtarget.getGRLenVT();
  SDValue Op0 = Op.getOperand(0);
  SDValue Op2 = Op.getOperand(2);
  const StringRef ErrorMsgOOR = "out of range";

  switch (Op.getConstantOperandVal(1)) {
  default:
    // TODO: Add more Intrinsics.
    return SDValue();
  case Intrinsic::loongarch_dbar: {
    unsigned Imm = cast<ConstantSDNode>(Op2)->getZExtValue();
    if (!isUInt<15>(Imm))
      return emitIntrinsicErrorMessage(Op, ErrorMsgOOR, DAG);

    return DAG.getNode(LoongArchISD::DBAR, DL, MVT::Other, Op0,
                       DAG.getConstant(Imm, DL, GRLenVT));
  }
  case Intrinsic::loongarch_ibar: {
    unsigned Imm = cast<ConstantSDNode>(Op2)->getZExtValue();
    if (!isUInt<15>(Imm))
      return emitIntrinsicErrorMessage(Op, ErrorMsgOOR, DAG);

    return DAG.getNode(LoongArchISD::IBAR, DL, MVT::Other, Op0,
                       DAG.getConstant(Imm, DL, GRLenVT));
  }
  case Intrinsic::loongarch_break: {
    unsigned Imm = cast<ConstantSDNode>(Op2)->getZExtValue();
    if (!isUInt<15>(Imm))
      return emitIntrinsicErrorMessage(Op, ErrorMsgOOR, DAG);

    return DAG.getNode(LoongArchISD::BREAK, DL, MVT::Other, Op0,
                       DAG.getConstant(Imm, DL, GRLenVT));
  }
  case Intrinsic::loongarch_syscall: {
    unsigned Imm = cast<ConstantSDNode>(Op2)->getZExtValue();
    if (!isUInt<15>(Imm))
      return emitIntrinsicErrorMessage(Op, ErrorMsgOOR, DAG);

    return DAG.getNode(LoongArchISD::SYSCALL, DL, MVT::Other, Op0,
                       DAG.getConstant(Imm, DL, GRLenVT));
  }
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

void LoongArchTargetLowering::ReplaceNodeResults(
    SDNode *N, SmallVectorImpl<SDValue> &Results, SelectionDAG &DAG) const {
  SDLoc DL(N);
  switch (N->getOpcode()) {
  default:
    llvm_unreachable("Don't know how to legalize this operation");
  case ISD::SHL:
  case ISD::SRA:
  case ISD::SRL:
  case ISD::ROTR:
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
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
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
    SDValue Src = N->getOperand(0);
    EVT VT = EVT::getFloatingPointVT(N->getValueSizeInBits(0));
    if (getTypeAction(*DAG.getContext(), Src.getValueType()) !=
        TargetLowering::TypeSoftenFloat) {
      SDValue Dst = DAG.getNode(LoongArchISD::FTINT, DL, VT, Src);
      Results.push_back(DAG.getNode(ISD::BITCAST, DL, N->getValueType(0), Dst));
      return;
    }
    // If the FP type needs to be softened, emit a library call using the 'si'
    // version. If we left it to default legalization we'd end up with 'di'.
    RTLIB::Libcall LC;
    LC = RTLIB::getFPTOSINT(Src.getValueType(), N->getValueType(0));
    MakeLibCallOptions CallOptions;
    EVT OpVT = Src.getValueType();
    CallOptions.setTypeListBeforeSoften(OpVT, N->getValueType(0), true);
    SDValue Chain = SDValue();
    SDValue Result;
    std::tie(Result, Chain) =
        makeLibCall(DAG, LC, N->getValueType(0), Src, CallOptions, DL, Chain);
    Results.push_back(Result);
    break;
  }
  case ISD::BITCAST: {
    EVT VT = N->getValueType(0);
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
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
    auto &TLI = DAG.getTargetLoweringInfo();
    SDValue Tmp1, Tmp2;
    TLI.expandFP_TO_UINT(N, Tmp1, Tmp2, DAG);
    Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, Tmp1));
    break;
  }
  case ISD::BSWAP: {
    SDValue Src = N->getOperand(0);
    EVT VT = N->getValueType(0);
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
    EVT VT = N->getValueType(0);
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
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
    Results.push_back(customLegalizeToWOp(N, DAG, 1));
    break;
  }
  case ISD::INTRINSIC_W_CHAIN: {
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");

    switch (N->getConstantOperandVal(1)) {
    default:
      llvm_unreachable("Unexpected Intrinsic.");
    case Intrinsic::loongarch_crc_w_d_w: {
      Results.push_back(DAG.getNode(
          ISD::TRUNCATE, DL, N->getValueType(0),
          DAG.getNode(
              LoongArchISD::CRC_W_D_W, DL, MVT::i64, N->getOperand(2),
              DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(3)))));
      Results.push_back(N->getOperand(0));
      break;
    }
    }
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

    // Return if the mask doesn't start at position 0.
    if (SMIdx)
      return SDValue();

    lsb = 0;
    NewOperand = FirstOperand;
  }
  msb = lsb + SMLen - 1;
  return DAG.getNode(LoongArchISD::BSTRPICK, DL, ValTy, NewOperand,
                     DAG.getConstant(msb, DL, GRLenVT),
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

MachineBasicBlock *LoongArchTargetLowering::EmitInstrWithCustomInserter(
    MachineInstr &MI, MachineBasicBlock *BB) const {

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
  }
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
    NODE_NAME_CASE(CRC_W_D_W)
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
  case LoongArchABI::ABI_LP64S:
  case LoongArchABI::ABI_ILP32F:
  case LoongArchABI::ABI_LP64F:
    report_fatal_error("Unimplemented ABI");
    break;
  case LoongArchABI::ABI_ILP32D:
  case LoongArchABI::ABI_LP64D:
    UseGPRForFloat = !IsFixed;
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
      LLVM_DEBUG(dbgs() << "InputArg #" << i << " has unhandled type "
                        << EVT(ArgVT).getEVTString() << '\n');
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
      LLVM_DEBUG(dbgs() << "OutputArg #" << i << " has unhandled type "
                        << EVT(ArgVT).getEVTString() << "\n");
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
  }

  EVT PtrVT = getPointerTy(DAG.getDataLayout());
  MVT GRLenVT = Subtarget.getGRLenVT();
  unsigned GRLenInBytes = Subtarget.getGRLen() / 8;
  // Used with varargs to acumulate store chains.
  std::vector<SDValue> OutChains;

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());

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
    ArrayRef<MCPhysReg> ArgRegs = makeArrayRef(ArgGPRs);
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
      VaArgOffset = CCInfo.getNextStackOffset();
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

// Check whether the call is eligible for tail call optimization.
bool LoongArchTargetLowering::isEligibleForTailCallOptimization(
    CCState &CCInfo, CallLoweringInfo &CLI, MachineFunction &MF,
    const SmallVectorImpl<CCValAssign> &ArgLocs) const {

  auto CalleeCC = CLI.CallConv;
  auto &Outs = CLI.Outs;
  auto &Caller = MF.getFunction();
  auto CallerCC = Caller.getCallingConv();

  // Do not tail call opt if the stack is used to pass parameters.
  if (CCInfo.getNextStackOffset() != 0)
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
  unsigned NumBytes = ArgCCInfo.getNextStackOffset();

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
    return DAG.getNode(LoongArchISD::TAIL, DL, NodeTys, Ops);
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
  if (AI->isFloatingPointOperation())
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

  Result =
      Builder.CreateCall(LlwOpScwLoop, {AlignedAddr, Incr, Mask, Ordering});

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

unsigned LoongArchTargetLowering::getInlineAsmMemConstraint(
    StringRef ConstraintCode) const {
  return StringSwitch<unsigned>(ConstraintCode)
      .Case("k", InlineAsm::Constraint_k)
      .Case("ZB", InlineAsm::Constraint_ZB)
      .Case("ZC", InlineAsm::Constraint_ZC)
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
  if (Constraint.startswith("{$r") || Constraint.startswith("{$f")) {
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
