//===-- M88kISelLowering.cpp - M88k DAG lowering implementation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the M88kTargetLowering class.
//
//===----------------------------------------------------------------------===//

#include "M88kISelLowering.h"
#include "M88kCondCode.h"
//#include "M88kCallingConv.h"
//#include "M88kConstantPoolValue.h"
//#include "M88kMachineFunctionInfo.h"
#include "M88kTargetMachine.h"
#include "MCTargetDesc/M88kBaseInfo.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
//#include "llvm/IR/IntrinsicsM88k.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/KnownBits.h"
#include <cctype>

using namespace llvm;

#define DEBUG_TYPE "m88k-lower"

// If I is a shifted mask, set the size (Width) and the first bit of the
// mask (Offset), and return true.
// For example, if I is 0x003e, (Width, Offset) = (5, 1).
static bool isShiftedMask(uint64_t I, uint64_t &Width, uint64_t &Offset) {
  if (!isShiftedMask_64(I))
    return false;

  Width = countPopulation(I);
  Offset = countTrailingZeros(I);
  return true;
}

static M88kCC::CondCode ISDCCtoM88kCC(ISD::CondCode isdCC) {
  switch (isdCC) {
  case ISD::SETUEQ:
    return M88kCC::EQ;
  case ISD::SETUGT:
    return M88kCC::HI;
  case ISD::SETUGE:
    return M88kCC::HS;
  case ISD::SETULT:
    return M88kCC::LO;
  case ISD::SETULE:
    return M88kCC::LS;
  case ISD::SETUNE:
    return M88kCC::NE;
  case ISD::SETEQ:
    return M88kCC::EQ;
  case ISD::SETGT:
    return M88kCC::GT;
  case ISD::SETGE:
    return M88kCC::GE;
  case ISD::SETLT:
    return M88kCC::LT;
  case ISD::SETLE:
    return M88kCC::LE;
  case ISD::SETNE:
    return M88kCC::NE;
  default:
    llvm_unreachable("Unhandled ISDCC code.");
  }
}

M88kTargetLowering::M88kTargetLowering(const TargetMachine &TM,
                                       const M88kSubtarget &STI)
    : TargetLowering(TM), Subtarget(STI) {
  addRegisterClass(MVT::i32, &M88k::GPRRegClass);
  addRegisterClass(MVT::f32, &M88k::GPRRegClass);

  // Compute derived properties from the register classes
  computeRegisterProperties(Subtarget.getRegisterInfo());

  // Set up special registers.
  setStackPointerRegisterToSaveRestore(M88k::R31);

  // How we extend i1 boolean values.
  setBooleanContents(ZeroOrOneBooleanContent);

  setOperationAction(ISD::GlobalAddress, MVT::i32, Custom);

  setMinFunctionAlignment(Align(4));
  setPrefFunctionAlignment(Align(4));

  // Branch operations.
  setOperationAction(ISD::BR_CC, MVT::i32, Legal);
  setOperationAction(ISD::BR_JT, MVT::Other, Expand);
  setOperationAction(ISD::BRCOND, MVT::Other, Legal);
  setOperationAction(ISD::SETCC, MVT::i32, Legal);
  setOperationAction(ISD::SELECT, MVT::i32, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::i32, Expand);

  // Add/sub with carry.
  // setOperationAction(ISD::CTLZ, MVT::i32, Custom);
  setOperationAction(ISD::CTTZ, MVT::i32, Expand);

  // Special DAG combiner for bit-field operations.
  setTargetDAGCombine(ISD::AND);
  setTargetDAGCombine(ISD::OR);
  setTargetDAGCombine(ISD::SHL);
}

SDValue M88kTargetLowering::LowerOperation(SDValue Op,
                                           SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  case ISD::GlobalAddress:
    return LowerGlobalAddress(Op, DAG);
  case ISD::RETURNADDR:
    return LowerRETURNADDR(Op, DAG);
  }
  return SDValue();
}

namespace {
SDValue performANDCombine(SDNode *N, TargetLowering::DAGCombinerInfo &DCI) {
  SelectionDAG &DAG = DCI.DAG;
  SDValue FirstOperand = N->getOperand(0);
  unsigned FirstOperandOpc = FirstOperand.getOpcode();
  // Second operand of and must be a constant.
  ConstantSDNode *Mask = dyn_cast<ConstantSDNode>(N->getOperand(1));
  if (!Mask)
    return SDValue();
  EVT ValTy = N->getValueType(0);
  SDLoc DL(N);

  SDValue NewOperand;
  unsigned Opc;

  uint64_t Offset;
  uint64_t MaskWidth, MaskOffset;
  if (isShiftedMask(Mask->getZExtValue(), MaskWidth, MaskOffset)) {
    if (FirstOperandOpc == ISD::SRL || FirstOperandOpc == ISD::SRA) {
      // Pattern match:
      // $dst = and (srl/sra $src, offset), (2**width - 1)
      // => EXTU $dst, $src, width<offset>

      // The second operand of the shift must be an immediate.
      ConstantSDNode *ShiftAmt =
          dyn_cast<ConstantSDNode>(FirstOperand.getOperand(1));
      if (!(ShiftAmt))
        return SDValue();

      Offset = ShiftAmt->getZExtValue();

      // Return if the shifted mask does not start at bit 0 or the sum of its
      // width and offset exceeds the word's size.
      if (MaskOffset != 0 || Offset + MaskWidth > ValTy.getSizeInBits())
        return SDValue();

      Opc = M88kISD::EXTU;
      NewOperand = FirstOperand.getOperand(0);
    } else
      return SDValue();
  } else if (isShiftedMask(~Mask->getZExtValue() &
                               ((0x1ULL << ValTy.getSizeInBits()) - 1),
                           MaskWidth, MaskOffset)) {
    // Pattern match:
    // $dst = and $src, ~((2**width - 1) << offset)
    // => CLR $dst, $src, width<offset>
    Opc = M88kISD::CLR;
    NewOperand = FirstOperand;
    Offset = MaskOffset;
  } else
    return SDValue();
  return DAG.getNode(Opc, DL, ValTy, NewOperand,
                     DAG.getConstant(MaskWidth << 5 | Offset, DL, MVT::i32));
}

SDValue performORCombine(SDNode *N, TargetLowering::DAGCombinerInfo &DCI) {
  SelectionDAG &DAG = DCI.DAG;
  uint64_t Width, Offset;

  // Second operand of or must be a constant shifted mask.
  ConstantSDNode *Mask = dyn_cast<ConstantSDNode>(N->getOperand(1));
  if (!Mask || !isShiftedMask(Mask->getZExtValue(), Width, Offset))
    return SDValue();

  // Pattern match:
  // $dst = or $src, ((2**width - 1) << offset
  // => SET $dst, $src, width<offset>
  EVT ValTy = N->getValueType(0);
  SDLoc DL(N);
  return DAG.getNode(M88kISD::SET, DL, ValTy, N->getOperand(0),
                     DAG.getConstant(Width << 5 | Offset, DL, MVT::i32));
}

SDValue performSHLCombine(SDNode *N, TargetLowering::DAGCombinerInfo &DCI) {
  // Pattern match:
  // $dst = shl (and $src, (2**width - 1)), offset
  // => MAK $dst, $src, width<offset>
  SelectionDAG &DAG = DCI.DAG;
  SDValue FirstOperand = N->getOperand(0);
  unsigned FirstOperandOpc = FirstOperand.getOpcode();
  // First operdns shl must be and, second operand must a constant.
  ConstantSDNode *ShiftAmt = dyn_cast<ConstantSDNode>(N->getOperand(1));
  if (!ShiftAmt || FirstOperandOpc != ISD::AND)
    return SDValue();
  EVT ValTy = N->getValueType(0);
  SDLoc DL(N);

  uint64_t Offset;
  uint64_t MaskWidth, MaskOffset;
  ConstantSDNode *Mask = dyn_cast<ConstantSDNode>(FirstOperand->getOperand(1));
  if (!Mask || !isShiftedMask(Mask->getZExtValue(), MaskWidth, MaskOffset))
    return SDValue();

  // The second operand of the shift must be an immediate.
  Offset = ShiftAmt->getZExtValue();

  // Return if the shifted mask does not start at bit 0 or the sum of its
  // width and offset exceeds the word's size.
  if (MaskOffset != 0 || Offset + MaskWidth > ValTy.getSizeInBits())
    return SDValue();

  return DAG.getNode(M88kISD::MAK, DL, ValTy, FirstOperand.getOperand(0),
                     DAG.getConstant(MaskWidth << 5 | Offset, DL, MVT::i32));
}
} // namespace

SDValue M88kTargetLowering::PerformDAGCombine(SDNode *N,
                                              DAGCombinerInfo &DCI) const {
  if (DCI.isBeforeLegalizeOps())
    return SDValue();
  LLVM_DEBUG(dbgs() << "In PerformDAGCombine\n");

  // TODO: Match certain and/or/shift ops to ext/mak.
  unsigned Opc = N->getOpcode();

  switch (Opc) {
  default:
    break;
  case ISD::AND:
    return performANDCombine(N, DCI);
  case ISD::OR:
    return performORCombine(N, DCI);
  case ISD::SHL:
    return performSHLCombine(N, DCI);
  }

  return SDValue();
}

//===----------------------------------------------------------------------===//
// Calling conventions
//===----------------------------------------------------------------------===//

#include "M88kGenCallingConv.inc"

static SDValue convertLocVTToValVT(SelectionDAG &DAG, const SDLoc &DL,
                                   CCValAssign &VA, SDValue Chain,
                                   SDValue Value) {
  // If the argument has been promoted from a smaller type, insert an
  // assertion to capture this.
  if (VA.getLocInfo() == CCValAssign::SExt)
    Value = DAG.getNode(ISD::AssertSext, DL, VA.getLocVT(), Value,
                        DAG.getValueType(VA.getValVT()));
  else if (VA.getLocInfo() == CCValAssign::ZExt)
    Value = DAG.getNode(ISD::AssertZext, DL, VA.getLocVT(), Value,
                        DAG.getValueType(VA.getValVT()));

  if (VA.isExtInLoc())
    Value = DAG.getNode(ISD::TRUNCATE, DL, VA.getValVT(), Value);
  else if (VA.getLocInfo() == CCValAssign::BCvt) {
    Value = DAG.getNode(ISD::BITCAST, DL, VA.getValVT(), Value);
  } else
    assert(VA.getLocInfo() == CCValAssign::Full && "Unsupported getLocInfo");
  return Value;
}

static SDValue convertValVTToLocVT(SelectionDAG &DAG, const SDLoc &DL,
                                   CCValAssign &VA, SDValue Value) {
  switch (VA.getLocInfo()) {
  case CCValAssign::SExt:
    return DAG.getNode(ISD::SIGN_EXTEND, DL, VA.getLocVT(), Value);
  case CCValAssign::ZExt:
    return DAG.getNode(ISD::ZERO_EXTEND, DL, VA.getLocVT(), Value);
  case CCValAssign::AExt:
    return DAG.getNode(ISD::ANY_EXTEND, DL, VA.getLocVT(), Value);
  case CCValAssign::Full:
    return Value;
  default:
    llvm_unreachable("Unhandled getLocInfo()");
  }
}

SDValue M88kTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  EVT PtrVT = getPointerTy(DAG.getDataLayout());

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());
  CCInfo.AnalyzeFormalArguments(Ins, CC_M88k);

  for (CCValAssign &VA : ArgLocs) {
    SDValue ArgValue;
    EVT LocVT = VA.getLocVT();
    if (VA.isRegLoc()) {
      // Arguments passed in registers.
      const TargetRegisterClass *RC;
      switch (LocVT.getSimpleVT().SimpleTy) {
      default:
        // Integers smaller than i32 should be promoted to i32.
        llvm_unreachable("Unexpected argument type");
      case MVT::i32:
        RC = &M88k::GPRRegClass;
        break;
      case MVT::f32:
        RC = &M88k::GPRRegClass;
        break;
      }

      Register VReg = MRI.createVirtualRegister(RC);
      MRI.addLiveIn(VA.getLocReg(), VReg);
      ArgValue = DAG.getCopyFromReg(Chain, DL, VReg, LocVT);
    } else {
      assert(VA.isMemLoc() && "Argument not register or memory");

      // Create the frame index object for this incoming parameter.
      int FI = MFI.CreateFixedObject(LocVT.getSizeInBits() / 8,
                                     VA.getLocMemOffset(), true);

      // Create the SelectionDAG nodes corresponding to a load
      // from this parameter.
      SDValue FIN = DAG.getFrameIndex(FI, PtrVT);
      ArgValue = DAG.getLoad(LocVT, DL, Chain, FIN,
                             MachinePointerInfo::getFixedStack(MF, FI));
    }

    InVals.push_back(convertLocVTToValVT(DAG, DL, VA, Chain, ArgValue));
  }

  if (IsVarArg) {
    llvm_unreachable("M88k - LowerFormalArguments - VarArgs not Implemented");
  }

  return Chain;
}

SDValue
M88kTargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                                bool IsVarArg,
                                const SmallVectorImpl<ISD::OutputArg> &Outs,
                                const SmallVectorImpl<SDValue> &OutVals,
                                const SDLoc &DL, SelectionDAG &DAG) const {

  MachineFunction &MF = DAG.getMachineFunction();

  // Assign locations to each returned value.
  SmallVector<CCValAssign, 16> RetLocs;
  CCState RetCCInfo(CallConv, IsVarArg, MF, RetLocs, *DAG.getContext());
  RetCCInfo.AnalyzeReturn(Outs, RetCC_M88k);

  // Quick exit for void returns
  if (RetLocs.empty())
    return DAG.getNode(M88kISD::RET_FLAG, DL, MVT::Other, Chain);

  SDValue Glue;
  SmallVector<SDValue, 4> RetOps;
  RetOps.push_back(Chain);
  for (unsigned I = 0, E = RetLocs.size(); I != E; ++I) {
    CCValAssign &VA = RetLocs[I];
    SDValue RetValue = OutVals[I];

    // Make the return register live on exit.
    assert(VA.isRegLoc() && "Can only return in registers!");

    // Promote the value as required.
    RetValue = convertValVTToLocVT(DAG, DL, VA, RetValue);

    // Chain and glue the copies together.
    Register Reg = VA.getLocReg();
    Chain = DAG.getCopyToReg(Chain, DL, Reg, RetValue, Glue);
    Glue = Chain.getValue(1);
    RetOps.push_back(DAG.getRegister(Reg, VA.getLocVT()));
  }

  // Update chain and glue.
  RetOps[0] = Chain;
  if (Glue.getNode())
    RetOps.push_back(Glue);

  return DAG.getNode(M88kISD::RET_FLAG, DL, MVT::Other, RetOps);
}

SDValue M88kTargetLowering::LowerCall(CallLoweringInfo &CLI,
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
  MachineFunction &MF = DAG.getMachineFunction();
  EVT PtrVT = getPointerTy(MF.getDataLayout());
  LLVMContext &Ctx = *DAG.getContext();

  // Analyze the operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState ArgCCInfo(CallConv, IsVarArg, MF, ArgLocs, Ctx);
  ArgCCInfo.AnalyzeCallOperands(Outs, CC_M88k);

  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NumBytes = ArgCCInfo.getNextStackOffset();

  Chain = DAG.getCALLSEQ_START(Chain, NumBytes, 0, DL);

  // Copy argument values to their designated locations.
  SmallVector<std::pair<unsigned, SDValue>, 9> RegsToPass;
  SmallVector<SDValue, 8> MemOpChains;
  SDValue StackPtr;
  for (unsigned I = 0, E = ArgLocs.size(); I != E; ++I) {
    CCValAssign &VA = ArgLocs[I];
    SDValue ArgValue = OutVals[I];

    assert(VA.getLocInfo() != CCValAssign::Indirect &&
           "Indirect args not yet supported");
    ArgValue = convertValVTToLocVT(DAG, DL, VA, ArgValue);

    if (VA.isRegLoc())
      // Queue up the argument copies and emit them at the end.
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), ArgValue));
    else {
      assert(VA.isMemLoc() && "Argument not register or memory");

      if (!StackPtr.getNode())
        StackPtr = DAG.getCopyFromReg(Chain, DL, M88k::R31, PtrVT);

      SDValue Address =
          DAG.getNode(ISD::ADD, DL, PtrVT, StackPtr,
                      DAG.getIntPtrConstant(VA.getLocMemOffset(), DL));

      // Emit the store.
      MemOpChains.push_back(
          DAG.getStore(Chain, DL, ArgValue, Address, MachinePointerInfo()));
    }
  }

  // Transform all store nodes into one single node because all store nodes are
  // independent of each other.
  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, MemOpChains);

  // If the callee is a GlobalAddress node (quite common, every direct call is)
  // turn it into a TargetGlobalAddress node so that legalize doesn't hack it.
  // Likewise ExternalSymbol -> TargetExternalSymbol.
  if (auto *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
    Callee = DAG.getTargetGlobalAddress(G->getGlobal(), DL, PtrVT);
  } else if (auto *E = dyn_cast<ExternalSymbolSDNode>(Callee)) {
    Callee = DAG.getTargetExternalSymbol(E->getSymbol(), PtrVT);
  }

  // Build a sequence of copy-to-reg nodes chained together with token chain and
  // flag operands which copy the outgoing args into registers.The Glue in
  // necessary since all emitted instructions must be stuck together.
  SDValue Glue;
  for (unsigned I = 0, E = RegsToPass.size(); I != E; ++I) {
    Chain = DAG.getCopyToReg(Chain, DL, RegsToPass[I].first,
                             RegsToPass[I].second, Glue);
    Glue = Chain.getValue(1);
  }

  // The first call operand is the chain and the second is the target address.
  SmallVector<SDValue, 8> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  // Add a register mask operand representing the call-preserved registers.
  const TargetRegisterInfo *TRI = Subtarget.getRegisterInfo();
  const uint32_t *Mask = TRI->getCallPreservedMask(MF, CallConv);
  assert(Mask && "Missing call preserved mask for calling convention");
  Ops.push_back(DAG.getRegisterMask(Mask));

  // Add argument registers to the end of the list so that they are
  // known live into the call.
  for (unsigned I = 0, E = RegsToPass.size(); I != E; ++I)
    Ops.push_back(DAG.getRegister(RegsToPass[I].first,
                                  RegsToPass[I].second.getValueType()));

  // Glue the call to the argument copies, if any.
  if (Glue.getNode())
    Ops.push_back(Glue);

  // Emit the call.
  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);
  Chain = DAG.getNode(M88kISD::CALL, DL, NodeTys, Ops);
  DAG.addNoMergeSiteInfo(Chain.getNode(), CLI.NoMerge);
  Glue = Chain.getValue(1);

  // Mark the end of the call, which is glued to the call itself.
  Chain = DAG.getCALLSEQ_END(Chain,
                             DAG.getConstant(NumBytes, DL, PtrVT, true),
                             DAG.getConstant(0, DL, PtrVT, true),
                             Glue, DL);
  Glue = Chain.getValue(1);

  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RetLocs;
  CCState RetCCInfo(CallConv, IsVarArg, MF, RetLocs, Ctx);
  RetCCInfo.AnalyzeCallResult(Ins, RetCC_M88k);

  // Copy all of the result registers out of their specified physreg.
  for (unsigned I = 0, E = RetLocs.size(); I != E; ++I) {
    CCValAssign &VA = RetLocs[I];

    // Copy the value out, gluing the copy to the end of the call sequence.
    SDValue RetValue = DAG.getCopyFromReg(Chain, DL, VA.getLocReg(),
                                          VA.getLocVT(), Glue);
    Chain = RetValue.getValue(1);
    Glue = RetValue.getValue(2);

    // Convert the value of the return register into the value that's
    // being returned.
    InVals.push_back(convertLocVTToValVT(DAG, DL, VA, Chain, RetValue));
  }
  return Chain;

//  llvm_unreachable("Not yet implemented");
}

const char *M88kTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
#define OPCODE(Opc)                                                            \
  case Opc:                                                                    \
    return #Opc
    OPCODE(M88kISD::RET_FLAG);
    OPCODE(M88kISD::CALL);
    OPCODE(M88kISD::Hi16);
    OPCODE(M88kISD::Lo16);
    OPCODE(M88kISD::CLR);
    OPCODE(M88kISD::SET);
    OPCODE(M88kISD::EXT);
    OPCODE(M88kISD::EXTU);
    OPCODE(M88kISD::MAK);
    OPCODE(M88kISD::ROT);
    OPCODE(M88kISD::FF1);
    OPCODE(M88kISD::FF0);
#undef OPCODE
  default:
    return nullptr;
  }
}

// TODO Do I need it?
SDValue M88kTargetLowering::getTargetNode(GlobalAddressSDNode *N, EVT Ty,
                                          SelectionDAG &DAG,
                                          unsigned Flag) const {
  return DAG.getTargetGlobalAddress(N->getGlobal(), SDLoc(N), Ty, 0, Flag);
}

SDValue M88kTargetLowering::LowerGlobalAddress(SDValue Op,
                                               SelectionDAG &DAG) const {
  EVT Ty = Op.getValueType();
  GlobalAddressSDNode *N = cast<GlobalAddressSDNode>(Op);
  //const GlobalValue *GV = N->getGlobal();

  if (!isPositionIndependent()) {
    SDLoc DL(N);
    SDValue Hi = getTargetNode(N, Ty, DAG, M88kII::MO_ABS_HI);
    SDValue Lo = getTargetNode(N, Ty, DAG, M88kII::MO_ABS_LO);
    return DAG.getNode(ISD::OR, DL, Ty, DAG.getNode(M88kISD::Hi16, DL, Ty, Hi),
                       DAG.getNode(M88kISD::Lo16, DL, Ty, Lo));
  }
  llvm_unreachable("Position-indepentend code not supported");
  return SDValue();
}

SDValue M88kTargetLowering::LowerRETURNADDR(SDValue Op,
                                            SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  //MVT VT = Op.getSimpleValueType();
  MFI.setReturnAddressIsTaken(true);

  // Return R1, which contains the return address. Mark it an implicit live-in.
  unsigned Reg = MF.addLiveIn(M88k::R1, getRegClassFor(MVT::i32));
  return DAG.getCopyFromReg(DAG.getEntryNode(), SDLoc(Op), Reg, MVT::i32);
}
