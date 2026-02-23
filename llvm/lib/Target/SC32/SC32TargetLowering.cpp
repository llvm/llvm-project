#include "SC32TargetLowering.h"
#include "MCTargetDesc/SC32MCTargetDesc.h"
#include "SC32RegisterInfo.h"
#include "SC32SelectionDAGInfo.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

using namespace llvm;

#include "SC32GenCallingConv.inc"

SC32TargetLowering::SC32TargetLowering(const TargetMachine &TM,
                                       const TargetSubtargetInfo &STI)
    : TargetLowering(TM, STI) {
  addRegisterClass(MVT::i32, &SC32::GPRegClass);

  setOperationAction(ISD::UREM, MVT::i32, Expand);
  setOperationAction(ISD::UDIV, MVT::i32, Expand);
  setOperationAction(ISD::MULHU, MVT::i32, Expand);
  setOperationAction(ISD::MUL, MVT::i32, Expand);

  setOperationAction(ISD::BR_CC, MVT::i32, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::i32, Custom);

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i8, Expand);

  setLoadExtAction(ISD::SEXTLOAD, MVT::i32, MVT::i8, Expand);

  computeRegisterProperties(STI.getRegisterInfo());
}

SDValue SC32TargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  MachineFunction &MF = DAG.getMachineFunction();

  MachineRegisterInfo &RI = MF.getRegInfo();

  EVT PtrVT = getPointerTy(DAG.getDataLayout());
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), ArgLocs,
                 *DAG.getContext());
  CCInfo.AnalyzeFormalArguments(Ins, CC_SC32);

  unsigned IndirectIdx = 0;
  for (size_t I = 0; I < ArgLocs.size(); I++, ++IndirectIdx) {
    if (ArgLocs[I].isRegLoc()) {
      Register VReg = RI.createVirtualRegister(&SC32::GPRegClass);
      RI.addLiveIn(ArgLocs[I].getLocReg(), VReg);
      InVals.push_back(DAG.getCopyFromReg(Chain, DL, VReg, MVT::i32));
    } else {
      // MemLoc
      unsigned Offset = ArgLocs[I].getLocMemOffset();
      int FI = MF.getFrameInfo().CreateFixedObject(4, Offset, true);
      SDValue FIPtr = DAG.getFrameIndex(FI, PtrVT);
      SDValue Load = DAG.getLoad(ArgLocs[I].getLocVT(), DL, Chain, FIPtr,
                                 MachinePointerInfo::getFixedStack(MF, FI));
      if (ArgLocs[I].getLocInfo() != CCValAssign::Indirect) {
        // Direct load
        InVals.push_back(Load);
        continue;
      }
      assert(ArgLocs[I].getLocInfo() == CCValAssign::Indirect);
      // Indrect load
      SDValue ArgValue = DAG.getLoad(ArgLocs[I].getValVT(), DL, Chain, Load,
                                     MachinePointerInfo());
      InVals.push_back(ArgValue);

      unsigned ArgIndex = Ins[IndirectIdx].OrigArgIndex;
      assert(Ins[IndirectIdx].PartOffset == 0);

      while (I + 1 != ArgLocs.size() &&
             Ins[IndirectIdx + 1].OrigArgIndex == ArgIndex) {
        CCValAssign &PartVA = ArgLocs[I + 1];
        unsigned PartOffset = Ins[IndirectIdx + 1].PartOffset;
        SDValue Address = DAG.getMemBasePlusOffset(
            ArgValue, TypeSize::getFixed(PartOffset), DL);
        InVals.push_back(DAG.getLoad(PartVA.getValVT(), DL, Chain, Address,
                                     MachinePointerInfo()));
        ++I;
        ++IndirectIdx;
      }
    }
  }

  return Chain;
}

SDValue
SC32TargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                                bool IsVarArg,
                                const SmallVectorImpl<ISD::OutputArg> &Outs,
                                const SmallVectorImpl<SDValue> &OutVals,
                                const SDLoc &DL, SelectionDAG &DAG) const {
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), RVLocs,
                 *DAG.getContext());
  CCInfo.AnalyzeReturn(Outs, RetCC_SC32);

  SmallVector<SDValue, 2> RetOps = {SDValue()};
  SDValue Glue;

  for (size_t I = 0; I < RVLocs.size(); I++) {
    Register Reg = RVLocs[I].getLocReg();
    Chain = DAG.getCopyToReg(Chain, DL, Reg, OutVals[I], Glue);
    Glue = Chain.getValue(1);
    RetOps.push_back(DAG.getRegister(Reg, MVT::i32));
  }

  RetOps[0] = Chain;

  if (Glue.getNode()) {
    RetOps.push_back(Glue);
  }

  return DAG.getNode(SC32ISD::RET_GLUE, DL, MVT::Other, RetOps);
}

SDValue SC32TargetLowering::LowerCall(CallLoweringInfo &CLI,
                                      SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG = CLI.DAG;
  SDLoc &DL = CLI.DL;
  SDValue Chain = CLI.Chain;
  SDValue Callee = CLI.Callee;
  CallingConv::ID CallConv = CLI.CallConv;
  bool IsVarArg = CLI.IsVarArg;

  CLI.IsTailCall = false;

  MachineFunction &MF = DAG.getMachineFunction();
  LLVMContext &Context = *DAG.getContext();

  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee))
    Callee = DAG.getTargetGlobalAddress(G->getGlobal(), DL, MVT::i32);
  else if (ExternalSymbolSDNode *E = dyn_cast<ExternalSymbolSDNode>(Callee))
    Callee = DAG.getTargetExternalSymbol(E->getSymbol(), MVT::i32);

  SmallVector<SDValue, 8> MemOpChains;
  SmallVector<SDValue, 8> Ops = {SDValue(), Callee};
  SDValue Glue;

  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, Context);
  CCInfo.AnalyzeCallOperands(CLI.Outs, CC_SC32);

  for (size_t I = 0; I < ArgLocs.size(); I++) {
    if (ArgLocs[I].isRegLoc()) {
      Register Reg = ArgLocs[I].getLocReg();
      Chain = DAG.getCopyToReg(Chain, DL, Reg, CLI.OutVals[I], Glue);
      Glue = Chain.getValue(1);
      Ops.push_back(DAG.getRegister(Reg, MVT::i32));
    } else {
      unsigned Offset = ArgLocs[I].getLocMemOffset();
      SDValue StackPtr = DAG.getRegister(SC32::GP29, MVT::i32);
      SDValue PtrOff = DAG.getIntPtrConstant(Offset, DL);
      PtrOff = DAG.getNode(ISD::ADD, DL, MVT::i32, StackPtr, PtrOff);
      MemOpChains.push_back(DAG.getStore(Chain, DL, CLI.OutVals[I], PtrOff,
                                         MachinePointerInfo()));
    }
  }

  Ops[0] = Chain;

  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
  const uint32_t *Mask = TRI.getCallPreservedMask(MF, CallConv);

  Ops.push_back(DAG.getRegisterMask(Mask));

  if (Glue.getNode()) {
    Ops.push_back(Glue);
  }

  Chain = DAG.getNode(SC32ISD::CALL, DL, {MVT::Other, MVT::Glue}, Ops);
  Glue = Chain.getValue(1);

  SmallVector<CCValAssign, 16> RVLocs;
  CCState RVInfo(CallConv, IsVarArg, MF, RVLocs, Context);
  RVInfo.AnalyzeCallResult(CLI.Ins, RetCC_SC32);

  if (!MemOpChains.empty()) {
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, MemOpChains);
  }
  for (size_t I = 0; I < RVLocs.size(); I++) {
    Register Reg = RVLocs[I].getLocReg();
    Chain = DAG.getCopyFromReg(Chain, DL, Reg, MVT::i32, Glue).getValue(1);
    Glue = Chain.getValue(2);
    InVals.push_back(Chain.getValue(0));
  }

  return Chain;
}

static SDValue LowerBR_CC(SDValue Op, SelectionDAG &DAG) {
  SDValue Chain = Op.getOperand(0);
  SDValue CC = Op.getOperand(1);
  SDValue LHS = Op.getOperand(2);
  SDValue RHS = Op.getOperand(3);
  SDValue Dest = Op.getOperand(4);

  SDLoc DL(Op);

  SDValue Glue = DAG.getNode(SC32ISD::CMP, DL, MVT::Glue, LHS, RHS);
  return DAG.getNode(SC32ISD::JMP, DL, MVT::Other, Chain, CC, Dest, Glue);
}

static SDValue LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) {
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  SDValue True = Op.getOperand(2);
  SDValue False = Op.getOperand(3);
  SDValue CC = Op.getOperand(4);

  SDLoc DL(Op);

  SDValue Glue = DAG.getNode(SC32ISD::CMP, DL, MVT::Glue, LHS, RHS);
  return DAG.getNode(SC32ISD::SELECT, DL, MVT::i32, CC, True, False, Glue);
}

SDValue SC32TargetLowering::LowerOperation(SDValue Op,
                                           SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  default:
    llvm_unreachable("Should not custom lower this!");
  case ISD::BR_CC:
    return LowerBR_CC(Op, DAG);
  case ISD::SELECT_CC:
    return LowerSELECT_CC(Op, DAG);
  }
}

static MachineBasicBlock *expandSelect(MachineInstr &MI, MachineBasicBlock *MBB,
                                       unsigned Jmp) {
  MachineFunction &MF = *MBB->getParent();
  MachineFunction::iterator I = std::next(MBB->getIterator());
  DebugLoc DL = MI.getDebugLoc();

  const BasicBlock *LLVM_BB = MBB->getBasicBlock();

  MachineBasicBlock *FalseMBB = MF.CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *SinkMBB = MF.CreateMachineBasicBlock(LLVM_BB);

  MF.insert(I, FalseMBB);
  MF.insert(I, SinkMBB);

  SinkMBB->splice(SinkMBB->begin(), MBB,
                  std::next(MachineBasicBlock::iterator(MI)), MBB->end());
  SinkMBB->transferSuccessorsAndUpdatePHIs(MBB);

  MBB->addSuccessor(FalseMBB);
  MBB->addSuccessor(SinkMBB);
  FalseMBB->addSuccessor(SinkMBB);

  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();

  BuildMI(MBB, DL, TII.get(Jmp)).addMBB(SinkMBB);

  BuildMI(*SinkMBB, SinkMBB->begin(), DL, TII.get(SC32::PHI),
          MI.getOperand(0).getReg())
      .addReg(MI.getOperand(1).getReg())
      .addMBB(MBB)
      .addReg(MI.getOperand(2).getReg())
      .addMBB(FalseMBB);

  MI.eraseFromParent();

  return SinkMBB;
}

MachineBasicBlock *
SC32TargetLowering::EmitInstrWithCustomInserter(MachineInstr &MI,
                                                MachineBasicBlock *MBB) const {
  switch (MI.getOpcode()) {
  default:
    llvm_unreachable("Should not custom insert this!");
  case SC32::SELECTEQ:
    return expandSelect(MI, MBB, SC32::JEQ);
  case SC32::SELECTNE:
    return expandSelect(MI, MBB, SC32::JNE);
  case SC32::SELECTLE:
    return expandSelect(MI, MBB, SC32::JLE);
  case SC32::SELECTLT:
    return expandSelect(MI, MBB, SC32::JLT);
  case SC32::SELECTGT:
    return expandSelect(MI, MBB, SC32::JGT);
  case SC32::SELECTGE:
    return expandSelect(MI, MBB, SC32::JGE);
  case SC32::SELECTLEU:
    return expandSelect(MI, MBB, SC32::JLEU);
  case SC32::SELECTLTU:
    return expandSelect(MI, MBB, SC32::JLTU);
  case SC32::SELECTGTU:
    return expandSelect(MI, MBB, SC32::JGTU);
  case SC32::SELECTGEU:
    return expandSelect(MI, MBB, SC32::JGEU);
  }
}
