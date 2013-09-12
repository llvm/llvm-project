//===-- SIISelLowering.cpp - SI DAG Lowering Implementation ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Custom DAG lowering for SI
//
//===----------------------------------------------------------------------===//

#include "SIISelLowering.h"
#include "AMDIL.h"
#include "AMDGPU.h"
#include "AMDILIntrinsicInfo.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"

using namespace llvm;

SITargetLowering::SITargetLowering(TargetMachine &TM) :
    AMDGPUTargetLowering(TM),
    TII(static_cast<const SIInstrInfo*>(TM.getInstrInfo())),
    TRI(TM.getRegisterInfo()) {

  addRegisterClass(MVT::i1, &AMDGPU::SReg_64RegClass);
  addRegisterClass(MVT::i64, &AMDGPU::SReg_64RegClass);

  addRegisterClass(MVT::v16i8, &AMDGPU::SReg_128RegClass);
  addRegisterClass(MVT::v32i8, &AMDGPU::SReg_256RegClass);
  addRegisterClass(MVT::v64i8, &AMDGPU::SReg_512RegClass);

  addRegisterClass(MVT::i32, &AMDGPU::VReg_32RegClass);
  addRegisterClass(MVT::f32, &AMDGPU::VReg_32RegClass);

  addRegisterClass(MVT::v1i32, &AMDGPU::VReg_32RegClass);

  addRegisterClass(MVT::v2i32, &AMDGPU::VReg_64RegClass);
  addRegisterClass(MVT::v2f32, &AMDGPU::VReg_64RegClass);

  addRegisterClass(MVT::v4i32, &AMDGPU::VReg_128RegClass);
  addRegisterClass(MVT::v4f32, &AMDGPU::VReg_128RegClass);
  addRegisterClass(MVT::i128, &AMDGPU::SReg_128RegClass);

  addRegisterClass(MVT::v8i32, &AMDGPU::VReg_256RegClass);
  addRegisterClass(MVT::v8f32, &AMDGPU::VReg_256RegClass);

  addRegisterClass(MVT::v16i32, &AMDGPU::VReg_512RegClass);
  addRegisterClass(MVT::v16f32, &AMDGPU::VReg_512RegClass);

  computeRegisterProperties();

  setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v8i32, Expand);
  setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v8f32, Expand);
  setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v16i32, Expand);
  setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v16f32, Expand);

  setOperationAction(ISD::ADD, MVT::i64, Legal);
  setOperationAction(ISD::ADD, MVT::i32, Legal);

  setOperationAction(ISD::SELECT_CC, MVT::f32, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::i32, Custom);

  setOperationAction(ISD::SELECT_CC, MVT::Other, Expand);

  setOperationAction(ISD::STORE, MVT::i32, Custom);
  setOperationAction(ISD::STORE, MVT::i64, Custom);

  setTargetDAGCombine(ISD::SELECT_CC);

  setTargetDAGCombine(ISD::SETCC);

  setSchedulingPreference(Sched::RegPressure);
}

SDValue SITargetLowering::LowerFormalArguments(
                                      SDValue Chain,
                                      CallingConv::ID CallConv,
                                      bool isVarArg,
                                      const SmallVectorImpl<ISD::InputArg> &Ins,
                                      DebugLoc DL, SelectionDAG &DAG,
                                      SmallVectorImpl<SDValue> &InVals) const {

  const TargetRegisterInfo *TRI = getTargetMachine().getRegisterInfo();

  MachineFunction &MF = DAG.getMachineFunction();
  FunctionType *FType = MF.getFunction()->getFunctionType();
  SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();

  assert(CallConv == CallingConv::C);

  SmallVector<ISD::InputArg, 16> Splits;
  uint32_t Skipped = 0;

  for (unsigned i = 0, e = Ins.size(), PSInputNum = 0; i != e; ++i) {
    const ISD::InputArg &Arg = Ins[i];
   
    // First check if it's a PS input addr 
    if (Info->ShaderType == ShaderType::PIXEL && !Arg.Flags.isInReg()) {

      assert((PSInputNum <= 15) && "Too many PS inputs!");

      if (!Arg.Used) {
        // We can savely skip PS inputs
        Skipped |= 1 << i;
        ++PSInputNum;
        continue;
      }

      Info->PSInputAddr |= 1 << PSInputNum++;
    }

    // Second split vertices into their elements
    if (Arg.VT.isVector()) {
      ISD::InputArg NewArg = Arg;
      NewArg.Flags.setSplit();
      NewArg.VT = Arg.VT.getVectorElementType();

      // We REALLY want the ORIGINAL number of vertex elements here, e.g. a
      // three or five element vertex only needs three or five registers,
      // NOT four or eigth.
      Type *ParamType = FType->getParamType(Arg.OrigArgIndex);
      unsigned NumElements = ParamType->getVectorNumElements();

      for (unsigned j = 0; j != NumElements; ++j) {
        Splits.push_back(NewArg);
        NewArg.PartOffset += NewArg.VT.getStoreSize();
      }

    } else {
      Splits.push_back(Arg);
    }
  }

  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), ArgLocs, *DAG.getContext());

  // At least one interpolation mode must be enabled or else the GPU will hang.
  if (Info->ShaderType == ShaderType::PIXEL && (Info->PSInputAddr & 0x7F) == 0) {
    Info->PSInputAddr |= 1;
    CCInfo.AllocateReg(AMDGPU::VGPR0);
    CCInfo.AllocateReg(AMDGPU::VGPR1);
  }

  AnalyzeFormalArguments(CCInfo, Splits);

  for (unsigned i = 0, e = Ins.size(), ArgIdx = 0; i != e; ++i) {

    if (Skipped & (1 << i)) {
      InVals.push_back(SDValue());
      continue;
    }

    CCValAssign &VA = ArgLocs[ArgIdx++];
    assert(VA.isRegLoc() && "Parameter must be in a register!");

    unsigned Reg = VA.getLocReg();
    MVT VT = VA.getLocVT();

    if (VT == MVT::i64) {
      // For now assume it is a pointer
      Reg = TRI->getMatchingSuperReg(Reg, AMDGPU::sub0,
                                     &AMDGPU::SReg_64RegClass);
      Reg = MF.addLiveIn(Reg, &AMDGPU::SReg_64RegClass);
      InVals.push_back(DAG.getCopyFromReg(Chain, DL, Reg, VT));
      continue;
    }

    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg, VT);

    Reg = MF.addLiveIn(Reg, RC);
    SDValue Val = DAG.getCopyFromReg(Chain, DL, Reg, VT);

    const ISD::InputArg &Arg = Ins[i];
    if (Arg.VT.isVector()) {

      // Build a vector from the registers
      Type *ParamType = FType->getParamType(Arg.OrigArgIndex);
      unsigned NumElements = ParamType->getVectorNumElements();

      SmallVector<SDValue, 4> Regs;
      Regs.push_back(Val);
      for (unsigned j = 1; j != NumElements; ++j) {
        Reg = ArgLocs[ArgIdx++].getLocReg();
        Reg = MF.addLiveIn(Reg, RC);
        Regs.push_back(DAG.getCopyFromReg(Chain, DL, Reg, VT));
      }

      // Fill up the missing vector elements
      NumElements = Arg.VT.getVectorNumElements() - NumElements;
      for (unsigned j = 0; j != NumElements; ++j)
        Regs.push_back(DAG.getUNDEF(VT));
 
      InVals.push_back(DAG.getNode(ISD::BUILD_VECTOR, DL, Arg.VT,
                                   Regs.data(), Regs.size()));
      continue;
    }

    InVals.push_back(Val);
  }
  return Chain;
}

MachineBasicBlock * SITargetLowering::EmitInstrWithCustomInserter(
    MachineInstr * MI, MachineBasicBlock * BB) const {

  switch (MI->getOpcode()) {
  default:
    return AMDGPUTargetLowering::EmitInstrWithCustomInserter(MI, BB);
  case AMDGPU::BRANCH: return BB;
  }
  return BB;
}

EVT SITargetLowering::getSetCCResultType(EVT VT) const {
  return MVT::i1;
}

MVT SITargetLowering::getScalarShiftAmountTy(EVT VT) const {
  return MVT::i32;
}

//===----------------------------------------------------------------------===//
// Custom DAG Lowering Operations
//===----------------------------------------------------------------------===//

SDValue SITargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  default: return AMDGPUTargetLowering::LowerOperation(Op, DAG);
  case ISD::BRCOND: return LowerBRCOND(Op, DAG);
  case ISD::SELECT_CC: return LowerSELECT_CC(Op, DAG);
  case ISD::STORE: return LowerSTORE(Op, DAG);
  }
  return SDValue();
}

/// \brief Helper function for LowerBRCOND
static SDNode *findUser(SDValue Value, unsigned Opcode) {

  SDNode *Parent = Value.getNode();
  for (SDNode::use_iterator I = Parent->use_begin(), E = Parent->use_end();
       I != E; ++I) {

    if (I.getUse().get() != Value)
      continue;

    if (I->getOpcode() == Opcode)
      return *I;
  }
  return 0;
}

/// This transforms the control flow intrinsics to get the branch destination as
/// last parameter, also switches branch target with BR if the need arise
SDValue SITargetLowering::LowerBRCOND(SDValue BRCOND,
                                      SelectionDAG &DAG) const {

  DebugLoc DL = BRCOND.getDebugLoc();

  SDNode *Intr = BRCOND.getOperand(1).getNode();
  SDValue Target = BRCOND.getOperand(2);
  SDNode *BR = 0;

  if (Intr->getOpcode() == ISD::SETCC) {
    // As long as we negate the condition everything is fine
    SDNode *SetCC = Intr;
    assert(SetCC->getConstantOperandVal(1) == 1);
    assert(cast<CondCodeSDNode>(SetCC->getOperand(2).getNode())->get() ==
           ISD::SETNE);
    Intr = SetCC->getOperand(0).getNode();

  } else {
    // Get the target from BR if we don't negate the condition
    BR = findUser(BRCOND, ISD::BR);
    Target = BR->getOperand(1);
  }

  assert(Intr->getOpcode() == ISD::INTRINSIC_W_CHAIN);

  // Build the result and
  SmallVector<EVT, 4> Res;
  for (unsigned i = 1, e = Intr->getNumValues(); i != e; ++i)
    Res.push_back(Intr->getValueType(i));

  // operands of the new intrinsic call
  SmallVector<SDValue, 4> Ops;
  Ops.push_back(BRCOND.getOperand(0));
  for (unsigned i = 1, e = Intr->getNumOperands(); i != e; ++i)
    Ops.push_back(Intr->getOperand(i));
  Ops.push_back(Target);

  // build the new intrinsic call
  SDNode *Result = DAG.getNode(
    Res.size() > 1 ? ISD::INTRINSIC_W_CHAIN : ISD::INTRINSIC_VOID, DL,
    DAG.getVTList(Res.data(), Res.size()), Ops.data(), Ops.size()).getNode();

  if (BR) {
    // Give the branch instruction our target
    SDValue Ops[] = {
      BR->getOperand(0),
      BRCOND.getOperand(2)
    };
    DAG.MorphNodeTo(BR, ISD::BR, BR->getVTList(), Ops, 2);
  }

  SDValue Chain = SDValue(Result, Result->getNumValues() - 1);

  // Copy the intrinsic results to registers
  for (unsigned i = 1, e = Intr->getNumValues() - 1; i != e; ++i) {
    SDNode *CopyToReg = findUser(SDValue(Intr, i), ISD::CopyToReg);
    if (!CopyToReg)
      continue;

    Chain = DAG.getCopyToReg(
      Chain, DL,
      CopyToReg->getOperand(1),
      SDValue(Result, i - 1),
      SDValue());

    DAG.ReplaceAllUsesWith(SDValue(CopyToReg, 0), CopyToReg->getOperand(0));
  }

  // Remove the old intrinsic from the chain
  DAG.ReplaceAllUsesOfValueWith(
    SDValue(Intr, Intr->getNumValues() - 1),
    Intr->getOperand(0));

  return Chain;
}

SDValue SITargetLowering::LowerSTORE(SDValue Op, SelectionDAG &DAG) const {
  const uint64_t RSRC_DATA_FORMAT = 0xf00000000000LL;
  StoreSDNode *StoreNode = cast<StoreSDNode>(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue Value = Op.getOperand(1);
  SDValue VirtualAddress = Op.getOperand(2);
  DebugLoc DL = Op.getDebugLoc();

  if (StoreNode->getAddressSpace() != AMDGPUAS::GLOBAL_ADDRESS) {
    return SDValue();
  }

  SDValue Zero = DAG.getConstant(0, MVT::i64);
  SDValue Format = DAG.getConstant(RSRC_DATA_FORMAT, MVT::i64);
  SDValue SrcSrc = DAG.getNode(ISD::BUILD_PAIR, DL, MVT::i128, Zero, Format);

  SDValue Ops[2];
  Ops[0] = DAG.getNode(AMDGPUISD::BUFFER_STORE, DL, MVT::Other, Chain,
                       Value, SrcSrc, VirtualAddress);
  Ops[1] = Chain;

  return DAG.getMergeValues(Ops, 2, DL);

}

SDValue SITargetLowering::LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const {
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  SDValue True = Op.getOperand(2);
  SDValue False = Op.getOperand(3);
  SDValue CC = Op.getOperand(4);
  EVT VT = Op.getValueType();
  DebugLoc DL = Op.getDebugLoc();

  // Possible Min/Max pattern
  SDValue MinMax = LowerMinMax(Op, DAG);
  if (MinMax.getNode()) {
    return MinMax;
  }

  SDValue Cond = DAG.getNode(ISD::SETCC, DL, MVT::i1, LHS, RHS, CC);
  return DAG.getNode(ISD::SELECT, DL, VT, Cond, True, False);
}

//===----------------------------------------------------------------------===//
// Custom DAG optimizations
//===----------------------------------------------------------------------===//

SDValue SITargetLowering::PerformDAGCombine(SDNode *N,
                                            DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  DebugLoc DL = N->getDebugLoc();
  EVT VT = N->getValueType(0);

  switch (N->getOpcode()) {
    default: break;
    case ISD::SELECT_CC: {
      N->dump();
      ConstantSDNode *True, *False;
      // i1 selectcc(l, r, -1, 0, cc) -> i1 setcc(l, r, cc)
      if ((True = dyn_cast<ConstantSDNode>(N->getOperand(2)))
          && (False = dyn_cast<ConstantSDNode>(N->getOperand(3)))
          && True->isAllOnesValue()
          && False->isNullValue()
          && VT == MVT::i1) {
        return DAG.getNode(ISD::SETCC, DL, VT, N->getOperand(0),
                           N->getOperand(1), N->getOperand(4));

      }
      break;
    }
    case ISD::SETCC: {
      SDValue Arg0 = N->getOperand(0);
      SDValue Arg1 = N->getOperand(1);
      SDValue CC = N->getOperand(2);
      ConstantSDNode * C = NULL;
      ISD::CondCode CCOp = dyn_cast<CondCodeSDNode>(CC)->get();

      // i1 setcc (sext(i1), 0, setne) -> i1 setcc(i1, 0, setne)
      if (VT == MVT::i1
          && Arg0.getOpcode() == ISD::SIGN_EXTEND
          && Arg0.getOperand(0).getValueType() == MVT::i1
          && (C = dyn_cast<ConstantSDNode>(Arg1))
          && C->isNullValue()
          && CCOp == ISD::SETNE) {
        return SimplifySetCC(VT, Arg0.getOperand(0),
                             DAG.getConstant(0, MVT::i1), CCOp, true, DCI, DL);
      }
      break;
    }
  }
  return SDValue();
}

/// \brief Test if RegClass is one of the VSrc classes 
static bool isVSrc(unsigned RegClass) {
  return AMDGPU::VSrc_32RegClassID == RegClass ||
         AMDGPU::VSrc_64RegClassID == RegClass;
}

/// \brief Test if RegClass is one of the SSrc classes 
static bool isSSrc(unsigned RegClass) {
  return AMDGPU::SSrc_32RegClassID == RegClass ||
         AMDGPU::SSrc_64RegClassID == RegClass;
}

/// \brief Analyze the possible immediate value Op
///
/// Returns -1 if it isn't an immediate, 0 if it's and inline immediate
/// and the immediate value if it's a literal immediate
int32_t SITargetLowering::analyzeImmediate(const SDNode *N) const {

  union {
    int32_t I;
    float F;
  } Imm;

  if (const ConstantSDNode *Node = dyn_cast<ConstantSDNode>(N)) {
    if (Node->getZExtValue() >> 32) {
        return -1;
    }
    Imm.I = Node->getSExtValue();
  } else if (const ConstantFPSDNode *Node = dyn_cast<ConstantFPSDNode>(N))
    Imm.F = Node->getValueAPF().convertToFloat();
  else
    return -1; // It isn't an immediate

  if ((Imm.I >= -16 && Imm.I <= 64) ||
      Imm.F == 0.5f || Imm.F == -0.5f ||
      Imm.F == 1.0f || Imm.F == -1.0f ||
      Imm.F == 2.0f || Imm.F == -2.0f ||
      Imm.F == 4.0f || Imm.F == -4.0f)
    return 0; // It's an inline immediate

  return Imm.I; // It's a literal immediate
}

/// \brief Try to fold an immediate directly into an instruction
bool SITargetLowering::foldImm(SDValue &Operand, int32_t &Immediate,
                               bool &ScalarSlotUsed) const {

  MachineSDNode *Mov = dyn_cast<MachineSDNode>(Operand);
  if (Mov == 0 || !TII->isMov(Mov->getMachineOpcode()))
    return false;

  const SDValue &Op = Mov->getOperand(0);
  int32_t Value = analyzeImmediate(Op.getNode());
  if (Value == -1) {
    // Not an immediate at all
    return false;

  } else if (Value == 0) {
    // Inline immediates can always be fold
    Operand = Op;
    return true;

  } else if (Value == Immediate) {
    // Already fold literal immediate
    Operand = Op;
    return true;

  } else if (!ScalarSlotUsed && !Immediate) {
    // Fold this literal immediate
    ScalarSlotUsed = true;
    Immediate = Value;
    Operand = Op;
    return true;

  }

  return false;
}

/// \brief Does "Op" fit into register class "RegClass" ?
bool SITargetLowering::fitsRegClass(SelectionDAG &DAG, SDValue &Op,
                                    unsigned RegClass) const {

  MachineRegisterInfo &MRI = DAG.getMachineFunction().getRegInfo(); 
  SDNode *Node = Op.getNode();

  const TargetRegisterClass *OpClass;
  if (MachineSDNode *MN = dyn_cast<MachineSDNode>(Node)) {
    const MCInstrDesc &Desc = TII->get(MN->getMachineOpcode());
    int OpClassID = Desc.OpInfo[Op.getResNo()].RegClass;
    if (OpClassID == -1)
      OpClass = getRegClassFor(Op.getSimpleValueType());
    else
      OpClass = TRI->getRegClass(OpClassID);

  } else if (Node->getOpcode() == ISD::CopyFromReg) {
    RegisterSDNode *Reg = cast<RegisterSDNode>(Node->getOperand(1).getNode());
    OpClass = MRI.getRegClass(Reg->getReg());

  } else
    return false;

  return TRI->getRegClass(RegClass)->hasSubClassEq(OpClass);
}

/// \brief Make sure that we don't exeed the number of allowed scalars
void SITargetLowering::ensureSRegLimit(SelectionDAG &DAG, SDValue &Operand,
                                       unsigned RegClass,
                                       bool &ScalarSlotUsed) const {

  // First map the operands register class to a destination class
  if (RegClass == AMDGPU::VSrc_32RegClassID)
    RegClass = AMDGPU::VReg_32RegClassID;
  else if (RegClass == AMDGPU::VSrc_64RegClassID)
    RegClass = AMDGPU::VReg_64RegClassID;
  else
    return;

  // Nothing todo if they fit naturaly
  if (fitsRegClass(DAG, Operand, RegClass))
    return;

  // If the scalar slot isn't used yet use it now
  if (!ScalarSlotUsed) {
    ScalarSlotUsed = true;
    return;
  }

  // This is a conservative aproach, it is possible that we can't determine
  // the correct register class and copy too often, but better save than sorry.
  SDValue RC = DAG.getTargetConstant(RegClass, MVT::i32);
  SDNode *Node = DAG.getMachineNode(TargetOpcode::COPY_TO_REGCLASS, DebugLoc(),
                                    Operand.getValueType(), Operand, RC);
  Operand = SDValue(Node, 0);
}

/// \brief Try to fold the Nodes operands into the Node
SDNode *SITargetLowering::foldOperands(MachineSDNode *Node,
                                       SelectionDAG &DAG) const {

  // Original encoding (either e32 or e64)
  int Opcode = Node->getMachineOpcode();
  const MCInstrDesc *Desc = &TII->get(Opcode);

  unsigned NumDefs = Desc->getNumDefs();
  unsigned NumOps = Desc->getNumOperands();

  // Commuted opcode if available
  int OpcodeRev = Desc->isCommutable() ? TII->commuteOpcode(Opcode) : -1;
  const MCInstrDesc *DescRev = OpcodeRev == -1 ? 0 : &TII->get(OpcodeRev);

  assert(!DescRev || DescRev->getNumDefs() == NumDefs);
  assert(!DescRev || DescRev->getNumOperands() == NumOps);

  // e64 version if available, -1 otherwise
  int OpcodeE64 = AMDGPU::getVOPe64(Opcode);
  const MCInstrDesc *DescE64 = OpcodeE64 == -1 ? 0 : &TII->get(OpcodeE64);

  assert(!DescE64 || DescE64->getNumDefs() == NumDefs);
  assert(!DescE64 || DescE64->getNumOperands() == (NumOps + 4));

  int32_t Immediate = Desc->getSize() == 4 ? 0 : -1;
  bool HaveVSrc = false, HaveSSrc = false;

  // First figure out what we alread have in this instruction
  for (unsigned i = 0, e = Node->getNumOperands(), Op = NumDefs;
       i != e && Op < NumOps; ++i, ++Op) {

    unsigned RegClass = Desc->OpInfo[Op].RegClass;
    if (isVSrc(RegClass))
      HaveVSrc = true;
    else if (isSSrc(RegClass))
      HaveSSrc = true;
    else
      continue;

    int32_t Imm = analyzeImmediate(Node->getOperand(i).getNode());
    if (Imm != -1 && Imm != 0) {
      // Literal immediate
      Immediate = Imm;
    }
  }

  // If we neither have VSrc nor SSrc it makes no sense to continue
  if (!HaveVSrc && !HaveSSrc)
    return Node;

  // No scalar allowed when we have both VSrc and SSrc
  bool ScalarSlotUsed = HaveVSrc && HaveSSrc;

  // Second go over the operands and try to fold them
  std::vector<SDValue> Ops;
  bool Promote2e64 = false;
  for (unsigned i = 0, e = Node->getNumOperands(), Op = NumDefs;
       i != e && Op < NumOps; ++i, ++Op) {

    const SDValue &Operand = Node->getOperand(i);
    Ops.push_back(Operand);

    // Already folded immediate ?
    if (isa<ConstantSDNode>(Operand.getNode()) ||
        isa<ConstantFPSDNode>(Operand.getNode()))
      continue;

    // Is this a VSrc or SSrc operand ?
    unsigned RegClass = Desc->OpInfo[Op].RegClass;
    if (isVSrc(RegClass) || isSSrc(RegClass)) {
      // Try to fold the immediates
      if (!foldImm(Ops[i], Immediate, ScalarSlotUsed)) {
        // Folding didn't worked, make sure we don't hit the SReg limit
        ensureSRegLimit(DAG, Ops[i], RegClass, ScalarSlotUsed);
      }
      continue;
    }

    if (i == 1 && DescRev && fitsRegClass(DAG, Ops[0], RegClass)) {

      unsigned OtherRegClass = Desc->OpInfo[NumDefs].RegClass;
      assert(isVSrc(OtherRegClass) || isSSrc(OtherRegClass));

      // Test if it makes sense to swap operands
      if (foldImm(Ops[1], Immediate, ScalarSlotUsed) ||
          (!fitsRegClass(DAG, Ops[1], RegClass) &&
           fitsRegClass(DAG, Ops[1], OtherRegClass))) {

        // Swap commutable operands
        SDValue Tmp = Ops[1];
        Ops[1] = Ops[0];
        Ops[0] = Tmp;

        Desc = DescRev;
        DescRev = 0;
        continue;
      }
    }

    if (DescE64 && !Immediate) {

      // Test if it makes sense to switch to e64 encoding
      unsigned OtherRegClass = DescE64->OpInfo[Op].RegClass;
      if (!isVSrc(OtherRegClass) && !isSSrc(OtherRegClass))
        continue;

      int32_t TmpImm = -1;
      if (foldImm(Ops[i], TmpImm, ScalarSlotUsed) ||
          (!fitsRegClass(DAG, Ops[i], RegClass) &&
           fitsRegClass(DAG, Ops[1], OtherRegClass))) {

        // Switch to e64 encoding
        Immediate = -1;
        Promote2e64 = true;
        Desc = DescE64;
        DescE64 = 0;
      }
    }
  }

  if (Promote2e64) {
    // Add the modifier flags while promoting
    for (unsigned i = 0; i < 4; ++i)
      Ops.push_back(DAG.getTargetConstant(0, MVT::i32));
  }

  // Add optional chain and glue
  for (unsigned i = NumOps - NumDefs, e = Node->getNumOperands(); i < e; ++i)
    Ops.push_back(Node->getOperand(i));

  // Create a complete new instruction
  return DAG.getMachineNode(Desc->Opcode, Node->getDebugLoc(),
                            Node->getVTList(), Ops);
}

/// \brief Helper function for adjustWritemask
unsigned SubIdx2Lane(unsigned Idx) {
  switch (Idx) {
  default: return 0;
  case AMDGPU::sub0: return 0;
  case AMDGPU::sub1: return 1;
  case AMDGPU::sub2: return 2;
  case AMDGPU::sub3: return 3;
  }
}

/// \brief Adjust the writemask of MIMG instructions
void SITargetLowering::adjustWritemask(MachineSDNode *&Node,
                                       SelectionDAG &DAG) const {
  SDNode *Users[4] = { };
  unsigned Writemask = 0, Lane = 0;

  // Try to figure out the used register components
  for (SDNode::use_iterator I = Node->use_begin(), E = Node->use_end();
       I != E; ++I) {

    // Abort if we can't understand the usage
    if (!I->isMachineOpcode() ||
        I->getMachineOpcode() != TargetOpcode::EXTRACT_SUBREG)
      return;

    Lane = SubIdx2Lane(I->getConstantOperandVal(1));

    // Abort if we have more than one user per component
    if (Users[Lane])
      return;

    Users[Lane] = *I;
    Writemask |= 1 << Lane;
  }

  // Abort if all components are used
  if (Writemask == 0xf)
    return;

  // Adjust the writemask in the node
  std::vector<SDValue> Ops;
  Ops.push_back(DAG.getTargetConstant(Writemask, MVT::i32));
  for (unsigned i = 1, e = Node->getNumOperands(); i != e; ++i)
    Ops.push_back(Node->getOperand(i));
  Node = (MachineSDNode*)DAG.UpdateNodeOperands(Node, Ops.data(), Ops.size());

  // If we only got one lane, replace it with a copy
  if (Writemask == (1U << Lane)) {
    SDValue RC = DAG.getTargetConstant(AMDGPU::VReg_32RegClassID, MVT::i32);
    SDNode *Copy = DAG.getMachineNode(TargetOpcode::COPY_TO_REGCLASS,
                                      DebugLoc(), Users[Lane]->getValueType(0),
                                      SDValue(Node, 0), RC);
    DAG.ReplaceAllUsesWith(Users[Lane], Copy);
    return;
  }

  // Update the users of the node with the new indices
  for (unsigned i = 0, Idx = AMDGPU::sub0; i < 4; ++i) {

    SDNode *User = Users[i];
    if (!User)
      continue;

    SDValue Op = DAG.getTargetConstant(Idx, MVT::i32);
    DAG.UpdateNodeOperands(User, User->getOperand(0), Op);

    switch (Idx) {
    default: break;
    case AMDGPU::sub0: Idx = AMDGPU::sub1; break;
    case AMDGPU::sub1: Idx = AMDGPU::sub2; break;
    case AMDGPU::sub2: Idx = AMDGPU::sub3; break;
    }
  }
}

/// \brief Fold the instructions after slecting them
SDNode *SITargetLowering::PostISelFolding(MachineSDNode *Node,
                                          SelectionDAG &DAG) const {

  if (AMDGPU::isMIMG(Node->getMachineOpcode()) != -1)
    adjustWritemask(Node, DAG);

  return foldOperands(Node, DAG);
}

/// \brief Assign the register class depending on the number of
/// bits set in the writemask
void SITargetLowering::AdjustInstrPostInstrSelection(MachineInstr *MI,
                                                     SDNode *Node) const {
  if (AMDGPU::isMIMG(MI->getOpcode()) == -1)
    return;

  unsigned VReg = MI->getOperand(0).getReg();
  unsigned Writemask = MI->getOperand(1).getImm();
  unsigned BitsSet = 0;
  for (unsigned i = 0; i < 4; ++i)
    BitsSet += Writemask & (1 << i) ? 1 : 0;

  const TargetRegisterClass *RC;
  switch (BitsSet) {
  default: return;
  case 1:  RC = &AMDGPU::VReg_32RegClass; break;
  case 2:  RC = &AMDGPU::VReg_64RegClass; break;
  case 3:  RC = &AMDGPU::VReg_96RegClass; break;
  }

  MachineRegisterInfo &MRI = MI->getParent()->getParent()->getRegInfo();
  MRI.setRegClass(VReg, RC);
}
