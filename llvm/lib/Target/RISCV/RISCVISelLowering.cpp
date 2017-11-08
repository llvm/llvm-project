//===-- RISCVISelLowering.cpp - RISCV DAG Lowering Implementation  --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that RISCV uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#include "RISCVISelLowering.h"
#include "RISCV.h"
#include "RISCVRegisterInfo.h"
#include "RISCVSubtarget.h"
#include "RISCVTargetMachine.h"
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
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-lower"

RISCVTargetLowering::RISCVTargetLowering(const TargetMachine &TM,
                                         const RISCVSubtarget &STI)
    : TargetLowering(TM), Subtarget(STI) {

  MVT XLenVT = Subtarget.getXLenVT();

  // Set up the register classes.
  addRegisterClass(XLenVT, &RISCV::GPRRegClass);

  // Compute derived properties from the register classes.
  computeRegisterProperties(STI.getRegisterInfo());

  setStackPointerRegisterToSaveRestore(RISCV::X2);

  for (auto N : {ISD::EXTLOAD, ISD::SEXTLOAD, ISD::ZEXTLOAD})
    setLoadExtAction(N, XLenVT, MVT::i1, Promote);

  // TODO: add all necessary setOperationAction calls.
  setOperationAction(ISD::GlobalAddress, XLenVT, Custom);

  setOperationAction(ISD::BR_CC, XLenVT, Expand);
  setBooleanContents(ZeroOrOneBooleanContent);

  // Function alignments (log2).
  setMinFunctionAlignment(3);
  setPrefFunctionAlignment(3);
}

SDValue RISCVTargetLowering::LowerOperation(SDValue Op,
                                            SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  default:
    report_fatal_error("unimplemented operand");
  case ISD::GlobalAddress:
    return lowerGlobalAddress(Op, DAG);
  }
}

SDValue RISCVTargetLowering::lowerGlobalAddress(SDValue Op,
                                                SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT Ty = Op.getValueType();
  GlobalAddressSDNode *N = cast<GlobalAddressSDNode>(Op);
  const GlobalValue *GV = N->getGlobal();
  int64_t Offset = N->getOffset();

  if (!isPositionIndependent() && !Subtarget.is64Bit()) {
    SDValue GAHi =
        DAG.getTargetGlobalAddress(GV, DL, Ty, Offset, RISCVII::MO_HI);
    SDValue GALo =
        DAG.getTargetGlobalAddress(GV, DL, Ty, Offset, RISCVII::MO_LO);
    SDValue MNHi = SDValue(DAG.getMachineNode(RISCV::LUI, DL, Ty, GAHi), 0);
    SDValue MNLo =
        SDValue(DAG.getMachineNode(RISCV::ADDI, DL, Ty, MNHi, GALo), 0);
    return MNLo;
  } else {
    report_fatal_error("Unable to lowerGlobalAddress");
  }
}

// Calling Convention Implementation.
#include "RISCVGenCallingConv.inc"

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
  MachineRegisterInfo &RegInfo = MF.getRegInfo();
  MVT XLenVT = Subtarget.getXLenVT();

  if (IsVarArg)
    report_fatal_error("VarArg not supported");

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());
  CCInfo.AnalyzeFormalArguments(Ins, CC_RISCV32);

  for (auto &VA : ArgLocs) {
    if (!VA.isRegLoc())
      report_fatal_error("Defined with too many args");

    // Arguments passed in registers.
    EVT RegVT = VA.getLocVT();
    if (RegVT != XLenVT) {
      DEBUG(dbgs() << "LowerFormalArguments Unhandled argument type: "
          << RegVT.getEVTString() << "\n");
      report_fatal_error("unhandled argument type");
    }
    const unsigned VReg =
      RegInfo.createVirtualRegister(&RISCV::GPRRegClass);
    RegInfo.addLiveIn(VA.getLocReg(), VReg);
    SDValue ArgIn = DAG.getCopyFromReg(Chain, DL, VReg, RegVT);

    InVals.push_back(ArgIn);
  }
  return Chain;
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
  CLI.IsTailCall = false;
  CallingConv::ID CallConv = CLI.CallConv;
  bool IsVarArg = CLI.IsVarArg;
  EVT PtrVT = getPointerTy(DAG.getDataLayout());

  if (IsVarArg) {
    report_fatal_error("LowerCall with varargs not implemented");
  }

  MachineFunction &MF = DAG.getMachineFunction();

  // Analyze the operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState ArgCCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());
  ArgCCInfo.AnalyzeCallOperands(Outs, CC_RISCV32);

  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NumBytes = ArgCCInfo.getNextStackOffset();

  for (auto &Arg : Outs) {
    if (!Arg.Flags.isByVal())
      continue;
    report_fatal_error("Passing arguments byval not yet implemented");
  }

  Chain = DAG.getCALLSEQ_START(Chain, NumBytes, 0, CLI.DL);

  // Copy argument values to their designated locations.
  SmallVector<std::pair<unsigned, SDValue>, 8> RegsToPass;
  SDValue StackPtr;
  for (unsigned I = 0, E = ArgLocs.size(); I != E; ++I) {
    CCValAssign &VA = ArgLocs[I];
    SDValue ArgValue = OutVals[I];

    // Promote the value if needed.
    // For now, only handle fully promoted arguments.
    switch (VA.getLocInfo()) {
    case CCValAssign::Full:
      break;
    default:
      llvm_unreachable("Unknown loc info!");
    }

    if (VA.isRegLoc()) {
      // Queue up the argument copies and emit them at the end.
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), ArgValue));
    } else {
      assert(VA.isMemLoc() && "Argument not register or memory");
      report_fatal_error("Passing arguments via the stack not yet implemented");
    }
  }

  SDValue Glue;

  // Build a sequence of copy-to-reg nodes, chained and glued together.
  for (auto &Reg : RegsToPass) {
    Chain = DAG.getCopyToReg(Chain, DL, Reg.first, Reg.second, Glue);
    Glue = Chain.getValue(1);
  }

  if (isa<GlobalAddressSDNode>(Callee)) {
    Callee = lowerGlobalAddress(Callee, DAG);
  } else if (isa<ExternalSymbolSDNode>(Callee)) {
    report_fatal_error(
        "lowerExternalSymbol, needed for lowerCall, not yet handled");
  }

  // The first call operand is the chain and the second is the target address.
  SmallVector<SDValue, 8> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  // Add argument registers to the end of the list so that they are
  // known live into the call.
  for (auto &Reg : RegsToPass)
    Ops.push_back(DAG.getRegister(Reg.first, Reg.second.getValueType()));

  // Add a register mask operand representing the call-preserved registers.
  const TargetRegisterInfo *TRI = Subtarget.getRegisterInfo();
  const uint32_t *Mask = TRI->getCallPreservedMask(MF, CallConv);
  assert(Mask && "Missing call preserved mask for calling convention");
  Ops.push_back(DAG.getRegisterMask(Mask));

  // Glue the call to the argument copies, if any.
  if (Glue.getNode())
    Ops.push_back(Glue);

  // Emit the call.
  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);
  Chain = DAG.getNode(RISCVISD::CALL, DL, NodeTys, Ops);
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
  RetCCInfo.AnalyzeCallResult(Ins, RetCC_RISCV32);

  // Copy all of the result registers out of their specified physreg.
  for (auto &VA : RVLocs) {
    // Copy the value out, gluing the copy to the end of the call sequence.
    SDValue RetValue = DAG.getCopyFromReg(Chain, DL, VA.getLocReg(),
                                          VA.getLocVT(), Glue);
    Chain = RetValue.getValue(1);
    Glue = RetValue.getValue(2);

    InVals.push_back(Chain.getValue(0));
  }

  return Chain;
}

SDValue
RISCVTargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                                 bool IsVarArg,
                                 const SmallVectorImpl<ISD::OutputArg> &Outs,
                                 const SmallVectorImpl<SDValue> &OutVals,
                                 const SDLoc &DL, SelectionDAG &DAG) const {
  if (IsVarArg) {
    report_fatal_error("VarArg not supported");
  }

  // Stores the assignment of the return value to a location.
  SmallVector<CCValAssign, 16> RVLocs;

  // Info about the registers and stack slot.
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), RVLocs,
                 *DAG.getContext());

  CCInfo.AnalyzeReturn(Outs, RetCC_RISCV32);

  SDValue Flag;
  SmallVector<SDValue, 4> RetOps(1, Chain);

  // Copy the result values into the output registers.
  for (unsigned i = 0, e = RVLocs.size(); i < e; ++i) {
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");

    Chain = DAG.getCopyToReg(Chain, DL, VA.getLocReg(), OutVals[i], Flag);

    // Guarantee that all emitted copies are stuck together.
    Flag = Chain.getValue(1);
    RetOps.push_back(DAG.getRegister(VA.getLocReg(), VA.getLocVT()));
  }

  RetOps[0] = Chain; // Update chain.

  // Add the flag if we have it.
  if (Flag.getNode()) {
    RetOps.push_back(Flag);
  }

  return DAG.getNode(RISCVISD::RET_FLAG, DL, MVT::Other, RetOps);
}

const char *RISCVTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch ((RISCVISD::NodeType)Opcode) {
  case RISCVISD::FIRST_NUMBER:
    break;
  case RISCVISD::RET_FLAG:
    return "RISCVISD::RET_FLAG";
  case RISCVISD::CALL:
    return "RISCVISD::CALL";
  }
  return nullptr;
}
