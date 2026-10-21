#include "GenpayISelLowering.h"
#include "GenpayRegisterInfo.h"
#include "GenpaySubtarget.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

// Args: first 8 i64 args in R0-R7, then stack (not supported yet)
// NOTE: LLVM 21's CCAssignFn has no ArgListEntry parameter (that is the
// LLVM 22-dev signature). This matches release/21.x CallingConvLower.h.
static bool CC_Genpay(unsigned ValNo, MVT ValVT, MVT LocVT,
                      CCValAssign::LocInfo LocInfo,
                      ISD::ArgFlagsTy ArgFlags, CCState &State) {
  static const MCPhysReg ArgRegs[] = {Genpay::R0, Genpay::R1, Genpay::R2,
                                      Genpay::R3, Genpay::R4, Genpay::R5,
                                      Genpay::R6, Genpay::R7};
  if (LocVT == MVT::i64) {
    if (unsigned Reg = State.AllocateReg(ArgRegs)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
    unsigned Offset = State.AllocateStack(8, Align(8));
    State.addLoc(CCValAssign::getMem(ValNo, ValVT, Offset, LocVT, LocInfo));
    return false;
  }
  return true;
}

// Return value: first i64 in R0
static bool RetCC_Genpay(unsigned ValNo, MVT ValVT, MVT LocVT,
                         CCValAssign::LocInfo LocInfo,
                         ISD::ArgFlagsTy ArgFlags, CCState &State) {
  if (LocVT == MVT::i64) {
    if (unsigned Reg = State.AllocateReg(Genpay::R0)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
    llvm_unreachable("Genpay: cannot return this value in a register");
  }
  return true;
}

GenpayTargetLowering::GenpayTargetLowering(const TargetMachine &TM,
                                           const GenpaySubtarget &STI)
    : TargetLowering(TM) {
  addRegisterClass(MVT::i64, &Genpay::GPRRegClass);
  computeRegisterProperties(STI.getRegisterInfo());

  // Lowered to libcalls / expanded sequences (Cpu0-style skeleton)
  setOperationAction(ISD::SHL, MVT::i64, Expand);
  setOperationAction(ISD::SRA, MVT::i64, Expand);
  setOperationAction(ISD::SRL, MVT::i64, Expand);
  setOperationAction(ISD::MUL, MVT::i64, Expand);
  setOperationAction(ISD::SDIV, MVT::i64, Expand);
  setOperationAction(ISD::SREM, MVT::i64, Expand);
  setOperationAction(ISD::UDIV, MVT::i64, Expand);
  setOperationAction(ISD::UREM, MVT::i64, Expand);
}

SDValue GenpayTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, RVLocs, *DAG.getContext());
  CCInfo.AnalyzeFormalArguments(Ins, CC_Genpay);

  for (const CCValAssign &VA : RVLocs) {
    if (VA.isRegLoc()) {
      if (VA.getLocVT() != MVT::i64)
        llvm_unreachable("Genpay: unsupported argument type");
      unsigned VReg = MRI.createVirtualRegister(&Genpay::GPRRegClass);
      MRI.addLiveIn(VA.getLocReg(), VReg);
      InVals.push_back(DAG.getCopyFromReg(Chain, DL, VReg, VA.getLocVT()));
    } else {
      llvm_unreachable("Genpay: stack arguments not supported yet");
    }
  }
  return Chain;
}

SDValue GenpayTargetLowering::LowerReturn(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::OutputArg> &Outs,
    const SmallVectorImpl<SDValue> &OutVals, const SDLoc &DL,
    SelectionDAG &DAG) const {
  SmallVector<CCValAssign, 4> RVLocs;
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), RVLocs,
                 *DAG.getContext());
  CCInfo.AnalyzeReturn(Outs, RetCC_Genpay);

  SDValue Flag;
  SmallVector<SDValue, 4> RetOps(1, Chain);

  for (const CCValAssign &VA : RVLocs) {
    assert(VA.getValNo() < Outs.size() && "Too many return values");
    SDValue Arg = OutVals[VA.getValNo()];

    if (VA.isRegLoc()) {
      Chain = DAG.getCopyToReg(Chain, DL, VA.getLocReg(), Arg, Flag);
      Flag = Chain.getValue(1);
      RetOps.push_back(DAG.getRegister(VA.getLocReg(), Arg.getValueType()));
    } else {
      llvm_unreachable("Genpay: stack return values not supported yet");
    }
  }

  RetOps[0] = Chain;
  if (Flag.getNode())
    RetOps.push_back(Flag);

  return DAG.getNode(GenpayISD::RET, DL, MVT::Other, RetOps);
}
