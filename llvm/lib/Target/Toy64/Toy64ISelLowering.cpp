//===-- Toy64ISelLowering.cpp - Toy64 DAG Lowering ------------------------===//

#include "Toy64ISelLowering.h"

#include "Toy64Subtarget.h"
#include "MCTargetDesc/Toy64MCTargetDesc.h"

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
// TargetLowering constructor
//===----------------------------------------------------------------------===//

Toy64TargetLowering::Toy64TargetLowering(
    const TargetMachine &TM,
    const Toy64Subtarget &STI)
    : TargetLowering(TM, STI) {

  // Toy64's native integer type is i64, held in GPRs.
  addRegisterClass(
      MVT::i64,
      &Toy64::GPRRegClass);

  // Tell SelectionDAG which register is the stack pointer.
  setStackPointerRegisterToSaveRestore(
      Toy64::X2);

  // Compute properties derived from the available register classes.
  computeRegisterProperties(
      STI.getRegisterInfo());
}

//===----------------------------------------------------------------------===//
// Formal argument lowering
//===----------------------------------------------------------------------===//

SDValue Toy64TargetLowering::LowerFormalArguments(
    SDValue Chain,
    CallingConv::ID CallConv,
    bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins,
    const SDLoc &DL,
    SelectionDAG &DAG,
    SmallVectorImpl<SDValue> &InVals) const {

  if (IsVarArg)
    report_fatal_error(
        "Toy64 varargs are not implemented");

  static constexpr MCPhysReg ArgRegs[] = {
      Toy64::X10,
      Toy64::X11,
      Toy64::X12,
      Toy64::X13,
      Toy64::X14,
      Toy64::X15,
      Toy64::X16,
      Toy64::X17
  };

  constexpr unsigned NumArgRegs = 8;

  if (Ins.size() > NumArgRegs)
    report_fatal_error(
        "Toy64 stack arguments are not implemented");

  MachineFunction &MF =
      DAG.getMachineFunction();

  // X1 contains the return address on function entry.
  //
  // There is no associated virtual register because ordinary LLVM IR
  // does not directly access the return address.
  MF.getRegInfo().addLiveIn(Toy64::X1);

  for (unsigned I = 0; I < Ins.size(); ++I) {

    if (Ins[I].VT != MVT::i64)
      report_fatal_error(
          "Toy64 currently supports only i64 arguments");

    // Associate the incoming physical ABI register with a virtual
    // register in Toy64's GPR class.
    Register VReg =
        MF.addLiveIn(
            ArgRegs[I],
            &Toy64::GPRRegClass);

    // SelectionDAG works with virtual values. Copy the incoming physical
    // ABI value into the virtual-register world.
    SDValue ArgValue =
        DAG.getCopyFromReg(
            Chain,
            DL,
            VReg,
            MVT::i64);

    Chain = ArgValue.getValue(1);

    InVals.push_back(ArgValue);
  }

  return Chain;
}

//===----------------------------------------------------------------------===//
// Return lowering
//===----------------------------------------------------------------------===//

SDValue Toy64TargetLowering::LowerReturn(
    SDValue Chain,
    CallingConv::ID CallConv,
    bool IsVarArg,
    const SmallVectorImpl<ISD::OutputArg> &Outs,
    const SmallVectorImpl<SDValue> &OutVals,
    const SDLoc &DL,
    SelectionDAG &DAG) const {

  if (IsVarArg)
    report_fatal_error(
        "Toy64 vararg returns are not implemented");

  if (Outs.size() > 1)
    report_fatal_error(
        "Toy64 currently supports only one return value");

  SmallVector<SDValue, 4> RetOps;

  RetOps.push_back(Chain);

  SDValue Glue;

  if (!Outs.empty()) {

    if (Outs[0].VT != MVT::i64)
      report_fatal_error(
          "Toy64 currently supports only i64 return values");

    // ABI: X10 holds the first return value.
    Chain =
        DAG.getCopyToReg(
            Chain,
            DL,
            Toy64::X10,
            OutVals[0],
            Glue);

    Glue = Chain.getValue(1);

    RetOps[0] = Chain;

    // Keep X10 live through the return node.
    RetOps.push_back(
        DAG.getRegister(
            Toy64::X10,
            MVT::i64));

    RetOps.push_back(Glue);
  }

  return DAG.getNode(
      Toy64ISD::RET_GLUE,
      DL,
      MVT::Other,
      RetOps);
}