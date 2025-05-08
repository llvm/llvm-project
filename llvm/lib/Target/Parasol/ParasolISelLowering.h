//=== ParasolISelLowering.h - Parasol DAG Lowering Interface --------------===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that Parasol uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PARASOL_PARASOLISELLOWERING_H
#define LLVM_LIB_TARGET_PARASOL_PARASOLISELLOWERING_H

#include "Parasol.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/IR/Function.h"

namespace llvm {
namespace ParasolISD {
enum NodeType {
  // Start the numbering from where ISD NodeType finishes.
  FIRST_NUMBER = ISD::BUILTIN_OP_END,

  // SUNSCREEN TODO: Do we need to fill this out? Is this used for anything?
  // Filled out in other backends like AArch64
  CALL,
  ADD,
  SUB,
  MUL,
  SELECT,

  // Return
  Ret,
};
}

class ParasolSubtarget;

class ParasolTargetLowering : public TargetLowering {
public:
  explicit ParasolTargetLowering(const TargetMachine &TM,
                                 const ParasolSubtarget &STI);

  const char *getTargetNodeName(unsigned Opcode) const override;

  SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;
  void ReplaceNodeResults(SDNode *N, SmallVectorImpl<SDValue> &Results,
                          SelectionDAG &DAG) const override;

  EVT getSetCCResultType(const DataLayout &DL, LLVMContext &Context,
                         EVT VT) const override;

protected:
  // Subtarget Info
  const ParasolSubtarget &Subtarget;

private:
  SDValue LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerBlockAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerConstantPool(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerRETURNADDR(SDValue Op, SelectionDAG &DAG) const;

  using RegsToPassVector = SmallVector<std::pair<unsigned, SDValue>, 8>;

  SDValue getGlobalAddressWrapper(SDValue GA, const GlobalValue *GV,
                                  SelectionDAG &DAG) const;

  SDValue LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv,
                               bool IsVarArg,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               const SDLoc &dl, SelectionDAG &DAG,
                               SmallVectorImpl<SDValue> &InVals) const override;

  SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      const SmallVectorImpl<SDValue> &OutVals, const SDLoc &dl,
                      SelectionDAG &DAG) const override;
  bool CanLowerReturn(CallingConv::ID CallConv, MachineFunction &MF,
                      bool isVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      LLVMContext &Context) const override;

  SDValue LowerCall(TargetLowering::CallLoweringInfo &CLI,
                    SmallVectorImpl<SDValue> &InVals) const override;
  void HandleByVal(CCState *, unsigned &, Align) const override;

  SDValue LowerCallResult(SDValue Chain, SDValue InFlag,
                          CallingConv::ID CallConv, bool IsVarArg,
                          const SmallVectorImpl<ISD::InputArg> &Ins,
                          const SDLoc &dl, SelectionDAG &DAG,
                          SmallVectorImpl<SDValue> &InVals, bool isThisReturn,
                          SDValue ThisVal) const;
  SDValue LowerMemOpCallTo(SDValue Chain, SDValue Arg, const SDLoc &dl,
                           SelectionDAG &DAG, const CCValAssign &VA,
                           ISD::ArgFlagsTy Flags) const;
};
} // namespace llvm

#endif // end LLVM_LIB_TARGET_PARASOL_PARASOLISELLOWERING_H
