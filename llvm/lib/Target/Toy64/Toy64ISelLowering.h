//===-- Toy64ISelLowering.h - Toy64 DAG Lowering Interface ----*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_TOY64_TOY64ISELLOWERING_H
#define LLVM_LIB_TARGET_TOY64_TOY64ISELLOWERING_H

#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {

class Toy64Subtarget;
class TargetMachine;

namespace Toy64ISD {

enum NodeType : unsigned {
  FIRST_NUMBER = ISD::BUILTIN_OP_END,

  // Function return.
  RET_GLUE
};

} // namespace Toy64ISD

class Toy64TargetLowering : public TargetLowering {
public:
  Toy64TargetLowering(
      const TargetMachine &TM,
      const Toy64Subtarget &STI);

  SDValue LowerFormalArguments(
      SDValue Chain,
      CallingConv::ID CallConv,
      bool IsVarArg,
      const SmallVectorImpl<ISD::InputArg> &Ins,
      const SDLoc &DL,
      SelectionDAG &DAG,
      SmallVectorImpl<SDValue> &InVals) const override;

  SDValue LowerReturn(
      SDValue Chain,
      CallingConv::ID CallConv,
      bool IsVarArg,
      const SmallVectorImpl<ISD::OutputArg> &Outs,
      const SmallVectorImpl<SDValue> &OutVals,
      const SDLoc &DL,
      SelectionDAG &DAG) const override;
};

} // namespace llvm

#endif