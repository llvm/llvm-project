//=- LoongArchISelLowering.h - LoongArch DAG Lowering Interface -*- C++ -*-===//
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

#ifndef LLVM_LIB_TARGET_LOONGARCH_LOONGARCHISELLOWERING_H
#define LLVM_LIB_TARGET_LOONGARCH_LOONGARCHISELLOWERING_H

#include "LoongArch.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {
class LoongArchSubtarget;
struct LoongArchRegisterInfo;
namespace LoongArchISD {
enum NodeType : unsigned {
  FIRST_NUMBER = ISD::BUILTIN_OP_END,

  // TODO: add more LoongArchISDs
  RET,

};
} // namespace LoongArchISD

class LoongArchTargetLowering : public TargetLowering {
  const LoongArchSubtarget &Subtarget;

public:
  explicit LoongArchTargetLowering(const TargetMachine &TM,
                                   const LoongArchSubtarget &STI);

  const LoongArchSubtarget &getSubtarget() const { return Subtarget; }

  // This method returns the name of a target specific DAG node.
  const char *getTargetNodeName(unsigned Opcode) const override;

  // Lower incoming arguments, copy physregs into vregs.
  SDValue LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv,
                               bool IsVarArg,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               const SDLoc &DL, SelectionDAG &DAG,
                               SmallVectorImpl<SDValue> &InVals) const override;
  bool CanLowerReturn(CallingConv::ID CallConv, MachineFunction &MF,
                      bool IsVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      LLVMContext &Context) const override;
  SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      const SmallVectorImpl<SDValue> &OutVals, const SDLoc &DL,
                      SelectionDAG &DAG) const override;

private:
  /// Target-specific function used to lower LoongArch calling conventions.
  typedef bool LoongArchCCAssignFn(unsigned ValNo, MVT ValVT,
                                   CCValAssign::LocInfo LocInfo,
                                   CCState &State);

  void analyzeInputArgs(CCState &CCInfo,
                        const SmallVectorImpl<ISD::InputArg> &Ins,
                        LoongArchCCAssignFn Fn) const;
  void analyzeOutputArgs(CCState &CCInfo,
                         const SmallVectorImpl<ISD::OutputArg> &Outs,
                         LoongArchCCAssignFn Fn) const;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_LOONGARCH_LOONGARCHISELLOWERING_H
