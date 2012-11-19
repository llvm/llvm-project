//===-- NVPTXISelLowering.h - NVPTX DAG Lowering Interface ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that NVPTX uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef NVPTXISELLOWERING_H
#define NVPTXISELLOWERING_H

#include "NVPTX.h"
#include "NVPTXSubtarget.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Target/TargetLowering.h"

namespace llvm {
namespace NVPTXISD {
enum NodeType {
  // Start the numbering from where ISD NodeType finishes.
  FIRST_NUMBER = ISD::BUILTIN_OP_END,
  Wrapper,
  CALL,
  RET_FLAG,
  LOAD_PARAM,
  NVBuiltin,
  DeclareParam,
  DeclareScalarParam,
  DeclareRetParam,
  DeclareRet,
  DeclareScalarRet,
  LoadParam,
  StoreParam,
  StoreParamS32, // to sext and store a <32bit value, not used currently
  StoreParamU32, // to zext and store a <32bit value, not used currently
  MoveToParam,
  PrintCall,
  PrintCallUni,
  CallArgBegin,
  CallArg,
  LastCallArg,
  CallArgEnd,
  CallVoid,
  CallVal,
  CallSymbol,
  Prototype,
  MoveParam,
  MoveRetval,
  MoveToRetval,
  StoreRetval,
  PseudoUseParam,
  RETURN,
  CallSeqBegin,
  CallSeqEnd,
  Dummy
};
}

//===--------------------------------------------------------------------===//
// TargetLowering Implementation
//===--------------------------------------------------------------------===//
class NVPTXTargetLowering : public TargetLowering {
public:
  explicit NVPTXTargetLowering(NVPTXTargetMachine &TM);
  virtual SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerGlobalAddress(const GlobalValue *GV, int64_t Offset,
                             SelectionDAG &DAG) const;

  virtual const char *getTargetNodeName(unsigned Opcode) const;

  bool isTypeSupportedInIntrinsic(MVT VT) const;

  bool getTgtMemIntrinsic(IntrinsicInfo& Info, const CallInst &I,
                          unsigned Intrinsic) const;

  /// isLegalAddressingMode - Return true if the addressing mode represented
  /// by AM is legal for this target, for a load/store of the specified type
  /// Used to guide target specific optimizations, like loop strength
  /// reduction (LoopStrengthReduce.cpp) and memory optimization for
  /// address mode (CodeGenPrepare.cpp)
  virtual bool isLegalAddressingMode(const AddrMode &AM, Type *Ty) const;

  /// getFunctionAlignment - Return the Log2 alignment of this function.
  virtual unsigned getFunctionAlignment(const Function *F) const;

  virtual EVT getSetCCResultType(EVT VT) const {
    return MVT::i1;
  }

  ConstraintType getConstraintType(const std::string &Constraint) const;
  std::pair<unsigned, const TargetRegisterClass*>
  getRegForInlineAsmConstraint(const std::string &Constraint, EVT VT) const;

  virtual SDValue
  LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
                       const SmallVectorImpl<ISD::InputArg> &Ins, DebugLoc dl,
                       SelectionDAG &DAG,
                       SmallVectorImpl<SDValue> &InVals) const;

  virtual SDValue
  LowerCall(CallLoweringInfo &CLI, SmallVectorImpl<SDValue> &InVals) const;

  std::string getPrototype(Type *, const ArgListTy &,
                           const SmallVectorImpl<ISD::OutputArg> &,
                           unsigned retAlignment) const;

  virtual SDValue
  LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
              const SmallVectorImpl<ISD::OutputArg> &Outs,
              const SmallVectorImpl<SDValue> &OutVals, DebugLoc dl,
              SelectionDAG &DAG) const;

  virtual void LowerAsmOperandForConstraint(SDValue Op, std::string &Constraint,
                                            std::vector<SDValue> &Ops,
                                            SelectionDAG &DAG) const;

  NVPTXTargetMachine *nvTM;

  // PTX always uses 32-bit shift amounts
  virtual MVT getShiftAmountTy(EVT LHSTy) const {
    return MVT::i32;
  }

private:
  const NVPTXSubtarget &nvptxSubtarget;  // cache the subtarget here

  SDValue getExtSymb(SelectionDAG &DAG, const char *name, int idx, EVT =
                         MVT::i32) const;
  SDValue getParamSymbol(SelectionDAG &DAG, int idx, EVT = MVT::i32) const;
  SDValue getParamHelpSymbol(SelectionDAG &DAG, int idx);

  SDValue LowerCONCAT_VECTORS(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerSTORE(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerLOAD(SDValue Op, SelectionDAG &DAG) const;
};
} // namespace llvm

#endif // NVPTXISELLOWERING_H
