//===-- NVPTXISelLowering.h - NVPTX DAG Lowering Interface ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that NVPTX uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NVPTX_NVPTXISELLOWERING_H
#define LLVM_LIB_TARGET_NVPTX_NVPTXISELLOWERING_H

#include "NVPTX.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {
namespace NVPTXISD {
enum NodeType : unsigned {
  // Start the numbering from where ISD NodeType finishes.
  FIRST_NUMBER = ISD::BUILTIN_OP_END,
  Wrapper,
  CALL,
  RET_GLUE,
  LOAD_PARAM,
  DeclareParam,
  DeclareScalarParam,
  DeclareRetParam,
  DeclareRet,
  DeclareScalarRet,
  PrintCall,
  PrintConvergentCall,
  PrintCallUni,
  PrintConvergentCallUni,
  CallArgBegin,
  CallArg,
  LastCallArg,
  CallArgEnd,
  CallVoid,
  CallVal,
  CallSymbol,
  Prototype,
  MoveParam,
  PseudoUseParam,
  RETURN,
  CallSeqBegin,
  CallSeqEnd,
  CallPrototype,
  ProxyReg,
  FSHL_CLAMP,
  FSHR_CLAMP,
  MUL_WIDE_SIGNED,
  MUL_WIDE_UNSIGNED,
  IMAD,
  SETP_F16X2,
  SETP_BF16X2,
  BFE,
  BFI,
  PRMT,
  FCOPYSIGN,
  DYNAMIC_STACKALLOC,
  STACKRESTORE,
  STACKSAVE,
  BrxStart,
  BrxItem,
  BrxEnd,
  Dummy,

  LoadV2 = ISD::FIRST_TARGET_MEMORY_OPCODE,
  LoadV4,
  LDUV2, // LDU.v2
  LDUV4, // LDU.v4
  StoreV2,
  StoreV4,
  LoadParam,
  LoadParamV2,
  LoadParamV4,
  StoreParam,
  StoreParamV2,
  StoreParamV4,
  StoreParamS32, // to sext and store a <32bit value, not used currently
  StoreParamU32, // to zext and store a <32bit value, not used currently
  StoreRetval,
  StoreRetvalV2,
  StoreRetvalV4,
};
}

class NVPTXSubtarget;

//===--------------------------------------------------------------------===//
// TargetLowering Implementation
//===--------------------------------------------------------------------===//
class NVPTXTargetLowering : public TargetLowering {
public:
  explicit NVPTXTargetLowering(const NVPTXTargetMachine &TM,
                               const NVPTXSubtarget &STI);
  SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;

  SDValue LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;

  const char *getTargetNodeName(unsigned Opcode) const override;

  bool getTgtMemIntrinsic(IntrinsicInfo &Info, const CallInst &I,
                          MachineFunction &MF,
                          unsigned Intrinsic) const override;

  Align getFunctionArgumentAlignment(const Function *F, Type *Ty, unsigned Idx,
                                     const DataLayout &DL) const;

  /// getFunctionParamOptimizedAlign - since function arguments are passed via
  /// .param space, we may want to increase their alignment in a way that
  /// ensures that we can effectively vectorize their loads & stores. We can
  /// increase alignment only if the function has internal or has private
  /// linkage as for other linkage types callers may already rely on default
  /// alignment. To allow using 128-bit vectorized loads/stores, this function
  /// ensures that alignment is 16 or greater.
  Align getFunctionParamOptimizedAlign(const Function *F, Type *ArgTy,
                                       const DataLayout &DL) const;

  /// Helper for computing alignment of a device function byval parameter.
  Align getFunctionByValParamAlign(const Function *F, Type *ArgTy,
                                   Align InitialAlign,
                                   const DataLayout &DL) const;

  // Helper for getting a function parameter name. Name is composed from
  // its index and the function name. Negative index corresponds to special
  // parameter (unsized array) used for passing variable arguments.
  std::string getParamName(const Function *F, int Idx) const;

  /// isLegalAddressingMode - Return true if the addressing mode represented
  /// by AM is legal for this target, for a load/store of the specified type
  /// Used to guide target specific optimizations, like loop strength
  /// reduction (LoopStrengthReduce.cpp) and memory optimization for
  /// address mode (CodeGenPrepare.cpp)
  bool isLegalAddressingMode(const DataLayout &DL, const AddrMode &AM, Type *Ty,
                             unsigned AS,
                             Instruction *I = nullptr) const override;

  bool isTruncateFree(Type *SrcTy, Type *DstTy) const override {
    // Truncating 64-bit to 32-bit is free in SASS.
    if (!SrcTy->isIntegerTy() || !DstTy->isIntegerTy())
      return false;
    return SrcTy->getPrimitiveSizeInBits() == 64 &&
           DstTy->getPrimitiveSizeInBits() == 32;
  }

  EVT getSetCCResultType(const DataLayout &DL, LLVMContext &Ctx,
                         EVT VT) const override {
    if (VT.isVector())
      return EVT::getVectorVT(Ctx, MVT::i1, VT.getVectorNumElements());
    return MVT::i1;
  }

  ConstraintType getConstraintType(StringRef Constraint) const override;
  std::pair<unsigned, const TargetRegisterClass *>
  getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                               StringRef Constraint, MVT VT) const override;

  SDValue LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv,
                               bool isVarArg,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               const SDLoc &dl, SelectionDAG &DAG,
                               SmallVectorImpl<SDValue> &InVals) const override;

  SDValue LowerCall(CallLoweringInfo &CLI,
                    SmallVectorImpl<SDValue> &InVals) const override;

  SDValue LowerDYNAMIC_STACKALLOC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSTACKSAVE(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSTACKRESTORE(SDValue Op, SelectionDAG &DAG) const;

  std::string
  getPrototype(const DataLayout &DL, Type *, const ArgListTy &,
               const SmallVectorImpl<ISD::OutputArg> &, MaybeAlign retAlignment,
               std::optional<std::pair<unsigned, const APInt &>> VAInfo,
               const CallBase &CB, unsigned UniqueCallSite) const;

  SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      const SmallVectorImpl<SDValue> &OutVals, const SDLoc &dl,
                      SelectionDAG &DAG) const override;

  void LowerAsmOperandForConstraint(SDValue Op, StringRef Constraint,
                                    std::vector<SDValue> &Ops,
                                    SelectionDAG &DAG) const override;

  const NVPTXTargetMachine *nvTM;

  // PTX always uses 32-bit shift amounts
  MVT getScalarShiftAmountTy(const DataLayout &, EVT) const override {
    return MVT::i32;
  }

  TargetLoweringBase::LegalizeTypeAction
  getPreferredVectorAction(MVT VT) const override;

  // Get the degree of precision we want from 32-bit floating point division
  // operations.
  //
  //  0 - Use ptx div.approx
  //  1 - Use ptx.div.full (approximate, but less so than div.approx)
  //  2 - Use IEEE-compliant div instructions, if available.
  int getDivF32Level() const;

  // Get whether we should use a precise or approximate 32-bit floating point
  // sqrt instruction.
  bool usePrecSqrtF32() const;

  // Get whether we should use instructions that flush floating-point denormals
  // to sign-preserving zero.
  bool useF32FTZ(const MachineFunction &MF) const;

  SDValue getSqrtEstimate(SDValue Operand, SelectionDAG &DAG, int Enabled,
                          int &ExtraSteps, bool &UseOneConst,
                          bool Reciprocal) const override;

  unsigned combineRepeatedFPDivisors() const override { return 2; }

  bool allowFMA(MachineFunction &MF, CodeGenOptLevel OptLevel) const;
  bool allowUnsafeFPMath(MachineFunction &MF) const;

  bool isFMAFasterThanFMulAndFAdd(const MachineFunction &MF,
                                  EVT) const override {
    return true;
  }

  // The default is the same as pointer type, but brx.idx only accepts i32
  MVT getJumpTableRegTy(const DataLayout &) const override { return MVT::i32; }

  unsigned getJumpTableEncoding() const override;

  bool enableAggressiveFMAFusion(EVT VT) const override { return true; }

  // The default is to transform llvm.ctlz(x, false) (where false indicates that
  // x == 0 is not undefined behavior) into a branch that checks whether x is 0
  // and avoids calling ctlz in that case.  We have a dedicated ctlz
  // instruction, so we say that ctlz is cheap to speculate.
  bool isCheapToSpeculateCtlz(Type *Ty) const override { return true; }

  AtomicExpansionKind shouldCastAtomicLoadInIR(LoadInst *LI) const override {
    return AtomicExpansionKind::None;
  }

  AtomicExpansionKind shouldCastAtomicStoreInIR(StoreInst *SI) const override {
    return AtomicExpansionKind::None;
  }

  AtomicExpansionKind
  shouldExpandAtomicRMWInIR(AtomicRMWInst *AI) const override;

  bool aggressivelyPreferBuildVectorSources(EVT VecVT) const override {
    // There's rarely any point of packing something into a vector type if we
    // already have the source data.
    return true;
  }

private:
  const NVPTXSubtarget &STI; // cache the subtarget here
  SDValue getParamSymbol(SelectionDAG &DAG, int idx, EVT) const;

  SDValue LowerBITCAST(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerBUILD_VECTOR(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerCONCAT_VECTORS(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerEXTRACT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerINSERT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerVECTOR_SHUFFLE(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerFCOPYSIGN(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerFROUND(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFROUND32(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFROUND64(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerINT_TO_FP(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFP_TO_INT(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerFP_ROUND(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFP_EXTEND(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerLOAD(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerLOADi1(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerSTORE(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSTOREi1(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSTOREVector(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerShiftRightParts(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerShiftLeftParts(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerSelect(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerBR_JT(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerVAARG(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerVASTART(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerCopyToReg_128(SDValue Op, SelectionDAG &DAG) const;
  unsigned getNumRegisters(LLVMContext &Context, EVT VT,
                           std::optional<MVT> RegisterVT) const override;
  bool
  splitValueIntoRegisterParts(SelectionDAG &DAG, const SDLoc &DL, SDValue Val,
                              SDValue *Parts, unsigned NumParts, MVT PartVT,
                              std::optional<CallingConv::ID> CC) const override;

  void ReplaceNodeResults(SDNode *N, SmallVectorImpl<SDValue> &Results,
                          SelectionDAG &DAG) const override;
  SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const override;

  Align getArgumentAlignment(const CallBase *CB, Type *Ty, unsigned Idx,
                             const DataLayout &DL) const;
};

} // namespace llvm

#endif
