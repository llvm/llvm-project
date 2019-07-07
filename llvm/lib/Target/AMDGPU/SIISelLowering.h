//===-- SIISelLowering.h - SI DAG Lowering Interface ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// SI DAG Lowering interface definition
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_SIISELLOWERING_H
#define LLVM_LIB_TARGET_AMDGPU_SIISELLOWERING_H

#include "AMDGPUISelLowering.h"
#include "AMDGPUArgumentUsageInfo.h"
#include "SIInstrInfo.h"

namespace llvm {

class SITargetLowering final : public AMDGPUTargetLowering {
private:
  const GCNSubtarget *Subtarget;

public:
  MVT getRegisterTypeForCallingConv(LLVMContext &Context,
                                    CallingConv::ID CC,
                                    EVT VT) const override;
  unsigned getNumRegistersForCallingConv(LLVMContext &Context,
                                         CallingConv::ID CC,
                                         EVT VT) const override;

  unsigned getVectorTypeBreakdownForCallingConv(
    LLVMContext &Context, CallingConv::ID CC, EVT VT, EVT &IntermediateVT,
    unsigned &NumIntermediates, MVT &RegisterVT) const override;

private:
  SDValue lowerKernArgParameterPtr(SelectionDAG &DAG, const SDLoc &SL,
                                   SDValue Chain, uint64_t Offset) const;
  SDValue getImplicitArgPtr(SelectionDAG &DAG, const SDLoc &SL) const;
  SDValue lowerKernargMemParameter(SelectionDAG &DAG, EVT VT, EVT MemVT,
                                   const SDLoc &SL, SDValue Chain,
                                   uint64_t Offset, unsigned Align, bool Signed,
                                   const ISD::InputArg *Arg = nullptr) const;

  SDValue lowerStackParameter(SelectionDAG &DAG, CCValAssign &VA,
                              const SDLoc &SL, SDValue Chain,
                              const ISD::InputArg &Arg) const;
  SDValue getPreloadedValue(SelectionDAG &DAG,
                            const SIMachineFunctionInfo &MFI,
                            EVT VT,
                            AMDGPUFunctionArgInfo::PreloadedValue) const;

  SDValue LowerGlobalAddress(AMDGPUMachineFunction *MFI, SDValue Op,
                             SelectionDAG &DAG) const override;
  SDValue lowerImplicitZextParam(SelectionDAG &DAG, SDValue Op,
                                 MVT VT, unsigned Offset) const;
  SDValue lowerImage(SDValue Op, const AMDGPU::ImageDimIntrinsicInfo *Intr,
                     SelectionDAG &DAG) const;
  SDValue lowerSBuffer(EVT VT, SDLoc DL, SDValue Rsrc, SDValue Offset,
                       SDValue GLC, SDValue DLC, SelectionDAG &DAG) const;

  SDValue LowerINTRINSIC_WO_CHAIN(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerINTRINSIC_W_CHAIN(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerINTRINSIC_VOID(SDValue Op, SelectionDAG &DAG) const;

  // The raw.tbuffer and struct.tbuffer intrinsics have two offset args: offset
  // (the offset that is included in bounds checking and swizzling, to be split
  // between the instruction's voffset and immoffset fields) and soffset (the
  // offset that is excluded from bounds checking and swizzling, to go in the
  // instruction's soffset field).  This function takes the first kind of
  // offset and figures out how to split it between voffset and immoffset.
  std::pair<SDValue, SDValue> splitBufferOffsets(SDValue Offset,
                                                 SelectionDAG &DAG) const;

  SDValue widenLoad(LoadSDNode *Ld, DAGCombinerInfo &DCI) const;
  SDValue LowerLOAD(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSELECT(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFastUnsafeFDIV(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFDIV_FAST(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFDIV16(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFDIV32(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFDIV64(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFDIV(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerINT_TO_FP(SDValue Op, SelectionDAG &DAG, bool Signed) const;
  SDValue LowerSTORE(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerTrig(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerATOMIC_CMP_SWAP(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerBRCOND(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerRETURNADDR(SDValue Op, SelectionDAG &DAG) const;
  SDValue adjustLoadValueType(unsigned Opcode, MemSDNode *M,
                              SelectionDAG &DAG, ArrayRef<SDValue> Ops,
                              bool IsIntrinsic = false) const;

  // Call DAG.getMemIntrinsicNode for a load, but first widen a dwordx3 type to
  // dwordx4 if on SI.
  SDValue getMemIntrinsicNode(unsigned Opcode, const SDLoc &DL, SDVTList VTList,
                              ArrayRef<SDValue> Ops, EVT MemVT,
                              MachineMemOperand *MMO, SelectionDAG &DAG) const;

  SDValue handleD16VData(SDValue VData, SelectionDAG &DAG) const;

  /// Converts \p Op, which must be of floating point type, to the
  /// floating point type \p VT, by either extending or truncating it.
  SDValue getFPExtOrFPTrunc(SelectionDAG &DAG,
                            SDValue Op,
                            const SDLoc &DL,
                            EVT VT) const;

  SDValue convertArgType(
    SelectionDAG &DAG, EVT VT, EVT MemVT, const SDLoc &SL, SDValue Val,
    bool Signed, const ISD::InputArg *Arg = nullptr) const;

  /// Custom lowering for ISD::FP_ROUND for MVT::f16.
  SDValue lowerFP_ROUND(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFMINNUM_FMAXNUM(SDValue Op, SelectionDAG &DAG) const;

  SDValue getSegmentAperture(unsigned AS, const SDLoc &DL,
                             SelectionDAG &DAG) const;

  SDValue lowerADDRSPACECAST(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerINSERT_SUBVECTOR(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerINSERT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerEXTRACT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVECTOR_SHUFFLE(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerBUILD_VECTOR(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerTRAP(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerDEBUGTRAP(SDValue Op, SelectionDAG &DAG) const;

  SDNode *adjustWritemask(MachineSDNode *&N, SelectionDAG &DAG) const;

  SDValue performUCharToFloatCombine(SDNode *N,
                                     DAGCombinerInfo &DCI) const;
  SDValue performSHLPtrCombine(SDNode *N,
                               unsigned AS,
                               EVT MemVT,
                               DAGCombinerInfo &DCI) const;

  SDValue performMemSDNodeCombine(MemSDNode *N, DAGCombinerInfo &DCI) const;

  SDValue splitBinaryBitConstantOp(DAGCombinerInfo &DCI, const SDLoc &SL,
                                   unsigned Opc, SDValue LHS,
                                   const ConstantSDNode *CRHS) const;

  SDValue performAndCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue performOrCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue performXorCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue performZeroExtendCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue performSignExtendInRegCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue performClassCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue getCanonicalConstantFP(SelectionDAG &DAG, const SDLoc &SL, EVT VT,
                                 const APFloat &C) const;
  SDValue performFCanonicalizeCombine(SDNode *N, DAGCombinerInfo &DCI) const;

  SDValue performFPMed3ImmCombine(SelectionDAG &DAG, const SDLoc &SL,
                                  SDValue Op0, SDValue Op1) const;
  SDValue performIntMed3ImmCombine(SelectionDAG &DAG, const SDLoc &SL,
                                   SDValue Op0, SDValue Op1, bool Signed) const;
  SDValue performMinMaxCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue performFMed3Combine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue performCvtPkRTZCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue performExtractVectorEltCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue performInsertVectorEltCombine(SDNode *N, DAGCombinerInfo &DCI) const;

  SDValue reassociateScalarOps(SDNode *N, SelectionDAG &DAG) const;
  unsigned getFusedOpcode(const SelectionDAG &DAG,
                          const SDNode *N0, const SDNode *N1) const;
  SDValue performAddCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue performAddCarrySubCarryCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue performSubCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue performFAddCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue performFSubCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue performFMACombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue performSetCCCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue performCvtF32UByteNCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue performClampCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue performRcpCombine(SDNode *N, DAGCombinerInfo &DCI) const;

  bool isLegalFlatAddressingMode(const AddrMode &AM) const;
  bool isLegalMUBUFAddressingMode(const AddrMode &AM) const;

  unsigned isCFIntrinsic(const SDNode *Intr) const;

  /// \returns True if fixup needs to be emitted for given global value \p GV,
  /// false otherwise.
  bool shouldEmitFixup(const GlobalValue *GV) const;

  /// \returns True if GOT relocation needs to be emitted for given global value
  /// \p GV, false otherwise.
  bool shouldEmitGOTReloc(const GlobalValue *GV) const;

  /// \returns True if PC-relative relocation needs to be emitted for given
  /// global value \p GV, false otherwise.
  bool shouldEmitPCReloc(const GlobalValue *GV) const;

  // Analyze a combined offset from an amdgcn_buffer_ intrinsic and store the
  // three offsets (voffset, soffset and instoffset) into the SDValue[3] array
  // pointed to by Offsets.
  void setBufferOffsets(SDValue CombinedOffset, SelectionDAG &DAG,
                        SDValue *Offsets, unsigned Align = 4) const;

  // Handle 8 bit and 16 bit buffer loads
  SDValue handleByteShortBufferLoads(SelectionDAG &DAG, EVT LoadVT, SDLoc DL,
                                     ArrayRef<SDValue> Ops, MemSDNode *M) const;

  // Handle 8 bit and 16 bit buffer stores
  SDValue handleByteShortBufferStores(SelectionDAG &DAG, EVT VDataType,
                                      SDLoc DL, SDValue Ops[],
                                      MemSDNode *M) const;

public:
  SITargetLowering(const TargetMachine &tm, const GCNSubtarget &STI);

  const GCNSubtarget *getSubtarget() const;

  bool isFPExtFoldable(unsigned Opcode, EVT DestVT, EVT SrcVT) const override;

  bool isShuffleMaskLegal(ArrayRef<int> /*Mask*/, EVT /*VT*/) const override;

  bool getTgtMemIntrinsic(IntrinsicInfo &, const CallInst &,
                          MachineFunction &MF,
                          unsigned IntrinsicID) const override;

  bool getAddrModeArguments(IntrinsicInst * /*I*/,
                            SmallVectorImpl<Value*> &/*Ops*/,
                            Type *&/*AccessTy*/) const override;

  bool isLegalGlobalAddressingMode(const AddrMode &AM) const;
  bool isLegalAddressingMode(const DataLayout &DL, const AddrMode &AM, Type *Ty,
                             unsigned AS,
                             Instruction *I = nullptr) const override;

  bool canMergeStoresTo(unsigned AS, EVT MemVT,
                        const SelectionDAG &DAG) const override;

  bool allowsMisalignedMemoryAccesses(
      EVT VT, unsigned AS, unsigned Align,
      MachineMemOperand::Flags Flags = MachineMemOperand::MONone,
      bool *IsFast = nullptr) const override;

  EVT getOptimalMemOpType(uint64_t Size, unsigned DstAlign,
                          unsigned SrcAlign, bool IsMemset,
                          bool ZeroMemset,
                          bool MemcpyStrSrc,
                          const AttributeList &FuncAttributes) const override;

  bool isMemOpUniform(const SDNode *N) const;
  bool isMemOpHasNoClobberedMemOperand(const SDNode *N) const;
  bool isNoopAddrSpaceCast(unsigned SrcAS, unsigned DestAS) const override;
  bool isFreeAddrSpaceCast(unsigned SrcAS, unsigned DestAS) const override;

  TargetLoweringBase::LegalizeTypeAction
  getPreferredVectorAction(MVT VT) const override;

  bool shouldConvertConstantLoadToIntImm(const APInt &Imm,
                                        Type *Ty) const override;

  bool isTypeDesirableForOp(unsigned Op, EVT VT) const override;

  bool isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const override;

  bool supportSplitCSR(MachineFunction *MF) const override;
  void initializeSplitCSR(MachineBasicBlock *Entry) const override;
  void insertCopiesSplitCSR(
    MachineBasicBlock *Entry,
    const SmallVectorImpl<MachineBasicBlock *> &Exits) const override;

  SDValue LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv,
                               bool isVarArg,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               const SDLoc &DL, SelectionDAG &DAG,
                               SmallVectorImpl<SDValue> &InVals) const override;

  bool CanLowerReturn(CallingConv::ID CallConv,
                      MachineFunction &MF, bool isVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      LLVMContext &Context) const override;

  SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      const SmallVectorImpl<SDValue> &OutVals, const SDLoc &DL,
                      SelectionDAG &DAG) const override;

  void passSpecialInputs(
    CallLoweringInfo &CLI,
    CCState &CCInfo,
    const SIMachineFunctionInfo &Info,
    SmallVectorImpl<std::pair<unsigned, SDValue>> &RegsToPass,
    SmallVectorImpl<SDValue> &MemOpChains,
    SDValue Chain) const;

  SDValue LowerCallResult(SDValue Chain, SDValue InFlag,
                          CallingConv::ID CallConv, bool isVarArg,
                          const SmallVectorImpl<ISD::InputArg> &Ins,
                          const SDLoc &DL, SelectionDAG &DAG,
                          SmallVectorImpl<SDValue> &InVals, bool isThisReturn,
                          SDValue ThisVal) const;

  bool mayBeEmittedAsTailCall(const CallInst *) const override;

  bool isEligibleForTailCallOptimization(
    SDValue Callee, CallingConv::ID CalleeCC, bool isVarArg,
    const SmallVectorImpl<ISD::OutputArg> &Outs,
    const SmallVectorImpl<SDValue> &OutVals,
    const SmallVectorImpl<ISD::InputArg> &Ins, SelectionDAG &DAG) const;

  SDValue LowerCall(CallLoweringInfo &CLI,
                    SmallVectorImpl<SDValue> &InVals) const override;

  unsigned getRegisterByName(const char* RegName, EVT VT,
                             SelectionDAG &DAG) const override;

  MachineBasicBlock *splitKillBlock(MachineInstr &MI,
                                    MachineBasicBlock *BB) const;

  MachineBasicBlock *emitGWSMemViolTestLoop(MachineInstr &MI,
                                            MachineBasicBlock *BB) const;

  MachineBasicBlock *
  EmitInstrWithCustomInserter(MachineInstr &MI,
                              MachineBasicBlock *BB) const override;

  bool hasBitPreservingFPLogic(EVT VT) const override;
  bool enableAggressiveFMAFusion(EVT VT) const override;
  EVT getSetCCResultType(const DataLayout &DL, LLVMContext &Context,
                         EVT VT) const override;
  MVT getScalarShiftAmountTy(const DataLayout &, EVT) const override;
  bool isFMAFasterThanFMulAndFAdd(EVT VT) const override;
  SDValue splitUnaryVectorOp(SDValue Op, SelectionDAG &DAG) const;
  SDValue splitBinaryVectorOp(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;

  void ReplaceNodeResults(SDNode *N, SmallVectorImpl<SDValue> &Results,
                          SelectionDAG &DAG) const override;

  SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const override;
  SDNode *PostISelFolding(MachineSDNode *N, SelectionDAG &DAG) const override;
  void AdjustInstrPostInstrSelection(MachineInstr &MI,
                                     SDNode *Node) const override;

  SDNode *legalizeTargetIndependentNode(SDNode *Node, SelectionDAG &DAG) const;

  MachineSDNode *wrapAddr64Rsrc(SelectionDAG &DAG, const SDLoc &DL,
                                SDValue Ptr) const;
  MachineSDNode *buildRSRC(SelectionDAG &DAG, const SDLoc &DL, SDValue Ptr,
                           uint32_t RsrcDword1, uint64_t RsrcDword2And3) const;
  std::pair<unsigned, const TargetRegisterClass *>
  getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                               StringRef Constraint, MVT VT) const override;
  ConstraintType getConstraintType(StringRef Constraint) const override;
  SDValue copyToM0(SelectionDAG &DAG, SDValue Chain, const SDLoc &DL,
                   SDValue V) const;

  void finalizeLowering(MachineFunction &MF) const override;

  void computeKnownBitsForFrameIndex(const SDValue Op,
                                     KnownBits &Known,
                                     const APInt &DemandedElts,
                                     const SelectionDAG &DAG,
                                     unsigned Depth = 0) const override;

  bool isSDNodeSourceOfDivergence(const SDNode *N,
    FunctionLoweringInfo *FLI, LegacyDivergenceAnalysis *DA) const override;

  bool isCanonicalized(SelectionDAG &DAG, SDValue Op,
                       unsigned MaxDepth = 5) const;
  bool denormalsEnabledForType(EVT VT) const;

  bool isKnownNeverNaNForTargetNode(SDValue Op,
                                    const SelectionDAG &DAG,
                                    bool SNaN = false,
                                    unsigned Depth = 0) const override;
  AtomicExpansionKind shouldExpandAtomicRMWInIR(AtomicRMWInst *) const override;

  unsigned getPrefLoopAlignment(MachineLoop *ML) const override;
};

} // End namespace llvm

#endif
