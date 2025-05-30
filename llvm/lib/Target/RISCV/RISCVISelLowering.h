//===-- RISCVISelLowering.h - RISC-V DAG Lowering Interface -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that RISC-V uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVISELLOWERING_H
#define LLVM_LIB_TARGET_RISCV_RISCVISELLOWERING_H

#include "RISCV.h"
#include "RISCVCallingConv.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLowering.h"
#include <optional>

namespace llvm {
class InstructionCost;
class RISCVSubtarget;
struct RISCVRegisterInfo;

class RISCVTargetLowering : public TargetLowering {
  const RISCVSubtarget &Subtarget;

public:
  explicit RISCVTargetLowering(const TargetMachine &TM,
                               const RISCVSubtarget &STI);

  const RISCVSubtarget &getSubtarget() const { return Subtarget; }

  bool getTgtMemIntrinsic(IntrinsicInfo &Info, const CallInst &I,
                          MachineFunction &MF,
                          unsigned Intrinsic) const override;
  bool isLegalAddressingMode(const DataLayout &DL, const AddrMode &AM, Type *Ty,
                             unsigned AS,
                             Instruction *I = nullptr) const override;
  bool isLegalICmpImmediate(int64_t Imm) const override;
  bool isLegalAddImmediate(int64_t Imm) const override;
  bool isTruncateFree(Type *SrcTy, Type *DstTy) const override;
  bool isTruncateFree(EVT SrcVT, EVT DstVT) const override;
  bool isTruncateFree(SDValue Val, EVT VT2) const override;
  bool isZExtFree(SDValue Val, EVT VT2) const override;
  bool isSExtCheaperThanZExt(EVT SrcVT, EVT DstVT) const override;
  bool signExtendConstant(const ConstantInt *CI) const override;
  bool isCheapToSpeculateCttz(Type *Ty) const override;
  bool isCheapToSpeculateCtlz(Type *Ty) const override;
  bool isMaskAndCmp0FoldingBeneficial(const Instruction &AndI) const override;
  bool hasAndNotCompare(SDValue Y) const override;
  bool hasAndNot(SDValue Y) const override;
  bool hasBitTest(SDValue X, SDValue Y) const override;
  bool shouldProduceAndByConstByHoistingConstFromShiftsLHSOfAnd(
      SDValue X, ConstantSDNode *XC, ConstantSDNode *CC, SDValue Y,
      unsigned OldShiftOpcode, unsigned NewShiftOpcode,
      SelectionDAG &DAG) const override;
  bool shouldScalarizeBinop(SDValue VecOp) const override;
  bool isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const override;
  int getLegalZfaFPImm(const APFloat &Imm, EVT VT) const;
  bool isFPImmLegal(const APFloat &Imm, EVT VT,
                    bool ForCodeSize) const override;
  bool isExtractSubvectorCheap(EVT ResVT, EVT SrcVT,
                               unsigned Index) const override;

  bool isIntDivCheap(EVT VT, AttributeList Attr) const override;

  bool preferScalarizeSplat(SDNode *N) const override;

  bool softPromoteHalfType() const override { return true; }

  /// Return the register type for a given MVT, ensuring vectors are treated
  /// as a series of gpr sized integers.
  MVT getRegisterTypeForCallingConv(LLVMContext &Context, CallingConv::ID CC,
                                    EVT VT) const override;

  /// Return the number of registers for a given MVT, for inline assembly
  unsigned
  getNumRegisters(LLVMContext &Context, EVT VT,
                  std::optional<MVT> RegisterVT = std::nullopt) const override;

  /// Return the number of registers for a given MVT, ensuring vectors are
  /// treated as a series of gpr sized integers.
  unsigned getNumRegistersForCallingConv(LLVMContext &Context,
                                         CallingConv::ID CC,
                                         EVT VT) const override;

  unsigned getVectorTypeBreakdownForCallingConv(LLVMContext &Context,
                                                CallingConv::ID CC, EVT VT,
                                                EVT &IntermediateVT,
                                                unsigned &NumIntermediates,
                                                MVT &RegisterVT) const override;

  bool shouldFoldSelectWithIdentityConstant(unsigned BinOpcode,
                                            EVT VT) const override;

  /// Return true if the given shuffle mask can be codegen'd directly, or if it
  /// should be stack expanded.
  bool isShuffleMaskLegal(ArrayRef<int> M, EVT VT) const override;

  bool isMultiStoresCheaperThanBitsMerge(EVT LTy, EVT HTy) const override {
    // If the pair to store is a mixture of float and int values, we will
    // save two bitwise instructions and one float-to-int instruction and
    // increase one store instruction. There is potentially a more
    // significant benefit because it avoids the float->int domain switch
    // for input value. So It is more likely a win.
    if ((LTy.isFloatingPoint() && HTy.isInteger()) ||
        (LTy.isInteger() && HTy.isFloatingPoint()))
      return true;
    // If the pair only contains int values, we will save two bitwise
    // instructions and increase one store instruction (costing one more
    // store buffer). Since the benefit is more blurred we leave such a pair
    // out until we get testcase to prove it is a win.
    return false;
  }

  bool
  shouldExpandBuildVectorWithShuffles(EVT VT,
                                      unsigned DefinedValues) const override;

  bool shouldExpandCttzElements(EVT VT) const override;

  /// Return the cost of LMUL for linear operations.
  InstructionCost getLMULCost(MVT VT) const;

  InstructionCost getVRGatherVVCost(MVT VT) const;
  InstructionCost getVRGatherVICost(MVT VT) const;
  InstructionCost getVSlideVXCost(MVT VT) const;
  InstructionCost getVSlideVICost(MVT VT) const;

  // Provide custom lowering hooks for some operations.
  SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;
  void ReplaceNodeResults(SDNode *N, SmallVectorImpl<SDValue> &Results,
                          SelectionDAG &DAG) const override;

  SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const override;

  bool targetShrinkDemandedConstant(SDValue Op, const APInt &DemandedBits,
                                    const APInt &DemandedElts,
                                    TargetLoweringOpt &TLO) const override;

  void computeKnownBitsForTargetNode(const SDValue Op,
                                     KnownBits &Known,
                                     const APInt &DemandedElts,
                                     const SelectionDAG &DAG,
                                     unsigned Depth) const override;
  unsigned ComputeNumSignBitsForTargetNode(SDValue Op,
                                           const APInt &DemandedElts,
                                           const SelectionDAG &DAG,
                                           unsigned Depth) const override;

  bool canCreateUndefOrPoisonForTargetNode(SDValue Op,
                                           const APInt &DemandedElts,
                                           const SelectionDAG &DAG,
                                           bool PoisonOnly, bool ConsiderFlags,
                                           unsigned Depth) const override;

  const Constant *getTargetConstantFromLoad(LoadSDNode *LD) const override;

  MachineMemOperand::Flags
  getTargetMMOFlags(const Instruction &I) const override;

  MachineMemOperand::Flags
  getTargetMMOFlags(const MemSDNode &Node) const override;

  bool
  areTwoSDNodeTargetMMOFlagsMergeable(const MemSDNode &NodeX,
                                      const MemSDNode &NodeY) const override;

  ConstraintType getConstraintType(StringRef Constraint) const override;

  InlineAsm::ConstraintCode
  getInlineAsmMemConstraint(StringRef ConstraintCode) const override;

  std::pair<unsigned, const TargetRegisterClass *>
  getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                               StringRef Constraint, MVT VT) const override;

  void LowerAsmOperandForConstraint(SDValue Op, StringRef Constraint,
                                    std::vector<SDValue> &Ops,
                                    SelectionDAG &DAG) const override;

  MachineBasicBlock *
  EmitInstrWithCustomInserter(MachineInstr &MI,
                              MachineBasicBlock *BB) const override;

  void AdjustInstrPostInstrSelection(MachineInstr &MI,
                                     SDNode *Node) const override;

  EVT getSetCCResultType(const DataLayout &DL, LLVMContext &Context,
                         EVT VT) const override;

  bool shouldFormOverflowOp(unsigned Opcode, EVT VT,
                            bool MathUsed) const override {
    if (VT == MVT::i8 || VT == MVT::i16)
      return false;

    return TargetLowering::shouldFormOverflowOp(Opcode, VT, MathUsed);
  }

  bool storeOfVectorConstantIsCheap(bool IsZero, EVT MemVT, unsigned NumElem,
                                    unsigned AddrSpace) const override {
    // If we can replace 4 or more scalar stores, there will be a reduction
    // in instructions even after we add a vector constant load.
    return NumElem >= 4;
  }

  bool convertSetCCLogicToBitwiseLogic(EVT VT) const override {
    return VT.isScalarInteger();
  }
  bool convertSelectOfConstantsToMath(EVT VT) const override { return true; }

  bool isCtpopFast(EVT VT) const override;

  unsigned getCustomCtpopCost(EVT VT, ISD::CondCode Cond) const override;

  bool preferZeroCompareBranch() const override { return true; }

  // Note that one specific case requires fence insertion for an
  // AtomicCmpXchgInst but is handled via the RISCVZacasABIFix pass rather
  // than this hook due to limitations in the interface here.
  bool shouldInsertFencesForAtomic(const Instruction *I) const override;

  Instruction *emitLeadingFence(IRBuilderBase &Builder, Instruction *Inst,
                                AtomicOrdering Ord) const override;
  Instruction *emitTrailingFence(IRBuilderBase &Builder, Instruction *Inst,
                                 AtomicOrdering Ord) const override;

  bool isFMAFasterThanFMulAndFAdd(const MachineFunction &MF,
                                  EVT VT) const override;

  ISD::NodeType getExtendForAtomicOps() const override {
    return ISD::SIGN_EXTEND;
  }

  ISD::NodeType getExtendForAtomicCmpSwapArg() const override;

  bool shouldTransformSignedTruncationCheck(EVT XVT,
                                            unsigned KeptBits) const override;

  TargetLowering::ShiftLegalizationStrategy
  preferredShiftLegalizationStrategy(SelectionDAG &DAG, SDNode *N,
                                     unsigned ExpansionFactor) const override {
    if (DAG.getMachineFunction().getFunction().hasMinSize())
      return ShiftLegalizationStrategy::LowerToLibcall;
    return TargetLowering::preferredShiftLegalizationStrategy(DAG, N,
                                                              ExpansionFactor);
  }

  bool isDesirableToCommuteWithShift(const SDNode *N,
                                     CombineLevel Level) const override;

  /// If a physical register, this returns the register that receives the
  /// exception address on entry to an EH pad.
  Register
  getExceptionPointerRegister(const Constant *PersonalityFn) const override;

  /// If a physical register, this returns the register that receives the
  /// exception typeid on entry to a landing pad.
  Register
  getExceptionSelectorRegister(const Constant *PersonalityFn) const override;

  bool shouldExtendTypeInLibCall(EVT Type) const override;
  bool shouldSignExtendTypeInLibCall(Type *Ty, bool IsSigned) const override;

  /// Returns the register with the specified architectural or ABI name. This
  /// method is necessary to lower the llvm.read_register.* and
  /// llvm.write_register.* intrinsics. Allocatable registers must be reserved
  /// with the clang -ffixed-xX flag for access to be allowed.
  Register getRegisterByName(const char *RegName, LLT VT,
                             const MachineFunction &MF) const override;

  // Lower incoming arguments, copy physregs into vregs
  SDValue LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv,
                               bool IsVarArg,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               const SDLoc &DL, SelectionDAG &DAG,
                               SmallVectorImpl<SDValue> &InVals) const override;
  bool CanLowerReturn(CallingConv::ID CallConv, MachineFunction &MF,
                      bool IsVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      LLVMContext &Context, const Type *RetTy) const override;
  SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      const SmallVectorImpl<SDValue> &OutVals, const SDLoc &DL,
                      SelectionDAG &DAG) const override;
  SDValue LowerCall(TargetLowering::CallLoweringInfo &CLI,
                    SmallVectorImpl<SDValue> &InVals) const override;

  bool shouldConvertConstantLoadToIntImm(const APInt &Imm,
                                         Type *Ty) const override;
  bool isUsedByReturnOnly(SDNode *N, SDValue &Chain) const override;
  bool mayBeEmittedAsTailCall(const CallInst *CI) const override;
  bool shouldConsiderGEPOffsetSplit() const override { return true; }

  bool decomposeMulByConstant(LLVMContext &Context, EVT VT,
                              SDValue C) const override;

  bool isMulAddWithConstProfitable(SDValue AddNode,
                                   SDValue ConstNode) const override;

  TargetLowering::AtomicExpansionKind
  shouldExpandAtomicRMWInIR(AtomicRMWInst *AI) const override;
  Value *emitMaskedAtomicRMWIntrinsic(IRBuilderBase &Builder, AtomicRMWInst *AI,
                                      Value *AlignedAddr, Value *Incr,
                                      Value *Mask, Value *ShiftAmt,
                                      AtomicOrdering Ord) const override;
  TargetLowering::AtomicExpansionKind
  shouldExpandAtomicCmpXchgInIR(AtomicCmpXchgInst *CI) const override;
  Value *emitMaskedAtomicCmpXchgIntrinsic(IRBuilderBase &Builder,
                                          AtomicCmpXchgInst *CI,
                                          Value *AlignedAddr, Value *CmpVal,
                                          Value *NewVal, Value *Mask,
                                          AtomicOrdering Ord) const override;

  /// Returns true if the target allows unaligned memory accesses of the
  /// specified type.
  bool allowsMisalignedMemoryAccesses(
      EVT VT, unsigned AddrSpace = 0, Align Alignment = Align(1),
      MachineMemOperand::Flags Flags = MachineMemOperand::MONone,
      unsigned *Fast = nullptr) const override;

  EVT getOptimalMemOpType(const MemOp &Op,
                          const AttributeList &FuncAttributes) const override;

  bool splitValueIntoRegisterParts(
      SelectionDAG & DAG, const SDLoc &DL, SDValue Val, SDValue *Parts,
      unsigned NumParts, MVT PartVT, std::optional<CallingConv::ID> CC)
      const override;

  SDValue joinRegisterPartsIntoValue(
      SelectionDAG & DAG, const SDLoc &DL, const SDValue *Parts,
      unsigned NumParts, MVT PartVT, EVT ValueVT,
      std::optional<CallingConv::ID> CC) const override;

  // Return the value of VLMax for the given vector type (i.e. SEW and LMUL)
  SDValue computeVLMax(MVT VecVT, const SDLoc &DL, SelectionDAG &DAG) const;

  static RISCVVType::VLMUL getLMUL(MVT VT);
  inline static unsigned computeVLMAX(unsigned VectorBits, unsigned EltSize,
                                      unsigned MinSize) {
    // Original equation:
    //   VLMAX = (VectorBits / EltSize) * LMUL
    //   where LMUL = MinSize / RISCV::RVVBitsPerBlock
    // The following equations have been reordered to prevent loss of precision
    // when calculating fractional LMUL.
    return ((VectorBits / EltSize) * MinSize) / RISCV::RVVBitsPerBlock;
  }

  // Return inclusive (low, high) bounds on the value of VLMAX for the
  // given scalable container type given known bounds on VLEN.
  static std::pair<unsigned, unsigned>
  computeVLMAXBounds(MVT ContainerVT, const RISCVSubtarget &Subtarget);

  static unsigned getRegClassIDForLMUL(RISCVVType::VLMUL LMul);
  static unsigned getSubregIndexByMVT(MVT VT, unsigned Index);
  static unsigned getRegClassIDForVecVT(MVT VT);
  static std::pair<unsigned, unsigned>
  decomposeSubvectorInsertExtractToSubRegs(MVT VecVT, MVT SubVecVT,
                                           unsigned InsertExtractIdx,
                                           const RISCVRegisterInfo *TRI);
  MVT getContainerForFixedLengthVector(MVT VT) const;

  bool shouldRemoveExtendFromGSIndex(SDValue Extend, EVT DataVT) const override;

  bool isLegalElementTypeForRVV(EVT ScalarTy) const;

  bool shouldConvertFpToSat(unsigned Op, EVT FPVT, EVT VT) const override;

  unsigned getJumpTableEncoding() const override;

  const MCExpr *LowerCustomJumpTableEntry(const MachineJumpTableInfo *MJTI,
                                          const MachineBasicBlock *MBB,
                                          unsigned uid,
                                          MCContext &Ctx) const override;

  bool isVScaleKnownToBeAPowerOfTwo() const override;

  bool getIndexedAddressParts(SDNode *Op, SDValue &Base, SDValue &Offset,
                              ISD::MemIndexedMode &AM, SelectionDAG &DAG) const;
  bool getPreIndexedAddressParts(SDNode *N, SDValue &Base, SDValue &Offset,
                                 ISD::MemIndexedMode &AM,
                                 SelectionDAG &DAG) const override;
  bool getPostIndexedAddressParts(SDNode *N, SDNode *Op, SDValue &Base,
                                  SDValue &Offset, ISD::MemIndexedMode &AM,
                                  SelectionDAG &DAG) const override;

  bool isLegalScaleForGatherScatter(uint64_t Scale,
                                    uint64_t ElemSize) const override {
    // Scaled addressing not supported on indexed load/stores
    return Scale == 1;
  }

  /// If the target has a standard location for the stack protector cookie,
  /// returns the address of that location. Otherwise, returns nullptr.
  Value *getIRStackGuard(IRBuilderBase &IRB) const override;

  /// Returns whether or not generating a interleaved load/store intrinsic for
  /// this type will be legal.
  bool isLegalInterleavedAccessType(VectorType *VTy, unsigned Factor,
                                    Align Alignment, unsigned AddrSpace,
                                    const DataLayout &) const;

  /// Return true if a stride load store of the given result type and
  /// alignment is legal.
  bool isLegalStridedLoadStore(EVT DataType, Align Alignment) const;

  unsigned getMaxSupportedInterleaveFactor() const override { return 8; }

  bool fallBackToDAGISel(const Instruction &Inst) const override;

  bool lowerInterleavedLoad(LoadInst *LI,
                            ArrayRef<ShuffleVectorInst *> Shuffles,
                            ArrayRef<unsigned> Indices,
                            unsigned Factor) const override;

  bool lowerInterleavedStore(StoreInst *SI, ShuffleVectorInst *SVI,
                             unsigned Factor) const override;

  bool lowerDeinterleaveIntrinsicToLoad(
      LoadInst *LI, ArrayRef<Value *> DeinterleaveValues) const override;

  bool lowerInterleaveIntrinsicToStore(
      StoreInst *SI, ArrayRef<Value *> InterleaveValues) const override;

  bool lowerInterleavedVPLoad(VPIntrinsic *Load, Value *Mask,
                              ArrayRef<Value *> DeinterleaveRes) const override;

  bool lowerInterleavedVPStore(VPIntrinsic *Store, Value *Mask,
                               ArrayRef<Value *> InterleaveOps) const override;

  bool supportKCFIBundles() const override { return true; }

  SDValue expandIndirectJTBranch(const SDLoc &dl, SDValue Value, SDValue Addr,
                                 int JTI, SelectionDAG &DAG) const override;

  MachineInstr *EmitKCFICheck(MachineBasicBlock &MBB,
                              MachineBasicBlock::instr_iterator &MBBI,
                              const TargetInstrInfo *TII) const override;

  /// True if stack clash protection is enabled for this functions.
  bool hasInlineStackProbe(const MachineFunction &MF) const override;

  unsigned getStackProbeSize(const MachineFunction &MF, Align StackAlign) const;

  MachineBasicBlock *emitDynamicProbedAlloc(MachineInstr &MI,
                                            MachineBasicBlock *MBB) const;

  ArrayRef<MCPhysReg> getRoundingControlRegisters() const override;

private:
  void analyzeInputArgs(MachineFunction &MF, CCState &CCInfo,
                        const SmallVectorImpl<ISD::InputArg> &Ins, bool IsRet,
                        RISCVCCAssignFn Fn) const;
  void analyzeOutputArgs(MachineFunction &MF, CCState &CCInfo,
                         const SmallVectorImpl<ISD::OutputArg> &Outs,
                         bool IsRet, CallLoweringInfo *CLI,
                         RISCVCCAssignFn Fn) const;

  template <class NodeTy>
  SDValue getAddr(NodeTy *N, SelectionDAG &DAG, bool IsLocal = true,
                  bool IsExternWeak = false) const;
  SDValue getStaticTLSAddr(GlobalAddressSDNode *N, SelectionDAG &DAG,
                           bool UseGOT) const;
  SDValue getDynamicTLSAddr(GlobalAddressSDNode *N, SelectionDAG &DAG) const;
  SDValue getTLSDescAddr(GlobalAddressSDNode *N, SelectionDAG &DAG) const;

  SDValue lowerConstantFP(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerBlockAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerConstantPool(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerJumpTable(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerGlobalTLSAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerSELECT(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerBRCOND(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVASTART(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFRAMEADDR(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerRETURNADDR(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerShiftLeftParts(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerShiftRightParts(SDValue Op, SelectionDAG &DAG, bool IsSRA) const;
  SDValue lowerSPLAT_VECTOR_PARTS(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVectorMaskSplat(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVectorMaskExt(SDValue Op, SelectionDAG &DAG,
                             int64_t ExtTrueVal) const;
  SDValue lowerVectorMaskTruncLike(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVectorTruncLike(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVectorFPExtendOrRoundLike(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerINSERT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerEXTRACT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerINTRINSIC_WO_CHAIN(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerINTRINSIC_W_CHAIN(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerINTRINSIC_VOID(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVPREDUCE(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVECREDUCE(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVectorMaskVecReduction(SDValue Op, SelectionDAG &DAG,
                                      bool IsVP) const;
  SDValue lowerFPVECREDUCE(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerINSERT_SUBVECTOR(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerEXTRACT_SUBVECTOR(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVECTOR_DEINTERLEAVE(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVECTOR_INTERLEAVE(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerSTEP_VECTOR(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVECTOR_REVERSE(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVECTOR_SPLICE(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerABS(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerMaskedLoad(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerMaskedStore(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVectorCompress(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFixedLengthVectorFCOPYSIGNToRVV(SDValue Op,
                                               SelectionDAG &DAG) const;
  SDValue lowerMaskedGather(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerMaskedScatter(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFixedLengthVectorLoadToRVV(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFixedLengthVectorStoreToRVV(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFixedLengthVectorSetccToRVV(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFixedLengthVectorSelectToRVV(SDValue Op,
                                            SelectionDAG &DAG) const;
  SDValue lowerToScalableOp(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerIS_FPCLASS(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVPOp(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerLogicVPOp(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVPExtMaskOp(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVPSetCCMaskOp(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVPMergeMask(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVPSplatExperimental(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVPSpliceExperimental(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVPReverseExperimental(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVPFPIntConvOp(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVPStridedLoad(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVPStridedStore(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVPCttzElements(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFixedLengthVectorExtendToRVV(SDValue Op, SelectionDAG &DAG,
                                            unsigned ExtendOpc) const;
  SDValue lowerGET_ROUNDING(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerSET_ROUNDING(SDValue Op, SelectionDAG &DAG) const;

  SDValue lowerEH_DWARF_CFA(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerCTLZ_CTTZ_ZERO_UNDEF(SDValue Op, SelectionDAG &DAG) const;

  SDValue lowerStrictFPExtendOrRoundLike(SDValue Op, SelectionDAG &DAG) const;

  SDValue lowerVectorStrictFSetcc(SDValue Op, SelectionDAG &DAG) const;

  SDValue lowerDYNAMIC_STACKALLOC(SDValue Op, SelectionDAG &DAG) const;

  SDValue expandUnalignedRVVLoad(SDValue Op, SelectionDAG &DAG) const;
  SDValue expandUnalignedRVVStore(SDValue Op, SelectionDAG &DAG) const;

  SDValue lowerINIT_TRAMPOLINE(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerADJUST_TRAMPOLINE(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerPARTIAL_REDUCE_MLA(SDValue Op, SelectionDAG &DAG) const;

  bool isEligibleForTailCallOptimization(
      CCState &CCInfo, CallLoweringInfo &CLI, MachineFunction &MF,
      const SmallVector<CCValAssign, 16> &ArgLocs) const;

  /// Generate error diagnostics if any register used by CC has been marked
  /// reserved.
  void validateCCReservedRegs(
      const SmallVectorImpl<std::pair<llvm::Register, llvm::SDValue>> &Regs,
      MachineFunction &MF) const;

  bool useRVVForFixedLengthVectorVT(MVT VT) const;

  MVT getVPExplicitVectorLengthTy() const override;

  bool shouldExpandGetVectorLength(EVT TripCountVT, unsigned VF,
                                   bool IsScalable) const override;

  /// RVV code generation for fixed length vectors does not lower all
  /// BUILD_VECTORs. This makes BUILD_VECTOR legalisation a source of stores to
  /// merge. However, merging them creates a BUILD_VECTOR that is just as
  /// illegal as the original, thus leading to an infinite legalisation loop.
  /// NOTE: Once BUILD_VECTOR can be custom lowered for all legal vector types,
  /// this override can be removed.
  bool mergeStoresAfterLegalization(EVT VT) const override;

  /// Disable normalizing
  /// select(N0&N1, X, Y) => select(N0, select(N1, X, Y), Y) and
  /// select(N0|N1, X, Y) => select(N0, select(N1, X, Y, Y))
  /// RISC-V doesn't have flags so it's better to perform the and/or in a GPR.
  bool shouldNormalizeToSelectSequence(LLVMContext &, EVT) const override {
    return false;
  }

  /// Disables storing and loading vectors by default when there are function
  /// calls between the load and store, since these are more expensive than just
  /// using scalars
  bool shouldMergeStoreOfLoadsOverCall(EVT SrcVT, EVT MergedVT) const override {
    return !MergedVT.isVector() || SrcVT.isVector();
  }

  /// For available scheduling models FDIV + two independent FMULs are much
  /// faster than two FDIVs.
  unsigned combineRepeatedFPDivisors() const override;

  SDValue BuildSDIVPow2(SDNode *N, const APInt &Divisor, SelectionDAG &DAG,
                        SmallVectorImpl<SDNode *> &Created) const override;

  bool shouldFoldSelectWithSingleBitTest(EVT VT,
                                         const APInt &AndMask) const override;

  unsigned getMinimumJumpTableEntries() const override;

  SDValue emitFlushICache(SelectionDAG &DAG, SDValue InChain, SDValue Start,
                          SDValue End, SDValue Flags, SDLoc DL) const;

  std::pair<const TargetRegisterClass *, uint8_t>
  findRepresentativeClass(const TargetRegisterInfo *TRI, MVT VT) const override;
};

namespace RISCVVIntrinsicsTable {

struct RISCVVIntrinsicInfo {
  unsigned IntrinsicID;
  uint8_t ScalarOperand;
  uint8_t VLOperand;
  bool hasScalarOperand() const {
    // 0xF is not valid. See NoScalarOperand in IntrinsicsRISCV.td.
    return ScalarOperand != 0xF;
  }
  bool hasVLOperand() const {
    // 0x1F is not valid. See NoVLOperand in IntrinsicsRISCV.td.
    return VLOperand != 0x1F;
  }
};

using namespace RISCV;

#define GET_RISCVVIntrinsicsTable_DECL
#include "RISCVGenSearchableTables.inc"
#undef GET_RISCVVIntrinsicsTable_DECL

} // end namespace RISCVVIntrinsicsTable

} // end namespace llvm

#endif
