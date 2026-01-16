//===-- HexagonISelLowering.h - Hexagon DAG Lowering Interface --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that Hexagon uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HEXAGON_HEXAGONISELLOWERING_H
#define LLVM_LIB_TARGET_HEXAGON_HEXAGONISELLOWERING_H

#include "Hexagon.h"
#include "MCTargetDesc/HexagonMCTargetDesc.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGenTypes/MachineValueType.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/InlineAsm.h"
#include <cstdint>
#include <utility>

namespace llvm {

class HexagonSubtarget;

class HexagonTargetLowering : public TargetLowering {
  int VarArgsFrameOffset;   // Frame offset to start of varargs area.
  const HexagonTargetMachine &HTM;
  const HexagonSubtarget &Subtarget;

public:
  explicit HexagonTargetLowering(const TargetMachine &TM,
                                 const HexagonSubtarget &ST);

  /// IsEligibleForTailCallOptimization - Check whether the call is eligible
  /// for tail call optimization. Targets which want to do tail call
  /// optimization should implement this function.
  bool IsEligibleForTailCallOptimization(SDValue Callee,
      CallingConv::ID CalleeCC, bool isVarArg, bool isCalleeStructRet,
      bool isCallerStructRet, const SmallVectorImpl<ISD::OutputArg> &Outs,
      const SmallVectorImpl<SDValue> &OutVals,
      const SmallVectorImpl<ISD::InputArg> &Ins, SelectionDAG& DAG) const;

  bool getTgtMemIntrinsic(IntrinsicInfo &Info, const CallBase &I,
                          MachineFunction &MF,
                          unsigned Intrinsic) const override;

  bool isTruncateFree(Type *Ty1, Type *Ty2) const override;
  bool isTruncateFree(EVT VT1, EVT VT2) const override;

  bool isCheapToSpeculateCttz(Type *) const override { return true; }
  bool isCheapToSpeculateCtlz(Type *) const override { return true; }
  bool isCtlzFast() const override { return true; }

  bool hasBitTest(SDValue X, SDValue Y) const override;

  bool allowTruncateForTailCall(Type *Ty1, Type *Ty2) const override;

  bool isMaskAndCmp0FoldingBeneficial(const Instruction &AndI) const override;

  bool isUsedByReturnOnly(SDNode *N, SDValue &Chain) const override;

  /// Return true if an FMA operation is faster than a pair of mul and add
  /// instructions. fmuladd intrinsics will be expanded to FMAs when this
  /// method returns true (and FMAs are legal), otherwise fmuladd is
  /// expanded to mul + add.
  bool isFMAFasterThanFMulAndFAdd(const MachineFunction &,
                                  EVT) const override;

  // Should we expand the build vector with shuffles?
  bool shouldExpandBuildVectorWithShuffles(EVT VT,
      unsigned DefinedValues) const override;
  bool isExtractSubvectorCheap(EVT ResVT, EVT SrcVT,
      unsigned Index) const override;

  bool isTargetCanonicalConstantNode(SDValue Op) const override;

  bool isShuffleMaskLegal(ArrayRef<int> Mask, EVT VT) const override;
  LegalizeTypeAction getPreferredVectorAction(MVT VT) const override;
  LegalizeAction getCustomOperationAction(SDNode &Op) const override;

  SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;
  void LowerOperationWrapper(SDNode *N, SmallVectorImpl<SDValue> &Results,
                             SelectionDAG &DAG) const override;
  void ReplaceNodeResults(SDNode *N, SmallVectorImpl<SDValue> &Results,
                          SelectionDAG &DAG) const override;

  std::pair<MVT, unsigned>
  handleMaskRegisterForCallingConv(const HexagonSubtarget &Subtarget,
                                   EVT VT) const;

  SDValue LowerBUILD_VECTOR(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerCONCAT_VECTORS(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerEXTRACT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerEXTRACT_SUBVECTOR(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerINSERT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerINSERT_SUBVECTOR(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerVECTOR_SHUFFLE(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerVECTOR_SHIFT(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerROTL(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerBITCAST(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerANY_EXTEND(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSIGN_EXTEND(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerZERO_EXTEND(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerLoad(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerStore(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerUnalignedLoad(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerUAddSubO(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerUAddSubOCarry(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerDYNAMIC_STACKALLOC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerINLINEASM(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFDIV(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerPREFETCH(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerREADCYCLECOUNTER(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerREADSTEADYCOUNTER(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerEH_LABEL(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerEH_RETURN(SDValue Op, SelectionDAG &DAG) const;
  SDValue
  LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
                       const SmallVectorImpl<ISD::InputArg> &Ins,
                       const SDLoc &dl, SelectionDAG &DAG,
                       SmallVectorImpl<SDValue> &InVals) const override;
  SDValue LowerGLOBALADDRESS(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerBlockAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerGlobalTLSAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerToTLSGeneralDynamicModel(GlobalAddressSDNode *GA,
      SelectionDAG &DAG) const;
  SDValue LowerToTLSInitialExecModel(GlobalAddressSDNode *GA,
      SelectionDAG &DAG) const;
  SDValue LowerToTLSLocalExecModel(GlobalAddressSDNode *GA,
      SelectionDAG &DAG) const;
  SDValue GetDynamicTLSAddr(SelectionDAG &DAG, SDValue Chain,
      GlobalAddressSDNode *GA, SDValue InGlue, EVT PtrVT,
      unsigned ReturnReg, unsigned char OperandGlues) const;
  SDValue LowerGLOBAL_OFFSET_TABLE(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerCall(TargetLowering::CallLoweringInfo &CLI,
      SmallVectorImpl<SDValue> &InVals) const override;
  SDValue LowerCallResult(SDValue Chain, SDValue InGlue,
                          CallingConv::ID CallConv, bool isVarArg,
                          const SmallVectorImpl<ISD::InputArg> &Ins,
                          const SDLoc &dl, SelectionDAG &DAG,
                          SmallVectorImpl<SDValue> &InVals,
                          const SmallVectorImpl<SDValue> &OutVals,
                          SDValue Callee) const;

  SDValue LowerSETCC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerVSELECT(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFRAMEADDR(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerATOMIC_FENCE(SDValue Op, SelectionDAG& DAG) const;
  SDValue LowerRETURNADDR(SDValue Op, SelectionDAG &DAG) const;

  bool CanLowerReturn(CallingConv::ID CallConv,
                      MachineFunction &MF, bool isVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      LLVMContext &Context, const Type *RetTy) const override;

  SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      const SmallVectorImpl<SDValue> &OutVals,
                      const SDLoc &dl, SelectionDAG &DAG) const override;

  SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const override;

  bool mayBeEmittedAsTailCall(const CallInst *CI) const override;

  Register getRegisterByName(const char* RegName, LLT VT,
                             const MachineFunction &MF) const override;

  unsigned getVectorTypeBreakdownForCallingConv(LLVMContext &Context,
                                                CallingConv::ID CC, EVT VT,
                                                EVT &IntermediateVT,
                                                unsigned &NumIntermediates,
                                                MVT &RegisterVT) const override;

  MVT getRegisterTypeForCallingConv(LLVMContext &Context, CallingConv::ID CC,
                                    EVT VT) const override;
  /// If a physical register, this returns the register that receives the
  /// exception address on entry to an EH pad.
  Register
  getExceptionPointerRegister(const Constant *PersonalityFn) const override {
    return Hexagon::R0;
  }

  /// If a physical register, this returns the register that receives the
  /// exception typeid on entry to a landing pad.
  Register
  getExceptionSelectorRegister(const Constant *PersonalityFn) const override {
    return Hexagon::R1;
  }

  SDValue LowerVASTART(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerVACOPY(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerConstantPool(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerJumpTable(SDValue Op, SelectionDAG &DAG) const;

  EVT getSetCCResultType(const DataLayout &, LLVMContext &C,
                         EVT VT) const override {
    if (!VT.isVector())
      return MVT::i1;
    else
      return EVT::getVectorVT(C, MVT::i1, VT.getVectorNumElements());
  }

  bool getPostIndexedAddressParts(SDNode *N, SDNode *Op,
                                  SDValue &Base, SDValue &Offset,
                                  ISD::MemIndexedMode &AM,
                                  SelectionDAG &DAG) const override;

  ConstraintType getConstraintType(StringRef Constraint) const override;

  std::pair<unsigned, const TargetRegisterClass *>
  getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                               StringRef Constraint, MVT VT) const override;

  // Intrinsics
  SDValue LowerINTRINSIC_WO_CHAIN(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerINTRINSIC_VOID(SDValue Op, SelectionDAG &DAG) const;
  /// isLegalAddressingMode - Return true if the addressing mode represented
  /// by AM is legal for this target, for a load/store of the specified type.
  /// The type may be VoidTy, in which case only return true if the addressing
  /// mode is legal for a load/store of any legal type.
  /// TODO: Handle pre/postinc as well.
  bool isLegalAddressingMode(const DataLayout &DL, const AddrMode &AM,
                             Type *Ty, unsigned AS,
                             Instruction *I = nullptr) const override;
  /// Return true if folding a constant offset with the given GlobalAddress
  /// is legal.  It is frequently not legal in PIC relocation models.
  bool isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const override;

  bool isFPImmLegal(const APFloat &Imm, EVT VT,
                    bool ForCodeSize) const override;

  /// isLegalICmpImmediate - Return true if the specified immediate is legal
  /// icmp immediate, that is the target has icmp instructions which can
  /// compare a register against the immediate without having to materialize
  /// the immediate into a register.
  bool isLegalICmpImmediate(int64_t Imm) const override;

  EVT getOptimalMemOpType(LLVMContext &Context, const MemOp &Op,
                          const AttributeList &FuncAttributes) const override;

  bool allowsMemoryAccess(LLVMContext &Context, const DataLayout &DL, EVT VT,
                          unsigned AddrSpace, Align Alignment,
                          MachineMemOperand::Flags Flags,
                          unsigned *Fast) const override;

  bool allowsMisalignedMemoryAccesses(EVT VT, unsigned AddrSpace,
                                      Align Alignment,
                                      MachineMemOperand::Flags Flags,
                                      unsigned *Fast) const override;

  /// Returns relocation base for the given PIC jumptable.
  SDValue getPICJumpTableRelocBase(SDValue Table, SelectionDAG &DAG)
                                   const override;

  /// Returns true if it is beneficial to convert a load of a constant
  /// to just the constant itself.
  bool shouldConvertConstantLoadToIntImm(const APInt &Imm,
                                         Type *Ty) const override;

  bool shouldReduceLoadWidth(SDNode *Load, ISD::LoadExtType ExtTy, EVT NewVT,
                             std::optional<unsigned> ByteOffset) const override;

  void AdjustInstrPostInstrSelection(MachineInstr &MI,
                                     SDNode *Node) const override;

  // Handling of atomic RMW instructions.
  Value *emitLoadLinked(IRBuilderBase &Builder, Type *ValueTy, Value *Addr,
                        AtomicOrdering Ord) const override;
  Value *emitStoreConditional(IRBuilderBase &Builder, Value *Val, Value *Addr,
                              AtomicOrdering Ord) const override;
  AtomicExpansionKind shouldExpandAtomicLoadInIR(LoadInst *LI) const override;
  AtomicExpansionKind shouldExpandAtomicStoreInIR(StoreInst *SI) const override;
  AtomicExpansionKind
  shouldExpandAtomicCmpXchgInIR(const AtomicCmpXchgInst *AI) const override;

  AtomicExpansionKind
  shouldExpandAtomicRMWInIR(const AtomicRMWInst *AI) const override {
    return AtomicExpansionKind::LLSC;
  }

private:
  void initializeHVXLowering();
  unsigned getPreferredHvxVectorAction(MVT VecTy) const;
  unsigned getCustomHvxOperationAction(SDNode &Op) const;

  bool validateConstPtrAlignment(SDValue Ptr, Align NeedAlign, const SDLoc &dl,
                                 SelectionDAG &DAG) const;
  SDValue replaceMemWithUndef(SDValue Op, SelectionDAG &DAG) const;

  std::pair<SDValue,int> getBaseAndOffset(SDValue Addr) const;

  bool getBuildVectorConstInts(ArrayRef<SDValue> Values, MVT VecTy,
                               SelectionDAG &DAG,
                               MutableArrayRef<ConstantInt*> Consts) const;
  SDValue buildVector32(ArrayRef<SDValue> Elem, const SDLoc &dl, MVT VecTy,
                        SelectionDAG &DAG) const;
  SDValue buildVector64(ArrayRef<SDValue> Elem, const SDLoc &dl, MVT VecTy,
                        SelectionDAG &DAG) const;
  SDValue extractVector(SDValue VecV, SDValue IdxV, const SDLoc &dl,
                        MVT ValTy, MVT ResTy, SelectionDAG &DAG) const;
  SDValue extractVectorPred(SDValue VecV, SDValue IdxV, const SDLoc &dl,
                            MVT ValTy, MVT ResTy, SelectionDAG &DAG) const;
  SDValue insertVector(SDValue VecV, SDValue ValV, SDValue IdxV,
                       const SDLoc &dl, MVT ValTy, SelectionDAG &DAG) const;
  SDValue insertVectorPred(SDValue VecV, SDValue ValV, SDValue IdxV,
                           const SDLoc &dl, MVT ValTy, SelectionDAG &DAG) const;
  SDValue expandPredicate(SDValue Vec32, const SDLoc &dl,
                          SelectionDAG &DAG) const;
  SDValue contractPredicate(SDValue Vec64, const SDLoc &dl,
                            SelectionDAG &DAG) const;
  SDValue getSplatValue(SDValue Op, SelectionDAG &DAG) const;
  SDValue getVectorShiftByInt(SDValue Op, SelectionDAG &DAG) const;
  SDValue appendUndef(SDValue Val, MVT ResTy, SelectionDAG &DAG) const;
  SDValue getCombine(SDValue Hi, SDValue Lo, const SDLoc &dl, MVT ResTy,
                     SelectionDAG &DAG) const;

  bool isUndef(SDValue Op) const {
    if (Op.isMachineOpcode())
      return Op.getMachineOpcode() == TargetOpcode::IMPLICIT_DEF;
    return Op.getOpcode() == ISD::UNDEF;
  }
  SDValue getInstr(unsigned MachineOpc, const SDLoc &dl, MVT Ty,
                   ArrayRef<SDValue> Ops, SelectionDAG &DAG) const {
    SDNode *N = DAG.getMachineNode(MachineOpc, dl, Ty, Ops);
    return SDValue(N, 0);
  }
  SDValue getZero(const SDLoc &dl, MVT Ty, SelectionDAG &DAG) const;

  using VectorPair = std::pair<SDValue, SDValue>;
  using TypePair = std::pair<MVT, MVT>;

  SDValue getInt(unsigned IntId, MVT ResTy, ArrayRef<SDValue> Ops,
                 const SDLoc &dl, SelectionDAG &DAG) const;

  MVT ty(SDValue Op) const {
    return Op.getValueType().getSimpleVT();
  }
  TypePair ty(const VectorPair &Ops) const {
    return { Ops.first.getValueType().getSimpleVT(),
             Ops.second.getValueType().getSimpleVT() };
  }
  MVT tyScalar(MVT Ty) const {
    if (!Ty.isVector())
      return Ty;
    return MVT::getIntegerVT(Ty.getSizeInBits());
  }
  MVT tyVector(MVT Ty, MVT ElemTy) const {
    if (Ty.isVector() && Ty.getVectorElementType() == ElemTy)
      return Ty;
    unsigned TyWidth = Ty.getSizeInBits();
    unsigned ElemWidth = ElemTy.getSizeInBits();
    assert((TyWidth % ElemWidth) == 0);
    return MVT::getVectorVT(ElemTy, TyWidth/ElemWidth);
  }

  MVT typeJoin(const TypePair &Tys) const;
  TypePair typeSplit(MVT Ty) const;
  MVT typeExtElem(MVT VecTy, unsigned Factor) const;
  MVT typeTruncElem(MVT VecTy, unsigned Factor) const;
  TypePair typeExtendToWider(MVT Ty0, MVT Ty1) const;
  TypePair typeWidenToWider(MVT Ty0, MVT Ty1) const;
  MVT typeLegalize(MVT Ty, SelectionDAG &DAG) const;
  MVT typeWidenToHvx(MVT Ty) const;

  SDValue opJoin(const VectorPair &Ops, const SDLoc &dl,
                 SelectionDAG &DAG) const;
  VectorPair opSplit(SDValue Vec, const SDLoc &dl, SelectionDAG &DAG) const;
  SDValue opCastElem(SDValue Vec, MVT ElemTy, SelectionDAG &DAG) const;

  SDValue LoHalf(SDValue V, SelectionDAG &DAG) const {
    MVT Ty = ty(V);
    const SDLoc &dl(V);
    if (!Ty.isVector()) {
      assert(Ty.getSizeInBits() == 64);
      return DAG.getTargetExtractSubreg(Hexagon::isub_lo, dl, MVT::i32, V);
    }
    MVT HalfTy = typeSplit(Ty).first;
    SDValue Idx = getZero(dl, MVT::i32, DAG);
    return DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, HalfTy, V, Idx);
  }
  SDValue HiHalf(SDValue V, SelectionDAG &DAG) const {
    MVT Ty = ty(V);
    const SDLoc &dl(V);
    if (!Ty.isVector()) {
      assert(Ty.getSizeInBits() == 64);
      return DAG.getTargetExtractSubreg(Hexagon::isub_hi, dl, MVT::i32, V);
    }
    MVT HalfTy = typeSplit(Ty).first;
    SDValue Idx = DAG.getConstant(HalfTy.getVectorNumElements(), dl, MVT::i32);
    return DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, HalfTy, V, Idx);
  }

  bool allowsHvxMemoryAccess(MVT VecTy, MachineMemOperand::Flags Flags,
                             unsigned *Fast) const;
  bool allowsHvxMisalignedMemoryAccesses(MVT VecTy,
                                         MachineMemOperand::Flags Flags,
                                         unsigned *Fast) const;
  void AdjustHvxInstrPostInstrSelection(MachineInstr &MI, SDNode *Node) const;

  bool isHvxSingleTy(MVT Ty) const;
  bool isHvxPairTy(MVT Ty) const;
  bool isHvxBoolTy(MVT Ty) const;
  SDValue convertToByteIndex(SDValue ElemIdx, MVT ElemTy,
                             SelectionDAG &DAG) const;
  SDValue getIndexInWord32(SDValue Idx, MVT ElemTy, SelectionDAG &DAG) const;
  SDValue getByteShuffle(const SDLoc &dl, SDValue Op0, SDValue Op1,
                         ArrayRef<int> Mask, SelectionDAG &DAG) const;

  SDValue buildHvxVectorReg(ArrayRef<SDValue> Values, const SDLoc &dl,
                            MVT VecTy, SelectionDAG &DAG) const;
  SDValue buildHvxVectorPred(ArrayRef<SDValue> Values, const SDLoc &dl,
                             MVT VecTy, SelectionDAG &DAG) const;
  SDValue createHvxPrefixPred(SDValue PredV, const SDLoc &dl,
                              unsigned BitBytes, bool ZeroFill,
                              SelectionDAG &DAG) const;
  SDValue extractHvxElementReg(SDValue VecV, SDValue IdxV, const SDLoc &dl,
                               MVT ResTy, SelectionDAG &DAG) const;
  SDValue extractHvxElementPred(SDValue VecV, SDValue IdxV, const SDLoc &dl,
                                MVT ResTy, SelectionDAG &DAG) const;
  SDValue insertHvxElementReg(SDValue VecV, SDValue IdxV, SDValue ValV,
                              const SDLoc &dl, SelectionDAG &DAG) const;
  SDValue insertHvxElementPred(SDValue VecV, SDValue IdxV, SDValue ValV,
                               const SDLoc &dl, SelectionDAG &DAG) const;
  SDValue extractHvxSubvectorReg(SDValue OrigOp, SDValue VecV, SDValue IdxV,
                                 const SDLoc &dl, MVT ResTy, SelectionDAG &DAG)
                                 const;
  SDValue extractHvxSubvectorPred(SDValue VecV, SDValue IdxV, const SDLoc &dl,
                                  MVT ResTy, SelectionDAG &DAG) const;
  SDValue insertHvxSubvectorReg(SDValue VecV, SDValue SubV, SDValue IdxV,
                                const SDLoc &dl, SelectionDAG &DAG) const;
  SDValue insertHvxSubvectorPred(SDValue VecV, SDValue SubV, SDValue IdxV,
                                 const SDLoc &dl, SelectionDAG &DAG) const;
  SDValue extendHvxVectorPred(SDValue VecV, const SDLoc &dl, MVT ResTy,
                              bool ZeroExt, SelectionDAG &DAG) const;
  SDValue compressHvxPred(SDValue VecQ, const SDLoc &dl, MVT ResTy,
                          SelectionDAG &DAG) const;
  SDValue resizeToWidth(SDValue VecV, MVT ResTy, bool Signed, const SDLoc &dl,
                        SelectionDAG &DAG) const;
  SDValue extractSubvector(SDValue Vec, MVT SubTy, unsigned SubIdx,
                           SelectionDAG &DAG) const;
  VectorPair emitHvxAddWithOverflow(SDValue A, SDValue B, const SDLoc &dl,
                                    bool Signed, SelectionDAG &DAG) const;
  VectorPair emitHvxShiftRightRnd(SDValue Val, unsigned Amt, bool Signed,
                                  SelectionDAG &DAG) const;
  SDValue emitHvxMulHsV60(SDValue A, SDValue B, const SDLoc &dl,
                          SelectionDAG &DAG) const;
  SDValue emitHvxMulLoHiV60(SDValue A, bool SignedA, SDValue B, bool SignedB,
                            const SDLoc &dl, SelectionDAG &DAG) const;
  SDValue emitHvxMulLoHiV62(SDValue A, bool SignedA, SDValue B, bool SignedB,
                            const SDLoc &dl, SelectionDAG &DAG) const;

  SDValue LowerHvxBuildVector(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerHvxSplatVector(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerHvxConcatVectors(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerHvxExtractElement(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerHvxInsertElement(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerHvxExtractSubvector(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerHvxInsertSubvector(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerHvxBitcast(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerHvxAnyExt(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerHvxSignExt(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerHvxZeroExt(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerHvxCttz(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerHvxMulh(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerHvxMulLoHi(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerHvxExtend(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerHvxSelect(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerHvxShift(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerHvxFunnelShift(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerHvxIntrinsic(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerHvxMaskedOp(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerHvxFpExtend(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerHvxFpToInt(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerHvxIntToFp(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerHvxPred32ToFp(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerHvxPred64ToFp(SDValue Op, SelectionDAG &DAG) const;
  SDValue ExpandHvxFpToInt(SDValue Op, SelectionDAG &DAG) const;
  SDValue ExpandHvxIntToFp(SDValue Op, SelectionDAG &DAG) const;

  VectorPair SplitVectorOp(SDValue Op, SelectionDAG &DAG) const;

  SDValue SplitHvxMemOp(SDValue Op, SelectionDAG &DAG) const;
  SDValue WidenHvxLoad(SDValue Op, SelectionDAG &DAG) const;
  SDValue WidenHvxStore(SDValue Op, SelectionDAG &DAG) const;
  SDValue WidenHvxSetCC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LegalizeHvxResize(SDValue Op, SelectionDAG &DAG) const;
  SDValue ExpandHvxResizeIntoSteps(SDValue Op, SelectionDAG &DAG) const;
  SDValue EqualizeFpIntConversion(SDValue Op, SelectionDAG &DAG) const;

  SDValue CreateTLWrapper(SDValue Op, SelectionDAG &DAG) const;
  SDValue RemoveTLWrapper(SDValue Op, SelectionDAG &DAG) const;

  std::pair<const TargetRegisterClass*, uint8_t>
  findRepresentativeClass(const TargetRegisterInfo *TRI, MVT VT)
      const override;

  bool shouldSplitToHvx(MVT Ty, SelectionDAG &DAG) const;
  bool shouldWidenToHvx(MVT Ty, SelectionDAG &DAG) const;
  bool isHvxOperation(SDNode *N, SelectionDAG &DAG) const;
  SDValue LowerHvxOperation(SDValue Op, SelectionDAG &DAG) const;
  void LowerHvxOperationWrapper(SDNode *N, SmallVectorImpl<SDValue> &Results,
                                SelectionDAG &DAG) const;
  void ReplaceHvxNodeResults(SDNode *N, SmallVectorImpl<SDValue> &Results,
                             SelectionDAG &DAG) const;

  SDValue combineTruncateBeforeLegal(SDValue Op, DAGCombinerInfo &DCI) const;
  SDValue combineConcatVectorsBeforeLegal(SDValue Op, DAGCombinerInfo & DCI)
      const;
  SDValue combineVectorShuffleBeforeLegal(SDValue Op, DAGCombinerInfo & DCI)
      const;

  SDValue PerformHvxDAGCombine(SDNode * N, DAGCombinerInfo & DCI) const;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_HEXAGON_HEXAGONISELLOWERING_H
