//===-- SystemZISelLowering.h - SystemZ DAG lowering interface --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that SystemZ uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZISELLOWERING_H
#define LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZISELLOWERING_H

#include "SystemZ.h"
#include "SystemZInstrInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLowering.h"
#include <optional>

namespace llvm {

namespace SystemZICMP {
// Describes whether an integer comparison needs to be signed or unsigned,
// or whether either type is OK.
enum {
  Any,
  UnsignedOnly,
  SignedOnly
};
} // end namespace SystemZICMP

class SystemZSubtarget;

class SystemZTargetLowering : public TargetLowering {
public:
  explicit SystemZTargetLowering(const TargetMachine &TM,
                                 const SystemZSubtarget &STI);

  bool useSoftFloat() const override;

  // Override TargetLowering.
  MVT getScalarShiftAmountTy(const DataLayout &, EVT) const override {
    return MVT::i32;
  }
  unsigned getVectorIdxWidth(const DataLayout &DL) const override {
    // Only the lower 12 bits of an element index are used, so we don't
    // want to clobber the upper 32 bits of a GPR unnecessarily.
    return 32;
  }
  TargetLoweringBase::LegalizeTypeAction getPreferredVectorAction(MVT VT)
    const override {
    // Widen subvectors to the full width rather than promoting integer
    // elements.  This is better because:
    //
    // (a) it means that we can handle the ABI for passing and returning
    //     sub-128 vectors without having to handle them as legal types.
    //
    // (b) we don't have instructions to extend on load and truncate on store,
    //     so promoting the integers is less efficient.
    //
    // (c) there are no multiplication instructions for the widest integer
    //     type (v2i64).
    if (VT.getScalarSizeInBits() % 8 == 0)
      return TypeWidenVector;
    return TargetLoweringBase::getPreferredVectorAction(VT);
  }
  unsigned
  getNumRegisters(LLVMContext &Context, EVT VT,
                  std::optional<MVT> RegisterVT) const override {
    // i128 inline assembly operand.
    if (VT == MVT::i128 && RegisterVT && *RegisterVT == MVT::Untyped)
      return 1;
    return TargetLowering::getNumRegisters(Context, VT);
  }
  MVT getRegisterTypeForCallingConv(LLVMContext &Context, CallingConv::ID CC,
                                    EVT VT) const override {
    // 128-bit single-element vector types are passed like other vectors,
    // not like their element type.
    if (VT.isVector() && VT.getSizeInBits() == 128 &&
        VT.getVectorNumElements() == 1)
      return MVT::v16i8;
    return TargetLowering::getRegisterTypeForCallingConv(Context, CC, VT);
  }
  bool isCheapToSpeculateCtlz(Type *) const override { return true; }
  bool isCheapToSpeculateCttz(Type *) const override { return true; }
  bool preferZeroCompareBranch() const override { return true; }
  bool isMaskAndCmp0FoldingBeneficial(const Instruction &AndI) const override {
    ConstantInt* Mask = dyn_cast<ConstantInt>(AndI.getOperand(1));
    return Mask && Mask->getValue().isIntN(16);
  }
  bool convertSetCCLogicToBitwiseLogic(EVT VT) const override {
    return VT.isScalarInteger();
  }
  EVT getSetCCResultType(const DataLayout &DL, LLVMContext &,
                         EVT) const override;
  bool isFMAFasterThanFMulAndFAdd(const MachineFunction &MF,
                                  EVT VT) const override;
  bool isFPImmLegal(const APFloat &Imm, EVT VT,
                    bool ForCodeSize) const override;
  bool ShouldShrinkFPConstant(EVT VT) const override {
    // Do not shrink 64-bit FP constpool entries since LDEB is slower than
    // LD, and having the full constant in memory enables reg/mem opcodes.
    return VT != MVT::f64;
  }
  MachineBasicBlock *emitEHSjLjSetJmp(MachineInstr &MI,
                                      MachineBasicBlock *MBB) const;

  MachineBasicBlock *emitEHSjLjLongJmp(MachineInstr &MI,
                                       MachineBasicBlock *MBB) const;

  bool hasInlineStackProbe(const MachineFunction &MF) const override;
  AtomicExpansionKind shouldCastAtomicLoadInIR(LoadInst *LI) const override;
  AtomicExpansionKind shouldCastAtomicStoreInIR(StoreInst *SI) const override;
  AtomicExpansionKind
  shouldExpandAtomicRMWInIR(AtomicRMWInst *RMW) const override;
  bool isLegalICmpImmediate(int64_t Imm) const override;
  bool isLegalAddImmediate(int64_t Imm) const override;
  bool isLegalAddressingMode(const DataLayout &DL, const AddrMode &AM, Type *Ty,
                             unsigned AS,
                             Instruction *I = nullptr) const override;
  bool allowsMisalignedMemoryAccesses(EVT VT, unsigned AS, Align Alignment,
                                      MachineMemOperand::Flags Flags,
                                      unsigned *Fast) const override;
  bool
  findOptimalMemOpLowering(LLVMContext &Context, std::vector<EVT> &MemOps,
                           unsigned Limit, const MemOp &Op, unsigned DstAS,
                           unsigned SrcAS,
                           const AttributeList &FuncAttributes) const override;
  EVT getOptimalMemOpType(LLVMContext &Context, const MemOp &Op,
                          const AttributeList &FuncAttributes) const override;
  bool isTruncateFree(Type *, Type *) const override;
  bool isTruncateFree(EVT, EVT) const override;

  bool shouldFormOverflowOp(unsigned Opcode, EVT VT,
                            bool MathUsed) const override {
    // Form add and sub with overflow intrinsics regardless of any extra
    // users of the math result.
    return VT == MVT::i32 || VT == MVT::i64 || VT == MVT::i128;
  }

  bool shouldConsiderGEPOffsetSplit() const override { return true; }

  bool preferSelectsOverBooleanArithmetic(EVT VT) const override {
    return true;
  }

  // This function currently returns cost for srl/ipm/cc sequence for merging.
  CondMergingParams
  getJumpConditionMergingParams(Instruction::BinaryOps Opc, const Value *Lhs,
                                const Value *Rhs) const override;

  // Handle Lowering flag assembly outputs.
  SDValue LowerAsmOutputForConstraint(SDValue &Chain, SDValue &Flag,
                                      const SDLoc &DL,
                                      const AsmOperandInfo &Constraint,
                                      SelectionDAG &DAG) const override;

  std::pair<unsigned, const TargetRegisterClass *>
  getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                               StringRef Constraint, MVT VT) const override;
  TargetLowering::ConstraintType
  getConstraintType(StringRef Constraint) const override;
  TargetLowering::ConstraintWeight
    getSingleConstraintMatchWeight(AsmOperandInfo &info,
                                   const char *constraint) const override;
  void LowerAsmOperandForConstraint(SDValue Op, StringRef Constraint,
                                    std::vector<SDValue> &Ops,
                                    SelectionDAG &DAG) const override;

  InlineAsm::ConstraintCode
  getInlineAsmMemConstraint(StringRef ConstraintCode) const override {
    if (ConstraintCode.size() == 1) {
      switch(ConstraintCode[0]) {
      default:
        break;
      case 'o':
        return InlineAsm::ConstraintCode::o;
      case 'Q':
        return InlineAsm::ConstraintCode::Q;
      case 'R':
        return InlineAsm::ConstraintCode::R;
      case 'S':
        return InlineAsm::ConstraintCode::S;
      case 'T':
        return InlineAsm::ConstraintCode::T;
      }
    } else if (ConstraintCode.size() == 2 && ConstraintCode[0] == 'Z') {
      switch (ConstraintCode[1]) {
      default:
        break;
      case 'Q':
        return InlineAsm::ConstraintCode::ZQ;
      case 'R':
        return InlineAsm::ConstraintCode::ZR;
      case 'S':
        return InlineAsm::ConstraintCode::ZS;
      case 'T':
        return InlineAsm::ConstraintCode::ZT;
      }
    }
    return TargetLowering::getInlineAsmMemConstraint(ConstraintCode);
  }

  Register getRegisterByName(const char *RegName, LLT VT,
                             const MachineFunction &MF) const override;

  /// If a physical register, this returns the register that receives the
  /// exception address on entry to an EH pad.
  Register
  getExceptionPointerRegister(const Constant *PersonalityFn) const override;

  /// If a physical register, this returns the register that receives the
  /// exception typeid on entry to a landing pad.
  Register
  getExceptionSelectorRegister(const Constant *PersonalityFn) const override;

  /// Override to support customized stack guard loading.
  bool useLoadStackGuardNode(const Module &M) const override { return true; }
  void insertSSPDeclarations(Module &M) const override {
  }

  MachineBasicBlock *
  EmitInstrWithCustomInserter(MachineInstr &MI,
                              MachineBasicBlock *BB) const override;
  SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;
  void LowerOperationWrapper(SDNode *N, SmallVectorImpl<SDValue> &Results,
                             SelectionDAG &DAG) const override;
  void ReplaceNodeResults(SDNode *N, SmallVectorImpl<SDValue>&Results,
                          SelectionDAG &DAG) const override;
  const MCPhysReg *getScratchRegisters(CallingConv::ID CC) const override;
  bool allowTruncateForTailCall(Type *, Type *) const override;
  bool mayBeEmittedAsTailCall(const CallInst *CI) const override;
  bool splitValueIntoRegisterParts(
      SelectionDAG & DAG, const SDLoc &DL, SDValue Val, SDValue *Parts,
      unsigned NumParts, MVT PartVT, std::optional<CallingConv::ID> CC)
      const override;
  SDValue joinRegisterPartsIntoValue(
      SelectionDAG & DAG, const SDLoc &DL, const SDValue *Parts,
      unsigned NumParts, MVT PartVT, EVT ValueVT,
      std::optional<CallingConv::ID> CC) const override;
  SDValue LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv,
                               bool isVarArg,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               const SDLoc &DL, SelectionDAG &DAG,
                               SmallVectorImpl<SDValue> &InVals) const override;
  SDValue LowerCall(CallLoweringInfo &CLI,
                    SmallVectorImpl<SDValue> &InVals) const override;

  std::pair<SDValue, SDValue>
  makeExternalCall(SDValue Chain, SelectionDAG &DAG, const char *CalleeName,
                   EVT RetVT, ArrayRef<SDValue> Ops, CallingConv::ID CallConv,
                   bool IsSigned, SDLoc DL, bool DoesNotReturn,
                   bool IsReturnValueUsed) const;

  SDValue useLibCall(SelectionDAG &DAG, RTLIB::Libcall LC, MVT VT, SDValue Arg,
                     SDLoc DL, SDValue Chain, bool IsStrict) const;

  bool CanLowerReturn(CallingConv::ID CallConv, MachineFunction &MF,
                      bool isVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      LLVMContext &Context,
                      const Type *RetTy) const override;
  SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      const SmallVectorImpl<SDValue> &OutVals, const SDLoc &DL,
                      SelectionDAG &DAG) const override;
  SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const override;

  /// Determine which of the bits specified in Mask are known to be either
  /// zero or one and return them in the KnownZero/KnownOne bitsets.
  void computeKnownBitsForTargetNode(const SDValue Op,
                                     KnownBits &Known,
                                     const APInt &DemandedElts,
                                     const SelectionDAG &DAG,
                                     unsigned Depth = 0) const override;

  /// Determine the number of bits in the operation that are sign bits.
  unsigned ComputeNumSignBitsForTargetNode(SDValue Op,
                                           const APInt &DemandedElts,
                                           const SelectionDAG &DAG,
                                           unsigned Depth) const override;

  bool isGuaranteedNotToBeUndefOrPoisonForTargetNode(
      SDValue Op, const APInt &DemandedElts, const SelectionDAG &DAG,
      bool PoisonOnly, unsigned Depth) const override;

  ISD::NodeType getExtendForAtomicOps() const override {
    return ISD::ANY_EXTEND;
  }
  ISD::NodeType getExtendForAtomicCmpSwapArg() const override {
    return ISD::ZERO_EXTEND;
  }

  bool supportSwiftError() const override {
    return true;
  }

  unsigned getStackProbeSize(const MachineFunction &MF) const;
  bool hasAndNot(SDValue Y) const override;

private:
  const SystemZSubtarget &Subtarget;

  // Implement LowerOperation for individual opcodes.
  SDValue getVectorCmp(SelectionDAG &DAG, unsigned Opcode,
                       const SDLoc &DL, EVT VT,
                       SDValue CmpOp0, SDValue CmpOp1, SDValue Chain) const;
  SDValue lowerVectorSETCC(SelectionDAG &DAG, const SDLoc &DL,
                           EVT VT, ISD::CondCode CC,
                           SDValue CmpOp0, SDValue CmpOp1,
                           SDValue Chain = SDValue(),
                           bool IsSignaling = false) const;
  SDValue lowerSETCC(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerSTRICT_FSETCC(SDValue Op, SelectionDAG &DAG,
                             bool IsSignaling) const;
  SDValue lowerBR_CC(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerGlobalAddress(GlobalAddressSDNode *Node,
                             SelectionDAG &DAG) const;
  SDValue lowerTLSGetOffset(GlobalAddressSDNode *Node,
                            SelectionDAG &DAG, unsigned Opcode,
                            SDValue GOTOffset) const;
  SDValue lowerThreadPointer(const SDLoc &DL, SelectionDAG &DAG) const;
  SDValue lowerGlobalTLSAddress(GlobalAddressSDNode *Node,
                                SelectionDAG &DAG) const;
  SDValue lowerBlockAddress(BlockAddressSDNode *Node,
                            SelectionDAG &DAG) const;
  SDValue lowerJumpTable(JumpTableSDNode *JT, SelectionDAG &DAG) const;
  SDValue lowerConstantPool(ConstantPoolSDNode *CP, SelectionDAG &DAG) const;
  SDValue lowerFRAMEADDR(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerRETURNADDR(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVASTART(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVASTART_ELF(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVASTART_XPLINK(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVACOPY(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerDYNAMIC_STACKALLOC(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerDYNAMIC_STACKALLOC_ELF(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerDYNAMIC_STACKALLOC_XPLINK(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerGET_DYNAMIC_AREA_OFFSET(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerMULH(SDValue Op, SelectionDAG &DAG, unsigned Opcode) const;
  SDValue lowerSMUL_LOHI(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerUMUL_LOHI(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerSDIVREM(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerUDIVREM(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerXALUO(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerUADDSUBO_CARRY(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerBITCAST(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerOR(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerCTPOP(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVECREDUCE_ADD(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerATOMIC_FENCE(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerATOMIC_LOAD(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerATOMIC_STORE(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerATOMIC_LDST_I128(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerATOMIC_LOAD_OP(SDValue Op, SelectionDAG &DAG,
                              unsigned Opcode) const;
  SDValue lowerATOMIC_LOAD_SUB(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerATOMIC_CMP_SWAP(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerSTACKSAVE(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerSTACKRESTORE(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerPREFETCH(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerINTRINSIC_W_CHAIN(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerINTRINSIC_WO_CHAIN(SDValue Op, SelectionDAG &DAG) const;
  bool isVectorElementLoad(SDValue Op) const;
  SDValue buildVector(SelectionDAG &DAG, const SDLoc &DL, EVT VT,
                      SmallVectorImpl<SDValue> &Elems) const;
  SDValue lowerBUILD_VECTOR(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVECTOR_SHUFFLE(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerSCALAR_TO_VECTOR(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerINSERT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerEXTRACT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerSIGN_EXTEND_VECTOR_INREG(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerZERO_EXTEND_VECTOR_INREG(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerShift(SDValue Op, SelectionDAG &DAG, unsigned ByScalar) const;
  SDValue lowerFSHL(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFSHR(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFP_EXTEND(SDValue Op, SelectionDAG &DAG) const;
  SDValue lower_FP_TO_INT(SDValue Op, SelectionDAG &DAG) const;
  SDValue lower_INT_TO_FP(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerLoadF16(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerStoreF16(SDValue Op, SelectionDAG &DAG) const;

  SDValue lowerIS_FPCLASS(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerGET_ROUNDING(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerREADCYCLECOUNTER(SDValue Op, SelectionDAG &DAG) const;

  bool canTreatAsByteVector(EVT VT) const;
  SDValue combineExtract(const SDLoc &DL, EVT ElemVT, EVT VecVT, SDValue OrigOp,
                         unsigned Index, DAGCombinerInfo &DCI,
                         bool Force) const;
  SDValue combineTruncateExtract(const SDLoc &DL, EVT TruncVT, SDValue Op,
                                 DAGCombinerInfo &DCI) const;
  SDValue combineZERO_EXTEND(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue combineSIGN_EXTEND(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue combineSIGN_EXTEND_INREG(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue combineMERGE(SDNode *N, DAGCombinerInfo &DCI) const;
  bool canLoadStoreByteSwapped(EVT VT) const;
  SDValue combineLOAD(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue combineSTORE(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue combineVECTOR_SHUFFLE(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue combineEXTRACT_VECTOR_ELT(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue combineJOIN_DWORDS(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue combineFP_ROUND(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue combineFP_EXTEND(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue combineINT_TO_FP(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue combineFCOPYSIGN(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue combineBSWAP(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue combineSETCC(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue combineBR_CCMASK(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue combineSELECT_CCMASK(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue combineGET_CCMASK(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue combineShiftToMulAddHigh(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue combineMUL(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue combineIntDIVREM(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue combineINTRINSIC(SDNode *N, DAGCombinerInfo &DCI) const;

  SDValue unwrapAddress(SDValue N) const override;

  // If the last instruction before MBBI in MBB was some form of COMPARE,
  // try to replace it with a COMPARE AND BRANCH just before MBBI.
  // CCMask and Target are the BRC-like operands for the branch.
  // Return true if the change was made.
  bool convertPrevCompareToBranch(MachineBasicBlock *MBB,
                                  MachineBasicBlock::iterator MBBI,
                                  unsigned CCMask,
                                  MachineBasicBlock *Target) const;

  // Implement EmitInstrWithCustomInserter for individual operation types.
  MachineBasicBlock *emitAdjCallStack(MachineInstr &MI,
                                      MachineBasicBlock *BB) const;
  MachineBasicBlock *emitSelect(MachineInstr &MI, MachineBasicBlock *BB) const;
  MachineBasicBlock *emitCondStore(MachineInstr &MI, MachineBasicBlock *BB,
                                   unsigned StoreOpcode, unsigned STOCOpcode,
                                   bool Invert) const;
  MachineBasicBlock *emitICmp128Hi(MachineInstr &MI, MachineBasicBlock *BB,
                                   bool Unsigned) const;
  MachineBasicBlock *emitPair128(MachineInstr &MI,
                                 MachineBasicBlock *MBB) const;
  MachineBasicBlock *emitExt128(MachineInstr &MI, MachineBasicBlock *MBB,
                                bool ClearEven) const;
  MachineBasicBlock *emitAtomicLoadBinary(MachineInstr &MI,
                                          MachineBasicBlock *BB,
                                          unsigned BinOpcode,
                                          bool Invert = false) const;
  MachineBasicBlock *emitAtomicLoadMinMax(MachineInstr &MI,
                                          MachineBasicBlock *MBB,
                                          unsigned CompareOpcode,
                                          unsigned KeepOldMask) const;
  MachineBasicBlock *emitAtomicCmpSwapW(MachineInstr &MI,
                                        MachineBasicBlock *BB) const;
  MachineBasicBlock *emitMemMemWrapper(MachineInstr &MI, MachineBasicBlock *BB,
                                       unsigned Opcode,
                                       bool IsMemset = false) const;
  MachineBasicBlock *emitStringWrapper(MachineInstr &MI, MachineBasicBlock *BB,
                                       unsigned Opcode) const;
  MachineBasicBlock *emitTransactionBegin(MachineInstr &MI,
                                          MachineBasicBlock *MBB,
                                          unsigned Opcode, bool NoFloat) const;
  MachineBasicBlock *emitLoadAndTestCmp0(MachineInstr &MI,
                                         MachineBasicBlock *MBB,
                                         unsigned Opcode) const;
  MachineBasicBlock *emitProbedAlloca(MachineInstr &MI,
                                      MachineBasicBlock *MBB) const;

  SDValue getBackchainAddress(SDValue SP, SelectionDAG &DAG) const;

  MachineMemOperand::Flags
  getTargetMMOFlags(const Instruction &I) const override;
  const TargetRegisterClass *getRepRegClassFor(MVT VT) const override;

private:
  bool isInternal(const Function *Fn) const;
  mutable std::map<const Function *, bool> IsInternalCache;
  void verifyNarrowIntegerArgs_Call(const SmallVectorImpl<ISD::OutputArg> &Outs,
                                    const Function *F, SDValue Callee) const;
  void verifyNarrowIntegerArgs_Ret(const SmallVectorImpl<ISD::OutputArg> &Outs,
                                   const Function *F) const;
  bool
  verifyNarrowIntegerArgs(const SmallVectorImpl<ISD::OutputArg> &Outs) const;

public:
};

struct SystemZVectorConstantInfo {
private:
  APInt IntBits;             // The 128 bits as an integer.
  APInt SplatBits;           // Smallest splat value.
  APInt SplatUndef;          // Bits correspoding to undef operands of the BVN.
  unsigned SplatBitSize = 0;
  bool isFP128 = false;
public:
  unsigned Opcode = 0;
  SmallVector<unsigned, 2> OpVals;
  MVT VecVT;
  SystemZVectorConstantInfo(APInt IntImm);
  SystemZVectorConstantInfo(APFloat FPImm)
      : SystemZVectorConstantInfo(FPImm.bitcastToAPInt()) {
    isFP128 = (&FPImm.getSemantics() == &APFloat::IEEEquad());
  }
  SystemZVectorConstantInfo(BuildVectorSDNode *BVN);
  bool isVectorConstantLegal(const SystemZSubtarget &Subtarget);
};

} // end namespace llvm

#endif
