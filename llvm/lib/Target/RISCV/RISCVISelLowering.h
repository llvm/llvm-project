//===-- RISCVISelLowering.h - RISCV DAG Lowering Interface ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that RISCV uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVISELLOWERING_H
#define LLVM_LIB_TARGET_RISCV_RISCVISELLOWERING_H

#include "RISCV.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {
class RISCVSubtarget;
struct RISCVRegisterInfo;
namespace RISCVISD {
enum NodeType : unsigned {
  FIRST_NUMBER = ISD::BUILTIN_OP_END,
  RET_FLAG,
  URET_FLAG,
  SRET_FLAG,
  MRET_FLAG,
  CALL,
  /// Select with condition operator - This selects between a true value and
  /// a false value (ops #3 and #4) based on the boolean result of comparing
  /// the lhs and rhs (ops #0 and #1) of a conditional expression with the
  /// condition code in op #2, a XLenVT constant from the ISD::CondCode enum.
  /// The lhs and rhs are XLenVT integers. The true and false values can be
  /// integer or floating point.
  SELECT_CC,
  BR_CC,
  BuildPairF64,
  SplitF64,
  TAIL,
  // Multiply high for signedxunsigned.
  MULHSU,
  // RV64I shifts, directly matching the semantics of the named RISC-V
  // instructions.
  SLLW,
  SRAW,
  SRLW,
  // 32-bit operations from RV64M that can't be simply matched with a pattern
  // at instruction selection time. These have undefined behavior for division
  // by 0 or overflow (divw) like their target independent counterparts.
  DIVW,
  DIVUW,
  REMUW,
  // RV64IB rotates, directly matching the semantics of the named RISC-V
  // instructions.
  ROLW,
  RORW,
  // RV64IZbb bit counting instructions directly matching the semantics of the
  // named RISC-V instructions.
  CLZW,
  CTZW,
  // RV64IB/RV32IB funnel shifts, with the semantics of the named RISC-V
  // instructions. Operand order is rs1, rs3, rs2/shamt.
  FSR,
  FSL,
  // RV64IB funnel shifts, with the semantics of the named RISC-V instructions.
  // Operand order is rs1, rs3, rs2/shamt.
  FSRW,
  FSLW,
  // FPR<->GPR transfer operations when the FPR is smaller than XLEN, needed as
  // XLEN is the only legal integer width.
  //
  // FMV_H_X matches the semantics of the FMV.H.X.
  // FMV_X_ANYEXTH is similar to FMV.X.H but has an any-extended result.
  // FMV_W_X_RV64 matches the semantics of the FMV.W.X.
  // FMV_X_ANYEXTW_RV64 is similar to FMV.X.W but has an any-extended result.
  //
  // This is a more convenient semantic for producing dagcombines that remove
  // unnecessary GPR->FPR->GPR moves.
  FMV_H_X,
  FMV_X_ANYEXTH,
  FMV_W_X_RV64,
  FMV_X_ANYEXTW_RV64,
  // FP to XLen int conversions. Corresponds to fcvt.l(u).s/d/h on RV64 and
  // fcvt.w(u).s/d/h on RV32. Unlike FP_TO_S/UINT these saturate out of
  // range inputs. These are used for FP_TO_S/UINT_SAT lowering. Rounding mode
  // is passed as a TargetConstant operand using the RISCVFPRndMode enum.
  FCVT_X,
  FCVT_XU,
  // FP to 32 bit int conversions for RV64. These are used to keep track of the
  // result being sign extended to 64 bit. These saturate out of range inputs.
  // Used for FP_TO_S/UINT and FP_TO_S/UINT_SAT lowering. Rounding mode
  // is passed as a TargetConstant operand using the RISCVFPRndMode enum.
  FCVT_W_RV64,
  FCVT_WU_RV64,
  // READ_CYCLE_WIDE - A read of the 64-bit cycle CSR on a 32-bit target
  // (returns (Lo, Hi)). It takes a chain operand.
  READ_CYCLE_WIDE,
  // Generalized Reverse and Generalized Or-Combine - directly matching the
  // semantics of the named RISC-V instructions. Lowered as custom nodes as
  // TableGen chokes when faced with commutative permutations in deeply-nested
  // DAGs. Each node takes an input operand and a control operand and outputs a
  // bit-manipulated version of input. All operands are i32 or XLenVT.
  GREV,
  GREVW,
  GORC,
  GORCW,
  SHFL,
  SHFLW,
  UNSHFL,
  UNSHFLW,
  // Bit Compress/Decompress implement the generic bit extract and bit deposit
  // functions. This operation is also referred to as bit gather/scatter, bit
  // pack/unpack, parallel extract/deposit, compress/expand, or right
  // compress/right expand.
  BCOMPRESS,
  BCOMPRESSW,
  BDECOMPRESS,
  BDECOMPRESSW,
  // The bit field place (bfp) instruction places up to XLEN/2 LSB bits from rs2
  // into the value in rs1. The upper bits of rs2 control the length of the bit
  // field and target position. The layout of rs2 is chosen in a way that makes
  // it possible to construct rs2 easily using pack[h] instructions and/or
  // andi/lui.
  BFP,
  BFPW,
  // Vector Extension
  // VMV_V_X_VL matches the semantics of vmv.v.x but includes an extra operand
  // for the VL value to be used for the operation.
  VMV_V_X_VL,
  // VFMV_V_F_VL matches the semantics of vfmv.v.f but includes an extra operand
  // for the VL value to be used for the operation.
  VFMV_V_F_VL,
  // VMV_X_S matches the semantics of vmv.x.s. The result is always XLenVT sign
  // extended from the vector element size.
  VMV_X_S,
  // VMV_S_X_VL matches the semantics of vmv.s.x. It carries a VL operand.
  VMV_S_X_VL,
  // VFMV_S_F_VL matches the semantics of vfmv.s.f. It carries a VL operand.
  VFMV_S_F_VL,
  // Splats an i64 scalar to a vector type (with element type i64) where the
  // scalar is a sign-extended i32.
  SPLAT_VECTOR_I64,
  // Splats an 64-bit value that has been split into two i32 parts. This is
  // expanded late to two scalar stores and a stride 0 vector load.
  SPLAT_VECTOR_SPLIT_I64_VL,
  // Read VLENB CSR
  READ_VLENB,
  // Truncates a RVV integer vector by one power-of-two. Carries both an extra
  // mask and VL operand.
  TRUNCATE_VECTOR_VL,
  // Matches the semantics of vslideup/vslidedown. The first operand is the
  // pass-thru operand, the second is the source vector, the third is the
  // XLenVT index (either constant or non-constant), the fourth is the mask
  // and the fifth the VL.
  VSLIDEUP_VL,
  VSLIDEDOWN_VL,
  // Matches the semantics of vslide1up/slide1down. The first operand is the
  // source vector, the second is the XLenVT scalar value. The third and fourth
  // operands are the mask and VL operands.
  VSLIDE1UP_VL,
  VSLIDE1DOWN_VL,
  // Matches the semantics of the vid.v instruction, with a mask and VL
  // operand.
  VID_VL,
  // Matches the semantics of the vfcnvt.rod function (Convert double-width
  // float to single-width float, rounding towards odd). Takes a double-width
  // float vector and produces a single-width float vector. Also has a mask and
  // VL operand.
  VFNCVT_ROD_VL,
  // These nodes match the semantics of the corresponding RVV vector reduction
  // instructions. They produce a vector result which is the reduction
  // performed over the second vector operand plus the first element of the
  // third vector operand. The first operand is the pass-thru operand. The
  // second operand is an unconstrained vector type, and the result, first, and
  // third operand's types are expected to be the corresponding full-width
  // LMUL=1 type for the second operand:
  //   nxv8i8 = vecreduce_add nxv8i8, nxv32i8, nxv8i8
  //   nxv2i32 = vecreduce_add nxv2i32, nxv8i32, nxv2i32
  // The different in types does introduce extra vsetvli instructions but
  // similarly it reduces the number of registers consumed per reduction.
  // Also has a mask and VL operand.
  VECREDUCE_ADD_VL,
  VECREDUCE_UMAX_VL,
  VECREDUCE_SMAX_VL,
  VECREDUCE_UMIN_VL,
  VECREDUCE_SMIN_VL,
  VECREDUCE_AND_VL,
  VECREDUCE_OR_VL,
  VECREDUCE_XOR_VL,
  VECREDUCE_FADD_VL,
  VECREDUCE_SEQ_FADD_VL,
  VECREDUCE_FMIN_VL,
  VECREDUCE_FMAX_VL,

  // Vector binary and unary ops with a mask as a third operand, and VL as a
  // fourth operand.
  // FIXME: Can we replace these with ISD::VP_*?
  ADD_VL,
  AND_VL,
  MUL_VL,
  OR_VL,
  SDIV_VL,
  SHL_VL,
  SREM_VL,
  SRA_VL,
  SRL_VL,
  SUB_VL,
  UDIV_VL,
  UREM_VL,
  XOR_VL,

  SADDSAT_VL,
  UADDSAT_VL,
  SSUBSAT_VL,
  USUBSAT_VL,

  FADD_VL,
  FSUB_VL,
  FMUL_VL,
  FDIV_VL,
  FNEG_VL,
  FABS_VL,
  FSQRT_VL,
  FMA_VL,
  FCOPYSIGN_VL,
  SMIN_VL,
  SMAX_VL,
  UMIN_VL,
  UMAX_VL,
  FMINNUM_VL,
  FMAXNUM_VL,
  MULHS_VL,
  MULHU_VL,
  FP_TO_SINT_VL,
  FP_TO_UINT_VL,
  SINT_TO_FP_VL,
  UINT_TO_FP_VL,
  FP_ROUND_VL,
  FP_EXTEND_VL,

  // Widening instructions
  VWMUL_VL,
  VWMULU_VL,
  VWADDU_VL,

  // Vector compare producing a mask. Fourth operand is input mask. Fifth
  // operand is VL.
  SETCC_VL,

  // Vector select with an additional VL operand. This operation is unmasked.
  VSELECT_VL,
  // Vector select with operand #2 (the value when the condition is false) tied
  // to the destination and an additional VL operand. This operation is
  // unmasked.
  VP_MERGE_VL,

  // Mask binary operators.
  VMAND_VL,
  VMOR_VL,
  VMXOR_VL,

  // Set mask vector to all zeros or ones.
  VMCLR_VL,
  VMSET_VL,

  // Matches the semantics of vrgather.vx and vrgather.vv with an extra operand
  // for VL.
  VRGATHER_VX_VL,
  VRGATHER_VV_VL,
  VRGATHEREI16_VV_VL,

  // Vector sign/zero extend with additional mask & VL operands.
  VSEXT_VL,
  VZEXT_VL,

  //  vcpop.m with additional mask and VL operands.
  VCPOP_VL,

  // Reads value of CSR.
  // The first operand is a chain pointer. The second specifies address of the
  // required CSR. Two results are produced, the read value and the new chain
  // pointer.
  READ_CSR,
  // Write value to CSR.
  // The first operand is a chain pointer, the second specifies address of the
  // required CSR and the third is the value to write. The result is the new
  // chain pointer.
  WRITE_CSR,
  // Read and write value of CSR.
  // The first operand is a chain pointer, the second specifies address of the
  // required CSR and the third is the value to write. Two results are produced,
  // the value read before the modification and the new chain pointer.
  SWAP_CSR,

  // FP to 32 bit int conversions for RV64. These are used to keep track of the
  // result being sign extended to 64 bit. These saturate out of range inputs.
  STRICT_FCVT_W_RV64 = ISD::FIRST_TARGET_STRICTFP_OPCODE,
  STRICT_FCVT_WU_RV64,

  // Memory opcodes start here.
  VLE_VL = ISD::FIRST_TARGET_MEMORY_OPCODE,
  VSE_VL,

  // WARNING: Do not add anything in the end unless you want the node to
  // have memop! In fact, starting from FIRST_TARGET_MEMORY_OPCODE all
  // opcodes will be thought as target memory ops!
};
} // namespace RISCVISD

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
  bool isZExtFree(SDValue Val, EVT VT2) const override;
  bool isSExtCheaperThanZExt(EVT SrcVT, EVT DstVT) const override;
  bool isCheapToSpeculateCttz() const override;
  bool isCheapToSpeculateCtlz() const override;
  bool hasAndNotCompare(SDValue Y) const override;
  bool shouldSinkOperands(Instruction *I,
                          SmallVectorImpl<Use *> &Ops) const override;
  bool isFPImmLegal(const APFloat &Imm, EVT VT,
                    bool ForCodeSize) const override;

  bool softPromoteHalfType() const override { return true; }

  /// Return the register type for a given MVT, ensuring vectors are treated
  /// as a series of gpr sized integers.
  MVT getRegisterTypeForCallingConv(LLVMContext &Context, CallingConv::ID CC,
                                    EVT VT) const override;

  /// Return the number of registers for a given MVT, ensuring vectors are
  /// treated as a series of gpr sized integers.
  unsigned getNumRegistersForCallingConv(LLVMContext &Context,
                                         CallingConv::ID CC,
                                         EVT VT) const override;

  /// Return true if the given shuffle mask can be codegen'd directly, or if it
  /// should be stack expanded.
  bool isShuffleMaskLegal(ArrayRef<int> M, EVT VT) const override;

  bool hasBitPreservingFPLogic(EVT VT) const override;
  bool
  shouldExpandBuildVectorWithShuffles(EVT VT,
                                      unsigned DefinedValues) const override;

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

  // This method returns the name of a target specific DAG node.
  const char *getTargetNodeName(unsigned Opcode) const override;

  ConstraintType getConstraintType(StringRef Constraint) const override;

  unsigned getInlineAsmMemConstraint(StringRef ConstraintCode) const override;

  std::pair<unsigned, const TargetRegisterClass *>
  getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                               StringRef Constraint, MVT VT) const override;

  void LowerAsmOperandForConstraint(SDValue Op, std::string &Constraint,
                                    std::vector<SDValue> &Ops,
                                    SelectionDAG &DAG) const override;

  MachineBasicBlock *
  EmitInstrWithCustomInserter(MachineInstr &MI,
                              MachineBasicBlock *BB) const override;

  void AdjustInstrPostInstrSelection(MachineInstr &MI,
                                     SDNode *Node) const override;

  EVT getSetCCResultType(const DataLayout &DL, LLVMContext &Context,
                         EVT VT) const override;

  bool convertSetCCLogicToBitwiseLogic(EVT VT) const override {
    return VT.isScalarInteger();
  }
  bool convertSelectOfConstantsToMath(EVT VT) const override { return true; }

  bool shouldInsertFencesForAtomic(const Instruction *I) const override {
    return isa<LoadInst>(I) || isa<StoreInst>(I);
  }
  Instruction *emitLeadingFence(IRBuilderBase &Builder, Instruction *Inst,
                                AtomicOrdering Ord) const override;
  Instruction *emitTrailingFence(IRBuilderBase &Builder, Instruction *Inst,
                                 AtomicOrdering Ord) const override;

  bool isFMAFasterThanFMulAndFAdd(const MachineFunction &MF,
                                  EVT VT) const override;

  ISD::NodeType getExtendForAtomicOps() const override {
    return ISD::SIGN_EXTEND;
  }

  ISD::NodeType getExtendForAtomicCmpSwapArg() const override {
    return ISD::SIGN_EXTEND;
  }

  bool shouldExpandShift(SelectionDAG &DAG, SDNode *N) const override {
    if (DAG.getMachineFunction().getFunction().hasMinSize())
      return false;
    return true;
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
  bool shouldSignExtendTypeInLibCall(EVT Type, bool IsSigned) const override;

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
                      LLVMContext &Context) const override;
  SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      const SmallVectorImpl<SDValue> &OutVals, const SDLoc &DL,
                      SelectionDAG &DAG) const override;
  SDValue LowerCall(TargetLowering::CallLoweringInfo &CLI,
                    SmallVectorImpl<SDValue> &InVals) const override;
  template <class NodeTy>
  SDValue getAddr(NodeTy *N, SelectionDAG &DAG, bool IsLocal = true) const;

  bool shouldConvertConstantLoadToIntImm(const APInt &Imm,
                                         Type *Ty) const override {
    return true;
  }
  bool mayBeEmittedAsTailCall(const CallInst *CI) const override;
  bool shouldConsiderGEPOffsetSplit() const override { return true; }

  bool decomposeMulByConstant(LLVMContext &Context, EVT VT,
                              SDValue C) const override;

  bool isMulAddWithConstProfitable(const SDValue &AddNode,
                                   const SDValue &ConstNode) const override;

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
      bool *Fast = nullptr) const override;

  bool splitValueIntoRegisterParts(SelectionDAG &DAG, const SDLoc &DL,
                                   SDValue Val, SDValue *Parts,
                                   unsigned NumParts, MVT PartVT,
                                   Optional<CallingConv::ID> CC) const override;

  SDValue
  joinRegisterPartsIntoValue(SelectionDAG &DAG, const SDLoc &DL,
                             const SDValue *Parts, unsigned NumParts,
                             MVT PartVT, EVT ValueVT,
                             Optional<CallingConv::ID> CC) const override;

  static RISCVII::VLMUL getLMUL(MVT VT);
  static unsigned getRegClassIDForLMUL(RISCVII::VLMUL LMul);
  static unsigned getSubregIndexByMVT(MVT VT, unsigned Index);
  static unsigned getRegClassIDForVecVT(MVT VT);
  static std::pair<unsigned, unsigned>
  decomposeSubvectorInsertExtractToSubRegs(MVT VecVT, MVT SubVecVT,
                                           unsigned InsertExtractIdx,
                                           const RISCVRegisterInfo *TRI);
  MVT getContainerForFixedLengthVector(MVT VT) const;

  bool shouldRemoveExtendFromGSIndex(EVT VT) const override;

  bool isLegalElementTypeForRVV(Type *ScalarTy) const;

  bool shouldConvertFpToSat(unsigned Op, EVT FPVT, EVT VT) const override;

  SDValue BuildSDIVPow2(SDNode *N, const APInt &Divisor, SelectionDAG &DAG,
                        SmallVectorImpl<SDNode *> &Created) const override;

  unsigned getJumpTableEncoding() const override;

  const MCExpr *LowerCustomJumpTableEntry(const MachineJumpTableInfo *MJTI,
                                          const MachineBasicBlock *MBB,
                                          unsigned uid,
                                          MCContext &Ctx) const override;

private:
  /// RISCVCCAssignFn - This target-specific function extends the default
  /// CCValAssign with additional information used to lower RISC-V calling
  /// conventions.
  typedef bool RISCVCCAssignFn(const DataLayout &DL, RISCVABI::ABI,
                               unsigned ValNo, MVT ValVT, MVT LocVT,
                               CCValAssign::LocInfo LocInfo,
                               ISD::ArgFlagsTy ArgFlags, CCState &State,
                               bool IsFixed, bool IsRet, Type *OrigTy,
                               const RISCVTargetLowering &TLI,
                               Optional<unsigned> FirstMaskArgument);

  void analyzeInputArgs(MachineFunction &MF, CCState &CCInfo,
                        const SmallVectorImpl<ISD::InputArg> &Ins, bool IsRet,
                        RISCVCCAssignFn Fn) const;
  void analyzeOutputArgs(MachineFunction &MF, CCState &CCInfo,
                         const SmallVectorImpl<ISD::OutputArg> &Outs,
                         bool IsRet, CallLoweringInfo *CLI,
                         RISCVCCAssignFn Fn) const;

  SDValue getStaticTLSAddr(GlobalAddressSDNode *N, SelectionDAG &DAG,
                           bool UseGOT) const;
  SDValue getDynamicTLSAddr(GlobalAddressSDNode *N, SelectionDAG &DAG) const;

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
  SDValue lowerVectorMaskTrunc(SDValue Op, SelectionDAG &DAG) const;
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
  SDValue lowerSTEP_VECTOR(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVECTOR_REVERSE(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerABS(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerMaskedLoad(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerMaskedStore(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFixedLengthVectorFCOPYSIGNToRVV(SDValue Op,
                                               SelectionDAG &DAG) const;
  SDValue lowerMaskedGather(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerMaskedScatter(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFixedLengthVectorLoadToRVV(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFixedLengthVectorStoreToRVV(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFixedLengthVectorSetccToRVV(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFixedLengthVectorLogicOpToRVV(SDValue Op, SelectionDAG &DAG,
                                             unsigned MaskOpc,
                                             unsigned VecOpc) const;
  SDValue lowerFixedLengthVectorShiftToRVV(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFixedLengthVectorSelectToRVV(SDValue Op,
                                            SelectionDAG &DAG) const;
  SDValue lowerToScalableOp(SDValue Op, SelectionDAG &DAG, unsigned NewOpc,
                            bool HasMask = true) const;
  SDValue lowerVPOp(SDValue Op, SelectionDAG &DAG, unsigned RISCVISDOpc) const;
  SDValue lowerLogicVPOp(SDValue Op, SelectionDAG &DAG, unsigned MaskOpc,
                         unsigned VecOpc) const;
  SDValue lowerFixedLengthVectorExtendToRVV(SDValue Op, SelectionDAG &DAG,
                                            unsigned ExtendOpc) const;
  SDValue lowerGET_ROUNDING(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerSET_ROUNDING(SDValue Op, SelectionDAG &DAG) const;

  SDValue expandUnalignedRVVLoad(SDValue Op, SelectionDAG &DAG) const;
  SDValue expandUnalignedRVVStore(SDValue Op, SelectionDAG &DAG) const;

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
  /// RISCV doesn't have flags so it's better to perform the and/or in a GPR.
  bool shouldNormalizeToSelectSequence(LLVMContext &, EVT) const override {
    return false;
  };
};

namespace RISCV {
// We use 64 bits as the known part in the scalable vector types.
static constexpr unsigned RVVBitsPerBlock = 64;
} // namespace RISCV

namespace RISCVVIntrinsicsTable {

struct RISCVVIntrinsicInfo {
  unsigned IntrinsicID;
  uint8_t SplatOperand;
  uint8_t VLOperand;
  bool hasSplatOperand() const {
    // 0xF is not valid. See NoSplatOperand in IntrinsicsRISCV.td.
    return SplatOperand != 0xF;
  }
  bool hasVLOperand() const {
    // 0x1F is not valid. See NoVLOperand in IntrinsicsRISCV.td.
    return VLOperand != 0x1F;
  }
};

using namespace RISCV;

#define GET_RISCVVIntrinsicsTable_DECL
#include "RISCVGenSearchableTables.inc"

} // end namespace RISCVVIntrinsicsTable

} // end namespace llvm

#endif
