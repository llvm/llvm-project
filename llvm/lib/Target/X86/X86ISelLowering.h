//===-- X86ISelLowering.h - X86 DAG Lowering Interface ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that X86 uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_X86_X86ISELLOWERING_H
#define LLVM_LIB_TARGET_X86_X86ISELLOWERING_H

#include "X86SelectionDAGInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {
  class X86Subtarget;
  class X86TargetMachine;

  namespace X86 {
    /// Current rounding mode is represented in bits 11:10 of FPSR. These
    /// values are same as corresponding constants for rounding mode used
    /// in glibc.
  enum RoundingMode {
    rmInvalid = -1,         // For handle Invalid rounding mode
    rmToNearest = 0,        // FE_TONEAREST
    rmDownward = 1 << 10,   // FE_DOWNWARD
    rmUpward = 2 << 10,     // FE_UPWARD
    rmTowardZero = 3 << 10, // FE_TOWARDZERO
    rmMask = 3 << 10        // Bit mask selecting rounding mode
  };
  }

  /// Define some predicates that are used for node matching.
  namespace X86 {
    /// Returns true if Elt is a constant zero or floating point constant +0.0.
    bool isZeroNode(SDValue Elt);

    /// Returns true of the given offset can be
    /// fit into displacement field of the instruction.
    bool isOffsetSuitableForCodeModel(int64_t Offset, CodeModel::Model M,
                                      bool hasSymbolicDisplacement);

    /// Determines whether the callee is required to pop its
    /// own arguments. Callee pop is necessary to support tail calls.
    bool isCalleePop(CallingConv::ID CallingConv,
                     bool is64Bit, bool IsVarArg, bool GuaranteeTCO);

    /// If Op is a constant whose elements are all the same constant or
    /// undefined, return true and return the constant value in \p SplatVal.
    /// If we have undef bits that don't cover an entire element, we treat these
    /// as zero if AllowPartialUndefs is set, else we fail and return false.
    bool isConstantSplat(SDValue Op, APInt &SplatVal,
                         bool AllowPartialUndefs = true);

    /// Check if Op is a load operation that could be folded into some other x86
    /// instruction as a memory operand. Example: vpaddd (%rdi), %xmm0, %xmm0.
    bool mayFoldLoad(SDValue Op, const X86Subtarget &Subtarget,
                     bool AssumeSingleUse = false,
                     bool IgnoreAlignment = false);

    /// Check if Op is a load operation that could be folded into a vector splat
    /// instruction as a memory operand. Example: vbroadcastss 16(%rdi), %xmm2.
    bool mayFoldLoadIntoBroadcastFromMem(SDValue Op, MVT EltVT,
                                         const X86Subtarget &Subtarget,
                                         bool AssumeSingleUse = false);

    /// Check if Op is a value that could be used to fold a store into some
    /// other x86 instruction as a memory operand. Ex: pextrb $0, %xmm0, (%rdi).
    bool mayFoldIntoStore(SDValue Op);

    /// Check if Op is an operation that could be folded into a zero extend x86
    /// instruction.
    bool mayFoldIntoZeroExtend(SDValue Op);

    /// True if the target supports the extended frame for async Swift
    /// functions.
    bool isExtendedSwiftAsyncFrameSupported(const X86Subtarget &Subtarget,
                                            const MachineFunction &MF);

    /// Convert LLVM rounding mode to X86 rounding mode.
    int getRoundingModeX86(unsigned RM);

  } // end namespace X86

  //===--------------------------------------------------------------------===//
  //  X86 Implementation of the TargetLowering interface
  class X86TargetLowering final : public TargetLowering {
    // Copying needed for an outgoing byval argument.
    enum ByValCopyKind {
      // Argument is already in the correct location, no copy needed.
      NoCopy,
      // Argument value is currently in the local stack frame, needs copying to
      // outgoing arguemnt area.
      CopyOnce,
      // Argument value is currently in the outgoing argument area, but not at
      // the correct offset, so needs copying via a temporary in local stack
      // space.
      CopyViaTemp,
    };

  public:
    explicit X86TargetLowering(const X86TargetMachine &TM,
                               const X86Subtarget &STI);

    unsigned getJumpTableEncoding() const override;
    bool useSoftFloat() const override;

    void markLibCallAttributes(MachineFunction *MF, unsigned CC,
                               ArgListTy &Args) const override;

    MVT getScalarShiftAmountTy(const DataLayout &, EVT VT) const override {
      return MVT::i8;
    }

    const MCExpr *
    LowerCustomJumpTableEntry(const MachineJumpTableInfo *MJTI,
                              const MachineBasicBlock *MBB, unsigned uid,
                              MCContext &Ctx) const override;

    /// Returns relocation base for the given PIC jumptable.
    SDValue getPICJumpTableRelocBase(SDValue Table,
                                     SelectionDAG &DAG) const override;
    const MCExpr *
    getPICJumpTableRelocBaseExpr(const MachineFunction *MF,
                                 unsigned JTI, MCContext &Ctx) const override;

    /// Return the desired alignment for ByVal aggregate
    /// function arguments in the caller parameter area. For X86, aggregates
    /// that contains are placed at 16-byte boundaries while the rest are at
    /// 4-byte boundaries.
    Align getByValTypeAlignment(Type *Ty, const DataLayout &DL) const override;

    EVT getOptimalMemOpType(LLVMContext &Context, const MemOp &Op,
                            const AttributeList &FuncAttributes) const override;

    /// Returns true if it's safe to use load / store of the
    /// specified type to expand memcpy / memset inline. This is mostly true
    /// for all types except for some special cases. For example, on X86
    /// targets without SSE2 f64 load / store are done with fldl / fstpl which
    /// also does type conversion. Note the specified type doesn't have to be
    /// legal as the hook is used before type legalization.
    bool isSafeMemOpType(MVT VT) const override;

    bool isMemoryAccessFast(EVT VT, Align Alignment) const;

    /// Returns true if the target allows unaligned memory accesses of the
    /// specified type. Returns whether it is "fast" in the last argument.
    bool allowsMisalignedMemoryAccesses(EVT VT, unsigned AS, Align Alignment,
                                        MachineMemOperand::Flags Flags,
                                        unsigned *Fast) const override;

    /// This function returns true if the memory access is aligned or if the
    /// target allows this specific unaligned memory access. If the access is
    /// allowed, the optional final parameter returns a relative speed of the
    /// access (as defined by the target).
    bool allowsMemoryAccess(
        LLVMContext &Context, const DataLayout &DL, EVT VT, unsigned AddrSpace,
        Align Alignment,
        MachineMemOperand::Flags Flags = MachineMemOperand::MONone,
        unsigned *Fast = nullptr) const override;

    bool allowsMemoryAccess(LLVMContext &Context, const DataLayout &DL, EVT VT,
                            const MachineMemOperand &MMO,
                            unsigned *Fast) const {
      return allowsMemoryAccess(Context, DL, VT, MMO.getAddrSpace(),
                                MMO.getAlign(), MMO.getFlags(), Fast);
    }

    /// Provide custom lowering hooks for some operations.
    ///
    SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;

    /// Replace the results of node with an illegal result
    /// type with new values built out of custom code.
    ///
    void ReplaceNodeResults(SDNode *N, SmallVectorImpl<SDValue>&Results,
                            SelectionDAG &DAG) const override;

    SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const override;

    bool preferABDSToABSWithNSW(EVT VT) const override;

    bool preferSextInRegOfTruncate(EVT TruncVT, EVT VT,
                                   EVT ExtVT) const override;

    bool isXAndYEqZeroPreferableToXAndYEqY(ISD::CondCode Cond,
                                           EVT VT) const override;

    /// Return true if the target has native support for
    /// the specified value type and it is 'desirable' to use the type for the
    /// given node type. e.g. On x86 i16 is legal, but undesirable since i16
    /// instruction encodings are longer and some i16 instructions are slow.
    bool isTypeDesirableForOp(unsigned Opc, EVT VT) const override;

    /// Return true if the target has native support for the
    /// specified value type and it is 'desirable' to use the type. e.g. On x86
    /// i16 is legal, but undesirable since i16 instruction encodings are longer
    /// and some i16 instructions are slow.
    bool IsDesirableToPromoteOp(SDValue Op, EVT &PVT) const override;

    /// Return prefered fold type, Abs if this is a vector, AddAnd if its an
    /// integer, None otherwise.
    TargetLowering::AndOrSETCCFoldKind
    isDesirableToCombineLogicOpOfSETCC(const SDNode *LogicOp,
                                       const SDNode *SETCC0,
                                       const SDNode *SETCC1) const override;

    /// Return the newly negated expression if the cost is not expensive and
    /// set the cost in \p Cost to indicate that if it is cheaper or neutral to
    /// do the negation.
    SDValue getNegatedExpression(SDValue Op, SelectionDAG &DAG,
                                 bool LegalOperations, bool ForCodeSize,
                                 NegatibleCost &Cost,
                                 unsigned Depth) const override;

    MachineBasicBlock *
    EmitInstrWithCustomInserter(MachineInstr &MI,
                                MachineBasicBlock *MBB) const override;

    /// Do not merge vector stores after legalization because that may conflict
    /// with x86-specific store splitting optimizations.
    bool mergeStoresAfterLegalization(EVT MemVT) const override {
      return !MemVT.isVector();
    }

    bool canMergeStoresTo(unsigned AddressSpace, EVT MemVT,
                          const MachineFunction &MF) const override;

    bool isCheapToSpeculateCttz(Type *Ty) const override;

    bool isCheapToSpeculateCtlz(Type *Ty) const override;

    bool isCtlzFast() const override;

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
      // store buffer). Since the benefit is more blurred so we leave
      // such pair out until we get testcase to prove it is a win.
      return false;
    }

    bool isMaskAndCmp0FoldingBeneficial(const Instruction &AndI) const override;

    bool hasAndNotCompare(SDValue Y) const override;

    bool hasAndNot(SDValue Y) const override;

    bool hasBitTest(SDValue X, SDValue Y) const override;

    bool shouldProduceAndByConstByHoistingConstFromShiftsLHSOfAnd(
        SDValue X, ConstantSDNode *XC, ConstantSDNode *CC, SDValue Y,
        unsigned OldShiftOpcode, unsigned NewShiftOpcode,
        SelectionDAG &DAG) const override;

    unsigned preferedOpcodeForCmpEqPiecesOfOperand(
        EVT VT, unsigned ShiftOpc, bool MayTransformRotate,
        const APInt &ShiftOrRotateAmt,
        const std::optional<APInt> &AndMask) const override;

    bool preferScalarizeSplat(SDNode *N) const override;

    CondMergingParams
    getJumpConditionMergingParams(Instruction::BinaryOps Opc, const Value *Lhs,
                                  const Value *Rhs) const override;

    bool shouldFoldConstantShiftPairToMask(const SDNode *N) const override;

    bool shouldFoldMaskToVariableShiftPair(SDValue Y) const override;

    bool
    shouldTransformSignedTruncationCheck(EVT XVT,
                                         unsigned KeptBits) const override {
      // For vectors, we don't have a preference..
      if (XVT.isVector())
        return false;

      auto VTIsOk = [](EVT VT) -> bool {
        return VT == MVT::i8 || VT == MVT::i16 || VT == MVT::i32 ||
               VT == MVT::i64;
      };

      // We are ok with KeptBitsVT being byte/word/dword, what MOVS supports.
      // XVT will be larger than KeptBitsVT.
      MVT KeptBitsVT = MVT::getIntegerVT(KeptBits);
      return VTIsOk(XVT) && VTIsOk(KeptBitsVT);
    }

    ShiftLegalizationStrategy
    preferredShiftLegalizationStrategy(SelectionDAG &DAG, SDNode *N,
                                       unsigned ExpansionFactor) const override;

    bool shouldSplatInsEltVarIndex(EVT VT) const override;

    bool shouldConvertFpToSat(unsigned Op, EVT FPVT, EVT VT) const override {
      // Converting to sat variants holds little benefit on X86 as we will just
      // need to saturate the value back using fp arithmatic.
      return Op != ISD::FP_TO_UINT_SAT && isOperationLegalOrCustom(Op, VT);
    }

    bool convertSetCCLogicToBitwiseLogic(EVT VT) const override {
      return VT.isScalarInteger();
    }

    /// Vector-sized comparisons are fast using PCMPEQ + PMOVMSK or PTEST.
    MVT hasFastEqualityCompare(unsigned NumBits) const override;

    /// Return the value type to use for ISD::SETCC.
    EVT getSetCCResultType(const DataLayout &DL, LLVMContext &Context,
                           EVT VT) const override;

    bool targetShrinkDemandedConstant(SDValue Op, const APInt &DemandedBits,
                                      const APInt &DemandedElts,
                                      TargetLoweringOpt &TLO) const override;

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

    bool SimplifyDemandedVectorEltsForTargetNode(SDValue Op,
                                                 const APInt &DemandedElts,
                                                 APInt &KnownUndef,
                                                 APInt &KnownZero,
                                                 TargetLoweringOpt &TLO,
                                                 unsigned Depth) const override;

    bool SimplifyDemandedVectorEltsForTargetShuffle(SDValue Op,
                                                    const APInt &DemandedElts,
                                                    unsigned MaskIndex,
                                                    TargetLoweringOpt &TLO,
                                                    unsigned Depth) const;

    bool SimplifyDemandedBitsForTargetNode(SDValue Op,
                                           const APInt &DemandedBits,
                                           const APInt &DemandedElts,
                                           KnownBits &Known,
                                           TargetLoweringOpt &TLO,
                                           unsigned Depth) const override;

    SDValue SimplifyMultipleUseDemandedBitsForTargetNode(
        SDValue Op, const APInt &DemandedBits, const APInt &DemandedElts,
        SelectionDAG &DAG, unsigned Depth) const override;

    bool isGuaranteedNotToBeUndefOrPoisonForTargetNode(
        SDValue Op, const APInt &DemandedElts, const SelectionDAG &DAG,
        bool PoisonOnly, unsigned Depth) const override;

    bool canCreateUndefOrPoisonForTargetNode(
        SDValue Op, const APInt &DemandedElts, const SelectionDAG &DAG,
        bool PoisonOnly, bool ConsiderFlags, unsigned Depth) const override;

    bool isSplatValueForTargetNode(SDValue Op, const APInt &DemandedElts,
                                   APInt &UndefElts, const SelectionDAG &DAG,
                                   unsigned Depth) const override;

    bool isTargetCanonicalConstantNode(SDValue Op) const override {
      // Peek through bitcasts/extracts/inserts to see if we have a vector
      // load/broadcast from memory.
      while (Op.getOpcode() == ISD::BITCAST ||
             Op.getOpcode() == ISD::EXTRACT_SUBVECTOR ||
             (Op.getOpcode() == ISD::INSERT_SUBVECTOR &&
              Op.getOperand(0).isUndef()))
        Op = Op.getOperand(Op.getOpcode() == ISD::INSERT_SUBVECTOR ? 1 : 0);

      return Op.getOpcode() == X86ISD::VBROADCAST_LOAD ||
             Op.getOpcode() == X86ISD::SUBV_BROADCAST_LOAD ||
             (Op.getOpcode() == ISD::LOAD &&
              getTargetConstantFromLoad(cast<LoadSDNode>(Op))) ||
             TargetLowering::isTargetCanonicalConstantNode(Op);
    }

    bool isTargetCanonicalSelect(SDNode *N) const override;

    const Constant *getTargetConstantFromLoad(LoadSDNode *LD) const override;

    SDValue unwrapAddress(SDValue N) const override;

    SDValue getReturnAddressFrameIndex(SelectionDAG &DAG) const;

    ConstraintType getConstraintType(StringRef Constraint) const override;

    /// Examine constraint string and operand type and determine a weight value.
    /// The operand object must already have been set up with the operand type.
    ConstraintWeight
      getSingleConstraintMatchWeight(AsmOperandInfo &Info,
                                     const char *Constraint) const override;

    const char *LowerXConstraint(EVT ConstraintVT) const override;

    /// Lower the specified operand into the Ops vector. If it is invalid, don't
    /// add anything to Ops. If hasMemory is true it means one of the asm
    /// constraint of the inline asm instruction being processed is 'm'.
    void LowerAsmOperandForConstraint(SDValue Op, StringRef Constraint,
                                      std::vector<SDValue> &Ops,
                                      SelectionDAG &DAG) const override;

    InlineAsm::ConstraintCode
    getInlineAsmMemConstraint(StringRef ConstraintCode) const override {
      if (ConstraintCode == "v")
        return InlineAsm::ConstraintCode::v;
      return TargetLowering::getInlineAsmMemConstraint(ConstraintCode);
    }

    /// Handle Lowering flag assembly outputs.
    SDValue LowerAsmOutputForConstraint(SDValue &Chain, SDValue &Flag,
                                        const SDLoc &DL,
                                        const AsmOperandInfo &Constraint,
                                        SelectionDAG &DAG) const override;

    /// Given a physical register constraint
    /// (e.g. {edx}), return the register number and the register class for the
    /// register.  This should only be used for C_Register constraints.  On
    /// error, this returns a register number of 0.
    std::pair<unsigned, const TargetRegisterClass *>
    getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                                 StringRef Constraint, MVT VT) const override;

    /// Return true if the addressing mode represented
    /// by AM is legal for this target, for a load/store of the specified type.
    bool isLegalAddressingMode(const DataLayout &DL, const AddrMode &AM,
                               Type *Ty, unsigned AS,
                               Instruction *I = nullptr) const override;

    bool addressingModeSupportsTLS(const GlobalValue &GV) const override;

    /// Return true if the specified immediate is legal
    /// icmp immediate, that is the target has icmp instructions which can
    /// compare a register against the immediate without having to materialize
    /// the immediate into a register.
    bool isLegalICmpImmediate(int64_t Imm) const override;

    /// Return true if the specified immediate is legal
    /// add immediate, that is the target has add instructions which can
    /// add a register and the immediate without having to materialize
    /// the immediate into a register.
    bool isLegalAddImmediate(int64_t Imm) const override;

    bool isLegalStoreImmediate(int64_t Imm) const override;

    /// Add x86-specific opcodes to the default list.
    bool isBinOp(unsigned Opcode) const override;

    /// Returns true if the opcode is a commutative binary operation.
    bool isCommutativeBinOp(unsigned Opcode) const override;

    /// Return true if it's free to truncate a value of
    /// type Ty1 to type Ty2. e.g. On x86 it's free to truncate a i32 value in
    /// register EAX to i16 by referencing its sub-register AX.
    bool isTruncateFree(Type *Ty1, Type *Ty2) const override;
    bool isTruncateFree(EVT VT1, EVT VT2) const override;

    bool allowTruncateForTailCall(Type *Ty1, Type *Ty2) const override;

    /// Return true if any actual instruction that defines a
    /// value of type Ty1 implicit zero-extends the value to Ty2 in the result
    /// register. This does not necessarily include registers defined in
    /// unknown ways, such as incoming arguments, or copies from unknown
    /// virtual registers. Also, if isTruncateFree(Ty2, Ty1) is true, this
    /// does not necessarily apply to truncate instructions. e.g. on x86-64,
    /// all instructions that define 32-bit values implicit zero-extend the
    /// result out to 64 bits.
    bool isZExtFree(Type *Ty1, Type *Ty2) const override;
    bool isZExtFree(EVT VT1, EVT VT2) const override;
    bool isZExtFree(SDValue Val, EVT VT2) const override;

    bool shouldConvertPhiType(Type *From, Type *To) const override;

    /// Return true if folding a vector load into ExtVal (a sign, zero, or any
    /// extend node) is profitable.
    bool isVectorLoadExtDesirable(SDValue) const override;

    /// Return true if an FMA operation is faster than a pair of fmul and fadd
    /// instructions. fmuladd intrinsics will be expanded to FMAs when this
    /// method returns true, otherwise fmuladd is expanded to fmul + fadd.
    bool isFMAFasterThanFMulAndFAdd(const MachineFunction &MF,
                                    EVT VT) const override;

    /// Return true if it's profitable to narrow operations of type SrcVT to
    /// DestVT. e.g. on x86, it's profitable to narrow from i32 to i8 but not
    /// from i32 to i16.
    bool isNarrowingProfitable(SDNode *N, EVT SrcVT, EVT DestVT) const override;

    bool shouldFoldSelectWithIdentityConstant(unsigned BinOpcode, EVT VT,
                                              unsigned SelectOpcode, SDValue X,
                                              SDValue Y) const override;

    /// Given an intrinsic, checks if on the target the intrinsic will need to
    /// map to a MemIntrinsicNode (touches memory). If this is the case, it
    /// returns true and stores the intrinsic information into the IntrinsicInfo
    /// that was passed to the function.
    void getTgtMemIntrinsic(SmallVectorImpl<IntrinsicInfo> &Infos,
                            const CallBase &I, MachineFunction &MF,
                            unsigned Intrinsic) const override;

    /// Returns true if the target can instruction select the
    /// specified FP immediate natively. If false, the legalizer will
    /// materialize the FP immediate as a load from a constant pool.
    bool isFPImmLegal(const APFloat &Imm, EVT VT,
                      bool ForCodeSize) const override;

    /// Targets can use this to indicate that they only support *some*
    /// VECTOR_SHUFFLE operations, those with specific masks. By default, if a
    /// target supports the VECTOR_SHUFFLE node, all mask values are assumed to
    /// be legal.
    bool isShuffleMaskLegal(ArrayRef<int> Mask, EVT VT) const override;

    /// Similar to isShuffleMaskLegal. Targets can use this to indicate if there
    /// is a suitable VECTOR_SHUFFLE that can be used to replace a VAND with a
    /// constant pool entry.
    bool isVectorClearMaskLegal(ArrayRef<int> Mask, EVT VT) const override;

    /// Returns true if lowering to a jump table is allowed.
    bool areJTsAllowed(const Function *Fn) const override;

    MVT getPreferredSwitchConditionType(LLVMContext &Context,
                                        EVT ConditionVT) const override;

    /// If true, then instruction selection should
    /// seek to shrink the FP constant of the specified type to a smaller type
    /// in order to save space and / or reduce runtime.
    bool ShouldShrinkFPConstant(EVT VT) const override;

    /// Return true if we believe it is correct and profitable to reduce the
    /// load node to a smaller type.
    bool
    shouldReduceLoadWidth(SDNode *Load, ISD::LoadExtType ExtTy, EVT NewVT,
                          std::optional<unsigned> ByteOffset) const override;

    /// Return true if the specified scalar FP type is computed in an SSE
    /// register, not on the X87 floating point stack.
    bool isScalarFPTypeInSSEReg(EVT VT) const;

    /// Returns true if it is beneficial to convert a load of a constant
    /// to just the constant itself.
    bool shouldConvertConstantLoadToIntImm(const APInt &Imm,
                                           Type *Ty) const override;

    bool reduceSelectOfFPConstantLoads(EVT CmpOpVT) const override;

    bool convertSelectOfConstantsToMath(EVT VT) const override;

    bool decomposeMulByConstant(LLVMContext &Context, EVT VT,
                                SDValue C) const override;

    /// Return true if EXTRACT_SUBVECTOR is cheap for this result type
    /// with this index.
    bool isExtractSubvectorCheap(EVT ResVT, EVT SrcVT,
                                 unsigned Index) const override;

    /// Scalar ops always have equal or better analysis/performance/power than
    /// the vector equivalent, so this always makes sense if the scalar op is
    /// supported.
    bool shouldScalarizeBinop(SDValue) const override;

    /// Extract of a scalar FP value from index 0 of a vector is free.
    bool isExtractVecEltCheap(EVT VT, unsigned Index) const override {
      EVT EltVT = VT.getScalarType();
      return (EltVT == MVT::f32 || EltVT == MVT::f64) && Index == 0;
    }

    /// Overflow nodes should get combined/lowered to optimal instructions
    /// (they should allow eliminating explicit compares by getting flags from
    /// math ops).
    bool shouldFormOverflowOp(unsigned Opcode, EVT VT,
                              bool MathUsed) const override;

    bool storeOfVectorConstantIsCheap(bool IsZero, EVT MemVT, unsigned NumElem,
                                      unsigned AddrSpace) const override {
      // If we can replace more than 2 scalar stores, there will be a reduction
      // in instructions even after we add a vector constant load.
      return IsZero || NumElem > 2;
    }

    bool isLoadBitCastBeneficial(EVT LoadVT, EVT BitcastVT,
                                 const SelectionDAG &DAG,
                                 const MachineMemOperand &MMO) const override;

    Register getRegisterByName(const char* RegName, LLT VT,
                               const MachineFunction &MF) const override;

    /// If a physical register, this returns the register that receives the
    /// exception address on entry to an EH pad.
    Register
    getExceptionPointerRegister(const Constant *PersonalityFn) const override;

    /// If a physical register, this returns the register that receives the
    /// exception typeid on entry to a landing pad.
    Register
    getExceptionSelectorRegister(const Constant *PersonalityFn) const override;

    bool needsFixedCatchObjects() const override;

    /// This method returns a target specific FastISel object,
    /// or null if the target does not support "fast" ISel.
    FastISel *
    createFastISel(FunctionLoweringInfo &funcInfo,
                   const TargetLibraryInfo *libInfo,
                   const LibcallLoweringInfo *libcallLowering) const override;

    /// If the target has a standard location for the stack protector cookie,
    /// returns the address of that location. Otherwise, returns nullptr.
    Value *getIRStackGuard(IRBuilderBase &IRB,
                           const LibcallLoweringInfo &Libcalls) const override;

    bool useLoadStackGuardNode(const Module &M) const override;
    bool useStackGuardXorFP() const override;
    void
    insertSSPDeclarations(Module &M,
                          const LibcallLoweringInfo &Libcalls) const override;
    SDValue emitStackGuardXorFP(SelectionDAG &DAG, SDValue Val,
                                const SDLoc &DL) const override;


    /// Return true if the target stores SafeStack pointer at a fixed offset in
    /// some non-standard address space, and populates the address space and
    /// offset as appropriate.
    Value *getSafeStackPointerLocation(
        IRBuilderBase &IRB, const LibcallLoweringInfo &Libcalls) const override;

    std::pair<SDValue, SDValue> BuildFILD(EVT DstVT, EVT SrcVT, const SDLoc &DL,
                                          SDValue Chain, SDValue Pointer,
                                          MachinePointerInfo PtrInfo,
                                          Align Alignment,
                                          SelectionDAG &DAG) const;

    /// Customize the preferred legalization strategy for certain types.
    LegalizeTypeAction getPreferredVectorAction(MVT VT) const override;

    MVT getRegisterTypeForCallingConv(LLVMContext &Context, CallingConv::ID CC,
                                      EVT VT) const override;

    unsigned getNumRegistersForCallingConv(LLVMContext &Context,
                                           CallingConv::ID CC,
                                           EVT VT) const override;

    unsigned getVectorTypeBreakdownForCallingConv(
        LLVMContext &Context, CallingConv::ID CC, EVT VT, EVT &IntermediateVT,
        unsigned &NumIntermediates, MVT &RegisterVT) const override;

    bool functionArgumentNeedsConsecutiveRegisters(
        Type *Ty, CallingConv::ID CallConv, bool isVarArg,
        const DataLayout &DL) const override;

    bool isIntDivCheap(EVT VT, AttributeList Attr) const override;

    bool supportSwiftError() const override;

    bool supportKCFIBundles() const override { return true; }

    MachineInstr *EmitKCFICheck(MachineBasicBlock &MBB,
                                MachineBasicBlock::instr_iterator &MBBI,
                                const TargetInstrInfo *TII) const override;

    bool hasStackProbeSymbol(const MachineFunction &MF) const override;
    bool hasInlineStackProbe(const MachineFunction &MF) const override;
    StringRef getStackProbeSymbolName(const MachineFunction &MF) const override;

    unsigned getStackProbeSize(const MachineFunction &MF) const;

    bool hasVectorBlend() const override { return true; }

    unsigned getMaxSupportedInterleaveFactor() const override { return 4; }

    bool isInlineAsmTargetBranch(const SmallVectorImpl<StringRef> &AsmStrs,
                                 unsigned OpNo) const override;

    SDValue visitMaskedLoad(SelectionDAG &DAG, const SDLoc &DL, SDValue Chain,
                            MachineMemOperand *MMO, SDValue &NewLoad,
                            SDValue Ptr, SDValue PassThru,
                            SDValue Mask) const override;
    SDValue visitMaskedStore(SelectionDAG &DAG, const SDLoc &DL, SDValue Chain,
                             MachineMemOperand *MMO, SDValue Ptr, SDValue Val,
                             SDValue Mask) const override;

    /// Lower interleaved load(s) into target specific
    /// instructions/intrinsics.
    bool lowerInterleavedLoad(Instruction *Load, Value *Mask,
                              ArrayRef<ShuffleVectorInst *> Shuffles,
                              ArrayRef<unsigned> Indices, unsigned Factor,
                              const APInt &GapMask) const override;

    /// Lower interleaved store(s) into target specific
    /// instructions/intrinsics.
    bool lowerInterleavedStore(Instruction *Store, Value *Mask,
                               ShuffleVectorInst *SVI, unsigned Factor,
                               const APInt &GapMask) const override;

    SDValue expandIndirectJTBranch(const SDLoc &dl, SDValue Value, SDValue Addr,
                                   int JTI, SelectionDAG &DAG) const override;

    Align getPrefLoopAlignment(MachineLoop *ML) const override;

    EVT getTypeToTransformTo(LLVMContext &Context, EVT VT) const override {
      if (VT == MVT::f80)
        return EVT::getIntegerVT(Context, 96);
      return TargetLoweringBase::getTypeToTransformTo(Context, VT);
    }

  protected:
    std::pair<const TargetRegisterClass *, uint8_t>
    findRepresentativeClass(const TargetRegisterInfo *TRI,
                            MVT VT) const override;

  private:
    /// Keep a reference to the X86Subtarget around so that we can
    /// make the right decision when generating code for different targets.
    const X86Subtarget &Subtarget;

    /// A list of legal FP immediates.
    std::vector<APFloat> LegalFPImmediates;

    /// Indicate that this x86 target can instruction
    /// select the specified FP immediate natively.
    void addLegalFPImmediate(const APFloat& Imm) {
      LegalFPImmediates.push_back(Imm);
    }

    SDValue LowerCallResult(SDValue Chain, SDValue InGlue,
                            CallingConv::ID CallConv, bool isVarArg,
                            const SmallVectorImpl<ISD::InputArg> &Ins,
                            const SDLoc &dl, SelectionDAG &DAG,
                            SmallVectorImpl<SDValue> &InVals,
                            uint32_t *RegMask) const;
    SDValue LowerMemArgument(SDValue Chain, CallingConv::ID CallConv,
                             const SmallVectorImpl<ISD::InputArg> &ArgInfo,
                             const SDLoc &dl, SelectionDAG &DAG,
                             const CCValAssign &VA, MachineFrameInfo &MFI,
                             unsigned i) const;
    SDValue LowerMemOpCallTo(SDValue Chain, SDValue StackPtr, SDValue Arg,
                             const SDLoc &dl, SelectionDAG &DAG,
                             const CCValAssign &VA,
                             ISD::ArgFlagsTy Flags, bool isByval) const;

    // Call lowering helpers.

    /// Check whether the call is eligible for sibling call optimization.
    bool
    isEligibleForSiblingCallOpt(TargetLowering::CallLoweringInfo &CLI,
                                CCState &CCInfo,
                                SmallVectorImpl<CCValAssign> &ArgLocs) const;
    SDValue EmitTailCallLoadRetAddr(SelectionDAG &DAG, SDValue &OutRetAddr,
                                    SDValue Chain, bool IsTailCall,
                                    bool Is64Bit, int FPDiff,
                                    const SDLoc &dl) const;

    unsigned GetAlignedArgumentStackSize(unsigned StackSize,
                                         SelectionDAG &DAG) const;

    unsigned getAddressSpace() const;

    SDValue FP_TO_INTHelper(SDValue Op, SelectionDAG &DAG, bool IsSigned,
                            SDValue &Chain) const;
    SDValue LRINT_LLRINTHelper(SDNode *N, SelectionDAG &DAG) const;

    SDValue LowerBUILD_VECTOR(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerVSELECT(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerEXTRACT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerINSERT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) const;

    unsigned getGlobalWrapperKind(const GlobalValue *GV,
                                  const unsigned char OpFlags) const;
    SDValue LowerConstantPool(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerBlockAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerGlobalTLSAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerExternalSymbol(SDValue Op, SelectionDAG &DAG) const;

    /// Creates target global address or external symbol nodes for calls or
    /// other uses.
    SDValue LowerGlobalOrExternal(SDValue Op, SelectionDAG &DAG, bool ForCall,
                                  bool *IsImpCall) const;

    SDValue LowerSINT_TO_FP(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerUINT_TO_FP(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerTRUNCATE(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerFP_TO_INT(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerFP_TO_INT_SAT(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerLRINT_LLRINT(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSETCC(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSETCCCARRY(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSELECT(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerConditionalBranch(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerJumpTable(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerDYNAMIC_STACKALLOC(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerVASTART(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerVAARG(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerRETURNADDR(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerADDROFRETURNADDR(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerFRAMEADDR(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerFRAME_TO_ARGS_OFFSET(SDValue Op, SelectionDAG &DAG) const;
    ByValCopyKind ByValNeedsCopyForTailCall(SelectionDAG &DAG, SDValue Src,
                                            SDValue Dst,
                                            ISD::ArgFlagsTy Flags) const;
    SDValue LowerEH_RETURN(SDValue Op, SelectionDAG &DAG) const;
    SDValue lowerEH_SJLJ_SETJMP(SDValue Op, SelectionDAG &DAG) const;
    SDValue lowerEH_SJLJ_LONGJMP(SDValue Op, SelectionDAG &DAG) const;
    SDValue lowerEH_SJLJ_SETUP_DISPATCH(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerINIT_TRAMPOLINE(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerGET_ROUNDING(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSET_ROUNDING(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerGET_FPENV_MEM(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSET_FPENV_MEM(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerRESET_FPENV(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerWin64_i128OP(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerWin64_FP_TO_INT128(SDValue Op, SelectionDAG &DAG,
                                    SDValue &Chain) const;
    SDValue LowerWin64_INT128_TO_FP(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerGC_TRANSITION(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerINTRINSIC_WO_CHAIN(SDValue Op, SelectionDAG &DAG) const;
    SDValue lowerFaddFsub(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerFP_EXTEND(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerFP_ROUND(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerFP_TO_BF16(SDValue Op, SelectionDAG &DAG) const;

    SDValue
    LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
                         const SmallVectorImpl<ISD::InputArg> &Ins,
                         const SDLoc &dl, SelectionDAG &DAG,
                         SmallVectorImpl<SDValue> &InVals) const override;
    SDValue LowerCall(CallLoweringInfo &CLI,
                      SmallVectorImpl<SDValue> &InVals) const override;

    SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
                        const SmallVectorImpl<ISD::OutputArg> &Outs,
                        const SmallVectorImpl<SDValue> &OutVals,
                        const SDLoc &dl, SelectionDAG &DAG) const override;

    bool supportSplitCSR(MachineFunction *MF) const override {
      return MF->getFunction().getCallingConv() == CallingConv::CXX_FAST_TLS &&
          MF->getFunction().hasFnAttribute(Attribute::NoUnwind);
    }
    void initializeSplitCSR(MachineBasicBlock *Entry) const override;
    void insertCopiesSplitCSR(
      MachineBasicBlock *Entry,
      const SmallVectorImpl<MachineBasicBlock *> &Exits) const override;

    bool isUsedByReturnOnly(SDNode *N, SDValue &Chain) const override;

    bool mayBeEmittedAsTailCall(const CallInst *CI) const override;

    EVT getTypeForExtReturn(LLVMContext &Context, EVT VT,
                            ISD::NodeType ExtendKind) const override;

    bool CanLowerReturn(CallingConv::ID CallConv, MachineFunction &MF,
                        bool isVarArg,
                        const SmallVectorImpl<ISD::OutputArg> &Outs,
                        LLVMContext &Context,
                        const Type *RetTy) const override;

    const MCPhysReg *getScratchRegisters(CallingConv::ID CC) const override;
    ArrayRef<MCPhysReg> getRoundingControlRegisters() const override;

    TargetLoweringBase::AtomicExpansionKind
    shouldExpandAtomicLoadInIR(LoadInst *LI) const override;
    TargetLoweringBase::AtomicExpansionKind
    shouldExpandAtomicStoreInIR(StoreInst *SI) const override;
    TargetLoweringBase::AtomicExpansionKind
    shouldExpandAtomicRMWInIR(const AtomicRMWInst *AI) const override;
    TargetLoweringBase::AtomicExpansionKind
    shouldExpandLogicAtomicRMWInIR(const AtomicRMWInst *AI) const;
    void emitBitTestAtomicRMWIntrinsic(AtomicRMWInst *AI) const override;
    void emitCmpArithAtomicRMWIntrinsic(AtomicRMWInst *AI) const override;

    LoadInst *
    lowerIdempotentRMWIntoFencedLoad(AtomicRMWInst *AI) const override;

    bool needsCmpXchgNb(Type *MemType) const;

    void SetupEntryBlockForSjLj(MachineInstr &MI, MachineBasicBlock *MBB,
                                MachineBasicBlock *DispatchBB, int FI) const;

    // Utility function to emit the low-level va_arg code for X86-64.
    MachineBasicBlock *
    EmitVAARGWithCustomInserter(MachineInstr &MI, MachineBasicBlock *MBB) const;

    /// Utility function to emit the xmm reg save portion of va_start.
    MachineBasicBlock *EmitLoweredCascadedSelect(MachineInstr &MI1,
                                                 MachineInstr &MI2,
                                                 MachineBasicBlock *BB) const;

    MachineBasicBlock *EmitLoweredSelect(MachineInstr &I,
                                         MachineBasicBlock *BB) const;

    MachineBasicBlock *EmitLoweredCatchRet(MachineInstr &MI,
                                           MachineBasicBlock *BB) const;

    MachineBasicBlock *EmitLoweredSegAlloca(MachineInstr &MI,
                                            MachineBasicBlock *BB) const;

    MachineBasicBlock *EmitLoweredProbedAlloca(MachineInstr &MI,
                                               MachineBasicBlock *BB) const;

    MachineBasicBlock *EmitLoweredTLSCall(MachineInstr &MI,
                                          MachineBasicBlock *BB) const;

    MachineBasicBlock *EmitLoweredIndirectThunk(MachineInstr &MI,
                                                MachineBasicBlock *BB) const;

    MachineBasicBlock *emitEHSjLjSetJmp(MachineInstr &MI,
                                        MachineBasicBlock *MBB) const;

    void emitSetJmpShadowStackFix(MachineInstr &MI,
                                  MachineBasicBlock *MBB) const;

    MachineBasicBlock *emitEHSjLjLongJmp(MachineInstr &MI,
                                         MachineBasicBlock *MBB) const;

    MachineBasicBlock *emitLongJmpShadowStackFix(MachineInstr &MI,
                                                 MachineBasicBlock *MBB) const;

    MachineBasicBlock *EmitSjLjDispatchBlock(MachineInstr &MI,
                                             MachineBasicBlock *MBB) const;

    MachineBasicBlock *emitPatchableEventCall(MachineInstr &MI,
                                              MachineBasicBlock *MBB) const;

    /// Emit flags for the given setcc condition and operands. Also returns the
    /// corresponding X86 condition code constant in X86CC.
    SDValue emitFlagsForSetcc(SDValue Op0, SDValue Op1, ISD::CondCode CC,
                              const SDLoc &dl, SelectionDAG &DAG,
                              SDValue &X86CC) const;

    bool optimizeFMulOrFDivAsShiftAddBitcast(SDNode *N, SDValue FPConst,
                                             SDValue IntPow2) const override;

    /// Check if replacement of SQRT with RSQRT should be disabled.
    bool isFsqrtCheap(SDValue Op, SelectionDAG &DAG) const override;

    /// Use rsqrt* to speed up sqrt calculations.
    SDValue getSqrtEstimate(SDValue Op, SelectionDAG &DAG, int Enabled,
                            int &RefinementSteps, bool &UseOneConstNR,
                            bool Reciprocal) const override;

    /// Use rcp* to speed up fdiv calculations.
    SDValue getRecipEstimate(SDValue Op, SelectionDAG &DAG, int Enabled,
                             int &RefinementSteps) const override;

    /// Reassociate floating point divisions into multiply by reciprocal.
    unsigned combineRepeatedFPDivisors() const override;

    SDValue BuildSDIVPow2(SDNode *N, const APInt &Divisor, SelectionDAG &DAG,
                          SmallVectorImpl<SDNode *> &Created) const override;

    SDValue getMOVL(SelectionDAG &DAG, const SDLoc &dl, MVT VT, SDValue V1,
                    SDValue V2) const;
  };

  namespace X86 {
  FastISel *createFastISel(FunctionLoweringInfo &funcInfo,
                           const TargetLibraryInfo *libInfo,
                           const LibcallLoweringInfo *libcallLowering);
  } // end namespace X86

  // X86 specific Gather/Scatter nodes.
  // The class has the same order of operands as MaskedGatherScatterSDNode for
  // convenience.
  class X86MaskedGatherScatterSDNode : public MemIntrinsicSDNode {
  public:
    // This is a intended as a utility and should never be directly created.
    X86MaskedGatherScatterSDNode() = delete;
    ~X86MaskedGatherScatterSDNode() = delete;

    const SDValue &getBasePtr() const { return getOperand(3); }
    const SDValue &getIndex()   const { return getOperand(4); }
    const SDValue &getMask()    const { return getOperand(2); }
    const SDValue &getScale()   const { return getOperand(5); }

    static bool classof(const SDNode *N) {
      return N->getOpcode() == X86ISD::MGATHER ||
             N->getOpcode() == X86ISD::MSCATTER;
    }
  };

  class X86MaskedGatherSDNode : public X86MaskedGatherScatterSDNode {
  public:
    const SDValue &getPassThru() const { return getOperand(1); }

    static bool classof(const SDNode *N) {
      return N->getOpcode() == X86ISD::MGATHER;
    }
  };

  class X86MaskedScatterSDNode : public X86MaskedGatherScatterSDNode {
  public:
    const SDValue &getValue() const { return getOperand(1); }

    static bool classof(const SDNode *N) {
      return N->getOpcode() == X86ISD::MSCATTER;
    }
  };

  /// Generate unpacklo/unpackhi shuffle mask.
  void createUnpackShuffleMask(EVT VT, SmallVectorImpl<int> &Mask, bool Lo,
                               bool Unary);

  /// Similar to unpacklo/unpackhi, but without the 128-bit lane limitation
  /// imposed by AVX and specific to the unary pattern. Example:
  /// v8iX Lo --> <0, 0, 1, 1, 2, 2, 3, 3>
  /// v8iX Hi --> <4, 4, 5, 5, 6, 6, 7, 7>
  void createSplat2ShuffleMask(MVT VT, SmallVectorImpl<int> &Mask, bool Lo);

} // end namespace llvm

#endif // LLVM_LIB_TARGET_X86_X86ISELLOWERING_H
