//===-- PPCISelLowering.h - PPC32 DAG Lowering Interface --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that PPC uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_POWERPC_PPCISELLOWERING_H
#define LLVM_LIB_TARGET_POWERPC_PPCISELLOWERING_H

#include "PPCInstrInfo.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGenTypes/MachineValueType.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Type.h"
#include <optional>
#include <utility>

namespace llvm {

  /// Define some predicates that are used for node matching.
  namespace PPC {

    /// isVPKUHUMShuffleMask - Return true if this is the shuffle mask for a
    /// VPKUHUM instruction.
    bool isVPKUHUMShuffleMask(ShuffleVectorSDNode *N, unsigned ShuffleKind,
                              SelectionDAG &DAG);

    /// isVPKUWUMShuffleMask - Return true if this is the shuffle mask for a
    /// VPKUWUM instruction.
    bool isVPKUWUMShuffleMask(ShuffleVectorSDNode *N, unsigned ShuffleKind,
                              SelectionDAG &DAG);

    /// isVPKUDUMShuffleMask - Return true if this is the shuffle mask for a
    /// VPKUDUM instruction.
    bool isVPKUDUMShuffleMask(ShuffleVectorSDNode *N, unsigned ShuffleKind,
                              SelectionDAG &DAG);

    /// isVMRGLShuffleMask - Return true if this is a shuffle mask suitable for
    /// a VRGL* instruction with the specified unit size (1,2 or 4 bytes).
    bool isVMRGLShuffleMask(ShuffleVectorSDNode *N, unsigned UnitSize,
                            unsigned ShuffleKind, SelectionDAG &DAG);

    /// isVMRGHShuffleMask - Return true if this is a shuffle mask suitable for
    /// a VRGH* instruction with the specified unit size (1,2 or 4 bytes).
    bool isVMRGHShuffleMask(ShuffleVectorSDNode *N, unsigned UnitSize,
                            unsigned ShuffleKind, SelectionDAG &DAG);

    /// isVMRGEOShuffleMask - Return true if this is a shuffle mask suitable for
    /// a VMRGEW or VMRGOW instruction
    bool isVMRGEOShuffleMask(ShuffleVectorSDNode *N, bool CheckEven,
                             unsigned ShuffleKind, SelectionDAG &DAG);
    /// isXXSLDWIShuffleMask - Return true if this is a shuffle mask suitable
    /// for a XXSLDWI instruction.
    bool isXXSLDWIShuffleMask(ShuffleVectorSDNode *N, unsigned &ShiftElts,
                              bool &Swap, bool IsLE);

    /// isXXBRHShuffleMask - Return true if this is a shuffle mask suitable
    /// for a XXBRH instruction.
    bool isXXBRHShuffleMask(ShuffleVectorSDNode *N);

    /// isXXBRWShuffleMask - Return true if this is a shuffle mask suitable
    /// for a XXBRW instruction.
    bool isXXBRWShuffleMask(ShuffleVectorSDNode *N);

    /// isXXBRDShuffleMask - Return true if this is a shuffle mask suitable
    /// for a XXBRD instruction.
    bool isXXBRDShuffleMask(ShuffleVectorSDNode *N);

    /// isXXBRQShuffleMask - Return true if this is a shuffle mask suitable
    /// for a XXBRQ instruction.
    bool isXXBRQShuffleMask(ShuffleVectorSDNode *N);

    /// isXXPERMDIShuffleMask - Return true if this is a shuffle mask suitable
    /// for a XXPERMDI instruction.
    bool isXXPERMDIShuffleMask(ShuffleVectorSDNode *N, unsigned &ShiftElts,
                              bool &Swap, bool IsLE);

    /// isVSLDOIShuffleMask - If this is a vsldoi shuffle mask, return the
    /// shift amount, otherwise return -1.
    int isVSLDOIShuffleMask(SDNode *N, unsigned ShuffleKind,
                            SelectionDAG &DAG);

    /// isSplatShuffleMask - Return true if the specified VECTOR_SHUFFLE operand
    /// specifies a splat of a single element that is suitable for input to
    /// VSPLTB/VSPLTH/VSPLTW.
    bool isSplatShuffleMask(ShuffleVectorSDNode *N, unsigned EltSize);

    /// isXXINSERTWMask - Return true if this VECTOR_SHUFFLE can be handled by
    /// the XXINSERTW instruction introduced in ISA 3.0. This is essentially any
    /// shuffle of v4f32/v4i32 vectors that just inserts one element from one
    /// vector into the other. This function will also set a couple of
    /// output parameters for how much the source vector needs to be shifted and
    /// what byte number needs to be specified for the instruction to put the
    /// element in the desired location of the target vector.
    bool isXXINSERTWMask(ShuffleVectorSDNode *N, unsigned &ShiftElts,
                         unsigned &InsertAtByte, bool &Swap, bool IsLE);

    /// getSplatIdxForPPCMnemonics - Return the splat index as a value that is
    /// appropriate for PPC mnemonics (which have a big endian bias - namely
    /// elements are counted from the left of the vector register).
    unsigned getSplatIdxForPPCMnemonics(SDNode *N, unsigned EltSize,
                                        SelectionDAG &DAG);

    /// get_VSPLTI_elt - If this is a build_vector of constants which can be
    /// formed by using a vspltis[bhw] instruction of the specified element
    /// size, return the constant being splatted.  The ByteSize field indicates
    /// the number of bytes of each element [124] -> [bhw].
    SDValue get_VSPLTI_elt(SDNode *N, unsigned ByteSize, SelectionDAG &DAG);

    // Flags for computing the optimal addressing mode for loads and stores.
    enum MemOpFlags {
      MOF_None = 0,

      // Extension mode for integer loads.
      MOF_SExt = 1,
      MOF_ZExt = 1 << 1,
      MOF_NoExt = 1 << 2,

      // Address computation flags.
      MOF_NotAddNorCst = 1 << 5,      // Not const. or sum of ptr and scalar.
      MOF_RPlusSImm16 = 1 << 6,       // Reg plus signed 16-bit constant.
      MOF_RPlusLo = 1 << 7,           // Reg plus signed 16-bit relocation
      MOF_RPlusSImm16Mult4 = 1 << 8,  // Reg plus 16-bit signed multiple of 4.
      MOF_RPlusSImm16Mult16 = 1 << 9, // Reg plus 16-bit signed multiple of 16.
      MOF_RPlusSImm34 = 1 << 10,      // Reg plus 34-bit signed constant.
      MOF_RPlusR = 1 << 11,           // Sum of two variables.
      MOF_PCRel = 1 << 12,            // PC-Relative relocation.
      MOF_AddrIsSImm32 = 1 << 13,     // A simple 32-bit constant.

      // The in-memory type.
      MOF_SubWordInt = 1 << 15,
      MOF_WordInt = 1 << 16,
      MOF_DoubleWordInt = 1 << 17,
      MOF_ScalarFloat = 1 << 18, // Scalar single or double precision.
      MOF_Vector = 1 << 19,      // Vector types and quad precision scalars.
      MOF_Vector256 = 1 << 20,

      // Subtarget features.
      MOF_SubtargetBeforeP9 = 1 << 22,
      MOF_SubtargetP9 = 1 << 23,
      MOF_SubtargetP10 = 1 << 24,
      MOF_SubtargetSPE = 1 << 25
    };

    // The addressing modes for loads and stores.
    enum AddrMode {
      AM_None,
      AM_DForm,
      AM_DSForm,
      AM_DQForm,
      AM_PrefixDForm,
      AM_XForm,
      AM_PCRel
    };
  } // end namespace PPC

  class PPCTargetLowering : public TargetLowering {
    const PPCSubtarget &Subtarget;

  public:
    explicit PPCTargetLowering(const PPCTargetMachine &TM,
                               const PPCSubtarget &STI);

    bool isSelectSupported(SelectSupportKind Kind) const override {
      // PowerPC does not support scalar condition selects on vectors.
      return (Kind != SelectSupportKind::ScalarCondVectorVal);
    }

    /// getPreferredVectorAction - The code we generate when vector types are
    /// legalized by promoting the integer element type is often much worse
    /// than code we generate if we widen the type for applicable vector types.
    /// The issue with promoting is that the vector is scalaraized, individual
    /// elements promoted and then the vector is rebuilt. So say we load a pair
    /// of v4i8's and shuffle them. This will turn into a mess of 8 extending
    /// loads, moves back into VSR's (or memory ops if we don't have moves) and
    /// then the VPERM for the shuffle. All in all a very slow sequence.
    TargetLoweringBase::LegalizeTypeAction getPreferredVectorAction(MVT VT)
      const override {
      // Default handling for scalable and single-element vectors.
      if (VT.isScalableVector() || VT.getVectorNumElements() == 1)
        return TargetLoweringBase::getPreferredVectorAction(VT);

      // Split and promote vNi1 vectors so we don't produce v256i1/v512i1
      // types as those are only for MMA instructions.
      if (VT.getScalarSizeInBits() == 1 && VT.getSizeInBits() > 16)
        return TypeSplitVector;
      if (VT.getScalarSizeInBits() == 1)
        return TypePromoteInteger;

      // Widen vectors that have reasonably sized elements.
      if (VT.getScalarSizeInBits() % 8 == 0)
        return TypeWidenVector;
      return TargetLoweringBase::getPreferredVectorAction(VT);
    }

    bool useSoftFloat() const override;

    bool hasSPE() const;

    MVT getScalarShiftAmountTy(const DataLayout &, EVT) const override {
      return MVT::i32;
    }

    bool isCheapToSpeculateCttz(Type *Ty) const override {
      return true;
    }

    bool isCheapToSpeculateCtlz(Type *Ty) const override {
      return true;
    }

    bool
    shallExtractConstSplatVectorElementToStore(Type *VectorTy,
                                               unsigned ElemSizeInBits,
                                               unsigned &Index) const override;

    bool isCtlzFast() const override {
      return true;
    }

    bool isEqualityCmpFoldedWithSignedCmp() const override {
      return false;
    }

    bool hasAndNotCompare(SDValue) const override {
      return true;
    }

    bool preferIncOfAddToSubOfNot(EVT VT) const override;

    bool convertSetCCLogicToBitwiseLogic(EVT VT) const override {
      return VT.isScalarInteger();
    }

    SDValue getNegatedExpression(SDValue Op, SelectionDAG &DAG, bool LegalOps,
                                 bool OptForSize, NegatibleCost &Cost,
                                 unsigned Depth = 0) const override;

    /// getSetCCResultType - Return the ISD::SETCC ValueType
    EVT getSetCCResultType(const DataLayout &DL, LLVMContext &Context,
                           EVT VT) const override;

    /// Return true if target always benefits from combining into FMA for a
    /// given value type. This must typically return false on targets where FMA
    /// takes more cycles to execute than FADD.
    bool enableAggressiveFMAFusion(EVT VT) const override;

    /// getPreIndexedAddressParts - returns true by value, base pointer and
    /// offset pointer and addressing mode by reference if the node's address
    /// can be legally represented as pre-indexed load / store address.
    bool getPreIndexedAddressParts(SDNode *N, SDValue &Base,
                                   SDValue &Offset,
                                   ISD::MemIndexedMode &AM,
                                   SelectionDAG &DAG) const override;

    /// SelectAddressEVXRegReg - Given the specified addressed, check to see if
    /// it can be more efficiently represented as [r+imm].
    bool SelectAddressEVXRegReg(SDValue N, SDValue &Base, SDValue &Index,
                                SelectionDAG &DAG) const;

    /// SelectAddressRegReg - Given the specified addressed, check to see if it
    /// can be more efficiently represented as [r+imm]. If \p EncodingAlignment
    /// is non-zero, only accept displacement which is not suitable for [r+imm].
    /// Returns false if it can be represented by [r+imm], which are preferred.
    bool SelectAddressRegReg(SDValue N, SDValue &Base, SDValue &Index,
                             SelectionDAG &DAG,
                             MaybeAlign EncodingAlignment = std::nullopt) const;

    /// SelectAddressRegImm - Returns true if the address N can be represented
    /// by a base register plus a signed 16-bit displacement [r+imm], and if it
    /// is not better represented as reg+reg. If \p EncodingAlignment is
    /// non-zero, only accept displacements suitable for instruction encoding
    /// requirement, i.e. multiples of 4 for DS form.
    bool SelectAddressRegImm(SDValue N, SDValue &Disp, SDValue &Base,
                             SelectionDAG &DAG,
                             MaybeAlign EncodingAlignment) const;
    bool SelectAddressRegImm34(SDValue N, SDValue &Disp, SDValue &Base,
                               SelectionDAG &DAG) const;

    /// SelectAddressRegRegOnly - Given the specified addressed, force it to be
    /// represented as an indexed [r+r] operation.
    bool SelectAddressRegRegOnly(SDValue N, SDValue &Base, SDValue &Index,
                                 SelectionDAG &DAG) const;

    /// SelectAddressPCRel - Represent the specified address as pc relative to
    /// be represented as [pc+imm]
    bool SelectAddressPCRel(SDValue N, SDValue &Base) const;

    Sched::Preference getSchedulingPreference(SDNode *N) const override;

    /// LowerOperation - Provide custom lowering hooks for some operations.
    ///
    SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;

    /// ReplaceNodeResults - Replace the results of node with an illegal result
    /// type with new values built out of custom code.
    ///
    void ReplaceNodeResults(SDNode *N, SmallVectorImpl<SDValue>&Results,
                            SelectionDAG &DAG) const override;

    SDValue expandVSXLoadForLE(SDNode *N, DAGCombinerInfo &DCI) const;
    SDValue expandVSXStoreForLE(SDNode *N, DAGCombinerInfo &DCI) const;

    SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const override;

    SDValue BuildSDIVPow2(SDNode *N, const APInt &Divisor, SelectionDAG &DAG,
                          SmallVectorImpl<SDNode *> &Created) const override;

    Register getRegisterByName(const char* RegName, LLT VT,
                               const MachineFunction &MF) const override;

    void computeKnownBitsForTargetNode(const SDValue Op,
                                       KnownBits &Known,
                                       const APInt &DemandedElts,
                                       const SelectionDAG &DAG,
                                       unsigned Depth = 0) const override;

    Align getPrefLoopAlignment(MachineLoop *ML) const override;

    bool shouldInsertFencesForAtomic(const Instruction *I) const override {
      return true;
    }

    Value *emitLoadLinked(IRBuilderBase &Builder, Type *ValueTy, Value *Addr,
                          AtomicOrdering Ord) const override;

    Value *emitStoreConditional(IRBuilderBase &Builder, Value *Val, Value *Addr,
                                AtomicOrdering Ord) const override;

    Instruction *emitLeadingFence(IRBuilderBase &Builder, Instruction *Inst,
                                  AtomicOrdering Ord) const override;
    Instruction *emitTrailingFence(IRBuilderBase &Builder, Instruction *Inst,
                                   AtomicOrdering Ord) const override;

    bool shouldInlineQuadwordAtomics() const;

    TargetLowering::AtomicExpansionKind
    shouldExpandAtomicRMWInIR(const AtomicRMWInst *AI) const override;

    TargetLowering::AtomicExpansionKind
    shouldExpandAtomicCmpXchgInIR(const AtomicCmpXchgInst *AI) const override;

    Value *emitMaskedAtomicRMWIntrinsic(IRBuilderBase &Builder,
                                        AtomicRMWInst *AI, Value *AlignedAddr,
                                        Value *Incr, Value *Mask,
                                        Value *ShiftAmt,
                                        AtomicOrdering Ord) const override;
    Value *emitMaskedAtomicCmpXchgIntrinsic(IRBuilderBase &Builder,
                                            AtomicCmpXchgInst *CI,
                                            Value *AlignedAddr, Value *CmpVal,
                                            Value *NewVal, Value *Mask,
                                            AtomicOrdering Ord) const override;

    MachineBasicBlock *
    EmitInstrWithCustomInserter(MachineInstr &MI,
                                MachineBasicBlock *MBB) const override;
    MachineBasicBlock *EmitAtomicBinary(MachineInstr &MI,
                                        MachineBasicBlock *MBB,
                                        unsigned AtomicSize,
                                        unsigned BinOpcode,
                                        unsigned CmpOpcode = 0,
                                        unsigned CmpPred = 0) const;
    MachineBasicBlock *EmitPartwordAtomicBinary(MachineInstr &MI,
                                                MachineBasicBlock *MBB,
                                                bool is8bit,
                                                unsigned Opcode,
                                                unsigned CmpOpcode = 0,
                                                unsigned CmpPred = 0) const;

    MachineBasicBlock *emitEHSjLjSetJmp(MachineInstr &MI,
                                        MachineBasicBlock *MBB) const;

    MachineBasicBlock *emitEHSjLjLongJmp(MachineInstr &MI,
                                         MachineBasicBlock *MBB) const;

    MachineBasicBlock *emitProbedAlloca(MachineInstr &MI,
                                        MachineBasicBlock *MBB) const;

    bool hasInlineStackProbe(const MachineFunction &MF) const override;

    unsigned getStackProbeSize(const MachineFunction &MF) const;

    ConstraintType getConstraintType(StringRef Constraint) const override;

    /// Examine constraint string and operand type and determine a weight value.
    /// The operand object must already have been set up with the operand type.
    ConstraintWeight getSingleConstraintMatchWeight(
      AsmOperandInfo &info, const char *constraint) const override;

    std::pair<unsigned, const TargetRegisterClass *>
    getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                                 StringRef Constraint, MVT VT) const override;

    /// getByValTypeAlignment - Return the desired alignment for ByVal aggregate
    /// function arguments in the caller parameter area.
    Align getByValTypeAlignment(Type *Ty, const DataLayout &DL) const override;

    /// LowerAsmOperandForConstraint - Lower the specified operand into the Ops
    /// vector.  If it is invalid, don't add anything to Ops.
    void LowerAsmOperandForConstraint(SDValue Op, StringRef Constraint,
                                      std::vector<SDValue> &Ops,
                                      SelectionDAG &DAG) const override;

    InlineAsm::ConstraintCode
    getInlineAsmMemConstraint(StringRef ConstraintCode) const override {
      if (ConstraintCode == "es")
        return InlineAsm::ConstraintCode::es;
      else if (ConstraintCode == "Q")
        return InlineAsm::ConstraintCode::Q;
      else if (ConstraintCode == "Z")
        return InlineAsm::ConstraintCode::Z;
      else if (ConstraintCode == "Zy")
        return InlineAsm::ConstraintCode::Zy;
      return TargetLowering::getInlineAsmMemConstraint(ConstraintCode);
    }

    void CollectTargetIntrinsicOperands(const CallInst &I,
                                 SmallVectorImpl<SDValue> &Ops,
                                 SelectionDAG &DAG) const override;

    /// isLegalAddressingMode - Return true if the addressing mode represented
    /// by AM is legal for this target, for a load/store of the specified type.
    bool isLegalAddressingMode(const DataLayout &DL, const AddrMode &AM,
                               Type *Ty, unsigned AS,
                               Instruction *I = nullptr) const override;

    /// isLegalICmpImmediate - Return true if the specified immediate is legal
    /// icmp immediate, that is the target has icmp instructions which can
    /// compare a register against the immediate without having to materialize
    /// the immediate into a register.
    bool isLegalICmpImmediate(int64_t Imm) const override;

    /// isLegalAddImmediate - Return true if the specified immediate is legal
    /// add immediate, that is the target has add instructions which can
    /// add a register and the immediate without having to materialize
    /// the immediate into a register.
    bool isLegalAddImmediate(int64_t Imm) const override;

    /// isTruncateFree - Return true if it's free to truncate a value of
    /// type Ty1 to type Ty2. e.g. On PPC it's free to truncate a i64 value in
    /// register X1 to i32 by referencing its sub-register R1.
    bool isTruncateFree(Type *Ty1, Type *Ty2) const override;
    bool isTruncateFree(EVT VT1, EVT VT2) const override;

    bool isZExtFree(SDValue Val, EVT VT2) const override;

    bool isFPExtFree(EVT DestVT, EVT SrcVT) const override;

    /// Returns true if it is beneficial to convert a load of a constant
    /// to just the constant itself.
    bool shouldConvertConstantLoadToIntImm(const APInt &Imm,
                                           Type *Ty) const override;

    bool convertSelectOfConstantsToMath(EVT VT) const override {
      return true;
    }

    bool decomposeMulByConstant(LLVMContext &Context, EVT VT,
                                SDValue C) const override;

    bool isDesirableToTransformToIntegerOp(unsigned Opc,
                                           EVT VT) const override {
      // Only handle float load/store pair because float(fpr) load/store
      // instruction has more cycles than integer(gpr) load/store in PPC.
      if (Opc != ISD::LOAD && Opc != ISD::STORE)
        return false;
      if (VT != MVT::f32 && VT != MVT::f64)
        return false;

      return true;
    }

    // Returns true if the address of the global is stored in TOC entry.
    bool isAccessedAsGotIndirect(SDValue N) const;

    bool isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const override;

    bool getTgtMemIntrinsic(IntrinsicInfo &Info, const CallBase &I,
                            MachineFunction &MF,
                            unsigned Intrinsic) const override;

    /// It returns EVT::Other if the type should be determined using generic
    /// target-independent logic.
    EVT getOptimalMemOpType(LLVMContext &Context, const MemOp &Op,
                            const AttributeList &FuncAttributes) const override;

    /// Is unaligned memory access allowed for the given type, and is it fast
    /// relative to software emulation.
    bool allowsMisalignedMemoryAccesses(
        EVT VT, unsigned AddrSpace, Align Alignment = Align(1),
        MachineMemOperand::Flags Flags = MachineMemOperand::MONone,
        unsigned *Fast = nullptr) const override;

    /// isFMAFasterThanFMulAndFAdd - Return true if an FMA operation is faster
    /// than a pair of fmul and fadd instructions. fmuladd intrinsics will be
    /// expanded to FMAs when this method returns true, otherwise fmuladd is
    /// expanded to fmul + fadd.
    bool isFMAFasterThanFMulAndFAdd(const MachineFunction &MF,
                                    EVT VT) const override;

    bool isFMAFasterThanFMulAndFAdd(const Function &F, Type *Ty) const override;

    /// isProfitableToHoist - Check if it is profitable to hoist instruction
    /// \p I to its dominator block.
    /// For example, it is not profitable if \p I and it's only user can form a
    /// FMA instruction, because Powerpc prefers FMADD.
    bool isProfitableToHoist(Instruction *I) const override;

    const MCPhysReg *getScratchRegisters(CallingConv::ID CC) const override;

    // Should we expand the build vector with shuffles?
    bool
    shouldExpandBuildVectorWithShuffles(EVT VT,
                                        unsigned DefinedValues) const override;

    // Keep the zero-extensions for arguments to libcalls.
    bool shouldKeepZExtForFP16Conv() const override { return true; }

    /// createFastISel - This method returns a target-specific FastISel object,
    /// or null if the target does not support "fast" instruction selection.
    FastISel *createFastISel(FunctionLoweringInfo &FuncInfo,
                             const TargetLibraryInfo *LibInfo) const override;

    /// Returns true if an argument of type Ty needs to be passed in a
    /// contiguous block of registers in calling convention CallConv.
    bool functionArgumentNeedsConsecutiveRegisters(
        Type *Ty, CallingConv::ID CallConv, bool isVarArg,
        const DataLayout &DL) const override {
      // We support any array type as "consecutive" block in the parameter
      // save area.  The element type defines the alignment requirement and
      // whether the argument should go in GPRs, FPRs, or VRs if available.
      //
      // Note that clang uses this capability both to implement the ELFv2
      // homogeneous float/vector aggregate ABI, and to avoid having to use
      // "byval" when passing aggregates that might fully fit in registers.
      return Ty->isArrayTy();
    }

    /// If a physical register, this returns the register that receives the
    /// exception address on entry to an EH pad.
    Register
    getExceptionPointerRegister(const Constant *PersonalityFn) const override;

    /// If a physical register, this returns the register that receives the
    /// exception typeid on entry to a landing pad.
    Register
    getExceptionSelectorRegister(const Constant *PersonalityFn) const override;

    /// Override to support customized stack guard loading.
    bool useLoadStackGuardNode(const Module &M) const override;

    bool isFPImmLegal(const APFloat &Imm, EVT VT,
                      bool ForCodeSize) const override;

    unsigned getJumpTableEncoding() const override;
    bool isJumpTableRelative() const override;
    SDValue getPICJumpTableRelocBase(SDValue Table,
                                     SelectionDAG &DAG) const override;
    const MCExpr *getPICJumpTableRelocBaseExpr(const MachineFunction *MF,
                                               unsigned JTI,
                                               MCContext &Ctx) const override;

    /// SelectOptimalAddrMode - Based on a node N and it's Parent (a MemSDNode),
    /// compute the address flags of the node, get the optimal address mode
    /// based on the flags, and set the Base and Disp based on the address mode.
    PPC::AddrMode SelectOptimalAddrMode(const SDNode *Parent, SDValue N,
                                        SDValue &Disp, SDValue &Base,
                                        SelectionDAG &DAG,
                                        MaybeAlign Align) const;
    /// SelectForceXFormMode - Given the specified address, force it to be
    /// represented as an indexed [r+r] operation (an XForm instruction).
    PPC::AddrMode SelectForceXFormMode(SDValue N, SDValue &Disp, SDValue &Base,
                                       SelectionDAG &DAG) const;

    bool splitValueIntoRegisterParts(
        SelectionDAG & DAG, const SDLoc &DL, SDValue Val, SDValue *Parts,
        unsigned NumParts, MVT PartVT, std::optional<CallingConv::ID> CC)
        const override;
    /// Structure that collects some common arguments that get passed around
    /// between the functions for call lowering.
    struct CallFlags {
      const CallingConv::ID CallConv;
      const bool IsTailCall : 1;
      const bool IsVarArg : 1;
      const bool IsPatchPoint : 1;
      const bool IsIndirect : 1;
      const bool HasNest : 1;
      const bool NoMerge : 1;

      CallFlags(CallingConv::ID CC, bool IsTailCall, bool IsVarArg,
                bool IsPatchPoint, bool IsIndirect, bool HasNest, bool NoMerge)
          : CallConv(CC), IsTailCall(IsTailCall), IsVarArg(IsVarArg),
            IsPatchPoint(IsPatchPoint), IsIndirect(IsIndirect),
            HasNest(HasNest), NoMerge(NoMerge) {}
    };

    CCAssignFn *ccAssignFnForCall(CallingConv::ID CC, bool Return,
                                  bool IsVarArg) const;
    bool supportsTailCallFor(const CallBase *CB) const;

    bool hasMultipleConditionRegisters(EVT VT) const override;

  private:
    struct ReuseLoadInfo {
      SDValue Ptr;
      SDValue Chain;
      SDValue ResChain;
      MachinePointerInfo MPI;
      bool IsDereferenceable = false;
      bool IsInvariant = false;
      Align Alignment;
      AAMDNodes AAInfo;
      const MDNode *Ranges = nullptr;

      ReuseLoadInfo() = default;

      MachineMemOperand::Flags MMOFlags() const {
        MachineMemOperand::Flags F = MachineMemOperand::MONone;
        if (IsDereferenceable)
          F |= MachineMemOperand::MODereferenceable;
        if (IsInvariant)
          F |= MachineMemOperand::MOInvariant;
        return F;
      }
    };

    // Map that relates a set of common address flags to PPC addressing modes.
    std::map<PPC::AddrMode, SmallVector<unsigned, 16>> AddrModesMap;
    void initializeAddrModeMap();

    bool canReuseLoadAddress(SDValue Op, EVT MemVT, ReuseLoadInfo &RLI,
                             SelectionDAG &DAG,
                             ISD::LoadExtType ET = ISD::NON_EXTLOAD) const;

    void LowerFP_TO_INTForReuse(SDValue Op, ReuseLoadInfo &RLI,
                                SelectionDAG &DAG, const SDLoc &dl) const;
    SDValue LowerFP_TO_INTDirectMove(SDValue Op, SelectionDAG &DAG,
                                     const SDLoc &dl) const;

    bool directMoveIsProfitable(const SDValue &Op) const;
    SDValue LowerINT_TO_FPDirectMove(SDValue Op, SelectionDAG &DAG,
                                     const SDLoc &dl) const;

    SDValue LowerINT_TO_FPVector(SDValue Op, SelectionDAG &DAG,
                                 const SDLoc &dl) const;

    SDValue LowerTRUNCATEVector(SDValue Op, SelectionDAG &DAG) const;

    SDValue getFramePointerFrameIndex(SelectionDAG & DAG) const;
    SDValue getReturnAddrFrameIndex(SelectionDAG & DAG) const;

    bool IsEligibleForTailCallOptimization(
        const GlobalValue *CalleeGV, CallingConv::ID CalleeCC,
        CallingConv::ID CallerCC, bool isVarArg,
        const SmallVectorImpl<ISD::InputArg> &Ins) const;

    bool IsEligibleForTailCallOptimization_64SVR4(
        const GlobalValue *CalleeGV, CallingConv::ID CalleeCC,
        CallingConv::ID CallerCC, const CallBase *CB, bool isVarArg,
        const SmallVectorImpl<ISD::OutputArg> &Outs,
        const SmallVectorImpl<ISD::InputArg> &Ins, const Function *CallerFunc,
        bool isCalleeExternalSymbol) const;

    bool isEligibleForTCO(const GlobalValue *CalleeGV, CallingConv::ID CalleeCC,
                          CallingConv::ID CallerCC, const CallBase *CB,
                          bool isVarArg,
                          const SmallVectorImpl<ISD::OutputArg> &Outs,
                          const SmallVectorImpl<ISD::InputArg> &Ins,
                          const Function *CallerFunc,
                          bool isCalleeExternalSymbol) const;

    SDValue EmitTailCallLoadFPAndRetAddr(SelectionDAG &DAG, int SPDiff,
                                         SDValue Chain, SDValue &LROpOut,
                                         SDValue &FPOpOut,
                                         const SDLoc &dl) const;

    SDValue getTOCEntry(SelectionDAG &DAG, const SDLoc &dl, SDValue GA) const;

    SDValue LowerRETURNADDR(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerFRAMEADDR(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerConstantPool(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerBlockAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerGlobalTLSAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerGlobalTLSAddressAIX(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerGlobalTLSAddressLinux(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerJumpTable(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSETCC(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSSUBO(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSADDO(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerINIT_TRAMPOLINE(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerADJUST_TRAMPOLINE(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerINLINEASM(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerVASTART(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerVAARG(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerVACOPY(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSTACKRESTORE(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerGET_DYNAMIC_AREA_OFFSET(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerDYNAMIC_STACKALLOC(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerEH_DWARF_CFA(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerLOAD(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSTORE(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerTRUNCATE(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerFP_TO_INT(SDValue Op, SelectionDAG &DAG,
                           const SDLoc &dl) const;
    SDValue LowerINT_TO_FP(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerGET_ROUNDING(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSET_ROUNDING(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSHL_PARTS(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSRL_PARTS(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSRA_PARTS(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerFunnelShift(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerBUILD_VECTOR(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerVECTOR_SHUFFLE(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerVPERM(SDValue Op, SelectionDAG &DAG, ArrayRef<int> PermMask,
                       EVT VT, SDValue V1, SDValue V2) const;
    SDValue LowerINSERT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerINTRINSIC_WO_CHAIN(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerINTRINSIC_VOID(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerBSWAP(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerATOMIC_CMP_SWAP(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerIS_FPCLASS(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerADDSUBO_CARRY(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerADDSUBO(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerUCMP(SDValue Op, SelectionDAG &DAG) const;
    SDValue lowerToLibCall(const char *LibCallName, SDValue Op,
                           SelectionDAG &DAG) const;
    SDValue lowerLibCallBasedOnType(const char *LibCallFloatName,
                                    const char *LibCallDoubleName, SDValue Op,
                                    SelectionDAG &DAG) const;
    bool isLowringToMASSFiniteSafe(SDValue Op) const;
    bool isLowringToMASSSafe(SDValue Op) const;
    bool isScalarMASSConversionEnabled() const;
    SDValue lowerLibCallBase(const char *LibCallDoubleName,
                             const char *LibCallFloatName,
                             const char *LibCallDoubleNameFinite,
                             const char *LibCallFloatNameFinite, SDValue Op,
                             SelectionDAG &DAG) const;
    SDValue lowerPow(SDValue Op, SelectionDAG &DAG) const;
    SDValue lowerSin(SDValue Op, SelectionDAG &DAG) const;
    SDValue lowerCos(SDValue Op, SelectionDAG &DAG) const;
    SDValue lowerLog(SDValue Op, SelectionDAG &DAG) const;
    SDValue lowerLog10(SDValue Op, SelectionDAG &DAG) const;
    SDValue lowerExp(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerATOMIC_LOAD_STORE(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSCALAR_TO_VECTOR(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerMUL(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerFP_EXTEND(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerFP_ROUND(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerROTL(SDValue Op, SelectionDAG &DAG) const;

    SDValue LowerVP_LOAD(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerVP_STORE(SDValue Op, SelectionDAG &DAG) const;

    SDValue LowerVectorLoad(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerVectorStore(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerDMFVectorLoad(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerDMFVectorStore(SDValue Op, SelectionDAG &DAG) const;
    SDValue DMFInsert1024(const SmallVectorImpl<SDValue> &Pairs,
                          const SDLoc &dl, SelectionDAG &DAG) const;

    SDValue LowerCallResult(SDValue Chain, SDValue InGlue,
                            CallingConv::ID CallConv, bool isVarArg,
                            const SmallVectorImpl<ISD::InputArg> &Ins,
                            const SDLoc &dl, SelectionDAG &DAG,
                            SmallVectorImpl<SDValue> &InVals) const;

    SDValue FinishCall(CallFlags CFlags, const SDLoc &dl, SelectionDAG &DAG,
                       SmallVector<std::pair<unsigned, SDValue>, 8> &RegsToPass,
                       SDValue InGlue, SDValue Chain, SDValue CallSeqStart,
                       SDValue &Callee, int SPDiff, unsigned NumBytes,
                       const SmallVectorImpl<ISD::InputArg> &Ins,
                       SmallVectorImpl<SDValue> &InVals,
                       const CallBase *CB) const;

    SDValue
    LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
                         const SmallVectorImpl<ISD::InputArg> &Ins,
                         const SDLoc &dl, SelectionDAG &DAG,
                         SmallVectorImpl<SDValue> &InVals) const override;

    SDValue LowerCall(TargetLowering::CallLoweringInfo &CLI,
                      SmallVectorImpl<SDValue> &InVals) const override;

    bool CanLowerReturn(CallingConv::ID CallConv, MachineFunction &MF,
                        bool isVarArg,
                        const SmallVectorImpl<ISD::OutputArg> &Outs,
                        LLVMContext &Context, const Type *RetTy) const override;

    SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
                        const SmallVectorImpl<ISD::OutputArg> &Outs,
                        const SmallVectorImpl<SDValue> &OutVals,
                        const SDLoc &dl, SelectionDAG &DAG) const override;

    SDValue extendArgForPPC64(ISD::ArgFlagsTy Flags, EVT ObjectVT,
                              SelectionDAG &DAG, SDValue ArgVal,
                              const SDLoc &dl) const;

    SDValue LowerFormalArguments_AIX(
        SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
        const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &dl,
        SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const;
    SDValue LowerFormalArguments_64SVR4(
        SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
        const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &dl,
        SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const;
    SDValue LowerFormalArguments_32SVR4(
        SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
        const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &dl,
        SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const;

    SDValue createMemcpyOutsideCallSeq(SDValue Arg, SDValue PtrOff,
                                       SDValue CallSeqStart,
                                       ISD::ArgFlagsTy Flags, SelectionDAG &DAG,
                                       const SDLoc &dl) const;

    SDValue LowerCall_64SVR4(SDValue Chain, SDValue Callee, CallFlags CFlags,
                             const SmallVectorImpl<ISD::OutputArg> &Outs,
                             const SmallVectorImpl<SDValue> &OutVals,
                             const SmallVectorImpl<ISD::InputArg> &Ins,
                             const SDLoc &dl, SelectionDAG &DAG,
                             SmallVectorImpl<SDValue> &InVals,
                             const CallBase *CB) const;
    SDValue LowerCall_32SVR4(SDValue Chain, SDValue Callee, CallFlags CFlags,
                             const SmallVectorImpl<ISD::OutputArg> &Outs,
                             const SmallVectorImpl<SDValue> &OutVals,
                             const SmallVectorImpl<ISD::InputArg> &Ins,
                             const SDLoc &dl, SelectionDAG &DAG,
                             SmallVectorImpl<SDValue> &InVals,
                             const CallBase *CB) const;
    SDValue LowerCall_AIX(SDValue Chain, SDValue Callee, CallFlags CFlags,
                          const SmallVectorImpl<ISD::OutputArg> &Outs,
                          const SmallVectorImpl<SDValue> &OutVals,
                          const SmallVectorImpl<ISD::InputArg> &Ins,
                          const SDLoc &dl, SelectionDAG &DAG,
                          SmallVectorImpl<SDValue> &InVals,
                          const CallBase *CB) const;

    SDValue lowerEH_SJLJ_SETJMP(SDValue Op, SelectionDAG &DAG) const;
    SDValue lowerEH_SJLJ_LONGJMP(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerBITCAST(SDValue Op, SelectionDAG &DAG) const;

    SDValue DAGCombineExtBoolTrunc(SDNode *N, DAGCombinerInfo &DCI) const;
    SDValue DAGCombineBuildVector(SDNode *N, DAGCombinerInfo &DCI) const;
    SDValue DAGCombineTruncBoolExt(SDNode *N, DAGCombinerInfo &DCI) const;
    SDValue combineStoreFPToInt(SDNode *N, DAGCombinerInfo &DCI) const;
    SDValue combineFPToIntToFP(SDNode *N, DAGCombinerInfo &DCI) const;
    SDValue combineSHL(SDNode *N, DAGCombinerInfo &DCI) const;
    SDValue combineVectorShift(SDNode *N, DAGCombinerInfo &DCI) const;
    SDValue combineSRA(SDNode *N, DAGCombinerInfo &DCI) const;
    SDValue combineSRL(SDNode *N, DAGCombinerInfo &DCI) const;
    SDValue combineMUL(SDNode *N, DAGCombinerInfo &DCI) const;
    SDValue combineADD(SDNode *N, DAGCombinerInfo &DCI) const;
    SDValue combineFMALike(SDNode *N, DAGCombinerInfo &DCI) const;
    SDValue combineTRUNCATE(SDNode *N, DAGCombinerInfo &DCI) const;
    SDValue combineSetCC(SDNode *N, DAGCombinerInfo &DCI) const;
    SDValue combineVectorShuffle(ShuffleVectorSDNode *SVN,
                                 SelectionDAG &DAG) const;
    SDValue combineVReverseMemOP(ShuffleVectorSDNode *SVN, LSBaseSDNode *LSBase,
                                 DAGCombinerInfo &DCI) const;

    /// ConvertSETCCToSubtract - looks at SETCC that compares ints. It replaces
    /// SETCC with integer subtraction when (1) there is a legal way of doing it
    /// (2) keeping the result of comparison in GPR has performance benefit.
    SDValue ConvertSETCCToSubtract(SDNode *N, DAGCombinerInfo &DCI) const;

    SDValue getSqrtEstimate(SDValue Operand, SelectionDAG &DAG, int Enabled,
                            int &RefinementSteps, bool &UseOneConstNR,
                            bool Reciprocal) const override;
    SDValue getRecipEstimate(SDValue Operand, SelectionDAG &DAG, int Enabled,
                             int &RefinementSteps) const override;
    SDValue getSqrtInputTest(SDValue Operand, SelectionDAG &DAG,
                             const DenormalMode &Mode) const override;
    SDValue getSqrtResultForDenormInput(SDValue Operand,
                                        SelectionDAG &DAG) const override;
    unsigned combineRepeatedFPDivisors() const override;

    SDValue
    combineElementTruncationToVectorTruncation(SDNode *N,
                                               DAGCombinerInfo &DCI) const;

    SDValue combineBVLoadsSpecialValue(SDValue Operand,
                                       SelectionDAG &DAG) const;

    /// lowerToVINSERTH - Return the SDValue if this VECTOR_SHUFFLE can be
    /// handled by the VINSERTH instruction introduced in ISA 3.0. This is
    /// essentially any shuffle of v8i16 vectors that just inserts one element
    /// from one vector into the other.
    SDValue lowerToVINSERTH(ShuffleVectorSDNode *N, SelectionDAG &DAG) const;

    /// lowerToVINSERTB - Return the SDValue if this VECTOR_SHUFFLE can be
    /// handled by the VINSERTB instruction introduced in ISA 3.0. This is
    /// essentially v16i8 vector version of VINSERTH.
    SDValue lowerToVINSERTB(ShuffleVectorSDNode *N, SelectionDAG &DAG) const;

    /// lowerToXXSPLTI32DX - Return the SDValue if this VECTOR_SHUFFLE can be
    /// handled by the XXSPLTI32DX instruction introduced in ISA 3.1.
    SDValue lowerToXXSPLTI32DX(ShuffleVectorSDNode *N, SelectionDAG &DAG) const;

    // Return whether the call instruction can potentially be optimized to a
    // tail call. This will cause the optimizers to attempt to move, or
    // duplicate return instructions to help enable tail call optimizations.
    bool mayBeEmittedAsTailCall(const CallInst *CI) const override;
    bool isMaskAndCmp0FoldingBeneficial(const Instruction &AndI) const override;

    /// getAddrModeForFlags - Based on the set of address flags, select the most
    /// optimal instruction format to match by.
    PPC::AddrMode getAddrModeForFlags(unsigned Flags) const;

    /// computeMOFlags - Given a node N and it's Parent (a MemSDNode), compute
    /// the address flags of the load/store instruction that is to be matched.
    /// The address flags are stored in a map, which is then searched
    /// through to determine the optimal load/store instruction format.
    unsigned computeMOFlags(const SDNode *Parent, SDValue N,
                            SelectionDAG &DAG) const;
  }; // end class PPCTargetLowering

  namespace PPC {

    FastISel *createFastISel(FunctionLoweringInfo &FuncInfo,
                             const TargetLibraryInfo *LibInfo);

  } // end namespace PPC

  bool isIntS16Immediate(SDNode *N, int16_t &Imm);
  bool isIntS16Immediate(SDValue Op, int16_t &Imm);
  bool isIntS34Immediate(SDNode *N, int64_t &Imm);
  bool isIntS34Immediate(SDValue Op, int64_t &Imm);

  bool convertToNonDenormSingle(APInt &ArgAPInt);
  bool convertToNonDenormSingle(APFloat &ArgAPFloat);
  bool checkConvertToNonDenormSingle(APFloat &ArgAPFloat);

} // end namespace llvm

#endif // LLVM_LIB_TARGET_POWERPC_PPCISELLOWERING_H
