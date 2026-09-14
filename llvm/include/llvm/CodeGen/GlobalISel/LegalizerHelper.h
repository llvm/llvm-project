//== llvm/CodeGen/GlobalISel/LegalizerHelper.h ---------------- -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file A pass to convert the target-illegal operations created by IR -> MIR
/// translation into ones the target expects to be able to select. This may
/// occur in multiple phases, for example G_ADD <2 x i8> -> G_ADD <2 x i16> ->
/// G_ADD <4 x i16>.
///
/// The LegalizerHelper class is where most of the work happens, and is
/// designed to be callable from other passes that find themselves with an
/// illegal instruction.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_LEGALIZERHELPER_H
#define LLVM_CODEGEN_GLOBALISEL_LEGALIZERHELPER_H

#include "llvm/CodeGen/GlobalISel/CallLowering.h"
#include "llvm/CodeGen/GlobalISel/GISelValueTracking.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/RuntimeLibcallUtil.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
// Forward declarations.
class APInt;
class GAnyLoad;
class GLoadStore;
class GStore;
class GenericMachineInstr;
class MachineFunction;
class MachineIRBuilder;
class MachineInstr;
class MachineInstrBuilder;
struct MachinePointerInfo;
template <typename T> class SmallVectorImpl;
class LegalizerInfo;
class MachineRegisterInfo;
class GISelChangeObserver;
class LostDebugLocObserver;
class TargetLowering;

class LegalizerHelper {
public:
  /// Expose MIRBuilder so clients can set their own RecordInsertInstruction
  /// functions
  MachineIRBuilder &MIRBuilder;

  /// To keep track of changes made by the LegalizerHelper.
  GISelChangeObserver &Observer;

private:
  MachineRegisterInfo &MRI;
  const LegalizerInfo &LI;
  const TargetLowering &TLI;

  const LibcallLoweringInfo *Libcalls = nullptr;
  GISelValueTracking *VT = nullptr;

public:
  enum LegalizeResult {
    /// Instruction was already legal and no change was made to the
    /// MachineFunction.
    AlreadyLegal,

    /// Instruction has been legalized and the MachineFunction changed.
    Legalized,

    /// Some kind of error has occurred and we could not legalize this
    /// instruction.
    UnableToLegalize,
  };

  /// Expose LegalizerInfo so the clients can re-use.
  const LegalizerInfo &getLegalizerInfo() const { return LI; }
  const TargetLowering &getTargetLowering() const { return TLI; }
  const LibcallLoweringInfo *getLibcallLoweringInfo() { return Libcalls; }
  GISelValueTracking *getValueTracking() const { return VT; }

  // FIXME: Should probably make Libcalls mandatory
  LLVM_ABI_NOT_EXPORTED LegalizerHelper(MachineFunction &MF, GISelChangeObserver &Observer,
                           MachineIRBuilder &B,
                           const LibcallLoweringInfo *Libcalls = nullptr);
  LLVM_ABI_NOT_EXPORTED LegalizerHelper(MachineFunction &MF, const LegalizerInfo &LI,
                           GISelChangeObserver &Observer, MachineIRBuilder &B,
                           const LibcallLoweringInfo *Libcalls = nullptr,
                           GISelValueTracking *VT = nullptr);

  /// Replace \p MI by a sequence of legal instructions that can implement the
  /// same operation. Note that this means \p MI may be deleted, so any iterator
  /// steps should be performed before calling this function. \p Helper should
  /// be initialized to the MachineFunction containing \p MI.
  ///
  /// Considered as an opaque blob, the legal code will use and define the same
  /// registers as \p MI.
  LLVM_ABI_NOT_EXPORTED LegalizeResult legalizeInstrStep(MachineInstr &MI,
                                            LostDebugLocObserver &LocObserver);

  /// Legalize an instruction by emiting a runtime library call instead.
  LLVM_ABI_NOT_EXPORTED LegalizeResult libcall(MachineInstr &MI,
                                  LostDebugLocObserver &LocObserver);

  /// Legalize an instruction by reducing the width of the underlying scalar
  /// type.
  LLVM_ABI_NOT_EXPORTED LegalizeResult narrowScalar(MachineInstr &MI, unsigned TypeIdx,
                                       LLT NarrowTy);

  /// Legalize an instruction by performing the operation on a wider scalar type
  /// (for example a 16-bit addition can be safely performed at 32-bits
  /// precision, ignoring the unused bits).
  LLVM_ABI_NOT_EXPORTED LegalizeResult widenScalar(MachineInstr &MI, unsigned TypeIdx,
                                      LLT WideTy);

  /// Legalize an instruction by replacing the value type
  LLVM_ABI_NOT_EXPORTED LegalizeResult bitcast(MachineInstr &MI, unsigned TypeIdx, LLT Ty);

  /// Legalize an instruction by splitting it into simpler parts, hopefully
  /// understood by the target.
  LLVM_ABI_NOT_EXPORTED LegalizeResult lower(MachineInstr &MI, unsigned TypeIdx, LLT Ty);

  /// Legalize a vector instruction by splitting into multiple components, each
  /// acting on the same scalar type as the original but with fewer elements.
  LLVM_ABI_NOT_EXPORTED LegalizeResult fewerElementsVector(MachineInstr &MI,
                                              unsigned TypeIdx, LLT NarrowTy);

  /// Legalize a vector instruction by increasing the number of vector elements
  /// involved and ignoring the added elements later.
  LLVM_ABI_NOT_EXPORTED LegalizeResult moreElementsVector(MachineInstr &MI, unsigned TypeIdx,
                                             LLT MoreTy);

  /// Cast the given value to an LLT::integer with an equivalent size. Returns
  /// the register to use if an instruction was inserted. Returns the original
  /// register if no coercion was necessary.
  //
  // This may also fail and return Register() if there is no legal way to cast.
  LLVM_ABI_NOT_EXPORTED Register coerceToInteger(Register Val);

  /// Legalize a single operand \p OpIdx of the machine instruction \p MI as a
  /// Use by extending the operand's type to \p WideTy using the specified \p
  /// ExtOpcode for the extension instruction, and replacing the vreg of the
  /// operand in place.
  LLVM_ABI_NOT_EXPORTED void widenScalarSrc(MachineInstr &MI, LLT WideTy, unsigned OpIdx,
                               unsigned ExtOpcode);

  /// Legalize a single operand \p OpIdx of the machine instruction \p MI as a
  /// Use by extending the operand's type to \p WideTy using the G_FPEXT for the
  /// extension instruction, and replacing the vreg of the operand in place.
  /// Flags are copied from MI to the new extend.
  LLVM_ABI_NOT_EXPORTED void widenScalarSrcUsingFPExt(MachineInstr &MI, LLT WideTy,
                                         unsigned OpIdx);

  /// Legalize a single operand \p OpIdx of the machine instruction \p MI as a
  /// Use by truncating the operand's type to \p NarrowTy using G_TRUNC, and
  /// replacing the vreg of the operand in place.
  LLVM_ABI_NOT_EXPORTED void narrowScalarSrc(MachineInstr &MI, LLT NarrowTy, unsigned OpIdx);

  /// Legalize a single operand \p OpIdx of the machine instruction \p MI as a
  /// Def by extending the operand's type to \p WideTy and truncating it back
  /// with the \p TruncOpcode, and replacing the vreg of the operand in place.
  LLVM_ABI_NOT_EXPORTED void widenScalarDst(MachineInstr &MI, LLT WideTy, unsigned OpIdx = 0,
                               unsigned TruncOpcode = TargetOpcode::G_TRUNC);

  /// Legalize a single operand \p OpIdx of the machine instruction \p MI as a
  /// Def by extending the operand's type to \p WideTy and truncating it back
  /// with G_FPTRUNC, and replacing the vreg of the operand in place. Flags are
  /// copied from MI to the new trunc.
  LLVM_ABI_NOT_EXPORTED void widenScalarDstUsingFPTrunc(MachineInstr &MI, LLT WideTy,
                                           unsigned OpIdx = 0);

  // Legalize a single operand \p OpIdx of the machine instruction \p MI as a
  // Def by truncating the operand's type to \p NarrowTy, replacing in place and
  // extending back with \p ExtOpcode.
  LLVM_ABI_NOT_EXPORTED void narrowScalarDst(MachineInstr &MI, LLT NarrowTy, unsigned OpIdx,
                                unsigned ExtOpcode);
  /// Legalize a single operand \p OpIdx of the machine instruction \p MI as a
  /// Def by performing it with additional vector elements and extracting the
  /// result elements, and replacing the vreg of the operand in place.
  LLVM_ABI_NOT_EXPORTED void moreElementsVectorDst(MachineInstr &MI, LLT MoreTy,
                                      unsigned OpIdx);

  /// Legalize a single operand \p OpIdx of the machine instruction \p MI as a
  /// Use by producing a vector with undefined high elements, extracting the
  /// original vector type, and replacing the vreg of the operand in place.
  LLVM_ABI_NOT_EXPORTED void moreElementsVectorSrc(MachineInstr &MI, LLT MoreTy,
                                      unsigned OpIdx);

  /// Legalize a single operand \p OpIdx of the machine instruction \p MI as a
  /// use by inserting a G_BITCAST to \p CastTy
  LLVM_ABI_NOT_EXPORTED void bitcastSrc(MachineInstr &MI, LLT CastTy, unsigned OpIdx);

  /// Legalize a single operand \p OpIdx of the machine instruction \p MI as a
  /// def by inserting a G_BITCAST from \p CastTy
  LLVM_ABI_NOT_EXPORTED void bitcastDst(MachineInstr &MI, LLT CastTy, unsigned OpIdx);

  // Useful for libcalls where all operands have the same type.
  LLVM_ABI_NOT_EXPORTED LegalizeResult
  simpleLibcall(MachineInstr &MI, MachineIRBuilder &MIRBuilder, unsigned Size,
                Type *OpType, LostDebugLocObserver &LocObserver) const;

  /// Helper function that creates a libcall to the given \p Name using the
  /// given calling convention \p CC.
  LLVM_ABI_NOT_EXPORTED LegalizeResult createLibcall(const char *Name,
                                        const CallLowering::ArgInfo &Result,
                                        ArrayRef<CallLowering::ArgInfo> Args,
                                        CallingConv::ID CC,
                                        LostDebugLocObserver &LocObserver,
                                        MachineInstr *MI = nullptr) const;

  /// Helper function that creates the given libcall.
  LLVM_ABI_NOT_EXPORTED LegalizeResult createLibcall(RTLIB::Libcall Libcall,
                                        const CallLowering::ArgInfo &Result,
                                        ArrayRef<CallLowering::ArgInfo> Args,
                                        LostDebugLocObserver &LocObserver,
                                        MachineInstr *MI = nullptr) const;

  LLVM_ABI_NOT_EXPORTED LegalizeResult conversionLibcall(MachineInstr &MI, Type *ToType,
                                            Type *FromType,
                                            LostDebugLocObserver &LocObserver,
                                            bool IsSigned = false) const;
  LLVM_ABI_NOT_EXPORTED LegalizerHelper::LegalizeResult
  createAtomicLibcall(MachineInstr &MI) const;

  /// Create a libcall to memcpy et al.
  LLVM_ABI_NOT_EXPORTED LegalizeResult
  createMemLibcall(MachineRegisterInfo &MRI, MachineInstr &MI,
                   LostDebugLocObserver &LocObserver) const;

private:
  LegalizeResult
  widenScalarMergeValues(MachineInstr &MI, unsigned TypeIdx, LLT WideTy);
  LegalizeResult
  widenScalarUnmergeValues(MachineInstr &MI, unsigned TypeIdx, LLT WideTy);
  LegalizeResult
  widenScalarExtract(MachineInstr &MI, unsigned TypeIdx, LLT WideTy);
  LegalizeResult
  widenScalarInsert(MachineInstr &MI, unsigned TypeIdx, LLT WideTy);
  LegalizeResult widenScalarAddSubOverflow(MachineInstr &MI, unsigned TypeIdx,
                                           LLT WideTy);
  LegalizeResult widenScalarAddSubShlSat(MachineInstr &MI, unsigned TypeIdx,
                                         LLT WideTy);
  LegalizeResult widenScalarMulo(MachineInstr &MI, unsigned TypeIdx,
                                 LLT WideTy);

  /// Helper function to build a wide generic register \p DstReg of type \p
  /// RegTy from smaller parts. This will produce a G_MERGE_VALUES,
  /// G_BUILD_VECTOR, G_CONCAT_VECTORS, or sequence of G_INSERT as appropriate
  /// for the types.
  ///
  /// \p PartRegs must be registers of type \p PartTy.
  ///
  /// If \p ResultTy does not evenly break into \p PartTy sized pieces, the
  /// remainder must be specified with \p LeftoverRegs of type \p LeftoverTy.
  void insertParts(Register DstReg, LLT ResultTy,
                   LLT PartTy, ArrayRef<Register> PartRegs,
                   LLT LeftoverTy = LLT(), ArrayRef<Register> LeftoverRegs = {});

  /// Merge \p PartRegs with different types into \p DstReg.
  void mergeMixedSubvectors(Register DstReg, ArrayRef<Register> PartRegs);

  void appendVectorElts(SmallVectorImpl<Register> &Elts, Register Reg);

  /// Unmerge \p SrcReg into smaller sized values, and append them to \p
  /// Parts. The elements of \p Parts will be the greatest common divisor type
  /// of \p DstTy, \p NarrowTy and the type of \p SrcReg. This will compute and
  /// return the GCD type.
  LLT extractGCDType(SmallVectorImpl<Register> &Parts, LLT DstTy,
                     LLT NarrowTy, Register SrcReg);

  /// Unmerge \p SrcReg into \p GCDTy typed registers. This will append all of
  /// the unpacked registers to \p Parts. This version is if the common unmerge
  /// type is already known.
  void extractGCDType(SmallVectorImpl<Register> &Parts, LLT GCDTy,
                      Register SrcReg);

  /// Produce a merge of values in \p VRegs to define \p DstReg. Perform a merge
  /// from the least common multiple type, and convert as appropriate to \p
  /// DstReg.
  ///
  /// \p VRegs should each have type \p GCDTy. This type should be greatest
  /// common divisor type of \p DstReg, \p NarrowTy, and an undetermined source
  /// type.
  ///
  /// \p NarrowTy is the desired result merge source type. If the source value
  /// needs to be widened to evenly cover \p DstReg, inserts high bits
  /// corresponding to the extension opcode \p PadStrategy.
  ///
  /// \p VRegs will be cleared, and the result \p NarrowTy register pieces
  /// will replace it. Returns The complete LCMTy that \p VRegs will cover when
  /// merged.
  LLT buildLCMMergePieces(LLT DstTy, LLT NarrowTy, LLT GCDTy,
                          SmallVectorImpl<Register> &VRegs,
                          unsigned PadStrategy = TargetOpcode::G_ANYEXT);

  /// Merge the values in \p RemergeRegs to an \p LCMTy typed value. Extract the
  /// low bits into \p DstReg. This is intended to use the outputs from
  /// buildLCMMergePieces after processing.
  void buildWidenedRemergeToDst(Register DstReg, LLT LCMTy,
                                ArrayRef<Register> RemergeRegs);

  /// Perform generic multiplication of values held in multiple registers.
  /// Generated instructions use only types NarrowTy and i1.
  /// Destination can be same or two times size of the source.
  void multiplyRegisters(SmallVectorImpl<Register> &DstRegs,
                         ArrayRef<Register> Src1Regs,
                         ArrayRef<Register> Src2Regs, LLT NarrowTy);

  void changeOpcode(MachineInstr &MI, unsigned NewOpcode);

  LegalizeResult tryNarrowPow2Reduction(MachineInstr &MI, Register SrcReg,
                                        LLT SrcTy, LLT NarrowTy,
                                        unsigned ScalarOpc);

  // Memcpy family legalization helpers.
  LegalizeResult lowerMemset(MachineInstr &MI, Register Dst, Register Val,
                             uint64_t KnownLen, Align Alignment,
                             bool DstAlignCanChange, ArrayRef<LLT> MemOps);
  LegalizeResult lowerMemcpy(MachineInstr &MI, Register Dst, Register Src,
                             uint64_t KnownLen, Align Alignment,
                             bool DstAlignCanChange, ArrayRef<LLT> MemOps);
  LegalizeResult lowerMemmove(MachineInstr &MI, Register Dst, Register Src,
                              uint64_t KnownLen, Align Alignment,
                              bool DstAlignCanChange, ArrayRef<LLT> MemOps);

  // Implements floating-point environment read/write via library function call.
  LegalizeResult createGetStateLibcall(MachineInstr &MI,
                                       LostDebugLocObserver &LocObserver);
  LegalizeResult createSetStateLibcall(MachineInstr &MI,
                                       LostDebugLocObserver &LocObserver);
  LegalizeResult createResetStateLibcall(MachineInstr &MI,
                                         LostDebugLocObserver &LocObserver);
  LegalizeResult createFCMPLibcall(MachineInstr &MI,
                                   LostDebugLocObserver &LocObserver);

  MachineInstrBuilder
  getNeutralElementForVecReduce(unsigned Opcode, MachineIRBuilder &MIRBuilder,
                                LLT Ty);

  LegalizeResult emitSincosLibcall(MachineInstr &MI,
                                   MachineIRBuilder &MIRBuilder, unsigned Size,
                                   Type *OpType,
                                   LostDebugLocObserver &LocObserver);

  LegalizeResult emitModfLibcall(MachineInstr &MI, MachineIRBuilder &MIRBuilder,
                                 unsigned Size, Type *OpType,
                                 LostDebugLocObserver &LocObserver);

public:
  /// Return the alignment to use for a stack temporary object with the given
  /// type.
  LLVM_ABI_NOT_EXPORTED Align getStackTemporaryAlignment(LLT Type,
                                            Align MinAlign = Align()) const;

  /// Create a stack temporary based on the size in bytes and the alignment
  LLVM_ABI_NOT_EXPORTED MachineInstrBuilder createStackTemporary(
      TypeSize Bytes, Align Alignment, MachinePointerInfo &PtrInfo);

  /// Create a store of \p Val to a stack temporary and return a load as the
  /// same type as \p Res.
  LLVM_ABI_NOT_EXPORTED MachineInstrBuilder createStackStoreLoad(const DstOp &Res,
                                                    const SrcOp &Val);

  /// Given a store of a boolean vector, scalarize it.
  LLVM_ABI_NOT_EXPORTED LegalizeResult scalarizeVectorBooleanStore(GStore &MI);

  /// Get a pointer to vector element \p Index located in memory for a vector of
  /// type \p VecTy starting at a base address of \p VecPtr. If \p Index is out
  /// of bounds the returned pointer is unspecified, but will be within the
  /// vector bounds.
  LLVM_ABI_NOT_EXPORTED Register getVectorElementPointer(Register VecPtr, LLT VecTy,
                                            Register Index);

  /// Handles most opcodes. Split \p MI into same instruction on sub-vectors or
  /// scalars with \p NumElts elements (1 for scalar). Supports uneven splits:
  /// there can be leftover sub-vector with fewer then \p NumElts or a leftover
  /// scalar. To avoid this use moreElements first and set MI number of elements
  /// to multiple of \p NumElts. Non-vector operands that should be used on all
  /// sub-instructions without split are listed in \p NonVecOpIndices.
  LLVM_ABI_NOT_EXPORTED LegalizeResult fewerElementsVectorMultiEltType(
      GenericMachineInstr &MI, unsigned NumElts,
      std::initializer_list<unsigned> NonVecOpIndices = {});

  LLVM_ABI_NOT_EXPORTED LegalizeResult fewerElementsVectorPhi(GenericMachineInstr &MI,
                                                 unsigned NumElts);

  LLVM_ABI_NOT_EXPORTED LegalizeResult moreElementsVectorPhi(MachineInstr &MI,
                                                unsigned TypeIdx, LLT MoreTy);
  LLVM_ABI_NOT_EXPORTED LegalizeResult moreElementsVectorShuffle(MachineInstr &MI,
                                                    unsigned TypeIdx,
                                                    LLT MoreTy);

  LLVM_ABI_NOT_EXPORTED LegalizeResult fewerElementsVectorUnmergeValues(MachineInstr &MI,
                                                           unsigned TypeIdx,
                                                           LLT NarrowTy);
  LLVM_ABI_NOT_EXPORTED LegalizeResult fewerElementsVectorMerge(MachineInstr &MI,
                                                   unsigned TypeIdx,
                                                   LLT NarrowTy);
  LLVM_ABI_NOT_EXPORTED LegalizeResult fewerElementsVectorExtractInsertVectorElt(
      MachineInstr &MI, unsigned TypeIdx, LLT NarrowTy);

  /// Equalize source and destination vector sizes of G_SHUFFLE_VECTOR.
  LLVM_ABI_NOT_EXPORTED LegalizeResult equalizeVectorShuffleLengths(MachineInstr &MI);

  LLVM_ABI_NOT_EXPORTED LegalizeResult reduceLoadStoreWidth(GLoadStore &MI, unsigned TypeIdx,
                                               LLT NarrowTy);

  LLVM_ABI_NOT_EXPORTED LegalizeResult narrowScalarShiftByConstant(MachineInstr &MI,
                                                      const APInt &Amt,
                                                      LLT HalfTy,
                                                      LLT ShiftAmtTy);

  /// Multi-way shift legalization: directly split wide shifts into target-sized
  /// parts in a single step, avoiding recursive binary splitting.
  LLVM_ABI_NOT_EXPORTED LegalizeResult narrowScalarShiftMultiway(MachineInstr &MI,
                                                    LLT TargetTy);

  /// Optimized path for constant shift amounts using static indexing.
  /// Directly calculates which source parts contribute to each output part
  /// without generating runtime select chains.
  LLVM_ABI_NOT_EXPORTED LegalizeResult narrowScalarShiftByConstantMultiway(MachineInstr &MI,
                                                              const APInt &Amt,
                                                              LLT TargetTy,
                                                              LLT ShiftAmtTy);

  struct ShiftParams {
    Register WordShift;   // Number of complete words to shift
    Register BitShift;    // Number of bits to shift within words
    Register InvBitShift; // Complement bit shift (TargetBits - BitShift)
    Register Zero;        // Zero constant for SHL/LSHR fill
    Register SignBit;     // Sign extension value for ASHR fill
  };

  /// Generates a single output part for constant shifts using direct indexing.
  /// Calculates which source parts contribute and how they're combined.
  LLVM_ABI_NOT_EXPORTED Register buildConstantShiftPart(unsigned Opcode, unsigned PartIdx,
                                           unsigned NumParts,
                                           ArrayRef<Register> SrcParts,
                                           const ShiftParams &Params,
                                           LLT TargetTy, LLT ShiftAmtTy);

  /// Generates a shift part with carry for variable shifts.
  /// Combines main operand shifted by BitShift with carry bits from adjacent
  /// operand.
  LLVM_ABI_NOT_EXPORTED Register buildVariableShiftPart(unsigned Opcode,
                                           Register MainOperand,
                                           Register ShiftAmt, LLT TargetTy,
                                           Register CarryOperand = Register());

  LLVM_ABI_NOT_EXPORTED LegalizeResult fewerElementsVectorReductions(MachineInstr &MI,
                                                        unsigned TypeIdx,
                                                        LLT NarrowTy);
  LLVM_ABI_NOT_EXPORTED LegalizeResult fewerElementsVectorSeqReductions(MachineInstr &MI,
                                                           unsigned TypeIdx,
                                                           LLT NarrowTy);

  // Fewer Elements for bitcast, ensuring that the size of the Src and Dst
  // registers will be the same
  LLVM_ABI_NOT_EXPORTED LegalizeResult fewerElementsBitcast(MachineInstr &MI,
                                               unsigned TypeIdx, LLT NarrowTy);

  LLVM_ABI_NOT_EXPORTED LegalizeResult fewerElementsVectorShuffle(MachineInstr &MI,
                                                     unsigned TypeIdx,
                                                     LLT NarrowTy);

  LLVM_ABI_NOT_EXPORTED LegalizeResult narrowScalarShift(MachineInstr &MI, unsigned TypeIdx,
                                            LLT Ty);
  LLVM_ABI_NOT_EXPORTED LegalizeResult narrowScalarAddSub(MachineInstr &MI, unsigned TypeIdx,
                                             LLT NarrowTy);
  LLVM_ABI_NOT_EXPORTED LegalizeResult narrowScalarMul(MachineInstr &MI, LLT Ty);
  LLVM_ABI_NOT_EXPORTED LegalizeResult narrowScalarFPTOI(MachineInstr &MI, unsigned TypeIdx,
                                            LLT Ty);
  LLVM_ABI_NOT_EXPORTED LegalizeResult narrowScalarExtract(MachineInstr &MI,
                                              unsigned TypeIdx, LLT Ty);
  LLVM_ABI_NOT_EXPORTED LegalizeResult narrowScalarInsert(MachineInstr &MI, unsigned TypeIdx,
                                             LLT Ty);

  LLVM_ABI_NOT_EXPORTED LegalizeResult narrowScalarBasic(MachineInstr &MI, unsigned TypeIdx,
                                            LLT Ty);
  LLVM_ABI_NOT_EXPORTED LegalizeResult narrowScalarExt(MachineInstr &MI, unsigned TypeIdx,
                                          LLT Ty);
  LLVM_ABI_NOT_EXPORTED LegalizeResult narrowScalarSelect(MachineInstr &MI, unsigned TypeIdx,
                                             LLT Ty);
  LLVM_ABI_NOT_EXPORTED LegalizeResult narrowScalarCTLZ(MachineInstr &MI, unsigned TypeIdx,
                                           LLT Ty);
  LLVM_ABI_NOT_EXPORTED LegalizeResult narrowScalarCTTZ(MachineInstr &MI, unsigned TypeIdx,
                                           LLT Ty);
  LLVM_ABI_NOT_EXPORTED LegalizeResult narrowScalarCTLS(MachineInstr &MI, unsigned TypeIdx,
                                           LLT Ty);
  LLVM_ABI_NOT_EXPORTED LegalizeResult narrowScalarCTPOP(MachineInstr &MI, unsigned TypeIdx,
                                            LLT Ty);
  LLVM_ABI_NOT_EXPORTED LegalizeResult narrowScalarFLDEXP(MachineInstr &MI, unsigned TypeIdx,
                                             LLT Ty);

  /// Perform Bitcast legalize action on G_EXTRACT_VECTOR_ELT.
  LLVM_ABI_NOT_EXPORTED LegalizeResult bitcastExtractVectorElt(MachineInstr &MI,
                                                  unsigned TypeIdx, LLT CastTy);

  /// Perform Bitcast legalize action on G_INSERT_VECTOR_ELT.
  LLVM_ABI_NOT_EXPORTED LegalizeResult bitcastInsertVectorElt(MachineInstr &MI,
                                                 unsigned TypeIdx, LLT CastTy);
  LLVM_ABI_NOT_EXPORTED LegalizeResult bitcastConcatVector(MachineInstr &MI,
                                              unsigned TypeIdx, LLT CastTy);
  LLVM_ABI_NOT_EXPORTED LegalizeResult bitcastShuffleVector(MachineInstr &MI,
                                               unsigned TypeIdx, LLT CastTy);
  LLVM_ABI_NOT_EXPORTED LegalizeResult bitcastExtractSubvector(MachineInstr &MI,
                                                  unsigned TypeIdx, LLT CastTy);
  LLVM_ABI_NOT_EXPORTED LegalizeResult bitcastInsertSubvector(MachineInstr &MI,
                                                 unsigned TypeIdx, LLT CastTy);

  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerConstant(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerFConstant(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerBitcast(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerLoad(GAnyLoad &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerStore(GStore &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerBitCount(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerFunnelShiftWithInverse(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerFunnelShiftAsShifts(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerFunnelShift(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerEXT(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerTRUNC(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerRotateWithReverseRotate(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerRotate(MachineInstr &MI);

  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerU64ToF32WithSITOFP(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerU64ToF64BitFloatOps(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerUITOFP(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerSITOFP(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerFPTOUI(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerFPTOSI(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerFPTOINT_SAT(MachineInstr &MI);

  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerFPExtAndTruncMem(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerFPEXT(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerFPEXT_BF16(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerFPTRUNC_F64_TO_F16(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerFPTRUNC_F32_TO_BF16(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED Register lowerRoundInexactToOdd(LLT ResultTy, Register Op);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerFPTRUNC_F64_TO_BF16(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerFPTRUNC(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerFPOWI(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerFMODF(MachineInstr &MI);

  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerISFPCLASS(MachineInstr &MI);

  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerThreewayCompare(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerMinMax(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerFCopySign(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerFMinNumMaxNum(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerFMinimumMaximum(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerFMad(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerIntrinsicRound(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerFFloor(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerMergeValues(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerUnmergeValues(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerExtractInsertVectorElt(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerShuffleVector(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerVECTOR_COMPRESS(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED Register getDynStackAllocTargetPtr(Register SPReg,
                                              Register AllocSize,
                                              Align Alignment, LLT PtrTy);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerDynStackAlloc(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerStackSave(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerStackRestore(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerExtract(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerInsert(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerSADDO_SSUBO(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerSADDE(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerSSUBE(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerAddSubSatToMinMax(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerAddSubSatToAddoSubo(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerShlSat(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerTruncSat(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerBswap(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerBitreverse(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerReadWriteRegister(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerSMULH_UMULH(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerSelect(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerDIVREM(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerAbsToAddXor(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerAbsToMaxNeg(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerAbsToCNeg(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerAbsDiffToSelect(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerAbsDiffToMinMax(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerFAbs(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerVectorReduction(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerMemCpyFamily(MachineInstr &MI, Register Dst,
                                            Register Src, uint64_t KnownLen,
                                            Align Alignment,
                                            bool DstAlignCanChange,
                                            ArrayRef<LLT> MemOps);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerMemCpyFamily(MachineInstr &MI,
                                            unsigned MaxLen = 0);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerVAArg(MachineInstr &MI);
  LLVM_ABI_NOT_EXPORTED LegalizeResult lowerMulfix(MachineInstr &MI);
};

} // End namespace llvm.

#endif
