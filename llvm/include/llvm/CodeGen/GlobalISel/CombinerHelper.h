//===-- llvm/CodeGen/GlobalISel/CombinerHelper.h --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//
/// \file
/// This contains common combine transformations that may be used in a combine
/// pass,or by the target elsewhere.
/// Targets can pick individual opcode transformations from the helper or use
/// tryCombine which invokes all transformations. All of the transformations
/// return true if the MachineInstruction changed and false otherwise.
///
//===--------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_COMBINERHELPER_H
#define LLVM_CODEGEN_GLOBALISEL_COMBINERHELPER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGenTypes/LowLevelType.h"
#include "llvm/IR/InstrTypes.h"
#include <functional>

namespace llvm {

class GISelChangeObserver;
class APInt;
class ConstantFP;
class GPtrAdd;
class GZExtLoad;
class MachineIRBuilder;
class MachineInstrBuilder;
class MachineRegisterInfo;
class MachineInstr;
class MachineOperand;
class GISelKnownBits;
class MachineDominatorTree;
class LegalizerInfo;
struct LegalityQuery;
class RegisterBank;
class RegisterBankInfo;
class TargetLowering;
class TargetRegisterInfo;

struct PreferredTuple {
  LLT Ty;                // The result type of the extend.
  unsigned ExtendOpcode; // G_ANYEXT/G_SEXT/G_ZEXT
  MachineInstr *MI;
};

struct IndexedLoadStoreMatchInfo {
  Register Addr;
  Register Base;
  Register Offset;
  bool RematOffset = false; // True if Offset is a constant that needs to be
                            // rematerialized before the new load/store.
  bool IsPre = false;
};

struct PtrAddChain {
  int64_t Imm;
  Register Base;
  const RegisterBank *Bank;
};

struct RegisterImmPair {
  Register Reg;
  int64_t Imm;
};

struct ShiftOfShiftedLogic {
  MachineInstr *Logic;
  MachineInstr *Shift2;
  Register LogicNonShiftReg;
  uint64_t ValSum;
};

using BuildFnTy = std::function<void(MachineIRBuilder &)>;

using OperandBuildSteps =
    SmallVector<std::function<void(MachineInstrBuilder &)>, 4>;
struct InstructionBuildSteps {
  unsigned Opcode = 0;          /// The opcode for the produced instruction.
  OperandBuildSteps OperandFns; /// Operands to be added to the instruction.
  InstructionBuildSteps() = default;
  InstructionBuildSteps(unsigned Opcode, const OperandBuildSteps &OperandFns)
      : Opcode(Opcode), OperandFns(OperandFns) {}
};

struct InstructionStepsMatchInfo {
  /// Describes instructions to be built during a combine.
  SmallVector<InstructionBuildSteps, 2> InstrsToBuild;
  InstructionStepsMatchInfo() = default;
  InstructionStepsMatchInfo(
      std::initializer_list<InstructionBuildSteps> InstrsToBuild)
      : InstrsToBuild(InstrsToBuild) {}
};

class CombinerHelper {
protected:
  MachineIRBuilder &Builder;
  MachineRegisterInfo &MRI;
  GISelChangeObserver &Observer;
  GISelKnownBits *KB;
  MachineDominatorTree *MDT;
  bool IsPreLegalize;
  const LegalizerInfo *LI;
  const RegisterBankInfo *RBI;
  const TargetRegisterInfo *TRI;

public:
  CombinerHelper(GISelChangeObserver &Observer, MachineIRBuilder &B,
                 bool IsPreLegalize,
                 GISelKnownBits *KB = nullptr,
                 MachineDominatorTree *MDT = nullptr,
                 const LegalizerInfo *LI = nullptr);

  GISelKnownBits *getKnownBits() const {
    return KB;
  }

  MachineIRBuilder &getBuilder() const {
    return Builder;
  }

  const TargetLowering &getTargetLowering() const;

  const MachineFunction &getMachineFunction() const;

  const DataLayout &getDataLayout() const;

  LLVMContext &getContext() const;

  /// \returns true if the combiner is running pre-legalization.
  bool isPreLegalize() const;

  /// \returns true if \p Query is legal on the target.
  bool isLegal(const LegalityQuery &Query) const;

  /// \return true if the combine is running prior to legalization, or if \p
  /// Query is legal on the target.
  bool isLegalOrBeforeLegalizer(const LegalityQuery &Query) const;

  /// \return true if the combine is running prior to legalization, or if \p Ty
  /// is a legal integer constant type on the target.
  bool isConstantLegalOrBeforeLegalizer(const LLT Ty) const;

  /// MachineRegisterInfo::replaceRegWith() and inform the observer of the changes
  void replaceRegWith(MachineRegisterInfo &MRI, Register FromReg, Register ToReg) const;

  /// Replace a single register operand with a new register and inform the
  /// observer of the changes.
  void replaceRegOpWith(MachineRegisterInfo &MRI, MachineOperand &FromRegOp,
                        Register ToReg) const;

  /// Replace the opcode in instruction with a new opcode and inform the
  /// observer of the changes.
  void replaceOpcodeWith(MachineInstr &FromMI, unsigned ToOpcode) const;

  /// Get the register bank of \p Reg.
  /// If Reg has not been assigned a register, a register class,
  /// or a register bank, then this returns nullptr.
  ///
  /// \pre Reg.isValid()
  const RegisterBank *getRegBank(Register Reg) const;

  /// Set the register bank of \p Reg.
  /// Does nothing if the RegBank is null.
  /// This is the counterpart to getRegBank.
  void setRegBank(Register Reg, const RegisterBank *RegBank) const;

  /// If \p MI is COPY, try to combine it.
  /// Returns true if MI changed.
  bool tryCombineCopy(MachineInstr &MI) const;
  bool matchCombineCopy(MachineInstr &MI) const;
  void applyCombineCopy(MachineInstr &MI) const;

  /// Returns true if \p DefMI precedes \p UseMI or they are the same
  /// instruction. Both must be in the same basic block.
  bool isPredecessor(const MachineInstr &DefMI,
                     const MachineInstr &UseMI) const;

  /// Returns true if \p DefMI dominates \p UseMI. By definition an
  /// instruction dominates itself.
  ///
  /// If we haven't been provided with a MachineDominatorTree during
  /// construction, this function returns a conservative result that tracks just
  /// a single basic block.
  bool dominates(const MachineInstr &DefMI, const MachineInstr &UseMI) const;

  /// If \p MI is extend that consumes the result of a load, try to combine it.
  /// Returns true if MI changed.
  bool tryCombineExtendingLoads(MachineInstr &MI) const;
  bool matchCombineExtendingLoads(MachineInstr &MI,
                                  PreferredTuple &MatchInfo) const;
  void applyCombineExtendingLoads(MachineInstr &MI,
                                  PreferredTuple &MatchInfo) const;

  /// Match (and (load x), mask) -> zextload x
  bool matchCombineLoadWithAndMask(MachineInstr &MI,
                                   BuildFnTy &MatchInfo) const;

  /// Combine a G_EXTRACT_VECTOR_ELT of a load into a narrowed
  /// load.
  bool matchCombineExtractedVectorLoad(MachineInstr &MI,
                                       BuildFnTy &MatchInfo) const;

  bool matchCombineIndexedLoadStore(MachineInstr &MI,
                                    IndexedLoadStoreMatchInfo &MatchInfo) const;
  void applyCombineIndexedLoadStore(MachineInstr &MI,
                                    IndexedLoadStoreMatchInfo &MatchInfo) const;

  bool matchSextTruncSextLoad(MachineInstr &MI) const;
  void applySextTruncSextLoad(MachineInstr &MI) const;

  /// Match sext_inreg(load p), imm -> sextload p
  bool matchSextInRegOfLoad(MachineInstr &MI,
                            std::tuple<Register, unsigned> &MatchInfo) const;
  void applySextInRegOfLoad(MachineInstr &MI,
                            std::tuple<Register, unsigned> &MatchInfo) const;

  /// Try to combine G_[SU]DIV and G_[SU]REM into a single G_[SU]DIVREM
  /// when their source operands are identical.
  bool matchCombineDivRem(MachineInstr &MI, MachineInstr *&OtherMI) const;
  void applyCombineDivRem(MachineInstr &MI, MachineInstr *&OtherMI) const;

  /// If a brcond's true block is not the fallthrough, make it so by inverting
  /// the condition and swapping operands.
  bool matchOptBrCondByInvertingCond(MachineInstr &MI,
                                     MachineInstr *&BrCond) const;
  void applyOptBrCondByInvertingCond(MachineInstr &MI,
                                     MachineInstr *&BrCond) const;

  /// If \p MI is G_CONCAT_VECTORS, try to combine it.
  /// Returns true if MI changed.
  /// Right now, we support:
  /// - concat_vector(undef, undef) => undef
  /// - concat_vector(build_vector(A, B), build_vector(C, D)) =>
  ///   build_vector(A, B, C, D)
  /// ==========================================================
  /// Check if the G_CONCAT_VECTORS \p MI is undef or if it
  /// can be flattened into a build_vector.
  /// In the first case \p Ops will be empty
  /// In the second case \p Ops will contain the operands
  /// needed to produce the flattened build_vector.
  ///
  /// \pre MI.getOpcode() == G_CONCAT_VECTORS.
  bool matchCombineConcatVectors(MachineInstr &MI,
                                 SmallVector<Register> &Ops) const;
  /// Replace \p MI with a flattened build_vector with \p Ops
  /// or an implicit_def if \p Ops is empty.
  void applyCombineConcatVectors(MachineInstr &MI,
                                 SmallVector<Register> &Ops) const;

  bool matchCombineShuffleConcat(MachineInstr &MI,
                                 SmallVector<Register> &Ops) const;
  /// Replace \p MI with a flattened build_vector with \p Ops
  /// or an implicit_def if \p Ops is empty.
  void applyCombineShuffleConcat(MachineInstr &MI,
                                 SmallVector<Register> &Ops) const;

  /// Try to combine G_SHUFFLE_VECTOR into G_CONCAT_VECTORS.
  /// Returns true if MI changed.
  ///
  /// \pre MI.getOpcode() == G_SHUFFLE_VECTOR.
  bool tryCombineShuffleVector(MachineInstr &MI) const;
  /// Check if the G_SHUFFLE_VECTOR \p MI can be replaced by a
  /// concat_vectors.
  /// \p Ops will contain the operands needed to produce the flattened
  /// concat_vectors.
  ///
  /// \pre MI.getOpcode() == G_SHUFFLE_VECTOR.
  bool matchCombineShuffleVector(MachineInstr &MI,
                                 SmallVectorImpl<Register> &Ops) const;
  /// Replace \p MI with a concat_vectors with \p Ops.
  void applyCombineShuffleVector(MachineInstr &MI,
                                 const ArrayRef<Register> Ops) const;
  bool matchShuffleToExtract(MachineInstr &MI) const;
  void applyShuffleToExtract(MachineInstr &MI) const;

  /// Optimize memcpy intrinsics et al, e.g. constant len calls.
  /// /p MaxLen if non-zero specifies the max length of a mem libcall to inline.
  ///
  /// For example (pre-indexed):
  ///
  ///     $addr = G_PTR_ADD $base, $offset
  ///     [...]
  ///     $val = G_LOAD $addr
  ///     [...]
  ///     $whatever = COPY $addr
  ///
  /// -->
  ///
  ///     $val, $addr = G_INDEXED_LOAD $base, $offset, 1 (IsPre)
  ///     [...]
  ///     $whatever = COPY $addr
  ///
  /// or (post-indexed):
  ///
  ///     G_STORE $val, $base
  ///     [...]
  ///     $addr = G_PTR_ADD $base, $offset
  ///     [...]
  ///     $whatever = COPY $addr
  ///
  /// -->
  ///
  ///     $addr = G_INDEXED_STORE $val, $base, $offset
  ///     [...]
  ///     $whatever = COPY $addr
  bool tryCombineMemCpyFamily(MachineInstr &MI, unsigned MaxLen = 0) const;

  bool matchPtrAddImmedChain(MachineInstr &MI, PtrAddChain &MatchInfo) const;
  void applyPtrAddImmedChain(MachineInstr &MI, PtrAddChain &MatchInfo) const;

  /// Fold (shift (shift base, x), y) -> (shift base (x+y))
  bool matchShiftImmedChain(MachineInstr &MI, RegisterImmPair &MatchInfo) const;
  void applyShiftImmedChain(MachineInstr &MI, RegisterImmPair &MatchInfo) const;

  /// If we have a shift-by-constant of a bitwise logic op that itself has a
  /// shift-by-constant operand with identical opcode, we may be able to convert
  /// that into 2 independent shifts followed by the logic op.
  bool matchShiftOfShiftedLogic(MachineInstr &MI,
                                ShiftOfShiftedLogic &MatchInfo) const;
  void applyShiftOfShiftedLogic(MachineInstr &MI,
                                ShiftOfShiftedLogic &MatchInfo) const;

  bool matchCommuteShift(MachineInstr &MI, BuildFnTy &MatchInfo) const;

  /// Transform a multiply by a power-of-2 value to a left shift.
  bool matchCombineMulToShl(MachineInstr &MI, unsigned &ShiftVal) const;
  void applyCombineMulToShl(MachineInstr &MI, unsigned &ShiftVal) const;

  // Transform a G_SUB with constant on the RHS to G_ADD.
  bool matchCombineSubToAdd(MachineInstr &MI, BuildFnTy &MatchInfo) const;

  // Transform a G_SHL with an extended source into a narrower shift if
  // possible.
  bool matchCombineShlOfExtend(MachineInstr &MI,
                               RegisterImmPair &MatchData) const;
  void applyCombineShlOfExtend(MachineInstr &MI,
                               const RegisterImmPair &MatchData) const;

  /// Fold away a merge of an unmerge of the corresponding values.
  bool matchCombineMergeUnmerge(MachineInstr &MI, Register &MatchInfo) const;

  /// Reduce a shift by a constant to an unmerge and a shift on a half sized
  /// type. This will not produce a shift smaller than \p TargetShiftSize.
  bool matchCombineShiftToUnmerge(MachineInstr &MI, unsigned TargetShiftSize,
                                  unsigned &ShiftVal) const;
  void applyCombineShiftToUnmerge(MachineInstr &MI,
                                  const unsigned &ShiftVal) const;
  bool tryCombineShiftToUnmerge(MachineInstr &MI,
                                unsigned TargetShiftAmount) const;

  /// Transform <ty,...> G_UNMERGE(G_MERGE ty X, Y, Z) -> ty X, Y, Z.
  bool matchCombineUnmergeMergeToPlainValues(
      MachineInstr &MI, SmallVectorImpl<Register> &Operands) const;
  void applyCombineUnmergeMergeToPlainValues(
      MachineInstr &MI, SmallVectorImpl<Register> &Operands) const;

  /// Transform G_UNMERGE Constant -> Constant1, Constant2, ...
  bool matchCombineUnmergeConstant(MachineInstr &MI,
                                   SmallVectorImpl<APInt> &Csts) const;
  void applyCombineUnmergeConstant(MachineInstr &MI,
                                   SmallVectorImpl<APInt> &Csts) const;

  /// Transform G_UNMERGE G_IMPLICIT_DEF -> G_IMPLICIT_DEF, G_IMPLICIT_DEF, ...
  bool matchCombineUnmergeUndef(
      MachineInstr &MI,
      std::function<void(MachineIRBuilder &)> &MatchInfo) const;

  /// Transform X, Y<dead> = G_UNMERGE Z -> X = G_TRUNC Z.
  bool matchCombineUnmergeWithDeadLanesToTrunc(MachineInstr &MI) const;
  void applyCombineUnmergeWithDeadLanesToTrunc(MachineInstr &MI) const;

  /// Transform X, Y = G_UNMERGE(G_ZEXT(Z)) -> X = G_ZEXT(Z); Y = G_CONSTANT 0
  bool matchCombineUnmergeZExtToZExt(MachineInstr &MI) const;
  void applyCombineUnmergeZExtToZExt(MachineInstr &MI) const;

  /// Transform fp_instr(cst) to constant result of the fp operation.
  void applyCombineConstantFoldFpUnary(MachineInstr &MI,
                                       const ConstantFP *Cst) const;

  /// Transform IntToPtr(PtrToInt(x)) to x if cast is in the same address space.
  bool matchCombineI2PToP2I(MachineInstr &MI, Register &Reg) const;
  void applyCombineI2PToP2I(MachineInstr &MI, Register &Reg) const;

  /// Transform PtrToInt(IntToPtr(x)) to x.
  void applyCombineP2IToI2P(MachineInstr &MI, Register &Reg) const;

  /// Transform G_ADD (G_PTRTOINT x), y -> G_PTRTOINT (G_PTR_ADD x, y)
  /// Transform G_ADD y, (G_PTRTOINT x) -> G_PTRTOINT (G_PTR_ADD x, y)
  bool
  matchCombineAddP2IToPtrAdd(MachineInstr &MI,
                             std::pair<Register, bool> &PtrRegAndCommute) const;
  void
  applyCombineAddP2IToPtrAdd(MachineInstr &MI,
                             std::pair<Register, bool> &PtrRegAndCommute) const;

  // Transform G_PTR_ADD (G_PTRTOINT C1), C2 -> C1 + C2
  bool matchCombineConstPtrAddToI2P(MachineInstr &MI, APInt &NewCst) const;
  void applyCombineConstPtrAddToI2P(MachineInstr &MI, APInt &NewCst) const;

  /// Transform anyext(trunc(x)) to x.
  bool matchCombineAnyExtTrunc(MachineInstr &MI, Register &Reg) const;

  /// Transform zext(trunc(x)) to x.
  bool matchCombineZextTrunc(MachineInstr &MI, Register &Reg) const;

  /// Transform trunc (shl x, K) to shl (trunc x), K
  ///    if K < VT.getScalarSizeInBits().
  ///
  /// Transforms trunc ([al]shr x, K) to (trunc ([al]shr (MidVT (trunc x)), K))
  ///    if K <= (MidVT.getScalarSizeInBits() - VT.getScalarSizeInBits())
  /// MidVT is obtained by finding a legal type between the trunc's src and dst
  /// types.
  bool
  matchCombineTruncOfShift(MachineInstr &MI,
                           std::pair<MachineInstr *, LLT> &MatchInfo) const;
  void
  applyCombineTruncOfShift(MachineInstr &MI,
                           std::pair<MachineInstr *, LLT> &MatchInfo) const;

  /// Return true if any explicit use operand on \p MI is defined by a
  /// G_IMPLICIT_DEF.
  bool matchAnyExplicitUseIsUndef(MachineInstr &MI) const;

  /// Return true if all register explicit use operands on \p MI are defined by
  /// a G_IMPLICIT_DEF.
  bool matchAllExplicitUsesAreUndef(MachineInstr &MI) const;

  /// Return true if a G_SHUFFLE_VECTOR instruction \p MI has an undef mask.
  bool matchUndefShuffleVectorMask(MachineInstr &MI) const;

  /// Return true if a G_STORE instruction \p MI is storing an undef value.
  bool matchUndefStore(MachineInstr &MI) const;

  /// Return true if a G_SELECT instruction \p MI has an undef comparison.
  bool matchUndefSelectCmp(MachineInstr &MI) const;

  /// Return true if a G_{EXTRACT,INSERT}_VECTOR_ELT has an out of range index.
  bool matchInsertExtractVecEltOutOfBounds(MachineInstr &MI) const;

  /// Return true if a G_SELECT instruction \p MI has a constant comparison. If
  /// true, \p OpIdx will store the operand index of the known selected value.
  bool matchConstantSelectCmp(MachineInstr &MI, unsigned &OpIdx) const;

  /// Replace an instruction with a G_FCONSTANT with value \p C.
  void replaceInstWithFConstant(MachineInstr &MI, double C) const;

  /// Replace an instruction with an G_FCONSTANT with value \p CFP.
  void replaceInstWithFConstant(MachineInstr &MI, ConstantFP *CFP) const;

  /// Replace an instruction with a G_CONSTANT with value \p C.
  void replaceInstWithConstant(MachineInstr &MI, int64_t C) const;

  /// Replace an instruction with a G_CONSTANT with value \p C.
  void replaceInstWithConstant(MachineInstr &MI, APInt C) const;

  /// Replace an instruction with a G_IMPLICIT_DEF.
  void replaceInstWithUndef(MachineInstr &MI) const;

  /// Delete \p MI and replace all of its uses with its \p OpIdx-th operand.
  void replaceSingleDefInstWithOperand(MachineInstr &MI, unsigned OpIdx) const;

  /// Delete \p MI and replace all of its uses with \p Replacement.
  void replaceSingleDefInstWithReg(MachineInstr &MI,
                                   Register Replacement) const;

  /// @brief Replaces the shift amount in \p MI with ShiftAmt % BW
  /// @param MI
  void applyFunnelShiftConstantModulo(MachineInstr &MI) const;

  /// Return true if \p MOP1 and \p MOP2 are register operands are defined by
  /// equivalent instructions.
  bool matchEqualDefs(const MachineOperand &MOP1,
                      const MachineOperand &MOP2) const;

  /// Return true if \p MOP is defined by a G_CONSTANT or splat with a value equal to
  /// \p C.
  bool matchConstantOp(const MachineOperand &MOP, int64_t C) const;

  /// Return true if \p MOP is defined by a G_FCONSTANT or splat with a value exactly
  /// equal to \p C.
  bool matchConstantFPOp(const MachineOperand &MOP, double C) const;

  /// @brief Checks if constant at \p ConstIdx is larger than \p MI 's bitwidth
  /// @param ConstIdx Index of the constant
  bool matchConstantLargerBitWidth(MachineInstr &MI, unsigned ConstIdx) const;

  /// Optimize (cond ? x : x) -> x
  bool matchSelectSameVal(MachineInstr &MI) const;

  /// Optimize (x op x) -> x
  bool matchBinOpSameVal(MachineInstr &MI) const;

  /// Check if operand \p OpIdx is zero.
  bool matchOperandIsZero(MachineInstr &MI, unsigned OpIdx) const;

  /// Check if operand \p OpIdx is undef.
  bool matchOperandIsUndef(MachineInstr &MI, unsigned OpIdx) const;

  /// Check if operand \p OpIdx is known to be a power of 2.
  bool matchOperandIsKnownToBeAPowerOfTwo(MachineInstr &MI,
                                          unsigned OpIdx) const;

  /// Erase \p MI
  void eraseInst(MachineInstr &MI) const;

  /// Return true if MI is a G_ADD which can be simplified to a G_SUB.
  bool matchSimplifyAddToSub(MachineInstr &MI,
                             std::tuple<Register, Register> &MatchInfo) const;
  void applySimplifyAddToSub(MachineInstr &MI,
                             std::tuple<Register, Register> &MatchInfo) const;

  /// Match (logic_op (op x...), (op y...)) -> (op (logic_op x, y))
  bool matchHoistLogicOpWithSameOpcodeHands(
      MachineInstr &MI, InstructionStepsMatchInfo &MatchInfo) const;

  /// Replace \p MI with a series of instructions described in \p MatchInfo.
  void applyBuildInstructionSteps(MachineInstr &MI,
                                  InstructionStepsMatchInfo &MatchInfo) const;

  /// Match ashr (shl x, C), C -> sext_inreg (C)
  bool matchAshrShlToSextInreg(MachineInstr &MI,
                               std::tuple<Register, int64_t> &MatchInfo) const;
  void applyAshShlToSextInreg(MachineInstr &MI,
                              std::tuple<Register, int64_t> &MatchInfo) const;

  /// Fold and(and(x, C1), C2) -> C1&C2 ? and(x, C1&C2) : 0
  bool matchOverlappingAnd(MachineInstr &MI, BuildFnTy &MatchInfo) const;

  /// \return true if \p MI is a G_AND instruction whose operands are x and y
  /// where x & y == x or x & y == y. (E.g., one of operands is all-ones value.)
  ///
  /// \param [in] MI - The G_AND instruction.
  /// \param [out] Replacement - A register the G_AND should be replaced with on
  /// success.
  bool matchRedundantAnd(MachineInstr &MI, Register &Replacement) const;

  /// \return true if \p MI is a G_OR instruction whose operands are x and y
  /// where x | y == x or x | y == y. (E.g., one of operands is all-zeros
  /// value.)
  ///
  /// \param [in] MI - The G_OR instruction.
  /// \param [out] Replacement - A register the G_OR should be replaced with on
  /// success.
  bool matchRedundantOr(MachineInstr &MI, Register &Replacement) const;

  /// \return true if \p MI is a G_SEXT_INREG that can be erased.
  bool matchRedundantSExtInReg(MachineInstr &MI) const;

  /// Combine inverting a result of a compare into the opposite cond code.
  bool matchNotCmp(MachineInstr &MI,
                   SmallVectorImpl<Register> &RegsToNegate) const;
  void applyNotCmp(MachineInstr &MI,
                   SmallVectorImpl<Register> &RegsToNegate) const;

  /// Fold (xor (and x, y), y) -> (and (not x), y)
  ///{
  bool matchXorOfAndWithSameReg(MachineInstr &MI,
                                std::pair<Register, Register> &MatchInfo) const;
  void applyXorOfAndWithSameReg(MachineInstr &MI,
                                std::pair<Register, Register> &MatchInfo) const;
  ///}

  /// Combine G_PTR_ADD with nullptr to G_INTTOPTR
  bool matchPtrAddZero(MachineInstr &MI) const;
  void applyPtrAddZero(MachineInstr &MI) const;

  /// Combine G_UREM x, (known power of 2) to an add and bitmasking.
  void applySimplifyURemByPow2(MachineInstr &MI) const;

  /// Push a binary operator through a select on constants.
  ///
  /// binop (select cond, K0, K1), K2 ->
  ///   select cond, (binop K0, K2), (binop K1, K2)
  bool matchFoldBinOpIntoSelect(MachineInstr &MI, unsigned &SelectOpNo) const;
  void applyFoldBinOpIntoSelect(MachineInstr &MI,
                                const unsigned &SelectOpNo) const;

  bool matchCombineInsertVecElts(MachineInstr &MI,
                                 SmallVectorImpl<Register> &MatchInfo) const;

  void applyCombineInsertVecElts(MachineInstr &MI,
                                 SmallVectorImpl<Register> &MatchInfo) const;

  /// Match expression trees of the form
  ///
  /// \code
  ///  sN *a = ...
  ///  sM val = a[0] | (a[1] << N) | (a[2] << 2N) | (a[3] << 3N) ...
  /// \endcode
  ///
  /// And check if the tree can be replaced with a M-bit load + possibly a
  /// bswap.
  bool matchLoadOrCombine(MachineInstr &MI, BuildFnTy &MatchInfo) const;

  bool matchExtendThroughPhis(MachineInstr &MI, MachineInstr *&ExtMI) const;
  void applyExtendThroughPhis(MachineInstr &MI, MachineInstr *&ExtMI) const;

  bool matchExtractVecEltBuildVec(MachineInstr &MI, Register &Reg) const;
  void applyExtractVecEltBuildVec(MachineInstr &MI, Register &Reg) const;

  bool matchExtractAllEltsFromBuildVector(
      MachineInstr &MI,
      SmallVectorImpl<std::pair<Register, MachineInstr *>> &MatchInfo) const;
  void applyExtractAllEltsFromBuildVector(
      MachineInstr &MI,
      SmallVectorImpl<std::pair<Register, MachineInstr *>> &MatchInfo) const;

  /// Use a function which takes in a MachineIRBuilder to perform a combine.
  /// By default, it erases the instruction \p MI from the function.
  void applyBuildFn(MachineInstr &MI, BuildFnTy &MatchInfo) const;
  /// Use a function which takes in a MachineIRBuilder to perform a combine.
  /// This variant does not erase \p MI after calling the build function.
  void applyBuildFnNoErase(MachineInstr &MI, BuildFnTy &MatchInfo) const;

  bool matchOrShiftToFunnelShift(MachineInstr &MI, BuildFnTy &MatchInfo) const;
  bool matchFunnelShiftToRotate(MachineInstr &MI) const;
  void applyFunnelShiftToRotate(MachineInstr &MI) const;
  bool matchRotateOutOfRange(MachineInstr &MI) const;
  void applyRotateOutOfRange(MachineInstr &MI) const;

  bool matchUseVectorTruncate(MachineInstr &MI, Register &MatchInfo) const;
  void applyUseVectorTruncate(MachineInstr &MI, Register &MatchInfo) const;

  /// \returns true if a G_ICMP instruction \p MI can be replaced with a true
  /// or false constant based off of KnownBits information.
  bool matchICmpToTrueFalseKnownBits(MachineInstr &MI,
                                     int64_t &MatchInfo) const;

  /// \returns true if a G_ICMP \p MI can be replaced with its LHS based off of
  /// KnownBits information.
  bool matchICmpToLHSKnownBits(MachineInstr &MI, BuildFnTy &MatchInfo) const;

  /// \returns true if (and (or x, c1), c2) can be replaced with (and x, c2)
  bool matchAndOrDisjointMask(MachineInstr &MI, BuildFnTy &MatchInfo) const;

  bool matchBitfieldExtractFromSExtInReg(MachineInstr &MI,
                                         BuildFnTy &MatchInfo) const;
  /// Match: and (lshr x, cst), mask -> ubfx x, cst, width
  bool matchBitfieldExtractFromAnd(MachineInstr &MI,
                                   BuildFnTy &MatchInfo) const;

  /// Match: shr (shl x, n), k -> sbfx/ubfx x, pos, width
  bool matchBitfieldExtractFromShr(MachineInstr &MI,
                                   BuildFnTy &MatchInfo) const;

  /// Match: shr (and x, n), k -> ubfx x, pos, width
  bool matchBitfieldExtractFromShrAnd(MachineInstr &MI,
                                      BuildFnTy &MatchInfo) const;

  // Helpers for reassociation:
  bool matchReassocConstantInnerRHS(GPtrAdd &MI, MachineInstr *RHS,
                                    BuildFnTy &MatchInfo) const;
  bool matchReassocFoldConstantsInSubTree(GPtrAdd &MI, MachineInstr *LHS,
                                          MachineInstr *RHS,
                                          BuildFnTy &MatchInfo) const;
  bool matchReassocConstantInnerLHS(GPtrAdd &MI, MachineInstr *LHS,
                                    MachineInstr *RHS,
                                    BuildFnTy &MatchInfo) const;
  /// Reassociate pointer calculations with G_ADD involved, to allow better
  /// addressing mode usage.
  bool matchReassocPtrAdd(MachineInstr &MI, BuildFnTy &MatchInfo) const;

  /// Try to reassociate to reassociate operands of a commutative binop.
  bool tryReassocBinOp(unsigned Opc, Register DstReg, Register Op0,
                       Register Op1, BuildFnTy &MatchInfo) const;
  /// Reassociate commutative binary operations like G_ADD.
  bool matchReassocCommBinOp(MachineInstr &MI, BuildFnTy &MatchInfo) const;

  /// Do constant folding when opportunities are exposed after MIR building.
  bool matchConstantFoldCastOp(MachineInstr &MI, APInt &MatchInfo) const;

  /// Do constant folding when opportunities are exposed after MIR building.
  bool matchConstantFoldBinOp(MachineInstr &MI, APInt &MatchInfo) const;

  /// Do constant FP folding when opportunities are exposed after MIR building.
  bool matchConstantFoldFPBinOp(MachineInstr &MI, ConstantFP *&MatchInfo) const;

  /// Constant fold G_FMA/G_FMAD.
  bool matchConstantFoldFMA(MachineInstr &MI, ConstantFP *&MatchInfo) const;

  /// \returns true if it is possible to narrow the width of a scalar binop
  /// feeding a G_AND instruction \p MI.
  bool matchNarrowBinopFeedingAnd(MachineInstr &MI, BuildFnTy &MatchInfo) const;

  /// Given an G_UDIV \p MI expressing a divide by constant, return an
  /// expression that implements it by multiplying by a magic number.
  /// Ref: "Hacker's Delight" or "The PowerPC Compiler Writer's Guide".
  MachineInstr *buildUDivUsingMul(MachineInstr &MI) const;
  /// Combine G_UDIV by constant into a multiply by magic constant.
  bool matchUDivByConst(MachineInstr &MI) const;
  void applyUDivByConst(MachineInstr &MI) const;

  /// Given an G_SDIV \p MI expressing a signed divide by constant, return an
  /// expression that implements it by multiplying by a magic number.
  /// Ref: "Hacker's Delight" or "The PowerPC Compiler Writer's Guide".
  MachineInstr *buildSDivUsingMul(MachineInstr &MI) const;
  bool matchSDivByConst(MachineInstr &MI) const;
  void applySDivByConst(MachineInstr &MI) const;

  /// Given an G_SDIV \p MI expressing a signed divided by a pow2 constant,
  /// return expressions that implements it by shifting.
  bool matchDivByPow2(MachineInstr &MI, bool IsSigned) const;
  void applySDivByPow2(MachineInstr &MI) const;
  /// Given an G_UDIV \p MI expressing an unsigned divided by a pow2 constant,
  /// return expressions that implements it by shifting.
  void applyUDivByPow2(MachineInstr &MI) const;

  // G_UMULH x, (1 << c)) -> x >> (bitwidth - c)
  bool matchUMulHToLShr(MachineInstr &MI) const;
  void applyUMulHToLShr(MachineInstr &MI) const;

  /// Try to transform \p MI by using all of the above
  /// combine functions. Returns true if changed.
  bool tryCombine(MachineInstr &MI) const;

  /// Emit loads and stores that perform the given memcpy.
  /// Assumes \p MI is a G_MEMCPY_INLINE
  /// TODO: implement dynamically sized inline memcpy,
  ///       and rename: s/bool tryEmit/void emit/
  bool tryEmitMemcpyInline(MachineInstr &MI) const;

  /// Match:
  ///   (G_UMULO x, 2) -> (G_UADDO x, x)
  ///   (G_SMULO x, 2) -> (G_SADDO x, x)
  bool matchMulOBy2(MachineInstr &MI, BuildFnTy &MatchInfo) const;

  /// Match:
  /// (G_*MULO x, 0) -> 0 + no carry out
  bool matchMulOBy0(MachineInstr &MI, BuildFnTy &MatchInfo) const;

  /// Match:
  /// (G_*ADDE x, y, 0) -> (G_*ADDO x, y)
  /// (G_*SUBE x, y, 0) -> (G_*SUBO x, y)
  bool matchAddEToAddO(MachineInstr &MI, BuildFnTy &MatchInfo) const;

  /// Transform (fadd x, fneg(y)) -> (fsub x, y)
  ///           (fadd fneg(x), y) -> (fsub y, x)
  ///           (fsub x, fneg(y)) -> (fadd x, y)
  ///           (fmul fneg(x), fneg(y)) -> (fmul x, y)
  ///           (fdiv fneg(x), fneg(y)) -> (fdiv x, y)
  ///           (fmad fneg(x), fneg(y), z) -> (fmad x, y, z)
  ///           (fma fneg(x), fneg(y), z) -> (fma x, y, z)
  bool matchRedundantNegOperands(MachineInstr &MI, BuildFnTy &MatchInfo) const;

  bool matchFsubToFneg(MachineInstr &MI, Register &MatchInfo) const;
  void applyFsubToFneg(MachineInstr &MI, Register &MatchInfo) const;

  bool canCombineFMadOrFMA(MachineInstr &MI, bool &AllowFusionGlobally,
                           bool &HasFMAD, bool &Aggressive,
                           bool CanReassociate = false) const;

  /// Transform (fadd (fmul x, y), z) -> (fma x, y, z)
  ///           (fadd (fmul x, y), z) -> (fmad x, y, z)
  bool matchCombineFAddFMulToFMadOrFMA(MachineInstr &MI,
                                       BuildFnTy &MatchInfo) const;

  /// Transform (fadd (fpext (fmul x, y)), z) -> (fma (fpext x), (fpext y), z)
  ///           (fadd (fpext (fmul x, y)), z) -> (fmad (fpext x), (fpext y), z)
  bool matchCombineFAddFpExtFMulToFMadOrFMA(MachineInstr &MI,
                                            BuildFnTy &MatchInfo) const;

  /// Transform (fadd (fma x, y, (fmul u, v)), z) -> (fma x, y, (fma u, v, z))
  ///          (fadd (fmad x, y, (fmul u, v)), z) -> (fmad x, y, (fmad u, v, z))
  bool matchCombineFAddFMAFMulToFMadOrFMA(MachineInstr &MI,
                                          BuildFnTy &MatchInfo) const;

  // Transform (fadd (fma x, y, (fpext (fmul u, v))), z)
  //            -> (fma x, y, (fma (fpext u), (fpext v), z))
  //           (fadd (fmad x, y, (fpext (fmul u, v))), z)
  //            -> (fmad x, y, (fmad (fpext u), (fpext v), z))
  bool
  matchCombineFAddFpExtFMulToFMadOrFMAAggressive(MachineInstr &MI,
                                                 BuildFnTy &MatchInfo) const;

  /// Transform (fsub (fmul x, y), z) -> (fma x, y, -z)
  ///           (fsub (fmul x, y), z) -> (fmad x, y, -z)
  bool matchCombineFSubFMulToFMadOrFMA(MachineInstr &MI,
                                       BuildFnTy &MatchInfo) const;

  /// Transform (fsub (fneg (fmul, x, y)), z) -> (fma (fneg x), y, (fneg z))
  ///           (fsub (fneg (fmul, x, y)), z) -> (fmad (fneg x), y, (fneg z))
  bool matchCombineFSubFNegFMulToFMadOrFMA(MachineInstr &MI,
                                           BuildFnTy &MatchInfo) const;

  /// Transform (fsub (fpext (fmul x, y)), z)
  ///           -> (fma (fpext x), (fpext y), (fneg z))
  ///           (fsub (fpext (fmul x, y)), z)
  ///           -> (fmad (fpext x), (fpext y), (fneg z))
  bool matchCombineFSubFpExtFMulToFMadOrFMA(MachineInstr &MI,
                                            BuildFnTy &MatchInfo) const;

  /// Transform (fsub (fpext (fneg (fmul x, y))), z)
  ///           -> (fneg (fma (fpext x), (fpext y), z))
  ///           (fsub (fpext (fneg (fmul x, y))), z)
  ///           -> (fneg (fmad (fpext x), (fpext y), z))
  bool matchCombineFSubFpExtFNegFMulToFMadOrFMA(MachineInstr &MI,
                                                BuildFnTy &MatchInfo) const;

  bool matchCombineFMinMaxNaN(MachineInstr &MI, unsigned &Info) const;

  /// Transform G_ADD(x, G_SUB(y, x)) to y.
  /// Transform G_ADD(G_SUB(y, x), x) to y.
  bool matchAddSubSameReg(MachineInstr &MI, Register &Src) const;

  bool matchBuildVectorIdentityFold(MachineInstr &MI,
                                    Register &MatchInfo) const;
  bool matchTruncBuildVectorFold(MachineInstr &MI, Register &MatchInfo) const;
  bool matchTruncLshrBuildVectorFold(MachineInstr &MI,
                                     Register &MatchInfo) const;

  /// Transform:
  ///   (x + y) - y -> x
  ///   (x + y) - x -> y
  ///   x - (y + x) -> 0 - y
  ///   x - (x + z) -> 0 - z
  bool matchSubAddSameReg(MachineInstr &MI, BuildFnTy &MatchInfo) const;

  /// \returns true if it is possible to simplify a select instruction \p MI
  /// to a min/max instruction of some sort.
  bool matchSimplifySelectToMinMax(MachineInstr &MI,
                                   BuildFnTy &MatchInfo) const;

  /// Transform:
  ///   (X + Y) == X -> Y == 0
  ///   (X - Y) == X -> Y == 0
  ///   (X ^ Y) == X -> Y == 0
  ///   (X + Y) != X -> Y != 0
  ///   (X - Y) != X -> Y != 0
  ///   (X ^ Y) != X -> Y != 0
  bool matchRedundantBinOpInEquality(MachineInstr &MI,
                                     BuildFnTy &MatchInfo) const;

  /// Match shifts greater or equal to the range (the bitwidth of the result
  /// datatype, or the effective bitwidth of the source value).
  bool matchShiftsTooBig(MachineInstr &MI,
                         std::optional<int64_t> &MatchInfo) const;

  /// Match constant LHS ops that should be commuted.
  bool matchCommuteConstantToRHS(MachineInstr &MI) const;

  /// Combine sext of trunc.
  bool matchSextOfTrunc(const MachineOperand &MO, BuildFnTy &MatchInfo) const;

  /// Combine zext of trunc.
  bool matchZextOfTrunc(const MachineOperand &MO, BuildFnTy &MatchInfo) const;

  /// Combine zext nneg to sext.
  bool matchNonNegZext(const MachineOperand &MO, BuildFnTy &MatchInfo) const;

  /// Match constant LHS FP ops that should be commuted.
  bool matchCommuteFPConstantToRHS(MachineInstr &MI) const;

  // Given a binop \p MI, commute operands 1 and 2.
  void applyCommuteBinOpOperands(MachineInstr &MI) const;

  /// Combine select to integer min/max.
  bool matchSelectIMinMax(const MachineOperand &MO, BuildFnTy &MatchInfo) const;

  /// Tranform (neg (min/max x, (neg x))) into (max/min x, (neg x)).
  bool matchSimplifyNegMinMax(MachineInstr &MI, BuildFnTy &MatchInfo) const;

  /// Combine selects.
  bool matchSelect(MachineInstr &MI, BuildFnTy &MatchInfo) const;

  /// Combine ands.
  bool matchAnd(MachineInstr &MI, BuildFnTy &MatchInfo) const;

  /// Combine ors.
  bool matchOr(MachineInstr &MI, BuildFnTy &MatchInfo) const;

  /// trunc (binop X, C) --> binop (trunc X, trunc C).
  bool matchNarrowBinop(const MachineInstr &TruncMI,
                        const MachineInstr &BinopMI,
                        BuildFnTy &MatchInfo) const;

  bool matchCastOfInteger(const MachineInstr &CastMI, APInt &MatchInfo) const;

  /// Combine addos.
  bool matchAddOverflow(MachineInstr &MI, BuildFnTy &MatchInfo) const;

  /// Combine extract vector element.
  bool matchExtractVectorElement(MachineInstr &MI, BuildFnTy &MatchInfo) const;

  /// Combine extract vector element with a build vector on the vector register.
  bool matchExtractVectorElementWithBuildVector(const MachineInstr &MI,
                                                const MachineInstr &MI2,
                                                BuildFnTy &MatchInfo) const;

  /// Combine extract vector element with a build vector trunc on the vector
  /// register.
  bool
  matchExtractVectorElementWithBuildVectorTrunc(const MachineOperand &MO,
                                                BuildFnTy &MatchInfo) const;

  /// Combine extract vector element with a shuffle vector on the vector
  /// register.
  bool matchExtractVectorElementWithShuffleVector(const MachineInstr &MI,
                                                  const MachineInstr &MI2,
                                                  BuildFnTy &MatchInfo) const;

  /// Combine extract vector element with a insert vector element on the vector
  /// register and different indices.
  bool
  matchExtractVectorElementWithDifferentIndices(const MachineOperand &MO,
                                                BuildFnTy &MatchInfo) const;

  /// Remove references to rhs if it is undef
  bool matchShuffleUndefRHS(MachineInstr &MI, BuildFnTy &MatchInfo) const;

  /// Turn shuffle a, b, mask -> shuffle undef, b, mask iff mask does not
  /// reference a.
  bool matchShuffleDisjointMask(MachineInstr &MI, BuildFnTy &MatchInfo) const;

  /// Use a function which takes in a MachineIRBuilder to perform a combine.
  /// By default, it erases the instruction def'd on \p MO from the function.
  void applyBuildFnMO(const MachineOperand &MO, BuildFnTy &MatchInfo) const;

  /// Match FPOWI if it's safe to extend it into a series of multiplications.
  bool matchFPowIExpansion(MachineInstr &MI, int64_t Exponent) const;

  /// Expands FPOWI into a series of multiplications and a division if the
  /// exponent is negative.
  void applyExpandFPowI(MachineInstr &MI, int64_t Exponent) const;

  /// Combine insert vector element OOB.
  bool matchInsertVectorElementOOB(MachineInstr &MI,
                                   BuildFnTy &MatchInfo) const;

  bool matchFreezeOfSingleMaybePoisonOperand(MachineInstr &MI,
                                             BuildFnTy &MatchInfo) const;

  bool matchAddOfVScale(const MachineOperand &MO, BuildFnTy &MatchInfo) const;

  bool matchMulOfVScale(const MachineOperand &MO, BuildFnTy &MatchInfo) const;

  bool matchSubOfVScale(const MachineOperand &MO, BuildFnTy &MatchInfo) const;

  bool matchShlOfVScale(const MachineOperand &MO, BuildFnTy &MatchInfo) const;

  /// Transform trunc ([asz]ext x) to x or ([asz]ext x) or (trunc x).
  bool matchTruncateOfExt(const MachineInstr &Root, const MachineInstr &ExtMI,
                          BuildFnTy &MatchInfo) const;

  bool matchCastOfSelect(const MachineInstr &Cast, const MachineInstr &SelectMI,
                         BuildFnTy &MatchInfo) const;
  bool matchFoldAPlusC1MinusC2(const MachineInstr &MI,
                               BuildFnTy &MatchInfo) const;

  bool matchFoldC2MinusAPlusC1(const MachineInstr &MI,
                               BuildFnTy &MatchInfo) const;

  bool matchFoldAMinusC1MinusC2(const MachineInstr &MI,
                                BuildFnTy &MatchInfo) const;

  bool matchFoldC1Minus2MinusC2(const MachineInstr &MI,
                                BuildFnTy &MatchInfo) const;

  // fold ((A-C1)+C2) -> (A+(C2-C1))
  bool matchFoldAMinusC1PlusC2(const MachineInstr &MI,
                               BuildFnTy &MatchInfo) const;

  bool matchExtOfExt(const MachineInstr &FirstMI, const MachineInstr &SecondMI,
                     BuildFnTy &MatchInfo) const;

  bool matchCastOfBuildVector(const MachineInstr &CastMI,
                              const MachineInstr &BVMI,
                              BuildFnTy &MatchInfo) const;

  bool matchCanonicalizeICmp(const MachineInstr &MI,
                             BuildFnTy &MatchInfo) const;
  bool matchCanonicalizeFCmp(const MachineInstr &MI,
                             BuildFnTy &MatchInfo) const;

  // unmerge_values(anyext(build vector)) -> build vector(anyext)
  bool matchUnmergeValuesAnyExtBuildVector(const MachineInstr &MI,
                                           BuildFnTy &MatchInfo) const;

  // merge_values(_, undef) -> anyext
  bool matchMergeXAndUndef(const MachineInstr &MI, BuildFnTy &MatchInfo) const;

  // merge_values(_, zero) -> zext
  bool matchMergeXAndZero(const MachineInstr &MI, BuildFnTy &MatchInfo) const;

  // overflow sub
  bool matchSuboCarryOut(const MachineInstr &MI, BuildFnTy &MatchInfo) const;

private:
  /// Checks for legality of an indexed variant of \p LdSt.
  bool isIndexedLoadStoreLegal(GLoadStore &LdSt) const;
  /// Given a non-indexed load or store instruction \p MI, find an offset that
  /// can be usefully and legally folded into it as a post-indexing operation.
  ///
  /// \returns true if a candidate is found.
  bool findPostIndexCandidate(GLoadStore &MI, Register &Addr, Register &Base,
                              Register &Offset, bool &RematOffset) const;

  /// Given a non-indexed load or store instruction \p MI, find an offset that
  /// can be usefully and legally folded into it as a pre-indexing operation.
  ///
  /// \returns true if a candidate is found.
  bool findPreIndexCandidate(GLoadStore &MI, Register &Addr, Register &Base,
                             Register &Offset) const;

  /// Helper function for matchLoadOrCombine. Searches for Registers
  /// which may have been produced by a load instruction + some arithmetic.
  ///
  /// \param [in] Root - The search root.
  ///
  /// \returns The Registers found during the search.
  std::optional<SmallVector<Register, 8>>
  findCandidatesForLoadOrCombine(const MachineInstr *Root) const;

  /// Helper function for matchLoadOrCombine.
  ///
  /// Checks if every register in \p RegsToVisit is defined by a load
  /// instruction + some arithmetic.
  ///
  /// \param [out] MemOffset2Idx - Maps the byte positions each load ends up
  /// at to the index of the load.
  /// \param [in] MemSizeInBits - The number of bits each load should produce.
  ///
  /// \returns On success, a 3-tuple containing lowest-index load found, the
  /// lowest index, and the last load in the sequence.
  std::optional<std::tuple<GZExtLoad *, int64_t, GZExtLoad *>>
  findLoadOffsetsForLoadOrCombine(
      SmallDenseMap<int64_t, int64_t, 8> &MemOffset2Idx,
      const SmallVector<Register, 8> &RegsToVisit,
      const unsigned MemSizeInBits) const;

  /// Examines the G_PTR_ADD instruction \p PtrAdd and determines if performing
  /// a re-association of its operands would break an existing legal addressing
  /// mode that the address computation currently represents.
  bool reassociationCanBreakAddressingModePattern(MachineInstr &PtrAdd) const;

  /// Behavior when a floating point min/max is given one NaN and one
  /// non-NaN as input.
  enum class SelectPatternNaNBehaviour {
    NOT_APPLICABLE = 0, /// NaN behavior not applicable.
    RETURNS_NAN,        /// Given one NaN input, returns the NaN.
    RETURNS_OTHER,      /// Given one NaN input, returns the non-NaN.
    RETURNS_ANY /// Given one NaN input, can return either (or both operands are
                /// known non-NaN.)
  };

  /// \returns which of \p LHS and \p RHS would be the result of a non-equality
  /// floating point comparison where one of \p LHS and \p RHS may be NaN.
  ///
  /// If both \p LHS and \p RHS may be NaN, returns
  /// SelectPatternNaNBehaviour::NOT_APPLICABLE.
  SelectPatternNaNBehaviour
  computeRetValAgainstNaN(Register LHS, Register RHS,
                          bool IsOrderedComparison) const;

  /// Determines the floating point min/max opcode which should be used for
  /// a G_SELECT fed by a G_FCMP with predicate \p Pred.
  ///
  /// \returns 0 if this G_SELECT should not be combined to a floating point
  /// min or max. If it should be combined, returns one of
  ///
  /// * G_FMAXNUM
  /// * G_FMAXIMUM
  /// * G_FMINNUM
  /// * G_FMINIMUM
  ///
  /// Helper function for matchFPSelectToMinMax.
  unsigned getFPMinMaxOpcForSelect(CmpInst::Predicate Pred, LLT DstTy,
                                   SelectPatternNaNBehaviour VsNaNRetVal) const;

  /// Handle floating point cases for matchSimplifySelectToMinMax.
  ///
  /// E.g.
  ///
  /// select (fcmp uge x, 1.0) x, 1.0 -> fmax x, 1.0
  /// select (fcmp uge x, 1.0) 1.0, x -> fminnm x, 1.0
  bool matchFPSelectToMinMax(Register Dst, Register Cond, Register TrueVal,
                             Register FalseVal, BuildFnTy &MatchInfo) const;

  /// Try to fold selects to logical operations.
  bool tryFoldBoolSelectToLogic(GSelect *Select, BuildFnTy &MatchInfo) const;

  bool tryFoldSelectOfConstants(GSelect *Select, BuildFnTy &MatchInfo) const;

  bool isOneOrOneSplat(Register Src, bool AllowUndefs) const;
  bool isZeroOrZeroSplat(Register Src, bool AllowUndefs) const;
  bool isConstantSplatVector(Register Src, int64_t SplatValue,
                             bool AllowUndefs) const;
  bool isConstantOrConstantVectorI(Register Src) const;

  std::optional<APInt> getConstantOrConstantSplatVector(Register Src) const;

  /// Fold (icmp Pred1 V1, C1) && (icmp Pred2 V2, C2)
  /// or   (icmp Pred1 V1, C1) || (icmp Pred2 V2, C2)
  /// into a single comparison using range-based reasoning.
  bool tryFoldAndOrOrICmpsUsingRanges(GLogicalBinOp *Logic,
                                      BuildFnTy &MatchInfo) const;

  // Simplify (cmp cc0 x, y) (&& or ||) (cmp cc1 x, y) -> cmp cc2 x, y.
  bool tryFoldLogicOfFCmps(GLogicalBinOp *Logic, BuildFnTy &MatchInfo) const;

  bool isCastFree(unsigned Opcode, LLT ToTy, LLT FromTy) const;

  bool constantFoldICmp(const GICmp &ICmp, const GIConstant &LHSCst,
                        const GIConstant &RHSCst, BuildFnTy &MatchInfo) const;
  bool constantFoldFCmp(const GFCmp &FCmp, const GFConstant &LHSCst,
                        const GFConstant &RHSCst, BuildFnTy &MatchInfo) const;
};
} // namespace llvm

#endif
