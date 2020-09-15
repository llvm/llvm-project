//===-- llvm/CodeGen/GlobalISel/CombinerHelper.h --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//
//
/// This contains common combine transformations that may be used in a combine
/// pass,or by the target elsewhere.
/// Targets can pick individual opcode transformations from the helper or use
/// tryCombine which invokes all transformations. All of the transformations
/// return true if the MachineInstruction changed and false otherwise.
//
//===--------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_COMBINER_HELPER_H
#define LLVM_CODEGEN_GLOBALISEL_COMBINER_HELPER_H

#include "llvm/CodeGen/LowLevelType.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/Support/Alignment.h"

namespace llvm {

class GISelChangeObserver;
class MachineIRBuilder;
class MachineInstrBuilder;
class MachineRegisterInfo;
class MachineInstr;
class MachineOperand;
class GISelKnownBits;
class MachineDominatorTree;
class LegalizerInfo;
struct LegalityQuery;
class TargetLowering;

struct PreferredTuple {
  LLT Ty;                // The result type of the extend.
  unsigned ExtendOpcode; // G_ANYEXT/G_SEXT/G_ZEXT
  MachineInstr *MI;
};

struct IndexedLoadStoreMatchInfo {
  Register Addr;
  Register Base;
  Register Offset;
  bool IsPre;
};

struct PtrAddChain {
  int64_t Imm;
  Register Base;
};

struct RegisterImmPair {
  Register Reg;
  int64_t Imm;
};

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
  const LegalizerInfo *LI;

public:
  CombinerHelper(GISelChangeObserver &Observer, MachineIRBuilder &B,
                 GISelKnownBits *KB = nullptr,
                 MachineDominatorTree *MDT = nullptr,
                 const LegalizerInfo *LI = nullptr);

  GISelKnownBits *getKnownBits() const {
    return KB;
  }

  const TargetLowering &getTargetLowering() const;

  /// \return true if the combine is running prior to legalization, or if \p
  /// Query is legal on the target.
  bool isLegalOrBeforeLegalizer(const LegalityQuery &Query) const;

  /// MachineRegisterInfo::replaceRegWith() and inform the observer of the changes
  void replaceRegWith(MachineRegisterInfo &MRI, Register FromReg, Register ToReg) const;

  /// Replace a single register operand with a new register and inform the
  /// observer of the changes.
  void replaceRegOpWith(MachineRegisterInfo &MRI, MachineOperand &FromRegOp,
                        Register ToReg) const;

  /// If \p MI is COPY, try to combine it.
  /// Returns true if MI changed.
  bool tryCombineCopy(MachineInstr &MI);
  bool matchCombineCopy(MachineInstr &MI);
  void applyCombineCopy(MachineInstr &MI);

  /// Returns true if \p DefMI precedes \p UseMI or they are the same
  /// instruction. Both must be in the same basic block.
  bool isPredecessor(const MachineInstr &DefMI, const MachineInstr &UseMI);

  /// Returns true if \p DefMI dominates \p UseMI. By definition an
  /// instruction dominates itself.
  ///
  /// If we haven't been provided with a MachineDominatorTree during
  /// construction, this function returns a conservative result that tracks just
  /// a single basic block.
  bool dominates(const MachineInstr &DefMI, const MachineInstr &UseMI);

  /// If \p MI is extend that consumes the result of a load, try to combine it.
  /// Returns true if MI changed.
  bool tryCombineExtendingLoads(MachineInstr &MI);
  bool matchCombineExtendingLoads(MachineInstr &MI, PreferredTuple &MatchInfo);
  void applyCombineExtendingLoads(MachineInstr &MI, PreferredTuple &MatchInfo);

  /// Combine \p MI into a pre-indexed or post-indexed load/store operation if
  /// legal and the surrounding code makes it useful.
  bool tryCombineIndexedLoadStore(MachineInstr &MI);
  bool matchCombineIndexedLoadStore(MachineInstr &MI, IndexedLoadStoreMatchInfo &MatchInfo);
  void applyCombineIndexedLoadStore(MachineInstr &MI, IndexedLoadStoreMatchInfo &MatchInfo);

  bool matchSextTruncSextLoad(MachineInstr &MI);
  bool applySextTruncSextLoad(MachineInstr &MI);

  /// Match sext_inreg(load p), imm -> sextload p
  bool matchSextInRegOfLoad(MachineInstr &MI, std::tuple<Register, unsigned> &MatchInfo);
  bool applySextInRegOfLoad(MachineInstr &MI, std::tuple<Register, unsigned> &MatchInfo);

  /// If a brcond's true block is not the fallthrough, make it so by inverting
  /// the condition and swapping operands.
  bool matchOptBrCondByInvertingCond(MachineInstr &MI);
  void applyOptBrCondByInvertingCond(MachineInstr &MI);

  /// If \p MI is G_CONCAT_VECTORS, try to combine it.
  /// Returns true if MI changed.
  /// Right now, we support:
  /// - concat_vector(undef, undef) => undef
  /// - concat_vector(build_vector(A, B), build_vector(C, D)) =>
  ///   build_vector(A, B, C, D)
  ///
  /// \pre MI.getOpcode() == G_CONCAT_VECTORS.
  bool tryCombineConcatVectors(MachineInstr &MI);
  /// Check if the G_CONCAT_VECTORS \p MI is undef or if it
  /// can be flattened into a build_vector.
  /// In the first case \p IsUndef will be true.
  /// In the second case \p Ops will contain the operands needed
  /// to produce the flattened build_vector.
  ///
  /// \pre MI.getOpcode() == G_CONCAT_VECTORS.
  bool matchCombineConcatVectors(MachineInstr &MI, bool &IsUndef,
                                 SmallVectorImpl<Register> &Ops);
  /// Replace \p MI with a flattened build_vector with \p Ops or an
  /// implicit_def if IsUndef is true.
  void applyCombineConcatVectors(MachineInstr &MI, bool IsUndef,
                                 const ArrayRef<Register> Ops);

  /// Try to combine G_SHUFFLE_VECTOR into G_CONCAT_VECTORS.
  /// Returns true if MI changed.
  ///
  /// \pre MI.getOpcode() == G_SHUFFLE_VECTOR.
  bool tryCombineShuffleVector(MachineInstr &MI);
  /// Check if the G_SHUFFLE_VECTOR \p MI can be replaced by a
  /// concat_vectors.
  /// \p Ops will contain the operands needed to produce the flattened
  /// concat_vectors.
  ///
  /// \pre MI.getOpcode() == G_SHUFFLE_VECTOR.
  bool matchCombineShuffleVector(MachineInstr &MI,
                                 SmallVectorImpl<Register> &Ops);
  /// Replace \p MI with a concat_vectors with \p Ops.
  void applyCombineShuffleVector(MachineInstr &MI,
                                 const ArrayRef<Register> Ops);

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
  bool tryCombineMemCpyFamily(MachineInstr &MI, unsigned MaxLen = 0);

  bool matchPtrAddImmedChain(MachineInstr &MI, PtrAddChain &MatchInfo);
  bool applyPtrAddImmedChain(MachineInstr &MI, PtrAddChain &MatchInfo);

  /// Transform a multiply by a power-of-2 value to a left shift.
  bool matchCombineMulToShl(MachineInstr &MI, unsigned &ShiftVal);
  bool applyCombineMulToShl(MachineInstr &MI, unsigned &ShiftVal);

  // Transform a G_SHL with an extended source into a narrower shift if
  // possible.
  bool matchCombineShlOfExtend(MachineInstr &MI, RegisterImmPair &MatchData);
  bool applyCombineShlOfExtend(MachineInstr &MI,
                               const RegisterImmPair &MatchData);

  /// Reduce a shift by a constant to an unmerge and a shift on a half sized
  /// type. This will not produce a shift smaller than \p TargetShiftSize.
  bool matchCombineShiftToUnmerge(MachineInstr &MI, unsigned TargetShiftSize,
                                 unsigned &ShiftVal);
  bool applyCombineShiftToUnmerge(MachineInstr &MI, const unsigned &ShiftVal);
  bool tryCombineShiftToUnmerge(MachineInstr &MI, unsigned TargetShiftAmount);

  /// Transform <ty,...> G_UNMERGE(G_MERGE ty X, Y, Z) -> ty X, Y, Z.
  bool
  matchCombineUnmergeMergeToPlainValues(MachineInstr &MI,
                                        SmallVectorImpl<Register> &Operands);
  bool
  applyCombineUnmergeMergeToPlainValues(MachineInstr &MI,
                                        SmallVectorImpl<Register> &Operands);

  /// Transform G_UNMERGE Constant -> Constant1, Constant2, ...
  bool matchCombineUnmergeConstant(MachineInstr &MI,
                                   SmallVectorImpl<APInt> &Csts);
  bool applyCombineUnmergeConstant(MachineInstr &MI,
                                   SmallVectorImpl<APInt> &Csts);

  /// Transform X, Y<dead> = G_UNMERGE Z -> X = G_TRUNC Z.
  bool matchCombineUnmergeWithDeadLanesToTrunc(MachineInstr &MI);
  bool applyCombineUnmergeWithDeadLanesToTrunc(MachineInstr &MI);

  /// Transform X, Y = G_UNMERGE(G_ZEXT(Z)) -> X = G_ZEXT(Z); Y = G_CONSTANT 0
  bool matchCombineUnmergeZExtToZExt(MachineInstr &MI);
  bool applyCombineUnmergeZExtToZExt(MachineInstr &MI);

  /// Transform IntToPtr(PtrToInt(x)) to x if cast is in the same address space.
  bool matchCombineI2PToP2I(MachineInstr &MI, Register &Reg);
  bool applyCombineI2PToP2I(MachineInstr &MI, Register &Reg);

  /// Transform PtrToInt(IntToPtr(x)) to x.
  bool matchCombineP2IToI2P(MachineInstr &MI, Register &Reg);
  bool applyCombineP2IToI2P(MachineInstr &MI, Register &Reg);

  /// Transform G_ADD (G_PTRTOINT x), y -> G_PTRTOINT (G_PTR_ADD x, y)
  /// Transform G_ADD y, (G_PTRTOINT x) -> G_PTRTOINT (G_PTR_ADD x, y)
  bool matchCombineAddP2IToPtrAdd(MachineInstr &MI,
                                  std::pair<Register, bool> &PtrRegAndCommute);
  bool applyCombineAddP2IToPtrAdd(MachineInstr &MI,
                                  std::pair<Register, bool> &PtrRegAndCommute);

  /// Transform anyext(trunc(x)) to x.
  bool matchCombineAnyExtTrunc(MachineInstr &MI, Register &Reg);
  bool applyCombineAnyExtTrunc(MachineInstr &MI, Register &Reg);

  /// Transform [asz]ext([asz]ext(x)) to [asz]ext x.
  bool matchCombineExtOfExt(MachineInstr &MI,
                            std::tuple<Register, unsigned> &MatchInfo);
  bool applyCombineExtOfExt(MachineInstr &MI,
                            std::tuple<Register, unsigned> &MatchInfo);

  /// Transform fneg(fneg(x)) to x.
  bool matchCombineFNegOfFNeg(MachineInstr &MI, Register &Reg);

  /// Match fabs(fabs(x)) to fabs(x).
  bool matchCombineFAbsOfFAbs(MachineInstr &MI, Register &Src);
  bool applyCombineFAbsOfFAbs(MachineInstr &MI, Register &Src);

  /// Return true if any explicit use operand on \p MI is defined by a
  /// G_IMPLICIT_DEF.
  bool matchAnyExplicitUseIsUndef(MachineInstr &MI);

  /// Return true if all register explicit use operands on \p MI are defined by
  /// a G_IMPLICIT_DEF.
  bool matchAllExplicitUsesAreUndef(MachineInstr &MI);

  /// Return true if a G_SHUFFLE_VECTOR instruction \p MI has an undef mask.
  bool matchUndefShuffleVectorMask(MachineInstr &MI);

  /// Return true if a G_STORE instruction \p MI is storing an undef value.
  bool matchUndefStore(MachineInstr &MI);

  /// Return true if a G_SELECT instruction \p MI has an undef comparison.
  bool matchUndefSelectCmp(MachineInstr &MI);

  /// Return true if a G_SELECT instruction \p MI has a constant comparison. If
  /// true, \p OpIdx will store the operand index of the known selected value.
  bool matchConstantSelectCmp(MachineInstr &MI, unsigned &OpIdx);

  /// Replace an instruction with a G_FCONSTANT with value \p C.
  bool replaceInstWithFConstant(MachineInstr &MI, double C);

  /// Replace an instruction with a G_CONSTANT with value \p C.
  bool replaceInstWithConstant(MachineInstr &MI, int64_t C);

  /// Replace an instruction with a G_IMPLICIT_DEF.
  bool replaceInstWithUndef(MachineInstr &MI);

  /// Delete \p MI and replace all of its uses with its \p OpIdx-th operand.
  bool replaceSingleDefInstWithOperand(MachineInstr &MI, unsigned OpIdx);

  /// Delete \p MI and replace all of its uses with \p Replacement.
  bool replaceSingleDefInstWithReg(MachineInstr &MI, Register Replacement);

  /// Return true if \p MOP1 and \p MOP2 are register operands are defined by
  /// equivalent instructions.
  bool matchEqualDefs(const MachineOperand &MOP1, const MachineOperand &MOP2);

  /// Return true if \p MOP is defined by a G_CONSTANT with a value equal to
  /// \p C.
  bool matchConstantOp(const MachineOperand &MOP, int64_t C);

  /// Optimize (cond ? x : x) -> x
  bool matchSelectSameVal(MachineInstr &MI);

  /// Optimize (x op x) -> x
  bool matchBinOpSameVal(MachineInstr &MI);

  /// Check if operand \p OpIdx is zero.
  bool matchOperandIsZero(MachineInstr &MI, unsigned OpIdx);

  /// Check if operand \p OpIdx is undef.
  bool matchOperandIsUndef(MachineInstr &MI, unsigned OpIdx);

  /// Erase \p MI
  bool eraseInst(MachineInstr &MI);

  /// Return true if MI is a G_ADD which can be simplified to a G_SUB.
  bool matchSimplifyAddToSub(MachineInstr &MI,
                             std::tuple<Register, Register> &MatchInfo);
  bool applySimplifyAddToSub(MachineInstr &MI,
                             std::tuple<Register, Register> &MatchInfo);

  /// Match (logic_op (op x...), (op y...)) -> (op (logic_op x, y))
  bool
  matchHoistLogicOpWithSameOpcodeHands(MachineInstr &MI,
                                       InstructionStepsMatchInfo &MatchInfo);

  /// Replace \p MI with a series of instructions described in \p MatchInfo.
  bool applyBuildInstructionSteps(MachineInstr &MI,
                                  InstructionStepsMatchInfo &MatchInfo);

  /// Match ashr (shl x, C), C -> sext_inreg (C)
  bool matchAshrShlToSextInreg(MachineInstr &MI,
                               std::tuple<Register, int64_t> &MatchInfo);
  bool applyAshShlToSextInreg(MachineInstr &MI,
                              std::tuple<Register, int64_t> &MatchInfo);
  /// \return true if \p MI is a G_AND instruction whose RHS is a mask where
  /// LHS & mask == LHS. (E.g., an all-ones value.)
  ///
  /// \param [in] MI - The G_AND instruction.
  /// \param [out] Replacement - A register the G_AND should be replaced with on
  /// success.
  bool matchAndWithTrivialMask(MachineInstr &MI, Register &Replacement);

  /// \return true if \p MI is a G_SEXT_INREG that can be erased.
  bool matchRedundantSExtInReg(MachineInstr &MI);

  /// Combine inverting a result of a compare into the opposite cond code.
  bool matchNotCmp(MachineInstr &MI, SmallVectorImpl<Register> &RegsToNegate);
  bool applyNotCmp(MachineInstr &MI, SmallVectorImpl<Register> &RegsToNegate);

  /// Try to transform \p MI by using all of the above
  /// combine functions. Returns true if changed.
  bool tryCombine(MachineInstr &MI);

private:
  // Memcpy family optimization helpers.
  bool optimizeMemcpy(MachineInstr &MI, Register Dst, Register Src,
                      unsigned KnownLen, Align DstAlign, Align SrcAlign,
                      bool IsVolatile);
  bool optimizeMemmove(MachineInstr &MI, Register Dst, Register Src,
                       unsigned KnownLen, Align DstAlign, Align SrcAlign,
                       bool IsVolatile);
  bool optimizeMemset(MachineInstr &MI, Register Dst, Register Val,
                      unsigned KnownLen, Align DstAlign, bool IsVolatile);

  /// Given a non-indexed load or store instruction \p MI, find an offset that
  /// can be usefully and legally folded into it as a post-indexing operation.
  ///
  /// \returns true if a candidate is found.
  bool findPostIndexCandidate(MachineInstr &MI, Register &Addr, Register &Base,
                              Register &Offset);

  /// Given a non-indexed load or store instruction \p MI, find an offset that
  /// can be usefully and legally folded into it as a pre-indexing operation.
  ///
  /// \returns true if a candidate is found.
  bool findPreIndexCandidate(MachineInstr &MI, Register &Addr, Register &Base,
                             Register &Offset);
};
} // namespace llvm

#endif
