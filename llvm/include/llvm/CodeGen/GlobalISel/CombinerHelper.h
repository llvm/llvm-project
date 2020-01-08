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

namespace llvm {

class GISelChangeObserver;
class MachineIRBuilder;
class MachineRegisterInfo;
class MachineInstr;
class MachineOperand;
class GISelKnownBits;
class MachineDominatorTree;

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

class CombinerHelper {
protected:
  MachineIRBuilder &Builder;
  MachineRegisterInfo &MRI;
  GISelChangeObserver &Observer;
  GISelKnownBits *KB;
  MachineDominatorTree *MDT;

public:
  CombinerHelper(GISelChangeObserver &Observer, MachineIRBuilder &B,
                 GISelKnownBits *KB = nullptr,
                 MachineDominatorTree *MDT = nullptr);

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
  bool isPredecessor(MachineInstr &DefMI, MachineInstr &UseMI);

  /// Returns true if \p DefMI dominates \p UseMI. By definition an
  /// instruction dominates itself.
  ///
  /// If we haven't been provided with a MachineDominatorTree during
  /// construction, this function returns a conservative result that tracks just
  /// a single basic block.
  bool dominates(MachineInstr &DefMI, MachineInstr &UseMI);

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

  bool matchElideBrByInvertingCond(MachineInstr &MI);
  void applyElideBrByInvertingCond(MachineInstr &MI);
  bool tryElideBrByInvertingCond(MachineInstr &MI);

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

  /// Try to transform \p MI by using all of the above
  /// combine functions. Returns true if changed.
  bool tryCombine(MachineInstr &MI);

private:
  // Memcpy family optimization helpers.
  bool optimizeMemcpy(MachineInstr &MI, Register Dst, Register Src,
                      unsigned KnownLen, unsigned DstAlign, unsigned SrcAlign,
                      bool IsVolatile);
  bool optimizeMemmove(MachineInstr &MI, Register Dst, Register Src,
                      unsigned KnownLen, unsigned DstAlign, unsigned SrcAlign,
                      bool IsVolatile);
  bool optimizeMemset(MachineInstr &MI, Register Dst, Register Val,
                      unsigned KnownLen, unsigned DstAlign, bool IsVolatile);

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
