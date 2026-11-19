//===-- AMDGPUBitOp3.h - V_BITOP3 cost check helpers ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// The V_BITOP3 cost check that SelectionDAG and GlobalISel share. It sees the
/// values of a selector through a view (BitOp3DAGView, BitOp3MIRView) with
/// these members:
///   isLogicOp(V), getOperands(V)  V is an AND, OR or XOR, and its operands
///   areUsersIn(V, Removed)        every user of V is in Removed
///   getOp(V), isAllOnes(V), isDivergent(V), hasOneUse(V)
///                                 for matchMultiOpInst
///   is16Bit(V), hasSDWA(), isByteMask(V)
///                                 for the SDWA byte mask rule
///   needsBus(V), isLiteral(V), hasVOP3Literal(), getBusLimit(),
///   hasOtherRegisterUse(V, Removed), isUsedInOtherBlock(V)
///                                 for getNumVOP3Moves
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUBITOP3_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUBITOP3_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include <optional>
#include <utility>

namespace llvm::AMDGPU {

/// Return the logic operations that one V_BITOP3 for \p Root replaces: \p Root
/// itself, and every other operation that BitOp3_Op matched (a logic operation
/// that is not one of the sources \p Src) whose users are all replaced too.
/// Only the first 16 matched operations are looked at and the rest count as
/// kept, so the result never overstates what V_BITOP3 replaces. Users come
/// before their operands.
template <typename ValueT, typename ViewT>
SmallVector<ValueT, 8> getBitOp3RemovedOps(ValueT Root, ArrayRef<ValueT> Src,
                                           const ViewT &View) {
  constexpr unsigned MaxMatched = 16;
  SmallVector<ValueT, 8> Matched;
  SmallVector<ValueT, 8> Worklist = {Root};
  while (!Worklist.empty() && Matched.size() < MaxMatched) {
    ValueT V = Worklist.pop_back_val();
    if (is_contained(Matched, V) || !View.isLogicOp(V) || is_contained(Src, V))
      continue;
    Matched.push_back(V);
    auto [A, B] = View.getOperands(V);
    Worklist.append({B, A});
  }

  SmallVector<ValueT, 8> Removed = {Root};
  for (bool Changed = true; Changed;) {
    Changed = false;
    for (ValueT V : Matched) {
      if (is_contained(Removed, V) ||
          !View.areUsersIn(V, ArrayRef<ValueT>(Removed)))
        continue;
      Removed.push_back(V);
      Changed = true;
    }
  }
  return Removed;
}

/// The logic operation that defines a value, as the V_BITOP3 cost check sees
/// it. Only 32-bit operations are reported: the instructions below have no
/// other forms.
enum class LogicOp { None, And, Or, Xor };

/// A VALU instruction other than V_BITOP3 that computes several logic
/// operations: the operations it absorbs besides its root, and its operands.
template <typename ValueT> struct MultiOpInst {
  SmallVector<ValueT, 3> Absorbed;
  SmallVector<ValueT, 3> Operands;
};

/// Without V_BITOP3 the selector covers these groups of logic operations with
/// one instruction each, so V_BITOP3 saves nothing on them:
///   V_OR3_B32     (a | b) | c
///   V_AND_OR_B32  (a & b) | c
///   V_BFI_B32     (a & b) | (~a & c),  c ^ (a & (b ^ c)),  b & ~a,  b | ~a
/// Return the one that matches at the divergent operation \p P, if any. An
/// absorbed operation must be divergent, have no other user, and be
/// available: \p IsAvailable(V) says whether V can still be absorbed.
template <typename ValueT, typename ViewT, typename AvailableFn>
std::optional<MultiOpInst<ValueT>> matchMultiOpInst(ValueT P, const ViewT &View,
                                                    AvailableFn IsAvailable) {
  LogicOp Op = View.getOp(P);
  if (Op == LogicOp::None || !View.isDivergent(P))
    return std::nullopt;
  auto IsAbsorbable = [&](ValueT V, LogicOp Want) {
    return View.getOp(V) == Want && View.isDivergent(V) && View.hasOneUse(V) &&
           IsAvailable(V);
  };
  // not(x) is xor(x, -1); return x.
  auto GetNotOperand = [&](ValueT V) -> std::optional<ValueT> {
    if (!IsAbsorbable(V, LogicOp::Xor))
      return std::nullopt;
    auto [A, B] = View.getOperands(V);
    if (View.isAllOnes(B))
      return A;
    if (View.isAllOnes(A))
      return B;
    return std::nullopt;
  };
  auto [L, R] = View.getOperands(P);
  const std::pair<ValueT, ValueT> Orders[] = {{L, R}, {R, L}};

  if (Op == LogicOp::Or) {
    // V_BFI_B32: (a & b) | (~a & c).
    for (auto [X, Y] : Orders) {
      if (!IsAbsorbable(X, LogicOp::And) || !IsAbsorbable(Y, LogicOp::And))
        continue;
      auto [XA, XB] = View.getOperands(X);
      auto [YA, YB] = View.getOperands(Y);
      for (auto [N, C] : {std::pair(YA, YB), std::pair(YB, YA)}) {
        std::optional<ValueT> A = GetNotOperand(N);
        if (A && (*A == XA || *A == XB))
          return MultiOpInst<ValueT>{{X, Y, N}, {*A, *A == XA ? XB : XA, C}};
      }
    }
    // V_OR3_B32: (a | b) | c, and V_AND_OR_B32: (a & b) | c.
    for (auto [X, Y] : Orders) {
      if (IsAbsorbable(X, LogicOp::Or) || IsAbsorbable(X, LogicOp::And)) {
        auto [A, B] = View.getOperands(X);
        return MultiOpInst<ValueT>{{X}, {A, B, Y}};
      }
    }
  }
  if (Op == LogicOp::Xor) {
    // V_BFI_B32: c ^ (a & (b ^ c)).
    for (auto [C, X] : Orders) {
      if (!IsAbsorbable(X, LogicOp::And))
        continue;
      auto [XA, XB] = View.getOperands(X);
      for (auto [A, T] : {std::pair(XA, XB), std::pair(XB, XA)}) {
        if (!IsAbsorbable(T, LogicOp::Xor))
          continue;
        auto [TA, TB] = View.getOperands(T);
        if (TA == C || TB == C)
          return MultiOpInst<ValueT>{{X, T}, {A, TA == C ? TB : TA, C}};
      }
    }
  }
  if (Op == LogicOp::And || Op == LogicOp::Or) {
    // V_BFI_B32: b & ~a, and b | ~a.
    for (auto [B, N] : Orders)
      if (std::optional<ValueT> A = GetNotOperand(N))
        return MultiOpInst<ValueT>{{N}, {*A, B}};
  }
  return std::nullopt;
}

/// Records which blocks of a function use each constant in an and, or or xor
/// instruction. The selectors see one block at a time, but MachineCSE and
/// MachineLICM can leave one move for a literal that several blocks use.
class LogicConstantBlocks {
  DenseMap<uint64_t, SmallVector<const BasicBlock *, 2>> Blocks;
  bool Built = false;

  /// Return \p V if it is an integer constant of at most 64 bits, or the
  /// splat value of a vector of them.
  static const ConstantInt *getIntConstant(const Value *V) {
    const auto *C = dyn_cast<ConstantInt>(V);
    if (!C && V->getType()->isVectorTy())
      if (const auto *CV = dyn_cast<Constant>(V))
        C = dyn_cast_or_null<ConstantInt>(CV->getSplatValue());
    return C && C->getBitWidth() <= 64 ? C : nullptr;
  }

  void build(const Function &F) {
    for (const BasicBlock &B : F) {
      for (const Instruction &I : B) {
        if (!I.isBitwiseLogicOp())
          continue;
        for (const Value *Op : I.operands()) {
          const ConstantInt *C = getIntConstant(Op);
          if (!C)
            continue;
          SmallVector<const BasicBlock *, 2> &UseBlocks =
              Blocks[C->getZExtValue()];
          if (!is_contained(UseBlocks, &B))
            UseBlocks.push_back(&B);
        }
      }
    }
    Built = true;
  }

public:
  /// Forget the previous function.
  void reset() {
    Blocks.clear();
    Built = false;
  }

  /// Return whether an and, or or xor instruction in another block of the
  /// function of \p BB uses the constant \p Val. \p BB may be null.
  bool isUsedInOtherBlock(const BasicBlock *BB, uint64_t Val) {
    if (!BB)
      return false;
    if (!Built)
      build(*BB->getParent());
    auto It = Blocks.find(Val);
    return It != Blocks.end() && any_of(It->second, llvm::not_equal_to(BB));
  }
};

/// Return how many moves a VOP3 needs for its sources \p Srcs: one for each
/// constant-bus read over the limit and, on subtargets without VOP3
/// literals, one for each literal. A literal that another instruction may
/// also need in a register, or that a logic operation in another block
/// uses, counts as a constant-bus read instead, as its move into an SGPR is
/// likely shared; uses by the operations in \p Removed do not count.
template <typename ValueT, typename ViewT>
unsigned getNumVOP3Moves(ArrayRef<ValueT> Srcs, ArrayRef<ValueT> Removed,
                         const ViewT &View) {
  unsigned Reads = 0, Literals = 0;
  for (auto [I, V] : enumerate(Srcs)) {
    if (is_contained(Srcs.take_front(I), V) || !View.needsBus(V))
      continue;
    if (!View.hasVOP3Literal() && View.isLiteral(V) &&
        !View.hasOtherRegisterUse(V, Removed) && !View.isUsedInOtherBlock(V))
      ++Literals;
    else
      ++Reads;
  }
  unsigned Limit = View.getBusLimit();
  return Literals + (Reads > Limit ? Reads - Limit : 0);
}

/// Return how many instructions the operations in \p Removed need without
/// V_BITOP3: one each, except that an instruction from matchMultiOpInst also
/// covers the operations it absorbs, plus the moves for its operands
/// (getNumVOP3Moves). \p Removed lists users before their operands, so the
/// instructions form from the top, as the selector forms them.
template <typename ValueT, typename ViewT>
unsigned getNumInstsWithoutBitOp3(ArrayRef<ValueT> Removed, const ViewT &View) {
  SmallVector<ValueT, 8> Absorbed;
  unsigned NumInsts = 0;
  for (ValueT P : Removed) {
    if (is_contained(Absorbed, P))
      continue;
    ++NumInsts;
    std::optional<MultiOpInst<ValueT>> Inst =
        matchMultiOpInst(P, View, [&](ValueT V) {
          return is_contained(Removed, V) && !is_contained(Absorbed, V);
        });
    if (!Inst)
      continue;
    unsigned Moves =
        getNumVOP3Moves(ArrayRef<ValueT>(Inst->Operands), Removed, View);
    // Use the instruction only if it beats one instruction per operation.
    if (Moves >= Inst->Absorbed.size())
      continue;
    Absorbed.append(Inst->Absorbed);
    NumInsts += Moves;
  }
  return NumInsts;
}

/// Return true if replacing the logic operations that BitOp3_Op matched
/// between the divergent \p Root and \p Src with one V_BITOP3 removes an
/// instruction. A matched operation is only removed if all of its users are
/// removed too, some groups of operations are one instruction anyway
/// (matchMultiOpInst), and V_BITOP3 needs moves for some of its sources
/// (getNumVOP3Moves).
template <typename ValueT, typename ViewT>
bool isBitOp3Profitable(ValueT Root, ArrayRef<ValueT> Src, const ViewT &View) {
  SmallVector<ValueT, 8> Removed = getBitOp3RemovedOps(Root, Src, View);
  unsigned NumInsts = getNumInstsWithoutBitOp3(ArrayRef<ValueT>(Removed), View);
  // With SDWA, a byte mask folds into the operand select of a VOP1 or VOP2
  // user, so it costs nothing without V_BITOP3, also at the root. 16-bit
  // roots come from byte packing.
  if (View.hasSDWA() && View.is16Bit(Root))
    NumInsts -= static_cast<unsigned>(
        count_if(Removed, [&](ValueT V) { return View.isByteMask(V); }));
  // V_BITOP3 and its moves must need at least one instruction less.
  return NumInsts >= getNumVOP3Moves(Src, ArrayRef<ValueT>(Removed), View) + 2;
}

} // namespace llvm::AMDGPU

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUBITOP3_H
