//===- KnownBitsDataflow.h - Cache and invalidate KnownBits ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Caches KnownBits for Values, aand invalidates the cached results on IR
// updates by walking the dataflow graph. Provides a custom Map-like container
// with lookup and insertion APIs.
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_KNOWNBITSDATAFLOW_H
#define LLVM_ANALYSIS_KNOWNBITSDATAFLOW_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Pass.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/KnownBits.h"
#include <memory>

namespace llvm {
class Function;
class DataLayout;
class raw_ostream;
class KnownBitsDataflow;

/// GraphTraits enabling depth_first over Values.
template <typename NodeRef, typename ChildIteratorType>
struct NodeGraphTraitsBase {
  static NodeRef getEntryNode(NodeRef N) { return N; }
  static ChildIteratorType child_begin(NodeRef N) { // NOLINT
    return N->user_begin();
  }
  static ChildIteratorType child_end(NodeRef N) { // NOLINT
    return N->user_end();
  }
};

template <>
struct GraphTraits<Value *>
    : public NodeGraphTraitsBase<Value *, Value::user_iterator> {
  using NodeRef = Value *;
  using ChildIteratorType = Value::user_iterator;
};

/// A custom ValueHandle with callback to erase KnownBits in the cache when an
/// Instruction is deleted, or invalidate dependent KnownBits when it is
/// RAUW'ed.
class KnownBitsVH : private CallbackVH {
  friend class KnownBitsDataflow;
  KnownBitsDataflow *KBD;

  virtual void anchor() override;
  virtual void deleted() override;
  virtual void allUsesReplacedWith(Value *New) override;

public:
  KnownBitsVH(const Value *V, KnownBitsDataflow *KBD)
      : CallbackVH(V), KBD(KBD) {}
  virtual ~KnownBitsVH() = default;
  using ValueHandleBase::getValPtr;
  using CallbackVH::operator Value *;
  bool operator==(const KnownBitsVH &Other) const {
    return getValPtr() == Other.getValPtr();
  }
  bool operator<(const KnownBitsVH &Other) const {
    return getValPtr() < Other.getValPtr();
  }
};

// The DenseMapInfo for our KnownBits ValueHandle is just the DenseMapInfo on
// the Value pointer: there is no other information in the ValueHandle that's
// relevant for a DenseMap.
template <> struct DenseMapInfo<KnownBitsVH> {
  static unsigned getHashValue(const KnownBitsVH &Val) {
    return DenseMapInfo<const Value *>::getHashValue(Val.getValPtr());
  }

  static bool isEqual(const KnownBitsVH &LHS, const KnownBitsVH &RHS) {
    return DenseMapInfo<const Value *>::isEqual(LHS.getValPtr(),
                                                RHS.getValPtr());
  }
};

/// A DenseMap holding a ValueHandle, that performs lookups based on the
/// underlying Value.
template <typename ValueT>
using DenseMapForVH =
    DenseMap<KnownBitsVH, ValueT, DenseMapInfo<const Value *>>;

/// The ValueT of our DenseMap is actually a KnownBits augmented with
/// context-instruction information.
struct KnownBitsWithCtxI : public KnownBits {
  WeakVH CtxI;
  KnownBitsWithCtxI() = default;
  KnownBitsWithCtxI(const KnownBits &Known, const Instruction *CtxI)
      : KnownBits(Known), CtxI(const_cast<Instruction *>(CtxI)) {}
  bool canUseWith(const Instruction *Other) const {
    // If the cached value was computed with a CtxI, and one without a CxtI is
    // requested, returning the cached value would yield a better optimization
    // result.
    return !Other || Other == CtxI;
  }
};

/// A structure keeps a mapping between a custom ValueHandle and
/// KnownBitsWithCtxI, with core functionality to cache KnownBits with automatic
/// invalidation on IR manipulation. We compute a deterministic ordering for
/// entries in the map for testing and debugging.
class LLVM_ABI KnownBitsDataflow : protected DenseMapForVH<KnownBitsWithCtxI> {
  friend class KnownBitsVH;

  /// Do a forward data-flow walk, and find all ValueHandles whose KnownBits
  /// depeends on the KnownBits of \p V. Returns a range of Values.
  auto forwardDataflow(const KnownBitsVH &V) const {
    return make_filter_range(depth_first(V.getValPtr()),
                             bind_front(&KnownBitsDataflow::contains, this));
  }

protected:
  using BaseT = DenseMapForVH<KnownBitsWithCtxI>;

  LLVM_ABI_FOR_TEST KnownBitsVH key_as(const Value *V) const { // NOLINT
    auto It = find_as(V);
    assert(It != end() && "Expected to find ValueHandle");
    return It->first;
  }
  LLVM_ABI_FOR_TEST KnownBitsWithCtxI &value_as(const Value *V) { // NOLINT
    auto It = find_as(V);
    assert(It != end() && "Expected to find ValueHandle");
    return It->second;
  }
  LLVM_ABI_FOR_TEST KnownBitsWithCtxI value_as(const Value *V) const { // NOLINT
    auto It = find_as(V);
    assert(It != end() && "Expected to find ValueHandle");
    return It->second;
  }

  /// Invalidates KnownBits in the entire subgraph found from the
  /// forwardDataflow walk starting from \p V, turning them into Unknown values.
  /// Triggered on IR manipulation events.
  LLVM_ABI_FOR_TEST void invalidate(const KnownBitsVH &V) {
    for (const Value *N : forwardDataflow(V))
      value_as(N).resetAll();
  }

  /// Range-based variant of forwardDataflow used in print.
  LLVM_ABI_FOR_TEST SmallVector<const Value *>
  forwardDataflow(ArrayRef<KnownBitsVH> Roots) const;

  /// Checks if \p V is present in the map.
  LLVM_ABI_FOR_TEST bool contains(const Value *V) const {
    return find_as(V) != end();
  }

  /// Roots are the function \p F's arguments, along with Instructions that
  /// expose a new root like phis and fptosi. This is used in print, to print
  /// entries in the map in deterministic order.
  LLVM_ABI_FOR_TEST SmallVector<KnownBitsVH>
  computeRoots(const Function &F) const;

public:
  LLVM_ABI KnownBitsDataflow() {}
  LLVM_ABI KnownBitsDataflow(const KnownBitsDataflow &) = delete;
  LLVM_ABI KnownBitsDataflow &operator=(const KnownBitsDataflow &) = delete;

  /// A small helper extracted from ValueTracking.
  LLVM_ABI static unsigned getBitWidth(Type *Ty, const DataLayout &DL);

  using BaseT::empty;
  using BaseT::size;

  /// Checks if \p V if it is present in the map, and if it has a valid
  /// (non-Unknown) KnownBits, returning it if so. Pass \p CtxI to filter on
  /// compatibility of context-instructions.
  std::optional<KnownBits>
      LLVM_ABI lookup(const Value *V, const Instruction *CtxI = nullptr) const {
    // Constants should never be inserted into the map. This is the fast
    // lookup-path.
    if (isa<Constant>(V))
      return std::nullopt;
    auto It = find_as(V);
    if (It == end())
      return std::nullopt;
    const KnownBitsWithCtxI &Known = It->second;
    if (Known.isUnknown() || !Known.canUseWith(CtxI))
      return std::nullopt;
    return Known;
  }

  /// Registers that \p V has KnownBits information \p Known, with
  /// context-instruction \p CtxI, overwriting any existing value. Is a no-op on
  /// constant \p V and unknown \p Known.
  void LLVM_ABI emplace_as(const Value *V, const KnownBits &Known, // NOLINT
                           const Instruction *CtxI = nullptr) {
    if (isa<Constant>(V) || Known.isUnknown())
      return;
    emplace_or_assign({V, this}, KnownBitsWithCtxI(Known, CtxI));
  }

  /// This routine prints in the entries in the map in deterministic order.
  LLVM_ABI void print(const Function &F, raw_ostream &OS) const;
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump(const Function &F) const;
#endif

  bool LLVM_ABI invalidate(Function &, const PreservedAnalyses &PA,
                           FunctionAnalysisManager::Invalidator &);
};

class LLVM_ABI KnownBitsDataflowAnalysis
    : public AnalysisInfoMixin<KnownBitsDataflowAnalysis> {
public:
  static AnalysisKey Key;
  using Result = KnownBitsDataflow;
  KnownBitsDataflow run(Function &F, FunctionAnalysisManager &);
};

/// Legacy PM wrapper pass.
class LLVM_ABI KnownBitsDataflowAnalysisWrapperPass : public FunctionPass {
  std::unique_ptr<KnownBitsDataflow> Result;

public:
  static char ID;

  KnownBitsDataflowAnalysisWrapperPass();
  KnownBitsDataflow &getResult() { return *Result; }
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnFunction(Function &F) override;
};
} // end namespace llvm

#endif // LLVM_ANALYSIS_KNOWNBITSDATAFLOW_H
