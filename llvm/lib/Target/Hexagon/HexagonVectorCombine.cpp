//===-- HexagonVectorCombine.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// HexagonVectorCombine is a utility class implementing a variety of functions
// that assist in vector-based optimizations.
//
// AlignVectors: replace unaligned vector loads and stores with aligned ones.
//===----------------------------------------------------------------------===//

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsHexagon.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

#include "HexagonSubtarget.h"
#include "HexagonTargetMachine.h"

#include <algorithm>
#include <deque>
#include <map>
#include <set>
#include <utility>
#include <vector>

#define DEBUG_TYPE "hexagon-vc"

using namespace llvm;

namespace {
class HexagonVectorCombine {
public:
  HexagonVectorCombine(Function &F_, AliasAnalysis &AA_, AssumptionCache &AC_,
                       DominatorTree &DT_, TargetLibraryInfo &TLI_,
                       const TargetMachine &TM_)
      : F(F_), DL(F.getParent()->getDataLayout()), AA(AA_), AC(AC_), DT(DT_),
        TLI(TLI_),
        HST(static_cast<const HexagonSubtarget &>(*TM_.getSubtargetImpl(F))) {}

  bool run();

  // Common integer type.
  IntegerType *getIntTy() const;
  // Byte type: either scalar (when Length = 0), or vector with given
  // element count.
  Type *getByteTy(int ElemCount = 0) const;
  // Boolean type: either scalar (when Length = 0), or vector with given
  // element count.
  Type *getBoolTy(int ElemCount = 0) const;
  // Create a ConstantInt of type returned by getIntTy with the value Val.
  ConstantInt *getConstInt(int Val) const;
  // Get the integer value of V, if it exists.
  Optional<APInt> getIntValue(const Value *Val) const;
  // Is V a constant 0, or a vector of 0s?
  bool isZero(const Value *Val) const;
  // Is V an undef value?
  bool isUndef(const Value *Val) const;

  int getSizeOf(const Value *Val) const;
  int getSizeOf(const Type *Ty) const;
  int getTypeAlignment(Type *Ty) const;

  VectorType *getByteVectorTy(int ScLen) const;
  Constant *getNullValue(Type *Ty) const;
  Constant *getFullValue(Type *Ty) const;

  Value *insertb(IRBuilder<> &Builder, Value *Dest, Value *Src, int Start,
                 int Length, int Where) const;
  Value *vlalignb(IRBuilder<> &Builder, Value *Lo, Value *Hi, Value *Amt) const;
  Value *vralignb(IRBuilder<> &Builder, Value *Lo, Value *Hi, Value *Amt) const;
  Value *concat(IRBuilder<> &Builder, ArrayRef<Value *> Vecs) const;
  Value *vresize(IRBuilder<> &Builder, Value *Val, int NewSize,
                 Value *Pad) const;
  Value *rescale(IRBuilder<> &Builder, Value *Mask, Type *FromTy,
                 Type *ToTy) const;
  Value *vlsb(IRBuilder<> &Builder, Value *Val) const;
  Value *vbytes(IRBuilder<> &Builder, Value *Val) const;

  Value *createHvxIntrinsic(IRBuilder<> &Builder, Intrinsic::ID IntID,
                            Type *RetTy, ArrayRef<Value *> Args) const;

  Optional<int> calculatePointerDifference(Value *Ptr0, Value *Ptr1) const;

  template <typename T = std::vector<Instruction *>>
  bool isSafeToMoveBeforeInBB(const Instruction &In,
                              BasicBlock::const_iterator To,
                              const T &Ignore = {}) const;

  Function &F;
  const DataLayout &DL;
  AliasAnalysis &AA;
  AssumptionCache &AC;
  DominatorTree &DT;
  TargetLibraryInfo &TLI;
  const HexagonSubtarget &HST;

private:
#ifndef NDEBUG
  // These two functions are only used for assertions at the moment.
  bool isByteVecTy(Type *Ty) const;
  bool isSectorTy(Type *Ty) const;
#endif
  Value *getElementRange(IRBuilder<> &Builder, Value *Lo, Value *Hi, int Start,
                         int Length) const;
};

class AlignVectors {
public:
  AlignVectors(HexagonVectorCombine &HVC_) : HVC(HVC_) {}

  bool run();

private:
  using InstList = std::vector<Instruction *>;

  struct Segment {
    void *Data;
    int Start;
    int Size;
  };

  struct AddrInfo {
    AddrInfo(const AddrInfo &) = default;
    AddrInfo(const HexagonVectorCombine &HVC, Instruction *I, Value *A, Type *T,
             Align H)
        : Inst(I), Addr(A), ValTy(T), HaveAlign(H),
          NeedAlign(HVC.getTypeAlignment(ValTy)) {}

    // XXX: add Size member?
    Instruction *Inst;
    Value *Addr;
    Type *ValTy;
    Align HaveAlign;
    Align NeedAlign;
    int Offset = 0; // Offset (in bytes) from the first member of the
                    // containing AddrList.
  };
  using AddrList = std::vector<AddrInfo>;

  struct InstrLess {
    bool operator()(const Instruction *A, const Instruction *B) const {
      return A->comesBefore(B);
    }
  };
  using DepList = std::set<Instruction *, InstrLess>;

  struct MoveGroup {
    MoveGroup(const AddrInfo &AI, Instruction *B, bool Hvx, bool Load)
        : Base(B), Main{AI.Inst}, IsHvx(Hvx), IsLoad(Load) {}
    Instruction *Base; // Base instruction of the parent address group.
    InstList Main;     // Main group of instructions.
    InstList Deps;     // List of dependencies.
    bool IsHvx;        // Is this group of HVX instructions?
    bool IsLoad;       // Is this a load group?
  };
  using MoveList = std::vector<MoveGroup>;

  struct ByteSpan {
    struct Segment {
      Segment(Value *Val, int Begin, int Len)
          : Val(Val), Start(Begin), Size(Len) {}
      Segment(const Segment &Seg) = default;
      Value *Val;
      int Start;
      int Size;
    };

    struct Block {
      Block(Value *Val, int Len, int Pos) : Seg(Val, 0, Len), Pos(Pos) {}
      Block(Value *Val, int Off, int Len, int Pos)
          : Seg(Val, Off, Len), Pos(Pos) {}
      Block(const Block &Blk) = default;
      Segment Seg;
      int Pos;
    };

    int extent() const;
    ByteSpan section(int Start, int Length) const;
    ByteSpan &normalize();

    int size() const { return Blocks.size(); }
    Block &operator[](int i) { return Blocks[i]; }

    std::vector<Block> Blocks;

    using iterator = decltype(Blocks)::iterator;
    iterator begin() { return Blocks.begin(); }
    iterator end() { return Blocks.end(); }
    using const_iterator = decltype(Blocks)::const_iterator;
    const_iterator begin() const { return Blocks.begin(); }
    const_iterator end() const { return Blocks.end(); }
  };

  Align getAlignFromValue(const Value *V) const;
  Optional<MemoryLocation> getLocation(const Instruction &In) const;
  Optional<AddrInfo> getAddrInfo(Instruction &In) const;
  bool isHvx(const AddrInfo &AI) const;

  Value *getPayload(Value *Val) const;
  Value *getMask(Value *Val) const;
  Value *getPassThrough(Value *Val) const;

  Value *createAdjustedPointer(IRBuilder<> &Builder, Value *Ptr, Type *ValTy,
                               int Adjust) const;
  Value *createAlignedPointer(IRBuilder<> &Builder, Value *Ptr, Type *ValTy,
                              int Alignment) const;
  Value *createAlignedLoad(IRBuilder<> &Builder, Type *ValTy, Value *Ptr,
                           int Alignment, Value *Mask, Value *PassThru) const;
  Value *createAlignedStore(IRBuilder<> &Builder, Value *Val, Value *Ptr,
                            int Alignment, Value *Mask) const;

  bool createAddressGroups();
  MoveList createLoadGroups(const AddrList &Group) const;
  MoveList createStoreGroups(const AddrList &Group) const;
  bool move(const MoveGroup &Move) const;
  bool realignGroup(const MoveGroup &Move) const;

  friend raw_ostream &operator<<(raw_ostream &OS, const AddrInfo &AI);
  friend raw_ostream &operator<<(raw_ostream &OS, const MoveGroup &MG);
  friend raw_ostream &operator<<(raw_ostream &OS, const ByteSpan &BS);

  std::map<Instruction *, AddrList> AddrGroups;
  HexagonVectorCombine &HVC;
};

LLVM_ATTRIBUTE_UNUSED
raw_ostream &operator<<(raw_ostream &OS, const AlignVectors::AddrInfo &AI) {
  OS << "Inst: " << AI.Inst << "  " << *AI.Inst << '\n';
  OS << "Addr: " << *AI.Addr << '\n';
  OS << "Type: " << *AI.ValTy << '\n';
  OS << "HaveAlign: " << AI.HaveAlign.value() << '\n';
  OS << "NeedAlign: " << AI.NeedAlign.value() << '\n';
  OS << "Offset: " << AI.Offset;
  return OS;
}

LLVM_ATTRIBUTE_UNUSED
raw_ostream &operator<<(raw_ostream &OS, const AlignVectors::MoveGroup &MG) {
  OS << "Main\n";
  for (Instruction *I : MG.Main)
    OS << "  " << *I << '\n';
  OS << "Deps\n";
  for (Instruction *I : MG.Deps)
    OS << "  " << *I << '\n';
  return OS;
}

LLVM_ATTRIBUTE_UNUSED
raw_ostream &operator<<(raw_ostream &OS, const AlignVectors::ByteSpan &BS) {
  OS << "ByteSpan[size=" << BS.size() << ", extent=" << BS.extent() << '\n';
  for (const AlignVectors::ByteSpan::Block &B : BS) {
    OS << "  @" << B.Pos << " [" << B.Seg.Start << ',' << B.Seg.Size << "] "
       << *B.Seg.Val << '\n';
  }
  OS << ']';
  return OS;
}

} // namespace

namespace {

template <typename T> T *getIfUnordered(T *MaybeT) {
  return MaybeT && MaybeT->isUnordered() ? MaybeT : nullptr;
}
template <typename T> T *isCandidate(Instruction *In) {
  return dyn_cast<T>(In);
}
template <> LoadInst *isCandidate<LoadInst>(Instruction *In) {
  return getIfUnordered(dyn_cast<LoadInst>(In));
}
template <> StoreInst *isCandidate<StoreInst>(Instruction *In) {
  return getIfUnordered(dyn_cast<StoreInst>(In));
}

#if !defined(_MSC_VER) || _MSC_VER >= 1924
// VS2017 has trouble compiling this:
// error C2976: 'std::map': too few template arguments
template <typename Pred, typename... Ts>
void erase_if(std::map<Ts...> &map, Pred p)
#else
template <typename Pred, typename T, typename U>
void erase_if(std::map<T, U> &map, Pred p)
#endif
{
  for (auto i = map.begin(), e = map.end(); i != e;) {
    if (p(*i))
      i = map.erase(i);
    else
      i = std::next(i);
  }
}

// Forward other erase_ifs to the LLVM implementations.
template <typename Pred, typename T> void erase_if(T &&container, Pred p) {
  llvm::erase_if(std::forward<T>(container), p);
}

} // namespace

// --- Begin AlignVectors

auto AlignVectors::ByteSpan::extent() const -> int {
  if (size() == 0)
    return 0;
  int Min = Blocks[0].Pos;
  int Max = Blocks[0].Pos + Blocks[0].Seg.Size;
  for (int i = 1, e = size(); i != e; ++i) {
    Min = std::min(Min, Blocks[i].Pos);
    Max = std::max(Max, Blocks[i].Pos + Blocks[i].Seg.Size);
  }
  return Max - Min;
}

auto AlignVectors::ByteSpan::section(int Start, int Length) const -> ByteSpan {
  ByteSpan Section;
  for (const ByteSpan::Block &B : Blocks) {
    int L = std::max(B.Pos, Start);                       // Left end.
    int R = std::min(B.Pos + B.Seg.Size, Start + Length); // Right end+1.
    if (L < R) {
      // How much to chop off the beginning of the segment:
      int Off = L > B.Pos ? L - B.Pos : 0;
      Section.Blocks.emplace_back(B.Seg.Val, B.Seg.Start + Off, R - L, L);
    }
  }
  return Section;
}

auto AlignVectors::ByteSpan::normalize() -> ByteSpan & {
  if (size() == 0)
    return *this;
  int Min = Blocks[0].Pos;
  for (int i = 1, e = size(); i != e; ++i)
    Min = std::min(Min, Blocks[i].Pos);
  if (Min != 0) {
    for (Block &B : Blocks)
      B.Pos -= Min;
  }
  return *this;
}

auto AlignVectors::getAlignFromValue(const Value *V) const -> Align {
  const auto *C = dyn_cast<ConstantInt>(V);
  assert(C && "Alignment must be a compile-time constant integer");
  return C->getAlignValue();
}

auto AlignVectors::getAddrInfo(Instruction &In) const -> Optional<AddrInfo> {
  if (auto *L = isCandidate<LoadInst>(&In))
    return AddrInfo(HVC, L, L->getPointerOperand(), L->getType(),
                    L->getAlign());
  if (auto *S = isCandidate<StoreInst>(&In))
    return AddrInfo(HVC, S, S->getPointerOperand(),
                    S->getValueOperand()->getType(), S->getAlign());
  if (auto *II = isCandidate<IntrinsicInst>(&In)) {
    Intrinsic::ID ID = II->getIntrinsicID();
    switch (ID) {
    case Intrinsic::masked_load:
      return AddrInfo(HVC, II, II->getArgOperand(0), II->getType(),
                      getAlignFromValue(II->getArgOperand(1)));
    case Intrinsic::masked_store:
      return AddrInfo(HVC, II, II->getArgOperand(1),
                      II->getArgOperand(0)->getType(),
                      getAlignFromValue(II->getArgOperand(2)));
    }
  }
  return Optional<AddrInfo>();
}

auto AlignVectors::isHvx(const AddrInfo &AI) const -> bool {
  return HVC.HST.isTypeForHVX(AI.ValTy);
}

auto AlignVectors::getPayload(Value *Val) const -> Value * {
  if (auto *In = dyn_cast<Instruction>(Val)) {
    Intrinsic::ID ID = 0;
    if (auto *II = dyn_cast<IntrinsicInst>(In))
      ID = II->getIntrinsicID();
    if (isa<StoreInst>(In) || ID == Intrinsic::masked_store)
      return In->getOperand(0);
  }
  return Val;
}

auto AlignVectors::getMask(Value *Val) const -> Value * {
  if (auto *II = dyn_cast<IntrinsicInst>(Val)) {
    switch (II->getIntrinsicID()) {
    case Intrinsic::masked_load:
      return II->getArgOperand(2);
    case Intrinsic::masked_store:
      return II->getArgOperand(3);
    }
  }

  Type *ValTy = getPayload(Val)->getType();
  if (auto *VecTy = dyn_cast<VectorType>(ValTy)) {
    int ElemCount = VecTy->getElementCount().getFixedValue();
    return HVC.getFullValue(HVC.getBoolTy(ElemCount));
  }
  return HVC.getFullValue(HVC.getBoolTy());
}

auto AlignVectors::getPassThrough(Value *Val) const -> Value * {
  if (auto *II = dyn_cast<IntrinsicInst>(Val)) {
    if (II->getIntrinsicID() == Intrinsic::masked_load)
      return II->getArgOperand(3);
  }
  return UndefValue::get(getPayload(Val)->getType());
}

auto AlignVectors::createAdjustedPointer(IRBuilder<> &Builder, Value *Ptr,
                                         Type *ValTy, int Adjust) const
    -> Value * {
  // The adjustment is in bytes, but if it's a multiple of the type size,
  // we don't need to do pointer casts.
  Type *ElemTy = cast<PointerType>(Ptr->getType())->getElementType();
  int ElemSize = HVC.getSizeOf(ElemTy);
  if (Adjust % ElemSize == 0) {
    Value *Tmp0 = Builder.CreateGEP(Ptr, HVC.getConstInt(Adjust / ElemSize));
    return Builder.CreatePointerCast(Tmp0, ValTy->getPointerTo());
  }

  PointerType *CharPtrTy = Type::getInt8PtrTy(HVC.F.getContext());
  Value *Tmp0 = Builder.CreatePointerCast(Ptr, CharPtrTy);
  Value *Tmp1 = Builder.CreateGEP(Tmp0, HVC.getConstInt(Adjust));
  return Builder.CreatePointerCast(Tmp1, ValTy->getPointerTo());
}

auto AlignVectors::createAlignedPointer(IRBuilder<> &Builder, Value *Ptr,
                                        Type *ValTy, int Alignment) const
    -> Value * {
  Value *AsInt = Builder.CreatePtrToInt(Ptr, HVC.getIntTy());
  Value *Mask = HVC.getConstInt(-Alignment);
  Value *And = Builder.CreateAnd(AsInt, Mask);
  return Builder.CreateIntToPtr(And, ValTy->getPointerTo());
}

auto AlignVectors::createAlignedLoad(IRBuilder<> &Builder, Type *ValTy,
                                     Value *Ptr, int Alignment, Value *Mask,
                                     Value *PassThru) const -> Value * {
  assert(!HVC.isUndef(Mask)); // Should this be allowed?
  if (HVC.isZero(Mask))
    return PassThru;
  if (Mask == ConstantInt::getTrue(Mask->getType()))
    return Builder.CreateAlignedLoad(ValTy, Ptr, Align(Alignment));
  return Builder.CreateMaskedLoad(Ptr, Align(Alignment), Mask, PassThru);
}

auto AlignVectors::createAlignedStore(IRBuilder<> &Builder, Value *Val,
                                      Value *Ptr, int Alignment,
                                      Value *Mask) const -> Value * {
  if (HVC.isZero(Mask) || HVC.isUndef(Val) || HVC.isUndef(Mask))
    return UndefValue::get(Val->getType());
  if (Mask == ConstantInt::getTrue(Mask->getType()))
    return Builder.CreateAlignedStore(Val, Ptr, Align(Alignment));
  return Builder.CreateMaskedStore(Val, Ptr, Align(Alignment), Mask);
}

auto AlignVectors::createAddressGroups() -> bool {
  // An address group created here may contain instructions spanning
  // multiple basic blocks.
  AddrList WorkStack;

  auto findBaseAndOffset = [&](AddrInfo &AI) -> std::pair<Instruction *, int> {
    for (AddrInfo &W : WorkStack) {
      if (auto D = HVC.calculatePointerDifference(AI.Addr, W.Addr))
        return std::make_pair(W.Inst, *D);
    }
    return std::make_pair(nullptr, 0);
  };

  auto traverseBlock = [&](DomTreeNode *DomN, auto Visit) -> void {
    BasicBlock &Block = *DomN->getBlock();
    for (Instruction &I : Block) {
      auto AI = this->getAddrInfo(I); // Use this-> for gcc6.
      if (!AI)
        continue;
      auto F = findBaseAndOffset(*AI);
      Instruction *GroupInst;
      if (Instruction *BI = F.first) {
        AI->Offset = F.second;
        GroupInst = BI;
      } else {
        WorkStack.push_back(*AI);
        GroupInst = AI->Inst;
      }
      AddrGroups[GroupInst].push_back(*AI);
    }

    for (DomTreeNode *C : DomN->children())
      Visit(C, Visit);

    while (!WorkStack.empty() && WorkStack.back().Inst->getParent() == &Block)
      WorkStack.pop_back();
  };

  traverseBlock(HVC.DT.getRootNode(), traverseBlock);
  assert(WorkStack.empty());

  // AddrGroups are formed.

  // Remove groups of size 1.
  erase_if(AddrGroups, [](auto &G) { return G.second.size() == 1; });
  // Remove groups that don't use HVX types.
  erase_if(AddrGroups, [&](auto &G) {
    return !llvm::any_of(
        G.second, [&](auto &I) { return HVC.HST.isTypeForHVX(I.ValTy); });
  });
  // Remove groups where everything is properly aligned.
  erase_if(AddrGroups, [&](auto &G) {
    return llvm::all_of(G.second,
                        [&](auto &I) { return I.HaveAlign >= I.NeedAlign; });
  });

  return !AddrGroups.empty();
}

auto AlignVectors::createLoadGroups(const AddrList &Group) const -> MoveList {
  // Form load groups.
  // To avoid complications with moving code across basic blocks, only form
  // groups that are contained within a single basic block.

  auto getUpwardDeps = [](Instruction *In, Instruction *Base) {
    BasicBlock *Parent = Base->getParent();
    assert(In->getParent() == Parent &&
           "Base and In should be in the same block");
    assert(Base->comesBefore(In) && "Base should come before In");

    DepList Deps;
    std::deque<Instruction *> WorkQ = {In};
    while (!WorkQ.empty()) {
      Instruction *D = WorkQ.front();
      WorkQ.pop_front();
      Deps.insert(D);
      for (Value *Op : D->operands()) {
        if (auto *I = dyn_cast<Instruction>(Op)) {
          if (I->getParent() == Parent && Base->comesBefore(I))
            WorkQ.push_back(I);
        }
      }
    }
    return Deps;
  };

  auto tryAddTo = [&](const AddrInfo &Info, MoveGroup &Move) {
    assert(!Move.Main.empty() && "Move group should have non-empty Main");
    // Don't mix HVX and non-HVX instructions.
    if (Move.IsHvx != isHvx(Info))
      return false;
    // Leading instruction in the load group.
    Instruction *Base = Move.Main.front();
    if (Base->getParent() != Info.Inst->getParent())
      return false;

    auto isSafeToMoveToBase = [&](const Instruction *I) {
      return HVC.isSafeToMoveBeforeInBB(*I, Base->getIterator());
    };
    DepList Deps = getUpwardDeps(Info.Inst, Base);
    if (!llvm::all_of(Deps, isSafeToMoveToBase))
      return false;

    // The dependencies will be moved together with the load, so make sure
    // that none of them could be moved independently in another group.
    Deps.erase(Info.Inst);
    auto inAddrMap = [&](Instruction *I) { return AddrGroups.count(I) > 0; };
    if (llvm::any_of(Deps, inAddrMap))
      return false;
    Move.Main.push_back(Info.Inst);
    llvm::append_range(Move.Deps, Deps);
    return true;
  };

  MoveList LoadGroups;

  for (const AddrInfo &Info : Group) {
    if (!Info.Inst->mayReadFromMemory())
      continue;
    if (LoadGroups.empty() || !tryAddTo(Info, LoadGroups.back()))
      LoadGroups.emplace_back(Info, Group.front().Inst, isHvx(Info), true);
  }

  // Erase singleton groups.
  erase_if(LoadGroups, [](const MoveGroup &G) { return G.Main.size() <= 1; });
  return LoadGroups;
}

auto AlignVectors::createStoreGroups(const AddrList &Group) const -> MoveList {
  // Form store groups.
  // To avoid complications with moving code across basic blocks, only form
  // groups that are contained within a single basic block.

  auto tryAddTo = [&](const AddrInfo &Info, MoveGroup &Move) {
    assert(!Move.Main.empty() && "Move group should have non-empty Main");
    // For stores with return values we'd have to collect downward depenencies.
    // There are no such stores that we handle at the moment, so omit that.
    assert(Info.Inst->getType()->isVoidTy() &&
           "Not handling stores with return values");
    // Don't mix HVX and non-HVX instructions.
    if (Move.IsHvx != isHvx(Info))
      return false;
    // For stores we need to be careful whether it's safe to move them.
    // Stores that are otherwise safe to move together may not appear safe
    // to move over one another (i.e. isSafeToMoveBefore may return false).
    Instruction *Base = Move.Main.front();
    if (Base->getParent() != Info.Inst->getParent())
      return false;
    if (!HVC.isSafeToMoveBeforeInBB(*Info.Inst, Base->getIterator(), Move.Main))
      return false;
    Move.Main.push_back(Info.Inst);
    return true;
  };

  MoveList StoreGroups;

  for (auto I = Group.rbegin(), E = Group.rend(); I != E; ++I) {
    const AddrInfo &Info = *I;
    if (!Info.Inst->mayWriteToMemory())
      continue;
    if (StoreGroups.empty() || !tryAddTo(Info, StoreGroups.back()))
      StoreGroups.emplace_back(Info, Group.front().Inst, isHvx(Info), false);
  }

  // Erase singleton groups.
  erase_if(StoreGroups, [](const MoveGroup &G) { return G.Main.size() <= 1; });
  return StoreGroups;
}

auto AlignVectors::move(const MoveGroup &Move) const -> bool {
  assert(!Move.Main.empty() && "Move group should have non-empty Main");
  Instruction *Where = Move.Main.front();

  if (Move.IsLoad) {
    // Move all deps to before Where, keeping order.
    for (Instruction *D : Move.Deps)
      D->moveBefore(Where);
    // Move all main instructions to after Where, keeping order.
    ArrayRef<Instruction *> Main(Move.Main);
    for (Instruction *M : Main.drop_front(1)) {
      M->moveAfter(Where);
      Where = M;
    }
  } else {
    // NOTE: Deps are empty for "store" groups. If they need to be
    // non-empty, decide on the order.
    assert(Move.Deps.empty());
    // Move all main instructions to before Where, inverting order.
    ArrayRef<Instruction *> Main(Move.Main);
    for (Instruction *M : Main.drop_front(1)) {
      M->moveBefore(Where);
      Where = M;
    }
  }

  return Move.Main.size() + Move.Deps.size() > 1;
}

auto AlignVectors::realignGroup(const MoveGroup &Move) const -> bool {
  // TODO: Needs support for masked loads/stores of "scalar" vectors.
  if (!Move.IsHvx)
    return false;

  // Return the element with the maximum alignment from Range,
  // where GetValue obtains the value to compare from an element.
  auto getMaxOf = [](auto Range, auto GetValue) {
    return *std::max_element(
        Range.begin(), Range.end(),
        [&GetValue](auto &A, auto &B) { return GetValue(A) < GetValue(B); });
  };

  const AddrList &BaseInfos = AddrGroups.at(Move.Base);

  // Conceptually, there is a vector of N bytes covering the addresses
  // starting from the minimum offset (i.e. Base.Addr+Start). This vector
  // represents a contiguous memory region that spans all accessed memory
  // locations.
  // The correspondence between loaded or stored values will be expressed
  // in terms of this vector. For example, the 0th element of the vector
  // from the Base address info will start at byte Start from the beginning
  // of this conceptual vector.
  //
  // This vector will be loaded/stored starting at the nearest down-aligned
  // address and the amount od the down-alignment will be AlignVal:
  //   valign(load_vector(align_down(Base+Start)), AlignVal)

  std::set<Instruction *> TestSet(Move.Main.begin(), Move.Main.end());
  AddrList MoveInfos;
  llvm::copy_if(
      BaseInfos, std::back_inserter(MoveInfos),
      [&TestSet](const AddrInfo &AI) { return TestSet.count(AI.Inst); });

  // Maximum alignment present in the whole address group.
  const AddrInfo &WithMaxAlign =
      getMaxOf(BaseInfos, [](const AddrInfo &AI) { return AI.HaveAlign; });
  Align MaxGiven = WithMaxAlign.HaveAlign;

  // Minimum alignment present in the move address group.
  const AddrInfo &WithMinOffset =
      getMaxOf(MoveInfos, [](const AddrInfo &AI) { return -AI.Offset; });

  const AddrInfo &WithMaxNeeded =
      getMaxOf(MoveInfos, [](const AddrInfo &AI) { return AI.NeedAlign; });
  Align MinNeeded = WithMaxNeeded.NeedAlign;

  // Set the builder at the top instruction in the move group.
  Instruction *TopIn = Move.IsLoad ? Move.Main.front() : Move.Main.back();
  IRBuilder<> Builder(TopIn);
  Value *AlignAddr = nullptr; // Actual aligned address.
  Value *AlignVal = nullptr;  // Right-shift amount (for valign).

  if (MinNeeded <= MaxGiven) {
    int Start = WithMinOffset.Offset;
    int OffAtMax = WithMaxAlign.Offset;
    // Shift the offset of the maximally aligned instruction (OffAtMax)
    // back by just enough multiples of the required alignment to cover the
    // distance from Start to OffAtMax.
    // Calculate the address adjustment amount based on the address with the
    // maximum alignment. This is to allow a simple gep instruction instead
    // of potential bitcasts to i8*.
    int Adjust = -alignTo(OffAtMax - Start, MinNeeded.value());
    AlignAddr = createAdjustedPointer(Builder, WithMaxAlign.Addr,
                                      WithMaxAlign.ValTy, Adjust);
    int Diff = Start - (OffAtMax + Adjust);
    AlignVal = HVC.getConstInt(Diff);
    // Sanity.
    assert(Diff >= 0);
    assert(static_cast<decltype(MinNeeded.value())>(Diff) < MinNeeded.value());
  } else {
    // WithMinOffset is the lowest address in the group,
    //   WithMinOffset.Addr = Base+Start.
    // Align instructions for both HVX (V6_valign) and scalar (S2_valignrb)
    // mask off unnecessary bits, so it's ok to just the original pointer as
    // the alignment amount.
    // Do an explicit down-alignment of the address to avoid creating an
    // aligned instruction with an address that is not really aligned.
    AlignAddr = createAlignedPointer(Builder, WithMinOffset.Addr,
                                     WithMinOffset.ValTy, MinNeeded.value());
    AlignVal = Builder.CreatePtrToInt(WithMinOffset.Addr, HVC.getIntTy());
  }

  ByteSpan VSpan;
  for (const AddrInfo &AI : MoveInfos) {
    VSpan.Blocks.emplace_back(AI.Inst, HVC.getSizeOf(AI.ValTy),
                              AI.Offset - WithMinOffset.Offset);
  }

  // The aligned loads/stores will use blocks that are either scalars,
  // or HVX vectors. Let "sector" be the unified term for such a block.
  // blend(scalar, vector) -> sector...
  int ScLen = Move.IsHvx ? HVC.HST.getVectorLength()
                         : std::max<int>(MinNeeded.value(), 4);
  assert(!Move.IsHvx || ScLen == 64 || ScLen == 128);
  assert(Move.IsHvx || ScLen == 4 || ScLen == 8);

  Type *SecTy = HVC.getByteTy(ScLen);
  int NumSectors = (VSpan.extent() + ScLen - 1) / ScLen;

  if (Move.IsLoad) {
    ByteSpan ASpan;
    auto *True = HVC.getFullValue(HVC.getBoolTy(ScLen));
    auto *Undef = UndefValue::get(SecTy);

    for (int i = 0; i != NumSectors + 1; ++i) {
      Value *Ptr = createAdjustedPointer(Builder, AlignAddr, SecTy, i * ScLen);
      // FIXME: generate a predicated load?
      Value *Load = createAlignedLoad(Builder, SecTy, Ptr, ScLen, True, Undef);
      ASpan.Blocks.emplace_back(Load, ScLen, i * ScLen);
    }

    for (int j = 0; j != NumSectors; ++j) {
      ASpan[j].Seg.Val = HVC.vralignb(Builder, ASpan[j].Seg.Val,
                                      ASpan[j + 1].Seg.Val, AlignVal);
    }

    for (ByteSpan::Block &B : VSpan) {
      ByteSpan Section = ASpan.section(B.Pos, B.Seg.Size).normalize();
      Value *Accum = UndefValue::get(HVC.getByteTy(B.Seg.Size));
      for (ByteSpan::Block &S : Section) {
        Value *Pay = HVC.vbytes(Builder, getPayload(S.Seg.Val));
        Accum =
            HVC.insertb(Builder, Accum, Pay, S.Seg.Start, S.Seg.Size, S.Pos);
      }
      // Instead of casting everything to bytes for the vselect, cast to the
      // original value type. This will avoid complications with casting masks.
      // For example, in cases when the original mask applied to i32, it could
      // be converted to a mask applicable to i8 via pred_typecast intrinsic,
      // but if the mask is not exactly of HVX length, extra handling would be
      // needed to make it work.
      Type *ValTy = getPayload(B.Seg.Val)->getType();
      Value *Cast = Builder.CreateBitCast(Accum, ValTy);
      Value *Sel = Builder.CreateSelect(getMask(B.Seg.Val), Cast,
                                        getPassThrough(B.Seg.Val));
      B.Seg.Val->replaceAllUsesWith(Sel);
    }
  } else {
    // Stores.
    ByteSpan ASpanV, ASpanM;

    // Return a vector value corresponding to the input value Val:
    // either <1 x Val> for scalar Val, or Val itself for vector Val.
    auto MakeVec = [](IRBuilder<> &Builder, Value *Val) -> Value * {
      Type *Ty = Val->getType();
      if (Ty->isVectorTy())
        return Val;
      auto *VecTy = VectorType::get(Ty, 1, /*Scalable*/ false);
      return Builder.CreateBitCast(Val, VecTy);
    };

    // Create an extra "undef" sector at the beginning and at the end.
    // They will be used as the left/right filler in the vlalign step.
    for (int i = -1; i != NumSectors + 1; ++i) {
      ByteSpan Section = VSpan.section(i * ScLen, ScLen).normalize();
      Value *AccumV = UndefValue::get(SecTy);
      Value *AccumM = HVC.getNullValue(SecTy);
      for (ByteSpan::Block &S : Section) {
        Value *Pay = getPayload(S.Seg.Val);
        Value *Mask = HVC.rescale(Builder, MakeVec(Builder, getMask(S.Seg.Val)),
                                  Pay->getType(), HVC.getByteTy());
        AccumM = HVC.insertb(Builder, AccumM, HVC.vbytes(Builder, Mask),
                             S.Seg.Start, S.Seg.Size, S.Pos);
        AccumV = HVC.insertb(Builder, AccumV, HVC.vbytes(Builder, Pay),
                             S.Seg.Start, S.Seg.Size, S.Pos);
      }
      ASpanV.Blocks.emplace_back(AccumV, ScLen, i * ScLen);
      ASpanM.Blocks.emplace_back(AccumM, ScLen, i * ScLen);
    }

    // vlalign
    for (int j = 1; j != NumSectors + 2; ++j) {
      ASpanV[j - 1].Seg.Val = HVC.vlalignb(Builder, ASpanV[j - 1].Seg.Val,
                                           ASpanV[j].Seg.Val, AlignVal);
      ASpanM[j - 1].Seg.Val = HVC.vlalignb(Builder, ASpanM[j - 1].Seg.Val,
                                           ASpanM[j].Seg.Val, AlignVal);
    }

    for (int i = 0; i != NumSectors + 1; ++i) {
      Value *Ptr = createAdjustedPointer(Builder, AlignAddr, SecTy, i * ScLen);
      Value *Val = ASpanV[i].Seg.Val;
      Value *Mask = ASpanM[i].Seg.Val; // bytes
      if (!HVC.isUndef(Val) && !HVC.isZero(Mask))
        createAlignedStore(Builder, Val, Ptr, ScLen, HVC.vlsb(Builder, Mask));
    }
  }

  for (auto *Inst : Move.Main)
    Inst->eraseFromParent();

  return true;
}

auto AlignVectors::run() -> bool {
  if (!createAddressGroups())
    return false;

  bool Changed = false;
  MoveList LoadGroups, StoreGroups;

  for (auto &G : AddrGroups) {
    llvm::append_range(LoadGroups, createLoadGroups(G.second));
    llvm::append_range(StoreGroups, createStoreGroups(G.second));
  }

  for (auto &M : LoadGroups)
    Changed |= move(M);
  for (auto &M : StoreGroups)
    Changed |= move(M);

  for (auto &M : LoadGroups)
    Changed |= realignGroup(M);
  for (auto &M : StoreGroups)
    Changed |= realignGroup(M);

  return Changed;
}

// --- End AlignVectors

auto HexagonVectorCombine::run() -> bool {
  if (!HST.useHVXOps())
    return false;

  bool Changed = AlignVectors(*this).run();
  return Changed;
}

auto HexagonVectorCombine::getIntTy() const -> IntegerType * {
  return Type::getInt32Ty(F.getContext());
}

auto HexagonVectorCombine::getByteTy(int ElemCount) const -> Type * {
  assert(ElemCount >= 0);
  IntegerType *ByteTy = Type::getInt8Ty(F.getContext());
  if (ElemCount == 0)
    return ByteTy;
  return VectorType::get(ByteTy, ElemCount, /*Scalable*/ false);
}

auto HexagonVectorCombine::getBoolTy(int ElemCount) const -> Type * {
  assert(ElemCount >= 0);
  IntegerType *BoolTy = Type::getInt1Ty(F.getContext());
  if (ElemCount == 0)
    return BoolTy;
  return VectorType::get(BoolTy, ElemCount, /*Scalable*/ false);
}

auto HexagonVectorCombine::getConstInt(int Val) const -> ConstantInt * {
  return ConstantInt::getSigned(getIntTy(), Val);
}

auto HexagonVectorCombine::isZero(const Value *Val) const -> bool {
  if (auto *C = dyn_cast<Constant>(Val))
    return C->isZeroValue();
  return false;
}

auto HexagonVectorCombine::getIntValue(const Value *Val) const
    -> Optional<APInt> {
  if (auto *CI = dyn_cast<ConstantInt>(Val))
    return CI->getValue();
  return None;
}

auto HexagonVectorCombine::isUndef(const Value *Val) const -> bool {
  return isa<UndefValue>(Val);
}

auto HexagonVectorCombine::getSizeOf(const Value *Val) const -> int {
  return getSizeOf(Val->getType());
}

auto HexagonVectorCombine::getSizeOf(const Type *Ty) const -> int {
  return DL.getTypeStoreSize(const_cast<Type *>(Ty)).getFixedValue();
}

auto HexagonVectorCombine::getTypeAlignment(Type *Ty) const -> int {
  // The actual type may be shorter than the HVX vector, so determine
  // the alignment based on subtarget info.
  if (HST.isTypeForHVX(Ty))
    return HST.getVectorLength();
  return DL.getABITypeAlign(Ty).value();
}

auto HexagonVectorCombine::getNullValue(Type *Ty) const -> Constant * {
  assert(Ty->isIntOrIntVectorTy());
  auto Zero = ConstantInt::get(Ty->getScalarType(), 0);
  if (auto *VecTy = dyn_cast<VectorType>(Ty))
    return ConstantVector::getSplat(VecTy->getElementCount(), Zero);
  return Zero;
}

auto HexagonVectorCombine::getFullValue(Type *Ty) const -> Constant * {
  assert(Ty->isIntOrIntVectorTy());
  auto Minus1 = ConstantInt::get(Ty->getScalarType(), -1);
  if (auto *VecTy = dyn_cast<VectorType>(Ty))
    return ConstantVector::getSplat(VecTy->getElementCount(), Minus1);
  return Minus1;
}

// Insert bytes [Start..Start+Length) of Src into Dst at byte Where.
auto HexagonVectorCombine::insertb(IRBuilder<> &Builder, Value *Dst, Value *Src,
                                   int Start, int Length, int Where) const
    -> Value * {
  assert(isByteVecTy(Dst->getType()) && isByteVecTy(Src->getType()));
  int SrcLen = getSizeOf(Src);
  int DstLen = getSizeOf(Dst);
  assert(0 <= Start && Start + Length <= SrcLen);
  assert(0 <= Where && Where + Length <= DstLen);

  int P2Len = PowerOf2Ceil(SrcLen | DstLen);
  auto *Undef = UndefValue::get(getByteTy());
  Value *P2Src = vresize(Builder, Src, P2Len, Undef);
  Value *P2Dst = vresize(Builder, Dst, P2Len, Undef);

  SmallVector<int, 256> SMask(P2Len);
  for (int i = 0; i != P2Len; ++i) {
    // If i is in [Where, Where+Length), pick Src[Start+(i-Where)].
    // Otherwise, pick Dst[i];
    SMask[i] =
        (Where <= i && i < Where + Length) ? P2Len + Start + (i - Where) : i;
  }

  Value *P2Insert = Builder.CreateShuffleVector(P2Dst, P2Src, SMask);
  return vresize(Builder, P2Insert, DstLen, Undef);
}

auto HexagonVectorCombine::vlalignb(IRBuilder<> &Builder, Value *Lo, Value *Hi,
                                    Value *Amt) const -> Value * {
  assert(Lo->getType() == Hi->getType() && "Argument type mismatch");
  assert(isSectorTy(Hi->getType()));
  if (isZero(Amt))
    return Hi;
  int VecLen = getSizeOf(Hi);
  if (auto IntAmt = getIntValue(Amt))
    return getElementRange(Builder, Lo, Hi, VecLen - IntAmt->getSExtValue(),
                           VecLen);

  if (HST.isTypeForHVX(Hi->getType())) {
    int HwLen = HST.getVectorLength();
    assert(VecLen == HwLen && "Expecting an exact HVX type");
    Intrinsic::ID V6_vlalignb = HwLen == 64
                                    ? Intrinsic::hexagon_V6_vlalignb
                                    : Intrinsic::hexagon_V6_vlalignb_128B;
    return createHvxIntrinsic(Builder, V6_vlalignb, Hi->getType(),
                              {Hi, Lo, Amt});
  }

  if (VecLen == 4) {
    Value *Pair = concat(Builder, {Lo, Hi});
    Value *Shift = Builder.CreateLShr(Builder.CreateShl(Pair, Amt), 32);
    Value *Trunc = Builder.CreateTrunc(Shift, Type::getInt32Ty(F.getContext()));
    return Builder.CreateBitCast(Trunc, Hi->getType());
  }
  if (VecLen == 8) {
    Value *Sub = Builder.CreateSub(getConstInt(VecLen), Amt);
    return vralignb(Builder, Lo, Hi, Sub);
  }
  llvm_unreachable("Unexpected vector length");
}

auto HexagonVectorCombine::vralignb(IRBuilder<> &Builder, Value *Lo, Value *Hi,
                                    Value *Amt) const -> Value * {
  assert(Lo->getType() == Hi->getType() && "Argument type mismatch");
  assert(isSectorTy(Lo->getType()));
  if (isZero(Amt))
    return Lo;
  int VecLen = getSizeOf(Lo);
  if (auto IntAmt = getIntValue(Amt))
    return getElementRange(Builder, Lo, Hi, IntAmt->getSExtValue(), VecLen);

  if (HST.isTypeForHVX(Lo->getType())) {
    int HwLen = HST.getVectorLength();
    assert(VecLen == HwLen && "Expecting an exact HVX type");
    Intrinsic::ID V6_valignb = HwLen == 64 ? Intrinsic::hexagon_V6_valignb
                                           : Intrinsic::hexagon_V6_valignb_128B;
    return createHvxIntrinsic(Builder, V6_valignb, Lo->getType(),
                              {Hi, Lo, Amt});
  }

  if (VecLen == 4) {
    Value *Pair = concat(Builder, {Lo, Hi});
    Value *Shift = Builder.CreateLShr(Pair, Amt);
    Value *Trunc = Builder.CreateTrunc(Shift, Type::getInt32Ty(F.getContext()));
    return Builder.CreateBitCast(Trunc, Lo->getType());
  }
  if (VecLen == 8) {
    Type *Int64Ty = Type::getInt64Ty(F.getContext());
    Value *Lo64 = Builder.CreateBitCast(Lo, Int64Ty);
    Value *Hi64 = Builder.CreateBitCast(Hi, Int64Ty);
    Function *FI = Intrinsic::getDeclaration(F.getParent(),
                                             Intrinsic::hexagon_S2_valignrb);
    Value *Call = Builder.CreateCall(FI, {Hi64, Lo64, Amt});
    return Builder.CreateBitCast(Call, Lo->getType());
  }
  llvm_unreachable("Unexpected vector length");
}

// Concatenates a sequence of vectors of the same type.
auto HexagonVectorCombine::concat(IRBuilder<> &Builder,
                                  ArrayRef<Value *> Vecs) const -> Value * {
  assert(!Vecs.empty());
  SmallVector<int, 256> SMask;
  std::vector<Value *> Work[2];
  int ThisW = 0, OtherW = 1;

  Work[ThisW].assign(Vecs.begin(), Vecs.end());
  while (Work[ThisW].size() > 1) {
    auto *Ty = cast<VectorType>(Work[ThisW].front()->getType());
    int ElemCount = Ty->getElementCount().getFixedValue();
    SMask.resize(ElemCount * 2);
    std::iota(SMask.begin(), SMask.end(), 0);

    Work[OtherW].clear();
    if (Work[ThisW].size() % 2 != 0)
      Work[ThisW].push_back(UndefValue::get(Ty));
    for (int i = 0, e = Work[ThisW].size(); i < e; i += 2) {
      Value *Joined = Builder.CreateShuffleVector(Work[ThisW][i],
                                                  Work[ThisW][i + 1], SMask);
      Work[OtherW].push_back(Joined);
    }
    std::swap(ThisW, OtherW);
  }

  // Since there may have been some undefs appended to make shuffle operands
  // have the same type, perform the last shuffle to only pick the original
  // elements.
  SMask.resize(Vecs.size() * getSizeOf(Vecs.front()->getType()));
  std::iota(SMask.begin(), SMask.end(), 0);
  Value *Total = Work[OtherW].front();
  return Builder.CreateShuffleVector(Total, SMask);
}

auto HexagonVectorCombine::vresize(IRBuilder<> &Builder, Value *Val,
                                   int NewSize, Value *Pad) const -> Value * {
  assert(isa<VectorType>(Val->getType()));
  auto *ValTy = cast<VectorType>(Val->getType());
  assert(ValTy->getElementType() == Pad->getType());

  int CurSize = ValTy->getElementCount().getFixedValue();
  if (CurSize == NewSize)
    return Val;
  // Truncate?
  if (CurSize > NewSize)
    return getElementRange(Builder, Val, /*Unused*/ Val, 0, NewSize);
  // Extend.
  SmallVector<int, 128> SMask(NewSize);
  std::iota(SMask.begin(), SMask.begin() + CurSize, 0);
  std::fill(SMask.begin() + CurSize, SMask.end(), CurSize);
  Value *PadVec = Builder.CreateVectorSplat(CurSize, Pad);
  return Builder.CreateShuffleVector(Val, PadVec, SMask);
}

auto HexagonVectorCombine::rescale(IRBuilder<> &Builder, Value *Mask,
                                   Type *FromTy, Type *ToTy) const -> Value * {
  // Mask is a vector <N x i1>, where each element corresponds to an
  // element of FromTy. Remap it so that each element will correspond
  // to an element of ToTy.
  assert(isa<VectorType>(Mask->getType()));

  Type *FromSTy = FromTy->getScalarType();
  Type *ToSTy = ToTy->getScalarType();
  if (FromSTy == ToSTy)
    return Mask;

  int FromSize = getSizeOf(FromSTy);
  int ToSize = getSizeOf(ToSTy);
  assert(FromSize % ToSize == 0 || ToSize % FromSize == 0);

  auto *MaskTy = cast<VectorType>(Mask->getType());
  int FromCount = MaskTy->getElementCount().getFixedValue();
  int ToCount = (FromCount * FromSize) / ToSize;
  assert((FromCount * FromSize) % ToSize == 0);

  // Mask <N x i1> -> sext to <N x FromTy> -> bitcast to <M x ToTy> ->
  // -> trunc to <M x i1>.
  Value *Ext = Builder.CreateSExt(
      Mask, VectorType::get(FromSTy, FromCount, /*Scalable*/ false));
  Value *Cast = Builder.CreateBitCast(
      Ext, VectorType::get(ToSTy, ToCount, /*Scalable*/ false));
  return Builder.CreateTrunc(
      Cast, VectorType::get(getBoolTy(), ToCount, /*Scalable*/ false));
}

// Bitcast to bytes, and return least significant bits.
auto HexagonVectorCombine::vlsb(IRBuilder<> &Builder, Value *Val) const
    -> Value * {
  Type *ScalarTy = Val->getType()->getScalarType();
  if (ScalarTy == getBoolTy())
    return Val;

  Value *Bytes = vbytes(Builder, Val);
  if (auto *VecTy = dyn_cast<VectorType>(Bytes->getType()))
    return Builder.CreateTrunc(Bytes, getBoolTy(getSizeOf(VecTy)));
  // If Bytes is a scalar (i.e. Val was a scalar byte), return i1, not
  // <1 x i1>.
  return Builder.CreateTrunc(Bytes, getBoolTy());
}

// Bitcast to bytes for non-bool. For bool, convert i1 -> i8.
auto HexagonVectorCombine::vbytes(IRBuilder<> &Builder, Value *Val) const
    -> Value * {
  Type *ScalarTy = Val->getType()->getScalarType();
  if (ScalarTy == getByteTy())
    return Val;

  if (ScalarTy != getBoolTy())
    return Builder.CreateBitCast(Val, getByteTy(getSizeOf(Val)));
  // For bool, return a sext from i1 to i8.
  if (auto *VecTy = dyn_cast<VectorType>(Val->getType()))
    return Builder.CreateSExt(Val, VectorType::get(getByteTy(), VecTy));
  return Builder.CreateSExt(Val, getByteTy());
}

auto HexagonVectorCombine::createHvxIntrinsic(IRBuilder<> &Builder,
                                              Intrinsic::ID IntID, Type *RetTy,
                                              ArrayRef<Value *> Args) const
    -> Value * {
  int HwLen = HST.getVectorLength();
  Type *BoolTy = Type::getInt1Ty(F.getContext());
  Type *Int32Ty = Type::getInt32Ty(F.getContext());
  // HVX vector -> v16i32/v32i32
  // HVX vector predicate -> v512i1/v1024i1
  auto getTypeForIntrin = [&](Type *Ty) -> Type * {
    if (HST.isTypeForHVX(Ty, /*IncludeBool*/ true)) {
      Type *ElemTy = cast<VectorType>(Ty)->getElementType();
      if (ElemTy == Int32Ty)
        return Ty;
      if (ElemTy == BoolTy)
        return VectorType::get(BoolTy, 8 * HwLen, /*Scalable*/ false);
      return VectorType::get(Int32Ty, HwLen / 4, /*Scalable*/ false);
    }
    // Non-HVX type. It should be a scalar.
    assert(Ty == Int32Ty || Ty->isIntegerTy(64));
    return Ty;
  };

  auto getCast = [&](IRBuilder<> &Builder, Value *Val,
                     Type *DestTy) -> Value * {
    Type *SrcTy = Val->getType();
    if (SrcTy == DestTy)
      return Val;
    if (HST.isTypeForHVX(SrcTy, /*IncludeBool*/ true)) {
      if (cast<VectorType>(SrcTy)->getElementType() == BoolTy) {
        // This should take care of casts the other way too, for example
        // v1024i1 -> v32i1.
        Intrinsic::ID TC = HwLen == 64
                               ? Intrinsic::hexagon_V6_pred_typecast
                               : Intrinsic::hexagon_V6_pred_typecast_128B;
        Function *FI = Intrinsic::getDeclaration(F.getParent(), TC,
                                                 {DestTy, Val->getType()});
        return Builder.CreateCall(FI, {Val});
      }
      // Non-predicate HVX vector.
      return Builder.CreateBitCast(Val, DestTy);
    }
    // Non-HVX type. It should be a scalar, and it should already have
    // a valid type.
    llvm_unreachable("Unexpected type");
  };

  SmallVector<Value *, 4> IntOps;
  for (Value *A : Args)
    IntOps.push_back(getCast(Builder, A, getTypeForIntrin(A->getType())));
  Function *FI = Intrinsic::getDeclaration(F.getParent(), IntID);
  Value *Call = Builder.CreateCall(FI, IntOps);

  Type *CallTy = Call->getType();
  if (CallTy == RetTy)
    return Call;
  // Scalar types should have RetTy matching the call return type.
  assert(HST.isTypeForHVX(CallTy, /*IncludeBool*/ true));
  if (cast<VectorType>(CallTy)->getElementType() == BoolTy)
    return getCast(Builder, Call, RetTy);
  return Builder.CreateBitCast(Call, RetTy);
}

auto HexagonVectorCombine::calculatePointerDifference(Value *Ptr0,
                                                      Value *Ptr1) const
    -> Optional<int> {
  struct Builder : IRBuilder<> {
    Builder(BasicBlock *B) : IRBuilder<>(B) {}
    ~Builder() {
      for (Instruction *I : llvm::reverse(ToErase))
        I->eraseFromParent();
    }
    SmallVector<Instruction *, 8> ToErase;
  };

#define CallBuilder(B, F)                                                      \
  [&](auto &B_) {                                                              \
    Value *V = B_.F;                                                           \
    if (auto *I = dyn_cast<Instruction>(V))                                    \
      B_.ToErase.push_back(I);                                                 \
    return V;                                                                  \
  }(B)

  auto Simplify = [&](Value *V) {
    if (auto *I = dyn_cast<Instruction>(V)) {
      SimplifyQuery Q(DL, &TLI, &DT, &AC, I);
      if (Value *S = SimplifyInstruction(I, Q))
        return S;
    }
    return V;
  };

  auto StripBitCast = [](Value *V) {
    while (auto *C = dyn_cast<BitCastInst>(V))
      V = C->getOperand(0);
    return V;
  };

  Ptr0 = StripBitCast(Ptr0);
  Ptr1 = StripBitCast(Ptr1);
  if (!isa<GetElementPtrInst>(Ptr0) || !isa<GetElementPtrInst>(Ptr1))
    return None;

  auto *Gep0 = cast<GetElementPtrInst>(Ptr0);
  auto *Gep1 = cast<GetElementPtrInst>(Ptr1);
  if (Gep0->getPointerOperand() != Gep1->getPointerOperand())
    return None;

  Builder B(Gep0->getParent());
  Value *BasePtr = Gep0->getPointerOperand();
  int Scale = DL.getTypeStoreSize(BasePtr->getType()->getPointerElementType());

  // FIXME: for now only check GEPs with a single index.
  if (Gep0->getNumOperands() != 2 || Gep1->getNumOperands() != 2)
    return None;

  Value *Idx0 = Gep0->getOperand(1);
  Value *Idx1 = Gep1->getOperand(1);

  // First, try to simplify the subtraction directly.
  if (auto *Diff = dyn_cast<ConstantInt>(
          Simplify(CallBuilder(B, CreateSub(Idx0, Idx1)))))
    return Diff->getSExtValue() * Scale;

  KnownBits Known0 = computeKnownBits(Idx0, DL, 0, &AC, Gep0, &DT);
  KnownBits Known1 = computeKnownBits(Idx1, DL, 0, &AC, Gep1, &DT);
  APInt Unknown = ~(Known0.Zero | Known0.One) | ~(Known1.Zero | Known1.One);
  if (Unknown.isAllOnesValue())
    return None;

  Value *MaskU = ConstantInt::get(Idx0->getType(), Unknown);
  Value *AndU0 = Simplify(CallBuilder(B, CreateAnd(Idx0, MaskU)));
  Value *AndU1 = Simplify(CallBuilder(B, CreateAnd(Idx1, MaskU)));
  Value *SubU = Simplify(CallBuilder(B, CreateSub(AndU0, AndU1)));
  int Diff0 = 0;
  if (auto *C = dyn_cast<ConstantInt>(SubU)) {
    Diff0 = C->getSExtValue();
  } else {
    return None;
  }

  Value *MaskK = ConstantInt::get(MaskU->getType(), ~Unknown);
  Value *AndK0 = Simplify(CallBuilder(B, CreateAnd(Idx0, MaskK)));
  Value *AndK1 = Simplify(CallBuilder(B, CreateAnd(Idx1, MaskK)));
  Value *SubK = Simplify(CallBuilder(B, CreateSub(AndK0, AndK1)));
  int Diff1 = 0;
  if (auto *C = dyn_cast<ConstantInt>(SubK)) {
    Diff1 = C->getSExtValue();
  } else {
    return None;
  }

  return (Diff0 + Diff1) * Scale;

#undef CallBuilder
}

template <typename T>
auto HexagonVectorCombine::isSafeToMoveBeforeInBB(const Instruction &In,
                                                  BasicBlock::const_iterator To,
                                                  const T &Ignore) const
    -> bool {
  auto getLocOrNone = [this](const Instruction &I) -> Optional<MemoryLocation> {
    if (const auto *II = dyn_cast<IntrinsicInst>(&I)) {
      switch (II->getIntrinsicID()) {
      case Intrinsic::masked_load:
        return MemoryLocation::getForArgument(II, 0, TLI);
      case Intrinsic::masked_store:
        return MemoryLocation::getForArgument(II, 1, TLI);
      }
    }
    return MemoryLocation::getOrNone(&I);
  };

  // The source and the destination must be in the same basic block.
  const BasicBlock &Block = *In.getParent();
  assert(Block.begin() == To || Block.end() == To || To->getParent() == &Block);
  // No PHIs.
  if (isa<PHINode>(In) || (To != Block.end() && isa<PHINode>(*To)))
    return false;

  if (!mayBeMemoryDependent(In))
    return true;
  bool MayWrite = In.mayWriteToMemory();
  auto MaybeLoc = getLocOrNone(In);

  auto From = In.getIterator();
  if (From == To)
    return true;
  bool MoveUp = (To != Block.end() && To->comesBefore(&In));
  auto Range =
      MoveUp ? std::make_pair(To, From) : std::make_pair(std::next(From), To);
  for (auto It = Range.first; It != Range.second; ++It) {
    const Instruction &I = *It;
    if (llvm::is_contained(Ignore, &I))
      continue;
    // Parts based on isSafeToMoveBefore from CoveMoverUtils.cpp.
    if (I.mayThrow())
      return false;
    if (auto *CB = dyn_cast<CallBase>(&I)) {
      if (!CB->hasFnAttr(Attribute::WillReturn))
        return false;
      if (!CB->hasFnAttr(Attribute::NoSync))
        return false;
    }
    if (I.mayReadOrWriteMemory()) {
      auto MaybeLocI = getLocOrNone(I);
      if (MayWrite || I.mayWriteToMemory()) {
        if (!MaybeLoc || !MaybeLocI)
          return false;
        if (!AA.isNoAlias(*MaybeLoc, *MaybeLocI))
          return false;
      }
    }
  }
  return true;
}

#ifndef NDEBUG
auto HexagonVectorCombine::isByteVecTy(Type *Ty) const -> bool {
  if (auto *VecTy = dyn_cast<VectorType>(Ty))
    return VecTy->getElementType() == getByteTy();
  return false;
}

auto HexagonVectorCombine::isSectorTy(Type *Ty) const -> bool {
  if (!isByteVecTy(Ty))
    return false;
  int Size = getSizeOf(Ty);
  if (HST.isTypeForHVX(Ty))
    return Size == static_cast<int>(HST.getVectorLength());
  return Size == 4 || Size == 8;
}
#endif

auto HexagonVectorCombine::getElementRange(IRBuilder<> &Builder, Value *Lo,
                                           Value *Hi, int Start,
                                           int Length) const -> Value * {
  assert(0 <= Start && Start < Length);
  SmallVector<int, 128> SMask(Length);
  std::iota(SMask.begin(), SMask.end(), Start);
  return Builder.CreateShuffleVector(Lo, Hi, SMask);
}

// Pass management.

namespace llvm {
void initializeHexagonVectorCombineLegacyPass(PassRegistry &);
FunctionPass *createHexagonVectorCombineLegacyPass();
} // namespace llvm

namespace {
class HexagonVectorCombineLegacy : public FunctionPass {
public:
  static char ID;

  HexagonVectorCombineLegacy() : FunctionPass(ID) {}

  StringRef getPassName() const override { return "Hexagon Vector Combine"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<TargetPassConfig>();
    FunctionPass::getAnalysisUsage(AU);
  }

  bool runOnFunction(Function &F) override {
    AliasAnalysis &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
    AssumptionCache &AC =
        getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
    DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    TargetLibraryInfo &TLI =
        getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
    auto &TM = getAnalysis<TargetPassConfig>().getTM<HexagonTargetMachine>();
    HexagonVectorCombine HVC(F, AA, AC, DT, TLI, TM);
    return HVC.run();
  }
};
} // namespace

char HexagonVectorCombineLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(HexagonVectorCombineLegacy, DEBUG_TYPE,
                      "Hexagon Vector Combine", false, false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(HexagonVectorCombineLegacy, DEBUG_TYPE,
                    "Hexagon Vector Combine", false, false)

FunctionPass *llvm::createHexagonVectorCombineLegacyPass() {
  return new HexagonVectorCombineLegacy();
}
