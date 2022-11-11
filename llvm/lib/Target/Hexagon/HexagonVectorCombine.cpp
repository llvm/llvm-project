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
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/InstSimplifyFolder.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsHexagon.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/Local.h"

#include "HexagonSubtarget.h"
#include "HexagonTargetMachine.h"

#include <algorithm>
#include <deque>
#include <map>
#include <optional>
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
  IntegerType *getIntTy(unsigned Width = 32) const;
  // Byte type: either scalar (when Length = 0), or vector with given
  // element count.
  Type *getByteTy(int ElemCount = 0) const;
  // Boolean type: either scalar (when Length = 0), or vector with given
  // element count.
  Type *getBoolTy(int ElemCount = 0) const;
  // Create a ConstantInt of type returned by getIntTy with the value Val.
  ConstantInt *getConstInt(int Val) const;
  // Get the integer value of V, if it exists.
  std::optional<APInt> getIntValue(const Value *Val) const;
  // Is V a constant 0, or a vector of 0s?
  bool isZero(const Value *Val) const;
  // Is V an undef value?
  bool isUndef(const Value *Val) const;

  // Get HVX vector type with the given element type.
  VectorType *getHvxTy(Type *ElemTy, bool Pair = false) const;

  enum SizeKind {
    Store, // Store size
    Alloc, // Alloc size
  };
  int getSizeOf(const Value *Val, SizeKind Kind = Store) const;
  int getSizeOf(const Type *Ty, SizeKind Kind = Store) const;
  int getTypeAlignment(Type *Ty) const;
  size_t length(Value *Val) const;
  size_t length(Type *Ty) const;

  Constant *getNullValue(Type *Ty) const;
  Constant *getFullValue(Type *Ty) const;
  Constant *getConstSplat(Type *Ty, int Val) const;

  Value *insertb(IRBuilderBase &Builder, Value *Dest, Value *Src, int Start,
                 int Length, int Where) const;
  Value *vlalignb(IRBuilderBase &Builder, Value *Lo, Value *Hi,
                  Value *Amt) const;
  Value *vralignb(IRBuilderBase &Builder, Value *Lo, Value *Hi,
                  Value *Amt) const;
  Value *concat(IRBuilderBase &Builder, ArrayRef<Value *> Vecs) const;
  Value *vresize(IRBuilderBase &Builder, Value *Val, int NewSize,
                 Value *Pad) const;
  Value *rescale(IRBuilderBase &Builder, Value *Mask, Type *FromTy,
                 Type *ToTy) const;
  Value *vlsb(IRBuilderBase &Builder, Value *Val) const;
  Value *vbytes(IRBuilderBase &Builder, Value *Val) const;
  Value *subvector(IRBuilderBase &Builder, Value *Val, unsigned Start,
                   unsigned Length) const;
  Value *sublo(IRBuilderBase &Builder, Value *Val) const;
  Value *subhi(IRBuilderBase &Builder, Value *Val) const;
  Value *vdeal(IRBuilderBase &Builder, Value *Val0, Value *Val1) const;
  Value *vshuff(IRBuilderBase &Builder, Value *Val0, Value *Val1) const;

  Value *createHvxIntrinsic(IRBuilderBase &Builder, Intrinsic::ID IntID,
                            Type *RetTy, ArrayRef<Value *> Args,
                            ArrayRef<Type *> ArgTys = None) const;
  SmallVector<Value *> splitVectorElements(IRBuilderBase &Builder, Value *Vec,
                                           unsigned ToWidth) const;
  Value *joinVectorElements(IRBuilderBase &Builder, ArrayRef<Value *> Values,
                            VectorType *ToType) const;

  std::optional<int> calculatePointerDifference(Value *Ptr0, Value *Ptr1) const;

  unsigned getNumSignificantBits(const Value *V,
                                 const Instruction *CtxI = nullptr) const;
  KnownBits getKnownBits(const Value *V,
                         const Instruction *CtxI = nullptr) const;

  template <typename T = std::vector<Instruction *>>
  bool isSafeToMoveBeforeInBB(const Instruction &In,
                              BasicBlock::const_iterator To,
                              const T &IgnoreInsts = {}) const;

  // This function is only used for assertions at the moment.
  [[maybe_unused]] bool isByteVecTy(Type *Ty) const;

  Function &F;
  const DataLayout &DL;
  AliasAnalysis &AA;
  AssumptionCache &AC;
  DominatorTree &DT;
  TargetLibraryInfo &TLI;
  const HexagonSubtarget &HST;

private:
  Value *getElementRange(IRBuilderBase &Builder, Value *Lo, Value *Hi,
                         int Start, int Length) const;
};

class AlignVectors {
public:
  AlignVectors(const HexagonVectorCombine &HVC_) : HVC(HVC_) {}

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
    AddrInfo &operator=(const AddrInfo &) = default;

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
      // Segment of a Value: 'Len' bytes starting at byte 'Begin'.
      Segment(Value *Val, int Begin, int Len)
          : Val(Val), Start(Begin), Size(Len) {}
      Segment(const Segment &Seg) = default;
      Segment &operator=(const Segment &Seg) = default;
      Value *Val; // Value representable as a sequence of bytes.
      int Start;  // First byte of the value that belongs to the segment.
      int Size;   // Number of bytes in the segment.
    };

    struct Block {
      Block(Value *Val, int Len, int Pos) : Seg(Val, 0, Len), Pos(Pos) {}
      Block(Value *Val, int Off, int Len, int Pos)
          : Seg(Val, Off, Len), Pos(Pos) {}
      Block(const Block &Blk) = default;
      Block &operator=(const Block &Blk) = default;
      Segment Seg; // Value segment.
      int Pos;     // Position (offset) of the segment in the Block.
    };

    int extent() const;
    ByteSpan section(int Start, int Length) const;
    ByteSpan &shift(int Offset);
    SmallVector<Value *, 8> values() const;

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
  std::optional<MemoryLocation> getLocation(const Instruction &In) const;
  std::optional<AddrInfo> getAddrInfo(Instruction &In) const;
  bool isHvx(const AddrInfo &AI) const;
  // This function is only used for assertions at the moment.
  [[maybe_unused]] bool isSectorTy(Type *Ty) const;

  Value *getPayload(Value *Val) const;
  Value *getMask(Value *Val) const;
  Value *getPassThrough(Value *Val) const;

  Value *createAdjustedPointer(IRBuilderBase &Builder, Value *Ptr, Type *ValTy,
                               int Adjust) const;
  Value *createAlignedPointer(IRBuilderBase &Builder, Value *Ptr, Type *ValTy,
                              int Alignment) const;
  Value *createAlignedLoad(IRBuilderBase &Builder, Type *ValTy, Value *Ptr,
                           int Alignment, Value *Mask, Value *PassThru) const;
  Value *createAlignedStore(IRBuilderBase &Builder, Value *Val, Value *Ptr,
                            int Alignment, Value *Mask) const;

  DepList getUpwardDeps(Instruction *In, Instruction *Base) const;
  bool createAddressGroups();
  MoveList createLoadGroups(const AddrList &Group) const;
  MoveList createStoreGroups(const AddrList &Group) const;
  bool move(const MoveGroup &Move) const;
  void realignLoadGroup(IRBuilderBase &Builder, const ByteSpan &VSpan,
                        int ScLen, Value *AlignVal, Value *AlignAddr) const;
  void realignStoreGroup(IRBuilderBase &Builder, const ByteSpan &VSpan,
                         int ScLen, Value *AlignVal, Value *AlignAddr) const;
  bool realignGroup(const MoveGroup &Move) const;

  friend raw_ostream &operator<<(raw_ostream &OS, const AddrInfo &AI);
  friend raw_ostream &operator<<(raw_ostream &OS, const MoveGroup &MG);
  friend raw_ostream &operator<<(raw_ostream &OS, const ByteSpan::Block &B);
  friend raw_ostream &operator<<(raw_ostream &OS, const ByteSpan &BS);

  std::map<Instruction *, AddrList> AddrGroups;
  const HexagonVectorCombine &HVC;
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
raw_ostream &operator<<(raw_ostream &OS,
                        const AlignVectors::ByteSpan::Block &B) {
  OS << "  @" << B.Pos << " [" << B.Seg.Start << ',' << B.Seg.Size << "] "
     << *B.Seg.Val;
  return OS;
}

LLVM_ATTRIBUTE_UNUSED
raw_ostream &operator<<(raw_ostream &OS, const AlignVectors::ByteSpan &BS) {
  OS << "ByteSpan[size=" << BS.size() << ", extent=" << BS.extent() << '\n';
  for (const AlignVectors::ByteSpan::Block &B : BS)
    OS << B << '\n';
  OS << ']';
  return OS;
}

class HvxIdioms {
public:
  HvxIdioms(const HexagonVectorCombine &HVC_) : HVC(HVC_) {
    auto *Int32Ty = HVC.getIntTy(32);
    HvxI32Ty = HVC.getHvxTy(Int32Ty, /*Pair=*/false);
    HvxP32Ty = HVC.getHvxTy(Int32Ty, /*Pair=*/true);
  }

  bool run();

private:
  struct FxpOp {
    unsigned Opcode;
    unsigned Frac; // Number of fraction bits
    Value *X, *Y;
    // If present, add 1 << RoundAt before shift:
    std::optional<unsigned> RoundAt;
  };

  // Value + sign
  // This is to distinguish multiplications: s*s, s*u, u*s, u*u.
  struct SValue {
    Value *Val;
    bool Signed;
  };

  std::optional<FxpOp> matchFxpMul(Instruction &In) const;
  Value *processFxpMul(Instruction &In, const FxpOp &Op) const;

  Value *processFxpMulChopped(IRBuilderBase &Builder, Instruction &In,
                              const FxpOp &Op) const;
  Value *createMulQ15(IRBuilderBase &Builder, Value *X, Value *Y,
                      bool Rounding) const;
  Value *createMulQ31(IRBuilderBase &Builder, Value *X, Value *Y,
                      bool Rounding) const;
  std::pair<Value *, Value *> createMul32(IRBuilderBase &Builder, SValue X,
                                          SValue Y) const;

  VectorType *HvxI32Ty;
  VectorType *HvxP32Ty;
  const HexagonVectorCombine &HVC;

  friend raw_ostream &operator<<(raw_ostream &, const FxpOp &);
};

[[maybe_unused]] raw_ostream &operator<<(raw_ostream &OS,
                                         const HvxIdioms::FxpOp &Op) {
  OS << Instruction::getOpcodeName(Op.Opcode) << '.' << Op.Frac;
  if (Op.RoundAt.has_value()) {
    if (Op.Frac != 0 && Op.RoundAt.value() == Op.Frac - 1) {
      OS << ":rnd";
    } else {
      OS << " + 1<<" << Op.RoundAt.value();
    }
  }
  OS << "\n  X:" << *Op.X << "\n  Y:" << *Op.Y;
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

#if !defined(_MSC_VER) || _MSC_VER >= 1926
// VS2017 and some versions of VS2019 have trouble compiling this:
// error C2976: 'std::map': too few template arguments
// VS 2019 16.x is known to work, except for 16.4/16.5 (MSC_VER 1924/1925)
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

auto AlignVectors::ByteSpan::shift(int Offset) -> ByteSpan & {
  for (Block &B : Blocks)
    B.Pos += Offset;
  return *this;
}

auto AlignVectors::ByteSpan::values() const -> SmallVector<Value *, 8> {
  SmallVector<Value *, 8> Values(Blocks.size());
  for (int i = 0, e = Blocks.size(); i != e; ++i)
    Values[i] = Blocks[i].Seg.Val;
  return Values;
}

auto AlignVectors::getAlignFromValue(const Value *V) const -> Align {
  const auto *C = dyn_cast<ConstantInt>(V);
  assert(C && "Alignment must be a compile-time constant integer");
  return C->getAlignValue();
}

auto AlignVectors::getAddrInfo(Instruction &In) const
    -> std::optional<AddrInfo> {
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
  return std::nullopt;
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
  if (auto *VecTy = dyn_cast<VectorType>(ValTy))
    return HVC.getFullValue(HVC.getBoolTy(HVC.length(VecTy)));
  return HVC.getFullValue(HVC.getBoolTy());
}

auto AlignVectors::getPassThrough(Value *Val) const -> Value * {
  if (auto *II = dyn_cast<IntrinsicInst>(Val)) {
    if (II->getIntrinsicID() == Intrinsic::masked_load)
      return II->getArgOperand(3);
  }
  return UndefValue::get(getPayload(Val)->getType());
}

auto AlignVectors::createAdjustedPointer(IRBuilderBase &Builder, Value *Ptr,
                                         Type *ValTy, int Adjust) const
    -> Value * {
  // The adjustment is in bytes, but if it's a multiple of the type size,
  // we don't need to do pointer casts.
  auto *PtrTy = cast<PointerType>(Ptr->getType());
  if (!PtrTy->isOpaque()) {
    Type *ElemTy = PtrTy->getNonOpaquePointerElementType();
    int ElemSize = HVC.getSizeOf(ElemTy, HVC.Alloc);
    if (Adjust % ElemSize == 0 && Adjust != 0) {
      Value *Tmp0 =
          Builder.CreateGEP(ElemTy, Ptr, HVC.getConstInt(Adjust / ElemSize));
      return Builder.CreatePointerCast(Tmp0, ValTy->getPointerTo());
    }
  }

  PointerType *CharPtrTy = Type::getInt8PtrTy(HVC.F.getContext());
  Value *Tmp0 = Builder.CreatePointerCast(Ptr, CharPtrTy);
  Value *Tmp1 = Builder.CreateGEP(Type::getInt8Ty(HVC.F.getContext()), Tmp0,
                                  HVC.getConstInt(Adjust));
  return Builder.CreatePointerCast(Tmp1, ValTy->getPointerTo());
}

auto AlignVectors::createAlignedPointer(IRBuilderBase &Builder, Value *Ptr,
                                        Type *ValTy, int Alignment) const
    -> Value * {
  Value *AsInt = Builder.CreatePtrToInt(Ptr, HVC.getIntTy());
  Value *Mask = HVC.getConstInt(-Alignment);
  Value *And = Builder.CreateAnd(AsInt, Mask);
  return Builder.CreateIntToPtr(And, ValTy->getPointerTo());
}

auto AlignVectors::createAlignedLoad(IRBuilderBase &Builder, Type *ValTy,
                                     Value *Ptr, int Alignment, Value *Mask,
                                     Value *PassThru) const -> Value * {
  assert(!HVC.isUndef(Mask)); // Should this be allowed?
  if (HVC.isZero(Mask))
    return PassThru;
  if (Mask == ConstantInt::getTrue(Mask->getType()))
    return Builder.CreateAlignedLoad(ValTy, Ptr, Align(Alignment));
  return Builder.CreateMaskedLoad(ValTy, Ptr, Align(Alignment), Mask, PassThru);
}

auto AlignVectors::createAlignedStore(IRBuilderBase &Builder, Value *Val,
                                      Value *Ptr, int Alignment,
                                      Value *Mask) const -> Value * {
  if (HVC.isZero(Mask) || HVC.isUndef(Val) || HVC.isUndef(Mask))
    return UndefValue::get(Val->getType());
  if (Mask == ConstantInt::getTrue(Mask->getType()))
    return Builder.CreateAlignedStore(Val, Ptr, Align(Alignment));
  return Builder.CreateMaskedStore(Val, Ptr, Align(Alignment), Mask);
}

auto AlignVectors::getUpwardDeps(Instruction *In, Instruction *Base) const
    -> DepList {
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
    return llvm::none_of(
        G.second, [&](auto &I) { return HVC.HST.isTypeForHVX(I.ValTy); });
  });

  return !AddrGroups.empty();
}

auto AlignVectors::createLoadGroups(const AddrList &Group) const -> MoveList {
  // Form load groups.
  // To avoid complications with moving code across basic blocks, only form
  // groups that are contained within a single basic block.

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

auto AlignVectors::realignLoadGroup(IRBuilderBase &Builder,
                                    const ByteSpan &VSpan, int ScLen,
                                    Value *AlignVal, Value *AlignAddr) const
    -> void {
  Type *SecTy = HVC.getByteTy(ScLen);
  int NumSectors = (VSpan.extent() + ScLen - 1) / ScLen;
  bool DoAlign = !HVC.isZero(AlignVal);
  BasicBlock::iterator BasePos = Builder.GetInsertPoint();
  BasicBlock *BaseBlock = Builder.GetInsertBlock();

  ByteSpan ASpan;
  auto *True = HVC.getFullValue(HVC.getBoolTy(ScLen));
  auto *Undef = UndefValue::get(SecTy);

  SmallVector<Instruction *> Loads(NumSectors + DoAlign, nullptr);

  // We could create all of the aligned loads, and generate the valigns
  // at the location of the first load, but for large load groups, this
  // could create highly suboptimal code (there have been groups of 140+
  // loads in real code).
  // Instead, place the loads/valigns as close to the users as possible.
  // In any case we need to have a mapping from the blocks of VSpan (the
  // span covered by the pre-existing loads) to ASpan (the span covered
  // by the aligned loads). There is a small problem, though: ASpan needs
  // to have pointers to the loads/valigns, but we don't know where to put
  // them yet. We can't use nullptr, because when we create sections of
  // ASpan (corresponding to blocks from VSpan), for each block in the
  // section we need to know which blocks of ASpan they are a part of.
  // To have 1-1 mapping between blocks of ASpan and the temporary value
  // pointers, use the addresses of the blocks themselves.

  // Populate the blocks first, to avoid reallocations of the vector
  // interfering with generating the placeholder addresses.
  for (int Index = 0; Index != NumSectors; ++Index)
    ASpan.Blocks.emplace_back(nullptr, ScLen, Index * ScLen);
  for (int Index = 0; Index != NumSectors; ++Index) {
    ASpan.Blocks[Index].Seg.Val =
        reinterpret_cast<Value *>(&ASpan.Blocks[Index]);
  }

  // Multiple values from VSpan can map to the same value in ASpan. Since we
  // try to create loads lazily, we need to find the earliest use for each
  // value from ASpan.
  DenseMap<void *, Instruction *> EarliestUser;
  auto isEarlier = [](Instruction *A, Instruction *B) {
    if (B == nullptr)
      return true;
    if (A == nullptr)
      return false;
    assert(A->getParent() == B->getParent());
    return A->comesBefore(B);
  };
  auto earliestUser = [&](const auto &Uses) {
    Instruction *User = nullptr;
    for (const Use &U : Uses) {
      auto *I = dyn_cast<Instruction>(U.getUser());
      assert(I != nullptr && "Load used in a non-instruction?");
      // Make sure we only consider at users in this block, but we need
      // to remember if there were users outside the block too. This is
      // because if there are no users, aligned loads will not be created.
      if (I->getParent() == BaseBlock) {
        if (!isa<PHINode>(I))
          User = std::min(User, I, isEarlier);
      } else {
        User = std::min(User, BaseBlock->getTerminator(), isEarlier);
      }
    }
    return User;
  };

  for (const ByteSpan::Block &B : VSpan) {
    ByteSpan ASection = ASpan.section(B.Pos, B.Seg.Size);
    for (const ByteSpan::Block &S : ASection) {
      EarliestUser[S.Seg.Val] = std::min(
          EarliestUser[S.Seg.Val], earliestUser(B.Seg.Val->uses()), isEarlier);
    }
  }

  auto createLoad = [&](IRBuilderBase &Builder, const ByteSpan &VSpan,
                        int Index) {
    Value *Ptr =
        createAdjustedPointer(Builder, AlignAddr, SecTy, Index * ScLen);
    // FIXME: generate a predicated load?
    Value *Load = createAlignedLoad(Builder, SecTy, Ptr, ScLen, True, Undef);
    // If vector shifting is potentially needed, accumulate metadata
    // from source sections of twice the load width.
    int Start = (Index - DoAlign) * ScLen;
    int Width = (1 + DoAlign) * ScLen;
    propagateMetadata(cast<Instruction>(Load),
                      VSpan.section(Start, Width).values());
    return cast<Instruction>(Load);
  };

  auto moveBefore = [this](Instruction *In, Instruction *To) {
    // Move In and its upward dependencies to before To.
    assert(In->getParent() == To->getParent());
    DepList Deps = getUpwardDeps(In, To);
    // DepList is sorted with respect to positions in the basic block.
    for (Instruction *I : Deps)
      I->moveBefore(To);
  };

  // Generate necessary loads at appropriate locations.
  for (int Index = 0; Index != NumSectors + 1; ++Index) {
    // In ASpan, each block will be either a single aligned load, or a
    // valign of a pair of loads. In the latter case, an aligned load j
    // will belong to the current valign, and the one in the previous
    // block (for j > 0).
    Instruction *PrevAt =
        DoAlign && Index > 0 ? EarliestUser[&ASpan[Index - 1]] : nullptr;
    Instruction *ThisAt =
        Index < NumSectors ? EarliestUser[&ASpan[Index]] : nullptr;
    if (auto *Where = std::min(PrevAt, ThisAt, isEarlier)) {
      Builder.SetInsertPoint(Where);
      Loads[Index] = createLoad(Builder, VSpan, Index);
      // We know it's safe to put the load at BasePos, so if it's not safe
      // to move it from this location to BasePos, then the current location
      // is not valid.
      // We can't do this check proactively because we need the load to exist
      // in order to check legality.
      if (!HVC.isSafeToMoveBeforeInBB(*Loads[Index], BasePos))
        moveBefore(Loads[Index], &*BasePos);
    }
  }
  // Generate valigns if needed, and fill in proper values in ASpan
  for (int Index = 0; Index != NumSectors; ++Index) {
    ASpan[Index].Seg.Val = nullptr;
    if (auto *Where = EarliestUser[&ASpan[Index]]) {
      Builder.SetInsertPoint(Where);
      Value *Val = Loads[Index];
      assert(Val != nullptr);
      if (DoAlign) {
        Value *NextLoad = Loads[Index + 1];
        assert(NextLoad != nullptr);
        Val = HVC.vralignb(Builder, Val, NextLoad, AlignVal);
      }
      ASpan[Index].Seg.Val = Val;
    }
  }

  for (const ByteSpan::Block &B : VSpan) {
    ByteSpan ASection = ASpan.section(B.Pos, B.Seg.Size).shift(-B.Pos);
    Value *Accum = UndefValue::get(HVC.getByteTy(B.Seg.Size));
    Builder.SetInsertPoint(cast<Instruction>(B.Seg.Val));

    for (ByteSpan::Block &S : ASection) {
      if (S.Seg.Val == nullptr)
        continue;
      // The processing of the data loaded by the aligned loads
      // needs to be inserted after the data is available.
      Instruction *SegI = cast<Instruction>(S.Seg.Val);
      Builder.SetInsertPoint(&*std::next(SegI->getIterator()));
      Value *Pay = HVC.vbytes(Builder, getPayload(S.Seg.Val));
      Accum = HVC.insertb(Builder, Accum, Pay, S.Seg.Start, S.Seg.Size, S.Pos);
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
}

auto AlignVectors::realignStoreGroup(IRBuilderBase &Builder,
                                     const ByteSpan &VSpan, int ScLen,
                                     Value *AlignVal, Value *AlignAddr) const
    -> void {
  Type *SecTy = HVC.getByteTy(ScLen);
  int NumSectors = (VSpan.extent() + ScLen - 1) / ScLen;
  bool DoAlign = !HVC.isZero(AlignVal);

  // Stores.
  ByteSpan ASpanV, ASpanM;

  // Return a vector value corresponding to the input value Val:
  // either <1 x Val> for scalar Val, or Val itself for vector Val.
  auto MakeVec = [](IRBuilderBase &Builder, Value *Val) -> Value * {
    Type *Ty = Val->getType();
    if (Ty->isVectorTy())
      return Val;
    auto *VecTy = VectorType::get(Ty, 1, /*Scalable=*/false);
    return Builder.CreateBitCast(Val, VecTy);
  };

  // Create an extra "undef" sector at the beginning and at the end.
  // They will be used as the left/right filler in the vlalign step.
  for (int i = (DoAlign ? -1 : 0); i != NumSectors + DoAlign; ++i) {
    // For stores, the size of each section is an aligned vector length.
    // Adjust the store offsets relative to the section start offset.
    ByteSpan VSection = VSpan.section(i * ScLen, ScLen).shift(-i * ScLen);
    Value *AccumV = UndefValue::get(SecTy);
    Value *AccumM = HVC.getNullValue(SecTy);
    for (ByteSpan::Block &S : VSection) {
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
  if (DoAlign) {
    for (int j = 1; j != NumSectors + 2; ++j) {
      Value *PrevV = ASpanV[j - 1].Seg.Val, *ThisV = ASpanV[j].Seg.Val;
      Value *PrevM = ASpanM[j - 1].Seg.Val, *ThisM = ASpanM[j].Seg.Val;
      assert(isSectorTy(PrevV->getType()) && isSectorTy(PrevM->getType()));
      ASpanV[j - 1].Seg.Val = HVC.vlalignb(Builder, PrevV, ThisV, AlignVal);
      ASpanM[j - 1].Seg.Val = HVC.vlalignb(Builder, PrevM, ThisM, AlignVal);
    }
  }

  for (int i = 0; i != NumSectors + DoAlign; ++i) {
    Value *Ptr = createAdjustedPointer(Builder, AlignAddr, SecTy, i * ScLen);
    Value *Val = ASpanV[i].Seg.Val;
    Value *Mask = ASpanM[i].Seg.Val; // bytes
    if (!HVC.isUndef(Val) && !HVC.isZero(Mask)) {
      Value *Store =
          createAlignedStore(Builder, Val, Ptr, ScLen, HVC.vlsb(Builder, Mask));
      // If vector shifting is potentially needed, accumulate metadata
      // from source sections of twice the store width.
      int Start = (i - DoAlign) * ScLen;
      int Width = (1 + DoAlign) * ScLen;
      propagateMetadata(cast<Instruction>(Store),
                        VSpan.section(Start, Width).values());
    }
  }
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
      getMaxOf(MoveInfos, [](const AddrInfo &AI) { return AI.HaveAlign; });
  Align MaxGiven = WithMaxAlign.HaveAlign;

  // Minimum alignment present in the move address group.
  const AddrInfo &WithMinOffset =
      getMaxOf(MoveInfos, [](const AddrInfo &AI) { return -AI.Offset; });

  const AddrInfo &WithMaxNeeded =
      getMaxOf(MoveInfos, [](const AddrInfo &AI) { return AI.NeedAlign; });
  Align MinNeeded = WithMaxNeeded.NeedAlign;

  // Set the builder's insertion point right before the load group, or
  // immediately after the store group. (Instructions in a store group are
  // listed in reverse order.)
  Instruction *InsertAt = Move.Main.front();
  if (!Move.IsLoad) {
    // There should be a terminator (which store isn't, but check anyways).
    assert(InsertAt->getIterator() != InsertAt->getParent()->end());
    InsertAt = &*std::next(InsertAt->getIterator());
  }

  IRBuilder Builder(InsertAt->getParent(), InsertAt->getIterator(),
                    InstSimplifyFolder(HVC.DL));
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

  if (Move.IsLoad)
    realignLoadGroup(Builder, VSpan, ScLen, AlignVal, AlignAddr);
  else
    realignStoreGroup(Builder, VSpan, ScLen, AlignVal, AlignAddr);

  for (auto *Inst : Move.Main)
    Inst->eraseFromParent();

  return true;
}

auto AlignVectors::isSectorTy(Type *Ty) const -> bool {
  if (!HVC.isByteVecTy(Ty))
    return false;
  int Size = HVC.getSizeOf(Ty);
  if (HVC.HST.isTypeForHVX(Ty))
    return Size == static_cast<int>(HVC.HST.getVectorLength());
  return Size == 4 || Size == 8;
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

// --- Begin HvxIdioms

// Match
//   (X * Y) [>> N], or
//   ((X * Y) + (1 << N-1)) >> N
auto HvxIdioms::matchFxpMul(Instruction &In) const -> std::optional<FxpOp> {
  using namespace PatternMatch;
  auto *Ty = In.getType();

  if (!Ty->isVectorTy() || !Ty->getScalarType()->isIntegerTy())
    return std::nullopt;

  unsigned Width = cast<IntegerType>(Ty->getScalarType())->getBitWidth();

  FxpOp Op;
  Value *Exp = &In;

  // Fixed-point multiplication is always shifted right (except when the
  // fraction is 0 bits).
  auto m_Shr = [](auto &&V, auto &&S) {
    return m_CombineOr(m_LShr(V, S), m_AShr(V, S));
  };

  const APInt *Qn = nullptr;
  if (Value * T; match(Exp, m_Shr(m_Value(T), m_APInt(Qn)))) {
    Op.Frac = Qn->getZExtValue();
    Exp = T;
  } else {
    Op.Frac = 0;
  }

  if (Op.Frac > Width)
    return std::nullopt;

  // Check if there is rounding added.
  const APInt *C = nullptr;
  if (Value * T; Op.Frac > 0 && match(Exp, m_Add(m_Value(T), m_APInt(C)))) {
    unsigned CV = C->getZExtValue();
    if (CV != 0 && !isPowerOf2_32(CV))
      return std::nullopt;
    if (CV != 0)
      Op.RoundAt = Log2_32(CV);
    Exp = T;
  }

  // Check if the rest is a multiplication.
  if (match(Exp, m_Mul(m_Value(Op.X), m_Value(Op.Y)))) {
    Op.Opcode = Instruction::Mul;
    return Op;
  }

  return std::nullopt;
}

auto HvxIdioms::processFxpMul(Instruction &In, const FxpOp &Op) const
    -> Value * {
  assert(Op.X->getType() == Op.Y->getType());

  auto *VecTy = cast<VectorType>(Op.X->getType());
  auto *ElemTy = cast<IntegerType>(VecTy->getElementType());
  unsigned ElemWidth = ElemTy->getBitWidth();
  if (ElemWidth < 8 || !isPowerOf2_32(ElemWidth))
    return nullptr;

  unsigned VecLen = HVC.length(VecTy);
  unsigned HvxLen = (8 * HVC.HST.getVectorLength()) / std::min(ElemWidth, 32u);
  if (VecLen % HvxLen != 0)
    return nullptr;

  // FIXME: handle 8-bit multiplications
  if (ElemWidth < 16)
    return nullptr;

  SmallVector<Value *> Results;
  FxpOp ChopOp;
  ChopOp.Opcode = Op.Opcode;
  ChopOp.Frac = Op.Frac;
  ChopOp.RoundAt = Op.RoundAt;

  IRBuilder<InstSimplifyFolder> Builder(In.getParent(), In.getIterator(),
                                        InstSimplifyFolder(HVC.DL));

  for (unsigned V = 0; V != VecLen / HvxLen; ++V) {
    ChopOp.X = HVC.subvector(Builder, Op.X, V * HvxLen, HvxLen);
    ChopOp.Y = HVC.subvector(Builder, Op.Y, V * HvxLen, HvxLen);
    Results.push_back(processFxpMulChopped(Builder, In, ChopOp));
    if (Results.back() == nullptr)
      break;
  }

  if (Results.back() == nullptr) {
    // FIXME: clean up leftover instructions
    return nullptr;
  }

  return HVC.concat(Builder, Results);
}

auto HvxIdioms::processFxpMulChopped(IRBuilderBase &Builder, Instruction &In,
                                     const FxpOp &Op) const -> Value * {
  // FIXME: make this more elegant
  struct TempValues {
    void insert(Value *V) { //
      Values.push_back(V);
    }
    void insert(ArrayRef<Value *> Vs) {
      Values.insert(Values.end(), Vs.begin(), Vs.end());
    }
    void clear() { //
      Values.clear();
    }
    ~TempValues() {
      for (Value *V : llvm::reverse(Values)) {
        if (auto *In = dyn_cast<Instruction>(V))
          In->eraseFromParent();
      }
    }
    SmallVector<Value *> Values;
  };
  TempValues DeleteOnFailure;

  // TODO: Make it general.
  // if (Op.Frac != 15 && Op.Frac != 31)
  //  return nullptr;

  enum Signedness { Positive, Signed, Unsigned };
  auto getNumSignificantBits =
      [this, &In](Value *V) -> std::pair<unsigned, Signedness> {
    unsigned Bits = HVC.getNumSignificantBits(V, &In);
    // The significant bits are calculated including the sign bit. This may
    // add an extra bit for zero-extended values, e.g. (zext i32 to i64) may
    // result in 33 significant bits. To avoid extra words, skip the extra
    // sign bit, but keep information that the value is to be treated as
    // unsigned.
    KnownBits Known = HVC.getKnownBits(V, &In);
    Signedness Sign = Signed;
    if (Bits > 1 && isPowerOf2_32(Bits - 1)) {
      if (Known.Zero.ashr(Bits - 1).isAllOnes()) {
        Sign = Unsigned;
        Bits--;
      }
    }
    // If the top bit of the nearest power-of-2 is zero, this value is
    // positive. It could be treated as either signed or unsigned.
    if (unsigned Pow2 = PowerOf2Ceil(Bits); Pow2 != Bits) {
      if (Known.Zero.ashr(Pow2 - 1).isAllOnes())
        Sign = Positive;
    }
    return {Bits, Sign};
  };

  auto *OrigTy = dyn_cast<VectorType>(Op.X->getType());
  if (OrigTy == nullptr)
    return nullptr;

  auto [BitsX, SignX] = getNumSignificantBits(Op.X);
  auto [BitsY, SignY] = getNumSignificantBits(Op.Y);
  unsigned Width = PowerOf2Ceil(std::max(BitsX, BitsY));

  if (!Op.RoundAt || *Op.RoundAt == Op.Frac - 1) {
    bool Rounding = Op.RoundAt.has_value();
    // The fixed-point intrinsics do signed multiplication.
    if (Width == Op.Frac + 1 && SignX != Unsigned && SignY != Unsigned) {
      auto *TruncTy = VectorType::get(HVC.getIntTy(Width), OrigTy);
      Value *TruncX = Builder.CreateTrunc(Op.X, TruncTy);
      Value *TruncY = Builder.CreateTrunc(Op.Y, TruncTy);
      Value *QMul = nullptr;
      if (Width == 16) {
        QMul = createMulQ15(Builder, TruncX, TruncY, Rounding);
      } else if (Width == 32) {
        QMul = createMulQ31(Builder, TruncX, TruncY, Rounding);
      }
      if (QMul != nullptr)
        return Builder.CreateSExt(QMul, OrigTy);

      if (TruncX != Op.X && isa<Instruction>(TruncX))
        cast<Instruction>(TruncX)->eraseFromParent();
      if (TruncY != Op.Y && isa<Instruction>(TruncY))
        cast<Instruction>(TruncY)->eraseFromParent();
    }
  }

  // FIXME: make it general, _64, addcarry
  if (!HVC.HST.useHVXV62Ops())
    return nullptr;

  // FIXME: make it general
  if (OrigTy->getScalarSizeInBits() < 32)
    return nullptr;

  if (Width > 64)
    return nullptr;

  // At this point, NewX and NewY may be truncated to different element
  // widths to save on the number of multiplications to perform.
  unsigned WidthX =
      PowerOf2Ceil(std::max(BitsX, 32u)); // FIXME: handle shorter ones
  unsigned WidthY = PowerOf2Ceil(std::max(BitsY, 32u));
  Value *NewX = Builder.CreateTrunc(
      Op.X, VectorType::get(HVC.getIntTy(WidthX), HVC.length(Op.X), false));
  Value *NewY = Builder.CreateTrunc(
      Op.Y, VectorType::get(HVC.getIntTy(WidthY), HVC.length(Op.Y), false));
  if (NewX != Op.X)
    DeleteOnFailure.insert(NewX);
  if (NewY != Op.Y)
    DeleteOnFailure.insert(NewY);

  // Break up the arguments NewX and NewY into vectors of smaller widths
  // in preparation of doing the multiplication via HVX intrinsics.
  // TODO:
  // Make sure that the number of elements in NewX/NewY is 32. In the future
  // add generic code that will break up a (presumable long) vector into
  // shorter pieces, pad the last one, then concatenate all the pieces back.
  if (HVC.length(NewX) != 32)
    return nullptr;
  auto WordX = HVC.splitVectorElements(Builder, NewX, /*ToWidth=*/32);
  auto WordY = HVC.splitVectorElements(Builder, NewY, /*ToWidth=*/32);
  auto HvxWordTy = WordX[0]->getType();

  SmallVector<SmallVector<Value *>> Products(WordX.size() + WordY.size());

  // WordX[i] * WordY[j] produces words i+j and i+j+1 of the results,
  // that is halves 2(i+j), 2(i+j)+1, 2(i+j)+2, 2(i+j)+3.
  for (int i = 0, e = WordX.size(); i != e; ++i) {
    for (int j = 0, f = WordY.size(); j != f; ++j) {
      bool SgnX = (i + 1 == e) && SignX != Unsigned;
      bool SgnY = (j + 1 == f) && SignY != Unsigned;
      auto [Lo, Hi] = createMul32(Builder, {WordX[i], SgnX}, {WordY[j], SgnY});
      Products[i + j + 0].push_back(Lo);
      Products[i + j + 1].push_back(Hi);
    }
  }

  // Add the optional rounding to the proper word.
  if (Op.RoundAt.has_value()) {
    Products[*Op.RoundAt / 32].push_back(
        HVC.getConstSplat(HvxWordTy, 1 << (*Op.RoundAt % 32)));
  }

  auto V6_vaddcarry = HVC.HST.getIntrinsicId(Hexagon::V6_vaddcarry);
  Value *NoCarry = HVC.getNullValue(HVC.getBoolTy(HVC.length(HvxWordTy)));
  auto pop_back_or_zero = [this, HvxWordTy](auto &Vector) -> Value * {
    if (Vector.empty())
      return HVC.getNullValue(HvxWordTy);
    auto Last = Vector.back();
    Vector.pop_back();
    return Last;
  };

  for (int i = 0, e = Products.size(); i != e; ++i) {
    while (Products[i].size() > 1) {
      Value *Carry = NoCarry;
      for (int j = i; j != e; ++j) {
        auto &ProdJ = Products[j];
        Value *Ret = HVC.createHvxIntrinsic(
            Builder, V6_vaddcarry, nullptr,
            {pop_back_or_zero(ProdJ), pop_back_or_zero(ProdJ), Carry});
        ProdJ.insert(ProdJ.begin(), Builder.CreateExtractValue(Ret, {0}));
        Carry = Builder.CreateExtractValue(Ret, {1});
      }
    }
  }

  SmallVector<Value *> WordP;
  for (auto &P : Products) {
    assert(P.size() == 1 && "Should have been added together");
    WordP.push_back(P.front());
  }

  // Shift all products right by Op.Frac.
  unsigned SkipWords = Op.Frac / 32;
  Constant *ShiftAmt = HVC.getConstSplat(HvxWordTy, Op.Frac % 32);

  for (int Dst = 0, End = WordP.size() - SkipWords; Dst != End; ++Dst) {
    int Src = Dst + SkipWords;
    Value *Lo = WordP[Src];
    if (Src + 1 < End) {
      Value *Hi = WordP[Src + 1];
      WordP[Dst] = Builder.CreateIntrinsic(HvxWordTy, Intrinsic::fshr,
                                           {Hi, Lo, ShiftAmt});
    } else {
      // The shift of the most significant word.
      WordP[Dst] = Builder.CreateAShr(Lo, ShiftAmt);
    }
  }
  if (SkipWords != 0)
    WordP.resize(WordP.size() - SkipWords);

  DeleteOnFailure.clear();
  Value *Ret = HVC.joinVectorElements(Builder, WordP, OrigTy);
  return Ret;
}

auto HvxIdioms::createMulQ15(IRBuilderBase &Builder, Value *X, Value *Y,
                             bool Rounding) const -> Value * {
  assert(X->getType() == Y->getType());
  assert(X->getType()->getScalarType() == HVC.getIntTy(16));
  if (!HVC.HST.isHVXVectorType(EVT::getEVT(X->getType(), false)))
    return nullptr;

  unsigned HwLen = HVC.HST.getVectorLength();

  if (Rounding) {
    auto V6_vmpyhvsrs = HVC.HST.getIntrinsicId(Hexagon::V6_vmpyhvsrs);
    return HVC.createHvxIntrinsic(Builder, V6_vmpyhvsrs, X->getType(), {X, Y});
  }
  // No rounding, do i16*i16 -> i32, << 1, take upper half.
  auto V6_vmpyhv = HVC.HST.getIntrinsicId(Hexagon::V6_vmpyhv);

  // i16*i16 -> i32 / interleaved
  Value *V1 = HVC.createHvxIntrinsic(Builder, V6_vmpyhv, HvxP32Ty, {X, Y});
  // <<1
  Value *V2 = Builder.CreateAdd(V1, V1);
  // i32 -> i32 deinterleave
  SmallVector<int, 64> DeintMask;
  for (int i = 0; i != static_cast<int>(HwLen) / 4; ++i) {
    DeintMask.push_back(i);
    DeintMask.push_back(i + HwLen / 4);
  }

  Value *V3 =
      HVC.vdeal(Builder, HVC.sublo(Builder, V2), HVC.subhi(Builder, V2));
  // High halves: i32 -> i16
  SmallVector<int, 64> HighMask;
  for (int i = 0; i != static_cast<int>(HwLen) / 2; ++i) {
    HighMask.push_back(2 * i + 1);
  }
  auto *HvxP16Ty = HVC.getHvxTy(HVC.getIntTy(16), /*Pair=*/true);
  Value *V4 = Builder.CreateBitCast(V3, HvxP16Ty);
  return Builder.CreateShuffleVector(V4, HighMask);
}

auto HvxIdioms::createMulQ31(IRBuilderBase &Builder, Value *X, Value *Y,
                             bool Rounding) const -> Value * {
  assert(X->getType() == Y->getType());
  assert(X->getType()->getScalarType() == HVC.getIntTy(32));
  if (!HVC.HST.isHVXVectorType(EVT::getEVT(X->getType(), false)))
    return nullptr;

  auto V6_vmpyewuh = HVC.HST.getIntrinsicId(Hexagon::V6_vmpyewuh);
  auto MpyOddAcc = Rounding
                       ? HVC.HST.getIntrinsicId(Hexagon::V6_vmpyowh_rnd_sacc)
                       : HVC.HST.getIntrinsicId(Hexagon::V6_vmpyowh_sacc);
  Value *V1 =
      HVC.createHvxIntrinsic(Builder, V6_vmpyewuh, X->getType(), {X, Y});
  return HVC.createHvxIntrinsic(Builder, MpyOddAcc, X->getType(), {V1, X, Y});
}

auto HvxIdioms::createMul32(IRBuilderBase &Builder, SValue X, SValue Y) const
    -> std::pair<Value *, Value *> {
  assert(X.Val->getType() == Y.Val->getType());
  assert(X.Val->getType() == HVC.getHvxTy(HVC.getIntTy(32), /*Pair=*/false));

  Intrinsic::ID V6_vmpy_parts;
  if (X.Signed == Y.Signed) {
    V6_vmpy_parts = X.Signed ? Intrinsic::hexagon_V6_vmpyss_parts
                             : Intrinsic::hexagon_V6_vmpyuu_parts;
  } else {
    if (X.Signed)
      std::swap(X, Y);
    V6_vmpy_parts = Intrinsic::hexagon_V6_vmpyus_parts;
  }

  Value *Parts = HVC.createHvxIntrinsic(Builder, V6_vmpy_parts, nullptr,
                                        {X.Val, Y.Val}, {HvxI32Ty});
  Value *Hi = Builder.CreateExtractValue(Parts, {0});
  Value *Lo = Builder.CreateExtractValue(Parts, {1});
  return {Lo, Hi};
}

auto HvxIdioms::run() -> bool {
  bool Changed = false;

  for (BasicBlock &B : HVC.F) {
    for (auto It = B.rbegin(); It != B.rend(); ++It) {
      if (auto Fxm = matchFxpMul(*It)) {
        Value *New = processFxpMul(*It, *Fxm);
        // Always report "changed" for now.
        Changed = true;
        if (!New)
          continue;
        bool StartOver = !isa<Instruction>(New);
        It->replaceAllUsesWith(New);
        RecursivelyDeleteTriviallyDeadInstructions(&*It, &HVC.TLI);
        It = StartOver ? B.rbegin()
                       : cast<Instruction>(New)->getReverseIterator();
        Changed = true;
      }
    }
  }

  return Changed;
}

// --- End HvxIdioms

auto HexagonVectorCombine::run() -> bool {
  if (!HST.useHVXOps())
    return false;

  bool Changed = false;
  Changed |= AlignVectors(*this).run();
  Changed |= HvxIdioms(*this).run();

  return Changed;
}

auto HexagonVectorCombine::getIntTy(unsigned Width) const -> IntegerType * {
  return IntegerType::get(F.getContext(), Width);
}

auto HexagonVectorCombine::getByteTy(int ElemCount) const -> Type * {
  assert(ElemCount >= 0);
  IntegerType *ByteTy = Type::getInt8Ty(F.getContext());
  if (ElemCount == 0)
    return ByteTy;
  return VectorType::get(ByteTy, ElemCount, /*Scalable=*/false);
}

auto HexagonVectorCombine::getBoolTy(int ElemCount) const -> Type * {
  assert(ElemCount >= 0);
  IntegerType *BoolTy = Type::getInt1Ty(F.getContext());
  if (ElemCount == 0)
    return BoolTy;
  return VectorType::get(BoolTy, ElemCount, /*Scalable=*/false);
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
    -> std::optional<APInt> {
  if (auto *CI = dyn_cast<ConstantInt>(Val))
    return CI->getValue();
  return std::nullopt;
}

auto HexagonVectorCombine::isUndef(const Value *Val) const -> bool {
  return isa<UndefValue>(Val);
}

auto HexagonVectorCombine::getHvxTy(Type *ElemTy, bool Pair) const
    -> VectorType * {
  EVT ETy = EVT::getEVT(ElemTy, false);
  assert(ETy.isSimple() && "Invalid HVX element type");
  // Do not allow boolean types here: they don't have a fixed length.
  assert(HST.isHVXElementType(ETy.getSimpleVT(), /*IncludeBool=*/false) &&
         "Invalid HVX element type");
  unsigned HwLen = HST.getVectorLength();
  unsigned NumElems = (8 * HwLen) / ETy.getSizeInBits();
  return VectorType::get(ElemTy, Pair ? 2 * NumElems : NumElems,
                         /*Scalable=*/false);
}

auto HexagonVectorCombine::getSizeOf(const Value *Val, SizeKind Kind) const
    -> int {
  return getSizeOf(Val->getType(), Kind);
}

auto HexagonVectorCombine::getSizeOf(const Type *Ty, SizeKind Kind) const
    -> int {
  auto *NcTy = const_cast<Type *>(Ty);
  switch (Kind) {
  case Store:
    return DL.getTypeStoreSize(NcTy).getFixedValue();
  case Alloc:
    return DL.getTypeAllocSize(NcTy).getFixedValue();
  }
  llvm_unreachable("Unhandled SizeKind enum");
}

auto HexagonVectorCombine::getTypeAlignment(Type *Ty) const -> int {
  // The actual type may be shorter than the HVX vector, so determine
  // the alignment based on subtarget info.
  if (HST.isTypeForHVX(Ty))
    return HST.getVectorLength();
  return DL.getABITypeAlign(Ty).value();
}

auto HexagonVectorCombine::length(Value *Val) const -> size_t {
  return length(Val->getType());
}

auto HexagonVectorCombine::length(Type *Ty) const -> size_t {
  auto *VecTy = dyn_cast<VectorType>(Ty);
  assert(VecTy && "Must be a vector type");
  return VecTy->getElementCount().getFixedValue();
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

auto HexagonVectorCombine::getConstSplat(Type *Ty, int Val) const
    -> Constant * {
  assert(Ty->isVectorTy());
  auto VecTy = cast<VectorType>(Ty);
  Type *ElemTy = VecTy->getElementType();
  // Add support for floats if needed.
  auto *Splat = ConstantVector::getSplat(VecTy->getElementCount(),
                                         ConstantInt::get(ElemTy, Val));
  return Splat;
}

// Insert bytes [Start..Start+Length) of Src into Dst at byte Where.
auto HexagonVectorCombine::insertb(IRBuilderBase &Builder, Value *Dst,
                                   Value *Src, int Start, int Length,
                                   int Where) const -> Value * {
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

auto HexagonVectorCombine::vlalignb(IRBuilderBase &Builder, Value *Lo,
                                    Value *Hi, Value *Amt) const -> Value * {
  assert(Lo->getType() == Hi->getType() && "Argument type mismatch");
  if (isZero(Amt))
    return Hi;
  int VecLen = getSizeOf(Hi);
  if (auto IntAmt = getIntValue(Amt))
    return getElementRange(Builder, Lo, Hi, VecLen - IntAmt->getSExtValue(),
                           VecLen);

  if (HST.isTypeForHVX(Hi->getType())) {
    assert(static_cast<unsigned>(VecLen) == HST.getVectorLength() &&
           "Expecting an exact HVX type");
    return createHvxIntrinsic(Builder, HST.getIntrinsicId(Hexagon::V6_vlalignb),
                              Hi->getType(), {Hi, Lo, Amt});
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

auto HexagonVectorCombine::vralignb(IRBuilderBase &Builder, Value *Lo,
                                    Value *Hi, Value *Amt) const -> Value * {
  assert(Lo->getType() == Hi->getType() && "Argument type mismatch");
  if (isZero(Amt))
    return Lo;
  int VecLen = getSizeOf(Lo);
  if (auto IntAmt = getIntValue(Amt))
    return getElementRange(Builder, Lo, Hi, IntAmt->getSExtValue(), VecLen);

  if (HST.isTypeForHVX(Lo->getType())) {
    assert(static_cast<unsigned>(VecLen) == HST.getVectorLength() &&
           "Expecting an exact HVX type");
    return createHvxIntrinsic(Builder, HST.getIntrinsicId(Hexagon::V6_valignb),
                              Lo->getType(), {Hi, Lo, Amt});
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
auto HexagonVectorCombine::concat(IRBuilderBase &Builder,
                                  ArrayRef<Value *> Vecs) const -> Value * {
  assert(!Vecs.empty());
  SmallVector<int, 256> SMask;
  std::vector<Value *> Work[2];
  int ThisW = 0, OtherW = 1;

  Work[ThisW].assign(Vecs.begin(), Vecs.end());
  while (Work[ThisW].size() > 1) {
    auto *Ty = cast<VectorType>(Work[ThisW].front()->getType());
    SMask.resize(length(Ty) * 2);
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
  SMask.resize(Vecs.size() * length(Vecs.front()->getType()));
  std::iota(SMask.begin(), SMask.end(), 0);
  Value *Total = Work[ThisW].front();
  return Builder.CreateShuffleVector(Total, SMask);
}

auto HexagonVectorCombine::vresize(IRBuilderBase &Builder, Value *Val,
                                   int NewSize, Value *Pad) const -> Value * {
  assert(isa<VectorType>(Val->getType()));
  auto *ValTy = cast<VectorType>(Val->getType());
  assert(ValTy->getElementType() == Pad->getType());

  int CurSize = length(ValTy);
  if (CurSize == NewSize)
    return Val;
  // Truncate?
  if (CurSize > NewSize)
    return getElementRange(Builder, Val, /*Ignored*/ Val, 0, NewSize);
  // Extend.
  SmallVector<int, 128> SMask(NewSize);
  std::iota(SMask.begin(), SMask.begin() + CurSize, 0);
  std::fill(SMask.begin() + CurSize, SMask.end(), CurSize);
  Value *PadVec = Builder.CreateVectorSplat(CurSize, Pad);
  return Builder.CreateShuffleVector(Val, PadVec, SMask);
}

auto HexagonVectorCombine::rescale(IRBuilderBase &Builder, Value *Mask,
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
  int FromCount = length(MaskTy);
  int ToCount = (FromCount * FromSize) / ToSize;
  assert((FromCount * FromSize) % ToSize == 0);

  auto *FromITy = getIntTy(FromSize * 8);
  auto *ToITy = getIntTy(ToSize * 8);

  // Mask <N x i1> -> sext to <N x FromTy> -> bitcast to <M x ToTy> ->
  // -> trunc to <M x i1>.
  Value *Ext = Builder.CreateSExt(
      Mask, VectorType::get(FromITy, FromCount, /*Scalable=*/false));
  Value *Cast = Builder.CreateBitCast(
      Ext, VectorType::get(ToITy, ToCount, /*Scalable=*/false));
  return Builder.CreateTrunc(
      Cast, VectorType::get(getBoolTy(), ToCount, /*Scalable=*/false));
}

// Bitcast to bytes, and return least significant bits.
auto HexagonVectorCombine::vlsb(IRBuilderBase &Builder, Value *Val) const
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
auto HexagonVectorCombine::vbytes(IRBuilderBase &Builder, Value *Val) const
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

auto HexagonVectorCombine::subvector(IRBuilderBase &Builder, Value *Val,
                                     unsigned Start, unsigned Length) const
    -> Value * {
  assert(Start + Length <= length(Val));
  return getElementRange(Builder, Val, /*Ignored*/ Val, Start, Length);
}

auto HexagonVectorCombine::sublo(IRBuilderBase &Builder, Value *Val) const
    -> Value * {
  size_t Len = length(Val);
  assert(Len % 2 == 0 && "Length should be even");
  return subvector(Builder, Val, 0, Len / 2);
}

auto HexagonVectorCombine::subhi(IRBuilderBase &Builder, Value *Val) const
    -> Value * {
  size_t Len = length(Val);
  assert(Len % 2 == 0 && "Length should be even");
  return subvector(Builder, Val, Len / 2, Len / 2);
}

auto HexagonVectorCombine::vdeal(IRBuilderBase &Builder, Value *Val0,
                                 Value *Val1) const -> Value * {
  assert(Val0->getType() == Val1->getType());
  int Len = length(Val0);
  SmallVector<int, 128> Mask(2 * Len);

  for (int i = 0; i != Len; ++i) {
    Mask[i] = 2 * i;           // Even
    Mask[i + Len] = 2 * i + 1; // Odd
  }
  return Builder.CreateShuffleVector(Val0, Val1, Mask);
}

auto HexagonVectorCombine::vshuff(IRBuilderBase &Builder, Value *Val0,
                                  Value *Val1) const -> Value * { //
  assert(Val0->getType() == Val1->getType());
  int Len = length(Val0);
  SmallVector<int, 128> Mask(2 * Len);

  for (int i = 0; i != Len; ++i) {
    Mask[2 * i + 0] = i;       // Val0
    Mask[2 * i + 1] = i + Len; // Val1
  }
  return Builder.CreateShuffleVector(Val0, Val1, Mask);
}

auto HexagonVectorCombine::createHvxIntrinsic(IRBuilderBase &Builder,
                                              Intrinsic::ID IntID, Type *RetTy,
                                              ArrayRef<Value *> Args,
                                              ArrayRef<Type *> ArgTys) const
    -> Value * {
  auto getCast = [&](IRBuilderBase &Builder, Value *Val,
                     Type *DestTy) -> Value * {
    Type *SrcTy = Val->getType();
    if (SrcTy == DestTy)
      return Val;

    // Non-HVX type. It should be a scalar, and it should already have
    // a valid type.
    assert(HST.isTypeForHVX(SrcTy, /*IncludeBool=*/true));

    Type *BoolTy = Type::getInt1Ty(F.getContext());
    if (cast<VectorType>(SrcTy)->getElementType() != BoolTy)
      return Builder.CreateBitCast(Val, DestTy);

    // Predicate HVX vector.
    unsigned HwLen = HST.getVectorLength();
    Intrinsic::ID TC = HwLen == 64 ? Intrinsic::hexagon_V6_pred_typecast
                                   : Intrinsic::hexagon_V6_pred_typecast_128B;
    Function *FI =
        Intrinsic::getDeclaration(F.getParent(), TC, {DestTy, Val->getType()});
    return Builder.CreateCall(FI, {Val});
  };

  Function *IntrFn = Intrinsic::getDeclaration(F.getParent(), IntID, ArgTys);
  FunctionType *IntrTy = IntrFn->getFunctionType();

  SmallVector<Value *, 4> IntrArgs;
  for (int i = 0, e = Args.size(); i != e; ++i) {
    Value *A = Args[i];
    Type *T = IntrTy->getParamType(i);
    if (A->getType() != T) {
      IntrArgs.push_back(getCast(Builder, A, T));
    } else {
      IntrArgs.push_back(A);
    }
  }
  Value *Call = Builder.CreateCall(IntrFn, IntrArgs);

  Type *CallTy = Call->getType();
  if (RetTy == nullptr || CallTy == RetTy)
    return Call;
  // Scalar types should have RetTy matching the call return type.
  assert(HST.isTypeForHVX(CallTy, /*IncludeBool=*/true));
  return getCast(Builder, Call, RetTy);
}

auto HexagonVectorCombine::splitVectorElements(IRBuilderBase &Builder,
                                               Value *Vec,
                                               unsigned ToWidth) const
    -> SmallVector<Value *> {
  // Break a vector of wide elements into a series of vectors with narrow
  // elements:
  //   (...c0:b0:a0, ...c1:b1:a1, ...c2:b2:a2, ...)
  // -->
  //   (a0, a1, a2, ...)    // lowest "ToWidth" bits
  //   (b0, b1, b2, ...)    // the next lowest...
  //   (c0, c1, c2, ...)    // ...
  //   ...
  //
  // The number of elements in each resulting vector is the same as
  // in the original vector.

  auto *VecTy = cast<VectorType>(Vec->getType());
  assert(VecTy->getElementType()->isIntegerTy());
  unsigned FromWidth = VecTy->getScalarSizeInBits();
  assert(isPowerOf2_32(ToWidth) && isPowerOf2_32(FromWidth));
  assert(ToWidth <= FromWidth && "Breaking up into wider elements?");
  unsigned NumResults = FromWidth / ToWidth;

  SmallVector<Value *> Results(NumResults);
  Results[0] = Vec;
  unsigned Length = length(VecTy);

  // Do it by splitting in half, since those operations correspond to deal
  // instructions.
  auto splitInHalf = [&](unsigned Begin, unsigned End, auto splitFunc) -> void {
    // Take V = Results[Begin], split it in L, H.
    // Store Results[Begin] = L, Results[(Begin+End)/2] = H
    // Call itself recursively split(Begin, Half), split(Half+1, End)
    if (Begin + 1 == End)
      return;

    Value *Val = Results[Begin];
    unsigned Width = Val->getType()->getScalarSizeInBits();

    auto *VTy = VectorType::get(getIntTy(Width / 2), 2 * Length, false);
    Value *VVal = Builder.CreateBitCast(Val, VTy);

    Value *Res = vdeal(Builder, sublo(Builder, VVal), subhi(Builder, VVal));

    unsigned Half = (Begin + End) / 2;
    Results[Begin] = sublo(Builder, Res);
    Results[Half] = subhi(Builder, Res);

    splitFunc(Begin, Half, splitFunc);
    splitFunc(Half, End, splitFunc);
  };

  splitInHalf(0, NumResults, splitInHalf);
  return Results;
}

auto HexagonVectorCombine::joinVectorElements(IRBuilderBase &Builder,
                                              ArrayRef<Value *> Values,
                                              VectorType *ToType) const
    -> Value * {
  assert(ToType->getElementType()->isIntegerTy());

  // If the list of values does not have power-of-2 elements, append copies
  // of the sign bit to it, to make the size be 2^n.
  // The reason for this is that the values will be joined in pairs, because
  // otherwise the shuffles will result in convoluted code. With pairwise
  // joins, the shuffles will hopefully be folded into a perfect shuffle.
  // The output will need to be sign-extended to a type with element width
  // being a power-of-2 anyways.
  SmallVector<Value *> Inputs(Values.begin(), Values.end());

  unsigned ToWidth = ToType->getScalarSizeInBits();
  unsigned Width = Inputs.front()->getType()->getScalarSizeInBits();
  assert(Width <= ToWidth);
  assert(isPowerOf2_32(Width) && isPowerOf2_32(ToWidth));
  unsigned Length = length(Inputs.front()->getType());

  unsigned NeedInputs = ToWidth / Width;
  if (Inputs.size() != NeedInputs) {
    Value *Last = Inputs.back();
    Value *Sign =
        Builder.CreateAShr(Last, getConstSplat(Last->getType(), Width - 1));
    Inputs.resize(NeedInputs, Sign);
  }

  while (Inputs.size() > 1) {
    Width *= 2;
    auto *VTy = VectorType::get(getIntTy(Width), Length, false);
    for (int i = 0, e = Inputs.size(); i < e; i += 2) {
      Value *Res = vshuff(Builder, Inputs[i], Inputs[i + 1]);
      Inputs[i / 2] = Builder.CreateBitCast(Res, VTy);
    }
    Inputs.resize(Inputs.size() / 2);
  }

  assert(Inputs.front()->getType() == ToType);
  return Inputs.front();
}

auto HexagonVectorCombine::calculatePointerDifference(Value *Ptr0,
                                                      Value *Ptr1) const
    -> std::optional<int> {
  struct Builder : IRBuilder<> {
    Builder(BasicBlock *B) : IRBuilder<>(B->getTerminator()) {}
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
      if (Value *S = simplifyInstruction(I, Q))
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
    return std::nullopt;

  auto *Gep0 = cast<GetElementPtrInst>(Ptr0);
  auto *Gep1 = cast<GetElementPtrInst>(Ptr1);
  if (Gep0->getPointerOperand() != Gep1->getPointerOperand())
    return std::nullopt;

  Builder B(Gep0->getParent());
  int Scale = getSizeOf(Gep0->getSourceElementType(), Alloc);

  // FIXME: for now only check GEPs with a single index.
  if (Gep0->getNumOperands() != 2 || Gep1->getNumOperands() != 2)
    return std::nullopt;

  Value *Idx0 = Gep0->getOperand(1);
  Value *Idx1 = Gep1->getOperand(1);

  // First, try to simplify the subtraction directly.
  if (auto *Diff = dyn_cast<ConstantInt>(
          Simplify(CallBuilder(B, CreateSub(Idx0, Idx1)))))
    return Diff->getSExtValue() * Scale;

  KnownBits Known0 = getKnownBits(Idx0, Gep0);
  KnownBits Known1 = getKnownBits(Idx1, Gep1);
  APInt Unknown = ~(Known0.Zero | Known0.One) | ~(Known1.Zero | Known1.One);
  if (Unknown.isAllOnes())
    return std::nullopt;

  Value *MaskU = ConstantInt::get(Idx0->getType(), Unknown);
  Value *AndU0 = Simplify(CallBuilder(B, CreateAnd(Idx0, MaskU)));
  Value *AndU1 = Simplify(CallBuilder(B, CreateAnd(Idx1, MaskU)));
  Value *SubU = Simplify(CallBuilder(B, CreateSub(AndU0, AndU1)));
  int Diff0 = 0;
  if (auto *C = dyn_cast<ConstantInt>(SubU)) {
    Diff0 = C->getSExtValue();
  } else {
    return std::nullopt;
  }

  Value *MaskK = ConstantInt::get(MaskU->getType(), ~Unknown);
  Value *AndK0 = Simplify(CallBuilder(B, CreateAnd(Idx0, MaskK)));
  Value *AndK1 = Simplify(CallBuilder(B, CreateAnd(Idx1, MaskK)));
  Value *SubK = Simplify(CallBuilder(B, CreateSub(AndK0, AndK1)));
  int Diff1 = 0;
  if (auto *C = dyn_cast<ConstantInt>(SubK)) {
    Diff1 = C->getSExtValue();
  } else {
    return std::nullopt;
  }

  return (Diff0 + Diff1) * Scale;

#undef CallBuilder
}

auto HexagonVectorCombine::getNumSignificantBits(const Value *V,
                                                 const Instruction *CtxI) const
    -> unsigned {
  return ComputeMaxSignificantBits(V, DL, /*Depth=*/0, &AC, CtxI, &DT);
}

auto HexagonVectorCombine::getKnownBits(const Value *V,
                                        const Instruction *CtxI) const
    -> KnownBits {
  return computeKnownBits(V, DL, /*Depth=*/0, &AC, CtxI, &DT, /*ORE=*/nullptr,
                          /*UseInstrInfo=*/true);
}

template <typename T>
auto HexagonVectorCombine::isSafeToMoveBeforeInBB(const Instruction &In,
                                                  BasicBlock::const_iterator To,
                                                  const T &IgnoreInsts) const
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

  if (!mayHaveNonDefUseDependency(In))
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
    if (llvm::is_contained(IgnoreInsts, &I))
      continue;
    // assume intrinsic can be ignored
    if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
      if (II->getIntrinsicID() == Intrinsic::assume)
        continue;
    }
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

auto HexagonVectorCombine::isByteVecTy(Type *Ty) const -> bool {
  if (auto *VecTy = dyn_cast<VectorType>(Ty))
    return VecTy->getElementType() == getByteTy();
  return false;
}

auto HexagonVectorCombine::getElementRange(IRBuilderBase &Builder, Value *Lo,
                                           Value *Hi, int Start,
                                           int Length) const -> Value * {
  assert(0 <= Start && size_t(Start + Length) < length(Lo) + length(Hi));
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
    if (skipFunction(F))
      return false;
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
