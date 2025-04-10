//===- BuildBuiltins.cpp - Utility builder for builtins -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/BuildBuiltins.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/BuildLibCalls.h"

using namespace llvm;

namespace {
static IntegerType *getIntTy(IRBuilderBase &B, const TargetLibraryInfo *TLI) {
  return B.getIntNTy(TLI->getIntSize());
}

static IntegerType *getSizeTTy(IRBuilderBase &B, const TargetLibraryInfo *TLI) {
  const Module *M = B.GetInsertBlock()->getModule();
  return B.getIntNTy(TLI->getSizeTSize(*M));
}

/// In order to use one of the sized library calls such as
/// __atomic_fetch_add_4, the alignment must be sufficient, the size
/// must be one of the potentially-specialized sizes, and the value
/// type must actually exist in C on the target (otherwise, the
/// function wouldn't actually be defined.)
static bool canUseSizedAtomicCall(unsigned Size, Align Alignment,
                                  const DataLayout &DL) {
  // TODO: "LargestSize" is an approximation for "largest type that
  // you can express in C". It seems to be the case that int128 is
  // supported on all 64-bit platforms, otherwise only up to 64-bit
  // integers are supported. If we get this wrong, then we'll try to
  // call a sized libcall that doesn't actually exist. There should
  // really be some more reliable way in LLVM of determining integer
  // sizes which are valid in the target's C ABI...
  unsigned LargestSize = DL.getLargestLegalIntTypeSizeInBits() >= 64 ? 16 : 8;
  return Alignment >= Size &&
         (Size == 1 || Size == 2 || Size == 4 || Size == 8 || Size == 16) &&
         Size <= LargestSize;
}

/// Move the instruction after an InsertPoint to the beginning of another
/// BasicBlock.
///
/// The instructions after \p IP are moved to the beginning of \p New which must
/// not have any PHINodes. If \p CreateBranch is true, a branch instruction to
/// \p New will be added such that there is no semantic change. Otherwise, the
/// \p IP insert block remains degenerate and it is up to the caller to insert a
/// terminator. \p DL is used as the debug location for the branch instruction
/// if one is created.
static void spliceBB(IRBuilderBase::InsertPoint IP, BasicBlock *New,
                     bool CreateBranch, DebugLoc DL) {
  assert(New->getFirstInsertionPt() == New->begin() &&
         "Target BB must not have PHI nodes");

  // Move instructions to new block.
  BasicBlock *Old = IP.getBlock();
  New->splice(New->begin(), Old, IP.getPoint(), Old->end());

  if (CreateBranch) {
    auto *NewBr = BranchInst::Create(New, Old);
    NewBr->setDebugLoc(DL);
  }
}

/// Split a BasicBlock at an InsertPoint, even if the block is degenerate
/// (missing the terminator).
///
/// llvm::SplitBasicBlock and BasicBlock::splitBasicBlock require a well-formed
/// BasicBlock. \p Name is used for the new successor block. If \p CreateBranch
/// is true, a branch to the new successor will new created such that
/// semantically there is no change; otherwise the block of the insertion point
/// remains degenerate and it is the caller's responsibility to insert a
/// terminator. \p DL is used as the debug location for the branch instruction
/// if one is created. Returns the new successor block.
static BasicBlock *splitBB(IRBuilderBase::InsertPoint IP, bool CreateBranch,
                           DebugLoc DL, llvm::Twine Name) {
  BasicBlock *Old = IP.getBlock();
  BasicBlock *New = BasicBlock::Create(
      Old->getContext(), Name.isTriviallyEmpty() ? Old->getName() : Name,
      Old->getParent(), Old->getNextNode());
  spliceBB(IP, New, CreateBranch, DL);
  New->replaceSuccessorsPhiUsesWith(Old, New);
  return New;
}

/// Split a BasicBlock at \p Builder's insertion point, even if the block is
/// degenerate (missing the terminator).  Its new insert location will stick to
/// after the instruction before the insertion point (instead of moving with the
/// instruction the InsertPoint stores internally).
static BasicBlock *splitBB(IRBuilderBase &Builder, bool CreateBranch,
                           llvm::Twine Name) {
  DebugLoc DebugLoc = Builder.getCurrentDebugLocation();
  BasicBlock *New = splitBB(Builder.saveIP(), CreateBranch, DebugLoc, Name);
  if (CreateBranch)
    Builder.SetInsertPoint(Builder.GetInsertBlock()->getTerminator());
  else
    Builder.SetInsertPoint(Builder.GetInsertBlock());
  // SetInsertPoint also updates the Builder's debug location, but we want to
  // keep the one the Builder was configured to use.
  Builder.SetCurrentDebugLocation(DebugLoc);
  return New;
}

// Helper to check if a type is in a variant
template <typename T, typename Variant> struct is_in_variant;

template <typename T, typename... Types>
struct is_in_variant<T, std::variant<Types...>>
    : std::disjunction<std::is_same<T, Types>...> {};

/// Alternative to std::holds_alternative that works even if the std::variant
/// cannot hold T.
template <typename T, typename Variant>
constexpr bool holds_alternative_if_exists(const Variant &v) {
  if constexpr (is_in_variant<T, Variant>::value) {
    return std::holds_alternative<T>(v);
  } else {
    // Type T is not in the variant, return false or handle accordingly
    return false;
  }
}

/// Common code for emitting an atomic builtin (load, store, cmpxchg).
class AtomicEmitter {
public:
  AtomicEmitter(
      Value *Ptr, std::variant<Type *, uint64_t> TypeOrSize,
      std::variant<Value *, bool> IsWeak, bool IsVolatile,
      std::variant<Value *, AtomicOrdering, AtomicOrderingCABI> SuccessMemorder,
      std::variant<std::monostate, Value *, AtomicOrdering, AtomicOrderingCABI>
          FailureMemorder,
      SyncScope::ID Scope, MaybeAlign Align, IRBuilderBase &Builder,
      AtomicEmitOptions EmitOptions, const llvm::Twine &Name)
      : Ctx(Builder.getContext()), CurFn(Builder.GetInsertBlock()->getParent()),
        AtomicPtr(Ptr), TypeOrSize(TypeOrSize), IsWeak(IsWeak),
        IsVolatile(IsVolatile), SuccessMemorder(SuccessMemorder),
        FailureMemorder(FailureMemorder), Scope(Scope), Align(Align),
        Builder(Builder), EmitOptions(std::move(EmitOptions)), Name(Name) {}
  virtual ~AtomicEmitter() = default;

protected:
  LLVMContext &Ctx;
  Function *CurFn;

  Value *AtomicPtr;
  std::variant<Type *, uint64_t> TypeOrSize;
  std::variant<Value *, bool> IsWeak;
  bool IsVolatile;
  std::variant<Value *, AtomicOrdering, AtomicOrderingCABI> SuccessMemorder;
  std::variant<std::monostate, Value *, AtomicOrdering, AtomicOrderingCABI>
      FailureMemorder;
  SyncScope::ID Scope;
  MaybeAlign Align;
  IRBuilderBase &Builder;
  AtomicEmitOptions EmitOptions;
  const Twine &Name;

  uint64_t DataSize;
  Type *CoercedTy = nullptr;
  Type *InstCoercedTy = nullptr;

  llvm::Align EffectiveAlign;
  std::optional<AtomicOrdering> SuccessMemorderConst;
  Value *SuccessMemorderCABI;
  std::optional<AtomicOrdering> FailureMemorderConst;
  Value *FailureMemorderCABI;
  std::optional<bool> IsWeakConst;
  Value *IsWeakVal;

  BasicBlock *createBasicBlock(const Twine &BBName) {
    return BasicBlock::Create(Ctx, Name + "." + getBuiltinSig() + "." + BBName,
                              CurFn);
  };

  virtual const char *getBuiltinSig() const { return "atomic"; }
  virtual bool supportsInstOnFloat() const { return true; }
  virtual bool supportsAcquireOrdering() const { return true; }
  virtual bool supportsReleaseOrdering() const { return true; }

  virtual void prepareInst() {}

  virtual Value *emitInst(bool IsWeak, AtomicOrdering SuccessMemorder,
                          AtomicOrdering FailureMemorder) = 0;

  Value *emitFailureMemorderSwitch(bool IsWeak,
                                   AtomicOrdering SuccessMemorder) {
    if (FailureMemorderConst) {
      // FIXME:  (from CGAtomic)
      // 31.7.2.18: "The failure argument shall not be memory_order_release
      // nor memory_order_acq_rel". Fallback to monotonic.
      //
      // Prior to c++17, "the failure argument shall be no stronger than the
      // success argument". This condition has been lifted and the only
      // precondition is 31.7.2.18. Effectively treat this as a DR and skip
      // language version checks.
      return emitInst(IsWeak, SuccessMemorder, *FailureMemorderConst);
    }

    Type *BoolTy = Builder.getInt1Ty();
    IntegerType *Int32Ty = Builder.getInt32Ty();

    // Create all the relevant BB's
    BasicBlock *ContBB =
        splitBB(Builder, /*CreateBranch=*/false,
                Name + "." + getBuiltinSig() + ".failorder.continue");
    BasicBlock *MonotonicBB = createBasicBlock("monotonic_fail");
    BasicBlock *AcquireBB = createBasicBlock("acquire_fail");
    BasicBlock *SeqCstBB = createBasicBlock("seqcst_fail");

    // MonotonicBB is arbitrarily chosen as the default case; in practice,
    // this doesn't matter unless someone is crazy enough to use something
    // that doesn't fold to a constant for the ordering.
    Value *Order = Builder.CreateIntCast(FailureMemorderCABI, Int32Ty, false);
    SwitchInst *SI = Builder.CreateSwitch(Order, MonotonicBB);

    // TODO: Do not insert PHINode if operation cannot fail
    Builder.SetInsertPoint(ContBB, ContBB->begin());
    PHINode *Result =
        Builder.CreatePHI(BoolTy, /*NumReservedValues=*/3,
                          Name + "." + getBuiltinSig() + ".failorder.success");
    IRBuilderBase::InsertPoint ContIP = Builder.saveIP();

    auto EmitCaseImpl = [&](BasicBlock *CaseBB, AtomicOrdering AO,
                            bool IsDefault = false) {
      if (!IsDefault) {
        for (auto CABI : seq<int>(0, 6)) {
          if (fromCABI(CABI) == AO)
            SI->addCase(Builder.getInt32(CABI), CaseBB);
        }
      }
      Builder.SetInsertPoint(CaseBB);
      Value *AtomicResult = emitInst(IsWeak, SuccessMemorder, AO);
      Builder.CreateBr(ContBB);
      Result->addIncoming(AtomicResult, Builder.GetInsertBlock());
    };

    EmitCaseImpl(MonotonicBB, AtomicOrdering::Monotonic, /*IsDefault=*/true);
    EmitCaseImpl(AcquireBB, AtomicOrdering::Acquire);
    EmitCaseImpl(SeqCstBB, AtomicOrdering::SequentiallyConsistent);

    Builder.restoreIP(ContIP);
    return Result;
  };

  Value *emitSuccessMemorderSwitch(bool IsWeak) {
    if (SuccessMemorderConst)
      return emitFailureMemorderSwitch(IsWeak, *SuccessMemorderConst);

    Type *BoolTy = Builder.getInt1Ty();
    IntegerType *Int32Ty = Builder.getInt32Ty();

    // Create all the relevant BB's
    BasicBlock *ContBB =
        splitBB(Builder, /*CreateBranch=*/false,
                Name + "." + getBuiltinSig() + ".memorder.continue");
    BasicBlock *MonotonicBB = createBasicBlock("monotonic");
    BasicBlock *AcquireBB =
        supportsAcquireOrdering() ? createBasicBlock("acquire") : nullptr;
    BasicBlock *ReleaseBB =
        supportsReleaseOrdering() ? createBasicBlock("release") : nullptr;
    BasicBlock *AcqRelBB =
        supportsAcquireOrdering() && supportsReleaseOrdering()
            ? createBasicBlock("acqrel")
            : nullptr;
    BasicBlock *SeqCstBB = createBasicBlock("seqcst");

    // Create the switch for the split
    // MonotonicBB is arbitrarily chosen as the default case; in practice,
    // this doesn't matter unless someone is crazy enough to use something
    // that doesn't fold to a constant for the ordering.
    Value *Order = Builder.CreateIntCast(SuccessMemorderCABI, Int32Ty, false);
    SwitchInst *SI = Builder.CreateSwitch(Order, MonotonicBB);

    // TODO: No PHI if operation cannot fail
    Builder.SetInsertPoint(ContBB, ContBB->begin());
    PHINode *Result =
        Builder.CreatePHI(BoolTy, /*NumReservedValues=*/5,
                          Name + "." + getBuiltinSig() + ".memorder.success");
    IRBuilderBase::InsertPoint ContIP = Builder.saveIP();

    auto EmitCaseImpl = [&](BasicBlock *CaseBB, AtomicOrdering AO,
                            bool IsDefault = false) {
      if (!CaseBB)
        return;

      if (!IsDefault) {
        for (auto CABI : seq<int>(0, 6)) {
          if (fromCABI(CABI) == AO)
            SI->addCase(Builder.getInt32(CABI), CaseBB);
        }
      }
      Builder.SetInsertPoint(CaseBB);
      Value *AtomicResult = emitFailureMemorderSwitch(IsWeak, AO);
      Builder.CreateBr(ContBB);
      Result->addIncoming(AtomicResult, Builder.GetInsertBlock());
    };

    // Emit all the different atomics.
    EmitCaseImpl(MonotonicBB, AtomicOrdering::Monotonic, /*IsDefault=*/true);
    EmitCaseImpl(AcquireBB, AtomicOrdering::Acquire);
    EmitCaseImpl(ReleaseBB, AtomicOrdering::Release);
    EmitCaseImpl(AcqRelBB, AtomicOrdering::AcquireRelease);
    EmitCaseImpl(SeqCstBB, AtomicOrdering::SequentiallyConsistent);

    Builder.restoreIP(ContIP);
    return Result;
  };

  Value *emitWeakSwitch() {
    if (IsWeakConst)
      return emitSuccessMemorderSwitch(*IsWeakConst);

    // Create all the relevant BBs
    BasicBlock *ContBB =
        splitBB(Builder, /*CreateBranch=*/false,
                Name + "." + getBuiltinSig() + ".weak.continue");
    BasicBlock *StrongBB = createBasicBlock("strong");
    BasicBlock *WeakBB = createBasicBlock("weak");

    // FIXME: Originally copied CGAtomic. Why does it use a switch?
    SwitchInst *SI = Builder.CreateSwitch(IsWeakVal, WeakBB);
    SI->addCase(Builder.getInt1(false), StrongBB);

    Builder.SetInsertPoint(ContBB, ContBB->begin());
    PHINode *Result =
        Builder.CreatePHI(Builder.getInt1Ty(), 2,
                          Name + "." + getBuiltinSig() + ".isweak.success");
    IRBuilderBase::InsertPoint ContIP = Builder.saveIP();

    Builder.SetInsertPoint(StrongBB);
    Value *StrongResult = emitSuccessMemorderSwitch(false);
    Builder.CreateBr(ContBB);
    Result->addIncoming(StrongResult, Builder.GetInsertBlock());

    Builder.SetInsertPoint(WeakBB);
    Value *WeakResult = emitSuccessMemorderSwitch(true);
    Builder.CreateBr(ContBB);
    Result->addIncoming(WeakResult, Builder.GetInsertBlock());

    Builder.restoreIP(ContIP);
    return Result;
  };

  virtual Expected<Value *> emitSizedLibcall() = 0;

  virtual Expected<Value *> emitLibcall() = 0;

  virtual Expected<Value *> makeFallbackError() = 0;

  Expected<Value *> emit() {
    assert(AtomicPtr->getType()->isPointerTy() &&
           "Atomic must apply on pointer");
    assert(EmitOptions.TLI && "TargetLibraryInfo is mandatory");

    unsigned MaxAtomicSizeSupported = 16;
    if (EmitOptions.TL)
      MaxAtomicSizeSupported =
          EmitOptions.TL->getMaxAtomicSizeInBitsSupported() / 8;

    // Determine data size. It is still possible to be unknown after
    // this with SVE types, but neither atomic instructions nor libcall
    // functions support that. After this, *DataSize can be assume to have a
    // value.
    Type *DataType = nullptr;
    if (std::holds_alternative<Type *>(TypeOrSize)) {
      DataType = std::get<Type *>(TypeOrSize);
      TypeSize DS = EmitOptions.DL.getTypeStoreSize(DataType);
      assert(DS.isFixed() && "Atomics on scalable types are invalid");
      DataSize = DS.getFixedValue();
    } else {
      DataSize = std::get<uint64_t>(TypeOrSize);
    }

#ifndef NDEBUG
    if (DataType) {
      // 'long double' (80-bit extended precision) behaves strange here.
      // DL.getTypeStoreSize says it is 10 bytes
      // Clang assumes it is 12 bytes
      // So AtomicExpandPass will disagree with CGAtomic (except for cmpxchg
      // which does not support floats, so AtomicExpandPass doesn't even know it
      // originally was an FP80)
      TypeSize DS = EmitOptions.DL.getTypeStoreSize(DataType);
      assert(DS.getKnownMinValue() <= DataSize &&
             "Must access at least all the relevant bits of the data, possibly "
             "some more for padding");
    }
#endif

    if (Align) {
      EffectiveAlign = *Align;
    } else {
      // https://llvm.org/docs/LangRef.html#cmpxchg-instruction
      //
      //   The alignment is only optional when parsing textual IR; for in-memory
      //   IR, it is always present. If unspecified, the alignment is assumed to
      //   be equal to the size of the ‘<value>’ type.
      //
      // We prefer safety here and assume no alignment, unless
      // getPointerAlignment() can determine the actual alignment.
      // TODO: Would be great if this could determine alignment through a GEP
      EffectiveAlign = AtomicPtr->getPointerAlignment(EmitOptions.DL);
    }

    // Only use the original data type if it is compatible with the atomic
    // instruction (and sized libcall function) and matches the preferred size.
    // No type punning needed when using the libcall function while only takes
    // pointers.
    if (!DataType)
      DataType = IntegerType::get(Ctx, DataSize * 8);

    // Additional type requirements when using an atomic instruction.
    // Since we don't know the size of SVE instructions, can only use keep the
    // original type. If the type is too large, we must not attempt to pass it
    // by value if it wasn't an integer already.
    if (DataType->isIntegerTy() || DataType->isPointerTy() ||
        (supportsInstOnFloat() && DataType->isFloatingPointTy()))
      InstCoercedTy = DataType;
    else if (DataSize > MaxAtomicSizeSupported)
      InstCoercedTy = nullptr;
    else
      InstCoercedTy = IntegerType::get(Ctx, DataSize * 8);

    Type *IntTy = getIntTy(Builder, EmitOptions.TLI);

    // For resolving the SuccessMemorder/FailureMemorder arguments. If it is
    // constant, determine the AtomicOrdering for use with the cmpxchg
    // instruction. Also determines the llvm::Value to be passed to
    // __atomic_compare_exchange in case cmpxchg is not legal.
    auto processMemorder = [&](auto MemorderVariant)
        -> std::pair<std::optional<AtomicOrdering>, Value *> {
      if (holds_alternative_if_exists<std::monostate>(MemorderVariant)) {
        // Derive FailureMemorder from SucccessMemorder
        if (SuccessMemorderConst) {
          MemorderVariant = AtomicCmpXchgInst::getStrongestFailureOrdering(
              *SuccessMemorderConst);
        } else {
          // TODO: If SucccessMemorder is not constant, emit logic that derives
          // the failure ordering from FailureMemorderCABI as
          // getStrongestFailureOrdering() would do. For now use the strongest
          // possible ordering
          MemorderVariant = AtomicOrderingCABI::seq_cst;
        }
      }

      if (std::holds_alternative<AtomicOrdering>(MemorderVariant)) {
        auto Memorder = std::get<AtomicOrdering>(MemorderVariant);
        return std::make_pair(
            Memorder,
            ConstantInt::get(IntTy, static_cast<uint64_t>(toCABI(Memorder))));
      }

      if (std::holds_alternative<AtomicOrderingCABI>(MemorderVariant)) {
        auto MemorderCABI = std::get<AtomicOrderingCABI>(MemorderVariant);
        return std::make_pair(
            fromCABI(MemorderCABI),
            ConstantInt::get(IntTy, static_cast<uint64_t>(MemorderCABI)));
      }

      auto *MemorderCABI = std::get<Value *>(MemorderVariant);
      if (auto *MO = dyn_cast<ConstantInt>(MemorderCABI)) {
        uint64_t MOInt = MO->getZExtValue();
        return std::make_pair(fromCABI(MOInt), MO);
      }

      return std::make_pair(std::nullopt, MemorderCABI);
    };

    auto processIsWeak =
        [&](auto WeakVariant) -> std::pair<std::optional<bool>, Value *> {
      if (std::holds_alternative<bool>(WeakVariant)) {
        bool IsWeakBool = std::get<bool>(WeakVariant);
        return std::make_pair(IsWeakBool, Builder.getInt1(IsWeakBool));
      }

      auto *BoolVal = std::get<Value *>(WeakVariant);
      if (auto *BoolConst = dyn_cast<ConstantInt>(BoolVal)) {
        uint64_t IsWeakBool = BoolConst->getZExtValue();
        return std::make_pair(IsWeakBool != 0, BoolVal);
      }

      return std::make_pair(std::nullopt, BoolVal);
    };

    std::tie(IsWeakConst, IsWeakVal) = processIsWeak(IsWeak);
    std::tie(SuccessMemorderConst, SuccessMemorderCABI) =
        processMemorder(SuccessMemorder);
    std::tie(FailureMemorderConst, FailureMemorderCABI) =
        processMemorder(FailureMemorder);

    // Fix malformed inputs. We do not want to emit illegal IR.
    //
    // https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html
    //
    //   [failure_memorder] This memory order cannot be __ATOMIC_RELEASE nor
    //   __ATOMIC_ACQ_REL. It also cannot be a stronger order than that
    //   specified by success_memorder.
    //
    // https://llvm.org/docs/LangRef.html#cmpxchg-instruction
    //
    //   Both ordering parameters must be at least monotonic, the failure
    //   ordering cannot be either release or acq_rel.
    //
    if (FailureMemorderConst &&
        ((*FailureMemorderConst == AtomicOrdering::Release) ||
         (*FailureMemorderConst == AtomicOrdering::AcquireRelease))) {
      // Fall back to monotonic atomic when illegal value is passed. As with the
      // dynamic case below, it is an arbitrary choice.
      FailureMemorderConst = AtomicOrdering::Monotonic;
    }
    if (FailureMemorderConst && SuccessMemorderConst &&
        !isAtLeastOrStrongerThan(*SuccessMemorderConst,
                                 *FailureMemorderConst)) {
      // Make SuccessMemorder as least as strong as FailureMemorder
      SuccessMemorderConst =
          getMergedAtomicOrdering(*SuccessMemorderConst, *FailureMemorderConst);
    }

    // https://llvm.org/docs/LangRef.html#cmpxchg-instruction
    //
    //   The type of ‘<cmp>’ must be an integer or pointer type whose bit width
    //   is a power of two greater than or equal to eight and less than or equal
    //   to a target-specific size limit.
    bool CanUseInst = DataSize <= MaxAtomicSizeSupported &&
                      llvm::isPowerOf2_64(DataSize) && InstCoercedTy;
    bool CanUseSingleInst = CanUseInst && SuccessMemorderConst &&
                            FailureMemorderConst && IsWeakConst;
    bool CanUseSizedLibcall =
        canUseSizedAtomicCall(DataSize, EffectiveAlign, EmitOptions.DL) &&
        Scope == SyncScope::System;
    bool CanUseLibcall = Scope == SyncScope::System;

    if (CanUseSingleInst && EmitOptions.AllowInstruction) {
      prepareInst();
      return emitInst(*IsWeakConst, *SuccessMemorderConst,
                      *FailureMemorderConst);
    }

    // Switching only needed for cmpxchg instruction which requires constant
    // arguments.
    // FIXME: If AtomicExpandPass later considers the cmpxchg not lowerable for
    // the given target, it will also generate a call to the
    // __atomic_compare_exchange function. In that case the switching was very
    // unnecessary but cannot be undone.
    if (CanUseInst && EmitOptions.AllowSwitch && EmitOptions.AllowInstruction) {
      prepareInst();
      return emitWeakSwitch();
    }

    // Fallback to a libcall function. From here on IsWeak/Scope/IsVolatile is
    // ignored. IsWeak is assumed to be false, Scope is assumed to be
    // SyncScope::System (strongest possible assumption synchronizing with
    // everything, instead of just a subset of sibling threads), and volatile
    // does not apply to function calls.

    if (CanUseSizedLibcall && EmitOptions.AllowSizedLibcall) {
      Expected<Value *> SizedLibcallResult = emitSizedLibcall();
      if (SizedLibcallResult)
        return SizedLibcallResult;
      consumeError(SizedLibcallResult.takeError());
    }

    if (CanUseLibcall && EmitOptions.AllowLibcall) {
      Expected<Value *> LibcallResult = emitLibcall();
      if (LibcallResult)
        return LibcallResult;
      consumeError(LibcallResult.takeError());
    }

    return makeFallbackError();
  }
};

class AtomicLoadEmitter final : public AtomicEmitter {
public:
  using AtomicEmitter::AtomicEmitter;

  Error emitLoad(Value *RetPtr) {
    assert(RetPtr->getType()->isPointerTy());
    this->RetPtr = RetPtr;
    return emit().takeError();
  }

protected:
  Value *RetPtr;

  bool supportsReleaseOrdering() const override { return false; }

  Value *emitInst(bool IsWeak, AtomicOrdering SuccessMemorder,
                  AtomicOrdering FailureMemorder) override {
    LoadInst *AtomicInst = Builder.CreateLoad(
        InstCoercedTy, AtomicPtr, IsVolatile, Name + ".atomic.load");
    AtomicInst->setAtomic(SuccessMemorder, Scope);
    AtomicInst->setAlignment(EffectiveAlign);

    // Store loaded result to where the caller expects it.
    // FIXME: Do we need to zero the padding, if any?
    Builder.CreateStore(AtomicInst, RetPtr, IsVolatile);
    return Builder.getTrue();
  }

  Expected<Value *> emitSizedLibcall() override {
    Value *LoadResult =
        emitAtomicLoadN(DataSize, AtomicPtr, SuccessMemorderCABI, Builder,
                        EmitOptions.DL, EmitOptions.TLI);
    LoadResult->setName(Name);
    if (LoadResult) {
      Builder.CreateStore(LoadResult, RetPtr);
      return Builder.getTrue();
    }

    // emitAtomicLoadN can return nullptr if the backend does not
    // support sized libcalls. Fall back to the non-sized libcall and remove the
    // unused load again.
    return make_error<StringError>("__atomic_load_N libcall absent",
                                   inconvertibleErrorCode());
  }

  Expected<Value *> emitLibcall() override {
    // Fallback to a libcall function. From here on IsWeak/Scope/IsVolatile is
    // ignored. IsWeak is assumed to be false, Scope is assumed to be
    // SyncScope::System (strongest possible assumption synchronizing with
    // everything, instead of just a subset of sibling threads), and volatile
    // does not apply to function calls.

    Value *DataSizeVal =
        ConstantInt::get(getSizeTTy(Builder, EmitOptions.TLI), DataSize);
    Value *LoadCall =
        emitAtomicLoad(DataSizeVal, AtomicPtr, RetPtr, SuccessMemorderCABI,
                       Builder, EmitOptions.DL, EmitOptions.TLI);
    if (!LoadCall)
      return make_error<StringError>("__atomic_load libcall absent",
                                     inconvertibleErrorCode());

    if (!LoadCall->getType()->isVoidTy())
      LoadCall->setName(Name);
    return Builder.getTrue();
  }

  Expected<Value *> makeFallbackError() override {
    return make_error<StringError>(
        "__atomic_load builtin not supported by any available means",
        inconvertibleErrorCode());
  }
};

class AtomicStoreEmitter final : public AtomicEmitter {
public:
  using AtomicEmitter::AtomicEmitter;

  Error emitStore(Value *ValPtr) {
    assert(ValPtr->getType()->isPointerTy());
    this->ValPtr = ValPtr;
    return emit().takeError();
  }

protected:
  Value *ValPtr;
  Value *Val;

  bool supportsAcquireOrdering() const override { return false; }

  void prepareInst() override {
    Val = Builder.CreateLoad(InstCoercedTy, ValPtr, Name + ".atomic.val");
  }

  Value *emitInst(bool IsWeak, AtomicOrdering SuccessMemorder,
                  AtomicOrdering FailureMemorder) override {
    StoreInst *AtomicInst = Builder.CreateStore(Val, AtomicPtr, IsVolatile);
    AtomicInst->setAtomic(SuccessMemorder, Scope);
    AtomicInst->setAlignment(EffectiveAlign);
    return Builder.getTrue();
  }

  Expected<Value *> emitSizedLibcall() override {
    Val = Builder.CreateLoad(CoercedTy, ValPtr, Name + ".atomic.val");
    Value *StoreCall =
        emitAtomicStoreN(DataSize, AtomicPtr, Val, SuccessMemorderCABI, Builder,
                         EmitOptions.DL, EmitOptions.TLI);
    StoreCall->setName(Name);
    if (StoreCall)
      return Builder.getTrue();

    // emitAtomiStoreN can return nullptr if the backend does not
    // support sized libcalls. Fall back to the non-sized libcall and remove the
    // unused load again.
    return make_error<StringError>("__atomic_store_N libcall absent",
                                   inconvertibleErrorCode());
  }

  Expected<Value *> emitLibcall() override {
    // Fallback to a libcall function. From here on IsWeak/Scope/IsVolatile is
    // ignored. IsWeak is assumed to be false, Scope is assumed to be
    // SyncScope::System (strongest possible assumption synchronizing with
    // everything, instead of just a subset of sibling threads), and volatile
    // does not apply to function calls.

    Value *DataSizeVal =
        ConstantInt::get(getSizeTTy(Builder, EmitOptions.TLI), DataSize);
    Value *StoreCall =
        emitAtomicStore(DataSizeVal, AtomicPtr, ValPtr, SuccessMemorderCABI,
                        Builder, EmitOptions.DL, EmitOptions.TLI);
    if (!StoreCall)
      return make_error<StringError>("__atomic_store libcall absent",
                                     inconvertibleErrorCode());

    return Builder.getTrue();
  }

  Expected<Value *> makeFallbackError() override {
    return make_error<StringError>(
        "__atomic_store builtin not supported by any available means",
        inconvertibleErrorCode());
  }
};

class AtomicCompareExchangeEmitter final : public AtomicEmitter {
public:
  using AtomicEmitter::AtomicEmitter;

  Expected<Value *> emitCmpXchg(Value *ExpectedPtr, Value *DesiredPtr,
                                Value *ActualPtr) {
    assert(ExpectedPtr->getType()->isPointerTy());
    assert(DesiredPtr->getType()->isPointerTy());
    assert(!ActualPtr || ActualPtr->getType()->isPointerTy());
    assert(AtomicPtr != ExpectedPtr);
    assert(AtomicPtr != DesiredPtr);
    assert(AtomicPtr != ActualPtr);
    assert(ActualPtr != DesiredPtr);

    this->ExpectedPtr = ExpectedPtr;
    this->DesiredPtr = DesiredPtr;
    this->ActualPtr = ActualPtr;
    return emit();
  }

protected:
  Value *ExpectedPtr;
  Value *DesiredPtr;
  Value *ActualPtr;
  Value *ExpectedVal;
  Value *DesiredVal;

  const char *getBuiltinSig() const override { return "cmpxchg"; }

  bool supportsInstOnFloat() const override { return false; }

  void prepareInst() override {
    ExpectedVal = Builder.CreateLoad(InstCoercedTy, ExpectedPtr,
                                     Name + ".cmpxchg.expected");
    DesiredVal = Builder.CreateLoad(InstCoercedTy, DesiredPtr,
                                    Name + ".cmpxchg.desired");
  }

  Value *emitInst(bool IsWeak, AtomicOrdering SuccessMemorder,
                  AtomicOrdering FailureMemorder) override {
    AtomicCmpXchgInst *AtomicInst =
        Builder.CreateAtomicCmpXchg(AtomicPtr, ExpectedVal, DesiredVal, Align,
                                    SuccessMemorder, FailureMemorder, Scope);
    AtomicInst->setName(Name + ".cmpxchg.pair");
    AtomicInst->setAlignment(EffectiveAlign);
    AtomicInst->setWeak(IsWeak);
    AtomicInst->setVolatile(IsVolatile);

    if (ActualPtr) {
      Value *ActualVal = Builder.CreateExtractValue(AtomicInst, /*Idxs=*/0,
                                                    Name + ".cmpxchg.prev");
      Builder.CreateStore(ActualVal, ActualPtr);
    }
    Value *SuccessFailureVal = Builder.CreateExtractValue(
        AtomicInst, /*Idxs=*/1, Name + ".cmpxchg.success");

    assert(SuccessFailureVal->getType()->isIntegerTy(1));
    return SuccessFailureVal;
  }

  Expected<Value *> emitSizedLibcall() override {
    LoadInst *DesiredVal =
        Builder.CreateLoad(IntegerType::get(Ctx, DataSize * 8), DesiredPtr,
                           Name + ".cmpxchg.desired");
    Value *SuccessResult = emitAtomicCompareExchangeN(
        DataSize, AtomicPtr, ExpectedPtr, DesiredVal, SuccessMemorderCABI,
        FailureMemorderCABI, Builder, EmitOptions.DL, EmitOptions.TLI);
    if (SuccessResult) {
      Value *SuccessBool =
          Builder.CreateCmp(CmpInst::Predicate::ICMP_EQ, SuccessResult,
                            Builder.getInt8(0), Name + ".cmpxchg.success");

      if (ActualPtr && ActualPtr != ExpectedPtr)
        Builder.CreateMemCpy(ActualPtr, {}, ExpectedPtr, {}, DataSize);
      return SuccessBool;
    }

    // emitAtomicCompareExchangeN can return nullptr if the backend does not
    // support sized libcalls. Fall back to the non-sized libcall and remove the
    // unused load again.
    DesiredVal->eraseFromParent();
    return make_error<StringError>("__atomic_compare_exchange_N libcall absent",
                                   inconvertibleErrorCode());
  }

  Expected<Value *> emitLibcall() override {
    // FIXME: Some AMDGCN regression tests the addrspace, but
    // __atomic_compare_exchange by definition is addrsspace(0) and
    // emitAtomicCompareExchange will complain about it.
    if (AtomicPtr->getType()->getPointerAddressSpace() ||
        ExpectedPtr->getType()->getPointerAddressSpace() ||
        DesiredPtr->getType()->getPointerAddressSpace())
      return Builder.getInt1(false);

    Value *SuccessResult = emitAtomicCompareExchange(
        ConstantInt::get(getSizeTTy(Builder, EmitOptions.TLI), DataSize),
        AtomicPtr, ExpectedPtr, DesiredPtr, SuccessMemorderCABI,
        FailureMemorderCABI, Builder, EmitOptions.DL, EmitOptions.TLI);
    if (!SuccessResult)
      return make_error<StringError>("__atomic_compare_exchange libcall absent",
                                     inconvertibleErrorCode());

    Value *SuccessBool =
        Builder.CreateCmp(CmpInst::Predicate::ICMP_EQ, SuccessResult,
                          Builder.getInt8(0), Name + ".cmpxchg.success");

    if (ActualPtr && ActualPtr != ExpectedPtr)
      Builder.CreateMemCpy(ActualPtr, {}, ExpectedPtr, {}, DataSize);
    return SuccessBool;
  }

  Expected<Value *> makeFallbackError() override {
    return make_error<StringError>("__atomic_compare_exchange builtin not "
                                   "supported by any available means",
                                   inconvertibleErrorCode());
  }
};

} // namespace

Error llvm::emitAtomicLoadBuiltin(
    Value *AtomicPtr, Value *RetPtr, std::variant<Type *, uint64_t> TypeOrSize,
    bool IsVolatile,
    std::variant<Value *, AtomicOrdering, AtomicOrderingCABI> Memorder,
    SyncScope::ID Scope, MaybeAlign Align, IRBuilderBase &Builder,
    AtomicEmitOptions EmitOptions, const Twine &Name) {
  AtomicLoadEmitter Emitter(AtomicPtr, TypeOrSize, false, IsVolatile, Memorder,
                            {}, Scope, Align, Builder, EmitOptions, Name);
  return Emitter.emitLoad(RetPtr);
}

Error llvm::emitAtomicStoreBuiltin(
    Value *AtomicPtr, Value *ValPtr, std::variant<Type *, uint64_t> TypeOrSize,
    bool IsVolatile,
    std::variant<Value *, AtomicOrdering, AtomicOrderingCABI> Memorder,
    SyncScope::ID Scope, MaybeAlign Align, IRBuilderBase &Builder,
    AtomicEmitOptions EmitOptions, const Twine &Name) {
  AtomicStoreEmitter Emitter(AtomicPtr, TypeOrSize, false, IsVolatile, Memorder,
                             {}, Scope, Align, Builder, EmitOptions, Name);
  return Emitter.emitStore(ValPtr);
}

Expected<Value *> llvm::emitAtomicCompareExchangeBuiltin(
    Value *AtomicPtr, Value *ExpectedPtr, Value *DesiredPtr,
    std::variant<Type *, uint64_t> TypeOrSize,
    std::variant<Value *, bool> IsWeak, bool IsVolatile,
    std::variant<Value *, AtomicOrdering, AtomicOrderingCABI> SuccessMemorder,
    std::variant<std::monostate, Value *, AtomicOrdering, AtomicOrderingCABI>
        FailureMemorder,
    SyncScope::ID Scope, Value *PrevPtr, MaybeAlign Align,
    IRBuilderBase &Builder, AtomicEmitOptions EmitOptions, const Twine &Name) {
  AtomicCompareExchangeEmitter Emitter(
      AtomicPtr, TypeOrSize, IsWeak, IsVolatile, SuccessMemorder,
      FailureMemorder, Scope, Align, Builder, EmitOptions, Name);
  return Emitter.emitCmpXchg(ExpectedPtr, DesiredPtr, PrevPtr);
}
