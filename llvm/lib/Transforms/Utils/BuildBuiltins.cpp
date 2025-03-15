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
      Value *Ptr, std::variant<Value *, bool> IsWeak, bool IsVolatile,
      std::variant<Value *, AtomicOrdering, AtomicOrderingCABI> SuccessMemorder,
      std::variant<std::monostate, Value *, AtomicOrdering, AtomicOrderingCABI>
          FailureMemorder,
      SyncScope::ID Scope, Type *DataTy, std::optional<uint64_t> DataSize,
      std::optional<uint64_t> AvailableSize, MaybeAlign Align,
      IRBuilderBase &Builder, AtomicEmitOptions EmitOptions, llvm::Twine Name)
      : Ctx(Builder.getContext()), CurFn(Builder.GetInsertBlock()->getParent()),
        AtomicPtr(Ptr), IsWeak(IsWeak), IsVolatile(IsVolatile),
        SuccessMemorder(SuccessMemorder), FailureMemorder(FailureMemorder),
        Scope(Scope), DataTy(DataTy), DataSize(DataSize),
        AvailableSize(AvailableSize), Align(Align), Builder(Builder),
        EmitOptions(std::move(EmitOptions)), Name(Name) {}

protected:
  LLVMContext &Ctx;
  Function *CurFn;

  Value *AtomicPtr;
  std::variant<Value *, bool> IsWeak;
  bool IsVolatile;
  std::variant<Value *, AtomicOrdering, AtomicOrderingCABI> SuccessMemorder;
  std::variant<std::monostate, Value *, AtomicOrdering, AtomicOrderingCABI>
      FailureMemorder;
  SyncScope::ID Scope;
  Type *DataTy;
  std::optional<uint64_t> DataSize = std::nullopt;
  std::optional<uint64_t> AvailableSize;
  uint64_t PreferredSize;
  MaybeAlign Align;
  IRBuilderBase &Builder;
  AtomicEmitOptions EmitOptions;
  llvm::Twine Name;

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
    return BasicBlock::Create(Ctx, Name + BBName, CurFn);
  };

  virtual bool supportsInstOnFloat() const { return true; }

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

    // Create all the relevant BB's
    BasicBlock *MonotonicBB = createBasicBlock("monotonic_fail");
    BasicBlock *AcquireBB = createBasicBlock("acquire_fail");
    BasicBlock *SeqCstBB = createBasicBlock("seqcst_fail");
    BasicBlock *ContBB = createBasicBlock("atomic.continue");

    // MonotonicBB is arbitrarily chosen as the default case; in practice,
    // this doesn't matter unless someone is crazy enough to use something
    // that doesn't fold to a constant for the ordering.
    llvm::SwitchInst *SI =
        Builder.CreateSwitch(FailureMemorderCABI, MonotonicBB);
    // Implemented as acquire, since it's the closest in LLVM.
    SI->addCase(
        Builder.getInt32(static_cast<int32_t>(AtomicOrderingCABI::consume)),
        AcquireBB);
    SI->addCase(
        Builder.getInt32(static_cast<int32_t>(AtomicOrderingCABI::acquire)),
        AcquireBB);
    SI->addCase(
        Builder.getInt32(static_cast<int32_t>(AtomicOrderingCABI::seq_cst)),
        SeqCstBB);

    // Emit all the different atomics
    Builder.SetInsertPoint(MonotonicBB);
    Value *MonotonicResult =
        emitInst(IsWeak, SuccessMemorder, AtomicOrdering::Monotonic);
    BasicBlock *MonotonicSourceBB = Builder.GetInsertBlock();
    Builder.CreateBr(ContBB);

    Builder.SetInsertPoint(AcquireBB);
    Value *AcquireResult =
        emitInst(IsWeak, SuccessMemorder, AtomicOrdering::Acquire);
    BasicBlock *AcquireSourceBB = Builder.GetInsertBlock();
    Builder.CreateBr(ContBB);

    Builder.SetInsertPoint(SeqCstBB);
    Value *SeqCstResult = emitInst(IsWeak, SuccessMemorder,
                                   AtomicOrdering::SequentiallyConsistent);
    BasicBlock *SeqCstSourceBB = Builder.GetInsertBlock();
    Builder.CreateBr(ContBB);

    Builder.SetInsertPoint(ContBB);
    PHINode *Result = Builder.CreatePHI(BoolTy, 3, Name + ".cmpxchg.success");
    Result->addIncoming(MonotonicResult, MonotonicSourceBB);
    Result->addIncoming(AcquireResult, AcquireSourceBB);
    Result->addIncoming(SeqCstResult, SeqCstSourceBB);
    return Result;
  };

  Value *emitSuccessMemorderSwitch(bool IsWeak) {
    if (SuccessMemorderConst)
      return emitFailureMemorderSwitch(IsWeak, *SuccessMemorderConst);

    Type *BoolTy = Builder.getInt1Ty();

    // Create all the relevant BB's
    BasicBlock *MonotonicBB = createBasicBlock(".monotonic");
    BasicBlock *AcquireBB = createBasicBlock(".acquire");
    BasicBlock *ReleaseBB = createBasicBlock(".release");
    BasicBlock *AcqRelBB = createBasicBlock(".acqrel");
    BasicBlock *SeqCstBB = createBasicBlock(".seqcst");
    BasicBlock *ContBB = createBasicBlock(".atomic.continue");

    // Create the switch for the split
    // MonotonicBB is arbitrarily chosen as the default case; in practice,
    // this doesn't matter unless someone is crazy enough to use something
    // that doesn't fold to a constant for the ordering.
    Value *Order =
        Builder.CreateIntCast(SuccessMemorderCABI, Builder.getInt32Ty(), false);
    llvm::SwitchInst *SI = Builder.CreateSwitch(Order, MonotonicBB);

    Builder.SetInsertPoint(ContBB);
    PHINode *Result = Builder.CreatePHI(BoolTy, 5, Name + ".cmpxchg.success");

    // Emit all the different atomics
    Builder.SetInsertPoint(MonotonicBB);
    Value *MonotonicResult =
        emitFailureMemorderSwitch(IsWeak, AtomicOrdering::Monotonic);
    Result->addIncoming(MonotonicResult, MonotonicBB);
    Builder.CreateBr(ContBB);

    Builder.SetInsertPoint(AcquireBB);
    Value *AcquireResult =
        emitFailureMemorderSwitch(IsWeak, AtomicOrdering::Acquire);
    Builder.CreateBr(ContBB);
    SI->addCase(
        Builder.getInt32(static_cast<uint32_t>(AtomicOrderingCABI::consume)),
        Builder.GetInsertBlock());
    SI->addCase(
        Builder.getInt32(static_cast<uint32_t>(AtomicOrderingCABI::acquire)),
        Builder.GetInsertBlock());
    Result->addIncoming(AcquireResult, AcquireBB);

    Builder.SetInsertPoint(ReleaseBB);
    Value *ReleaseResult =
        emitFailureMemorderSwitch(IsWeak, AtomicOrdering::Release);
    Builder.CreateBr(ContBB);
    SI->addCase(
        Builder.getInt32(static_cast<uint32_t>(AtomicOrderingCABI::release)),
        Builder.GetInsertBlock());
    Result->addIncoming(ReleaseResult, Builder.GetInsertBlock());

    Builder.SetInsertPoint(AcqRelBB);
    Value *AcqRelResult =
        emitFailureMemorderSwitch(IsWeak, AtomicOrdering::AcquireRelease);
    Builder.CreateBr(ContBB);
    SI->addCase(
        Builder.getInt32(static_cast<uint32_t>(AtomicOrderingCABI::acq_rel)),
        AcqRelBB);
    Result->addIncoming(AcqRelResult, Builder.GetInsertBlock());

    Builder.SetInsertPoint(SeqCstBB);
    Value *SeqCstResult = emitFailureMemorderSwitch(
        IsWeak, AtomicOrdering::SequentiallyConsistent);
    Builder.CreateBr(ContBB);
    SI->addCase(
        Builder.getInt32(static_cast<uint32_t>(AtomicOrderingCABI::seq_cst)),
        SeqCstBB);
    Result->addIncoming(SeqCstResult, Builder.GetInsertBlock());

    Builder.SetInsertPoint(Result->getNextNode());
    return Result;
  };

  Value *emitWeakSwitch() {
    if (IsWeakConst)
      return emitSuccessMemorderSwitch(*IsWeakConst);

    Type *BoolTy = Builder.getInt1Ty();

    // Create all the relevant BB's
    BasicBlock *StrongBB = createBasicBlock(".cmpxchg.strong");
    BasicBlock *WeakBB = createBasicBlock(".cmpxchg.weak");
    BasicBlock *ContBB = createBasicBlock(".cmpxchg.continue");

    // FIXME: Why is this a switch?
    llvm::SwitchInst *SI = Builder.CreateSwitch(IsWeakVal, WeakBB);
    SI->addCase(Builder.getInt1(false), StrongBB);

    Builder.SetInsertPoint(StrongBB);
    Value *StrongResult = emitSuccessMemorderSwitch(false);
    BasicBlock *StrongSourceBB = Builder.GetInsertBlock();
    Builder.CreateBr(ContBB);

    Builder.SetInsertPoint(WeakBB);
    Value *WeakResult = emitSuccessMemorderSwitch(true);
    BasicBlock *WeakSourceBB = Builder.GetInsertBlock();
    Builder.CreateBr(ContBB);

    Builder.SetInsertPoint(ContBB);
    PHINode *Result =
        Builder.CreatePHI(BoolTy, 2, Name + ".cmpxchg.isweak.success");
    Result->addIncoming(WeakResult, WeakSourceBB);
    Result->addIncoming(StrongResult, StrongSourceBB);
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

    // Automatically derive data size. It is still possible to be unknown after
    // this with SVE types, but neither atomic instructions nor libcall
    // functions support that. After this, *DataSize can be assume to have a
    // value.
    if (!DataSize) {
      TypeSize DS = EmitOptions.DL.getTypeStoreSize(DataTy);
      assert(DS.isFixed() && "Atomics on scalable types are invalid");
      DataSize = DS.getFixedValue();
    }

    // At least DataSize is available
    // After this, *AvailableSize can be assume to have a value.
    AvailableSize = AvailableSize.value_or(*DataSize);
    assert(*DataSize <= *AvailableSize);

    // Choose a preferred size between DataSize and AvailableSize which is
    // power-of-2 and at most MaxAtomicSizeSupported
    PreferredSize = *DataSize;
    uint64_t PowerOfTwo = PowerOf2Ceil(*DataSize);
    if (*DataSize <= PowerOfTwo && PowerOfTwo <= *AvailableSize &&
        PowerOfTwo <= MaxAtomicSizeSupported)
      PreferredSize = PowerOfTwo;

#ifndef NDEBUG
    if (DataTy) {
      // 'long double' (80-bit extended precision) behaves strange here.
      // DL.getTypeStoreSize says it is 10 bytes
      // Clang says it is 12 bytes
      // So AtomicExpandPass will disagree with CGAtomic (except for cmpxchg
      // which does not support floats, so AtomicExpandPass doesn't even know it
      // originally was an FP80)
      TypeSize DS = EmitOptions.DL.getTypeStoreSize(DataTy);
      assert(DS.getKnownMinValue() <= *DataSize &&
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

    Type *IntTy = getIntTy(Builder, EmitOptions.TLI);

    // Only use the original data type if it is compatible with the atomic
    // instruction (and sized libcall function) and matches the preferred size.
    // No type punning needed when using the libcall function while only takes
    // pointers.
    CoercedTy = DataTy;
    // If we have rounded-up the data size, unconditionally coerce to a
    // different type.
    if (*DataSize != PreferredSize)
      CoercedTy = IntegerType::get(Ctx, PreferredSize * 8);

    // Additional type requirements when using an atomic instruction.
    // Since we don't know the size of SVE instructions, can only use keep the
    // original type.
    if (CoercedTy->isIntegerTy() || CoercedTy->isPointerTy() ||
        (supportsInstOnFloat() && CoercedTy->isFloatingPointTy()))
      InstCoercedTy = CoercedTy;
    else
      InstCoercedTy = IntegerType::get(Ctx, PreferredSize * 8);

    // For resolving the SuccessMemorder/FailureMemorder arguments. If it is
    // constant, determine the AtomicOrdering for use with the cmpxchg
    // instruction. Also determines the llvm::Value to be passed to
    // __atomic_compare_exchange in case cmpxchg is not legal.
    auto processMemorder = [&](auto MemorderVariant)
        -> std::pair<std::optional<AtomicOrdering>, Value *> {
      if (holds_alternative_if_exists<std::monostate>(MemorderVariant)) {
        // Derive FailureMemorder from SucccessMemorder
        if (SuccessMemorderConst) {
          AtomicOrdering MOFailure =
              AtomicCmpXchgInst::getStrongestFailureOrdering(
                  *SuccessMemorderConst);
          MemorderVariant = MOFailure;
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
    bool CanUseInst = PreferredSize <= MaxAtomicSizeSupported &&
                      llvm::isPowerOf2_64(PreferredSize) && InstCoercedTy;
    bool CanUseSingleInst = CanUseInst && SuccessMemorderConst &&
                            FailureMemorderConst && IsWeakConst;
    bool CanUseSizedLibcall =
        canUseSizedAtomicCall(PreferredSize, EffectiveAlign, EmitOptions.DL) &&
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
    }

    if (CanUseLibcall && EmitOptions.AllowLibcall) {
      Expected<Value *> LibcallResult = emitSizedLibcall();
      if (LibcallResult)
        return LibcallResult;
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

  Value *emitInst(bool IsWeak, AtomicOrdering SuccessMemorder,
                  AtomicOrdering FailureMemorder) override {
    LoadInst *AtomicInst = Builder.CreateLoad(
        InstCoercedTy, AtomicPtr, IsVolatile, Name + ".atomic.load");
    AtomicInst->setAtomic(SuccessMemorder, Scope);
    AtomicInst->setAlignment(EffectiveAlign);

    // Store loaded result to where the caller expects it.
    // FIXME: Do we need to zero the padding, if any?
    Builder.CreateStore(AtomicInst, RetPtr, IsVolatile);
    return nullptr;
  }

  Expected<Value *> emitSizedLibcall() override {
    Value *LoadResult =
        emitAtomicLoadN(PreferredSize, AtomicPtr, SuccessMemorderCABI, Builder,
                        EmitOptions.DL, EmitOptions.TLI);
    LoadResult->setName(Name);
    if (LoadResult) {
      Builder.CreateStore(LoadResult, RetPtr);
      return nullptr;
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
        ConstantInt::get(getSizeTTy(Builder, EmitOptions.TLI), *DataSize);
    Value *LoadCall =
        emitAtomicLoad(DataSizeVal, AtomicPtr, RetPtr, SuccessMemorderCABI,
                       Builder, EmitOptions.DL, EmitOptions.TLI);
    if (!LoadCall)
      return make_error<StringError>("__atomic_load libcall absent",
                                     inconvertibleErrorCode());

    LoadCall->setName(Name);
    return nullptr;
  }

  Expected<Value *> makeFallbackError() override {
    return make_error<StringError>(
        "__atomic_laod builtin not supported by any available means",
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

  void prepareInst() override {
    Val = Builder.CreateLoad(InstCoercedTy, ValPtr, Name + ".atomic.val");
  }

  Value *emitInst(bool IsWeak, AtomicOrdering SuccessMemorder,
                  AtomicOrdering FailureMemorder) override {
    StoreInst *AtomicInst = Builder.CreateStore(Val, AtomicPtr, IsVolatile);
    AtomicInst->setAtomic(SuccessMemorder, Scope);
    AtomicInst->setAlignment(EffectiveAlign);
    return nullptr;
  }

  Expected<Value *> emitSizedLibcall() override {
    Val = Builder.CreateLoad(CoercedTy, ValPtr, Name + ".atomic.val");
    Value *StoreCall =
        emitAtomicStoreN(*DataSize, AtomicPtr, Val, SuccessMemorderCABI,
                         Builder, EmitOptions.DL, EmitOptions.TLI);
    StoreCall->setName(Name);
    if (StoreCall)
      return nullptr;

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
        ConstantInt::get(getSizeTTy(Builder, EmitOptions.TLI), *DataSize);
    Value *StoreCall =
        emitAtomicStore(DataSizeVal, AtomicPtr, ValPtr, SuccessMemorderCABI,
                        Builder, EmitOptions.DL, EmitOptions.TLI);
    if (!StoreCall)
      return make_error<StringError>("__atomic_store libcall absent",
                                     inconvertibleErrorCode());

    return nullptr;
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
        Builder.CreateLoad(IntegerType::get(Ctx, PreferredSize * 8), DesiredPtr,
                           Name + ".cmpxchg.desired");
    Value *SuccessResult = emitAtomicCompareExchangeN(
        PreferredSize, AtomicPtr, ExpectedPtr, DesiredVal, SuccessMemorderCABI,
        FailureMemorderCABI, Builder, EmitOptions.DL, EmitOptions.TLI);
    if (SuccessResult) {
      Value *SuccessBool =
          Builder.CreateCmp(CmpInst::Predicate::ICMP_EQ, SuccessResult,
                            Builder.getInt8(0), Name + ".cmpxchg.success");

      if (ActualPtr && ActualPtr != ExpectedPtr)
        Builder.CreateMemCpy(ActualPtr, {}, ExpectedPtr, {}, *DataSize);
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
        ConstantInt::get(getSizeTTy(Builder, EmitOptions.TLI), *DataSize),
        AtomicPtr, ExpectedPtr, DesiredPtr, SuccessMemorderCABI,
        FailureMemorderCABI, Builder, EmitOptions.DL, EmitOptions.TLI);
    if (!SuccessResult)
      return make_error<StringError>("__atomic_compare_exchange libcall absent",
                                     inconvertibleErrorCode());

    Value *SuccessBool =
        Builder.CreateCmp(CmpInst::Predicate::ICMP_EQ, SuccessResult,
                          Builder.getInt8(0), Name + ".cmpxchg.success");

    if (ActualPtr && ActualPtr != ExpectedPtr)
      Builder.CreateMemCpy(ActualPtr, {}, ExpectedPtr, {}, *DataSize);
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
    Value *AtomicPtr, Value *RetPtr, bool IsVolatile,
    std::variant<Value *, AtomicOrdering, AtomicOrderingCABI> Memorder,
    SyncScope::ID Scope, Type *DataTy, std::optional<uint64_t> DataSize,
    std::optional<uint64_t> AvailableSize, MaybeAlign Align,
    IRBuilderBase &Builder, AtomicEmitOptions EmitOptions, llvm::Twine Name) {
  AtomicLoadEmitter Emitter(AtomicPtr, false, IsVolatile, Memorder, {}, Scope,
                            DataTy, DataSize, AvailableSize, Align, Builder,
                            EmitOptions, Name);
  return Emitter.emitLoad(RetPtr);
}

Error llvm::emitAtomicStoreBuiltin(
    Value *AtomicPtr, Value *ValPtr, bool IsVolatile,
    std::variant<Value *, AtomicOrdering, AtomicOrderingCABI> Memorder,
    SyncScope::ID Scope, Type *DataTy, std::optional<uint64_t> DataSize,
    std::optional<uint64_t> AvailableSize, MaybeAlign Align,
    IRBuilderBase &Builder, AtomicEmitOptions EmitOptions, llvm::Twine Name) {
  AtomicStoreEmitter Emitter(AtomicPtr, false, IsVolatile, Memorder, {}, Scope,
                             DataTy, DataSize, AvailableSize, Align, Builder,
                             EmitOptions, Name);
  return Emitter.emitStore(ValPtr);
}

Expected<Value *> llvm::emitAtomicCompareExchangeBuiltin(
    Value *AtomicPtr, Value *ExpectedPtr, Value *DesiredPtr,
    std::variant<Value *, bool> IsWeak, bool IsVolatile,
    std::variant<Value *, AtomicOrdering, AtomicOrderingCABI> SuccessMemorder,
    std::variant<std::monostate, Value *, AtomicOrdering, AtomicOrderingCABI>
        FailureMemorder,
    SyncScope::ID Scope, Value *PrevPtr, Type *DataTy,
    std::optional<uint64_t> DataSize, std::optional<uint64_t> AvailableSize,
    MaybeAlign Align, IRBuilderBase &Builder, AtomicEmitOptions EmitOptions,
    llvm::Twine Name) {
  AtomicCompareExchangeEmitter Emitter(
      AtomicPtr, IsWeak, IsVolatile, SuccessMemorder, FailureMemorder, Scope,
      DataTy, DataSize, AvailableSize, Align, Builder, EmitOptions, Name);
  return Emitter.emitCmpXchg(ExpectedPtr, DesiredPtr, PrevPtr);
}
