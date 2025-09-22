//===- PreISelIntrinsicLowering.cpp - Pre-ISel intrinsic lowering pass ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements IR lowering for the llvm.memcpy, llvm.memmove,
// llvm.memset, llvm.load.relative and llvm.objc.* intrinsics.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/PreISelIntrinsicLowering.h"
#include "llvm/Analysis/ObjCARCInstKind.h"
#include "llvm/Analysis/ObjCARCUtil.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/ExpandVectorPredication.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/RuntimeLibcalls.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Scalar/LowerConstantIntrinsics.h"
#include "llvm/Transforms/Utils/BuildLibCalls.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"
#include "llvm/Transforms/Utils/LowerVectorIntrinsics.h"

using namespace llvm;

/// Threshold to leave statically sized memory intrinsic calls. Calls of known
/// size larger than this will be expanded by the pass. Calls of unknown or
/// lower size will be left for expansion in codegen.
static cl::opt<int64_t> MemIntrinsicExpandSizeThresholdOpt(
    "mem-intrinsic-expand-size",
    cl::desc("Set minimum mem intrinsic size to expand in IR"), cl::init(-1),
    cl::Hidden);

namespace {

struct PreISelIntrinsicLowering {
  const TargetMachine *TM;
  const function_ref<TargetTransformInfo &(Function &)> LookupTTI;
  const function_ref<TargetLibraryInfo &(Function &)> LookupTLI;

  /// If this is true, assume it's preferably to leave memory intrinsic calls
  /// for replacement with a library call later. Otherwise this depends on
  /// TargetLoweringInfo availability of the corresponding function.
  const bool UseMemIntrinsicLibFunc;

  explicit PreISelIntrinsicLowering(
      const TargetMachine *TM_,
      function_ref<TargetTransformInfo &(Function &)> LookupTTI_,
      function_ref<TargetLibraryInfo &(Function &)> LookupTLI_,
      bool UseMemIntrinsicLibFunc_ = true)
      : TM(TM_), LookupTTI(LookupTTI_), LookupTLI(LookupTLI_),
        UseMemIntrinsicLibFunc(UseMemIntrinsicLibFunc_) {}

  static bool shouldExpandMemIntrinsicWithSize(Value *Size,
                                               const TargetTransformInfo &TTI);
  bool
  expandMemIntrinsicUses(Function &F,
                         DenseMap<Constant *, GlobalVariable *> &CMap) const;
  bool lowerIntrinsics(Module &M) const;
};

} // namespace

template <class T> static bool forEachCall(Function &Intrin, T Callback) {
  // Lowering all intrinsics in a function will delete multiple uses, so we
  // can't use an early-inc-range. In case some remain, we don't want to look
  // at them again. Unfortunately, Value::UseList is private, so we can't use a
  // simple Use**. If LastUse is null, the next use to consider is
  // Intrin.use_begin(), otherwise it's LastUse->getNext().
  Use *LastUse = nullptr;
  bool Changed = false;
  while (!Intrin.use_empty() && (!LastUse || LastUse->getNext())) {
    Use *U = LastUse ? LastUse->getNext() : &*Intrin.use_begin();
    bool Removed = false;
    // An intrinsic cannot have its address taken, so it cannot be an argument
    // operand. It might be used as operand in debug metadata, though.
    if (auto CI = dyn_cast<CallInst>(U->getUser()))
      Changed |= Removed = Callback(CI);
    if (!Removed)
      LastUse = U;
  }
  return Changed;
}

static bool lowerLoadRelative(Function &F) {
  if (F.use_empty())
    return false;

  bool Changed = false;
  Type *Int32Ty = Type::getInt32Ty(F.getContext());

  for (Use &U : llvm::make_early_inc_range(F.uses())) {
    auto CI = dyn_cast<CallInst>(U.getUser());
    if (!CI || CI->getCalledOperand() != &F)
      continue;

    IRBuilder<> B(CI);
    Value *OffsetPtr =
        B.CreatePtrAdd(CI->getArgOperand(0), CI->getArgOperand(1));
    Value *OffsetI32 = B.CreateAlignedLoad(Int32Ty, OffsetPtr, Align(4));

    Value *ResultPtr = B.CreatePtrAdd(CI->getArgOperand(0), OffsetI32);

    CI->replaceAllUsesWith(ResultPtr);
    CI->eraseFromParent();
    Changed = true;
  }

  return Changed;
}

// ObjCARC has knowledge about whether an obj-c runtime function needs to be
// always tail-called or never tail-called.
static CallInst::TailCallKind getOverridingTailCallKind(const Function &F) {
  objcarc::ARCInstKind Kind = objcarc::GetFunctionClass(&F);
  if (objcarc::IsAlwaysTail(Kind))
    return CallInst::TCK_Tail;
  else if (objcarc::IsNeverTail(Kind))
    return CallInst::TCK_NoTail;
  return CallInst::TCK_None;
}

static bool lowerObjCCall(Function &F, RTLIB::LibcallImpl NewFn,
                          bool setNonLazyBind = false) {
  assert(IntrinsicInst::mayLowerToFunctionCall(F.getIntrinsicID()) &&
         "Pre-ISel intrinsics do lower into regular function calls");
  if (F.use_empty())
    return false;

  // FIXME: When RuntimeLibcalls is an analysis, check if the function is really
  // supported, and go through RTLIB::Libcall.
  StringRef NewFnName = RTLIB::RuntimeLibcallsInfo::getLibcallImplName(NewFn);

  // If we haven't already looked up this function, check to see if the
  // program already contains a function with this name.
  Module *M = F.getParent();
  FunctionCallee FCache =
      M->getOrInsertFunction(NewFnName, F.getFunctionType());

  if (Function *Fn = dyn_cast<Function>(FCache.getCallee())) {
    Fn->setLinkage(F.getLinkage());
    if (setNonLazyBind && !Fn->isWeakForLinker()) {
      // If we have Native ARC, set nonlazybind attribute for these APIs for
      // performance.
      Fn->addFnAttr(Attribute::NonLazyBind);
    }
  }

  CallInst::TailCallKind OverridingTCK = getOverridingTailCallKind(F);

  for (Use &U : llvm::make_early_inc_range(F.uses())) {
    auto *CB = cast<CallBase>(U.getUser());

    if (CB->getCalledFunction() != &F) {
      assert(objcarc::getAttachedARCFunction(CB) == &F &&
             "use expected to be the argument of operand bundle "
             "\"clang.arc.attachedcall\"");
      U.set(FCache.getCallee());
      continue;
    }

    auto *CI = cast<CallInst>(CB);
    assert(CI->getCalledFunction() && "Cannot lower an indirect call!");

    IRBuilder<> Builder(CI->getParent(), CI->getIterator());
    SmallVector<Value *, 8> Args(CI->args());
    SmallVector<llvm::OperandBundleDef, 1> BundleList;
    CI->getOperandBundlesAsDefs(BundleList);
    CallInst *NewCI = Builder.CreateCall(FCache, Args, BundleList);
    NewCI->setName(CI->getName());

    // Try to set the most appropriate TailCallKind based on both the current
    // attributes and the ones that we could get from ObjCARC's special
    // knowledge of the runtime functions.
    //
    // std::max respects both requirements of notail and tail here:
    // * notail on either the call or from ObjCARC becomes notail
    // * tail on either side is stronger than none, but not notail
    CallInst::TailCallKind TCK = CI->getTailCallKind();
    NewCI->setTailCallKind(std::max(TCK, OverridingTCK));

    // Transfer the 'returned' attribute from the intrinsic to the call site.
    // By applying this only to intrinsic call sites, we avoid applying it to
    // non-ARC explicit calls to things like objc_retain which have not been
    // auto-upgraded to use the intrinsics.
    unsigned Index;
    if (F.getAttributes().hasAttrSomewhere(Attribute::Returned, &Index) &&
        Index)
      NewCI->addParamAttr(Index - AttributeList::FirstArgIndex,
                          Attribute::Returned);

    if (!CI->use_empty())
      CI->replaceAllUsesWith(NewCI);
    CI->eraseFromParent();
  }

  return true;
}

// TODO: Should refine based on estimated number of accesses (e.g. does it
// require splitting based on alignment)
bool PreISelIntrinsicLowering::shouldExpandMemIntrinsicWithSize(
    Value *Size, const TargetTransformInfo &TTI) {
  ConstantInt *CI = dyn_cast<ConstantInt>(Size);
  if (!CI)
    return true;
  uint64_t Threshold = MemIntrinsicExpandSizeThresholdOpt.getNumOccurrences()
                           ? MemIntrinsicExpandSizeThresholdOpt
                           : TTI.getMaxMemIntrinsicInlineSizeThreshold();
  uint64_t SizeVal = CI->getZExtValue();

  // Treat a threshold of 0 as a special case to force expansion of all
  // intrinsics, including size 0.
  return SizeVal > Threshold || Threshold == 0;
}

static bool canEmitLibcall(const TargetMachine *TM, Function *F,
                           RTLIB::Libcall LC) {
  // TODO: Should this consider the address space of the memcpy?
  if (!TM)
    return true;
  const TargetLowering *TLI = TM->getSubtargetImpl(*F)->getTargetLowering();
  return TLI->getLibcallName(LC) != nullptr;
}

static bool canEmitMemcpy(const TargetMachine *TM, Function *F) {
  // TODO: Should this consider the address space of the memcpy?
  if (!TM)
    return true;
  const TargetLowering *TLI = TM->getSubtargetImpl(*F)->getTargetLowering();
  return TLI->getMemcpyName() != nullptr;
}

// Return a value appropriate for use with the memset_pattern16 libcall, if
// possible and if we know how. (Adapted from equivalent helper in
// LoopIdiomRecognize).
static Constant *getMemSetPattern16Value(MemSetPatternInst *Inst,
                                         const TargetLibraryInfo &TLI) {
  // TODO: This could check for UndefValue because it can be merged into any
  // other valid pattern.

  // Don't emit libcalls if a non-default address space is being used.
  if (Inst->getRawDest()->getType()->getPointerAddressSpace() != 0)
    return nullptr;

  Value *V = Inst->getValue();
  Type *VTy = V->getType();
  const DataLayout &DL = Inst->getDataLayout();
  Module *M = Inst->getModule();

  if (!isLibFuncEmittable(M, &TLI, LibFunc_memset_pattern16))
    return nullptr;

  // If the value isn't a constant, we can't promote it to being in a constant
  // array.  We could theoretically do a store to an alloca or something, but
  // that doesn't seem worthwhile.
  Constant *C = dyn_cast<Constant>(V);
  if (!C || isa<ConstantExpr>(C))
    return nullptr;

  // Only handle simple values that are a power of two bytes in size.
  uint64_t Size = DL.getTypeSizeInBits(VTy);
  if (!DL.typeSizeEqualsStoreSize(VTy) || !isPowerOf2_64(Size))
    return nullptr;

  // Don't care enough about darwin/ppc to implement this.
  if (DL.isBigEndian())
    return nullptr;

  // Convert to size in bytes.
  Size /= 8;

  // TODO: If CI is larger than 16-bytes, we can try slicing it in half to see
  // if the top and bottom are the same (e.g. for vectors and large integers).
  if (Size > 16)
    return nullptr;

  // If the constant is exactly 16 bytes, just use it.
  if (Size == 16)
    return C;

  // Otherwise, we'll use an array of the constants.
  uint64_t ArraySize = 16 / Size;
  ArrayType *AT = ArrayType::get(V->getType(), ArraySize);
  return ConstantArray::get(AT, std::vector<Constant *>(ArraySize, C));
}

// TODO: Handle atomic memcpy and memcpy.inline
// TODO: Pass ScalarEvolution
bool PreISelIntrinsicLowering::expandMemIntrinsicUses(
    Function &F, DenseMap<Constant *, GlobalVariable *> &CMap) const {
  Intrinsic::ID ID = F.getIntrinsicID();
  bool Changed = false;

  for (User *U : llvm::make_early_inc_range(F.users())) {
    Instruction *Inst = cast<Instruction>(U);

    switch (ID) {
    case Intrinsic::memcpy: {
      auto *Memcpy = cast<MemCpyInst>(Inst);
      Function *ParentFunc = Memcpy->getFunction();
      const TargetTransformInfo &TTI = LookupTTI(*ParentFunc);
      if (shouldExpandMemIntrinsicWithSize(Memcpy->getLength(), TTI)) {
        if (UseMemIntrinsicLibFunc && canEmitMemcpy(TM, ParentFunc))
          break;

        // TODO: For optsize, emit the loop into a separate function
        expandMemCpyAsLoop(Memcpy, TTI);
        Changed = true;
        Memcpy->eraseFromParent();
      }

      break;
    }
    case Intrinsic::memcpy_inline: {
      // Only expand llvm.memcpy.inline with non-constant length in this
      // codepath, leaving the current SelectionDAG expansion for constant
      // length memcpy intrinsics undisturbed.
      auto *Memcpy = cast<MemCpyInst>(Inst);
      if (isa<ConstantInt>(Memcpy->getLength()))
        break;

      Function *ParentFunc = Memcpy->getFunction();
      const TargetTransformInfo &TTI = LookupTTI(*ParentFunc);
      expandMemCpyAsLoop(Memcpy, TTI);
      Changed = true;
      Memcpy->eraseFromParent();
      break;
    }
    case Intrinsic::memmove: {
      auto *Memmove = cast<MemMoveInst>(Inst);
      Function *ParentFunc = Memmove->getFunction();
      const TargetTransformInfo &TTI = LookupTTI(*ParentFunc);
      if (shouldExpandMemIntrinsicWithSize(Memmove->getLength(), TTI)) {
        if (UseMemIntrinsicLibFunc &&
            canEmitLibcall(TM, ParentFunc, RTLIB::MEMMOVE))
          break;

        if (expandMemMoveAsLoop(Memmove, TTI)) {
          Changed = true;
          Memmove->eraseFromParent();
        }
      }

      break;
    }
    case Intrinsic::memset: {
      auto *Memset = cast<MemSetInst>(Inst);
      Function *ParentFunc = Memset->getFunction();
      const TargetTransformInfo &TTI = LookupTTI(*ParentFunc);
      if (shouldExpandMemIntrinsicWithSize(Memset->getLength(), TTI)) {
        if (UseMemIntrinsicLibFunc &&
            canEmitLibcall(TM, ParentFunc, RTLIB::MEMSET))
          break;

        expandMemSetAsLoop(Memset);
        Changed = true;
        Memset->eraseFromParent();
      }

      break;
    }
    case Intrinsic::memset_inline: {
      // Only expand llvm.memset.inline with non-constant length in this
      // codepath, leaving the current SelectionDAG expansion for constant
      // length memset intrinsics undisturbed.
      auto *Memset = cast<MemSetInst>(Inst);
      if (isa<ConstantInt>(Memset->getLength()))
        break;

      expandMemSetAsLoop(Memset);
      Changed = true;
      Memset->eraseFromParent();
      break;
    }
    case Intrinsic::experimental_memset_pattern: {
      auto *Memset = cast<MemSetPatternInst>(Inst);
      const TargetLibraryInfo &TLI = LookupTLI(*Memset->getFunction());
      Constant *PatternValue = getMemSetPattern16Value(Memset, TLI);
      if (!PatternValue) {
        // If it isn't possible to emit a memset_pattern16 libcall, expand to
        // a loop instead.
        expandMemSetPatternAsLoop(Memset);
        Changed = true;
        Memset->eraseFromParent();
        break;
      }
      // FIXME: There is currently no profitability calculation for emitting
      // the libcall vs expanding the memset.pattern directly.
      IRBuilder<> Builder(Inst);
      Module *M = Memset->getModule();
      const DataLayout &DL = Memset->getDataLayout();

      Type *DestPtrTy = Memset->getRawDest()->getType();
      Type *SizeTTy = TLI.getSizeTType(*M);
      StringRef FuncName = "memset_pattern16";
      FunctionCallee MSP = getOrInsertLibFunc(M, TLI, LibFunc_memset_pattern16,
                                              Builder.getVoidTy(), DestPtrTy,
                                              Builder.getPtrTy(), SizeTTy);
      inferNonMandatoryLibFuncAttrs(M, FuncName, TLI);

      // Otherwise we should form a memset_pattern16.  PatternValue is known
      // to be an constant array of 16-bytes. Put the value into a mergable
      // global.
      assert(Memset->getRawDest()->getType()->getPointerAddressSpace() == 0 &&
             "Should have skipped if non-zero AS");
      GlobalVariable *GV;
      auto It = CMap.find(PatternValue);
      if (It != CMap.end()) {
        GV = It->second;
      } else {
        GV = new GlobalVariable(
            *M, PatternValue->getType(), /*isConstant=*/true,
            GlobalValue::PrivateLinkage, PatternValue, ".memset_pattern");
        GV->setUnnamedAddr(
            GlobalValue::UnnamedAddr::Global); // Ok to merge these.
        // TODO: Consider relaxing alignment requirement.
        GV->setAlignment(Align(16));
        CMap[PatternValue] = GV;
      }
      Value *PatternPtr = GV;
      Value *NumBytes = Builder.CreateMul(
          TLI.getAsSizeT(DL.getTypeAllocSize(Memset->getValue()->getType()),
                         *M),
          Builder.CreateZExtOrTrunc(Memset->getLength(), SizeTTy));
      CallInst *MemsetPattern16Call =
          Builder.CreateCall(MSP, {Memset->getRawDest(), PatternPtr, NumBytes});
      MemsetPattern16Call->setAAMetadata(Memset->getAAMetadata());
      // Preserve any call site attributes on the destination pointer
      // argument (e.g. alignment).
      AttrBuilder ArgAttrs(Memset->getContext(),
                           Memset->getAttributes().getParamAttrs(0));
      MemsetPattern16Call->setAttributes(
          MemsetPattern16Call->getAttributes().addParamAttributes(
              Memset->getContext(), 0, ArgAttrs));
      Changed = true;
      Memset->eraseFromParent();
      break;
    }
    default:
      llvm_unreachable("unhandled intrinsic");
    }
  }

  return Changed;
}

bool PreISelIntrinsicLowering::lowerIntrinsics(Module &M) const {
  // Map unique constants to globals.
  DenseMap<Constant *, GlobalVariable *> CMap;
  bool Changed = false;
  for (Function &F : M) {
    switch (F.getIntrinsicID()) {
    default:
      break;
    case Intrinsic::memcpy:
    case Intrinsic::memcpy_inline:
    case Intrinsic::memmove:
    case Intrinsic::memset:
    case Intrinsic::memset_inline:
    case Intrinsic::experimental_memset_pattern:
      Changed |= expandMemIntrinsicUses(F, CMap);
      break;
    case Intrinsic::load_relative:
      Changed |= lowerLoadRelative(F);
      break;
    case Intrinsic::is_constant:
    case Intrinsic::objectsize:
      Changed |= forEachCall(F, [&](CallInst *CI) {
        Function *Parent = CI->getParent()->getParent();
        TargetLibraryInfo &TLI = LookupTLI(*Parent);
        // Intrinsics in unreachable code are not lowered.
        bool Changed = lowerConstantIntrinsics(*Parent, TLI, /*DT=*/nullptr);
        return Changed;
      });
      break;
#define BEGIN_REGISTER_VP_INTRINSIC(VPID, MASKPOS, VLENPOS)                    \
  case Intrinsic::VPID:
#include "llvm/IR/VPIntrinsics.def"
      forEachCall(F, [&](CallInst *CI) {
        Function *Parent = CI->getParent()->getParent();
        const TargetTransformInfo &TTI = LookupTTI(*Parent);
        auto *VPI = cast<VPIntrinsic>(CI);
        VPExpansionDetails ED = expandVectorPredicationIntrinsic(*VPI, TTI);
        // Expansion of VP intrinsics may change the IR but not actually
        // replace the intrinsic, so update Changed for the pass
        // and compute Removed for forEachCall.
        Changed |= ED != VPExpansionDetails::IntrinsicUnchanged;
        bool Removed = ED == VPExpansionDetails::IntrinsicReplaced;
        return Removed;
      });
      break;
    case Intrinsic::objc_autorelease:
      Changed |= lowerObjCCall(F, RTLIB::impl_objc_autorelease);
      break;
    case Intrinsic::objc_autoreleasePoolPop:
      Changed |= lowerObjCCall(F, RTLIB::impl_objc_autoreleasePoolPop);
      break;
    case Intrinsic::objc_autoreleasePoolPush:
      Changed |= lowerObjCCall(F, RTLIB::impl_objc_autoreleasePoolPush);
      break;
    case Intrinsic::objc_autoreleaseReturnValue:
      Changed |= lowerObjCCall(F, RTLIB::impl_objc_autoreleaseReturnValue);
      break;
    case Intrinsic::objc_copyWeak:
      Changed |= lowerObjCCall(F, RTLIB::impl_objc_copyWeak);
      break;
    case Intrinsic::objc_destroyWeak:
      Changed |= lowerObjCCall(F, RTLIB::impl_objc_destroyWeak);
      break;
    case Intrinsic::objc_initWeak:
      Changed |= lowerObjCCall(F, RTLIB::impl_objc_initWeak);
      break;
    case Intrinsic::objc_loadWeak:
      Changed |= lowerObjCCall(F, RTLIB::impl_objc_loadWeak);
      break;
    case Intrinsic::objc_loadWeakRetained:
      Changed |= lowerObjCCall(F, RTLIB::impl_objc_loadWeakRetained);
      break;
    case Intrinsic::objc_moveWeak:
      Changed |= lowerObjCCall(F, RTLIB::impl_objc_moveWeak);
      break;
    case Intrinsic::objc_release:
      Changed |= lowerObjCCall(F, RTLIB::impl_objc_release, true);
      break;
    case Intrinsic::objc_retain:
      Changed |= lowerObjCCall(F, RTLIB::impl_objc_retain, true);
      break;
    case Intrinsic::objc_retainAutorelease:
      Changed |= lowerObjCCall(F, RTLIB::impl_objc_retainAutorelease);
      break;
    case Intrinsic::objc_retainAutoreleaseReturnValue:
      Changed |=
          lowerObjCCall(F, RTLIB::impl_objc_retainAutoreleaseReturnValue);
      break;
    case Intrinsic::objc_retainAutoreleasedReturnValue:
      Changed |=
          lowerObjCCall(F, RTLIB::impl_objc_retainAutoreleasedReturnValue);
      break;
    case Intrinsic::objc_claimAutoreleasedReturnValue:
      Changed |=
          lowerObjCCall(F, RTLIB::impl_objc_claimAutoreleasedReturnValue);
      break;
    case Intrinsic::objc_retainBlock:
      Changed |= lowerObjCCall(F, RTLIB::impl_objc_retainBlock);
      break;
    case Intrinsic::objc_storeStrong:
      Changed |= lowerObjCCall(F, RTLIB::impl_objc_storeStrong);
      break;
    case Intrinsic::objc_storeWeak:
      Changed |= lowerObjCCall(F, RTLIB::impl_objc_storeWeak);
      break;
    case Intrinsic::objc_unsafeClaimAutoreleasedReturnValue:
      Changed |=
          lowerObjCCall(F, RTLIB::impl_objc_unsafeClaimAutoreleasedReturnValue);
      break;
    case Intrinsic::objc_retainedObject:
      Changed |= lowerObjCCall(F, RTLIB::impl_objc_retainedObject);
      break;
    case Intrinsic::objc_unretainedObject:
      Changed |= lowerObjCCall(F, RTLIB::impl_objc_unretainedObject);
      break;
    case Intrinsic::objc_unretainedPointer:
      Changed |= lowerObjCCall(F, RTLIB::impl_objc_unretainedPointer);
      break;
    case Intrinsic::objc_retain_autorelease:
      Changed |= lowerObjCCall(F, RTLIB::impl_objc_retain_autorelease);
      break;
    case Intrinsic::objc_sync_enter:
      Changed |= lowerObjCCall(F, RTLIB::impl_objc_sync_enter);
      break;
    case Intrinsic::objc_sync_exit:
      Changed |= lowerObjCCall(F, RTLIB::impl_objc_sync_exit);
      break;
    case Intrinsic::exp:
    case Intrinsic::exp2:
    case Intrinsic::log:
      Changed |= forEachCall(F, [&](CallInst *CI) {
        Type *Ty = CI->getArgOperand(0)->getType();
        if (!isa<ScalableVectorType>(Ty))
          return false;
        const TargetLowering *TL = TM->getSubtargetImpl(F)->getTargetLowering();
        unsigned Op = TL->IntrinsicIDToISD(F.getIntrinsicID());
        assert(Op != ISD::DELETED_NODE && "unsupported intrinsic");
        if (!TL->isOperationExpand(Op, EVT::getEVT(Ty)))
          return false;
        return lowerUnaryVectorIntrinsicAsLoop(M, CI);
      });
      break;
    }
  }
  return Changed;
}

namespace {

class PreISelIntrinsicLoweringLegacyPass : public ModulePass {
public:
  static char ID;

  PreISelIntrinsicLoweringLegacyPass() : ModulePass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<TargetPassConfig>();
  }

  bool runOnModule(Module &M) override {
    auto LookupTTI = [this](Function &F) -> TargetTransformInfo & {
      return this->getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    };
    auto LookupTLI = [this](Function &F) -> TargetLibraryInfo & {
      return this->getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
    };

    const auto *TM = &getAnalysis<TargetPassConfig>().getTM<TargetMachine>();
    PreISelIntrinsicLowering Lowering(TM, LookupTTI, LookupTLI);
    return Lowering.lowerIntrinsics(M);
  }
};

} // end anonymous namespace

char PreISelIntrinsicLoweringLegacyPass::ID;

INITIALIZE_PASS_BEGIN(PreISelIntrinsicLoweringLegacyPass,
                      "pre-isel-intrinsic-lowering",
                      "Pre-ISel Intrinsic Lowering", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(PreISelIntrinsicLoweringLegacyPass,
                    "pre-isel-intrinsic-lowering",
                    "Pre-ISel Intrinsic Lowering", false, false)

ModulePass *llvm::createPreISelIntrinsicLoweringPass() {
  return new PreISelIntrinsicLoweringLegacyPass();
}

PreservedAnalyses PreISelIntrinsicLoweringPass::run(Module &M,
                                                    ModuleAnalysisManager &AM) {
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  auto LookupTTI = [&FAM](Function &F) -> TargetTransformInfo & {
    return FAM.getResult<TargetIRAnalysis>(F);
  };
  auto LookupTLI = [&FAM](Function &F) -> TargetLibraryInfo & {
    return FAM.getResult<TargetLibraryAnalysis>(F);
  };

  PreISelIntrinsicLowering Lowering(TM, LookupTTI, LookupTLI);
  if (!Lowering.lowerIntrinsics(M))
    return PreservedAnalyses::all();
  else
    return PreservedAnalyses::none();
}
