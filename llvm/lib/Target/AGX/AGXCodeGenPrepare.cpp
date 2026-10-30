//===-- AGXCodeGenPrepare.cpp - AGX IR preparation ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// IR transformations that AGX code generation relies on:
//
//  * AGXAlwaysInline normally marks every non-kernel function always_inline
//    (and internal, so it disappears once inlined). The opt-in call path keeps
//    eligible functions out of this expansion and removes unused functions.
//    The functions it keeps that use the global data buffer get its address
//    as a parameter (AGXGlobalDataArgs, after inlining): only a kernel has it,
//    in a uniform register.
//  * AGXCodeGenPrepare expands what the hardware has no instruction or
//    library for: integer division, and memory intrinsics that are not
//    small enough to be expanded by instruction selection. There is no
//    floating point environment, so constrained floating point intrinsics
//    become their ordinary operations. It also unifies the function exits,
//    which the control flow structurizer expects.
//  * On subtargets where simd_shuffle merges the low index bits of each quad
//    (quad-shuffle-index-merge), shuffles with divergent indices are
//    expanded into four shuffles whose low index bits are uniform, like
//    Mesa's agx_nir_lower_subgroups.
//  * The hardware flushes 32-bit denormals but keeps 16-bit ones. Functions
//    whose denormal mode for half asks for flushing get explicit flushes of
//    the operands and results of their half operations.
//  * There is no double precision hardware: the double precision arithmetic
//    that AGXSoftF64 does not implement is diagnosed as unsupported.
//  * Tangent, the inverse and the hyperbolic functions are built from the
//    functions the hardware has (AGXMathExpansion.cpp).
//  * A threadgroup of at most 4 SIMD-groups is one aligned block, whose
//    barrier is cheaper: on subtargets with block-barrier its threadgroup
//    barriers become block barriers.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "AGXMachineFunctionInfo.h"
#include "AGXSubtarget.h"
#include "llvm/ADT/GenericUniformityImpl.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/CycleAnalysis.h"
#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/BypassSlowDivision.h"
#include "llvm/Transforms/Utils/CallPromotionUtils.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAGX.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include <cmath>
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/IntegerDivision.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/ADT/SetVector.h"

using namespace llvm;

#define DEBUG_TYPE "agx-codegenprepare"

//===----------------------------------------------------------------------===//
// AGXAlwaysInline
//===----------------------------------------------------------------------===//

namespace {

class AGXAlwaysInline : public ModulePass {
public:
  static char ID;
  AGXAlwaysInline() : ModulePass(ID) {}

  StringRef getPassName() const override { return "AGX Always Inline"; }
  bool runOnModule(Module &M) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};

} // namespace

char AGXAlwaysInline::ID = 0;

INITIALIZE_PASS(AGXAlwaysInline, "agx-always-inline", "AGX Always Inline",
                false, false)

ModulePass *llvm::createAGXAlwaysInlinePass() { return new AGXAlwaysInline(); }

static bool allocateExperimentalCallType(Type *Ty, unsigned &Slots) {
  if (Ty->isVoidTy())
    return true;
  if (auto *Array = dyn_cast<ArrayType>(Ty)) {
    for (uint64_t I = 0; I < Array->getNumElements(); ++I)
      if (!allocateExperimentalCallType(Array->getElementType(), Slots))
        return false;
    return true;
  }
  if (Ty->isPointerTy() || Ty->isIntegerTy(64)) {
    Slots = alignTo(Slots, 2) + 2;
  } else if (Ty->isIntegerTy(16) || Ty->isIntegerTy(32) ||
             Ty->isHalfTy() || Ty->isFloatTy()) {
    ++Slots;
  } else {
    return false;
  }
  return Slots <= 28;
}

static bool canKeepExperimentalCall(const Function &F) {
  if (F.isDeclaration() || F.hasFnAttribute(Attribute::AlwaysInline) ||
      F.isVarArg() || F.hasStructRetAttr())
    return false;
  // (two slots kept for the global data buffer's address, which
  // AGXGlobalDataArgs may add)
  unsigned ArgSlots = 2;
  for (const Argument &Arg : F.args())
    if (!allocateExperimentalCallType(Arg.getType(), ArgSlots))
      return false;
  unsigned ResultSlots = 0;
  if (!allocateExperimentalCallType(F.getReturnType(), ResultSlots))
    return false;
  unsigned Returns = 0;
  for (const BasicBlock &BB : F)
    for (const Instruction &I : BB) {
      if (isa<CallBrInst>(I))
        return false;
      if (const auto *AI = dyn_cast<AllocaInst>(&I))
        if (!AI->isStaticAlloca())
          return false;
      if (isa<ReturnInst>(I))
        ++Returns;
      if (const auto *CB = dyn_cast<CallBase>(&I)) {
        const Function *Callee = CB->getCalledFunction();
        if (!Callee ||
            (Callee->isIntrinsic() &&
             (Callee->getIntrinsicID() == Intrinsic::agx_texture_sample ||
              Callee->getIntrinsicID() == Intrinsic::agx_threadgroups_per_grid_x ||
              Callee->getIntrinsicID() == Intrinsic::agx_threadgroups_per_grid_y ||
              Callee->getIntrinsicID() == Intrinsic::agx_threadgroups_per_grid_z)))
          return false;
      }
    }
  return Returns == 1;
}

/// The default backend inlines all calls. An indirect call becomes a test of
/// the callee against each address-taken function of its type, with a direct
/// call (inlined later) for each, and a trap if none
/// matches. The addresses of these functions, as values, become their
/// numbers (1, 2, ...), so that they can be stored, compared and passed.
static bool lowerIndirectCalls(Module &M) {
  SmallVector<CallBase *, 8> Indirect;
  for (Function &F : M)
    for (Instruction &I : instructions(F))
      if (auto *CB = dyn_cast<CallBase>(&I); CB && CB->isIndirectCall() &&
                                             !CB->isInlineAsm())
        Indirect.push_back(CB);
  SmallVector<Function *, 8> Taken;
  for (Function &F : M)
    if (!F.isIntrinsic() && F.getCallingConv() != CallingConv::AGX_KERNEL &&
        F.hasAddressTaken(nullptr, /*IgnoreCallbackUses=*/false,
                          /*IgnoreAssumeLikeCalls=*/true,
                          /*IngoreLLVMUsed=*/true))
      Taken.push_back(&F);
  if (Indirect.empty() && Taken.empty())
    return false;
  for (CallBase *CB : Indirect) {
    CallBase *Rest = CB;
    for (Function *F : Taken) {
      if (F->getFunctionType() != CB->getFunctionType())
        continue;
      // The direct call is in the then block; the indirect one stays in the
      // else block.
      promoteCallWithIfThenElse(*Rest, F);
    }
    // No address-taken function of this type matches.
    IRBuilder<> B(Rest);
    B.CreateIntrinsic(Intrinsic::trap, {});
    if (!Rest->getType()->isVoidTy())
      Rest->replaceAllUsesWith(PoisonValue::get(Rest->getType()));
    Rest->eraseFromParent();
  }
  for (auto [N, F] : enumerate(Taken)) {
    Constant *Id = ConstantExpr::getIntToPtr(
        ConstantInt::get(Type::getInt64Ty(M.getContext()), N + 1), F->getType());
    F->replaceUsesWithIf(Id, [](Use &U) {
      auto *CB = dyn_cast<CallBase>(U.getUser());
      return !CB || !CB->isCallee(&U);
    });
  }
  return true;
}

bool AGXAlwaysInline::runOnModule(Module &M) {
  bool Changed = lowerIndirectCalls(M);
  Changed |= lowerAGXRecursion(M);
  // Modules without any agx_kernel function (for example tests of single
  // functions) compile every external function as an entry point.
  bool HasKernels = any_of(M, [](const Function &F) {
    return F.getCallingConv() == CallingConv::AGX_KERNEL;
  });

  // Compute a conservative static frame depth before inlining wrappers.
  // A shared callee uses its deepest path, leaving every active ancestor's
  // scratch and saved return address in a different region/register.
  DenseMap<const Function *, unsigned> Depth;
  bool CallsWithinLimit = true;
  if (agxExperimentalLeafCallsEnabled()) {
    for (Function &F : M)
      if (!F.isDeclaration() &&
          (F.getCallingConv() == CallingConv::AGX_KERNEL ||
           (!HasKernels && !F.hasLocalLinkage())))
        Depth[&F] = 0;
    for (unsigned Iter = 0; Iter != 8; ++Iter) {
      bool Updated = false;
      for (Function &F : M) {
        auto It = Depth.find(&F);
        if (It == Depth.end())
          continue;
        // (by value: inserting a callee below may move the map's entries)
        unsigned Next = It->second + 1;
        for (const Instruction &I : instructions(F)) {
          const auto *CB = dyn_cast<CallBase>(&I);
          const Function *Callee = CB ? CB->getCalledFunction() : nullptr;
          if (!Callee || Callee->isDeclaration() || Callee->isIntrinsic())
            continue;
          if (Next > Depth.lookup(Callee)) {
            Depth[Callee] = Next;
            Updated = true;
          }
        }
      }
      if (!Updated)
        break;
    }
    unsigned MaxDepth = 0;
    for (auto [F, D] : Depth)
      MaxDepth = std::max(MaxDepth, D);
    CallsWithinLimit = MaxDepth <= 7;
    if (CallsWithinLimit) {
      for (Function &F : M)
        if (auto It = Depth.find(&F); It != Depth.end()) {
          F.addFnAttr("agx-call-depth", Twine(It->second).str());
          Changed = true;
        }
      for (Function &F : M)
        if (F.getCallingConv() == CallingConv::AGX_KERNEL) {
          F.addFnAttr("agx-max-call-depth", Twine(MaxDepth).str());
          Changed = true;
        }
    }
  }

  SmallVector<Function *, 8> Dead;
  for (Function &F : M) {
    if (F.isDeclaration() || F.getCallingConv() == CallingConv::AGX_KERNEL)
      continue;
    if (!HasKernels && !F.hasLocalLinkage())
      continue;
    // With kernels in the module, other functions are only reachable from
    // them: internalize them so that they disappear once inlined, and drop
    // the unused ones (for example from linked device libraries).
    if (HasKernels && !F.hasLocalLinkage()) {
      F.setLinkage(GlobalValue::InternalLinkage);
      F.setVisibility(GlobalValue::DefaultVisibility);
      Changed = true;
    }
    if (F.use_empty()) {
      Dead.push_back(&F);
      continue;
    }
    if (F.hasFnAttribute(Attribute::NoInline))
      continue;
    if (agxExperimentalLeafCallsEnabled() && CallsWithinLimit &&
        canKeepExperimentalCall(F)) {
      F.addFnAttr(Attribute::NoInline);
      Changed = true;
      continue;
    }
    F.addFnAttr(Attribute::AlwaysInline);
    Changed = true;
  }
  for (Function *F : Dead)
    F->eraseFromParent();
  Changed |= !Dead.empty();
  // Called functions after every kernel: a kernel's program runs from its
  // symbol to the end of .text, which must hold what it calls (a module from
  // AIR lists its functions before its kernels).
  if (agxExperimentalLeafCallsEnabled() && HasKernels) {
    SmallVector<Function *, 16> Called;
    for (Function &F : M)
      if (!F.isDeclaration() && F.getCallingConv() != CallingConv::AGX_KERNEL)
        Called.push_back(&F);
    for (Function *F : Called) {
      F->removeFromParent();
      M.getFunctionList().push_back(F);
    }
    Changed |= !Called.empty();
  }
  return Changed;
}

//===----------------------------------------------------------------------===//
// AGXGlobalDataArgs
//===----------------------------------------------------------------------===//

namespace {

/// After inlining, the functions that are still called (the experimental
/// call path) and use the global data buffer, and their callers up to the
/// kernels, get its address as a last i64 parameter: a kernel has it in a
/// uniform register (LowerFormalArguments), a called function has no access
/// to it. In those functions each global data variable becomes that address
/// plus its offset in computeGlobalDataLayout's layout, which instruction
/// selection uses for the kernels' own accesses (no global is added or removed
/// after this pass). A kernel passes the address of the layout's first
/// variable less its offset.
class AGXGlobalDataArgs : public ModulePass {
public:
  static char ID;
  AGXGlobalDataArgs() : ModulePass(ID) {}
  StringRef getPassName() const override { return "AGX Global Data Arguments"; }
  bool runOnModule(Module &M) override;
};

} // namespace

char AGXGlobalDataArgs::ID = 0;

ModulePass *llvm::createAGXGlobalDataArgsPass() {
  return new AGXGlobalDataArgs();
}

bool AGXGlobalDataArgs::runOnModule(Module &M) {
  auto IsKernel = [](const Function &F) {
    return F.getCallingConv() == CallingConv::AGX_KERNEL;
  };
  SetVector<Function *> Need;
  for (Function &F : M)
    if (!F.isDeclaration() && !IsKernel(F) && AGX::usesGlobalData(F))
      Need.insert(&F);
  for (unsigned I = 0; I < Need.size(); ++I)
    for (User *U : Need[I]->users())
      if (auto *CB = dyn_cast<CallBase>(U))
        if (Function *Caller = CB->getFunction(); !IsKernel(*Caller))
          Need.insert(Caller);
  if (Need.empty())
    return false;
  AGX::GlobalDataLayout Layout = AGX::computeGlobalDataLayout(M);
  if (Layout.empty())
    return false;
  LLVMContext &Ctx = M.getContext();
  Type *I64 = Type::getInt64Ty(Ctx);

  // Each function with the parameter added (its body moved over).
  DenseMap<Function *, Function *> New;
  for (Function *F : Need) {
    FunctionType *FT = F->getFunctionType();
    SmallVector<Type *, 8> Params(FT->params());
    Params.push_back(I64);
    Function *NF = Function::Create(
        FunctionType::get(FT->getReturnType(), Params, false), F->getLinkage(),
        F->getAddressSpace(), "", &M);
    NF->copyAttributesFrom(F);
    NF->copyMetadata(F, 0);
    NF->setCallingConv(F->getCallingConv());
    NF->splice(NF->begin(), F);
    for (auto [Old, NewArg] : zip(F->args(), NF->args())) {
      NewArg.takeName(&Old);
      Old.replaceAllUsesWith(&NewArg);
    }
    NF->getArg(FT->getNumParams())->setName("agx.global.data");
    NF->takeName(F);
    New[F] = NF;
  }

  // The global data variables' uses in those functions, as base + offset.
  SmallVector<Constant *, 16> Globals;
  for (auto &[GV, Offset] : Layout.Offsets)
    Globals.push_back(const_cast<GlobalVariable *>(GV));
  for (auto &[F, NF] : New) {
    convertUsersOfConstantsToInstructions(Globals, NF);
    Argument *Base = NF->getArg(NF->arg_size() - 1);
    for (Instruction &I : instructions(*NF))
      for (Use &U : I.operands()) {
        auto *GV = dyn_cast<GlobalVariable>(U.get());
        if (!GV)
          continue;
        std::optional<uint64_t> Offset = Layout.getOffset(GV);
        if (!Offset)
          continue;
        auto *Phi = dyn_cast<PHINode>(&I);
        IRBuilder<> B(Phi ? Phi->getIncomingBlock(U)->getTerminator() : &I);
        U.set(B.CreateIntToPtr(B.CreateAdd(Base, B.getInt64(*Offset)),
                               GV->getType()));
      }
  }

  // The calls, with the caller's address (a kernel's from the layout's
  // first variable).
  // (the calls are in the new functions, whose bodies they are now)
  SmallPtrSet<Function *, 16> Added;
  for (auto &[F, NF] : New)
    Added.insert(NF);
  DenseMap<Function *, Value *> KernelBase;
  auto BaseIn = [&](Function *Caller) -> Value * {
    if (Added.contains(Caller))
      return Caller->getArg(Caller->arg_size() - 1);
    Value *&V = KernelBase[Caller];
    if (!V) {
      auto &[GV, Offset] = Layout.Offsets.front();
      IRBuilder<> B(&*Caller->getEntryBlock().getFirstInsertionPt());
      V = B.CreateSub(
          B.CreatePtrToInt(const_cast<GlobalVariable *>(GV), I64),
          B.getInt64(Offset), "agx.global.data");
    }
    return V;
  };
  for (auto &[F, NF] : New) {
    for (User *U : make_early_inc_range(F->users())) {
      auto *CB = dyn_cast<CallInst>(U);
      if (!CB || CB->getCalledOperand() != F)
        report_fatal_error("AGX: a function using global data is used other "
                           "than called");
      SmallVector<Value *, 8> Args(CB->args());
      Args.push_back(BaseIn(CB->getFunction()));
      auto *NC = CallInst::Create(NF, Args, "", CB->getIterator());
      NC->setCallingConv(CB->getCallingConv());
      NC->setAttributes(CB->getAttributes());
      NC->setTailCallKind(CB->getTailCallKind());
      NC->setDebugLoc(CB->getDebugLoc());
      NC->takeName(CB);
      CB->replaceAllUsesWith(NC);
      CB->eraseFromParent();
    }
    F->eraseFromParent();
  }
  return true;
}

//===----------------------------------------------------------------------===//
// AGXCodeGenPrepare
//===----------------------------------------------------------------------===//

namespace {

class AGXCodeGenPrepare : public FunctionPass {
  bool expandDivisions(Function &F);
  bool expandMemIntrinsics(Function &F, const TargetTransformInfo &TTI);
  bool unifyExits(Function &F);
  bool lowerConstrainedFP(Function &F);
  bool lowerDivergentShuffles(Function &F, const TargetTransformInfo &TTI);
  bool lowerMatrixFMA(Function &F, const AGXSubtarget &ST);
  bool flushHalfDenormals(Function &F);
  bool flushFloatMinMax(Function &F);
  bool diagnoseDoubles(Function &F);
  bool fenceHostCoherentAccesses(Function &F);
  bool useBlockBarriers(Function &F, const AGXSubtarget &ST);
  bool expandMath(Function &F, const AGXSubtarget &ST);
  bool combineUniformAtomics(Function &F, const TargetTransformInfo &TTI);

public:
  static char ID;
  AGXCodeGenPrepare() : FunctionPass(ID) {}

  StringRef getPassName() const override { return "AGX CodeGen Prepare"; }
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<TargetPassConfig>();
  }
};

} // namespace

char AGXCodeGenPrepare::ID = 0;

INITIALIZE_PASS_BEGIN(AGXCodeGenPrepare, DEBUG_TYPE, "AGX CodeGen Prepare",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(AGXCodeGenPrepare, DEBUG_TYPE, "AGX CodeGen Prepare",
                    false, false)

FunctionPass *llvm::createAGXCodeGenPreparePass() {
  return new AGXCodeGenPrepare();
}

/// Division by a power of two is a shift, and 32-bit division by another
/// constant is a multiply-high; both are left to instruction selection.
static bool isCheapDivision(const BinaryOperator &BO) {
  auto *C = dyn_cast<ConstantInt>(BO.getOperand(1));
  if (!C)
    return false;
  if (C->getValue().isPowerOf2())
    return true;
  return BO.getType()->isIntegerTy(32);
}

static cl::opt<bool> ShareDivisors(
    "agx-share-divisors", cl::init(true), cl::Hidden,
    cl::desc("Compute the reciprocal of a 32-bit divisor once where the "
             "divisor is defined, shared by every division by it"));

namespace {
/// The divisor-only part of a 32-bit division: |y|, whether y < 0, and the
/// estimate z of 2^32 / |y|.
struct Divisor {
  Value *Y, *Neg, *Z;
};

/// Divisors computed once each. A divisor that is a cheap pure expression
/// (ZLUDA's umax(y, 1), a trunc of a 64-bit value) is rebuilt right after
/// the latest of its leaves, so that copies of it in different blocks (the
/// unrolled iterations of a loop, the fast paths of 64-bit divisions) share
/// one reciprocal.
class DivisorCache {
  Function &F;
  DominatorTree DT;
  DenseMap<Value *, Value *> Hoisted;
  std::map<SmallVector<uintptr_t, 6>, Value *> Exprs;
  DenseMap<std::pair<Value *, unsigned>, Divisor> Divisors;

  static bool isCheapPure(const Instruction &I) {
    if (isa<CastInst, BinaryOperator, ICmpInst, SelectInst>(I))
      return !isa<BinaryOperator>(I) || !I.isIntDivRem();
    if (auto *II = dyn_cast<IntrinsicInst>(&I))
      switch (II->getIntrinsicID()) {
      case Intrinsic::umax:
      case Intrinsic::umin:
      case Intrinsic::smax:
      case Intrinsic::smin:
      case Intrinsic::abs:
        return true;
      default:
        break;
      }
    return false;
  }

  /// Where a value defined by V can first be used.
  BasicBlock::iterator after(Value *V) {
    if (auto *I = dyn_cast_or_null<Instruction>(V)) {
      if (isa<PHINode>(I))
        return I->getParent()->getFirstInsertionPt();
      return std::next(I->getIterator());
    }
    return F.getEntryBlock().getFirstInsertionPt();
  }

  /// The later of two definition points on one dominator chain.
  Value *later(Value *A, Value *B) {
    auto *IA = dyn_cast<Instruction>(A), *IB = dyn_cast<Instruction>(B);
    if (!IA)
      return B;
    if (!IB)
      return A;
    return DT.dominates(IA, IB) ? B : A;
  }

  /// V rebuilt after its leaves; Last is set to the latest leaf (or null).
  Value *hoist(Value *V, Value *&Last, unsigned Depth = 0) {
    auto *I = dyn_cast<Instruction>(V);
    if (!I) {
      Last = nullptr;
      return V;
    }
    if (Depth > 4 || !isCheapPure(*I)) {
      Last = I;
      return I;
    }
    if (Value *H = Hoisted.lookup(I)) {
      Last = H;
      return H;
    }
    SmallVector<Value *, 3> Ops;
    Value *Latest = nullptr;
    SmallVector<uintptr_t, 6> Key{I->getOpcode(),
                                  reinterpret_cast<uintptr_t>(I->getType())};
    if (auto *Cmp = dyn_cast<CmpInst>(I))
      Key.push_back(Cmp->getPredicate());
    if (auto *II = dyn_cast<IntrinsicInst>(I))
      Key.push_back(II->getIntrinsicID());
    unsigned NOps = isa<CallInst>(I) ? cast<CallInst>(I)->arg_size()
                                     : I->getNumOperands();
    for (unsigned N = 0; N < NOps; ++N) {
      Value *L;
      Value *Op = hoist(I->getOperand(N), L, Depth + 1);
      Latest = L ? (Latest ? later(Latest, L) : L) : Latest;
      Ops.push_back(Op);
      Key.push_back(reinterpret_cast<uintptr_t>(Op));
    }
    if (I->getType()->isVectorTy()) {
      Last = I;
      return I;
    }
    Value *&E = Exprs[Key];
    if (!E) {
      Instruction *C = I->clone();
      C->setName(I->getName() + ".div");
      for (unsigned N = 0; N < NOps; ++N)
        C->setOperand(N, Ops[N]);
      C->dropPoisonGeneratingAnnotations();
      C->insertBefore(after(Latest));
      E = C;
    }
    Hoisted[I] = E;
    Last = E;
    return E;
  }

public:
  DivisorCache(Function &F) : F(F), DT(F) {}

  Divisor get(Value *Y, bool IsSigned, BinaryOperator &At) {
    Value *Last = nullptr;
    Value *H = ShareDivisors ? hoist(Y, Last) : Y;
    auto [It, New] = Divisors.try_emplace({H, unsigned(IsSigned)});
    if (!New)
      return It->second;
    IRBuilder<> B(ShareDivisors ? &*after(Last) : &At);
    Value *YAbs = H, *Neg = nullptr;
    if (IsSigned) {
      Neg = B.CreateICmpSLT(H, B.getInt32(0));
      YAbs = B.CreateSelect(Neg, B.CreateNeg(H), H);
    }
    Value *FY = B.CreateUIToFP(YAbs, B.getFloatTy());
    Value *Rcp = B.CreateUnaryIntrinsic(Intrinsic::agx_rcp, FY);
    Value *Scaled =
        B.CreateFMul(Rcp, ConstantFP::get(B.getFloatTy(), 4294966784.0));
    Value *Z = B.CreateFPToUI(Scaled, B.getInt32Ty());
    Value *P = B.CreateMul(
        B.CreateZExt(Z, B.getInt64Ty()),
        B.CreateZExt(B.CreateMul(B.CreateNeg(YAbs), Z), B.getInt64Ty()));
    Z = B.CreateAdd(Z, B.CreateTrunc(B.CreateLShr(P, 32), B.getInt32Ty()));
    Divisor D{YAbs, Neg, Z};
    if (ShareDivisors)
      It->second = D;
    else
      Divisors.erase(It);
    return D;
  }
};
} // namespace

/// A 32-bit division or remainder without a loop, as AMDGPU expands them
/// (AMDGPUCodeGenPrepare::expandDivRem32): an estimate of 2^32 / y from the
/// f32 reciprocal, one Newton-Raphson step in integers, then two corrections,
/// which is exact when the reciprocal is within 1 ulp. The hardware's rcp is
/// within 0.75 ulp for every mantissa (Apple-M2-Playground 2026-10-20-fdiv32).
/// Division by zero gives all ones (remainder: x), as on NVIDIA GPUs. The
/// divisor's part comes from Divisors, shared by divisions by the same value.
static Value *expandDivRem32(BinaryOperator &BO, DivisorCache &Divisors) {
  IRBuilder<> B(&BO);
  Instruction::BinaryOps Op = BO.getOpcode();
  bool IsSigned = Op == Instruction::SDiv || Op == Instruction::SRem;
  bool IsRem = Op == Instruction::URem || Op == Instruction::SRem;
  Type *I32 = B.getInt32Ty();
  Value *X = BO.getOperand(0);
  Divisor D = Divisors.get(BO.getOperand(1), IsSigned, BO);
  Value *Y = D.Y, *Z = D.Z;
  Value *XNeg = nullptr, *Neg = nullptr;
  if (IsSigned) {
    XNeg = B.CreateICmpSLT(X, B.getInt32(0));
    Neg = IsRem ? XNeg : B.CreateXor(XNeg, D.Neg);
    X = B.CreateSelect(XNeg, B.CreateNeg(X), X);
  }
  auto MulHi = [&](Value *A, Value *C) {
    Value *P = B.CreateMul(B.CreateZExt(A, B.getInt64Ty()), B.CreateZExt(C, B.getInt64Ty()));
    return B.CreateTrunc(B.CreateLShr(P, 32), I32);
  };
  Value *Q = MulHi(X, Z);
  Value *R = B.CreateSub(X, B.CreateMul(Q, Y));
  for (int I = 0; I < 2; ++I) {
    Value *C = B.CreateICmpUGE(R, Y);
    Q = B.CreateSelect(C, B.CreateAdd(Q, B.getInt32(1)), Q);
    R = B.CreateSelect(C, B.CreateSub(R, Y), R);
  }
  Value *Res = IsRem ? R : Q;
  if (IsSigned)
    Res = B.CreateSelect(Neg, B.CreateNeg(Res), Res);
  Value *Zero = B.CreateICmpEQ(BO.getOperand(1), B.getInt32(0));
  return B.CreateSelect(Zero, IsRem ? BO.getOperand(0) : B.getInt32(-1), Res);
}

static cl::opt<bool> AGXLaunchDivisors(
    "agx-launch-divisors", cl::init(false), cl::Hidden,
    cl::desc("Divide by a 32-bit kernel argument with a multiplier and "
             "shifts the launch computes (.agx.kernels extension 7)"));

/// At most this many divisors per kernel take their multipliers from the
/// launch (three uniforms each).
static constexpr unsigned MaxLaunchDivisors = 8;

/// The kernel argument and 32-bit word of it that divisor Y is, if any: an
/// i32 argument, or the low or high word of an i64 one or the low word of an
/// extended i32 one (as the 32-bit fast path of a 64-bit division takes
/// them).
static std::optional<std::pair<unsigned, unsigned>>
launchDivisorSource(Value *Y) {
  using namespace PatternMatch;
  auto Unfrozen = [](Value *V) {
    auto *Fr = dyn_cast<FreezeInst>(V);
    return Fr ? Fr->getOperand(0) : V;
  };
  Y = Unfrozen(Y);
  // umax(d, 1), ZLUDA's guard of a PTX division's divisor: the launch
  // computes the values of max(d, 1) (word 2 and 3 stand for words 0 and 1
  // so guarded)
  Value *Guarded;
  if (match(Y, m_UMax(m_Value(Guarded), m_One()))) {
    auto Src = launchDivisorSource(Guarded);
    if (!Src || Src->second >= 2)
      return std::nullopt;
    return std::pair(Src->first, Src->second + 2);
  }
  if (auto *A = dyn_cast<Argument>(Y))
    return A->getType()->isIntegerTy(32)
               ? std::optional(std::pair(A->getArgNo(), 0u))
               : std::nullopt;
  auto *T = dyn_cast<TruncInst>(Y);
  if (!T || !T->getType()->isIntegerTy(32))
    return std::nullopt;
  Value *V = Unfrozen(T->getOperand(0));
  unsigned Word = 0;
  const APInt *Sh;
  if (match(V, m_LShr(m_Value(V), m_APInt(Sh)))) {
    if (*Sh != 32)
      return std::nullopt;
    V = Unfrozen(V);
    Word = 1;
  }
  if (auto *A = dyn_cast<Argument>(V); A && A->getType()->isIntegerTy(64))
    return std::pair(A->getArgNo(), Word);
  // the low word of an extended i32 argument (an index built from 32-bit
  // arguments) is the argument
  Value *Narrow;
  if (Word == 0 && match(V, m_ZExtOrSExt(m_Value(Narrow))))
    if (auto *A = dyn_cast<Argument>(Narrow); A && A->getType()->isIntegerTy(32))
      return std::pair(A->getArgNo(), 0u);
  return std::nullopt;
}

/// A 32-bit division or remainder by a kernel argument, with the multiplier
/// m and shifts s1, s2 of its divisor that the launch computes
/// (llvm.agx.launch.value): t = mulhi(x, m), q = (t + ((x - t) >> s1)) >> s2,
/// exact for every 32-bit x (Granlund and Montgomery, "Division by Invariant
/// Integers using Multiplication", PLDI 1994, section 4). A zero divisor has
/// m = s1 = s2 = 0: q = x, so the remainder is x; the quotient is all ones,
/// as in expandDivRem32.
static Value *expandLaunchDivRem32(BinaryOperator &BO, unsigned Arg,
                                   unsigned Word) {
  IRBuilder<> B(&BO);
  Instruction::BinaryOps Op = BO.getOpcode();
  bool IsSigned = Op == Instruction::SDiv || Op == Instruction::SRem;
  bool IsRem = Op == Instruction::URem || Op == Instruction::SRem;
  Type *I32 = B.getInt32Ty();
  Function *LV = Intrinsic::getOrInsertDeclaration(
      BO.getModule(), Intrinsic::agx_launch_value);
  // words 2 and 3: max(d, 1) of word 0 or 1 (kinds 6 to 8)
  bool Guarded = Word >= 2;
  auto Launch = [&](unsigned Kind) {
    return B.CreateCall(
        LV, {B.getInt32(Arg), B.getInt32(Word & 1),
             B.getInt32(Kind + (Guarded ? 6 : IsSigned ? 3 : 0))});
  };
  Value *X = BO.getOperand(0), *Y = BO.getOperand(1);
  Value *Neg = nullptr;
  if (IsSigned) {
    Value *XNeg = B.CreateICmpSLT(X, B.getInt32(0));
    Value *YNeg = B.CreateICmpSLT(Y, B.getInt32(0));
    Neg = IsRem ? XNeg : B.CreateXor(XNeg, YNeg);
    X = B.CreateSelect(XNeg, B.CreateNeg(X), X);
    Y = B.CreateSelect(YNeg, B.CreateNeg(Y), Y);
  }
  Value *P = B.CreateMul(B.CreateZExt(X, B.getInt64Ty()),
                         B.CreateZExt(Launch(0), B.getInt64Ty()));
  Value *T = B.CreateTrunc(B.CreateLShr(P, 32), I32);
  Value *Q = B.CreateLShr(
      B.CreateAdd(T, B.CreateLShr(B.CreateSub(X, T), Launch(1))), Launch(2));
  Value *Res = IsRem ? B.CreateSub(X, B.CreateMul(Q, Y)) : Q;
  if (IsSigned)
    Res = B.CreateSelect(Neg, B.CreateNeg(Res), Res);
  if (IsRem || Guarded)
    return Res;
  Value *Zero = B.CreateICmpEQ(BO.getOperand(1), B.getInt32(0));
  return B.CreateSelect(Zero, B.getInt32(-1), Res);
}

bool AGXCodeGenPrepare::expandDivisions(Function &F) {
  // 64-bit divisions whose operands fit in 32 bits (checked at run time, or
  // known) take the 32-bit expansion. Division and remainder of the same
  // operands share one.
  bool Bypassed = false;
  SmallVector<BasicBlock *, 16> Blocks;
  for (BasicBlock &BB : F)
    Blocks.push_back(&BB);
  DenseMap<unsigned, unsigned> BypassWidth = {{64, 32}};
  for (BasicBlock *BB : Blocks)
    Bypassed |= bypassSlowDivision(BB, BypassWidth);

  SmallVector<BinaryOperator *, 8> Worklist;
  for (Instruction &I : instructions(F)) {
    auto *BO = dyn_cast<BinaryOperator>(&I);
    if (!BO || !BO->getType()->isIntegerTy())
      continue;
    switch (BO->getOpcode()) {
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::URem:
    case Instruction::SRem:
      if (!isCheapDivision(*BO) &&
          BO->getType()->getIntegerBitWidth() <= 64)
        Worklist.push_back(BO);
      break;
    default:
      break;
    }
  }

  // The 32-bit expansions first: they keep the CFG (and the dominator tree
  // of the divisor cache) as it is; the loops of the wider ones do not.
  std::optional<DivisorCache> Divisors;
  SmallVector<BinaryOperator *, 8> Wide;
  // Divisors whose multipliers the launch computes, while the uniforms have
  // room for them (AGXPromoteKernelArgs fills at most 1024 bytes less 20).
  SmallVector<std::pair<unsigned, unsigned>, 4> LaunchDivisors;
  bool Kernel = AGXLaunchDivisors &&
                F.getCallingConv() == CallingConv::AGX_KERNEL;
  if (Kernel) {
    uint64_t Bytes = 20 + 8;
    for (const Argument &A : F.args())
      Bytes += alignTo(F.getDataLayout().getTypeStoreSize(A.getType()), 4);
    Kernel = Bytes + MaxLaunchDivisors * 2 * 12 <= 1024;
  }
  for (BinaryOperator *BO : Worklist) {
    if (!BO->getType()->isIntegerTy(32)) {
      Wide.push_back(BO);
      continue;
    }
    auto Src = Kernel ? launchDivisorSource(BO->getOperand(1)) : std::nullopt;
    // max(d, 1) has unsigned values only
    if (Src && Src->second >= 2 && (BO->getOpcode() == Instruction::SDiv ||
                                     BO->getOpcode() == Instruction::SRem))
      Src = std::nullopt;
    if (Src) {
      bool Known = is_contained(LaunchDivisors, *Src);
      if (Known || LaunchDivisors.size() < MaxLaunchDivisors) {
        if (!Known)
          LaunchDivisors.push_back(*Src);
        Value *R = expandLaunchDivRem32(*BO, Src->first, Src->second);
        R->takeName(BO);
        BO->replaceAllUsesWith(R);
        BO->eraseFromParent();
        continue;
      }
    }
    if (!Divisors)
      Divisors.emplace(F);
    Value *R = expandDivRem32(*BO, *Divisors);
    R->takeName(BO);
    BO->replaceAllUsesWith(R);
    BO->eraseFromParent();
  }
  for (BinaryOperator *BO : Wide) {
    bool IsRem = BO->getOpcode() == Instruction::URem ||
                 BO->getOpcode() == Instruction::SRem;
    bool Is64 = BO->getType()->getIntegerBitWidth() > 32;
    if (IsRem)
      Is64 ? expandRemainderUpTo64Bits(BO) : expandRemainderUpTo32Bits(BO);
    else
      Is64 ? expandDivisionUpTo64Bits(BO) : expandDivisionUpTo32Bits(BO);
  }
  return Bypassed || !Worklist.empty();
}

bool AGXCodeGenPrepare::expandMemIntrinsics(Function &F,
                                            const TargetTransformInfo &TTI) {
  // Instruction selection expands memory intrinsics of a known, small size
  // into loads and stores; everything else becomes a loop.
  constexpr uint64_t MaxInlineSize = 128;
  SmallVector<MemIntrinsic *, 4> Worklist;
  for (Instruction &I : instructions(F)) {
    auto *MI = dyn_cast<MemIntrinsic>(&I);
    if (!MI)
      continue;
    auto *Len = dyn_cast<ConstantInt>(MI->getLength());
    if (Len && Len->getZExtValue() <= MaxInlineSize)
      continue;
    Worklist.push_back(MI);
  }

  for (MemIntrinsic *MI : Worklist) {
    if (auto *Cpy = dyn_cast<MemCpyInst>(MI)) {
      expandMemCpyAsLoop(Cpy, TTI);
    } else if (auto *Move = dyn_cast<MemMoveInst>(MI)) {
      if (!expandMemMoveAsLoop(Move, TTI))
        continue;
    } else if (auto *Set = dyn_cast<MemSetInst>(MI)) {
      expandMemSetAsLoop(Set, TTI);
    } else {
      continue;
    }
    MI->eraseFromParent();
  }
  return !Worklist.empty();
}

bool AGXCodeGenPrepare::lowerConstrainedFP(Function &F) {
  SmallVector<ConstrainedFPIntrinsic *, 8> Worklist;
  for (Instruction &I : instructions(F))
    if (auto *CI = dyn_cast<ConstrainedFPIntrinsic>(&I))
      Worklist.push_back(CI);

  for (ConstrainedFPIntrinsic *CI : Worklist) {
    IRBuilder<> B(CI);
    SmallVector<Value *, 4> Args(CI->args().begin(),
                                 CI->args().begin() +
                                     CI->getNonMetadataArgCount());
    Value *Res = nullptr;
    switch (CI->getIntrinsicID()) {
#define INSTRUCTION(NAME, NARGS, ROUND_MODE, INTRINSIC)                        \
  case Intrinsic::INTRINSIC:                                                   \
    if (Instruction::isBinaryOp(Instruction::NAME))                            \
      Res = B.CreateBinOp(Instruction::BinaryOps(Instruction::NAME), Args[0],  \
                          Args[1]);                                            \
    else                                                                       \
      Res = B.CreateCast(Instruction::CastOps(Instruction::NAME), Args[0],     \
                         CI->getType());                                       \
    break;
#define CMP_INSTRUCTION(NAME, NARGS, ROUND_MODE, INTRINSIC, DAGN)              \
  case Intrinsic::INTRINSIC:                                                   \
    Res = B.CreateFCmp(cast<ConstrainedFPCmpIntrinsic>(CI)->getPredicate(),    \
                       Args[0], Args[1]);                                      \
    break;
#define FUNCTION(NAME, NARGS, ROUND_MODE, INTRINSIC)                           \
  case Intrinsic::INTRINSIC: {                                                 \
    SmallVector<Type *, 4> ArgTys;                                             \
    for (Value *A : Args)                                                      \
      ArgTys.push_back(A->getType());                                          \
    SmallVector<Type *, 2> OverloadTys;                                        \
    FunctionType *FT = FunctionType::get(CI->getType(), ArgTys, false);        \
    if (Intrinsic::isSignatureValid(Intrinsic::NAME, FT, OverloadTys))         \
      Res = B.CreateCall(Intrinsic::getOrInsertDeclaration(                    \
                             F.getParent(), Intrinsic::NAME, OverloadTys),     \
                         Args);                                                \
    break;                                                                     \
  }
#include "llvm/IR/ConstrainedOps.def"
    default:
      break;
    }
    if (!Res)
      continue;
    Res->takeName(CI);
    CI->replaceAllUsesWith(Res);
    CI->eraseFromParent();
  }
  return !Worklist.empty();
}

/// Replace the returns, and unreachable terminators, by branches to a single
/// return block.
bool AGXCodeGenPrepare::unifyExits(Function &F) {
  SmallVector<Instruction *, 4> Exits;
  for (BasicBlock &BB : F) {
    Instruction *T = BB.getTerminator();
    if (isa<ReturnInst>(T) || isa<UnreachableInst>(T))
      Exits.push_back(T);
  }
  if (!F.getReturnType()->isVoidTy())
    return false;
  // A thread that reaches unreachable (after llvm.trap, as in __assertfail)
  // returns: an executed trap does nothing on G14C (ptx-re-blockers B21), and
  // the code after the block would otherwise run on.
  if (Exits.size() == 1) {
    if (!isa<UnreachableInst>(Exits.front()))
      return false;
    ReturnInst::Create(F.getContext(), nullptr, Exits.front()->getIterator());
    Exits.front()->eraseFromParent();
    return true;
  }
  if (Exits.empty())
    return false;

  BasicBlock *Ret = BasicBlock::Create(F.getContext(), "unified.return", &F);
  ReturnInst::Create(F.getContext(), Ret);
  for (Instruction *T : Exits) {
    UncondBrInst::Create(Ret, T->getIterator());
    T->eraseFromParent();
  }
  return true;
}

/// simd_shuffle reads from each quad the thread at the OR of the low two
/// index bits of the quad's threads (applegpu documents this, Mesa works
/// around it). A shuffle whose index is the same in a quad (an immediate, or
/// low bits that are uniform) reads the thread it names. Divergent indices
/// are split into the four quad positions:
///
///   r = shuffle(v, idx & ~3); r = idx & 3 == k ? shuffle(v, idx & ~3 | k) : r
///
/// leaving out the positions that the known bits of the index exclude.
/// Quad shuffles take the four immediate positions instead. Relative
/// shuffles with divergent deltas become shuffles of the computed thread;
/// threads whose source is outside the SIMD-group (or quad) keep their own
/// value, as Metal documents for simd_shuffle_up and simd_shuffle_down.
bool AGXCodeGenPrepare::lowerDivergentShuffles(Function &F,
                                               const TargetTransformInfo &TTI) {
  SmallVector<IntrinsicInst *, 8> Worklist;
  for (Instruction &I : instructions(F)) {
    auto *II = dyn_cast<IntrinsicInst>(&I);
    if (!II)
      continue;
    switch (II->getIntrinsicID()) {
    case Intrinsic::agx_simd_shuffle:
    case Intrinsic::agx_simd_shuffle_xor:
    case Intrinsic::agx_simd_shuffle_up:
    case Intrinsic::agx_simd_shuffle_down:
    case Intrinsic::agx_quad_shuffle:
    case Intrinsic::agx_quad_shuffle_xor:
    case Intrinsic::agx_quad_shuffle_up:
    case Intrinsic::agx_quad_shuffle_down:
      if (!isa<Constant>(II->getArgOperand(1)))
        Worklist.push_back(II);
      break;
    default:
      break;
    }
  }

  // Most compute kernels have no variable shuffle indices. Do not compute
  // whole-function divergence (and its dominator/cycle analyses) for them.
  // Build it after matrix lowering, which can itself introduce shuffles.
  if (Worklist.empty())
    return false;
  DominatorTree DT(F);
  CycleInfo CI;
  CI.compute(F);
  UniformityInfo UI(DT, CI, &TTI);
  UI.compute();
  llvm::erase_if(Worklist, [&](IntrinsicInst *II) {
    return !UI.isDivergentAtUse(II->getArgOperandUse(1));
  });

  for (IntrinsicInst *II : Worklist) {
    IRBuilder<> B(II);
    Value *V = II->getArgOperand(0);
    Value *Idx = II->getArgOperand(1);
    Type *Ty = V->getType();
    Intrinsic::ID ID = II->getIntrinsicID();
    bool Quad = ID == Intrinsic::agx_quad_shuffle ||
                ID == Intrinsic::agx_quad_shuffle_xor ||
                ID == Intrinsic::agx_quad_shuffle_up ||
                ID == Intrinsic::agx_quad_shuffle_down;
    unsigned Size = Quad ? 4 : 32;

    // The thread to read, relative to the start of the quad for quad
    // shuffles.
    Value *Lane = nullptr;
    if (ID != Intrinsic::agx_simd_shuffle && ID != Intrinsic::agx_quad_shuffle) {
      Lane = B.CreateIntrinsic(Intrinsic::agx_thread_index_in_simdgroup, {});
      if (Quad)
        Lane = B.CreateAnd(Lane, 3);
    }
    switch (ID) {
    case Intrinsic::agx_simd_shuffle_xor:
    case Intrinsic::agx_quad_shuffle_xor:
      Idx = B.CreateAnd(B.CreateXor(Lane, Idx), Size - 1);
      break;
    case Intrinsic::agx_simd_shuffle_up:
    case Intrinsic::agx_quad_shuffle_up:
      Idx = B.CreateSelect(B.CreateICmpUGE(Lane, Idx), B.CreateSub(Lane, Idx),
                           Lane);
      break;
    case Intrinsic::agx_simd_shuffle_down:
    case Intrinsic::agx_quad_shuffle_down:
      Idx = B.CreateSelect(
          B.CreateICmpULT(Idx, B.CreateSub(B.getInt32(Size), Lane)),
          B.CreateAdd(Lane, Idx), Lane);
      break;
    default:
      break;
    }

    Intrinsic::ID Shuffle =
        Quad ? Intrinsic::agx_quad_shuffle : Intrinsic::agx_simd_shuffle;
    Value *Low = B.CreateAnd(Idx, 3);
    Value *Base = Quad ? nullptr : B.CreateAnd(Idx, 0x1c);
    auto Position = [&](unsigned K) -> Value * {
      if (Quad)
        return B.getInt32(K);
      return K ? B.CreateOr(Base, K) : Base;
    };
    // Positions the low bits cannot take need no shuffle.
    KnownBits Known = computeKnownBits(Idx, F.getDataLayout());
    SmallVector<unsigned, 4> Positions;
    for (unsigned K = 0; K < 4; ++K)
      if ((K & Known.Zero.getZExtValue() & 3) == 0 &&
          (~K & Known.One.getZExtValue() & 3) == 0)
        Positions.push_back(K);
    Value *Result =
        B.CreateIntrinsic(Shuffle, {Ty}, {V, Position(Positions[0])});
    for (unsigned K : drop_begin(Positions)) {
      Value *R = B.CreateIntrinsic(Shuffle, {Ty}, {V, Position(K)});
      Result = B.CreateSelect(B.CreateICmpEQ(Low, B.getInt32(K)), R, Result);
    }
    Result->takeName(II);
    II->replaceAllUsesWith(Result);
    II->eraseFromParent();
  }
  return !Worklist.empty();
}

/// Whether I computes a floating point value from its operands (and so
/// reads and produces denormals according to the denormal mode).
static bool isFPArithmetic(const Instruction &I) {
  switch (I.getOpcode()) {
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::FRem:
    return true;
  default:
    break;
  }
  const auto *II = dyn_cast<IntrinsicInst>(&I);
  if (!II)
    return false;
  switch (II->getIntrinsicID()) {
  case Intrinsic::fma:
  case Intrinsic::fmuladd:
  case Intrinsic::sqrt:
  case Intrinsic::minnum:
  case Intrinsic::maxnum:
  case Intrinsic::minimum:
  case Intrinsic::maximum:
  case Intrinsic::minimumnum:
  case Intrinsic::maximumnum:
  case Intrinsic::exp2:
  case Intrinsic::log2:
  case Intrinsic::sin:
  case Intrinsic::cos:
  case Intrinsic::agx_exp2:
  case Intrinsic::agx_log2:
  case Intrinsic::agx_sin:
  case Intrinsic::agx_cos:
  case Intrinsic::floor:
  case Intrinsic::ceil:
  case Intrinsic::trunc:
  case Intrinsic::rint:
  case Intrinsic::nearbyint:
  case Intrinsic::roundeven:
  case Intrinsic::canonicalize:
  case Intrinsic::agx_rcp:
  case Intrinsic::agx_rsqrt:
    return true;
  default:
    return false;
  }
}

static bool isHalf(const Type *Ty) { return Ty->getScalarType()->isHalfTy(); }

/// llvm.agx.simdgroup.matrix.fma: D = A * B + C on 8x8 matrices spread over
/// the SIMD-group. Thread t holds the elements (r, c) and (r, c + 1) of each
/// matrix, with r = 4 * (t / 16) + (t / 2) % 4 and c = 4 * ((t / 8) % 2) +
/// 2 * (t % 2). Each thread gathers its row of A and its two columns of B
/// with shuffles and accumulates in order of k with fused multiply-adds.
///
/// On G14 the matrix unit (simd_matrix_fmadd32 / 16) computes the same chain
/// of fused multiply-adds for the same layout (measured: Apple-M2-Playground
/// results/2026-10-21-b15-simd-matrix): float flushing denormals like the
/// ALU, half keeping them. It is used where the shuffled form gives the same
/// results: float without IEEE denormals in software (ieee-fp32), half with
/// IEEE denormals, and no canonical NaNs.
bool AGXCodeGenPrepare::lowerMatrixFMA(Function &F, const AGXSubtarget &ST) {
  SmallVector<IntrinsicInst *, 2> Calls;
  for (Instruction &I : instructions(F))
    if (auto *II = dyn_cast<IntrinsicInst>(&I);
        II && II->getIntrinsicID() == Intrinsic::agx_simdgroup_matrix_fma)
      Calls.push_back(II);
  bool Unit = ST.hasSIMDMatrix() && !ST.hasMesaISA() && !ST.hasCanonicalNaN() &&
              ST.getGeneration() == AGXSubtarget::G14;
  bool FloatUnit =
      Unit && !(ST.hasIEEEFP32() &&
                F.getDenormalMode(APFloat::IEEEsingle()) == DenormalMode::getIEEE());
  bool HalfUnit =
      Unit && F.getDenormalMode(APFloat::IEEEhalf()) == DenormalMode::getIEEE();
  for (IntrinsicInst *II : Calls) {
    IRBuilder<> B(II);
    Type *VT = II->getType();
    Value *A = II->getArgOperand(0), *Bm = II->getArgOperand(1),
          *C = II->getArgOperand(2);
    Type *ET = VT->getScalarType();
    if ((ET->isFloatTy() && FloatUnit) || (ET->isHalfTy() && HalfUnit)) {
      Type *IT = ET->isFloatTy() ? B.getInt64Ty() : B.getInt32Ty();
      Intrinsic::ID ID = ET->isFloatTy() ? Intrinsic::agx_simd_matrix_fmadd32
                                         : Intrinsic::agx_simd_matrix_fmadd16;
      Value *D = B.CreateIntrinsic(ID, {},
                                   {B.CreateBitCast(A, IT), B.CreateBitCast(Bm, IT),
                                    B.CreateBitCast(C, IT)});
      II->replaceAllUsesWith(B.CreateBitCast(D, VT));
      II->eraseFromParent();
      continue;
    }
    Value *T = B.CreateIntrinsic(Intrinsic::agx_thread_index_in_simdgroup, {});
    auto Bits = [&](Value *V, unsigned Shift, unsigned Mask) {
      return B.CreateAnd(B.CreateLShr(V, Shift), Mask);
    };
    // The thread holding the elements 2j and 2j + 1 of row r.
    auto Holder = [&](Value *R, Value *J) {
      Value *L = B.CreateShl(B.CreateLShr(R, 2), 4);
      L = B.CreateOr(L, B.CreateShl(B.CreateLShr(J, 1), 3));
      L = B.CreateOr(L, B.CreateShl(B.CreateAnd(R, 3), 1));
      return B.CreateOr(L, B.CreateAnd(J, 1));
    };
    auto Shuffle = [&](Value *V, unsigned Elt, Value *Lane) {
      Value *X = B.CreateExtractElement(V, Elt);
      return B.CreateIntrinsic(Intrinsic::agx_simd_shuffle, {X->getType()}, {X, Lane});
    };
    Value *Row = B.CreateOr(B.CreateShl(Bits(T, 4, 1), 2), Bits(T, 1, 3));
    Value *ColPair = B.CreateOr(B.CreateShl(Bits(T, 3, 1), 1), Bits(T, 0, 1));
    SmallVector<Value *, 8> RowA, ColB0, ColB1;
    for (unsigned J = 0; J < 4; ++J) {
      Value *Lane = Holder(Row, B.getInt32(J));
      RowA.push_back(Shuffle(A, 0, Lane));
      RowA.push_back(Shuffle(A, 1, Lane));
    }
    for (unsigned K = 0; K < 8; ++K) {
      Value *Lane = Holder(B.getInt32(K), ColPair);
      ColB0.push_back(Shuffle(Bm, 0, Lane));
      ColB1.push_back(Shuffle(Bm, 1, Lane));
    }
    Value *D0 = B.CreateExtractElement(C, uint64_t(0));
    Value *D1 = B.CreateExtractElement(C, 1);
    for (unsigned K = 0; K < 8; ++K) {
      D0 = B.CreateIntrinsic(Intrinsic::fma, {D0->getType()}, {RowA[K], ColB0[K], D0});
      D1 = B.CreateIntrinsic(Intrinsic::fma, {D1->getType()}, {RowA[K], ColB1[K], D1});
    }
    Value *R = B.CreateInsertElement(PoisonValue::get(VT), D0, uint64_t(0));
    R = B.CreateInsertElement(R, D1, 1);
    II->replaceAllUsesWith(R);
    II->eraseFromParent();
  }
  return !Calls.empty();
}

/// The hardware minimum and maximum return a denormal operand as it is, also
/// where float denormals are flushed: flush their results.
bool AGXCodeGenPrepare::flushFloatMinMax(Function &F) {
  if (F.getDenormalMode(APFloat::IEEEsingle()).Output == DenormalMode::IEEE)
    return false;
  SmallVector<IntrinsicInst *, 8> MinMax;
  for (Instruction &I : instructions(F)) {
    auto *II = dyn_cast<IntrinsicInst>(&I);
    if (!II || !II->getType()->isFloatTy() || II->hasMetadata("agx.ieee") ||
        II->hasMetadata("agx.exact"))
      continue;
    switch (II->getIntrinsicID()) {
    case Intrinsic::minnum:
    case Intrinsic::maxnum:
    case Intrinsic::minimum:
    case Intrinsic::maximum:
    case Intrinsic::minimumnum:
    case Intrinsic::maximumnum:
      MinMax.push_back(II);
      break;
    default:
      break;
    }
  }
  for (IntrinsicInst *II : MinMax) {
    IRBuilder<> B(II->getNextNode());
    SmallVector<Use *, 4> Uses;
    for (Use &U : II->uses())
      Uses.push_back(&U);
    Value *R = B.CreateUnaryIntrinsic(Intrinsic::canonicalize, II);
    for (Use *U : Uses)
      U->set(R);
  }
  return !MinMax.empty();
}

bool AGXCodeGenPrepare::flushHalfDenormals(Function &F) {
  DenormalMode Mode = F.getDenormalMode(APFloat::IEEEhalf());
  bool FlushInputs = Mode.Input != DenormalMode::IEEE &&
                     Mode.Input != DenormalMode::Dynamic;
  bool FlushOutputs = Mode.Output != DenormalMode::IEEE &&
                      Mode.Output != DenormalMode::Dynamic;
  if (!FlushInputs && !FlushOutputs)
    return false;
  bool PositiveZero = Mode.Output == DenormalMode::PositiveZero;

  // flush(x) = |x| < 2^-14 ? +-0 : x. NaNs compare false and pass through.
  SmallPtrSet<Value *, 16> Flushed;
  auto Flush = [&](IRBuilder<> &B, Value *X, bool Positive) -> Value * {
    if (Flushed.contains(X))
      return X;
    if (auto *C = dyn_cast<ConstantFP>(X))
      if (!C->getValueAPF().isDenormal())
        return X;
    Type *Ty = X->getType();
    Value *Abs = B.CreateUnaryIntrinsic(Intrinsic::fabs, X);
    Value *Small = B.CreateFCmpOLT(
        Abs, ConstantFP::get(Ty, APFloat::getSmallestNormalized(
                                     APFloat::IEEEhalf())));
    Value *Zero = Positive ? ConstantFP::get(Ty, 0.0)
                           : B.CreateBinaryIntrinsic(Intrinsic::copysign,
                                                     ConstantFP::get(Ty, 0.0),
                                                     X);
    Value *R = B.CreateSelect(Small, Zero, X);
    Flushed.insert(R);
    return R;
  };

  SmallVector<Instruction *, 32> Worklist;
  for (Instruction &I : instructions(F)) {
    // Operations with their own denormal handling (from AGXLowerFPEnv).
    if (I.hasMetadata("agx.ieee") || I.hasMetadata("agx.exact"))
      continue;
    bool Arith = isFPArithmetic(I) && isHalf(I.getType());
    bool Reads = Arith || (isa<FCmpInst>(I) &&
                           isHalf(I.getOperand(0)->getType())) ||
                 (isa<FPExtInst>(I) && isHalf(I.getOperand(0)->getType())) ||
                 ((isa<FPToSIInst>(I) || isa<FPToUIInst>(I)) &&
                  isHalf(I.getOperand(0)->getType()));
    bool Writes = Arith || (isa<FPTruncInst>(I) && isHalf(I.getType()));
    if (Reads || Writes)
      Worklist.push_back(&I);
  }

  for (Instruction *I : Worklist) {
    IRBuilder<> B(I);
    bool Arith = isFPArithmetic(*I) && isHalf(I->getType());
    bool Reads = !isa<FPTruncInst>(I);
    if (FlushInputs && Reads) {
      unsigned NumOps = isa<CallBase>(I) ? cast<CallBase>(I)->arg_size()
                                         : I->getNumOperands();
      for (unsigned Op = 0; Op < NumOps; ++Op)
        if (I->getOperand(Op)->getType()->isFPOrFPVectorTy() &&
            isHalf(I->getOperand(Op)->getType()))
          I->setOperand(Op, Flush(B, I->getOperand(Op), false));
    }
    if (auto *II = dyn_cast<IntrinsicInst>(I);
        II && II->getIntrinsicID() == Intrinsic::canonicalize) {
      // The operand is flushed already; nothing else to canonicalize.
      Value *Op = II->getArgOperand(0);
      if (FlushOutputs && !Flushed.contains(Op))
        Op = Flush(B, Op, PositiveZero);
      II->replaceAllUsesWith(Op);
      II->eraseFromParent();
      continue;
    }
    if (FlushOutputs && (Arith || isa<FPTruncInst>(I))) {
      B.SetInsertPoint(I->getNextNode());
      SmallVector<User *, 4> OldUsers(I->users());
      Value *R = Flush(B, I, PositiveZero);
      I->replaceUsesWithIf(R, [&](Use &U) {
        return is_contained(OldUsers, U.getUser());
      });
    }
  }
  // The flushes are explicit now. Later combines would otherwise assume that
  // the hardware flushes and fold them away.
  AttrBuilder AB(F.getContext());
  AB.addDenormalFPEnvAttr(DenormalFPEnv(
      DenormalMode::getIEEE(), F.getDenormalMode(APFloat::IEEEsingle())));
  F.addFnAttrs(AB);
  return true;
}

bool AGXCodeGenPrepare::diagnoseDoubles(Function &F) {
  SmallVector<Instruction *, 8> Unsupported;
  for (Instruction &I : instructions(F)) {
    bool Double = (isFPArithmetic(I) || isa<FCmpInst>(I) ||
                   isa<FPToSIInst>(I) || isa<FPToUIInst>(I) ||
                   isa<FPTruncInst>(I)) &&
                  I.getOperand(0)->getType()->getScalarType()->isDoubleTy();
    Double |= (isa<SIToFPInst>(I) || isa<UIToFPInst>(I) ||
               isa<FPExtInst>(I) || isFPArithmetic(I)) &&
              I.getType()->getScalarType()->isDoubleTy();
    if (Double)
      Unsupported.push_back(&I);
  }
  if (Unsupported.empty())
    return false;
  F.getContext().diagnose(DiagnosticInfoUnsupported(
      F, "double precision floating point arithmetic (AGX has no double "
         "precision hardware)",
      Unsupported.front()->getDebugLoc()));
  for (Instruction *I : Unsupported) {
    I->replaceAllUsesWith(PoisonValue::get(I->getType()));
    I->eraseFromParent();
  }
  return true;
}

bool AGXCodeGenPrepare::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;
  const TargetTransformInfo &TTI =
      getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
  const TargetMachine &TM =
      getAnalysis<TargetPassConfig>().getTM<TargetMachine>();
  const auto &ST = TM.getSubtarget<AGXSubtarget>(F);
  bool Changed = diagnoseDoubles(F);
  Changed |= lowerMatrixFMA(F, ST);
  if (ST.hasQuadShuffleIndexMerge())
    Changed |= lowerDivergentShuffles(F, TTI);
  Changed |= combineUniformAtomics(F, TTI);
  Changed |= lowerConstrainedFP(F);
  Changed |= expandMath(F, ST);
  Changed |= flushHalfDenormals(F);
  Changed |= flushFloatMinMax(F);
  Changed |= expandDivisions(F);
  Changed |= expandMemIntrinsics(F, TTI);
  Changed |= fenceHostCoherentAccesses(F);
  Changed |= useBlockBarriers(F, ST);
  Changed |= unifyExits(F);
  return Changed;
}

static cl::opt<bool> AGXAtomicOptimizer(
    "agx-atomic-optimizer", cl::init(false), cl::Hidden,
    cl::desc("Combine the atomics of a SIMD-group to one uniform address into "
             "one atomic of the group's reduction (as AMDGPUAtomicOptimizer)"));

/// The AGX SIMD-group reduction operation (AGX::SIMDOp numbering of
/// llvm.agx.simd.reduce) for an atomicrmw, or -1.
static int simdOpFor(const AtomicRMWInst &RMW) {
  bool Int32 = RMW.getType()->isIntegerTy(32);
  switch (RMW.getOperation()) {
  case AtomicRMWInst::Add:
  case AtomicRMWInst::Sub:
    return Int32 ? 16 : -1;
  case AtomicRMWInst::And:
    return Int32 ? 0 : -1;
  case AtomicRMWInst::Or:
    return Int32 ? 2 : -1;
  case AtomicRMWInst::Xor:
    return Int32 ? 4 : -1;
  case AtomicRMWInst::Min:
    return Int32 ? 20 : -1;
  case AtomicRMWInst::Max:
    return Int32 ? 22 : -1;
  case AtomicRMWInst::UMin:
    return Int32 ? 28 : -1;
  case AtomicRMWInst::UMax:
    return Int32 ? 30 : -1;
  default:
    // Not fadd: a reduction rounds and flushes denormals unlike the atomic
    // (PTX-formalization atom-f32-ftz).
    return -1;
  }
}

/// Atomics of every active thread of a SIMD-group to one (uniform) address
/// become one atomic, done by the first active thread, of the reduction of
/// their operands; the others' results are rebuilt from the old value and an
/// exclusive scan (only for add and sub, which have one). 32-bit integers. Same-address
/// atomics serialize in the memory system, which agx_cost does not model:
/// off by default until hardware measurements decide it (EXPERIMENTS.md).
bool AGXCodeGenPrepare::combineUniformAtomics(Function &F,
                                              const TargetTransformInfo &TTI) {
  if (!AGXAtomicOptimizer)
    return false;
  SmallVector<AtomicRMWInst *, 8> Candidates;
  for (Instruction &I : instructions(F))
    if (auto *RMW = dyn_cast<AtomicRMWInst>(&I))
      if (!RMW->isVolatile() && simdOpFor(*RMW) >= 0)
        Candidates.push_back(RMW);
  if (Candidates.empty())
    return false;
  DominatorTree DT(F);
  CycleInfo CI;
  CI.compute(F);
  UniformityInfo UI(DT, CI, &TTI);
  UI.compute();

  bool Changed = false;
  for (AtomicRMWInst *RMW : Candidates) {
    Value *Ptr = RMW->getPointerOperand();
    if (UI.isDivergentAtUse(RMW->getOperandUse(AtomicRMWInst::getPointerOperandIndex())))
      continue;
    AtomicRMWInst::BinOp Op = RMW->getOperation();
    bool Scan = Op == AtomicRMWInst::Add || Op == AtomicRMWInst::Sub;
    if (!RMW->use_empty() && !Scan)
      continue;
    Type *Ty = RMW->getType();
    Value *V = RMW->getValOperand();
    int HWOp = simdOpFor(*RMW);
    IRBuilder<> B(RMW);
    Value *Total, *Before = nullptr;
    // From the ballot of the active threads (the active-thread special
    // registers disagree with it in agx_run).
    Value *Ballot = B.CreateIntrinsic(Intrinsic::agx_simd_ballot,
                                      ArrayRef<Type *>(), {B.getTrue()});
    Value *Lane = B.CreateIntrinsic(Intrinsic::agx_thread_index_in_simdgroup,
                                    ArrayRef<Type *>(), ArrayRef<Value *>());
    Value *Active =
        B.CreateIntrinsic(Intrinsic::ctpop, {B.getInt32Ty()}, {Ballot});
    Value *Below = B.CreateAnd(
        Ballot, B.CreateSub(B.CreateShl(B.getInt32(1), Lane), B.getInt32(1)));
    Value *Index =
        B.CreateIntrinsic(Intrinsic::ctpop, {B.getInt32Ty()}, {Below});
    if (!UI.isDivergentAtUse(RMW->getOperandUse(1)) && Ty->isIntegerTy()) {
      // A uniform operand: n times it for add and sub, itself otherwise.
      bool Sum = Op == AtomicRMWInst::Add || Op == AtomicRMWInst::Sub;
      Total = Sum ? B.CreateMul(V, Active) : V;
      if (Sum && !RMW->use_empty())
        Before = B.CreateMul(V, Index);
    } else {
      Total = B.CreateIntrinsic(Intrinsic::agx_simd_reduce, {Ty},
                                {V, B.getInt32(HWOp)});
      if (!RMW->use_empty())
        Before = B.CreateIntrinsic(Intrinsic::agx_simd_prefix, {Ty},
                                   {V, B.getInt32(HWOp)});
    }
    Value *First = B.CreateICmpEQ(Index, B.getInt32(0));
    Instruction *Then = SplitBlockAndInsertIfThen(First, RMW, false);
    BasicBlock *ThenBB = Then->getParent(), *Tail = RMW->getParent();
    B.SetInsertPoint(Then);
    AtomicRMWInst *One = B.CreateAtomicRMW(Op, Ptr, Total, RMW->getAlign(),
                                           RMW->getOrdering(),
                                           RMW->getSyncScopeID());
    One->setVolatile(false);
    if (!RMW->use_empty()) {
      B.SetInsertPoint(RMW);
      PHINode *Old = B.CreatePHI(Ty, 2);
      Old->addIncoming(One, ThenBB);
      Old->addIncoming(PoisonValue::get(Ty), ThenBB->getSinglePredecessor());
      // The first active thread's lane: the lowest set bit of the ballot.
      Value *FirstLane = B.CreateIntrinsic(Intrinsic::cttz, {B.getInt32Ty()},
                                           {Ballot, B.getTrue()});
      Value *Base = B.CreateIntrinsic(Intrinsic::agx_simd_shuffle, {Ty},
                                      {Old, FirstLane});
      Value *Result = Op == AtomicRMWInst::Sub ? B.CreateSub(Base, Before)
                                               : B.CreateAdd(Base, Before);
      RMW->replaceAllUsesWith(Result);
    }
    (void)Tail;
    RMW->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

/// powi(x, n) with a variable n: pow(|x|, n) (AGX has no powi library
/// function, and SelectionDAG's fallback, pow(x, n), is NaN for a negative x),
/// with the sign of x for an odd n. A constant n is left to SelectionDAG's
/// multiplications.
static bool expandPowI(Function &F) {
  SmallVector<IntrinsicInst *, 4> Calls;
  for (Instruction &I : instructions(F))
    if (auto *II = dyn_cast<IntrinsicInst>(&I);
        II && II->getIntrinsicID() == Intrinsic::powi &&
        !isa<Constant>(II->getArgOperand(1)) &&
        (II->getType()->isFloatTy() || II->getType()->isHalfTy()))
      Calls.push_back(II);
  for (IntrinsicInst *II : Calls) {
    IRBuilder<> B(II);
    B.setFastMathFlags(II->getFastMathFlags());
    Type *F32 = B.getFloatTy();
    Value *X = B.CreateFPExt(II->getArgOperand(0), F32);
    Value *N = II->getArgOperand(1);
    Value *Mag = B.CreateBinaryIntrinsic(
        Intrinsic::pow, B.CreateUnaryIntrinsic(Intrinsic::fabs, X),
        B.CreateSIToFP(N, F32));
    Value *Odd = B.CreateTrunc(N, B.getInt1Ty());
    Value *R = B.CreateCopySign(
        Mag, B.CreateSelect(Odd, X, ConstantFP::get(F32, 1.0)));
    R = B.CreateFPTrunc(R, II->getType());
    R->takeName(II);
    II->replaceAllUsesWith(R);
    II->eraseFromParent();
  }
  return !Calls.empty();
}

/// frem of float and half (in float): fmod, exact. A step reduces a by a
/// divisor d with a / d < 2^24: q = trunc(a / d) is the integer part n or,
/// rounded up, n + 1 (the quotient is correctly rounded and n representable),
/// so the remainder is a - q d or a - (q - 1) d, whichever is in [0, d)
/// (exact, or flushed). While a exceeds b 2^23 by more than an exponent
/// step, a is first reduced by b 2^j, a multiple of b, with a / (b 2^j) <
/// 2^24 (each step takes 23 bits off the exponent). A denormal b (IEEE float)
/// is normalized by 2^24 for the loop, with a last step by b itself. The
/// result has the sign of a; fmod(x, +-inf) is x (also a denormal x), and
/// NaN for an infinite or NaN x, a NaN or zero y.
static bool expandFRem(Function &F, const AGXSubtarget &ST) {
  SmallVector<BinaryOperator *, 4> Rems;
  for (Instruction &I : instructions(F))
    if (I.getOpcode() == Instruction::FRem &&
        (I.getType()->isFloatTy() || I.getType()->isHalfTy()))
      Rems.push_back(cast<BinaryOperator>(&I));
  bool DenormalB =
      ST.hasIEEEFP32() &&
      F.getDenormalMode(APFloat::IEEEsingle()).Input == DenormalMode::IEEE;
  for (BinaryOperator *Rem : Rems) {
    LLVMContext &Ctx = F.getContext();
    BasicBlock *Before = Rem->getParent();
    BasicBlock *After = Before->splitBasicBlock(Rem, "frem.done");
    BasicBlock *Header = BasicBlock::Create(Ctx, "frem.loop", &F, After);
    BasicBlock *Body = BasicBlock::Create(Ctx, "frem.reduce", &F, After);
    Before->getTerminator()->eraseFromParent();

    IRBuilder<> B(Before);
    Type *F32 = B.getFloatTy(), *I32 = B.getInt32Ty();
    auto K = [&](float V) { return ConstantFP::get(F32, V); };
    auto Bits = [&](Value *V) { return B.CreateBitCast(V, I32); };
    auto Exponent = [&](Value *V) { return B.CreateLShr(Bits(V), 23); };
    // a mod d for a / d < 2^24 (a, d positive)
    auto Step = [&](Value *A, Value *D) {
      Value *Q = B.CreateUnaryIntrinsic(Intrinsic::trunc, B.CreateFDiv(A, D));
      auto Rem = [&](Value *Q) {
        return B.CreateIntrinsic(Intrinsic::fma, {F32},
                                 {B.CreateFNeg(Q), D, A});
      };
      // With q one too large, a - (q - 1) d is the remainder (below d);
      // with the right q it is at least d. (a - q d < 0 would not show an
      // overestimate whose remainder is flushed.)
      Value *Low = Rem(B.CreateFSub(Q, K(1.0f)));
      return B.CreateSelect(B.CreateFCmpOLT(Low, D), Low, Rem(Q));
    };

    Value *X = B.CreateFPExt(Rem->getOperand(0), F32);
    Value *Y = B.CreateFPExt(Rem->getOperand(1), F32);
    Value *AX = B.CreateUnaryIntrinsic(Intrinsic::fabs, X);
    Value *AY = B.CreateUnaryIntrinsic(Intrinsic::fabs, Y);
    Value *BN = AY;
    if (DenormalB)
      BN = B.CreateSelect(B.CreateFCmpOLT(AY, K(0x1p-126f)),
                          B.CreateFMul(AY, K(0x1p24f)), AY);
    // NaN, or x itself for an infinite y: no reduction.
    Value *NaN = B.CreateOr(B.CreateFCmpUGE(AX, K(INFINITY)),
                            B.CreateFCmpULE(BN, K(0.0f)));
    Value *YInf = B.CreateFCmpOEQ(AY, K(INFINITY));
    Value *Start = B.CreateSelect(B.CreateOr(NaN, YInf), K(0.0f), AX);
    Value *EB = Exponent(BN);
    B.CreateBr(Header);

    B.SetInsertPoint(Header);
    PHINode *A = B.CreatePHI(F32, 2);
    A->addIncoming(Start, Before);
    Value *J = B.CreateSub(B.CreateSub(Exponent(A), EB), B.getInt32(23));
    B.CreateCondBr(B.CreateICmpSGT(J, B.getInt32(0)), Body, After);

    B.SetInsertPoint(Body);
    Value *BK = B.CreateBitCast(B.CreateAdd(Bits(BN), B.CreateShl(J, 23)), F32);
    A->addIncoming(Step(A, BK), Body);
    B.CreateBr(Header);

    B.SetInsertPoint(Rem);
    Value *R = Step(A, BN);
    if (DenormalB)
      R = Step(R, AY);
    R = B.CreateCopySign(R, X);
    R = B.CreateSelect(YInf, X, R);
    R = B.CreateSelect(NaN, K(NAN), R);
    R = B.CreateFPTrunc(R, Rem->getType());
    R->takeName(Rem);
    Rem->replaceAllUsesWith(R);
    Rem->eraseFromParent();
  }
  return !Rems.empty();
}

bool AGXCodeGenPrepare::expandMath(Function &F, const AGXSubtarget &ST) {
  bool Changed = expandPowI(F);
  Changed |= expandFRem(F, ST);
  // An expansion may use functions that are expanded in turn (tan's sin).
  for (;;) {
    SmallVector<IntrinsicInst *, 8> Calls;
    for (Instruction &I : instructions(F)) {
      auto *II = dyn_cast<IntrinsicInst>(&I);
      if (!II)
        continue;
      // With ieee-fp32, AGXStrictFP computes the precise functions exactly.
      bool Approx = (isa<FPMathOperator>(II) && II->hasApproxFunc()) ||
                    ST.hasIEEEFP32();
      // The type of the operand: sincos returns a pair.
      if (II->arg_size() &&
          AGX::canExpandMath(II->getIntrinsicID(),
                             II->getArgOperand(0)->getType(), Approx))
        Calls.push_back(II);
    }
    if (Calls.empty())
      return Changed;
    for (IntrinsicInst *II : Calls) {
      IRBuilder<> B(II);
      B.setFastMathFlags(II->getFastMathFlags());
      SmallVector<Value *, 2> Args(II->args());
      Value *R = AGX::expandMath(B, II->getIntrinsicID(), Args);
      R->takeName(II);
      II->replaceAllUsesWith(R);
      II->eraseFromParent();
    }
    Changed = true;
  }
}

/// With "agx-max-threads-per-threadgroup" at most 128, the threadgroup is one
/// aligned block of 4 SIMD-groups (or less: a block cut short by the end of
/// the threadgroup does not wait for the missing groups). Its block_barrier
/// waits for the same threads as threadgroup_barrier, about 3x faster on G14,
/// and orders their threadgroup and device memory, so the workgroup-scope
/// fences right around the barrier go too.
bool AGXCodeGenPrepare::useBlockBarriers(Function &F, const AGXSubtarget &ST) {
  unsigned Threads =
      F.getFnAttributeAsParsedInteger("agx-max-threads-per-threadgroup", 0);
  if (!ST.hasBlockBarrier() || Threads == 0 || Threads > 128)
    return false;
  SmallVector<IntrinsicInst *, 4> Barriers;
  for (Instruction &I : instructions(F))
    if (auto *II = dyn_cast<IntrinsicInst>(&I);
        II && II->getIntrinsicID() == Intrinsic::agx_threadgroup_barrier)
      Barriers.push_back(II);
  SyncScope::ID Workgroup = F.getContext().getOrInsertSyncScopeID("workgroup");
  auto IsWorkgroupFence = [&](Instruction *I) {
    auto *Fence = dyn_cast_or_null<FenceInst>(I);
    return Fence && Fence->getSyncScopeID() == Workgroup;
  };
  for (IntrinsicInst *II : Barriers) {
    for (Instruction *Fence : {II->getPrevNode(), II->getNextNode()})
      if (IsWorkgroupFence(Fence))
        Fence->eraseFromParent();
    IRBuilder<> B(II);
    B.CreateIntrinsic(Intrinsic::agx_block_barrier, {});
    II->eraseFromParent();
  }
  return !Barriers.empty();
}

static bool isSystemScope(SyncScope::ID Scope, LLVMContext &Ctx) {
  return Scope == SyncScope::System ||
         Scope == Ctx.getOrInsertSyncScopeID("one-as");
}

bool AGX::isHostCoherentAccess(const Instruction &I) {
  unsigned AS;
  if (const auto *L = dyn_cast<LoadInst>(&I)) {
    if (!L->isVolatile() &&
        !(L->isAtomic() && isSystemScope(L->getSyncScopeID(), I.getContext())))
      return false;
    AS = L->getPointerAddressSpace();
  } else if (const auto *S = dyn_cast<StoreInst>(&I)) {
    if (!S->isVolatile() &&
        !(S->isAtomic() && isSystemScope(S->getSyncScopeID(), I.getContext())))
      return false;
    AS = S->getPointerAddressSpace();
  } else {
    return false;
  }
  return AS == AGX::DEVICE_ADDRESS || AS == AGX::GENERIC_ADDRESS;
}

bool AGX::isHostCoherentAccess(const MachineInstr &MI) {
  LLVMContext &Ctx = MI.getMF()->getFunction().getContext();
  for (const MachineMemOperand *MMO : MI.memoperands())
    if (MMO->isVolatile() ||
        (MMO->isAtomic() && isSystemScope(MMO->getSyncScopeID(), Ctx)))
      return true;
  return false;
}

/// Host memory written during a dispatch is seen only by a coherent load
/// after an f59d, and the host sees a store during the dispatch only when it
/// is coherent and followed by an f533 (ptx-re-blockers B7, Apple-M2-Playground
/// 2026-10-16-cpu-gpu-visibility). The coherency bit is set when the access is
/// encoded (AGXMCInstLower); the barriers go here.
bool AGXCodeGenPrepare::fenceHostCoherentAccesses(Function &F) {
  SmallVector<Instruction *, 8> Accesses;
  for (Instruction &I : instructions(F))
    if (AGX::isHostCoherentAccess(I))
      Accesses.push_back(&I);
  if (Accesses.empty())
    return false;
  Module &M = *F.getParent();
  Function *Barrier = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::agx_memory_barrier);
  for (Instruction *I : Accesses) {
    IRBuilder<> B(I);
    if (isa<StoreInst>(I))
      B.SetInsertPoint(I->getNextNode());
    // f59d is memory_barrier 3, 1, 9; f533 is 0, 3, 3.
    unsigned A = isa<LoadInst>(I) ? 3 : 0, Bv = isa<LoadInst>(I) ? 1 : 3,
             C = isa<LoadInst>(I) ? 9 : 3;
    B.CreateCall(Barrier, {B.getInt32(A), B.getInt32(Bv), B.getInt32(C)});
  }
  return true;
}
