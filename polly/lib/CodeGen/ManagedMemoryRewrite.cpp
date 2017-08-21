//===---- ManagedMemoryRewrite.cpp - Rewrite global & malloc'd memory -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Take a module and rewrite:
// 1. `malloc` -> `polly_mallocManaged`
// 2. `free` -> `polly_freeManaged`
// 3. global arrays with initializers -> global arrays that are initialized
//                                       with a constructor call to
//                                       `polly_mallocManaged`.
//
//===----------------------------------------------------------------------===//

#include "polly/CodeGen/CodeGeneration.h"
#include "polly/CodeGen/IslAst.h"
#include "polly/CodeGen/IslNodeBuilder.h"
#include "polly/CodeGen/PPCGCodeGeneration.h"
#include "polly/CodeGen/Utils.h"
#include "polly/DependenceInfo.h"
#include "polly/LinkAllPasses.h"
#include "polly/Options.h"
#include "polly/ScopDetection.h"
#include "polly/ScopInfo.h"
#include "polly/Support/SCEVValidator.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/ScalarEvolutionAliasAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

static cl::opt<bool> RewriteAllocas(
    "polly-acc-rewrite-allocas",
    cl::desc(
        "Ask the managed memory rewriter to also rewrite alloca instructions"),
    cl::Hidden, cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool> IgnoreLinkageForGlobals(
    "polly-acc-rewrite-ignore-linkage-for-globals",
    cl::desc(
        "By default, we only rewrite globals with internal linkage. This flag "
        "enables rewriting of globals regardless of linkage"),
    cl::Hidden, cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

#define DEBUG_TYPE "polly-acc-rewrite-managed-memory"
namespace {

static llvm::Function *getOrCreatePollyMallocManaged(Module &M) {
  const char *Name = "polly_mallocManaged";
  Function *F = M.getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    PollyIRBuilder Builder(M.getContext());
    // TODO: How do I get `size_t`? I assume from DataLayout?
    FunctionType *Ty = FunctionType::get(Builder.getInt8PtrTy(),
                                         {Builder.getInt64Ty()}, false);
    F = Function::Create(Ty, Linkage, Name, &M);
  }

  return F;
}

static llvm::Function *getOrCreatePollyFreeManaged(Module &M) {
  const char *Name = "polly_freeManaged";
  Function *F = M.getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    PollyIRBuilder Builder(M.getContext());
    // TODO: How do I get `size_t`? I assume from DataLayout?
    FunctionType *Ty =
        FunctionType::get(Builder.getVoidTy(), {Builder.getInt8PtrTy()}, false);
    F = Function::Create(Ty, Linkage, Name, &M);
  }

  return F;
}

// Expand a constant expression `Cur`, which is used at instruction `Parent`
// at index `index`.
// Since a constant expression can expand to multiple instructions, store all
// the expands into a set called `Expands`.
// Note that this goes inorder on the constant expression tree.
// A * ((B * D) + C)
// will be processed with first A, then B * D, then B, then D, and then C.
// Though ConstantExprs are not treated as "trees" but as DAGs, since you can
// have something like this:
//    *
//   /  \
//   \  /
//    (D)
//
// For the purposes of this expansion, we expand the two occurences of D
// separately. Therefore, we expand the DAG into the tree:
//  *
// / \
// D  D
// TODO: We don't _have_to do this, but this is the simplest solution.
// We can write a solution that keeps track of which constants have been
// already expanded.
static void expandConstantExpr(ConstantExpr *Cur, PollyIRBuilder &Builder,
                               Instruction *Parent, int index,
                               SmallPtrSet<Instruction *, 4> &Expands) {
  assert(Cur && "invalid constant expression passed");
  Instruction *I = Cur->getAsInstruction();
  assert(I && "unable to convert ConstantExpr to Instruction");

  DEBUG(dbgs() << "Expanding ConstantExpression: (" << *Cur
               << ") in Instruction: (" << *I << ")\n";);

  // Invalidate `Cur` so that no one after this point uses `Cur`. Rather,
  // they should mutate `I`.
  Cur = nullptr;

  Expands.insert(I);
  Parent->setOperand(index, I);

  // The things that `Parent` uses (its operands) should be created
  // before `Parent`.
  Builder.SetInsertPoint(Parent);
  Builder.Insert(I);

  for (unsigned i = 0; i < I->getNumOperands(); i++) {
    Value *Op = I->getOperand(i);
    assert(isa<Constant>(Op) && "constant must have a constant operand");

    if (ConstantExpr *CExprOp = dyn_cast<ConstantExpr>(Op))
      expandConstantExpr(CExprOp, Builder, I, i, Expands);
  }
}

// Edit all uses of `OldVal` to NewVal` in `Inst`. This will rewrite
// `ConstantExpr`s that are used in the `Inst`.
// Note that `replaceAllUsesWith` is insufficient for this purpose because it
// does not rewrite values in `ConstantExpr`s.
static void rewriteOldValToNew(Instruction *Inst, Value *OldVal, Value *NewVal,
                               PollyIRBuilder &Builder) {

  // This contains a set of instructions in which OldVal must be replaced.
  // We start with `Inst`, and we fill it up with the expanded `ConstantExpr`s
  // from `Inst`s arguments.
  // We need to go through this process because `replaceAllUsesWith` does not
  // actually edit `ConstantExpr`s.
  SmallPtrSet<Instruction *, 4> InstsToVisit = {Inst};

  // Expand all `ConstantExpr`s and place it in `InstsToVisit`.
  for (unsigned i = 0; i < Inst->getNumOperands(); i++) {
    Value *Operand = Inst->getOperand(i);
    if (ConstantExpr *ValueConstExpr = dyn_cast<ConstantExpr>(Operand))
      expandConstantExpr(ValueConstExpr, Builder, Inst, i, InstsToVisit);
  }

  // Now visit each instruction and use `replaceUsesOfWith`. We know that
  // will work because `I` cannot have any `ConstantExpr` within it.
  for (Instruction *I : InstsToVisit)
    I->replaceUsesOfWith(OldVal, NewVal);
}

// Given a value `Current`, return all Instructions that may contain `Current`
// in an expression.
// We need this auxiliary function, because if we have a
// `Constant` that is a user of `V`, we need to recurse into the
// `Constant`s uses to gather the root instruciton.
static void getInstructionUsersOfValue(Value *V,
                                       SmallVector<Instruction *, 4> &Owners) {
  if (auto *I = dyn_cast<Instruction>(V)) {
    Owners.push_back(I);
  } else {
    // Anything that is a `User` must be a constant or an instruction.
    auto *C = cast<Constant>(V);
    for (Use &CUse : C->uses())
      getInstructionUsersOfValue(CUse.getUser(), Owners);
  }
}

static void
replaceGlobalArray(Module &M, const DataLayout &DL, GlobalVariable &Array,
                   SmallPtrSet<GlobalVariable *, 4> &ReplacedGlobals) {
  // We only want arrays.
  ArrayType *ArrayTy = dyn_cast<ArrayType>(Array.getType()->getElementType());
  if (!ArrayTy)
    return;
  Type *ElemTy = ArrayTy->getElementType();
  PointerType *ElemPtrTy = ElemTy->getPointerTo();

  // We only wish to replace arrays that are visible in the module they
  // inhabit. Otherwise, our type edit from [T] to T* would be illegal across
  // modules.
  const bool OnlyVisibleInsideModule = Array.hasPrivateLinkage() ||
                                       Array.hasInternalLinkage() ||
                                       IgnoreLinkageForGlobals;
  if (!OnlyVisibleInsideModule) {
    DEBUG(dbgs() << "Not rewriting (" << Array
                 << ") to managed memory "
                    "because it could be visible externally. To force rewrite, "
                    "use -polly-acc-rewrite-ignore-linkage-for-globals.\n");
    return;
  }

  if (!Array.hasInitializer() ||
      !isa<ConstantAggregateZero>(Array.getInitializer())) {
    DEBUG(dbgs() << "Not rewriting (" << Array
                 << ") to managed memory "
                    "because it has an initializer which is "
                    "not a zeroinitializer.\n");
    return;
  }

  // At this point, we have committed to replacing this array.
  ReplacedGlobals.insert(&Array);

  std::string NewName = Array.getName();
  NewName += ".toptr";
  GlobalVariable *ReplacementToArr =
      cast<GlobalVariable>(M.getOrInsertGlobal(NewName, ElemPtrTy));
  ReplacementToArr->setInitializer(ConstantPointerNull::get(ElemPtrTy));

  Function *PollyMallocManaged = getOrCreatePollyMallocManaged(M);
  std::string FnName = Array.getName();
  FnName += ".constructor";
  PollyIRBuilder Builder(M.getContext());
  FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), false);
  const GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
  Function *F = Function::Create(Ty, Linkage, FnName, &M);
  BasicBlock *Start = BasicBlock::Create(M.getContext(), "entry", F);
  Builder.SetInsertPoint(Start);

  int ArraySizeInt = DL.getTypeAllocSizeInBits(ArrayTy) / 8;
  Value *ArraySize = Builder.getInt64(ArraySizeInt);
  ArraySize->setName("array.size");

  Value *AllocatedMemRaw =
      Builder.CreateCall(PollyMallocManaged, {ArraySize}, "mem.raw");
  Value *AllocatedMemTyped =
      Builder.CreatePointerCast(AllocatedMemRaw, ElemPtrTy, "mem.typed");
  Builder.CreateStore(AllocatedMemTyped, ReplacementToArr);
  Builder.CreateRetVoid();

  const int Priority = 0;
  appendToGlobalCtors(M, F, Priority, ReplacementToArr);

  SmallVector<Instruction *, 4> ArrayUserInstructions;
  // Get all instructions that use array. We need to do this weird thing
  // because `Constant`s that contain this array neeed to be expanded into
  // instructions so that we can replace their parameters. `Constant`s cannot
  // be edited easily, so we choose to convert all `Constant`s to
  // `Instruction`s and handle all of the uses of `Array` uniformly.
  for (Use &ArrayUse : Array.uses())
    getInstructionUsersOfValue(ArrayUse.getUser(), ArrayUserInstructions);

  for (Instruction *UserOfArrayInst : ArrayUserInstructions) {

    Builder.SetInsertPoint(UserOfArrayInst);
    // <ty>** -> <ty>*
    Value *ArrPtrLoaded = Builder.CreateLoad(ReplacementToArr, "arrptr.load");
    // <ty>* -> [ty]*
    Value *ArrPtrLoadedBitcasted = Builder.CreateBitCast(
        ArrPtrLoaded, ArrayTy->getPointerTo(), "arrptr.bitcast");
    rewriteOldValToNew(UserOfArrayInst, &Array, ArrPtrLoadedBitcasted, Builder);
  }
}

// We return all `allocas` that may need to be converted to a call to
// cudaMallocManaged.
static void getAllocasToBeManaged(Function &F,
                                  SmallSet<AllocaInst *, 4> &Allocas) {
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      auto *Alloca = dyn_cast<AllocaInst>(&I);
      if (!Alloca)
        continue;
      DEBUG(dbgs() << "Checking if (" << *Alloca << ") may be captured: ");

      if (PointerMayBeCaptured(Alloca, /* ReturnCaptures */ false,
                               /* StoreCaptures */ true)) {
        Allocas.insert(Alloca);
        DEBUG(dbgs() << "YES (captured).\n");
      } else {
        DEBUG(dbgs() << "NO (not captured).\n");
      }
    }
  }
}

static void rewriteAllocaAsManagedMemory(AllocaInst *Alloca,
                                         const DataLayout &DL) {
  DEBUG(dbgs() << "rewriting: (" << *Alloca << ") to managed mem.\n");
  Module *M = Alloca->getModule();
  assert(M && "Alloca does not have a module");

  PollyIRBuilder Builder(M->getContext());
  Builder.SetInsertPoint(Alloca);

  Value *MallocManagedFn = getOrCreatePollyMallocManaged(*Alloca->getModule());
  const int Size = DL.getTypeAllocSize(Alloca->getType()->getElementType());
  Value *SizeVal = Builder.getInt64(Size);
  Value *RawManagedMem = Builder.CreateCall(MallocManagedFn, {SizeVal});
  Value *Bitcasted = Builder.CreateBitCast(RawManagedMem, Alloca->getType());

  Function *F = Alloca->getFunction();
  assert(F && "Alloca has invalid function");

  Bitcasted->takeName(Alloca);
  Alloca->replaceAllUsesWith(Bitcasted);
  Alloca->eraseFromParent();

  for (BasicBlock &BB : *F) {
    ReturnInst *Return = dyn_cast<ReturnInst>(BB.getTerminator());
    if (!Return)
      continue;
    Builder.SetInsertPoint(Return);

    Value *FreeManagedFn = getOrCreatePollyFreeManaged(*M);
    Builder.CreateCall(FreeManagedFn, {RawManagedMem});
  }
}

// Replace all uses of `Old` with `New`, even inside `ConstantExpr`.
//
// `replaceAllUsesWith` does replace values in `ConstantExpr`. This function
// actually does replace it in `ConstantExpr`. The caveat is that if there is
// a use that is *outside* a function (say, at global declarations), we fail.
// So, this is meant to be used on values which we know will only be used
// within functions.
//
// This process works by looking through the uses of `Old`. If it finds a
// `ConstantExpr`, it recursively looks for the owning instruction.
// Then, it expands all the `ConstantExpr` to instructions and replaces
// `Old` with `New` in the expanded instructions.
static void replaceAllUsesAndConstantUses(Value *Old, Value *New,
                                          PollyIRBuilder &Builder) {
  SmallVector<Instruction *, 4> UserInstructions;
  // Get all instructions that use array. We need to do this weird thing
  // because `Constant`s that contain this array neeed to be expanded into
  // instructions so that we can replace their parameters. `Constant`s cannot
  // be edited easily, so we choose to convert all `Constant`s to
  // `Instruction`s and handle all of the uses of `Array` uniformly.
  for (Use &ArrayUse : Old->uses())
    getInstructionUsersOfValue(ArrayUse.getUser(), UserInstructions);

  for (Instruction *I : UserInstructions)
    rewriteOldValToNew(I, Old, New, Builder);
}

class ManagedMemoryRewritePass : public ModulePass {
public:
  static char ID;
  GPUArch Architecture;
  GPURuntime Runtime;

  ManagedMemoryRewritePass() : ModulePass(ID) {}
  virtual bool runOnModule(Module &M) {
    const DataLayout &DL = M.getDataLayout();

    Function *Malloc = M.getFunction("malloc");

    if (Malloc) {
      PollyIRBuilder Builder(M.getContext());
      Function *PollyMallocManaged = getOrCreatePollyMallocManaged(M);
      assert(PollyMallocManaged && "unable to create polly_mallocManaged");

      replaceAllUsesAndConstantUses(Malloc, PollyMallocManaged, Builder);
      Malloc->eraseFromParent();
    }

    Function *Free = M.getFunction("free");

    if (Free) {
      PollyIRBuilder Builder(M.getContext());
      Function *PollyFreeManaged = getOrCreatePollyFreeManaged(M);
      assert(PollyFreeManaged && "unable to create polly_freeManaged");

      replaceAllUsesAndConstantUses(Free, PollyFreeManaged, Builder);
      Free->eraseFromParent();
    }

    SmallPtrSet<GlobalVariable *, 4> GlobalsToErase;
    for (GlobalVariable &Global : M.globals())
      replaceGlobalArray(M, DL, Global, GlobalsToErase);
    for (GlobalVariable *G : GlobalsToErase)
      G->eraseFromParent();

    // Rewrite allocas to cudaMallocs if we are asked to do so.
    if (RewriteAllocas) {
      SmallSet<AllocaInst *, 4> AllocasToBeManaged;
      for (Function &F : M.functions())
        getAllocasToBeManaged(F, AllocasToBeManaged);

      for (AllocaInst *Alloca : AllocasToBeManaged)
        rewriteAllocaAsManagedMemory(Alloca, DL);
    }

    return true;
  }
};

} // namespace
char ManagedMemoryRewritePass::ID = 42;

Pass *polly::createManagedMemoryRewritePassPass(GPUArch Arch,
                                                GPURuntime Runtime) {
  ManagedMemoryRewritePass *pass = new ManagedMemoryRewritePass();
  pass->Runtime = Runtime;
  pass->Architecture = Arch;
  return pass;
}

INITIALIZE_PASS_BEGIN(
    ManagedMemoryRewritePass, "polly-acc-rewrite-managed-memory",
    "Polly - Rewrite all allocations in heap & data section to managed memory",
    false, false)
INITIALIZE_PASS_DEPENDENCY(PPCGCodeGeneration);
INITIALIZE_PASS_DEPENDENCY(DependenceInfo);
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass);
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass);
INITIALIZE_PASS_DEPENDENCY(RegionInfoPass);
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass);
INITIALIZE_PASS_DEPENDENCY(ScopDetectionWrapperPass);
INITIALIZE_PASS_END(
    ManagedMemoryRewritePass, "polly-acc-rewrite-managed-memory",
    "Polly - Rewrite all allocations in heap & data section to managed memory",
    false, false)
