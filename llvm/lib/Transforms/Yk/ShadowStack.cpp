//===- ShadowStack.cpp - Pass to add shadow stacks to the AOT module --===//
//
// Add shadow stacks to store variables that may have their references taken.
// Storing such variables on a shadow stack allows AOT to share them with
// compiled traces, and back (i.e. references created inside a trace will still
// be valid when we return from the trace via deoptimisation).
// YKFIXME: This can be optimised by only putting variables on the shadow stack
// that actually have their reference taken.

#include "llvm/Transforms/Yk/ShadowStack.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Yk/LivenessAnalysis.h"

#include <map>

#define DEBUG_TYPE "yk-shadowstack"
#define YK_MT_NEW "yk_mt_new"
#define G_SHADOW_STACK "shadowstack_0"
// The size of the shadow stack. Defaults to 1MB.
// YKFIXME: Make this adjustable by a compiler flag.
#define SHADOW_STACK_SIZE 1000000

using namespace llvm;

namespace llvm {
void initializeYkShadowStackPass(PassRegistry &);
} // namespace llvm

namespace {
class YkShadowStack : public ModulePass {
public:
  static char ID;
  YkShadowStack() : ModulePass(ID) {
    initializeYkShadowStackPass(*PassRegistry::getPassRegistry());
  }

  // Checks whether the given instruction is the alloca of the call to
  // `yk_mt_new`.
  bool isYkMTNewAlloca(Instruction *I) {
    for (User *U : I->users()) {
      if (U && isa<StoreInst>(U)) {
        Value *V = cast<StoreInst>(U)->getValueOperand();
        if (isa<CallInst>(V)) {
          CallInst *CI = cast<CallInst>(V);
          if (CI->isInlineAsm())
            return false;
          if (!CI->getCalledFunction())
            return false;
          return (CI->getCalledFunction()->getName() == YK_MT_NEW);
        }
      }
    }
    return false;
  }

  bool runOnModule(Module &M) override {
    LLVMContext &Context = M.getContext();

    DataLayout DL(&M);
    Type *Int8Ty = Type::getInt8Ty(Context);
    Type *Int32Ty = Type::getInt32Ty(Context);
    Type *PointerSizedIntTy = DL.getIntPtrType(Context);
    Type *Int8PtrTy = Type::getInt8PtrTy(Context);

    // Create a global variable which will store the pointer to the heap memory
    // allocated for the shadow stack.
    Constant *GShadowStackPtr = M.getOrInsertGlobal(G_SHADOW_STACK, Int8PtrTy);
    GlobalVariable *GVar = M.getNamedGlobal(G_SHADOW_STACK);
    GVar->setInitializer(
        ConstantPointerNull::get(cast<PointerType>(Int8PtrTy)));

    // We only need to create one shadow stack per module so we'll do this
    // inside the module's entry point.
    // YKFIXME: Investigate languages that don't have/use main as the first
    // entry point.
    Function *Main = M.getFunction("main");
    if (Main == nullptr) {
      Context.emitError(
          "Unable to add shadow stack: could not find \"main\" function!");
      return false;
    }
    Instruction *First = Main->getEntryBlock().getFirstNonPHI();
    IRBuilder<> Builder(First);

    // Now create some memory on the heap for the shadow stack.
    FunctionCallee MF =
        M.getOrInsertFunction("malloc", Int8PtrTy, PointerSizedIntTy);
    CallInst *Malloc = Builder.CreateCall(
        MF, {ConstantInt::get(PointerSizedIntTy, SHADOW_STACK_SIZE)}, "");
    Builder.CreateStore(Malloc, GShadowStackPtr);

    Value *SSPtr;
    for (Function &F : M) {
      if (F.empty()) // skip declarations.
        continue;

      if (&F != Main) {
        // At the top of each function in the module, load the heap pointer
        // from the global shadow stack variable.
        Builder.SetInsertPoint(F.getEntryBlock().getFirstNonPHI());
        SSPtr = Builder.CreateLoad(Int8PtrTy, GShadowStackPtr);
      } else {
        SSPtr = cast<Value>(Malloc);
      }

      size_t Offset = 0;
      // Remember which allocas were replaced, so we can remove them later in
      // one swoop. Removing them here messes up the loop.
      std::vector<Instruction *> RemoveAllocas;
      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          if (isa<AllocaInst>(I)) {
            // Replace allocas with pointers into the shadow stack.
            AllocaInst &AI = cast<AllocaInst>(I);
            if (isYkMTNewAlloca(&AI)) {
              // The variable created by `yk_mt_new` will never be traced, so
              // there's no need to store it on the shadow stack.
              continue;
            }
            if (isa<StructType>(AI.getAllocatedType())) {
              StructType *ST = cast<StructType>(AI.getAllocatedType());
              // Some yk specific variables that will never be traced and thus
              // can live happily on the normal stack.
              // YKFIXME: This is somewhat fragile since `struct.YkLocation` is
              // a name given by LLVM which could theoretically change. Luckily,
              // this should all go away once we only move variables to the
              // shadowstack that have their reference taken.
              if (!ST->isLiteral()) {
                if (ST->getName() == "YkCtrlPointVars" ||
                    ST->getName() == "struct.YkLocation") {
                  continue;
                }
              }
            }
            Builder.SetInsertPoint(&I);
            auto AllocaSizeInBits = AI.getAllocationSizeInBits(DL);
            if (!AllocaSizeInBits) {
              // YKFIXME: Deal with functions where the stack size isn't know at
              // compile time, e.g. when `alloca` is used.
              Context.emitError("Unable to add shadow stack: function has "
                                "dynamically sized stack!");
              return false;
            }
            // Calculate this `AllocaInst`s size, aligning its pointer if
            // necessary, and create a replacement pointer into the shadow
            // stack.
            size_t AllocaSize = *AllocaSizeInBits / sizeof(uintptr_t);
            size_t Align = AI.getAlign().value();
            Offset = int((Offset + (Align - 1)) / Align) * Align;
            GetElementPtrInst *GEP = GetElementPtrInst::Create(
                Int8Ty, SSPtr, {ConstantInt::get(Int32Ty, Offset)}, "",
                cast<Instruction>(&AI));
            Builder.SetInsertPoint(GEP);
            Builder.CreateBitCast(GEP, AI.getAllocatedType()->getPointerTo());
            cast<Value>(I).replaceAllUsesWith(GEP);
            RemoveAllocas.push_back(cast<Instruction>(&AI));
            Offset += AllocaSize;
          } else if (isa<CallInst>(I)) {
            // When we see a call, we need make space for a new stack frame. We
            // do this by simply adjusting the pointer stored in the global
            // shadow stack. When the function returns the global is reset. This
            // is similar to how the RSP is adjusted inside the
            // prologue/epilogue of a function, but here the prologue/epilogue
            // are handled by the caller.
            CallInst &CI = cast<CallInst>(I);
            if (&CI == Malloc) {
              // Don't do this for the `malloc` that created the shadow stack.
              continue;
            }
            // Inline asm can't be traced.
            if (CI.isInlineAsm()) {
              continue;
            }

            // YKFIXME: Skip functions that are marked with `yk_outline`
            // (as those won't be traced and thus don't require a shadow
            // stack).
            // YKFIXME: Skip functions (direct or indirect) that we don't have
            // IR for.

            if (CI.getCalledFunction()) {
              // Skip some known intrinsics. YKFIXME: Is there a more general
              // solution, e.g. skip all intrinsics?
              if (CI.getCalledFunction()->getName() ==
                  "llvm.experimental.stackmap") {
                continue;
              } else if (CI.getCalledFunction()->getName() ==
                         "llvm.dbg.declare") {
                continue;
              }
            }

            // Adjust shadow stack pointer before a call, and reset it back to
            // its previous value upon returning. Make sure to align the shadow
            // stack to a 16 byte boundary before calling, as required by the
            // calling convention.
#ifdef __x86_64__
            Offset = int((Offset + (16 - 1)) / 16) * 16;
#else
#error unknown platform
#endif
            GetElementPtrInst *GEP = GetElementPtrInst::Create(
                Int8Ty, SSPtr, {ConstantInt::get(Int32Ty, Offset)}, "", &I);
            Builder.SetInsertPoint(&I);
            Builder.CreateStore(GEP, GShadowStackPtr);
            Builder.SetInsertPoint(I.getNextNonDebugInstruction());
            Builder.CreateStore(SSPtr, GShadowStackPtr);
          }
        }
      }
      for (Instruction *I : RemoveAllocas) {
        I->removeFromParent();
      }
      RemoveAllocas.clear();
    }

#ifndef NDEBUG
    // Our pass runs after LLVM normally does its verify pass. In debug builds
    // we run it again to check that our pass is generating valid IR.
    if (verifyModule(M, &errs())) {
      Context.emitError("ShadowStack insertion pass generated invalid IR!");
      return false;
    }
#endif
    return true;
  }
};
} // namespace

char YkShadowStack::ID = 0;
INITIALIZE_PASS(YkShadowStack, DEBUG_TYPE, "yk shadowstack", false, false)

ModulePass *llvm::createYkShadowStackPass() { return new YkShadowStack(); }
