#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <string>

using namespace llvm;

namespace {
// Hello2 - The second implementation with getAnalysisUsage implemented.
struct Cfcss : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  Cfcss() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {

    for (Function &F : M) {

      if (F.getName() != "__cfcss_error" && F.getName() != "printf" &&
          F.getName() != "exit") {

        IRBuilder<> Builder((F.begin())->getFirstNonPHI());

        GlobalVariable *GV = new llvm::GlobalVariable(
            *F.getParent(), IntegerType::getInt32Ty((F.getContext())), false,
            llvm::GlobalValue::InternalLinkage, Builder.getInt32(0), "G");

        GlobalVariable *Dg = new llvm::GlobalVariable(
            *F.getParent(), IntegerType::getInt32Ty((F.getContext())), false,
            llvm::GlobalValue::InternalLinkage, Builder.getInt32(0), "D");

        //While iterating over BB we might get new BB and it is not-exiting
        SmallVector<llvm::BasicBlock*> VBasicBlock;
        llvm::DenseMap<BasicBlock *, int> SigMap;
        llvm::DenseMap<BasicBlock *, int> Dsig;
        llvm::DenseMap<BasicBlock *, int> Diffsig;
        llvm::DenseMap<BasicBlock *, Instruction *> BrIMap;
        int SigCount = 1;
        BasicBlock *Pbb;
        LLVMContext &Ctx = M.getContext();
        FunctionCallee ErrorFunc =
            M.getOrInsertFunction("__cfcss_error", Builder.getVoidTy());
        SmallVector<Value *> Arguments;

        // Checking the branch/return instruction of each BB and storing it into
        // BrIMap.
        for (BasicBlock &BB : F) {
          for (Instruction &I : BB) {
            if (isa<BranchInst, ReturnInst>(I)) {
              BrIMap[&BB] = &I;
              break;
            }
          }
        }

        // Calculating Signature(s) of each BB and storing it into SigMap.
        for (BasicBlock &BB : F) {
          SigMap[&BB] = SigCount;
          SigCount++;
          VBasicBlock.push_back(&BB);
        }

        Builder.CreateStore(Builder.getInt32(1), GV);

        // Calculating Dsig, by xoring Source (S) and Destination (sd) sig.
        for (BasicBlock &BB : F) {
          if (BB.hasNPredecessors(1)) {
            Dsig[&BB] = SigMap[&BB] ^ SigMap[BB.getSinglePredecessor()];
          }
          // Calculating Dsig of BB of one predecessors , if BB contains 2
          // predecessors.
          if (BB.hasNPredecessorsOrMore(2)) {
            for (BasicBlock *Pred : predecessors(&BB)) {
              Pbb = Pred;
              Dsig[&BB] = SigMap[&BB] ^ SigMap[Pbb];
              break;
            }
            // Calculating Diffsig of each , if BB contains 2 predecessors.
            for (BasicBlock *Pred : predecessors(&BB)) {
              Builder.SetInsertPoint((Pred)->getFirstNonPHI());
              Diffsig[Pred] = SigMap[Pbb] ^ SigMap[Pred];
              Builder.CreateStore(Builder.getInt32(Diffsig[Pred]), Dg);
            }
          }
        }


        // Creating a new BB to emit errors.
        BasicBlock* ErrorBlock = BasicBlock::Create(Ctx, "ErrorBlock", &F);
        Builder.SetInsertPoint(ErrorBlock);
        Builder.CreateCall(ErrorFunc); //To display error message
        Value *Rzero=Builder.getInt32(0);
        Builder.CreateRet(Rzero);


        // Calculating G, and comparing it with source Signature by calling
        // error function.
        for (BasicBlock* BB : VBasicBlock) {

          // G1=s1
          // if it has 0 predecessor then no need to call error function.as G{i}
          // and s(i) are initialized to same value.

          // G=Gs^dsig
          if (BB->hasNPredecessors(1)) {
            Builder.SetInsertPoint((BB)->getFirstNonPHI());
            LoadInst *LI = Builder.CreateLoad(Builder.getInt32Ty(), GV);
            Value *Diff = Builder.CreateXor(LI, Dsig[BB]);
            Builder.CreateStore(Diff, GV);

            // Value *Args[] = {Diff, Builder.getInt32(SigMap[&BB])};
            // Builder.SetInsertPoint(BrIMap[&BB]);
            // Builder.CreateCall(ErrorFunc, Args);
            Value *Fail=Builder.CreateICmpNE(Diff,Builder.getInt32(SigMap[BB]), "failure" );
            BasicBlock *Dd=(BB)->splitBasicBlock(dyn_cast<Instruction>(Fail)->getNextNode(), "split");
            (BB->getTerminator())->eraseFromParent();
            Builder.SetInsertPoint(BB);
            Builder.CreateCondBr(Fail,ErrorBlock, Dd);
          }
          // G=Gs^dsig; G=G^D
          if (BB->hasNPredecessorsOrMore(2)) {
            Builder.SetInsertPoint((BB)->getFirstNonPHI());
            LoadInst *LI = Builder.CreateLoad(Builder.getInt32Ty(), GV);
            LoadInst *DI = Builder.CreateLoad(Builder.getInt32Ty(), Dg);
            Value *Diff = Builder.CreateXor(LI, Dsig[BB]);
            Value *Diff1 = Builder.CreateXor(Diff, DI);
            Builder.CreateStore(Diff1, GV);

            // Value *Args[] = {Diff1, Builder.getInt32(SigMap[&BB])};
            // Builder.SetInsertPoint(BrIMap[&BB]);
            // Builder.CreateCall(ErrorFunc, Args);
            Value *Fail=Builder.CreateICmpNE(Diff1,Builder.getInt32(SigMap[BB]), "failure" );
            BasicBlock *Dd=(BB)->splitBasicBlock(dyn_cast<Instruction>(Fail)->getNextNode(), "split");
            (BB->getTerminator())->eraseFromParent();
            Builder.SetInsertPoint(BB);
            Builder.CreateCondBr(Fail,ErrorBlock, Dd);
          }
        }
      }
    }

    return false;
  }

  // We don't modify the program, so we preserve all analyses.
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};
} // namespace

char Cfcss::ID = 0;
static RegisterPass<Cfcss> Y("cfcss",
                             "Cfcss Pass (with getAnalysisUsage implemented)");