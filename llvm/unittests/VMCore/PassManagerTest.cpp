//===- llvm/unittest/VMCore/PassManager.cpp - Constants unit tests ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/LLVMContext.h"
#include "llvm/PassManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/CallGraphSCCPass.h"
#include "llvm/DataLayout.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Constants.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Function.h"
#include "llvm/CallingConv.h"
#include "llvm/BasicBlock.h"
#include "llvm/Instructions.h"
#include "llvm/InlineAsm.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/PassManager.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace llvm {
  void initializeModuleNDMPass(PassRegistry&);
  void initializeFPassPass(PassRegistry&);
  void initializeCGPassPass(PassRegistry&);
  void initializeLPassPass(PassRegistry&);
  void initializeBPassPass(PassRegistry&);

  namespace {
    // ND = no deps
    // NM = no modifications
    struct ModuleNDNM: public ModulePass {
    public:
      static char run;
      static char ID;
      ModuleNDNM() : ModulePass(ID) { }
      virtual bool runOnModule(Module &M) {
        run++;
        return false;
      }
      virtual void getAnalysisUsage(AnalysisUsage &AU) const {
        AU.setPreservesAll();
      }
    };
    char ModuleNDNM::ID=0;
    char ModuleNDNM::run=0;

    struct ModuleNDM : public ModulePass {
    public:
      static char run;
      static char ID;
      ModuleNDM() : ModulePass(ID) {}
      virtual bool runOnModule(Module &M) {
        run++;
        return true;
      }
    };
    char ModuleNDM::ID=0;
    char ModuleNDM::run=0;

    struct ModuleNDM2 : public ModulePass {
    public:
      static char run;
      static char ID;
      ModuleNDM2() : ModulePass(ID) {}
      virtual bool runOnModule(Module &M) {
        run++;
        return true;
      }
    };
    char ModuleNDM2::ID=0;
    char ModuleNDM2::run=0;

    struct ModuleDNM : public ModulePass {
    public:
      static char run;
      static char ID;
      ModuleDNM() : ModulePass(ID) {
        initializeModuleNDMPass(*PassRegistry::getPassRegistry());
      }
      virtual bool runOnModule(Module &M) {
        EXPECT_TRUE(getAnalysisIfAvailable<DataLayout>());
        run++;
        return false;
      }
      virtual void getAnalysisUsage(AnalysisUsage &AU) const {
        AU.addRequired<ModuleNDM>();
        AU.setPreservesAll();
      }
    };
    char ModuleDNM::ID=0;
    char ModuleDNM::run=0;

    template<typename P>
    struct PassTestBase : public P {
    protected:
      static int runc;
      static bool initialized;
      static bool finalized;
      int allocated;
      void run() {
        EXPECT_TRUE(initialized);
        EXPECT_FALSE(finalized);
        EXPECT_EQ(0, allocated);
        allocated++;
        runc++;
      }
    public:
      static char ID;
      static void finishedOK(int run) {
        EXPECT_GT(runc, 0);
        EXPECT_TRUE(initialized);
        EXPECT_TRUE(finalized);
        EXPECT_EQ(run, runc);
      }
      PassTestBase() : P(ID), allocated(0) {
        initialized = false;
        finalized = false;
        runc = 0;
      }

      virtual void releaseMemory() {
        EXPECT_GT(runc, 0);
        EXPECT_GT(allocated, 0);
        allocated--;
      }
    };
    template<typename P> char PassTestBase<P>::ID;
    template<typename P> int PassTestBase<P>::runc;
    template<typename P> bool PassTestBase<P>::initialized;
    template<typename P> bool PassTestBase<P>::finalized;

    template<typename T, typename P>
    struct PassTest : public PassTestBase<P> {
    public:
      virtual bool doInitialization(T &t) {
        EXPECT_FALSE(PassTestBase<P>::initialized);
        PassTestBase<P>::initialized = true;
        return false;
      }
      virtual bool doFinalization(T &t) {
        EXPECT_FALSE(PassTestBase<P>::finalized);
        PassTestBase<P>::finalized = true;
        EXPECT_EQ(0, PassTestBase<P>::allocated);
        return false;
      }
    };

    struct CGPass : public PassTest<CallGraph, CallGraphSCCPass> {
    public:
      CGPass() {
        initializeCGPassPass(*PassRegistry::getPassRegistry());
      }
      virtual bool runOnSCC(CallGraphSCC &SCMM) {
        EXPECT_TRUE(getAnalysisIfAvailable<DataLayout>());
        run();
        return false;
      }
    };

    struct FPass : public PassTest<Module, FunctionPass> {
    public:
      virtual bool runOnFunction(Function &F) {
        // FIXME: PR4112
        // EXPECT_TRUE(getAnalysisIfAvailable<DataLayout>());
        run();
        return false;
      }
    };

    struct LPass : public PassTestBase<LoopPass> {
    private:
      static int initcount;
      static int fincount;
    public:
      LPass() {
        initializeLPassPass(*PassRegistry::getPassRegistry());
        initcount = 0; fincount=0;
        EXPECT_FALSE(initialized);
      }
      static void finishedOK(int run, int finalized) {
        PassTestBase<LoopPass>::finishedOK(run);
        EXPECT_EQ(run, initcount);
        EXPECT_EQ(finalized, fincount);
      }
      virtual bool doInitialization(Loop* L, LPPassManager &LPM) {
        initialized = true;
        initcount++;
        return false;
      }
      virtual bool runOnLoop(Loop *L, LPPassManager &LPM) {
        EXPECT_TRUE(getAnalysisIfAvailable<DataLayout>());
        run();
        return false;
      }
      virtual bool doFinalization() {
        fincount++;
        finalized = true;
        return false;
      }
    };
    int LPass::initcount=0;
    int LPass::fincount=0;

    struct BPass : public PassTestBase<BasicBlockPass> {
    private:
      static int inited;
      static int fin;
    public:
      static void finishedOK(int run, int N) {
        PassTestBase<BasicBlockPass>::finishedOK(run);
        EXPECT_EQ(inited, N);
        EXPECT_EQ(fin, N);
      }
      BPass() {
        inited = 0;
        fin = 0;
      }
      virtual bool doInitialization(Module &M) {
        EXPECT_FALSE(initialized);
        initialized = true;
        return false;
      }
      virtual bool doInitialization(Function &F) {
        inited++;
        return false;
      }
      virtual bool runOnBasicBlock(BasicBlock &BB) {
        EXPECT_TRUE(getAnalysisIfAvailable<DataLayout>());
        run();
        return false;
      }
      virtual bool doFinalization(Function &F) {
        fin++;
        return false;
      }
      virtual bool doFinalization(Module &M) {
        EXPECT_FALSE(finalized);
        finalized = true;
        EXPECT_EQ(0, allocated);
        return false;
      }
    };
    int BPass::inited=0;
    int BPass::fin=0;

    struct OnTheFlyTest: public ModulePass {
    public:
      static char ID;
      OnTheFlyTest() : ModulePass(ID) {
        initializeFPassPass(*PassRegistry::getPassRegistry());
      }
      virtual bool runOnModule(Module &M) {
        EXPECT_TRUE(getAnalysisIfAvailable<DataLayout>());
        for (Module::iterator I=M.begin(),E=M.end(); I != E; ++I) {
          Function &F = *I;
          {
            SCOPED_TRACE("Running on the fly function pass");
            getAnalysis<FPass>(F);
          }
        }
        return false;
      }
      virtual void getAnalysisUsage(AnalysisUsage &AU) const {
        AU.addRequired<FPass>();
      }
    };
    char OnTheFlyTest::ID=0;

    TEST(PassManager, RunOnce) {
      Module M("test-once", getGlobalContext());
      struct ModuleNDNM *mNDNM = new ModuleNDNM();
      struct ModuleDNM *mDNM = new ModuleDNM();
      struct ModuleNDM *mNDM = new ModuleNDM();
      struct ModuleNDM2 *mNDM2 = new ModuleNDM2();

      mNDM->run = mNDNM->run = mDNM->run = mNDM2->run = 0;

      PassManager Passes;
      Passes.add(new DataLayout(&M));
      Passes.add(mNDM2);
      Passes.add(mNDM);
      Passes.add(mNDNM);
      Passes.add(mDNM);

      Passes.run(M);
      // each pass must be run exactly once, since nothing invalidates them
      EXPECT_EQ(1, mNDM->run);
      EXPECT_EQ(1, mNDNM->run);
      EXPECT_EQ(1, mDNM->run);
      EXPECT_EQ(1, mNDM2->run);
    }

    TEST(PassManager, ReRun) {
      Module M("test-rerun", getGlobalContext());
      struct ModuleNDNM *mNDNM = new ModuleNDNM();
      struct ModuleDNM *mDNM = new ModuleDNM();
      struct ModuleNDM *mNDM = new ModuleNDM();
      struct ModuleNDM2 *mNDM2 = new ModuleNDM2();

      mNDM->run = mNDNM->run = mDNM->run = mNDM2->run = 0;

      PassManager Passes;
      Passes.add(new DataLayout(&M));
      Passes.add(mNDM);
      Passes.add(mNDNM);
      Passes.add(mNDM2);// invalidates mNDM needed by mDNM
      Passes.add(mDNM);

      Passes.run(M);
      // Some passes must be rerun because a pass that modified the
      // module/function was run in between
      EXPECT_EQ(2, mNDM->run);
      EXPECT_EQ(1, mNDNM->run);
      EXPECT_EQ(1, mNDM2->run);
      EXPECT_EQ(1, mDNM->run);
    }

    Module* makeLLVMModule();

    template<typename T>
    void MemoryTestHelper(int run) {
      OwningPtr<Module> M(makeLLVMModule());
      T *P = new T();
      PassManager Passes;
      Passes.add(new DataLayout(M.get()));
      Passes.add(P);
      Passes.run(*M);
      T::finishedOK(run);
    }

    template<typename T>
    void MemoryTestHelper(int run, int N) {
      Module *M = makeLLVMModule();
      T *P = new T();
      PassManager Passes;
      Passes.add(new DataLayout(M));
      Passes.add(P);
      Passes.run(*M);
      T::finishedOK(run, N);
      delete M;
    }

    TEST(PassManager, Memory) {
      // SCC#1: test1->test2->test3->test1
      // SCC#2: test4
      // SCC#3: indirect call node
      {
        SCOPED_TRACE("Callgraph pass");
        MemoryTestHelper<CGPass>(3);
      }

      {
        SCOPED_TRACE("Function pass");
        MemoryTestHelper<FPass>(4);// 4 functions
      }

      {
        SCOPED_TRACE("Loop pass");
        MemoryTestHelper<LPass>(2, 1); //2 loops, 1 function
      }
      {
        SCOPED_TRACE("Basic block pass");
        MemoryTestHelper<BPass>(7, 4); //9 basic blocks
      }

    }

    TEST(PassManager, MemoryOnTheFly) {
      Module *M = makeLLVMModule();
      {
        SCOPED_TRACE("Running OnTheFlyTest");
        struct OnTheFlyTest *O = new OnTheFlyTest();
        PassManager Passes;
        Passes.add(new DataLayout(M));
        Passes.add(O);
        Passes.run(*M);

        FPass::finishedOK(4);
      }
      delete M;
    }

    Module* makeLLVMModule() {
      // Module Construction
      Module* mod = new Module("test-mem", getGlobalContext());
      mod->setDataLayout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                         "i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-"
                         "a0:0:64-s0:64:64-f80:128:128");
      mod->setTargetTriple("x86_64-unknown-linux-gnu");

      // Type Definitions
      std::vector<Type*>FuncTy_0_args;
      FunctionType* FuncTy_0 = FunctionType::get(
        /*Result=*/IntegerType::get(getGlobalContext(), 32),
        /*Params=*/FuncTy_0_args,
        /*isVarArg=*/false);

      std::vector<Type*>FuncTy_2_args;
      FuncTy_2_args.push_back(IntegerType::get(getGlobalContext(), 1));
      FunctionType* FuncTy_2 = FunctionType::get(
        /*Result=*/Type::getVoidTy(getGlobalContext()),
        /*Params=*/FuncTy_2_args,
        /*isVarArg=*/false);


      // Function Declarations

      Function* func_test1 = Function::Create(
        /*Type=*/FuncTy_0,
        /*Linkage=*/GlobalValue::ExternalLinkage,
        /*Name=*/"test1", mod);
      func_test1->setCallingConv(CallingConv::C);
      AttrListPtr func_test1_PAL;
      func_test1->setAttributes(func_test1_PAL);

      Function* func_test2 = Function::Create(
        /*Type=*/FuncTy_0,
        /*Linkage=*/GlobalValue::ExternalLinkage,
        /*Name=*/"test2", mod);
      func_test2->setCallingConv(CallingConv::C);
      AttrListPtr func_test2_PAL;
      func_test2->setAttributes(func_test2_PAL);

      Function* func_test3 = Function::Create(
        /*Type=*/FuncTy_0,
        /*Linkage=*/GlobalValue::ExternalLinkage,
        /*Name=*/"test3", mod);
      func_test3->setCallingConv(CallingConv::C);
      AttrListPtr func_test3_PAL;
      func_test3->setAttributes(func_test3_PAL);

      Function* func_test4 = Function::Create(
        /*Type=*/FuncTy_2,
        /*Linkage=*/GlobalValue::ExternalLinkage,
        /*Name=*/"test4", mod);
      func_test4->setCallingConv(CallingConv::C);
      AttrListPtr func_test4_PAL;
      func_test4->setAttributes(func_test4_PAL);

      // Global Variable Declarations


      // Constant Definitions

      // Global Variable Definitions

      // Function Definitions

      // Function: test1 (func_test1)
      {

        BasicBlock* label_entry = BasicBlock::Create(getGlobalContext(), "entry",func_test1,0);

        // Block entry (label_entry)
        CallInst* int32_3 = CallInst::Create(func_test2, "", label_entry);
        int32_3->setCallingConv(CallingConv::C);
        int32_3->setTailCall(false);AttrListPtr int32_3_PAL;
        int32_3->setAttributes(int32_3_PAL);

        ReturnInst::Create(getGlobalContext(), int32_3, label_entry);

      }

      // Function: test2 (func_test2)
      {

        BasicBlock* label_entry_5 = BasicBlock::Create(getGlobalContext(), "entry",func_test2,0);

        // Block entry (label_entry_5)
        CallInst* int32_6 = CallInst::Create(func_test3, "", label_entry_5);
        int32_6->setCallingConv(CallingConv::C);
        int32_6->setTailCall(false);AttrListPtr int32_6_PAL;
        int32_6->setAttributes(int32_6_PAL);

        ReturnInst::Create(getGlobalContext(), int32_6, label_entry_5);

      }

      // Function: test3 (func_test3)
      {

        BasicBlock* label_entry_8 = BasicBlock::Create(getGlobalContext(), "entry",func_test3,0);

        // Block entry (label_entry_8)
        CallInst* int32_9 = CallInst::Create(func_test1, "", label_entry_8);
        int32_9->setCallingConv(CallingConv::C);
        int32_9->setTailCall(false);AttrListPtr int32_9_PAL;
        int32_9->setAttributes(int32_9_PAL);

        ReturnInst::Create(getGlobalContext(), int32_9, label_entry_8);

      }

      // Function: test4 (func_test4)
      {
        Function::arg_iterator args = func_test4->arg_begin();
        Value* int1_f = args++;
        int1_f->setName("f");

        BasicBlock* label_entry_11 = BasicBlock::Create(getGlobalContext(), "entry",func_test4,0);
        BasicBlock* label_bb = BasicBlock::Create(getGlobalContext(), "bb",func_test4,0);
        BasicBlock* label_bb1 = BasicBlock::Create(getGlobalContext(), "bb1",func_test4,0);
        BasicBlock* label_return = BasicBlock::Create(getGlobalContext(), "return",func_test4,0);

        // Block entry (label_entry_11)
        BranchInst::Create(label_bb, label_entry_11);

        // Block bb (label_bb)
        BranchInst::Create(label_bb, label_bb1, int1_f, label_bb);

        // Block bb1 (label_bb1)
        BranchInst::Create(label_bb1, label_return, int1_f, label_bb1);

        // Block return (label_return)
        ReturnInst::Create(getGlobalContext(), label_return);

      }
      return mod;
    }

  }
}

INITIALIZE_PASS(ModuleNDM, "mndm", "mndm", false, false)
INITIALIZE_PASS_BEGIN(CGPass, "cgp","cgp", false, false)
INITIALIZE_AG_DEPENDENCY(CallGraph)
INITIALIZE_PASS_END(CGPass, "cgp","cgp", false, false)
INITIALIZE_PASS(FPass, "fp","fp", false, false)
INITIALIZE_PASS_BEGIN(LPass, "lp","lp", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfo)
INITIALIZE_PASS_END(LPass, "lp","lp", false, false)
INITIALIZE_PASS(BPass, "bp","bp", false, false)
