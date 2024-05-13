//===- llvm/unittest/IR/LegacyPassManager.cpp - Legacy PassManager tests --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This unit test exercises the legacy pass manager infrastructure. We use the
// old names as well to ensure that the source-level compatibility is preserved
// where possible.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/AbstractCallSite.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/OptBisect.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/CallGraphUpdater.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace llvm {
  void initializeModuleNDMPass(PassRegistry&);
  void initializeFPassPass(PassRegistry&);
  void initializeCGPassPass(PassRegistry&);
  void initializeLPassPass(PassRegistry&);

  namespace {
    // ND = no deps
    // NM = no modifications
    struct ModuleNDNM: public ModulePass {
    public:
      static char run;
      static char ID;
      ModuleNDNM() : ModulePass(ID) { }
      bool runOnModule(Module &M) override {
        run++;
        return false;
      }
      void getAnalysisUsage(AnalysisUsage &AU) const override {
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
      bool runOnModule(Module &M) override {
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
      bool runOnModule(Module &M) override {
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
      bool runOnModule(Module &M) override {
        run++;
        return false;
      }
      void getAnalysisUsage(AnalysisUsage &AU) const override {
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

      void releaseMemory() override {
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
#ifndef _MSC_VER // MSVC complains that Pass is not base class.
      using llvm::Pass::doInitialization;
      using llvm::Pass::doFinalization;
#endif
      bool doInitialization(T &t) override {
        EXPECT_FALSE(PassTestBase<P>::initialized);
        PassTestBase<P>::initialized = true;
        return false;
      }
      bool doFinalization(T &t) override {
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
      bool runOnSCC(CallGraphSCC &SCMM) override {
        run();
        return false;
      }
    };

    struct FPass : public PassTest<Module, FunctionPass> {
    public:
      bool runOnFunction(Function &F) override {
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
      using llvm::Pass::doInitialization;
      using llvm::Pass::doFinalization;
      bool doInitialization(Loop* L, LPPassManager &LPM) override {
        initialized = true;
        initcount++;
        return false;
      }
      bool runOnLoop(Loop *L, LPPassManager &LPM) override {
        run();
        return false;
      }
      bool doFinalization() override {
        fincount++;
        finalized = true;
        return false;
      }
    };
    int LPass::initcount=0;
    int LPass::fincount=0;

    struct OnTheFlyTest: public ModulePass {
    public:
      static char ID;
      OnTheFlyTest() : ModulePass(ID) {
        initializeFPassPass(*PassRegistry::getPassRegistry());
      }
      bool runOnModule(Module &M) override {
        for (Module::iterator I=M.begin(),E=M.end(); I != E; ++I) {
          Function &F = *I;
          {
            SCOPED_TRACE("Running on the fly function pass");
            getAnalysis<FPass>(F);
          }
        }
        return false;
      }
      void getAnalysisUsage(AnalysisUsage &AU) const override {
        AU.addRequired<FPass>();
      }
    };
    char OnTheFlyTest::ID=0;

    TEST(PassManager, RunOnce) {
      LLVMContext Context;
      Module M("test-once", Context);
      struct ModuleNDNM *mNDNM = new ModuleNDNM();
      struct ModuleDNM *mDNM = new ModuleDNM();
      struct ModuleNDM *mNDM = new ModuleNDM();
      struct ModuleNDM2 *mNDM2 = new ModuleNDM2();

      mNDM->run = mNDNM->run = mDNM->run = mNDM2->run = 0;

      legacy::PassManager Passes;
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
      LLVMContext Context;
      Module M("test-rerun", Context);
      struct ModuleNDNM *mNDNM = new ModuleNDNM();
      struct ModuleDNM *mDNM = new ModuleDNM();
      struct ModuleNDM *mNDM = new ModuleNDM();
      struct ModuleNDM2 *mNDM2 = new ModuleNDM2();

      mNDM->run = mNDNM->run = mDNM->run = mNDM2->run = 0;

      legacy::PassManager Passes;
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

    Module *makeLLVMModule(LLVMContext &Context);

    template<typename T>
    void MemoryTestHelper(int run) {
      LLVMContext Context;
      std::unique_ptr<Module> M(makeLLVMModule(Context));
      T *P = new T();
      legacy::PassManager Passes;
      Passes.add(P);
      Passes.run(*M);
      T::finishedOK(run);
    }

    template<typename T>
    void MemoryTestHelper(int run, int N) {
      LLVMContext Context;
      Module *M = makeLLVMModule(Context);
      T *P = new T();
      legacy::PassManager Passes;
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

    }

    TEST(PassManager, MemoryOnTheFly) {
      LLVMContext Context;
      Module *M = makeLLVMModule(Context);
      {
        SCOPED_TRACE("Running OnTheFlyTest");
        struct OnTheFlyTest *O = new OnTheFlyTest();
        legacy::PassManager Passes;
        Passes.add(O);
        Passes.run(*M);

        FPass::finishedOK(4);
      }
      delete M;
    }

    // Skips or runs optional passes.
    struct CustomOptPassGate : public OptPassGate {
      bool Skip;
      CustomOptPassGate(bool Skip) : Skip(Skip) { }
      bool shouldRunPass(const StringRef PassName, StringRef IRDescription) override {
        return !Skip;
      }
      bool isEnabled() const override { return true; }
    };

    // Optional module pass.
    struct ModuleOpt: public ModulePass {
      char run = 0;
      static char ID;
      ModuleOpt() : ModulePass(ID) { }
      bool runOnModule(Module &M) override {
        if (!skipModule(M))
          run++;
        return false;
      }
    };
    char ModuleOpt::ID=0;

    TEST(PassManager, CustomOptPassGate) {
      LLVMContext Context0;
      LLVMContext Context1;
      LLVMContext Context2;
      CustomOptPassGate SkipOptionalPasses(true);
      CustomOptPassGate RunOptionalPasses(false);

      Module M0("custom-opt-bisect", Context0);
      Module M1("custom-opt-bisect", Context1);
      Module M2("custom-opt-bisect2", Context2);
      struct ModuleOpt *mOpt0 = new ModuleOpt();
      struct ModuleOpt *mOpt1 = new ModuleOpt();
      struct ModuleOpt *mOpt2 = new ModuleOpt();

      mOpt0->run = mOpt1->run = mOpt2->run = 0;

      legacy::PassManager Passes0;
      legacy::PassManager Passes1;
      legacy::PassManager Passes2;

      Passes0.add(mOpt0);
      Passes1.add(mOpt1);
      Passes2.add(mOpt2);

      Context1.setOptPassGate(SkipOptionalPasses);
      Context2.setOptPassGate(RunOptionalPasses);

      Passes0.run(M0);
      Passes1.run(M1);
      Passes2.run(M2);

      // By default optional passes are run.
      EXPECT_EQ(1, mOpt0->run);

      // The first context skips optional passes.
      EXPECT_EQ(0, mOpt1->run);

      // The second context runs optional passes.
      EXPECT_EQ(1, mOpt2->run);
    }

    Module *makeLLVMModule(LLVMContext &Context) {
      // Module Construction
      Module *mod = new Module("test-mem", Context);
      mod->setDataLayout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                         "i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-"
                         "a:0:64-s:64:64-f80:128:128");
      mod->setTargetTriple("x86_64-unknown-linux-gnu");

      // Type Definitions
      std::vector<Type*>FuncTy_0_args;
      FunctionType *FuncTy_0 = FunctionType::get(
          /*Result=*/IntegerType::get(Context, 32),
          /*Params=*/FuncTy_0_args,
          /*isVarArg=*/false);

      std::vector<Type*>FuncTy_2_args;
      FuncTy_2_args.push_back(IntegerType::get(Context, 1));
      FunctionType *FuncTy_2 = FunctionType::get(
          /*Result=*/Type::getVoidTy(Context),
          /*Params=*/FuncTy_2_args,
          /*isVarArg=*/false);

      // Function Declarations

      Function* func_test1 = Function::Create(
        /*Type=*/FuncTy_0,
        /*Linkage=*/GlobalValue::ExternalLinkage,
        /*Name=*/"test1", mod);
      func_test1->setCallingConv(CallingConv::C);
      AttributeList func_test1_PAL;
      func_test1->setAttributes(func_test1_PAL);

      Function* func_test2 = Function::Create(
        /*Type=*/FuncTy_0,
        /*Linkage=*/GlobalValue::ExternalLinkage,
        /*Name=*/"test2", mod);
      func_test2->setCallingConv(CallingConv::C);
      AttributeList func_test2_PAL;
      func_test2->setAttributes(func_test2_PAL);

      Function* func_test3 = Function::Create(
        /*Type=*/FuncTy_0,
        /*Linkage=*/GlobalValue::InternalLinkage,
        /*Name=*/"test3", mod);
      func_test3->setCallingConv(CallingConv::C);
      AttributeList func_test3_PAL;
      func_test3->setAttributes(func_test3_PAL);

      Function* func_test4 = Function::Create(
        /*Type=*/FuncTy_2,
        /*Linkage=*/GlobalValue::ExternalLinkage,
        /*Name=*/"test4", mod);
      func_test4->setCallingConv(CallingConv::C);
      AttributeList func_test4_PAL;
      func_test4->setAttributes(func_test4_PAL);

      // Global Variable Declarations


      // Constant Definitions

      // Global Variable Definitions

      // Function Definitions

      // Function: test1 (func_test1)
      {

        BasicBlock *label_entry =
            BasicBlock::Create(Context, "entry", func_test1, nullptr);

        // Block entry (label_entry)
        CallInst* int32_3 = CallInst::Create(func_test2, "", label_entry);
        int32_3->setCallingConv(CallingConv::C);
        int32_3->setTailCall(false);
        AttributeList int32_3_PAL;
        int32_3->setAttributes(int32_3_PAL);

        ReturnInst::Create(Context, int32_3, label_entry);
      }

      // Function: test2 (func_test2)
      {

        BasicBlock *label_entry_5 =
            BasicBlock::Create(Context, "entry", func_test2, nullptr);

        // Block entry (label_entry_5)
        CallInst* int32_6 = CallInst::Create(func_test3, "", label_entry_5);
        int32_6->setCallingConv(CallingConv::C);
        int32_6->setTailCall(false);
        AttributeList int32_6_PAL;
        int32_6->setAttributes(int32_6_PAL);

        ReturnInst::Create(Context, int32_6, label_entry_5);
      }

      // Function: test3 (func_test3)
      {

        BasicBlock *label_entry_8 =
            BasicBlock::Create(Context, "entry", func_test3, nullptr);

        // Block entry (label_entry_8)
        CallInst* int32_9 = CallInst::Create(func_test1, "", label_entry_8);
        int32_9->setCallingConv(CallingConv::C);
        int32_9->setTailCall(false);
        AttributeList int32_9_PAL;
        int32_9->setAttributes(int32_9_PAL);

        ReturnInst::Create(Context, int32_9, label_entry_8);
      }

      // Function: test4 (func_test4)
      {
        Function::arg_iterator args = func_test4->arg_begin();
        Value *int1_f = &*args++;
        int1_f->setName("f");

        BasicBlock *label_entry_11 =
            BasicBlock::Create(Context, "entry", func_test4, nullptr);
        BasicBlock *label_bb =
            BasicBlock::Create(Context, "bb", func_test4, nullptr);
        BasicBlock *label_bb1 =
            BasicBlock::Create(Context, "bb1", func_test4, nullptr);
        BasicBlock *label_return =
            BasicBlock::Create(Context, "return", func_test4, nullptr);

        // Block entry (label_entry_11)
        auto *AI = new AllocaInst(func_test3->getType(), 0, "func3ptr",
                                  label_entry_11);
        new StoreInst(func_test3, AI, label_entry_11);
        BranchInst::Create(label_bb, label_entry_11);

        // Block bb (label_bb)
        BranchInst::Create(label_bb, label_bb1, int1_f, label_bb);

        // Block bb1 (label_bb1)
        BranchInst::Create(label_bb1, label_return, int1_f, label_bb1);

        // Block return (label_return)
        ReturnInst::Create(Context, label_return);
      }
      return mod;
    }

    /// Split a simple function which contains only a call and a return into two
    /// such that the first calls the second and the second whoever was called
    /// initially.
    Function *splitSimpleFunction(Function &F) {
      LLVMContext &Context = F.getContext();
      Function *SF = Function::Create(F.getFunctionType(), F.getLinkage(),
                                      F.getName() + "b", F.getParent());
      F.setName(F.getName() + "a");
      BasicBlock *Entry = BasicBlock::Create(Context, "entry", SF, nullptr);
      CallInst &CI = cast<CallInst>(F.getEntryBlock().front());
      CI.clone()->insertBefore(ReturnInst::Create(Context, Entry));
      CI.setCalledFunction(SF);
      return SF;
    }

    struct CGModifierPass : public CGPass {
      unsigned NumSCCs = 0;
      unsigned NumFns = 0;
      unsigned NumFnDecls = 0;
      unsigned SetupWorked = 0;
      unsigned NumExtCalledBefore = 0;
      unsigned NumExtCalledAfter = 0;

      CallGraphUpdater CGU;

      bool runOnSCC(CallGraphSCC &SCMM) override {
        ++NumSCCs;
        for (CallGraphNode *N : SCMM) {
          if (N->getFunction()){
            ++NumFns;
            NumFnDecls += N->getFunction()->isDeclaration();
          }
        }
        CGPass::run();

        CallGraph &CG = const_cast<CallGraph &>(SCMM.getCallGraph());
        CallGraphNode *ExtCallingNode = CG.getExternalCallingNode();
        NumExtCalledBefore = ExtCallingNode->size();

        if (SCMM.size() <= 1)
          return false;

        CallGraphNode *N = *(SCMM.begin());
        Function *F = N->getFunction();
        Module *M = F->getParent();
        Function *Test1F = M->getFunction("test1");
        Function *Test2aF = M->getFunction("test2a");
        Function *Test2bF = M->getFunction("test2b");
        Function *Test3F = M->getFunction("test3");

        auto InSCC = [&](Function *Fn) {
          return llvm::any_of(SCMM, [Fn](CallGraphNode *CGN) {
            return CGN->getFunction() == Fn;
          });
        };

        if (!Test1F || !Test2aF || !Test2bF || !Test3F || !InSCC(Test1F) ||
            !InSCC(Test2aF) || !InSCC(Test2bF) || !InSCC(Test3F))
          return false;

        CallInst *CI = dyn_cast<CallInst>(&Test1F->getEntryBlock().front());
        if (!CI || CI->getCalledFunction() != Test2aF)
          return false;

        SetupWorked += 1;

        // Create a replica of test3 and just move the blocks there.
        Function *Test3FRepl = Function::Create(
            /*Type=*/Test3F->getFunctionType(),
            /*Linkage=*/GlobalValue::InternalLinkage,
            /*Name=*/"test3repl", Test3F->getParent());
        while (!Test3F->empty()) {
          BasicBlock &BB = Test3F->front();
          BB.removeFromParent();
          BB.insertInto(Test3FRepl);
        }

        CGU.initialize(CG, SCMM);

        // Replace test3 with the replica. This is legal as it is actually
        // internal and the "capturing use" is not really capturing anything.
        CGU.replaceFunctionWith(*Test3F, *Test3FRepl);
        Test3F->replaceAllUsesWith(Test3FRepl);

        // Rewrite the call in test1 to point to the replica of 3 not test2.
        CI->setCalledFunction(Test3FRepl);

        // Delete test2a and test2b and reanalyze 1 as we changed calls inside.
        CGU.removeFunction(*Test2aF);
        CGU.removeFunction(*Test2bF);
        CGU.reanalyzeFunction(*Test1F);

        return true;
      }

      bool doFinalization(CallGraph &CG) override {
        CGU.finalize();
        // We removed test2 and replaced the internal test3.
        NumExtCalledAfter = CG.getExternalCallingNode()->size();
        return true;
      }
    };

    TEST(PassManager, CallGraphUpdater0) {
      // SCC#1: test1->test2a->test2b->test3->test1
      // SCC#2: test4
      // SCC#3: test3 (the empty function declaration as we replaced it with
      //               test3repl when we visited SCC#1)
      // SCC#4: test2a->test2b (the empty function declarations as we deleted
      //                        these functions when we visited SCC#1)
      // SCC#5: indirect call node

      LLVMContext Context;
      std::unique_ptr<Module> M(makeLLVMModule(Context));
      ASSERT_EQ(M->getFunctionList().size(), 4U);
      Function *F = M->getFunction("test2");
      Function *SF = splitSimpleFunction(*F);
      CallInst::Create(F, "", &*SF->getEntryBlock().getFirstInsertionPt());
      ASSERT_EQ(M->getFunctionList().size(), 5U);
      CGModifierPass *P = new CGModifierPass();
      legacy::PassManager Passes;
      Passes.add(P);
      Passes.run(*M);
      ASSERT_EQ(P->SetupWorked, 1U);
      ASSERT_EQ(P->NumSCCs, 4U);
      ASSERT_EQ(P->NumFns, 6U);
      ASSERT_EQ(P->NumFnDecls, 1U);
      ASSERT_EQ(M->getFunctionList().size(), 3U);
      ASSERT_EQ(P->NumExtCalledBefore, /* test1, 2a, 2b, 3, 4 */ 5U);
      ASSERT_EQ(P->NumExtCalledAfter, /* test1, 3repl, 4 */ 3U);
    }

    // Test for call graph SCC pass that replaces all callback call instructions
    // with clones and updates CallGraph by calling CallGraph::replaceCallEdge()
    // method. Test is expected to complete successfully after running pass on
    // all SCCs in the test module.
    struct CallbackCallsModifierPass : public CGPass {
      bool runOnSCC(CallGraphSCC &SCC) override {
        CGPass::run();

        CallGraph &CG = const_cast<CallGraph &>(SCC.getCallGraph());

        bool Changed = false;
        for (CallGraphNode *CGN : SCC) {
          Function *F = CGN->getFunction();
          if (!F || F->isDeclaration())
            continue;

          SmallVector<CallBase *, 4u> Calls;
          for (Use &U : F->uses()) {
            AbstractCallSite ACS(&U);
            if (!ACS || !ACS.isCallbackCall() || !ACS.isCallee(&U))
              continue;
            Calls.push_back(cast<CallBase>(ACS.getInstruction()));
          }
          if (Calls.empty())
            continue;

          for (CallBase *OldCB : Calls) {
            CallGraphNode *CallerCGN = CG[OldCB->getParent()->getParent()];
            assert(any_of(*CallerCGN,
                          [CGN](const CallGraphNode::CallRecord &CallRecord) {
                            return CallRecord.second == CGN;
                          }) &&
                   "function is not a callee");

            CallBase *NewCB = cast<CallBase>(OldCB->clone());

            NewCB->insertBefore(OldCB);
            NewCB->takeName(OldCB);

            CallerCGN->replaceCallEdge(*OldCB, *NewCB, CG[F]);

            OldCB->replaceAllUsesWith(NewCB);
            OldCB->eraseFromParent();
          }
          Changed = true;
        }
        return Changed;
      }
    };

    TEST(PassManager, CallbackCallsModifier0) {
      LLVMContext Context;

      const char *IR = "define void @foo() {\n"
                       "  call void @broker(void (i8*)* @callback0, i8* null)\n"
                       "  call void @broker(void (i8*)* @callback1, i8* null)\n"
                       "  ret void\n"
                       "}\n"
                       "\n"
                       "declare !callback !0 void @broker(void (i8*)*, i8*)\n"
                       "\n"
                       "define internal void @callback0(i8* %arg) {\n"
                       "  ret void\n"
                       "}\n"
                       "\n"
                       "define internal void @callback1(i8* %arg) {\n"
                       "  ret void\n"
                       "}\n"
                       "\n"
                       "!0 = !{!1}\n"
                       "!1 = !{i64 0, i64 1, i1 false}";

      SMDiagnostic Err;
      std::unique_ptr<Module> M = parseAssemblyString(IR, Err, Context);
      if (!M)
        Err.print("LegacyPassManagerTest", errs());

      CallbackCallsModifierPass *P = new CallbackCallsModifierPass();
      legacy::PassManager Passes;
      Passes.add(P);
      Passes.run(*M);
    }
  }
}

INITIALIZE_PASS(ModuleNDM, "mndm", "mndm", false, false)
INITIALIZE_PASS_BEGIN(CGPass, "cgp","cgp", false, false)
INITIALIZE_PASS_DEPENDENCY(CallGraphWrapperPass)
INITIALIZE_PASS_END(CGPass, "cgp","cgp", false, false)
INITIALIZE_PASS(FPass, "fp","fp", false, false)
INITIALIZE_PASS_BEGIN(LPass, "lp","lp", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(LPass, "lp","lp", false, false)
