//===- unittests/MIR/PassBuilderCallbacksTest.cpp - PB Callback Tests -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Testing/Support/Error.h"
#include <functional>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <llvm/ADT/Any.h>
#include <llvm/AsmParser/Parser.h>
#include <llvm/CodeGen/MIRParser/MIRParser.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/MachinePassManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/PassInstrumentation.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Regex.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>

using namespace llvm;

namespace {
using testing::_;
using testing::AnyNumber;
using testing::DoAll;
using testing::Not;
using testing::Return;
using testing::WithArgs;

StringRef MIRString = R"MIR(
--- |
  define void @test() {
    ret void
  }
...
---
name:            test
body:             |
  bb.0 (%ir-block.0):
    RET64
...
)MIR";

/// Helper for HasName matcher that returns getName both for IRUnit and
/// for IRUnit pointer wrapper into llvm::Any (wrapped by PassInstrumentation).
template <typename IRUnitT> std::string getName(const IRUnitT &IR) {
  return std::string(IR.getName());
}

template <> std::string getName(const StringRef &name) {
  return std::string(name);
}

template <> std::string getName(const Any &WrappedIR) {
  if (const auto *const *M = llvm::any_cast<const Module *>(&WrappedIR))
    return (*M)->getName().str();
  if (const auto *const *F = llvm::any_cast<const Function *>(&WrappedIR))
    return (*F)->getName().str();
  if (const auto *const *MF =
          llvm::any_cast<const MachineFunction *>(&WrappedIR))
    return (*MF)->getName().str();
  return "<UNKNOWN>";
}
/// Define a custom matcher for objects which support a 'getName' method.
///
/// LLVM often has IR objects or analysis objects which expose a name
/// and in tests it is convenient to match these by name for readability.
/// Usually, this name is either a StringRef or a plain std::string. This
/// matcher supports any type exposing a getName() method of this form whose
/// return value is compatible with an std::ostream. For StringRef, this uses
/// the shift operator defined above.
///
/// It should be used as:
///
///   HasName("my_function")
///
/// No namespace or other qualification is required.
MATCHER_P(HasName, Name, "") {
  *result_listener << "has name '" << getName(arg) << "'";
  return Name == getName(arg);
}

MATCHER_P(HasNameRegex, Name, "") {
  *result_listener << "has name '" << getName(arg) << "'";
  llvm::Regex r(Name);
  return r.match(getName(arg));
}

struct MockPassInstrumentationCallbacks {
  PassInstrumentationCallbacks Callbacks;

  MockPassInstrumentationCallbacks() {
    ON_CALL(*this, runBeforePass(_, _)).WillByDefault(Return(true));
  }
  MOCK_METHOD2(runBeforePass, bool(StringRef PassID, llvm::Any));
  MOCK_METHOD2(runBeforeSkippedPass, void(StringRef PassID, llvm::Any));
  MOCK_METHOD2(runBeforeNonSkippedPass, void(StringRef PassID, llvm::Any));
  MOCK_METHOD3(runAfterPass,
               void(StringRef PassID, llvm::Any, const PreservedAnalyses &PA));
  MOCK_METHOD2(runAfterPassInvalidated,
               void(StringRef PassID, const PreservedAnalyses &PA));
  MOCK_METHOD2(runBeforeAnalysis, void(StringRef PassID, llvm::Any));
  MOCK_METHOD2(runAfterAnalysis, void(StringRef PassID, llvm::Any));

  void registerPassInstrumentation() {
    Callbacks.registerShouldRunOptionalPassCallback(
        [this](StringRef P, llvm::Any IR) {
          return this->runBeforePass(P, IR);
        });
    Callbacks.registerBeforeSkippedPassCallback(
        [this](StringRef P, llvm::Any IR) {
          this->runBeforeSkippedPass(P, IR);
        });
    Callbacks.registerBeforeNonSkippedPassCallback(
        [this](StringRef P, llvm::Any IR) {
          this->runBeforeNonSkippedPass(P, IR);
        });
    Callbacks.registerAfterPassCallback(
        [this](StringRef P, llvm::Any IR, const PreservedAnalyses &PA) {
          this->runAfterPass(P, IR, PA);
        });
    Callbacks.registerAfterPassInvalidatedCallback(
        [this](StringRef P, const PreservedAnalyses &PA) {
          this->runAfterPassInvalidated(P, PA);
        });
    Callbacks.registerBeforeAnalysisCallback([this](StringRef P, llvm::Any IR) {
      return this->runBeforeAnalysis(P, IR);
    });
    Callbacks.registerAfterAnalysisCallback(
        [this](StringRef P, llvm::Any IR) { this->runAfterAnalysis(P, IR); });
  }

  void ignoreNonMockPassInstrumentation(StringRef IRName) {
    // Generic EXPECT_CALLs are needed to match instrumentation on unimportant
    // parts of a pipeline that we do not care about (e.g. various passes added
    // by default by PassBuilder - Verifier pass etc).
    // Make sure to avoid ignoring Mock passes/analysis, we definitely want
    // to check these explicitly.
    EXPECT_CALL(*this,
                runBeforePass(Not(HasNameRegex("Mock")), HasName(IRName)))
        .Times(AnyNumber());
    EXPECT_CALL(
        *this, runBeforeSkippedPass(Not(HasNameRegex("Mock")), HasName(IRName)))
        .Times(AnyNumber());
    EXPECT_CALL(*this, runBeforeNonSkippedPass(Not(HasNameRegex("Mock")),
                                               HasName(IRName)))
        .Times(AnyNumber());
    EXPECT_CALL(*this,
                runAfterPass(Not(HasNameRegex("Mock")), HasName(IRName), _))
        .Times(AnyNumber());
    EXPECT_CALL(*this, runBeforeAnalysis(HasNameRegex("MachineModuleAnalysis"),
                                         HasName(IRName)))
        .Times(AnyNumber());
    EXPECT_CALL(*this,
                runBeforeAnalysis(Not(HasNameRegex("Mock")), HasName(IRName)))
        .Times(AnyNumber());
    EXPECT_CALL(*this, runAfterAnalysis(HasNameRegex("MachineModuleAnalysis"),
                                        HasName(IRName)))
        .Times(AnyNumber());
    EXPECT_CALL(*this,
                runAfterAnalysis(Not(HasNameRegex("Mock")), HasName(IRName)))
        .Times(AnyNumber());
  }
};

template <typename DerivedT> class MockAnalysisHandleBase {
public:
  class Analysis : public AnalysisInfoMixin<Analysis> {
    friend AnalysisInfoMixin<Analysis>;
    friend MockAnalysisHandleBase;
    static AnalysisKey Key;

    DerivedT *Handle;

    Analysis(DerivedT &Handle) : Handle(&Handle) {
      static_assert(std::is_base_of<MockAnalysisHandleBase, DerivedT>::value,
                    "Must pass the derived type to this template!");
    }

  public:
    class Result {
      friend MockAnalysisHandleBase;

      DerivedT *Handle;

      Result(DerivedT &Handle) : Handle(&Handle) {}

    public:
      // Forward invalidation events to the mock handle.
      bool invalidate(MachineFunction &IR, const PreservedAnalyses &PA,
                      MachineFunctionAnalysisManager::Invalidator &Inv) {
        return Handle->invalidate(IR, PA, Inv);
      }
    };

    Result run(MachineFunction &IR, MachineFunctionAnalysisManager::Base &AM) {
      return Handle->run(IR, AM);
    }
  };

  Analysis getAnalysis() { return Analysis(static_cast<DerivedT &>(*this)); }
  typename Analysis::Result getResult() {
    return typename Analysis::Result(static_cast<DerivedT &>(*this));
  }
  static StringRef getName() { return llvm::getTypeName<DerivedT>(); }

protected:
  // FIXME: MSVC seems unable to handle a lambda argument to Invoke from within
  // the template, so we use a boring static function.
  static bool
  invalidateCallback(MachineFunction &IR, const PreservedAnalyses &PA,
                     MachineFunctionAnalysisManager::Invalidator &Inv) {
    auto PAC = PA.template getChecker<Analysis>();
    return !PAC.preserved() &&
           !PAC.template preservedSet<AllAnalysesOn<MachineFunction>>();
  }

  /// Derived classes should call this in their constructor to set up default
  /// mock actions. (We can't do this in our constructor because this has to
  /// run after the DerivedT is constructed.)
  void setDefaults() {
    ON_CALL(static_cast<DerivedT &>(*this), run(_, _))
        .WillByDefault(Return(this->getResult()));
    ON_CALL(static_cast<DerivedT &>(*this), invalidate(_, _, _))
        .WillByDefault(&invalidateCallback);
  }
};

template <typename DerivedT> class MockPassHandleBase {
public:
  class Pass : public MachinePassInfoMixin<Pass> {
    friend MockPassHandleBase;

    DerivedT *Handle;

    Pass(DerivedT &Handle) : Handle(&Handle) {
      static_assert(std::is_base_of<MockPassHandleBase, DerivedT>::value,
                    "Must pass the derived type to this template!");
    }

  public:
    PreservedAnalyses run(MachineFunction &IR,
                          MachineFunctionAnalysisManager::Base &AM) {
      return Handle->run(IR, AM);
    }
  };

  static StringRef getName() { return llvm::getTypeName<DerivedT>(); }

  Pass getPass() { return Pass(static_cast<DerivedT &>(*this)); }

protected:
  /// Derived classes should call this in their constructor to set up default
  /// mock actions. (We can't do this in our constructor because this has to
  /// run after the DerivedT is constructed.)
  void setDefaults() {
    ON_CALL(static_cast<DerivedT &>(*this), run(_, _))
        .WillByDefault(Return(PreservedAnalyses::all()));
  }
};

struct MockAnalysisHandle : public MockAnalysisHandleBase<MockAnalysisHandle> {
  MOCK_METHOD2(run, Analysis::Result(MachineFunction &,
                                     MachineFunctionAnalysisManager::Base &));

  MOCK_METHOD3(invalidate, bool(MachineFunction &, const PreservedAnalyses &,
                                MachineFunctionAnalysisManager::Invalidator &));

  MockAnalysisHandle() { setDefaults(); }
};

template <typename DerivedT>
AnalysisKey MockAnalysisHandleBase<DerivedT>::Analysis::Key;

class MockPassHandle : public MockPassHandleBase<MockPassHandle> {
public:
  MOCK_METHOD2(run, PreservedAnalyses(MachineFunction &,
                                      MachineFunctionAnalysisManager::Base &));

  MockPassHandle() { setDefaults(); }
};

class MachineFunctionCallbacksTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    InitializeAllTargetInfos();
    InitializeAllTargets();
    InitializeAllTargetMCs();
  }

  std::unique_ptr<LLVMTargetMachine> TM;
  std::unique_ptr<MachineModuleInfo> MMI;

  LLVMContext Context;
  std::unique_ptr<Module> M;
  std::unique_ptr<MIRParser> MIR;

  MockPassInstrumentationCallbacks CallbacksHandle;

  PassBuilder PB;
  ModulePassManager PM;
  MachineFunctionPassManager MFPM;
  FunctionAnalysisManager FAM;
  ModuleAnalysisManager AM;
  MachineFunctionAnalysisManager MFAM;

  MockPassHandle PassHandle;
  MockAnalysisHandle AnalysisHandle;

  std::unique_ptr<Module> parseMIR(const TargetMachine &TM, StringRef MIRCode,
                                   MachineModuleInfo &MMI) {
    SMDiagnostic Diagnostic;
    std::unique_ptr<MemoryBuffer> MBuffer = MemoryBuffer::getMemBuffer(MIRCode);
    MIR = createMIRParser(std::move(MBuffer), Context);
    if (!MIR)
      return nullptr;

    std::unique_ptr<Module> Mod = MIR->parseIRModule();
    if (!Mod)
      return nullptr;

    Mod->setDataLayout(TM.createDataLayout());

    if (MIR->parseMachineFunctions(*Mod, MMI)) {
      M.reset();
      return nullptr;
    }
    return Mod;
  }

  static PreservedAnalyses
  getAnalysisResult(MachineFunction &U,
                    MachineFunctionAnalysisManager::Base &AM) {
    auto &MFAM = static_cast<MachineFunctionAnalysisManager &>(AM);
    MFAM.getResult<MockAnalysisHandle::Analysis>(U);
    return PreservedAnalyses::all();
  }

  void SetUp() override {
    std::string Error;
    auto TripleName = "x86_64-pc-linux-gnu";
    auto *T = TargetRegistry::lookupTarget(TripleName, Error);
    if (!T)
      GTEST_SKIP();
    TM = std::unique_ptr<LLVMTargetMachine>(
        static_cast<LLVMTargetMachine *>(T->createTargetMachine(
            TripleName, "", "", TargetOptions(), std::nullopt)));
    if (!TM)
      GTEST_SKIP();
    MMI = std::make_unique<MachineModuleInfo>(TM.get());
    M = parseMIR(*TM, MIRString, *MMI);
    AM.registerPass([&] { return MachineModuleAnalysis(*MMI); });
  }

  MachineFunctionCallbacksTest()
      : CallbacksHandle(), PB(nullptr, PipelineTuningOptions(), std::nullopt,
                              &CallbacksHandle.Callbacks),
        PM(), FAM(), AM(), MFAM(FAM, AM) {

    EXPECT_TRUE(&CallbacksHandle.Callbacks ==
                PB.getPassInstrumentationCallbacks());

    /// Register a callback for analysis registration.
    ///
    /// The callback is a function taking a reference to an AnalyisManager
    /// object. When called, the callee gets to register its own analyses with
    /// this PassBuilder instance.
    PB.registerAnalysisRegistrationCallback(
        [this](MachineFunctionAnalysisManager &AM) {
          // Register our mock analysis
          AM.registerPass([this] { return AnalysisHandle.getAnalysis(); });
        });

    /// Register a callback for pipeline parsing.
    ///
    /// During parsing of a textual pipeline, the PassBuilder will call these
    /// callbacks for each encountered pass name that it does not know. This
    /// includes both simple pass names as well as names of sub-pipelines. In
    /// the latter case, the InnerPipeline is not empty.
    PB.registerPipelineParsingCallback(
        [this](StringRef Name, MachineFunctionPassManager &PM) {
          if (parseAnalysisUtilityPasses<MockAnalysisHandle::Analysis>(
                  "test-analysis", Name, PM))
            return true;

          /// Parse the name of our pass mock handle
          if (Name == "test-transform") {
            MFPM.addPass(PassHandle.getPass());
            return true;
          }
          return false;
        });

    /// Register builtin analyses and cross-register the analysis proxies
    PB.registerModuleAnalyses(AM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerMachineFunctionAnalyses(MFAM);
  }
};

TEST_F(MachineFunctionCallbacksTest, Passes) {
  EXPECT_CALL(AnalysisHandle, run(HasName("test"), _));
  EXPECT_CALL(PassHandle, run(HasName("test"), _)).WillOnce(&getAnalysisResult);

  StringRef PipelineText = "test-transform";
  ASSERT_THAT_ERROR(PB.parsePassPipeline(MFPM, PipelineText), Succeeded())
      << "Pipeline was: " << PipelineText;
  ASSERT_THAT_ERROR(MFPM.run(*M, MFAM), Succeeded());
}

TEST_F(MachineFunctionCallbacksTest, InstrumentedPasses) {
  CallbacksHandle.registerPassInstrumentation();
  // Non-mock instrumentation not specifically mentioned below can be ignored.
  CallbacksHandle.ignoreNonMockPassInstrumentation("<string>");
  CallbacksHandle.ignoreNonMockPassInstrumentation("test");
  CallbacksHandle.ignoreNonMockPassInstrumentation("");

  // PassInstrumentation calls should happen in-sequence, in the same order
  // as passes/analyses are scheduled.
  ::testing::Sequence PISequence;
  EXPECT_CALL(CallbacksHandle,
              runBeforePass(HasNameRegex("MockPassHandle"), HasName("test")))
      .InSequence(PISequence);
  EXPECT_CALL(
      CallbacksHandle,
      runBeforeNonSkippedPass(HasNameRegex("MockPassHandle"), HasName("test")))
      .InSequence(PISequence);
  EXPECT_CALL(CallbacksHandle,
              runBeforeAnalysis(HasNameRegex("MockAnalysisHandle"), _))
      .InSequence(PISequence);
  EXPECT_CALL(CallbacksHandle,
              runAfterAnalysis(HasNameRegex("MockAnalysisHandle"), _))
      .InSequence(PISequence);
  EXPECT_CALL(CallbacksHandle,
              runAfterPass(HasNameRegex("MockPassHandle"), HasName("test"), _))
      .InSequence(PISequence);

  EXPECT_CALL(AnalysisHandle, run(HasName("test"), _));
  EXPECT_CALL(PassHandle, run(HasName("test"), _)).WillOnce(&getAnalysisResult);

  StringRef PipelineText = "test-transform";
  ASSERT_THAT_ERROR(PB.parsePassPipeline(MFPM, PipelineText), Succeeded())
      << "Pipeline was: " << PipelineText;
  ASSERT_THAT_ERROR(MFPM.run(*M, MFAM), Succeeded());
}

TEST_F(MachineFunctionCallbacksTest, InstrumentedSkippedPasses) {
  CallbacksHandle.registerPassInstrumentation();
  // Non-mock instrumentation run here can safely be ignored.
  CallbacksHandle.ignoreNonMockPassInstrumentation("<string>");
  CallbacksHandle.ignoreNonMockPassInstrumentation("test");
  CallbacksHandle.ignoreNonMockPassInstrumentation("");

  // Skip the pass by returning false.
  EXPECT_CALL(CallbacksHandle,
              runBeforePass(HasNameRegex("MockPassHandle"), HasName("test")))
      .WillOnce(Return(false));

  EXPECT_CALL(
      CallbacksHandle,
      runBeforeSkippedPass(HasNameRegex("MockPassHandle"), HasName("test")))
      .Times(1);

  EXPECT_CALL(AnalysisHandle, run(HasName("test"), _)).Times(0);
  EXPECT_CALL(PassHandle, run(HasName("test"), _)).Times(0);

  // As the pass is skipped there is no afterPass, beforeAnalysis/afterAnalysis
  // as well.
  EXPECT_CALL(CallbacksHandle,
              runBeforeNonSkippedPass(HasNameRegex("MockPassHandle"), _))
      .Times(0);
  EXPECT_CALL(CallbacksHandle,
              runAfterPass(HasNameRegex("MockPassHandle"), _, _))
      .Times(0);
  EXPECT_CALL(CallbacksHandle,
              runAfterPassInvalidated(HasNameRegex("MockPassHandle"), _))
      .Times(0);
  EXPECT_CALL(CallbacksHandle,
              runAfterPass(HasNameRegex("MockPassHandle"), _, _))
      .Times(0);
  EXPECT_CALL(CallbacksHandle,
              runBeforeAnalysis(HasNameRegex("MockAnalysisHandle"), _))
      .Times(0);
  EXPECT_CALL(CallbacksHandle,
              runAfterAnalysis(HasNameRegex("MockAnalysisHandle"), _))
      .Times(0);

  StringRef PipelineText = "test-transform";
  ASSERT_THAT_ERROR(PB.parsePassPipeline(MFPM, PipelineText), Succeeded())
      << "Pipeline was: " << PipelineText;
  ASSERT_THAT_ERROR(MFPM.run(*M, MFAM), Succeeded());
}

} // end anonymous namespace
