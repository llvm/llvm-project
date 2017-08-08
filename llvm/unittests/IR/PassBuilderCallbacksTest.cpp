//===- unittests/IR/PassBuilderCallbacksTest.cpp - PB Callback Tests --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/AsmParser/Parser.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Transforms/Scalar/LoopPassManager.h>

using namespace llvm;

namespace llvm {
/// Provide an ostream operator for StringRef.
///
/// For convenience we provide a custom matcher below for IRUnit's and analysis
/// result's getName functions, which most of the time returns a StringRef. The
/// matcher makes use of this operator.
static std::ostream &operator<<(std::ostream &O, StringRef S) {
  return O << S.str();
}
}

namespace {
using testing::DoDefault;
using testing::Return;
using testing::Expectation;
using testing::Invoke;
using testing::WithArgs;
using testing::_;

/// \brief A CRTP base for analysis mock handles
///
/// This class reconciles mocking with the value semantics implementation of the
/// AnalysisManager. Analysis mock handles should derive from this class and
/// call \c setDefault() in their constroctur for wiring up the defaults defined
/// by this base with their mock run() and invalidate() implementations.
template <typename DerivedT, typename IRUnitT,
          typename AnalysisManagerT = AnalysisManager<IRUnitT>,
          typename... ExtraArgTs>
class MockAnalysisHandleBase {
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
      bool invalidate(IRUnitT &IR, const PreservedAnalyses &PA,
                      typename AnalysisManagerT::Invalidator &Inv) {
        return Handle->invalidate(IR, PA, Inv);
      }
    };

    Result run(IRUnitT &IR, AnalysisManagerT &AM, ExtraArgTs... ExtraArgs) {
      return Handle->run(IR, AM, ExtraArgs...);
    }
  };

  Analysis getAnalysis() { return Analysis(static_cast<DerivedT &>(*this)); }
  typename Analysis::Result getResult() {
    return typename Analysis::Result(static_cast<DerivedT &>(*this));
  }

protected:
  // FIXME: MSVC seems unable to handle a lambda argument to Invoke from within
  // the template, so we use a boring static function.
  static bool invalidateCallback(IRUnitT &IR, const PreservedAnalyses &PA,
                                 typename AnalysisManagerT::Invalidator &Inv) {
    auto PAC = PA.template getChecker<Analysis>();
    return !PAC.preserved() &&
           !PAC.template preservedSet<AllAnalysesOn<IRUnitT>>();
  }

  /// Derived classes should call this in their constructor to set up default
  /// mock actions. (We can't do this in our constructor because this has to
  /// run after the DerivedT is constructed.)
  void setDefaults() {
    ON_CALL(static_cast<DerivedT &>(*this),
            run(_, _, testing::Matcher<ExtraArgTs>(_)...))
        .WillByDefault(Return(this->getResult()));
    ON_CALL(static_cast<DerivedT &>(*this), invalidate(_, _, _))
        .WillByDefault(Invoke(&invalidateCallback));
  }
};

/// \brief A CRTP base for pass mock handles
///
/// This class reconciles mocking with the value semantics implementation of the
/// PassManager. Pass mock handles should derive from this class and
/// call \c setDefault() in their constroctur for wiring up the defaults defined
/// by this base with their mock run() and invalidate() implementations.
template <typename DerivedT, typename IRUnitT, typename AnalysisManagerT,
          typename... ExtraArgTs>
AnalysisKey MockAnalysisHandleBase<DerivedT, IRUnitT, AnalysisManagerT,
                                   ExtraArgTs...>::Analysis::Key;

template <typename DerivedT, typename IRUnitT,
          typename AnalysisManagerT = AnalysisManager<IRUnitT>,
          typename... ExtraArgTs>
class MockPassHandleBase {
public:
  class Pass : public PassInfoMixin<Pass> {
    friend MockPassHandleBase;

    DerivedT *Handle;

    Pass(DerivedT &Handle) : Handle(&Handle) {
      static_assert(std::is_base_of<MockPassHandleBase, DerivedT>::value,
                    "Must pass the derived type to this template!");
    }

  public:
    PreservedAnalyses run(IRUnitT &IR, AnalysisManagerT &AM,
                          ExtraArgTs... ExtraArgs) {
      return Handle->run(IR, AM, ExtraArgs...);
    }
  };

  Pass getPass() { return Pass(static_cast<DerivedT &>(*this)); }

protected:
  /// Derived classes should call this in their constructor to set up default
  /// mock actions. (We can't do this in our constructor because this has to
  /// run after the DerivedT is constructed.)
  void setDefaults() {
    ON_CALL(static_cast<DerivedT &>(*this),
            run(_, _, testing::Matcher<ExtraArgTs>(_)...))
        .WillByDefault(Return(PreservedAnalyses::all()));
  }
};

/// Mock handles for passes for the IRUnits Module, CGSCC, Function, Loop.
/// These handles define the appropriate run() mock interface for the respective
/// IRUnit type.
template <typename IRUnitT> struct MockPassHandle;
template <>
struct MockPassHandle<Loop>
    : MockPassHandleBase<MockPassHandle<Loop>, Loop, LoopAnalysisManager,
                         LoopStandardAnalysisResults &, LPMUpdater &> {
  MOCK_METHOD4(run,
               PreservedAnalyses(Loop &, LoopAnalysisManager &,
                                 LoopStandardAnalysisResults &, LPMUpdater &));
  MockPassHandle() { setDefaults(); }
};

template <>
struct MockPassHandle<Function>
    : MockPassHandleBase<MockPassHandle<Function>, Function> {
  MOCK_METHOD2(run, PreservedAnalyses(Function &, FunctionAnalysisManager &));

  MockPassHandle() { setDefaults(); }
};

template <>
struct MockPassHandle<LazyCallGraph::SCC>
    : MockPassHandleBase<MockPassHandle<LazyCallGraph::SCC>, LazyCallGraph::SCC,
                         CGSCCAnalysisManager, LazyCallGraph &,
                         CGSCCUpdateResult &> {
  MOCK_METHOD4(run,
               PreservedAnalyses(LazyCallGraph::SCC &, CGSCCAnalysisManager &,
                                 LazyCallGraph &G, CGSCCUpdateResult &UR));

  MockPassHandle() { setDefaults(); }
};

template <>
struct MockPassHandle<Module>
    : MockPassHandleBase<MockPassHandle<Module>, Module> {
  MOCK_METHOD2(run, PreservedAnalyses(Module &, ModuleAnalysisManager &));

  MockPassHandle() { setDefaults(); }
};

/// Mock handles for analyses for the IRUnits Module, CGSCC, Function, Loop.
/// These handles define the appropriate run() and invalidate() mock interfaces
/// for the respective IRUnit type.
template <typename IRUnitT> struct MockAnalysisHandle;
template <>
struct MockAnalysisHandle<Loop>
    : MockAnalysisHandleBase<MockAnalysisHandle<Loop>, Loop,
                             LoopAnalysisManager,
                             LoopStandardAnalysisResults &> {

  MOCK_METHOD3_T(run, typename Analysis::Result(Loop &, LoopAnalysisManager &,
                                                LoopStandardAnalysisResults &));

  MOCK_METHOD3_T(invalidate, bool(Loop &, const PreservedAnalyses &,
                                  LoopAnalysisManager::Invalidator &));

  MockAnalysisHandle<Loop>() { this->setDefaults(); }
};

template <>
struct MockAnalysisHandle<Function>
    : MockAnalysisHandleBase<MockAnalysisHandle<Function>, Function> {
  MOCK_METHOD2(run, Analysis::Result(Function &, FunctionAnalysisManager &));

  MOCK_METHOD3(invalidate, bool(Function &, const PreservedAnalyses &,
                                FunctionAnalysisManager::Invalidator &));

  MockAnalysisHandle<Function>() { setDefaults(); }
};

template <>
struct MockAnalysisHandle<LazyCallGraph::SCC>
    : MockAnalysisHandleBase<MockAnalysisHandle<LazyCallGraph::SCC>,
                             LazyCallGraph::SCC, CGSCCAnalysisManager,
                             LazyCallGraph &> {
  MOCK_METHOD3(run, Analysis::Result(LazyCallGraph::SCC &,
                                     CGSCCAnalysisManager &, LazyCallGraph &));

  MOCK_METHOD3(invalidate, bool(LazyCallGraph::SCC &, const PreservedAnalyses &,
                                CGSCCAnalysisManager::Invalidator &));

  MockAnalysisHandle<LazyCallGraph::SCC>() { setDefaults(); }
};

template <>
struct MockAnalysisHandle<Module>
    : MockAnalysisHandleBase<MockAnalysisHandle<Module>, Module> {
  MOCK_METHOD2(run, Analysis::Result(Module &, ModuleAnalysisManager &));

  MOCK_METHOD3(invalidate, bool(Module &, const PreservedAnalyses &,
                                ModuleAnalysisManager::Invalidator &));

  MockAnalysisHandle<Module>() { setDefaults(); }
};

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  return parseAssemblyString(IR, Err, C);
}

template <typename PassManagerT> class PassBuilderCallbacksTest;

/// This test fixture is shared between all the actual tests below and
/// takes care of setting up appropriate defaults.
///
/// The template specialization serves to extract the IRUnit and AM types from
/// the given PassManagerT.
template <typename TestIRUnitT, typename... ExtraPassArgTs,
          typename... ExtraAnalysisArgTs>
class PassBuilderCallbacksTest<PassManager<
    TestIRUnitT, AnalysisManager<TestIRUnitT, ExtraAnalysisArgTs...>,
    ExtraPassArgTs...>> : public testing::Test {
protected:
  using IRUnitT = TestIRUnitT;
  using AnalysisManagerT = AnalysisManager<TestIRUnitT, ExtraAnalysisArgTs...>;
  using PassManagerT =
      PassManager<TestIRUnitT, AnalysisManagerT, ExtraPassArgTs...>;
  using AnalysisT = typename MockAnalysisHandle<IRUnitT>::Analysis;

  LLVMContext Context;
  std::unique_ptr<Module> M;

  PassBuilder PB;
  ModulePassManager PM;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager AM;

  MockPassHandle<IRUnitT> PassHandle;
  MockAnalysisHandle<IRUnitT> AnalysisHandle;

  static PreservedAnalyses getAnalysisResult(IRUnitT &U, AnalysisManagerT &AM,
                                             ExtraAnalysisArgTs &&... Args) {
    (void)AM.template getResult<AnalysisT>(
        U, std::forward<ExtraAnalysisArgTs>(Args)...);
    return PreservedAnalyses::all();
  }

  PassBuilderCallbacksTest()
      : M(parseIR(Context,
                  "declare void @bar()\n"
                  "define void @foo(i32 %n) {\n"
                  "entry:\n"
                  "  br label %loop\n"
                  "loop:\n"
                  "  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]\n"
                  "  %iv.next = add i32 %iv, 1\n"
                  "  tail call void @bar()\n"
                  "  %cmp = icmp eq i32 %iv, %n\n"
                  "  br i1 %cmp, label %exit, label %loop\n"
                  "exit:\n"
                  "  ret void\n"
                  "}\n")),
        PM(true), LAM(true), FAM(true), CGAM(true), AM(true) {

    /// Register a callback for analysis registration.
    ///
    /// The callback is a function taking a reference to an AnalyisManager
    /// object. When called, the callee gets to register its own analyses with
    /// this PassBuilder instance.
    PB.registerAnalysisRegistrationCallback([this](AnalysisManagerT &AM) {
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
        [this](StringRef Name, PassManagerT &PM,
               ArrayRef<PassBuilder::PipelineElement> InnerPipeline) {
          /// Handle parsing of the names of analysis utilities such as
          /// require<test-analysis> and invalidate<test-analysis> for our
          /// analysis mock handle
          if (parseAnalysisUtilityPasses<AnalysisT>("test-analysis", Name, PM))
            return true;

          /// Parse the name of our pass mock handle
          if (Name == "test-transform") {
            PM.addPass(PassHandle.getPass());
            return true;
          }
          return false;
        });

    /// Register builtin analyses and cross-register the analysis proxies
    PB.registerModuleAnalyses(AM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, AM);
  }
};

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
  *result_listener << "has name '" << arg.getName() << "'";
  return Name == arg.getName();
}

using ModuleCallbacksTest = PassBuilderCallbacksTest<ModulePassManager>;
using CGSCCCallbacksTest = PassBuilderCallbacksTest<CGSCCPassManager>;
using FunctionCallbacksTest = PassBuilderCallbacksTest<FunctionPassManager>;
using LoopCallbacksTest = PassBuilderCallbacksTest<LoopPassManager>;

/// Test parsing of the name of our mock pass for all IRUnits.
///
/// The pass should by default run our mock analysis and then preserve it.
TEST_F(ModuleCallbacksTest, Passes) {
  EXPECT_CALL(AnalysisHandle, run(HasName("<string>"), _));
  EXPECT_CALL(PassHandle, run(HasName("<string>"), _))
      .WillOnce(Invoke(getAnalysisResult));

  StringRef PipelineText = "test-transform";
  ASSERT_TRUE(PB.parsePassPipeline(PM, PipelineText, true))
      << "Pipeline was: " << PipelineText;
  PM.run(*M, AM);
}

TEST_F(FunctionCallbacksTest, Passes) {
  EXPECT_CALL(AnalysisHandle, run(HasName("foo"), _));
  EXPECT_CALL(PassHandle, run(HasName("foo"), _))
      .WillOnce(Invoke(getAnalysisResult));

  StringRef PipelineText = "test-transform";
  ASSERT_TRUE(PB.parsePassPipeline(PM, PipelineText, true))
      << "Pipeline was: " << PipelineText;
  PM.run(*M, AM);
}

TEST_F(LoopCallbacksTest, Passes) {
  EXPECT_CALL(AnalysisHandle, run(HasName("loop"), _, _));
  EXPECT_CALL(PassHandle, run(HasName("loop"), _, _, _))
      .WillOnce(WithArgs<0, 1, 2>(Invoke(getAnalysisResult)));

  StringRef PipelineText = "test-transform";
  ASSERT_TRUE(PB.parsePassPipeline(PM, PipelineText, true))
      << "Pipeline was: " << PipelineText;
  PM.run(*M, AM);
}

TEST_F(CGSCCCallbacksTest, Passes) {
  EXPECT_CALL(AnalysisHandle, run(HasName("(foo)"), _, _));
  EXPECT_CALL(PassHandle, run(HasName("(foo)"), _, _, _))
      .WillOnce(WithArgs<0, 1, 2>(Invoke(getAnalysisResult)));

  StringRef PipelineText = "test-transform";
  ASSERT_TRUE(PB.parsePassPipeline(PM, PipelineText, true))
      << "Pipeline was: " << PipelineText;
  PM.run(*M, AM);
}

/// Test parsing of the names of analysis utilities for our mock analysis
/// for all IRUnits.
///
/// We first require<>, then invalidate<> it, expecting the analysis to be run
/// once and subsequently invalidated.
TEST_F(ModuleCallbacksTest, AnalysisUtilities) {
  EXPECT_CALL(AnalysisHandle, run(HasName("<string>"), _));
  EXPECT_CALL(AnalysisHandle, invalidate(HasName("<string>"), _, _));

  StringRef PipelineText = "require<test-analysis>,invalidate<test-analysis>";
  ASSERT_TRUE(PB.parsePassPipeline(PM, PipelineText, true))
      << "Pipeline was: " << PipelineText;
  PM.run(*M, AM);
}

TEST_F(CGSCCCallbacksTest, PassUtilities) {
  EXPECT_CALL(AnalysisHandle, run(HasName("(foo)"), _, _));
  EXPECT_CALL(AnalysisHandle, invalidate(HasName("(foo)"), _, _));

  StringRef PipelineText = "require<test-analysis>,invalidate<test-analysis>";
  ASSERT_TRUE(PB.parsePassPipeline(PM, PipelineText, true))
      << "Pipeline was: " << PipelineText;
  PM.run(*M, AM);
}

TEST_F(FunctionCallbacksTest, AnalysisUtilities) {
  EXPECT_CALL(AnalysisHandle, run(HasName("foo"), _));
  EXPECT_CALL(AnalysisHandle, invalidate(HasName("foo"), _, _));

  StringRef PipelineText = "require<test-analysis>,invalidate<test-analysis>";
  ASSERT_TRUE(PB.parsePassPipeline(PM, PipelineText, true))
      << "Pipeline was: " << PipelineText;
  PM.run(*M, AM);
}

TEST_F(LoopCallbacksTest, PassUtilities) {
  EXPECT_CALL(AnalysisHandle, run(HasName("loop"), _, _));
  EXPECT_CALL(AnalysisHandle, invalidate(HasName("loop"), _, _));

  StringRef PipelineText = "require<test-analysis>,invalidate<test-analysis>";

  ASSERT_TRUE(PB.parsePassPipeline(PM, PipelineText, true))
      << "Pipeline was: " << PipelineText;
  PM.run(*M, AM);
}

/// Test parsing of the top-level pipeline.
///
/// The ParseTopLevelPipeline callback takes over parsing of the entire pipeline
/// from PassBuilder if it encounters an unknown pipeline entry at the top level
/// (i.e., the first entry on the pipeline).
/// This test parses a pipeline named 'another-pipeline', whose only elements
/// may be the test-transform pass or the analysis utilities
TEST_F(ModuleCallbacksTest, ParseTopLevelPipeline) {
  PB.registerParseTopLevelPipelineCallback([this](
      ModulePassManager &MPM, ArrayRef<PassBuilder::PipelineElement> Pipeline,
      bool VerifyEachPass, bool DebugLogging) {
    auto &FirstName = Pipeline.front().Name;
    auto &InnerPipeline = Pipeline.front().InnerPipeline;
    if (FirstName == "another-pipeline") {
      for (auto &E : InnerPipeline) {
        if (parseAnalysisUtilityPasses<AnalysisT>("test-analysis", E.Name, PM))
          continue;

        if (E.Name == "test-transform") {
          PM.addPass(PassHandle.getPass());
          continue;
        }
        return false;
      }
    }
    return true;
  });

  EXPECT_CALL(AnalysisHandle, run(HasName("<string>"), _));
  EXPECT_CALL(PassHandle, run(HasName("<string>"), _))
      .WillOnce(Invoke(getAnalysisResult));
  EXPECT_CALL(AnalysisHandle, invalidate(HasName("<string>"), _, _));

  StringRef PipelineText =
      "another-pipeline(test-transform,invalidate<test-analysis>)";
  ASSERT_TRUE(PB.parsePassPipeline(PM, PipelineText, true))
      << "Pipeline was: " << PipelineText;
  PM.run(*M, AM);

  /// Test the negative case
  PipelineText = "another-pipeline(instcombine)";
  ASSERT_FALSE(PB.parsePassPipeline(PM, PipelineText, true))
      << "Pipeline was: " << PipelineText;
}
} // end anonymous namespace
