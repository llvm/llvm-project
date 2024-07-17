//===- PGOInstrumentationTest.cpp - Instrumentation unit tests ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/PGOInstrumentation.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/ProfileData/InstrProf.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <tuple>

namespace {

using namespace llvm;

using testing::_;
using ::testing::DoDefault;
using ::testing::Invoke;
using ::testing::IsNull;
using ::testing::NotNull;
using ::testing::Ref;
using ::testing::Return;
using ::testing::Sequence;
using ::testing::Test;
using ::testing::TestParamInfo;
using ::testing::Values;
using ::testing::WithParamInterface;

template <typename Derived> class MockAnalysisHandleBase {
public:
  class Analysis : public AnalysisInfoMixin<Analysis> {
  public:
    class Result {
    public:
      // Forward invalidation events to the mock handle.
      bool invalidate(Module &M, const PreservedAnalyses &PA,
                      ModuleAnalysisManager::Invalidator &Inv) {
        return Handle->invalidate(M, PA, Inv);
      }

    private:
      explicit Result(Derived *Handle) : Handle(Handle) {}

      friend MockAnalysisHandleBase;
      Derived *Handle;
    };

    Result run(Module &M, ModuleAnalysisManager &AM) {
      return Handle->run(M, AM);
    }

  private:
    friend AnalysisInfoMixin<Analysis>;
    friend MockAnalysisHandleBase;
    static inline AnalysisKey Key;

    Derived *Handle;

    explicit Analysis(Derived *Handle) : Handle(Handle) {}
  };

  Analysis getAnalysis() { return Analysis(static_cast<Derived *>(this)); }

  typename Analysis::Result getResult() {
    return typename Analysis::Result(static_cast<Derived *>(this));
  }

protected:
  void setDefaults() {
    ON_CALL(static_cast<Derived &>(*this), run(_, _))
        .WillByDefault(Return(this->getResult()));
    ON_CALL(static_cast<Derived &>(*this), invalidate(_, _, _))
        .WillByDefault(Invoke([](Module &M, const PreservedAnalyses &PA,
                                 ModuleAnalysisManager::Invalidator &) {
          auto PAC = PA.template getChecker<Analysis>();
          return !PAC.preserved() &&
                 !PAC.template preservedSet<AllAnalysesOn<Module>>();
        }));
  }

private:
  friend Derived;
  MockAnalysisHandleBase() = default;
};

class MockModuleAnalysisHandle
    : public MockAnalysisHandleBase<MockModuleAnalysisHandle> {
public:
  MockModuleAnalysisHandle() { setDefaults(); }

  MOCK_METHOD(typename Analysis::Result, run,
              (Module &, ModuleAnalysisManager &));

  MOCK_METHOD(bool, invalidate,
              (Module &, const PreservedAnalyses &,
               ModuleAnalysisManager::Invalidator &));
};

struct PGOInstrumentationGenTest
    : public Test,
      WithParamInterface<std::tuple<StringRef, StringRef>> {
  LLVMContext Ctx;
  ModulePassManager MPM;
  PassBuilder PB;
  MockModuleAnalysisHandle MMAHandle;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  LLVMContext Context;
  std::unique_ptr<Module> M;

  PGOInstrumentationGenTest() {
    MAM.registerPass([&] { return MMAHandle.getAnalysis(); });
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
    MPM.addPass(
        RequireAnalysisPass<MockModuleAnalysisHandle::Analysis, Module>());
    MPM.addPass(PGOInstrumentationGen());
  }

  void parseAssembly(const StringRef IR) {
    SMDiagnostic Error;
    M = parseAssemblyString(IR, Error, Context);
    std::string ErrMsg;
    raw_string_ostream OS(ErrMsg);
    Error.print("", OS);

    // A failure here means that the test itself is buggy.
    if (!M)
      report_fatal_error(OS.str().c_str());
  }
};

static constexpr StringRef CodeWithFuncDefs = R"(
  define i32 @f(i32 %n) {
  entry:
    ret i32 0
  })";

static constexpr StringRef CodeWithFuncDecls = R"(
  declare i32 @f(i32);
)";

static constexpr StringRef CodeWithGlobals = R"(
  @foo.table = internal unnamed_addr constant [1 x ptr] [ptr @f]
  declare i32 @f(i32);
)";

INSTANTIATE_TEST_SUITE_P(
    PGOInstrumetationGenTestSuite, PGOInstrumentationGenTest,
    Values(std::make_tuple(CodeWithFuncDefs, "instrument_function_defs"),
           std::make_tuple(CodeWithFuncDecls, "instrument_function_decls"),
           std::make_tuple(CodeWithGlobals, "instrument_globals")),
    [](const TestParamInfo<PGOInstrumentationGenTest::ParamType> &Info) {
      return std::get<1>(Info.param).str();
    });

TEST_P(PGOInstrumentationGenTest, Instrumented) {
  const StringRef Code = std::get<0>(GetParam());
  parseAssembly(Code);

  ASSERT_THAT(M, NotNull());

  Sequence PassSequence;
  EXPECT_CALL(MMAHandle, run(Ref(*M), _))
      .InSequence(PassSequence)
      .WillOnce(DoDefault());
  EXPECT_CALL(MMAHandle, invalidate(Ref(*M), _, _))
      .InSequence(PassSequence)
      .WillOnce(DoDefault());

  MPM.run(*M, MAM);

  const auto *IRInstrVar =
      M->getNamedGlobal(INSTR_PROF_QUOTE(INSTR_PROF_RAW_VERSION_VAR));
  EXPECT_THAT(IRInstrVar, NotNull());
  EXPECT_FALSE(IRInstrVar->isDeclaration());
}

} // end anonymous namespace
