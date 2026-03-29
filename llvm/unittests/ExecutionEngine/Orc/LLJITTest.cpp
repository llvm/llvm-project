#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "OrcTestCommon.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Testing/Support/Error.h"

using namespace llvm;
using namespace llvm::orc;

namespace {

static ThreadSafeModule parseModule(llvm::StringRef Source,
                                    llvm::StringRef Name) {
  auto Ctx = std::make_unique<LLVMContext>();
  SMDiagnostic Err;
  auto M = parseIR(MemoryBufferRef(Source, Name), Err, *Ctx);
  if (!M) {
    Err.print("Testcase source failed to parse: ", errs());
    exit(1);
  }
  return ThreadSafeModule(std::move(M), std::move(Ctx));
}

TEST(LLJITTest, CleanupFailedInitializers) {
  OrcNativeTarget::initialize();
  auto J = cantFail(LLJITBuilder().create());
  auto &JD = J->getMainJITDylib();

  // ctor references undefined symbol 'testing'
  auto TSM_A = parseModule(R"(
    declare void @testing()

    define internal void @ctor_A() {
      call void @testing()
      ret void
    }

    @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [
      { i32, ptr, ptr } { i32 65535, ptr @ctor_A, ptr null }
    ]
  )",
                           "A");

  cantFail(J->addIRModule(std::move(TSM_A)));

  // Initialize fails: "Symbols not found: [ testing ]"
  EXPECT_THAT_ERROR(J->initialize(JD), Failed());

  // Clean module should succeed if A's bookkeeping was cleaned up
  auto TSM_B = parseModule(R"(
    @i = global i32 42
  )",
                           "B");

  cantFail(J->addIRModule(std::move(TSM_B)));

  EXPECT_THAT_ERROR(J->initialize(JD), Succeeded());
}

TEST(LLJITTest, RepeatedInitializationFailures) {
  // Consecutive failures don't accumulate stale state
  OrcNativeTarget::initialize();
  auto J = cantFail(LLJITBuilder().create());
  auto &JD = J->getMainJITDylib();

  // First failure
  auto TSM_A = parseModule(R"(
    declare void @undefined_a()
    define internal void @ctor_A() {
      call void @undefined_a()
      ret void
    }
    @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [
      { i32, ptr, ptr } { i32 65535, ptr @ctor_A, ptr null }
    ]
  )",
                           "A");
  cantFail(J->addIRModule(std::move(TSM_A)));
  EXPECT_THAT_ERROR(J->initialize(JD), Failed());

  // Second failure
  auto TSM_B = parseModule(R"(
    declare void @undefined_b()
    define internal void @ctor_B() {
      call void @undefined_b()
      ret void
    }
    @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [
      { i32, ptr, ptr } { i32 65535, ptr @ctor_B, ptr null }
    ]
  )",
                           "B");
  cantFail(J->addIRModule(std::move(TSM_B)));
  EXPECT_THAT_ERROR(J->initialize(JD), Failed());

  // Should succeed, both A and B cleaned up
  auto TSM_C = parseModule(R"(
    @x = global i32 0
  )",
                           "C");
  cantFail(J->addIRModule(std::move(TSM_C)));
  EXPECT_THAT_ERROR(J->initialize(JD), Succeeded());
}

} // anonymous namespace
