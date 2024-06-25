#include "OrcTestCommon.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/LazyReexports.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;

class LazyReexportsTest : public CoreAPIsBasedStandardTest {};

static int dummyTarget() { return 42; }

TEST_F(LazyReexportsTest, BasicLocalCallThroughManagerOperation) {
  // Create a callthrough manager for the host (if possible) and verify that
  // a call to the lazy call-through:
  // (1) Materializes the MU. This verifies that the symbol was looked up, and
  //     that we didn't arrive at the target via some other path
  // (2) Returns the expected value (which we take as proof that the call
  //     reached the target).

  auto JTMB = JITTargetMachineBuilder::detectHost();

  // Bail out if we can not detect the host.
  if (!JTMB) {
    consumeError(JTMB.takeError());
    GTEST_SKIP();
  }

  // Bail out if we can not build a local call-through manager.
  auto LCTM = createLocalLazyCallThroughManager(JTMB->getTargetTriple(), ES,
                                                ExecutorAddr());
  if (!LCTM) {
    consumeError(LCTM.takeError());
    GTEST_SKIP();
  }

  auto DummyTarget = ES.intern("DummyTarget");

  bool DummyTargetMaterialized = false;

  cantFail(JD.define(std::make_unique<SimpleMaterializationUnit>(
      SymbolFlagsMap({{DummyTarget, JITSymbolFlags::Exported}}),
      [&](std::unique_ptr<MaterializationResponsibility> R) {
        DummyTargetMaterialized = true;
        // No dependencies registered, can't fail.
        cantFail(R->notifyResolved({{DummyTarget,
                                     {ExecutorAddr::fromPtr(&dummyTarget),
                                      JITSymbolFlags::Exported}}}));
        cantFail(R->notifyEmitted({}));
      })));

  unsigned NotifyResolvedCount = 0;
  auto NotifyResolved = [&](ExecutorAddr ResolvedAddr) {
    ++NotifyResolvedCount;
    return Error::success();
  };

  auto CallThroughTrampoline = cantFail((*LCTM)->getCallThroughTrampoline(
      JD, DummyTarget, std::move(NotifyResolved)));

  auto CTTPtr = CallThroughTrampoline.toPtr<int (*)()>();

  // Call twice to verify nothing unexpected happens on redundant calls.
  auto Result = CTTPtr();
  (void)CTTPtr();

  EXPECT_TRUE(DummyTargetMaterialized)
      << "CallThrough did not materialize target";
  EXPECT_EQ(NotifyResolvedCount, 1U)
      << "CallThrough should have generated exactly one 'NotifyResolved' call";
  EXPECT_EQ(Result, 42) << "Failed to call through to target";
}
