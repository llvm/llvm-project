#include "OrcTestCommon.h"
#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/JITLinkRedirectableSymbolManager.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::jitlink;

static int initialTarget() { return 42; }
static int middleTarget() { return 13; }
static int finalTarget() { return 53; }

class JITLinkRedirectionManagerTest : public testing::Test {
public:
  ~JITLinkRedirectionManagerTest() {
    if (ES)
      if (auto Err = ES->endSession())
        ES->reportError(std::move(Err));
  }

protected:
  void SetUp() override {
    auto JTMB = JITTargetMachineBuilder::detectHost();
    // Bail out if we can not detect the host.
    if (!JTMB) {
      consumeError(JTMB.takeError());
      GTEST_SKIP();
    }
    auto DLOrErr = JTMB->getDefaultDataLayoutForTarget();
    if (!DLOrErr) {
      consumeError(DLOrErr.takeError());
      GTEST_SKIP();
    }

    ES = std::make_unique<ExecutionSession>(
        std::make_unique<UnsupportedExecutorProcessControl>(
            nullptr, nullptr, JTMB->getTargetTriple().getTriple()));
    JD = &ES->createBareJITDylib("main");
    ObjLinkingLayer = std::make_unique<ObjectLinkingLayer>(
        *ES, std::make_unique<InProcessMemoryManager>(16384));
    DL = std::make_unique<DataLayout>(std::move(*DLOrErr));
  }
  JITDylib *JD{nullptr};
  std::unique_ptr<ExecutionSession> ES;
  std::unique_ptr<ObjectLinkingLayer> ObjLinkingLayer;
  std::unique_ptr<DataLayout> DL;
};

TEST_F(JITLinkRedirectionManagerTest, BasicRedirectionOperation) {
  auto RM = JITLinkRedirectableSymbolManager::Create(*ObjLinkingLayer, *JD);
  // Bail out if we can not create
  if (!RM) {
    consumeError(RM.takeError());
    GTEST_SKIP();
  }

  auto DefineTarget = [&](StringRef TargetName, ExecutorAddr Addr) {
    SymbolStringPtr Target = ES->intern(TargetName);
    cantFail(JD->define(std::make_unique<SimpleMaterializationUnit>(
        SymbolFlagsMap({{Target, JITSymbolFlags::Exported}}),
        [&](std::unique_ptr<MaterializationResponsibility> R) -> void {
          // No dependencies registered, can't fail.
          cantFail(
              R->notifyResolved({{Target, {Addr, JITSymbolFlags::Exported}}}));
          cantFail(R->notifyEmitted({}));
        })));
    return cantFail(ES->lookup({JD}, TargetName));
  };

  auto InitialTarget =
      DefineTarget("InitialTarget", ExecutorAddr::fromPtr(&initialTarget));
  auto MiddleTarget =
      DefineTarget("MiddleTarget", ExecutorAddr::fromPtr(&middleTarget));
  auto FinalTarget =
      DefineTarget("FinalTarget", ExecutorAddr::fromPtr(&finalTarget));

  auto RedirectableSymbol = ES->intern("RedirectableTarget");
  EXPECT_THAT_ERROR(
      (*RM)->createRedirectableSymbols(JD->getDefaultResourceTracker(),
                                       {{RedirectableSymbol, InitialTarget}}),
      Succeeded());
  auto RTDef = cantFail(ES->lookup({JD}, RedirectableSymbol));

  auto RTPtr = RTDef.getAddress().toPtr<int (*)()>();
  auto Result = RTPtr();
  EXPECT_EQ(Result, 42) << "Failed to call initial target";

  EXPECT_THAT_ERROR((*RM)->redirect(*JD, {{RedirectableSymbol, MiddleTarget}}),
                    Succeeded());
  Result = RTPtr();
  EXPECT_EQ(Result, 13) << "Failed to call middle redirected target";

  EXPECT_THAT_ERROR((*RM)->redirect(*JD, {{RedirectableSymbol, FinalTarget}}),
                    Succeeded());
  Result = RTPtr();
  EXPECT_EQ(Result, 53) << "Failed to call redirected target";
}
