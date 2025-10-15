#include "llvm/ExecutionEngine/Orc/ReOptimizeLayer.h"
#include "OrcTestCommon.h"
#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRPartitionLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/JITLinkRedirectableSymbolManager.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/MapperJITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/ObjectTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/SelfExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::jitlink;

class ReOptimizeLayerTest : public testing::Test {
public:
  ~ReOptimizeLayerTest() {
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

    // COFF-ARM64 is not supported yet
    auto Triple = JTMB->getTargetTriple();
    if (Triple.isOSBinFormatCOFF() && Triple.isAArch64())
      GTEST_SKIP();

    // SystemZ is not supported yet.
    if (Triple.isSystemZ())
      GTEST_SKIP();

    // 32-bit X86 is not supported yet.
    if (Triple.isX86() && Triple.isArch32Bit())
      GTEST_SKIP();

    if (Triple.isPPC())
      GTEST_SKIP();

    // RISC-V is not supported yet
    if (Triple.isRISCV())
      GTEST_SKIP();

    // ARM is not supported yet.
    if (Triple.isARM())
      GTEST_SKIP();

    auto EPC = SelfExecutorProcessControl::Create();
    if (!EPC) {
      consumeError(EPC.takeError());
      GTEST_SKIP();
    }

    auto DLOrErr = JTMB->getDefaultDataLayoutForTarget();
    if (!DLOrErr) {
      consumeError(DLOrErr.takeError());
      GTEST_SKIP();
    }

    auto PageSize = sys::Process::getPageSize();
    if (!PageSize) {
      consumeError(PageSize.takeError());
      GTEST_SKIP();
    }

    ES = std::make_unique<ExecutionSession>(std::move(*EPC));
    JD = &ES->createBareJITDylib("main");

    ObjLinkingLayer = std::make_unique<ObjectLinkingLayer>(
        *ES, std::make_unique<MapperJITLinkMemoryManager>(
                 10 * 1024 * 1024,
                 std::make_unique<InProcessMemoryMapper>(*PageSize)));
    DL = std::make_unique<DataLayout>(std::move(*DLOrErr));

    auto TM = JTMB->createTargetMachine();
    if (!TM) {
      consumeError(TM.takeError());
      GTEST_SKIP();
    }
    auto CompileFunction =
        std::make_unique<TMOwningSimpleCompiler>(std::move(*TM));
    CompileLayer = std::make_unique<IRCompileLayer>(*ES, *ObjLinkingLayer,
                                                    std::move(CompileFunction));
  }

  Error addIRModule(ResourceTrackerSP RT, ThreadSafeModule TSM) {
    assert(TSM && "Can not add null module");

    TSM.withModuleDo([&](Module &M) { M.setDataLayout(*DL); });

    return ROLayer->add(std::move(RT), std::move(TSM));
  }

  JITDylib *JD{nullptr};
  std::unique_ptr<ExecutionSession> ES;
  std::unique_ptr<ObjectLinkingLayer> ObjLinkingLayer;
  std::unique_ptr<IRCompileLayer> CompileLayer;
  std::unique_ptr<ReOptimizeLayer> ROLayer;
  std::unique_ptr<DataLayout> DL;
};

static Function *createRetFunction(Module *M, StringRef Name,
                                   uint32_t ReturnCode) {
  Function *Result = Function::Create(
      FunctionType::get(Type::getInt32Ty(M->getContext()), {}, false),
      GlobalValue::ExternalLinkage, Name, M);

  BasicBlock *BB = BasicBlock::Create(M->getContext(), Name, Result);
  IRBuilder<> Builder(M->getContext());
  Builder.SetInsertPoint(BB);

  Value *RetValue = ConstantInt::get(M->getContext(), APInt(32, ReturnCode));
  Builder.CreateRet(RetValue);
  return Result;
}

TEST_F(ReOptimizeLayerTest, BasicReOptimization) {
  MangleAndInterner Mangle(*ES, *DL);

  auto &EPC = ES->getExecutorProcessControl();
  EXPECT_THAT_ERROR(JD->define(absoluteSymbols(
                        {{Mangle("__orc_rt_jit_dispatch"),
                          {EPC.getJITDispatchInfo().JITDispatchFunction,
                           JITSymbolFlags::Exported}},
                         {Mangle("__orc_rt_jit_dispatch_ctx"),
                          {EPC.getJITDispatchInfo().JITDispatchContext,
                           JITSymbolFlags::Exported}},
                         {Mangle("__orc_rt_reoptimize_tag"),
                          {ExecutorAddr(), JITSymbolFlags::Exported}}})),
                    Succeeded());

  auto RM = JITLinkRedirectableSymbolManager::Create(*ObjLinkingLayer);
  EXPECT_THAT_ERROR(RM.takeError(), Succeeded());

  ROLayer = std::make_unique<ReOptimizeLayer>(*ES, *DL, *CompileLayer, **RM);
  ROLayer->setReoptimizeFunc(
      [&](ReOptimizeLayer &Parent,
          ReOptimizeLayer::ReOptMaterializationUnitID MUID, unsigned CurVerison,
          ResourceTrackerSP OldRT, ThreadSafeModule &TSM) {
        TSM.withModuleDo([&](Module &M) {
          for (auto &F : M) {
            if (F.isDeclaration())
              continue;
            for (auto &B : F) {
              for (auto &I : B) {
                if (ReturnInst *Ret = dyn_cast<ReturnInst>(&I)) {
                  Value *RetValue =
                      ConstantInt::get(M.getContext(), APInt(32, 53));
                  Ret->setOperand(0, RetValue);
                }
              }
            }
          }
        });
        return Error::success();
      });
  EXPECT_THAT_ERROR(ROLayer->reigsterRuntimeFunctions(*JD), Succeeded());

  auto Ctx = std::make_unique<LLVMContext>();
  auto M = std::make_unique<Module>("<main>", *Ctx);
  M->setTargetTriple(Triple(sys::getProcessTriple()));

  (void)createRetFunction(M.get(), "main", 42);

  EXPECT_THAT_ERROR(addIRModule(JD->getDefaultResourceTracker(),
                                ThreadSafeModule(std::move(M), std::move(Ctx))),
                    Succeeded());

  auto Result = cantFail(ES->lookup({JD}, Mangle("main")));
  auto FuncPtr = Result.getAddress().toPtr<int (*)()>();
  for (size_t I = 0; I <= ReOptimizeLayer::CallCountThreshold; I++)
    EXPECT_EQ(FuncPtr(), 42);
  EXPECT_EQ(FuncPtr(), 53);
}
