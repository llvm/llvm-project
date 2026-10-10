#include "llvm/Target/TargetOptions.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace llvm {
  void initializeTestPassPass(PassRegistry &);
}

namespace {

void initLLVM() {
  InitializeAllTargetInfos();
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  PassRegistry *Registry = PassRegistry::getPassRegistry();
  initializeCore(*Registry);
  initializeCodeGen(*Registry);
}

/// Create a TargetMachine. We need a target that doesn't have IPRA enabled by
/// default. That turns out to be all targets at the moment, so just use X86.
std::unique_ptr<TargetMachine> createTargetMachine(bool EnableIPRA) {
  Triple TargetTriple("x86_64--");
  std::string Error;
  const Target *T = TargetRegistry::lookupTarget("", TargetTriple, Error);
  if (!T)
    return nullptr;

  TargetOptions Options;
  Options.EnableIPRA = EnableIPRA;
  return std::unique_ptr<TargetMachine>(
      T->createTargetMachine(TargetTriple, "", "", Options, std::nullopt,
                             std::nullopt, CodeGenOptLevel::Aggressive));
}

typedef std::function<void(bool)> TargetOptionsTest;

static void targetOptionsTest(bool EnableIPRA) {
  std::unique_ptr<TargetMachine> TM = createTargetMachine(EnableIPRA);
  // This test is designed for the X86 backend; stop if it is not available.
  if (!TM)
    GTEST_SKIP();
  legacy::PassManager PM;

  TargetPassConfig *TPC = TM->createPassConfig(PM);
  (void)TPC;

  ASSERT_TRUE(TM->Options.EnableIPRA == EnableIPRA);

  delete TPC;
}

} // End of anonymous namespace.

TEST(TargetOptionsTest, IPRASetToOff) {
  targetOptionsTest(false);
}

TEST(TargetOptionsTest, IPRASetToOn) {
  targetOptionsTest(true);
}

TEST(TargetOptionsTest, SubtargetCopyPreservesHwMode) {
  for (StringRef TripleName :
       {"x86_64-unknown-linux-gnu", "riscv64-unknown-elf"}) {
    SCOPED_TRACE(TripleName);
    Triple TT(TripleName);
    std::string Error;
    const Target *TheTarget = TargetRegistry::lookupTarget(TT, Error);
    if (!TheTarget)
      continue;
    TargetOptions Options;
    std::unique_ptr<TargetMachine> TM(
        TheTarget->createTargetMachine(TT, "", "", Options, std::nullopt));
    ASSERT_TRUE(TM);
    LLVMContext LLVMCtx;
    Module M("test", LLVMCtx);
    Function *F =
        Function::Create(FunctionType::get(Type::getVoidTy(LLVMCtx), false),
                         Function::ExternalLinkage, "foo", M);
    const TargetSubtargetInfo *TST = TM->getSubtargetImpl(*F);
    ASSERT_TRUE(TST);
    ASSERT_NE(TST->getHwMode(), 0u);
    ASSERT_NE(TST->getHwModeSet(), 0u);
    MCContext Ctx(TT, TM->getMCAsmInfo(), TM->getMCRegisterInfo(), *TST);
    MCSubtargetInfo &TSTCopy = Ctx.getSubtargetCopy(*TST);
    // FIXME: MCContext::getSubtargetCopy invokes the base MCSubtargetInfo copy
    // constructor, resetting the vtable to MCSubtargetInfo and losing the
    // <Target>GenSubtargetInfo overrides for getHwMode() and getHwModeSet().
    EXPECT_NE(TSTCopy.getHwMode(), TST->getHwMode());
    EXPECT_NE(TSTCopy.getHwModeSet(), TST->getHwModeSet());
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  initLLVM();
  return RUN_ALL_TESTS();
}
