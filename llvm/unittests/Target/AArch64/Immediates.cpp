#include "AArch64Subtarget.h"
#include "AArch64TargetMachine.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"

#include "gtest/gtest.h"
#include <initializer_list>
#include <memory>

using namespace llvm;

namespace {

struct TestCase {
  int64_t Imm;
  bool Result;
};

const std::initializer_list<TestCase> Tests = {
    // ScalableImm, Result
    // No change, easily 'supported'
    {0, true},

    // addvl increments by whole registers, range [-32,31]
    // +(16 * vscale), one register's worth
    {16, true},
    // +(8 * vscale), half a register's worth
    {8, false},
    // -(32 * 16 * vscale)
    {-512, true},
    // -(33 * 16 * vscale)
    {-528, false},
    // +(31 * 16 * vscale)
    {496, true},
    // +(32 * 16 * vscale)
    {512, false},
};
} // namespace

TEST(Immediates, Immediates) {
  LLVMInitializeAArch64TargetInfo();
  LLVMInitializeAArch64Target();
  LLVMInitializeAArch64TargetMC();

  std::string Error;
  auto TT = Triple::normalize("aarch64");
  const Target *T = TargetRegistry::lookupTarget(TT, Error);

  std::unique_ptr<TargetMachine> TM(T->createTargetMachine(
      TT, "generic", "+sve", TargetOptions(), std::nullopt, std::nullopt,
      CodeGenOptLevel::Default));
  AArch64Subtarget ST(TM->getTargetTriple(), TM->getTargetCPU(),
                      TM->getTargetCPU(), TM->getTargetFeatureString(), *TM,
                      true);

  auto *TLI = ST.getTargetLowering();

  for (const auto &Test : Tests) {
    ASSERT_EQ(TLI->isLegalAddScalableImmediate(Test.Imm), Test.Result);
  }
}
