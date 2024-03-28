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
    // -(32 * 16 * vscale)
    {-512, true},
    // -(33 * 16 * vscale)
    {-528, false},
    // +(31 * 16 * vscale)
    {496, true},
    // +(32 * 16 * vscale)
    {512, false},

    // inc[h|w|d] increments by the number of 16/32/64bit elements in a
    // register. mult_imm is in the range [1,16]
    // +(mult_imm * num_elts * vscale)
    // +(1 * 8 * vscale), 16 bit
    {8, true},
    // +(15 * 8 * vscale), 16 bit
    {120, true},
    // +(1 * 4 * vscale), 32 bit
    {4, true},
    // +(7 * 4 * vscale), 32 bit
    {28, true},
    // +(1 * 2 * vscale), 64 bit
    {2, true},
    // +(13 * 2 * vscale), 64 bit
    {26, true},
    // +(17 * 8 * vscale), 16 bit, out of range.
    {136, false},
    // +(19 * 2 * vscale), 64 bit, out of range.
    {38, false},
    // +(21 * 4 * vscale), 32 bit, out of range.
    {84, false},

    // dec[h|w|d] -- Same as above, but negative.
    // -(mult_imm * num_elts * vscale)
    // -(1 * 8 * vscale), 16 bit
    {-8, true},
    // -(15 * 8 * vscale), 16 bit
    {-120, true},
    // -(1 * 4 * vscale), 32 bit
    {-4, true},
    // -(7 * 4 * vscale), 32 bit
    {-28, true},
    // -(1 * 2 * vscale), 64 bit
    {-2, true},
    // -(13 * 2 * vscale), 64 bit
    {-26, true},
    // -(17 * 8 * vscale), 16 bit, out of range.
    {-136, false},
    // -(19 * 2 * vscale), 64 bit, out of range.
    {-38, false},
    // -(21 * 4 * vscale), 32 bit, out of range.
    {-84, false},

    // Invalid; not divisible by the above powers of 2.
    {5, false},
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
      TT, "generic", "+sve2", TargetOptions(), std::nullopt, std::nullopt,
      CodeGenOptLevel::Default));
  AArch64Subtarget ST(TM->getTargetTriple(), TM->getTargetCPU(),
                      TM->getTargetCPU(), TM->getTargetFeatureString(), *TM,
                      true);

  auto *TLI = ST.getTargetLowering();

  for (const auto &Test : Tests) {
    ASSERT_EQ(TLI->isLegalAddScalableImmediate(Test.Imm), Test.Result);
  }
}
