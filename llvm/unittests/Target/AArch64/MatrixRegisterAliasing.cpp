#include "AArch64Subtarget.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {
std::unique_ptr<LLVMTargetMachine> createTargetMachine() {
  auto TT(Triple::normalize("aarch64--"));
  std::string CPU("generic");
  std::string FS("+sme");

  LLVMInitializeAArch64TargetInfo();
  LLVMInitializeAArch64Target();
  LLVMInitializeAArch64TargetMC();

  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(TT, Error);

  return std::unique_ptr<LLVMTargetMachine>(
      static_cast<LLVMTargetMachine *>(TheTarget->createTargetMachine(
          TT, CPU, FS, TargetOptions(), None, None, CodeGenOpt::Default)));
}

std::unique_ptr<AArch64InstrInfo> createInstrInfo(TargetMachine *TM) {
  AArch64Subtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                      std::string(TM->getTargetCPU()),
                      std::string(TM->getTargetFeatureString()), *TM,
                      /* isLittle */ false);
  return std::make_unique<AArch64InstrInfo>(ST);
}

TEST(MatrixRegisterAliasing, Aliasing) {
  std::unique_ptr<LLVMTargetMachine> TM = createTargetMachine();
  ASSERT_TRUE(TM);
  std::unique_ptr<AArch64InstrInfo> II = createInstrInfo(TM.get());

  const AArch64RegisterInfo &TRI = II->getRegisterInfo();

  // za overlaps with za.b
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZA, AArch64::ZAB0));

  // za0.b overlaps with all tiles
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAB0, AArch64::ZAQ0));
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAB0, AArch64::ZAQ15));
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAB0, AArch64::ZAD0));
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAB0, AArch64::ZAD7));
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAB0, AArch64::ZAS0));
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAB0, AArch64::ZAS3));
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAB0, AArch64::ZAH0));
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAB0, AArch64::ZAH1));

  // za0.h aliases with za0.q, za2.q, ..
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAH0, AArch64::ZAQ0));
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAH0, AArch64::ZAQ2));
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAH0, AArch64::ZAQ4));
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAH0, AArch64::ZAQ6));
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAH0, AArch64::ZAQ8));
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAH0, AArch64::ZAQ10));
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAH0, AArch64::ZAQ12));
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAH0, AArch64::ZAQ14));

  // za1.h aliases with za1.q, za3.q, ...
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAH1, AArch64::ZAQ1));
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAH1, AArch64::ZAQ3));
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAH1, AArch64::ZAQ5));
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAH1, AArch64::ZAQ7));
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAH1, AArch64::ZAQ9));
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAH1, AArch64::ZAQ11));
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAH1, AArch64::ZAQ13));
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAH1, AArch64::ZAQ15));

  // za1.h doesn't alias with za0.q, za2.q, ..
  ASSERT_FALSE(TRI.regsOverlap(AArch64::ZAH1, AArch64::ZAQ0));
  ASSERT_FALSE(TRI.regsOverlap(AArch64::ZAH1, AArch64::ZAQ2));
  ASSERT_FALSE(TRI.regsOverlap(AArch64::ZAH1, AArch64::ZAQ4));
  ASSERT_FALSE(TRI.regsOverlap(AArch64::ZAH1, AArch64::ZAQ6));
  ASSERT_FALSE(TRI.regsOverlap(AArch64::ZAH1, AArch64::ZAQ8));
  ASSERT_FALSE(TRI.regsOverlap(AArch64::ZAH1, AArch64::ZAQ10));
  ASSERT_FALSE(TRI.regsOverlap(AArch64::ZAH1, AArch64::ZAQ12));
  ASSERT_FALSE(TRI.regsOverlap(AArch64::ZAH1, AArch64::ZAQ14));

  // za0.h doesn't alias with za1.q, za3.q, ..
  ASSERT_FALSE(TRI.regsOverlap(AArch64::ZAH0, AArch64::ZAQ1));
  ASSERT_FALSE(TRI.regsOverlap(AArch64::ZAH0, AArch64::ZAQ3));
  ASSERT_FALSE(TRI.regsOverlap(AArch64::ZAH0, AArch64::ZAQ5));
  ASSERT_FALSE(TRI.regsOverlap(AArch64::ZAH0, AArch64::ZAQ7));
  ASSERT_FALSE(TRI.regsOverlap(AArch64::ZAH0, AArch64::ZAQ9));
  ASSERT_FALSE(TRI.regsOverlap(AArch64::ZAH0, AArch64::ZAQ11));
  ASSERT_FALSE(TRI.regsOverlap(AArch64::ZAH0, AArch64::ZAQ13));
  ASSERT_FALSE(TRI.regsOverlap(AArch64::ZAH0, AArch64::ZAQ15));

  // za0.s aliases with za0.q, za4.q, za8.q, za12.q
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAS0, AArch64::ZAQ0));
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAS0, AArch64::ZAQ4));
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAS0, AArch64::ZAQ8));
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAS0, AArch64::ZAQ12));

  // za1.s aliases with za1.q, za5.q, za9.q, za13.q
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAS1, AArch64::ZAQ1));
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAS1, AArch64::ZAQ5));
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAS1, AArch64::ZAQ9));
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAS1, AArch64::ZAQ13));

  // za0.s doesn't alias with za1.q, za5.q, za9.q, za13.q
  ASSERT_FALSE(TRI.regsOverlap(AArch64::ZAS0, AArch64::ZAQ1));
  ASSERT_FALSE(TRI.regsOverlap(AArch64::ZAS0, AArch64::ZAQ5));
  ASSERT_FALSE(TRI.regsOverlap(AArch64::ZAS0, AArch64::ZAQ9));
  ASSERT_FALSE(TRI.regsOverlap(AArch64::ZAS0, AArch64::ZAQ13));

  // za1.s doesn't alias with za0.q, za4.q, za8.q, za12.q
  ASSERT_FALSE(TRI.regsOverlap(AArch64::ZAS1, AArch64::ZAQ0));
  ASSERT_FALSE(TRI.regsOverlap(AArch64::ZAS1, AArch64::ZAQ4));
  ASSERT_FALSE(TRI.regsOverlap(AArch64::ZAS1, AArch64::ZAQ8));
  ASSERT_FALSE(TRI.regsOverlap(AArch64::ZAS1, AArch64::ZAQ12));

  // za0.d aliases za0.q and za8.q
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAD0, AArch64::ZAQ0));
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAD0, AArch64::ZAQ8));

  // za1.d aliases za1.q and za9.q
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAD1, AArch64::ZAQ1));
  ASSERT_TRUE(TRI.regsOverlap(AArch64::ZAD1, AArch64::ZAQ9));

  // za0.d doesn't alias with za1.q and za9.q
  ASSERT_FALSE(TRI.regsOverlap(AArch64::ZAD0, AArch64::ZAQ1));
  ASSERT_FALSE(TRI.regsOverlap(AArch64::ZAD0, AArch64::ZAQ9));

  // za1.d doesn't alias with za0.q and za8.q
  ASSERT_FALSE(TRI.regsOverlap(AArch64::ZAD1, AArch64::ZAQ0));
  ASSERT_FALSE(TRI.regsOverlap(AArch64::ZAD1, AArch64::ZAQ8));
}

} // end anonymous namespace
