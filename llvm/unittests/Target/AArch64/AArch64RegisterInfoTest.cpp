#include "AArch64RegisterInfo.h"
#include "AArch64InstrInfo.h"
#include "AArch64Subtarget.h"
#include "AArch64TargetMachine.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

#include "gtest/gtest.h"

#define GET_COMPUTE_FEATURES
#include "AArch64GenInstrInfo.inc"

using namespace llvm;

namespace {

std::unique_ptr<TargetMachine> createTargetMachine(const std::string &CPU) {
  auto TT(Triple::normalize("aarch64--"));

  LLVMInitializeAArch64TargetInfo();
  LLVMInitializeAArch64Target();
  LLVMInitializeAArch64TargetMC();

  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(TT, Error);

  return std::unique_ptr<TargetMachine>(
      TheTarget->createTargetMachine(TT, CPU, "", TargetOptions(), std::nullopt,
                                     std::nullopt, CodeGenOptLevel::Default));
}

std::unique_ptr<AArch64InstrInfo> createInstrInfo(TargetMachine *TM) {
  AArch64Subtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                      std::string(TM->getTargetCPU()),
                      std::string(TM->getTargetFeatureString()), *TM, true);
  return std::make_unique<AArch64InstrInfo>(ST);
}

TEST(AArch64LaneBitmasks, SubRegs) {
  std::unique_ptr<TargetMachine> TM = createTargetMachine("");
  ASSERT_TRUE(TM);

  std::unique_ptr<AArch64InstrInfo> II = createInstrInfo(TM.get());
  ASSERT_TRUE(II);

  const AArch64RegisterInfo &TRI = II->getRegisterInfo();

  // Test that the lane masks for the subregisters 'bsub, hsub, ssub, etc'
  // are composed correctly.
  EXPECT_EQ(TRI.getSubRegIndexLaneMask(AArch64::bsub) |
                TRI.getSubRegIndexLaneMask(AArch64::bsub_hi),
            TRI.getSubRegIndexLaneMask(AArch64::hsub));

  EXPECT_EQ(TRI.getSubRegIndexLaneMask(AArch64::hsub) |
                TRI.getSubRegIndexLaneMask(AArch64::hsub_hi),
            TRI.getSubRegIndexLaneMask(AArch64::ssub));

  EXPECT_EQ(TRI.getSubRegIndexLaneMask(AArch64::ssub) |
                TRI.getSubRegIndexLaneMask(AArch64::ssub_hi),
            TRI.getSubRegIndexLaneMask(AArch64::dsub));

  EXPECT_EQ(TRI.getSubRegIndexLaneMask(AArch64::dsub) |
                TRI.getSubRegIndexLaneMask(AArch64::dsub_hi),
            TRI.getSubRegIndexLaneMask(AArch64::zsub));

  EXPECT_EQ(TRI.getSubRegIndexLaneMask(AArch64::zsub) |
                TRI.getSubRegIndexLaneMask(AArch64::zsub_hi),
            TRI.getSubRegIndexLaneMask(AArch64::zsub0));

  // Test that the lane masks for tuples are composed correctly.
  EXPECT_EQ(TRI.getSubRegIndexLaneMask(AArch64::dsub1_then_bsub) |
                TRI.getSubRegIndexLaneMask(AArch64::dsub1_then_bsub_hi),
            TRI.getSubRegIndexLaneMask(AArch64::dsub1_then_hsub));

  EXPECT_EQ(TRI.getSubRegIndexLaneMask(AArch64::dsub1_then_hsub) |
                TRI.getSubRegIndexLaneMask(AArch64::dsub1_then_hsub_hi),
            TRI.getSubRegIndexLaneMask(AArch64::dsub1_then_ssub));

  EXPECT_EQ(TRI.getSubRegIndexLaneMask(AArch64::dsub1_then_ssub) |
                TRI.getSubRegIndexLaneMask(AArch64::dsub1_then_ssub_hi),
            TRI.getSubRegIndexLaneMask(AArch64::dsub1));

  EXPECT_EQ(TRI.getSubRegIndexLaneMask(AArch64::dsub1) |
                TRI.getSubRegIndexLaneMask(AArch64::qsub1_then_dsub_hi),
            TRI.getSubRegIndexLaneMask(AArch64::qsub1));

  EXPECT_EQ(TRI.getSubRegIndexLaneMask(AArch64::sub_32) |
                TRI.getSubRegIndexLaneMask(AArch64::sub_32_hi),
            TRI.getSubRegIndexLaneMask(AArch64::sube64));

  EXPECT_EQ(TRI.getSubRegIndexLaneMask(AArch64::subo64_then_sub_32) |
                TRI.getSubRegIndexLaneMask(AArch64::subo64_then_sub_32_hi),
            TRI.getSubRegIndexLaneMask(AArch64::subo64));

  // Test that there is no overlap between different (sub)registers
  // in a tuple.
  EXPECT_EQ(TRI.getSubRegIndexLaneMask(AArch64::dsub0) &
                TRI.getSubRegIndexLaneMask(AArch64::dsub1) &
                TRI.getSubRegIndexLaneMask(AArch64::dsub2) &
                TRI.getSubRegIndexLaneMask(AArch64::dsub3),
            LaneBitmask::getNone());

  EXPECT_EQ(TRI.getSubRegIndexLaneMask(AArch64::qsub0) &
                TRI.getSubRegIndexLaneMask(AArch64::qsub1) &
                TRI.getSubRegIndexLaneMask(AArch64::qsub2) &
                TRI.getSubRegIndexLaneMask(AArch64::qsub3),
            LaneBitmask::getNone());

  EXPECT_EQ(TRI.getSubRegIndexLaneMask(AArch64::zsub0) &
                TRI.getSubRegIndexLaneMask(AArch64::zsub1) &
                TRI.getSubRegIndexLaneMask(AArch64::zsub2) &
                TRI.getSubRegIndexLaneMask(AArch64::zsub3),
            LaneBitmask::getNone());

  EXPECT_EQ(TRI.getSubRegIndexLaneMask(AArch64::sube32) &
                TRI.getSubRegIndexLaneMask(AArch64::subo32),
            LaneBitmask::getNone());

  EXPECT_EQ(TRI.getSubRegIndexLaneMask(AArch64::sube64) &
                TRI.getSubRegIndexLaneMask(AArch64::subo64),
            LaneBitmask::getNone());

  // Test that getting a subregister results in the expected subregister.
  EXPECT_EQ(TRI.getSubReg(AArch64::Z0_Z8, AArch64::bsub), AArch64::B0);
  EXPECT_EQ(TRI.getSubReg(AArch64::Z0_Z8, AArch64::hsub), AArch64::H0);
  EXPECT_EQ(TRI.getSubReg(AArch64::Z0_Z8, AArch64::ssub), AArch64::S0);
  EXPECT_EQ(TRI.getSubReg(AArch64::Z0_Z8, AArch64::dsub), AArch64::D0);
  EXPECT_EQ(TRI.getSubReg(AArch64::Z0_Z8, AArch64::zsub), AArch64::Q0);
  EXPECT_EQ(TRI.getSubReg(AArch64::Z0_Z8, AArch64::zsub0), AArch64::Z0);

  EXPECT_EQ(TRI.getSubReg(AArch64::Z0_Z8, AArch64::dsub1_then_bsub),
            AArch64::B8);
  EXPECT_EQ(TRI.getSubReg(AArch64::Z0_Z8, AArch64::dsub1_then_hsub),
            AArch64::H8);
  EXPECT_EQ(TRI.getSubReg(AArch64::Z0_Z8, AArch64::dsub1_then_ssub),
            AArch64::S8);
  EXPECT_EQ(TRI.getSubReg(AArch64::Z0_Z8, AArch64::dsub1), AArch64::D8);
  EXPECT_EQ(TRI.getSubReg(AArch64::Z0_Z8, AArch64::qsub1), AArch64::Q8);
  EXPECT_EQ(TRI.getSubReg(AArch64::Z0_Z8, AArch64::zsub1), AArch64::Z8);

  EXPECT_EQ(TRI.getSubReg(AArch64::X0_X1, AArch64::sube64), AArch64::X0);
  EXPECT_EQ(TRI.getSubReg(AArch64::X0_X1, AArch64::subo64), AArch64::X1);
  EXPECT_EQ(TRI.getSubReg(AArch64::X0_X1, AArch64::sub_32), AArch64::W0);
  EXPECT_EQ(TRI.getSubReg(AArch64::X0_X1, AArch64::subo64_then_sub_32),
            AArch64::W1);
}

} // namespace
