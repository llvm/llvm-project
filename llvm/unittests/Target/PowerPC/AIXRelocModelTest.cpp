#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class AIXRelocModelTest : public ::testing::Test {
protected:
  static void SetUpTestCase() {
    LLVMInitializePowerPCTargetInfo();
    LLVMInitializePowerPCTarget();
    LLVMInitializePowerPCTargetMC();
  }
};

TEST_F(AIXRelocModelTest, DefalutToPIC) {
  Triple TheTriple(/*ArchStr*/ "powerpc", /*VendorStr*/ "", /*OSStr*/ "aix");
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget("", TheTriple, Error);
  ASSERT_TRUE(TheTarget) << Error;

  TargetOptions Options;
  // Create a TargetMachine for powerpc--aix target, and deliberately leave its
  // relocation model unset.
  std::unique_ptr<TargetMachine> Target(TheTarget->createTargetMachine(
      /*TT*/ TheTriple.getTriple(), /*CPU*/ "", /*Features*/ "",
      /*Options*/ Options, /*RM*/ std::nullopt, /*CM*/ std::nullopt,
      /*OL*/ CodeGenOptLevel::Default));
  ASSERT_TRUE(Target) << "Could not allocate target machine!";

  // The relocation model on AIX should be forced to PIC regardless.
  EXPECT_TRUE(Target->getRelocationModel() == Reloc::PIC_);
}

} // end of anonymous namespace
