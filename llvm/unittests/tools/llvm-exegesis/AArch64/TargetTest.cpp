#include "Target.h"

#include <cassert>
#include <memory>

#include "MCTargetDesc/AArch64MCTargetDesc.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace exegesis {

void InitializeAArch64ExegesisTarget();

namespace {

using testing::Gt;
using testing::NotNull;
using testing::SizeIs;

constexpr const char kTriple[] = "aarch64-unknown-linux";

class AArch64TargetTest : public ::testing::Test {
protected:
  AArch64TargetTest()
      : ExegesisTarget_(ExegesisTarget::lookup(llvm::Triple(kTriple))) {
    EXPECT_THAT(ExegesisTarget_, NotNull());
    std::string error;
    Target_ = llvm::TargetRegistry::lookupTarget(kTriple, error);
    EXPECT_THAT(Target_, NotNull());
  }
  static void SetUpTestCase() {
    LLVMInitializeAArch64TargetInfo();
    LLVMInitializeAArch64Target();
    LLVMInitializeAArch64TargetMC();
    InitializeAArch64ExegesisTarget();
  }

  const llvm::Target *Target_;
  const ExegesisTarget *const ExegesisTarget_;
};

TEST_F(AArch64TargetTest, SetRegToConstant) {
  const std::unique_ptr<llvm::MCSubtargetInfo> STI(
      Target_->createMCSubtargetInfo(kTriple, "generic", ""));
  // The AArch64 target currently doesn't know how to set register values
  const auto Insts = ExegesisTarget_->setRegToConstant(*STI, llvm::AArch64::X0);
  EXPECT_THAT(Insts, SizeIs(0));
}

} // namespace
} // namespace exegesis
