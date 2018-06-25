#include "Target.h"

#include <cassert>
#include <memory>

#include "MCTargetDesc/X86MCTargetDesc.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace exegesis {

void InitializeX86ExegesisTarget();

namespace {

using testing::Gt;
using testing::NotNull;
using testing::SizeIs;

class X86TargetTest : public ::testing::Test {
protected:
  X86TargetTest()
      : Target_(ExegesisTarget::lookup(llvm::Triple("x86_64-unknown-linux"))) {
    EXPECT_THAT(Target_, NotNull());
  }
  static void SetUpTestCase() { InitializeX86ExegesisTarget(); }

  const ExegesisTarget *const Target_;
};

TEST_F(X86TargetTest, SetRegToConstantGPR) {
  const auto Insts = Target_->setRegToConstant(llvm::X86::EAX);
  EXPECT_THAT(Insts, SizeIs(1));
  EXPECT_EQ(Insts[0].getOpcode(), llvm::X86::MOV32ri);
  EXPECT_EQ(Insts[0].getOperand(0).getReg(), llvm::X86::EAX);
}

TEST_F(X86TargetTest, SetRegToConstantXMM) {
  const auto Insts = Target_->setRegToConstant(llvm::X86::XMM1);
  EXPECT_THAT(Insts, SizeIs(Gt(0)));
}

} // namespace
} // namespace exegesis
