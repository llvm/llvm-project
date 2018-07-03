#include "Target.h"

#include <cassert>
#include <memory>

#include "MCTargetDesc/X86MCTargetDesc.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace exegesis {

void InitializeX86ExegesisTarget();

namespace {

using testing::Gt;
using testing::NotNull;
using testing::SizeIs;

constexpr const char kTriple[] = "x86_64-unknown-linux";

class X86TargetTest : public ::testing::Test {
protected:
  X86TargetTest()
      : ExegesisTarget_(ExegesisTarget::lookup(llvm::Triple(kTriple))) {
    EXPECT_THAT(ExegesisTarget_, NotNull());
    std::string error;
    Target_ = llvm::TargetRegistry::lookupTarget(kTriple, error);
    EXPECT_THAT(Target_, NotNull());
  }
  static void SetUpTestCase() {
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86Target();
    LLVMInitializeX86TargetMC();
    InitializeX86ExegesisTarget();
  }

  const llvm::Target *Target_;
  const ExegesisTarget *const ExegesisTarget_;
};

TEST_F(X86TargetTest, SetRegToConstantGPR) {
  const std::unique_ptr<llvm::MCSubtargetInfo> STI(
      Target_->createMCSubtargetInfo(kTriple, "core2", ""));
  const auto Insts = ExegesisTarget_->setRegToConstant(*STI, llvm::X86::EAX);
  EXPECT_THAT(Insts, SizeIs(1));
  EXPECT_EQ(Insts[0].getOpcode(), llvm::X86::MOV32ri);
  EXPECT_EQ(Insts[0].getOperand(0).getReg(), llvm::X86::EAX);
}

TEST_F(X86TargetTest, SetRegToConstantXMM_SSE2) {
  const std::unique_ptr<llvm::MCSubtargetInfo> STI(
      Target_->createMCSubtargetInfo(kTriple, "core2", ""));
  const auto Insts = ExegesisTarget_->setRegToConstant(*STI, llvm::X86::XMM1);
  EXPECT_THAT(Insts, SizeIs(7U));
  EXPECT_EQ(Insts[0].getOpcode(), llvm::X86::SUB64ri8);
  EXPECT_EQ(Insts[1].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[2].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[3].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[4].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[5].getOpcode(), llvm::X86::MOVDQUrm);
  EXPECT_EQ(Insts[6].getOpcode(), llvm::X86::ADD64ri8);
}

TEST_F(X86TargetTest, SetRegToConstantXMM_AVX) {
  const std::unique_ptr<llvm::MCSubtargetInfo> STI(
      Target_->createMCSubtargetInfo(kTriple, "core2", "+avx"));
  const auto Insts = ExegesisTarget_->setRegToConstant(*STI, llvm::X86::XMM1);
  EXPECT_THAT(Insts, SizeIs(7U));
  EXPECT_EQ(Insts[0].getOpcode(), llvm::X86::SUB64ri8);
  EXPECT_EQ(Insts[1].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[2].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[3].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[4].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[5].getOpcode(), llvm::X86::VMOVDQUrm);
  EXPECT_EQ(Insts[6].getOpcode(), llvm::X86::ADD64ri8);
}

TEST_F(X86TargetTest, SetRegToConstantXMM_AVX512) {
  const std::unique_ptr<llvm::MCSubtargetInfo> STI(
      Target_->createMCSubtargetInfo(kTriple, "core2", "+avx512vl"));
  const auto Insts = ExegesisTarget_->setRegToConstant(*STI, llvm::X86::XMM1);
  EXPECT_THAT(Insts, SizeIs(7U));
  EXPECT_EQ(Insts[0].getOpcode(), llvm::X86::SUB64ri8);
  EXPECT_EQ(Insts[1].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[2].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[3].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[4].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[5].getOpcode(), llvm::X86::VMOVDQU32Z128rm);
  EXPECT_EQ(Insts[6].getOpcode(), llvm::X86::ADD64ri8);
}

TEST_F(X86TargetTest, SetRegToConstantMMX) {
  const std::unique_ptr<llvm::MCSubtargetInfo> STI(
      Target_->createMCSubtargetInfo(kTriple, "core2", ""));
  const auto Insts = ExegesisTarget_->setRegToConstant(*STI, llvm::X86::MM1);
  EXPECT_THAT(Insts, SizeIs(5U));
  EXPECT_EQ(Insts[0].getOpcode(), llvm::X86::SUB64ri8);
  EXPECT_EQ(Insts[1].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[2].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[3].getOpcode(), llvm::X86::MMX_MOVQ64rm);
  EXPECT_EQ(Insts[4].getOpcode(), llvm::X86::ADD64ri8);
}

TEST_F(X86TargetTest, SetRegToConstantYMM_AVX) {
  const std::unique_ptr<llvm::MCSubtargetInfo> STI(
      Target_->createMCSubtargetInfo(kTriple, "core2", "+avx"));
  const auto Insts = ExegesisTarget_->setRegToConstant(*STI, llvm::X86::YMM1);
  EXPECT_THAT(Insts, SizeIs(11U));
  EXPECT_EQ(Insts[0].getOpcode(), llvm::X86::SUB64ri8);
  EXPECT_EQ(Insts[1].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[2].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[3].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[4].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[5].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[6].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[7].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[8].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[9].getOpcode(), llvm::X86::VMOVDQUYrm);
  EXPECT_EQ(Insts[10].getOpcode(), llvm::X86::ADD64ri8);
}

TEST_F(X86TargetTest, SetRegToConstantYMM_AVX512) {
  const std::unique_ptr<llvm::MCSubtargetInfo> STI(
      Target_->createMCSubtargetInfo(kTriple, "core2", "+avx512vl"));
  const auto Insts = ExegesisTarget_->setRegToConstant(*STI, llvm::X86::YMM1);
  EXPECT_THAT(Insts, SizeIs(11U));
  EXPECT_EQ(Insts[0].getOpcode(), llvm::X86::SUB64ri8);
  EXPECT_EQ(Insts[1].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[2].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[3].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[4].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[5].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[6].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[7].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[8].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[9].getOpcode(), llvm::X86::VMOVDQU32Z256rm);
  EXPECT_EQ(Insts[10].getOpcode(), llvm::X86::ADD64ri8);
}

TEST_F(X86TargetTest, SetRegToConstantZMM_AVX512) {
  const std::unique_ptr<llvm::MCSubtargetInfo> STI(
      Target_->createMCSubtargetInfo(kTriple, "core2", "+avx512vl"));
  const auto Insts = ExegesisTarget_->setRegToConstant(*STI, llvm::X86::ZMM1);
  EXPECT_THAT(Insts, SizeIs(19U));
  EXPECT_EQ(Insts[0].getOpcode(), llvm::X86::SUB64ri8);
  EXPECT_EQ(Insts[1].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[2].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[3].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[4].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[5].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[6].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[7].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[8].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[9].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[10].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[11].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[12].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[13].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[14].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[15].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[16].getOpcode(), llvm::X86::MOV32mi);
  EXPECT_EQ(Insts[17].getOpcode(), llvm::X86::VMOVDQU32Zrm);
  EXPECT_EQ(Insts[18].getOpcode(), llvm::X86::ADD64ri8);
}

} // namespace
} // namespace exegesis
