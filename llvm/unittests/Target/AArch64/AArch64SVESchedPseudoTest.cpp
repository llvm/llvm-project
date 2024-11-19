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

/// Returns true if the instruction is enabled under a feature that the
/// CPU supports.
static bool isInstructionSupportedByCPU(unsigned Opcode,
                                        FeatureBitset Features) {
  FeatureBitset AvailableFeatures =
      llvm::AArch64_MC::computeAvailableFeatures(Features);
  FeatureBitset RequiredFeatures =
      llvm::AArch64_MC::computeRequiredFeatures(Opcode);
  FeatureBitset MissingFeatures =
      (AvailableFeatures & RequiredFeatures) ^ RequiredFeatures;
  return MissingFeatures.none();
}

void runSVEPseudoTestForCPU(const std::string &CPU) {

  std::unique_ptr<TargetMachine> TM = createTargetMachine(CPU);
  ASSERT_TRUE(TM);
  std::unique_ptr<AArch64InstrInfo> II = createInstrInfo(TM.get());
  ASSERT_TRUE(II);

  const MCSubtargetInfo *STI = TM->getMCSubtargetInfo();
  MCSchedModel SchedModel = STI->getSchedModel();

  for (unsigned i = 0; i < AArch64::INSTRUCTION_LIST_END; ++i) {
    // Check if instruction is in the pseudo table
    // i holds the opcode of the pseudo, OrigInstr holds the opcode of the
    // original instruction
    int OrigInstr = AArch64::getSVEPseudoMap(i);
    if (OrigInstr == -1)
      continue;

    // Ignore any pseudos/instructions which may not be part of the scheduler
    // model for the CPU we're testing. This avoids this test from failing when
    // new instructions are added that are not yet covered by the scheduler
    // model.
    if (!isInstructionSupportedByCPU(OrigInstr, STI->getFeatureBits()))
      continue;

    const MCInstrDesc &Desc = II->get(i);
    unsigned SCClass = Desc.getSchedClass();
    const MCSchedClassDesc *SCDesc = SchedModel.getSchedClassDesc(SCClass);

    const MCInstrDesc &DescOrig = II->get(OrigInstr);
    unsigned SCClassOrig = DescOrig.getSchedClass();
    const MCSchedClassDesc *SCDescOrig =
        SchedModel.getSchedClassDesc(SCClassOrig);

    int Latency = 0;
    int LatencyOrig = 0;

    for (unsigned DefIdx = 0, DefEnd = SCDesc->NumWriteLatencyEntries;
         DefIdx != DefEnd; ++DefIdx) {
      const MCWriteLatencyEntry *WLEntry =
          STI->getWriteLatencyEntry(SCDesc, DefIdx);
      const MCWriteLatencyEntry *WLEntryOrig =
          STI->getWriteLatencyEntry(SCDescOrig, DefIdx);
      Latency = std::max(Latency, static_cast<int>(WLEntry->Cycles));
      LatencyOrig = std::max(Latency, static_cast<int>(WLEntryOrig->Cycles));
    }

    ASSERT_EQ(Latency, LatencyOrig);
    ASSERT_TRUE(SCDesc->isValid());
  }
}

// TODO : Add more CPUs that support SVE/SVE2
TEST(AArch64SVESchedPseudoTesta510, IsCorrect) {
  runSVEPseudoTestForCPU("cortex-a510");
}

TEST(AArch64SVESchedPseudoTestn1, IsCorrect) {
  runSVEPseudoTestForCPU("neoverse-n2");
}

TEST(AArch64SVESchedPseudoTestn3, IsCorrect) {
  runSVEPseudoTestForCPU("neoverse-n3");
}

TEST(AArch64SVESchedPseudoTestv1, IsCorrect) {
  runSVEPseudoTestForCPU("neoverse-v1");
}

TEST(AArch64SVESchedPseudoTestv2, IsCorrect) {
  runSVEPseudoTestForCPU("neoverse-v2");
}

} // namespace
