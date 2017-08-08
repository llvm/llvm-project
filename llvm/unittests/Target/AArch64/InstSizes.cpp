#include "AArch64Subtarget.h"
#include "AArch64TargetMachine.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {
std::unique_ptr<TargetMachine> createTargetMachine() {
  auto TT(Triple::normalize("aarch64--"));
  std::string CPU("generic");
  std::string FS("");

  LLVMInitializeAArch64TargetInfo();
  LLVMInitializeAArch64Target();
  LLVMInitializeAArch64TargetMC();

  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(TT, Error);

  return std::unique_ptr<TargetMachine>(
      TheTarget->createTargetMachine(TT, CPU, FS, TargetOptions(), None,
                                     CodeModel::Default, CodeGenOpt::Default));
}

std::unique_ptr<AArch64InstrInfo> createInstrInfo(TargetMachine *TM) {
  AArch64Subtarget ST(TM->getTargetTriple(), TM->getTargetCPU(),
                      TM->getTargetFeatureString(), *TM, /* isLittle */ false);
  return llvm::make_unique<AArch64InstrInfo>(ST);
}

/// The \p InputIRSnippet is only needed for things that can't be expressed in
/// the \p InputMIRSnippet (global variables etc)
/// TODO: Some of this might be useful for other architectures as well - extract
///       the platform-independent parts somewhere they can be reused.
void runChecks(
    TargetMachine *TM, AArch64InstrInfo *II, const StringRef InputIRSnippet,
    const StringRef InputMIRSnippet,
    std::function<void(AArch64InstrInfo &, MachineFunction &)> Checks) {
  LLVMContext Context;

  auto MIRString =
    "--- |\n"
    "  declare void @sizes()\n"
    + InputIRSnippet.str() +
    "...\n"
    "---\n"
    "name: sizes\n"
    "body: |\n"
    "  bb.0:\n"
    + InputMIRSnippet.str();

  std::unique_ptr<MemoryBuffer> MBuffer = MemoryBuffer::getMemBuffer(MIRString);
  std::unique_ptr<MIRParser> MParser =
      createMIRParser(std::move(MBuffer), Context);
  ASSERT_TRUE(MParser);

  std::unique_ptr<Module> M = MParser->parseIRModule();
  ASSERT_TRUE(M);

  M->setTargetTriple(TM->getTargetTriple().getTriple());
  M->setDataLayout(TM->createDataLayout());

  MachineModuleInfo MMI(TM);
  bool Res = MParser->parseMachineFunctions(*M, MMI);
  ASSERT_FALSE(Res);

  auto F = M->getFunction("sizes");
  ASSERT_TRUE(F != nullptr);
  auto &MF = MMI.getOrCreateMachineFunction(*F);

  Checks(*II, MF);
}

} // anonymous namespace

TEST(InstSizes, STACKMAP) {
  std::unique_ptr<TargetMachine> TM = createTargetMachine();
  ASSERT_TRUE(TM);
  std::unique_ptr<AArch64InstrInfo> II = createInstrInfo(TM.get());

  runChecks(TM.get(), II.get(), "", "    STACKMAP 0, 16\n"
                                    "    STACKMAP 1, 32\n",
            [](AArch64InstrInfo &II, MachineFunction &MF) {
              auto I = MF.begin()->begin();
              EXPECT_EQ(16u, II.getInstSizeInBytes(*I));
              ++I;
              EXPECT_EQ(32u, II.getInstSizeInBytes(*I));
            });
}

TEST(InstSizes, PATCHPOINT) {
  std::unique_ptr<TargetMachine> TM = createTargetMachine();
  std::unique_ptr<AArch64InstrInfo> II = createInstrInfo(TM.get());

  runChecks(TM.get(), II.get(), "",
            "    PATCHPOINT 0, 16, 0, 0, 0, csr_aarch64_aapcs\n"
            "    PATCHPOINT 1, 32, 0, 0, 0, csr_aarch64_aapcs\n",
            [](AArch64InstrInfo &II, MachineFunction &MF) {
              auto I = MF.begin()->begin();
              EXPECT_EQ(16u, II.getInstSizeInBytes(*I));
              ++I;
              EXPECT_EQ(32u, II.getInstSizeInBytes(*I));
            });
}

TEST(InstSizes, TLSDESC_CALLSEQ) {
  std::unique_ptr<TargetMachine> TM = createTargetMachine();
  std::unique_ptr<AArch64InstrInfo> II = createInstrInfo(TM.get());

  runChecks(
      TM.get(), II.get(),
      "  @ThreadLocalGlobal = external thread_local global i32, align 8\n",
      "    TLSDESC_CALLSEQ target-flags(aarch64-tls) @ThreadLocalGlobal\n",
      [](AArch64InstrInfo &II, MachineFunction &MF) {
        auto I = MF.begin()->begin();
        EXPECT_EQ(16u, II.getInstSizeInBytes(*I));
      });
}
