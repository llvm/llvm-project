#include "LoongArchSubtarget.h"
#include "LoongArchTargetMachine.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {
std::unique_ptr<LLVMTargetMachine> createTargetMachine() {
  auto TT(Triple::normalize("loongarch64--"));
  std::string CPU("generic-la64");
  std::string FS("+64bit");

  LLVMInitializeLoongArchTargetInfo();
  LLVMInitializeLoongArchTarget();
  LLVMInitializeLoongArchTargetMC();

  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(TT, Error);

  return std::unique_ptr<LLVMTargetMachine>(
      static_cast<LLVMTargetMachine *>(TheTarget->createTargetMachine(
          TT, CPU, FS, TargetOptions(), None, None, CodeGenOpt::Default)));
}

std::unique_ptr<LoongArchInstrInfo> createInstrInfo(TargetMachine *TM) {
  LoongArchSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                        std::string(TM->getTargetCPU()),
                        std::string(TM->getTargetFeatureString()), "lp64d",
                        *TM);
  return std::make_unique<LoongArchInstrInfo>(ST);
}

/// The \p InputIRSnippet is only needed for things that can't be expressed in
/// the \p InputMIRSnippet (global variables etc)
/// Inspired by AArch64
void runChecks(
    LLVMTargetMachine *TM, LoongArchInstrInfo *II,
    const StringRef InputIRSnippet, const StringRef InputMIRSnippet,
    std::function<void(LoongArchInstrInfo &, MachineFunction &)> Checks) {
  LLVMContext Context;

  auto MIRString = "--- |\n"
                   "  declare void @sizes()\n" +
                   InputIRSnippet.str() +
                   "...\n"
                   "---\n"
                   "name: sizes\n"
                   "jumpTable:\n"
                   "  kind:            block-address\n"
                   "  entries:\n"
                   "    - id:              0\n"
                   "      blocks:          [ '%bb.0' ]\n"
                   "body: |\n"
                   "  bb.0:\n" +
                   InputMIRSnippet.str();

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

TEST(InstSizes, INLINEASM_BR) {
  std::unique_ptr<LLVMTargetMachine> TM = createTargetMachine();
  std::unique_ptr<LoongArchInstrInfo> II = createInstrInfo(TM.get());

  runChecks(TM.get(), II.get(), "",
            // clang-format off
            "  INLINEASM_BR &nop, 1 /* sideeffect attdialect */, 13 /* imm */, %jump-table.0\n",
            // clang-format on
            [](LoongArchInstrInfo &II, MachineFunction &MF) {
              auto I = MF.begin()->begin();
              EXPECT_EQ(4u, II.getInstSizeInBytes(*I));
            });
}

TEST(InstSizes, SPACE) {
  std::unique_ptr<LLVMTargetMachine> TM = createTargetMachine();
  std::unique_ptr<LoongArchInstrInfo> II = createInstrInfo(TM.get());

  runChecks(TM.get(), II.get(), "", "  INLINEASM &\".space 1024\", 1\n",
            [](LoongArchInstrInfo &II, MachineFunction &MF) {
              auto I = MF.begin()->begin();
              EXPECT_EQ(1024u, II.getInstSizeInBytes(*I));
            });
}

TEST(InstSizes, AtomicPseudo) {
  std::unique_ptr<LLVMTargetMachine> TM = createTargetMachine();
  std::unique_ptr<LoongArchInstrInfo> II = createInstrInfo(TM.get());

  runChecks(
      TM.get(), II.get(), "",
      // clang-format off
      "    dead early-clobber renamable $r10, dead early-clobber renamable $r11 = PseudoMaskedAtomicLoadAdd32 renamable $r7, renamable $r6, renamable $r8, 4\n"
      "    dead early-clobber renamable $r10, dead early-clobber renamable $r11 = PseudoAtomicLoadAdd32 renamable $r7, renamable $r6, renamable $r8\n"
      "    dead early-clobber renamable $r5, dead early-clobber renamable $r9, dead early-clobber renamable $r10 = PseudoMaskedAtomicLoadUMax32 renamable $r7, renamable $r6, renamable $r8, 4\n"
      "    early-clobber renamable $r9, dead early-clobber renamable $r10, dead early-clobber renamable $r11 = PseudoMaskedAtomicLoadMax32 killed renamable $r6, killed renamable $r5, killed renamable $r7, killed renamable $r8, 4\n"
      "    dead early-clobber renamable $r5, dead early-clobber renamable $r9 = PseudoCmpXchg32 renamable $r7, renamable $r4, renamable $r6\n"
      "    dead early-clobber renamable $r5, dead early-clobber renamable $r9 = PseudoMaskedCmpXchg32 killed renamable $r7, killed renamable $r4, killed renamable $r6, killed renamable $r8, 4\n",
      // clang-format on
      [](LoongArchInstrInfo &II, MachineFunction &MF) {
        auto I = MF.begin()->begin();
        EXPECT_EQ(36u, II.getInstSizeInBytes(*I));
        ++I;
        EXPECT_EQ(24u, II.getInstSizeInBytes(*I));
        ++I;
        EXPECT_EQ(48u, II.getInstSizeInBytes(*I));
        ++I;
        EXPECT_EQ(56u, II.getInstSizeInBytes(*I));
        ++I;
        EXPECT_EQ(36u, II.getInstSizeInBytes(*I));
        ++I;
        EXPECT_EQ(44u, II.getInstSizeInBytes(*I));
      });
}
