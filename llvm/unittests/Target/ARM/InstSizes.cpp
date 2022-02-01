#include "ARMInstrInfo.h"
#include "ARMSubtarget.h"
#include "ARMTargetMachine.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {
/// The \p InputIRSnippet is only needed for things that can't be expressed in
/// the \p InputMIRSnippet (global variables etc)
/// TODO: Some of this might be useful for other architectures as well - extract
///       the platform-independent parts somewhere they can be reused.
void runChecks(
    LLVMTargetMachine *TM, const ARMBaseInstrInfo *II,
    const StringRef InputIRSnippet, const StringRef InputMIRSnippet,
    unsigned Expected,
    std::function<void(const ARMBaseInstrInfo &, MachineFunction &, unsigned &)>
        Checks) {
  LLVMContext Context;

  auto MIRString = "--- |\n"
                   "  declare void @sizes()\n" +
                   InputIRSnippet.str() +
                   "...\n"
                   "---\n"
                   "name: sizes\n"
                   "constants:\n"
                   "  - id:        0\n"
                   "    value:     i32 12345678\n"
                   "    alignment: 4\n"
                   "jumpTable:\n"
                   "  kind:    inline\n"
                   "  entries:\n"
                   "    - id:     0\n"
                   "      blocks: [ '%bb.0' ]\n"
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

  Checks(*II, MF, Expected);
}

} // anonymous namespace

TEST(InstSizes, PseudoInst) {
  LLVMInitializeARMTargetInfo();
  LLVMInitializeARMTarget();
  LLVMInitializeARMTargetMC();

  auto TT(Triple::normalize("thumbv8.1m.main-none-none-eabi"));
  std::string Error;
  const Target *T = TargetRegistry::lookupTarget(TT, Error);
  if (!T) {
    dbgs() << Error;
    return;
  }

  TargetOptions Options;
  auto TM = std::unique_ptr<LLVMTargetMachine>(
      static_cast<LLVMTargetMachine *>(T->createTargetMachine(
          TT, "generic", "", Options, None, None, CodeGenOpt::Default)));
  ARMSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                  std::string(TM->getTargetFeatureString()),
                  *static_cast<const ARMBaseTargetMachine *>(TM.get()), false);
  const ARMBaseInstrInfo *II = ST.getInstrInfo();

  auto cmpInstSize = [](const ARMBaseInstrInfo &II, MachineFunction &MF,
                        unsigned &Expected) {
    auto I = MF.begin()->begin();
    EXPECT_EQ(Expected, II.getInstSizeInBytes(*I));
  };

  runChecks(TM.get(), II, "",
            "    $r0 = MOVi16_ga_pcrel"
            " target-flags(arm-lo16, arm-nonlazy) @sizes, 0\n",
            4u, cmpInstSize);

  runChecks(TM.get(), II, "",
            "    $r0 = MOVTi16_ga_pcrel $r0,"
            " target-flags(arm-hi16, arm-nonlazy) @sizes, 0\n",
            4u, cmpInstSize);

  runChecks(TM.get(), II, "",
            "    $r0 = t2MOVi16_ga_pcrel"
            " target-flags(arm-lo16, arm-nonlazy) @sizes, 0\n",
            4u, cmpInstSize);

  runChecks(TM.get(), II, "",
            "    $r0 = t2MOVTi16_ga_pcrel $r0,"
            " target-flags(arm-hi16, arm-nonlazy) @sizes, 0\n",
            4u, cmpInstSize);

  runChecks(TM.get(), II, "", "    $r0 = MOVi32imm 2\n", 8u, cmpInstSize);

  runChecks(TM.get(), II, "", "    $r0 = t2MOVi32imm 2\n", 8u, cmpInstSize);

  runChecks(TM.get(), II, "",
            "    SpeculationBarrierISBDSBEndBB\n"
            "    tBX_RET 14, $noreg, implicit $r0\n",
            8u, cmpInstSize);

  runChecks(TM.get(), II, "",
            "    t2SpeculationBarrierISBDSBEndBB\n"
            "    tBX_RET 14, $noreg, implicit $r0\n",
            8u, cmpInstSize);

  runChecks(TM.get(), II, "",
            "    SpeculationBarrierSBEndBB\n"
            "    tBX_RET 14, $noreg, implicit $r0\n",
            4u, cmpInstSize);

  runChecks(TM.get(), II, "",
            "    t2SpeculationBarrierSBEndBB\n"
            "    tBX_RET 14, $noreg, implicit $r0\n",
            4u, cmpInstSize);

  runChecks(TM.get(), II, "",
            "    Int_eh_sjlj_longjmp $r0, $r1, implicit-def $r7,"
            " implicit-def $lr, implicit-def $sp\n",
            16u, cmpInstSize);

  runChecks(TM.get(), II, "",
            "    tInt_eh_sjlj_longjmp $r0, $r1, implicit-def $r7,"
            " implicit-def $lr, implicit-def $sp\n",
            10u, cmpInstSize);

  runChecks(TM.get(), II, "",
            "    tInt_WIN_eh_sjlj_longjmp $r0, $r1, implicit-def $r11,"
            " implicit-def $lr, implicit-def $sp\n",
            12u, cmpInstSize);

  runChecks(TM.get(), II, "",
            "    Int_eh_sjlj_setjmp $r0, $r1, implicit-def $r0,"
            " implicit-def $r1, implicit-def $r2, implicit-def $r3,"
            " implicit-def $r4, implicit-def $r5, implicit-def $r6,"
            " implicit-def $r7, implicit-def $r8, implicit-def $r9,"
            " implicit-def $r10, implicit-def $r11, implicit-def $r12,"
            " implicit-def $lr, implicit-def $cpsr, implicit-def $q0,"
            " implicit-def $q1, implicit-def $q2, implicit-def $q3,"
            " implicit-def $q4, implicit-def $q5, implicit-def $q6,"
            " implicit-def $q7, implicit-def $q8, implicit-def $q9,"
            " implicit-def $q10, implicit-def $q11, implicit-def $q12,"
            " implicit-def $q13, implicit-def $q14, implicit-def $q15\n"
            "    tBX_RET 14, $noreg, implicit $r0\n",
            20u, cmpInstSize);

  runChecks(TM.get(), II, "",
            "    Int_eh_sjlj_setjmp_nofp $r0, $r1, implicit-def $r0,"
            " implicit-def $r1, implicit-def $r2, implicit-def $r3,"
            " implicit-def $r4, implicit-def $r5, implicit-def $r6,"
            " implicit-def $r7, implicit-def $r8, implicit-def $r9,"
            " implicit-def $r10, implicit-def $r11, implicit-def $r12,"
            " implicit-def $lr, implicit-def $cpsr\n"
            "    tBX_RET 14, $noreg, implicit $r0\n",
            20u, cmpInstSize);

  runChecks(TM.get(), II, "",
            "    tInt_eh_sjlj_setjmp $r0, $r1, implicit-def $r0,"
            " implicit-def $r1, implicit-def $r2, implicit-def $r3,"
            " implicit-def $r4, implicit-def $r5, implicit-def $r6,"
            " implicit-def $r7, implicit-def $r12, implicit-def $cpsr\n"
            "    tBX_RET 14, $noreg, implicit $r0\n",
            12u, cmpInstSize);

  runChecks(TM.get(), II, "",
            "    t2Int_eh_sjlj_setjmp $r0, $r1, implicit-def $r0,"
            " implicit-def $r1, implicit-def $r2, implicit-def $r3,"
            " implicit-def $r4, implicit-def $r5, implicit-def $r6,"
            " implicit-def $r7, implicit-def $r8, implicit-def $r9,"
            " implicit-def $r10, implicit-def $r11, implicit-def $r12,"
            " implicit-def $lr, implicit-def $cpsr, implicit-def $q0,"
            " implicit-def $q1, implicit-def $q2, implicit-def $q3,"
            " implicit-def $q8, implicit-def $q9, implicit-def $q10,"
            " implicit-def $q11, implicit-def $q12, implicit-def $q13,"
            " implicit-def $q14, implicit-def $q15\n"
            "    tBX_RET 14, $noreg, implicit $r0\n",
            12u, cmpInstSize);

  runChecks(TM.get(), II, "",
            "    t2Int_eh_sjlj_setjmp_nofp $r0, $r1, implicit-def $r0,"
            " implicit-def $r1, implicit-def $r2, implicit-def $r3,"
            " implicit-def $r4, implicit-def $r5, implicit-def $r6,"
            " implicit-def $r7, implicit-def $r8, implicit-def $r9,"
            " implicit-def $r10, implicit-def $r11, implicit-def $r12,"
            " implicit-def $lr, implicit-def $cpsr\n"
            "    tBX_RET 14, $noreg, implicit $r0\n",
            12u, cmpInstSize);

  runChecks(TM.get(), II, "", "  CONSTPOOL_ENTRY 3, %const.0, 8\n", 8u,
            cmpInstSize);

  runChecks(TM.get(), II, "", "  JUMPTABLE_ADDRS 0, %jump-table.0, 123\n", 123u,
            cmpInstSize);

  runChecks(TM.get(), II, "", "  JUMPTABLE_INSTS 0, %jump-table.0, 456\n", 456u,
            cmpInstSize);

  runChecks(TM.get(), II, "", "  JUMPTABLE_TBB 0, %jump-table.0, 789\n", 789u,
            cmpInstSize);

  runChecks(TM.get(), II, "", "  JUMPTABLE_TBH 0, %jump-table.0, 188\n", 188u,
            cmpInstSize);

  runChecks(TM.get(), II, "", "  $r0 = SPACE 40, undef $r0\n", 40u,
            cmpInstSize);

  runChecks(TM.get(), II, "", "  INLINEASM &\"movs  r0, #42\", 1\n", 6u,
            cmpInstSize);

  runChecks(TM.get(), II,
            "  define void @foo() {\n"
            "  entry:\n"
            "    ret void\n"
            "  }\n",
            "  INLINEASM_BR &\"b ${0:l}\", 1, 13, blockaddress(@foo, "
            "%ir-block.entry)\n",
            6u, cmpInstSize);
}
