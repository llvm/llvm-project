#include "ARMBaseInstrInfo.h"
#include "ARMSubtarget.h"
#include "ARMTargetMachine.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

#include "gtest/gtest.h"

using namespace llvm;

// Test for instructions that aren't immediately obviously valid within a
// tail-predicated loop. This should be marked up in their tablegen
// descriptions. Currently the horizontal vector operations are tagged.
// TODO Add instructions that perform:
// - truncation,
// - extensions,
// - byte swapping,
// - others?
TEST(MachineInstrInvalidTailPredication, IsCorrect) {
  LLVMInitializeARMTargetInfo();
  LLVMInitializeARMTarget();
  LLVMInitializeARMTargetMC();

  auto TT(Triple::normalize("thumbv8.1m.main-arm-none-eabi"));
  std::string Error;
  const Target *T = TargetRegistry::lookupTarget(TT, Error);
  if (!T) {
    dbgs() << Error;
    return;
  }

  TargetOptions Options;
  auto TM = std::unique_ptr<LLVMTargetMachine>(
    static_cast<LLVMTargetMachine*>(
      T->createTargetMachine(TT, "generic", "", Options, None, None,
                             CodeGenOpt::Default)));
  auto MII = TM->getMCInstrInfo();

  using namespace ARM;

  auto IsInvalidTPOpcode = [](unsigned Opcode) {
    switch (Opcode) {
    case MVE_VABAVs8:
    case MVE_VABAVs16:
    case MVE_VABAVs32:
    case MVE_VABAVu8:
    case MVE_VABAVu16:
    case MVE_VABAVu32:
    case MVE_VADDVs8acc:
    case MVE_VADDVs16acc:
    case MVE_VADDVs32acc:
    case MVE_VADDVu8acc:
    case MVE_VADDVu16acc:
    case MVE_VADDVu32acc:
    case MVE_VADDVs8no_acc:
    case MVE_VADDVs16no_acc:
    case MVE_VADDVs32no_acc:
    case MVE_VADDVu8no_acc:
    case MVE_VADDVu16no_acc:
    case MVE_VADDVu32no_acc:
    case MVE_VADDLVs32no_acc:
    case MVE_VADDLVu32no_acc:
    case MVE_VADDLVs32acc:
    case MVE_VADDLVu32acc:
    case MVE_VMLADAVas16:
    case MVE_VMLADAVas32:
    case MVE_VMLADAVas8:
    case MVE_VMLADAVau16:
    case MVE_VMLADAVau32:
    case MVE_VMLADAVau8:
    case MVE_VMLADAVaxs16:
    case MVE_VMLADAVaxs32:
    case MVE_VMLADAVaxs8:
    case MVE_VMLADAVs16:
    case MVE_VMLADAVs32:
    case MVE_VMLADAVs8:
    case MVE_VMLADAVu16:
    case MVE_VMLADAVu32:
    case MVE_VMLADAVu8:
    case MVE_VMLADAVxs16:
    case MVE_VMLADAVxs32:
    case MVE_VMLADAVxs8:
    case MVE_VMLALDAVas16:
    case MVE_VMLALDAVas32:
    case MVE_VMLALDAVau16:
    case MVE_VMLALDAVau32:
    case MVE_VMLALDAVaxs16:
    case MVE_VMLALDAVaxs32:
    case MVE_VMLALDAVs16:
    case MVE_VMLALDAVs32:
    case MVE_VMLALDAVu16:
    case MVE_VMLALDAVu32:
    case MVE_VMLALDAVxs16:
    case MVE_VMLALDAVxs32:
    case MVE_VMLSDAVas16:
    case MVE_VMLSDAVas32:
    case MVE_VMLSDAVas8:
    case MVE_VMLSDAVaxs16:
    case MVE_VMLSDAVaxs32:
    case MVE_VMLSDAVaxs8:
    case MVE_VMLSDAVs16:
    case MVE_VMLSDAVs32:
    case MVE_VMLSDAVs8:
    case MVE_VMLSDAVxs16:
    case MVE_VMLSDAVxs32:
    case MVE_VMLSDAVxs8:
    case MVE_VMLSLDAVas16:
    case MVE_VMLSLDAVas32:
    case MVE_VMLSLDAVaxs16:
    case MVE_VMLSLDAVaxs32:
    case MVE_VMLSLDAVs16:
    case MVE_VMLSLDAVs32:
    case MVE_VMLSLDAVxs16:
    case MVE_VMLSLDAVxs32:
    case MVE_VRMLALDAVHas32:
    case MVE_VRMLALDAVHau32:
    case MVE_VRMLALDAVHaxs32:
    case MVE_VRMLALDAVHs32:
    case MVE_VRMLALDAVHu32:
    case MVE_VRMLALDAVHxs32:
    case MVE_VRMLSLDAVHas32:
    case MVE_VRMLSLDAVHaxs32:
    case MVE_VRMLSLDAVHs32:
    case MVE_VRMLSLDAVHxs32:
    case MVE_VMAXNMVf16:
    case MVE_VMINNMVf16:
    case MVE_VMAXNMVf32:
    case MVE_VMINNMVf32:
    case MVE_VMAXNMAVf16:
    case MVE_VMINNMAVf16:
    case MVE_VMAXNMAVf32:
    case MVE_VMINNMAVf32:
    case MVE_VMAXVs8:
    case MVE_VMAXVs16:
    case MVE_VMAXVs32:
    case MVE_VMAXVu8:
    case MVE_VMAXVu16:
    case MVE_VMAXVu32:
    case MVE_VMINVs8:
    case MVE_VMINVs16:
    case MVE_VMINVs32:
    case MVE_VMINVu8:
    case MVE_VMINVu16:
    case MVE_VMINVu32:
    case MVE_VMAXAVs8:
    case MVE_VMAXAVs16:
    case MVE_VMAXAVs32:
    case MVE_VMINAVs8:
    case MVE_VMINAVs16:
    case MVE_VMINAVs32:
      return true;
    default:
      return false;
    }
  };

  for (unsigned i = 0; i < ARM::INSTRUCTION_LIST_END; ++i) {
    uint64_t Flags = MII->get(i).TSFlags;
    bool Invalid = (Flags & ARMII::InvalidForTailPredication) != 0;
    ASSERT_EQ(IsInvalidTPOpcode(i), Invalid)
        << MII->getName(i)
        << ": mismatched expectation for tail-predicated safety\n";
  }
}
