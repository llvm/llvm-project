#include "ARMBaseInstrInfo.h"
#include "ARMSubtarget.h"
#include "ARMTargetMachine.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

#include "gtest/gtest.h"

using namespace llvm;

TEST(MachineInstructionDoubleWidthResult, IsCorrect) {
  using namespace ARM;

  auto DoubleWidthResult = [](unsigned Opcode) {
    switch (Opcode) {
    default:
      break;
    case MVE_VMULLBp16:
    case MVE_VMULLBp8:
    case MVE_VMULLBs16:
    case MVE_VMULLBs32:
    case MVE_VMULLBs8:
    case MVE_VMULLBu16:
    case MVE_VMULLBu32:
    case MVE_VMULLBu8:
    case MVE_VMULLTp16:
    case MVE_VMULLTp8:
    case MVE_VMULLTs16:
    case MVE_VMULLTs32:
    case MVE_VMULLTs8:
    case MVE_VMULLTu16:
    case MVE_VMULLTu32:
    case MVE_VMULLTu8:
    case MVE_VQDMULL_qr_s16bh:
    case MVE_VQDMULL_qr_s16th:
    case MVE_VQDMULL_qr_s32bh:
    case MVE_VQDMULL_qr_s32th:
    case MVE_VQDMULLs16bh:
    case MVE_VQDMULLs16th:
    case MVE_VQDMULLs32bh:
    case MVE_VQDMULLs32th:
    case MVE_VMOVLs16bh:
    case MVE_VMOVLs16th:
    case MVE_VMOVLs8bh:
    case MVE_VMOVLs8th:
    case MVE_VMOVLu16bh:
    case MVE_VMOVLu16th:
    case MVE_VMOVLu8bh:
    case MVE_VMOVLu8th:
    case MVE_VSHLL_imms16bh:
    case MVE_VSHLL_imms16th:
    case MVE_VSHLL_imms8bh:
    case MVE_VSHLL_imms8th:
    case MVE_VSHLL_immu16bh:
    case MVE_VSHLL_immu16th:
    case MVE_VSHLL_immu8bh:
    case MVE_VSHLL_immu8th:
    case MVE_VSHLL_lws16bh:
    case MVE_VSHLL_lws16th:
    case MVE_VSHLL_lws8bh:
    case MVE_VSHLL_lws8th:
    case MVE_VSHLL_lwu16bh:
    case MVE_VSHLL_lwu16th:
    case MVE_VSHLL_lwu8bh:
    case MVE_VSHLL_lwu8th:
      return true;
    }
    return false;
  };

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
    static_cast<LLVMTargetMachine*>(
      T->createTargetMachine(TT, "generic", "", Options, None, None,
                             CodeGenOpt::Default)));
  ARMSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                  std::string(TM->getTargetFeatureString()),
                  *static_cast<const ARMBaseTargetMachine *>(TM.get()), false);
  const ARMBaseInstrInfo *TII = ST.getInstrInfo();
  auto MII = TM->getMCInstrInfo();

  for (unsigned i = 0; i < ARM::INSTRUCTION_LIST_END; ++i) {
    const MCInstrDesc &Desc = TII->get(i);

    uint64_t Flags = Desc.TSFlags;
    if ((Flags & ARMII::DomainMask) != ARMII::DomainMVE)
      continue;

    bool Valid = (Flags & ARMII::DoubleWidthResult) != 0;
    ASSERT_EQ(DoubleWidthResult(i), Valid)
              << MII->getName(i)
              << ": mismatched expectation for tail-predicated safety\n";
  }
}

TEST(MachineInstructionHorizontalReduction, IsCorrect) {
  using namespace ARM;

  auto HorizontalReduction = [](unsigned Opcode) {
    switch (Opcode) {
    default:
      break;
    case MVE_VABAVs16:
    case MVE_VABAVs32:
    case MVE_VABAVs8:
    case MVE_VABAVu16:
    case MVE_VABAVu32:
    case MVE_VABAVu8:
    case MVE_VADDLVs32acc:
    case MVE_VADDLVs32no_acc:
    case MVE_VADDLVu32acc:
    case MVE_VADDLVu32no_acc:
    case MVE_VADDVs16acc:
    case MVE_VADDVs16no_acc:
    case MVE_VADDVs32acc:
    case MVE_VADDVs32no_acc:
    case MVE_VADDVs8acc:
    case MVE_VADDVs8no_acc:
    case MVE_VADDVu16acc:
    case MVE_VADDVu16no_acc:
    case MVE_VADDVu32acc:
    case MVE_VADDVu32no_acc:
    case MVE_VADDVu8acc:
    case MVE_VADDVu8no_acc:
    case MVE_VMAXAVs16:
    case MVE_VMAXAVs32:
    case MVE_VMAXAVs8:
    case MVE_VMAXNMAVf16:
    case MVE_VMAXNMAVf32:
    case MVE_VMAXNMVf16:
    case MVE_VMAXNMVf32:
    case MVE_VMAXVs16:
    case MVE_VMAXVs32:
    case MVE_VMAXVs8:
    case MVE_VMAXVu16:
    case MVE_VMAXVu32:
    case MVE_VMAXVu8:
    case MVE_VMINAVs16:
    case MVE_VMINAVs32:
    case MVE_VMINAVs8:
    case MVE_VMINNMAVf16:
    case MVE_VMINNMAVf32:
    case MVE_VMINNMVf16:
    case MVE_VMINNMVf32:
    case MVE_VMINVs16:
    case MVE_VMINVs32:
    case MVE_VMINVs8:
    case MVE_VMINVu16:
    case MVE_VMINVu32:
    case MVE_VMINVu8:
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
      return true;
    }
    return false;
  };

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
    static_cast<LLVMTargetMachine*>(
      T->createTargetMachine(TT, "generic", "", Options, None, None,
                             CodeGenOpt::Default)));
  ARMSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                  std::string(TM->getTargetFeatureString()),
                  *static_cast<const ARMBaseTargetMachine *>(TM.get()), false);
  const ARMBaseInstrInfo *TII = ST.getInstrInfo();
  auto MII = TM->getMCInstrInfo();

  for (unsigned i = 0; i < ARM::INSTRUCTION_LIST_END; ++i) {
    const MCInstrDesc &Desc = TII->get(i);

    uint64_t Flags = Desc.TSFlags;
    if ((Flags & ARMII::DomainMask) != ARMII::DomainMVE)
      continue;
    bool Valid = (Flags & ARMII::HorizontalReduction) != 0;
    ASSERT_EQ(HorizontalReduction(i), Valid)
              << MII->getName(i)
              << ": mismatched expectation for tail-predicated safety\n";
  }
}

TEST(MachineInstructionRetainsPreviousHalfElement, IsCorrect) {
  using namespace ARM;

  auto RetainsPreviousHalfElement = [](unsigned Opcode) {
    switch (Opcode) {
    default:
      break;
    case MVE_VMOVNi16bh:
    case MVE_VMOVNi16th:
    case MVE_VMOVNi32bh:
    case MVE_VMOVNi32th:
    case MVE_VQMOVNs16bh:
    case MVE_VQMOVNs16th:
    case MVE_VQMOVNs32bh:
    case MVE_VQMOVNs32th:
    case MVE_VQMOVNu16bh:
    case MVE_VQMOVNu16th:
    case MVE_VQMOVNu32bh:
    case MVE_VQMOVNu32th:
    case MVE_VQMOVUNs16bh:
    case MVE_VQMOVUNs16th:
    case MVE_VQMOVUNs32bh:
    case MVE_VQMOVUNs32th:
    case MVE_VQRSHRNbhs16:
    case MVE_VQRSHRNbhs32:
    case MVE_VQRSHRNbhu16:
    case MVE_VQRSHRNbhu32:
    case MVE_VQRSHRNths16:
    case MVE_VQRSHRNths32:
    case MVE_VQRSHRNthu16:
    case MVE_VQRSHRNthu32:
    case MVE_VQRSHRUNs16bh:
    case MVE_VQRSHRUNs16th:
    case MVE_VQRSHRUNs32bh:
    case MVE_VQRSHRUNs32th:
    case MVE_VQSHRNbhs16:
    case MVE_VQSHRNbhs32:
    case MVE_VQSHRNbhu16:
    case MVE_VQSHRNbhu32:
    case MVE_VQSHRNths16:
    case MVE_VQSHRNths32:
    case MVE_VQSHRNthu16:
    case MVE_VQSHRNthu32:
    case MVE_VQSHRUNs16bh:
    case MVE_VQSHRUNs16th:
    case MVE_VQSHRUNs32bh:
    case MVE_VQSHRUNs32th:
    case MVE_VRSHRNi16bh:
    case MVE_VRSHRNi16th:
    case MVE_VRSHRNi32bh:
    case MVE_VRSHRNi32th:
    case MVE_VSHRNi16bh:
    case MVE_VSHRNi16th:
    case MVE_VSHRNi32bh:
    case MVE_VSHRNi32th:
    case MVE_VCVTf16f32bh:
    case MVE_VCVTf16f32th:
    case MVE_VCVTf32f16bh:
    case MVE_VCVTf32f16th:
      return true;
    }
    return false;
  };

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
    static_cast<LLVMTargetMachine*>(
      T->createTargetMachine(TT, "generic", "", Options, None, None,
                             CodeGenOpt::Default)));
  ARMSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                  std::string(TM->getTargetFeatureString()),
                  *static_cast<const ARMBaseTargetMachine *>(TM.get()), false);
  const ARMBaseInstrInfo *TII = ST.getInstrInfo();
  auto MII = TM->getMCInstrInfo();

  for (unsigned i = 0; i < ARM::INSTRUCTION_LIST_END; ++i) {
    const MCInstrDesc &Desc = TII->get(i);

    uint64_t Flags = Desc.TSFlags;
    if ((Flags & ARMII::DomainMask) != ARMII::DomainMVE)
      continue;

    bool Valid = (Flags & ARMII::RetainsPreviousHalfElement) != 0;
    ASSERT_EQ(RetainsPreviousHalfElement(i), Valid)
              << MII->getName(i)
              << ": mismatched expectation for tail-predicated safety\n";
  }
}
// Test for instructions that aren't immediately obviously valid within a
// tail-predicated loop. This should be marked up in their tablegen
// descriptions. Currently we, conservatively, disallow:
// - cross beat carries.
// - complex operations.
// - horizontal operations with exchange.
// - byte swapping.
// - interleaved memory instructions.
// TODO: Add to this list once we can handle them safely.
TEST(MachineInstrValidTailPredication, IsCorrect) {

  using namespace ARM;

  auto IsValidTPOpcode = [](unsigned Opcode) {
    switch (Opcode) {
    default:
      return false;
    case MVE_ASRLi:
    case MVE_ASRLr:
    case MVE_LSRL:
    case MVE_LSLLi:
    case MVE_LSLLr:
    case MVE_SQRSHR:
    case MVE_SQRSHRL:
    case MVE_SQSHL:
    case MVE_SQSHLL:
    case MVE_SRSHR:
    case MVE_SRSHRL:
    case MVE_UQRSHL:
    case MVE_UQRSHLL:
    case MVE_UQSHL:
    case MVE_UQSHLL:
    case MVE_URSHR:
    case MVE_URSHRL:
    case MVE_VABDf16:
    case MVE_VABDf32:
    case MVE_VABDs16:
    case MVE_VABDs32:
    case MVE_VABDs8:
    case MVE_VABDu16:
    case MVE_VABDu32:
    case MVE_VABDu8:
    case MVE_VABSf16:
    case MVE_VABSf32:
    case MVE_VABSs16:
    case MVE_VABSs32:
    case MVE_VABSs8:
    case MVE_VADD_qr_f16:
    case MVE_VADD_qr_f32:
    case MVE_VADD_qr_i16:
    case MVE_VADD_qr_i32:
    case MVE_VADD_qr_i8:
    case MVE_VADDVs16acc:
    case MVE_VADDVs16no_acc:
    case MVE_VADDVs32acc:
    case MVE_VADDVs32no_acc:
    case MVE_VADDVs8acc:
    case MVE_VADDVs8no_acc:
    case MVE_VADDVu16acc:
    case MVE_VADDVu16no_acc:
    case MVE_VADDVu32acc:
    case MVE_VADDVu32no_acc:
    case MVE_VADDVu8acc:
    case MVE_VADDVu8no_acc:
    case MVE_VADDf16:
    case MVE_VADDf32:
    case MVE_VADDi16:
    case MVE_VADDi32:
    case MVE_VADDi8:
    case MVE_VAND:
    case MVE_VBIC:
    case MVE_VBICimmi16:
    case MVE_VBICimmi32:
    case MVE_VBRSR16:
    case MVE_VBRSR32:
    case MVE_VBRSR8:
    case MVE_VCLSs16:
    case MVE_VCLSs32:
    case MVE_VCLSs8:
    case MVE_VCLZs16:
    case MVE_VCLZs32:
    case MVE_VCLZs8:
    case MVE_VCMPf16:
    case MVE_VCMPf16r:
    case MVE_VCMPf32:
    case MVE_VCMPf32r:
    case MVE_VCMPi16:
    case MVE_VCMPi16r:
    case MVE_VCMPi32:
    case MVE_VCMPi32r:
    case MVE_VCMPi8:
    case MVE_VCMPi8r:
    case MVE_VCMPs16:
    case MVE_VCMPs16r:
    case MVE_VCMPs32:
    case MVE_VCMPs32r:
    case MVE_VCMPs8:
    case MVE_VCMPs8r:
    case MVE_VCMPu16:
    case MVE_VCMPu16r:
    case MVE_VCMPu32:
    case MVE_VCMPu32r:
    case MVE_VCMPu8:
    case MVE_VCMPu8r:
    case MVE_VCTP16:
    case MVE_VCTP32:
    case MVE_VCTP64:
    case MVE_VCTP8:
    case MVE_VCVTf16s16_fix:
    case MVE_VCVTf16s16n:
    case MVE_VCVTf16u16_fix:
    case MVE_VCVTf16u16n:
    case MVE_VCVTf32s32_fix:
    case MVE_VCVTf32s32n:
    case MVE_VCVTf32u32_fix:
    case MVE_VCVTf32u32n:
    case MVE_VCVTs16f16_fix:
    case MVE_VCVTs16f16a:
    case MVE_VCVTs16f16m:
    case MVE_VCVTs16f16n:
    case MVE_VCVTs16f16p:
    case MVE_VCVTs16f16z:
    case MVE_VCVTs32f32_fix:
    case MVE_VCVTs32f32a:
    case MVE_VCVTs32f32m:
    case MVE_VCVTs32f32n:
    case MVE_VCVTs32f32p:
    case MVE_VCVTs32f32z:
    case MVE_VCVTu16f16_fix:
    case MVE_VCVTu16f16a:
    case MVE_VCVTu16f16m:
    case MVE_VCVTu16f16n:
    case MVE_VCVTu16f16p:
    case MVE_VCVTu16f16z:
    case MVE_VCVTu32f32_fix:
    case MVE_VCVTu32f32a:
    case MVE_VCVTu32f32m:
    case MVE_VCVTu32f32n:
    case MVE_VCVTu32f32p:
    case MVE_VCVTu32f32z:
    case MVE_VDDUPu16:
    case MVE_VDDUPu32:
    case MVE_VDDUPu8:
    case MVE_VDUP16:
    case MVE_VDUP32:
    case MVE_VDUP8:
    case MVE_VDWDUPu16:
    case MVE_VDWDUPu32:
    case MVE_VDWDUPu8:
    case MVE_VEOR:
    case MVE_VFMA_qr_Sf16:
    case MVE_VFMA_qr_Sf32:
    case MVE_VFMA_qr_f16:
    case MVE_VFMA_qr_f32:
    case MVE_VFMAf16:
    case MVE_VFMAf32:
    case MVE_VFMSf16:
    case MVE_VFMSf32:
    case MVE_VMAXAs16:
    case MVE_VMAXAs32:
    case MVE_VMAXAs8:
    case MVE_VMAXs16:
    case MVE_VMAXs32:
    case MVE_VMAXs8:
    case MVE_VMAXu16:
    case MVE_VMAXu32:
    case MVE_VMAXu8:
    case MVE_VMAXNMf16:
    case MVE_VMAXNMf32:
    case MVE_VMAXNMAf16:
    case MVE_VMAXNMAf32:
    case MVE_VMINAs16:
    case MVE_VMINAs32:
    case MVE_VMINAs8:
    case MVE_VMINs16:
    case MVE_VMINs32:
    case MVE_VMINs8:
    case MVE_VMINu16:
    case MVE_VMINu32:
    case MVE_VMINu8:
    case MVE_VMINNMf16:
    case MVE_VMINNMf32:
    case MVE_VMINNMAf16:
    case MVE_VMINNMAf32:
    case MVE_VMLADAVas16:
    case MVE_VMLADAVas32:
    case MVE_VMLADAVas8:
    case MVE_VMLADAVau16:
    case MVE_VMLADAVau32:
    case MVE_VMLADAVau8:
    case MVE_VMLADAVs16:
    case MVE_VMLADAVs32:
    case MVE_VMLADAVs8:
    case MVE_VMLADAVu16:
    case MVE_VMLADAVu32:
    case MVE_VMLADAVu8:
    case MVE_VMLALDAVs16:
    case MVE_VMLALDAVs32:
    case MVE_VMLALDAVu16:
    case MVE_VMLALDAVu32:
    case MVE_VMLALDAVas16:
    case MVE_VMLALDAVas32:
    case MVE_VMLALDAVau16:
    case MVE_VMLALDAVau32:
    case MVE_VMLSDAVas16:
    case MVE_VMLSDAVas32:
    case MVE_VMLSDAVas8:
    case MVE_VMLSDAVs16:
    case MVE_VMLSDAVs32:
    case MVE_VMLSDAVs8:
    case MVE_VMLSLDAVas16:
    case MVE_VMLSLDAVas32:
    case MVE_VMLSLDAVs16:
    case MVE_VMLSLDAVs32:
    case MVE_VRMLALDAVHas32:
    case MVE_VRMLALDAVHau32:
    case MVE_VRMLALDAVHs32:
    case MVE_VRMLALDAVHu32:
    case MVE_VRMLSLDAVHas32:
    case MVE_VRMLSLDAVHs32:
    case MVE_VMLAS_qr_i16:
    case MVE_VMLAS_qr_i32:
    case MVE_VMLAS_qr_i8:
    case MVE_VMLA_qr_i16:
    case MVE_VMLA_qr_i32:
    case MVE_VMLA_qr_i8:
    case MVE_VHADD_qr_s16:
    case MVE_VHADD_qr_s32:
    case MVE_VHADD_qr_s8:
    case MVE_VHADD_qr_u16:
    case MVE_VHADD_qr_u32:
    case MVE_VHADD_qr_u8:
    case MVE_VHADDs16:
    case MVE_VHADDs32:
    case MVE_VHADDs8:
    case MVE_VHADDu16:
    case MVE_VHADDu32:
    case MVE_VHADDu8:
    case MVE_VHSUB_qr_s16:
    case MVE_VHSUB_qr_s32:
    case MVE_VHSUB_qr_s8:
    case MVE_VHSUB_qr_u16:
    case MVE_VHSUB_qr_u32:
    case MVE_VHSUB_qr_u8:
    case MVE_VHSUBs16:
    case MVE_VHSUBs32:
    case MVE_VHSUBs8:
    case MVE_VHSUBu16:
    case MVE_VHSUBu32:
    case MVE_VHSUBu8:
    case MVE_VIDUPu16:
    case MVE_VIDUPu32:
    case MVE_VIDUPu8:
    case MVE_VIWDUPu16:
    case MVE_VIWDUPu32:
    case MVE_VIWDUPu8:
    case MVE_VLD20_8:
    case MVE_VLD21_8:
    case MVE_VLD20_16:
    case MVE_VLD21_16:
    case MVE_VLD20_32:
    case MVE_VLD21_32:
    case MVE_VLD20_8_wb:
    case MVE_VLD21_8_wb:
    case MVE_VLD20_16_wb:
    case MVE_VLD21_16_wb:
    case MVE_VLD20_32_wb:
    case MVE_VLD21_32_wb:
    case MVE_VLD40_8:
    case MVE_VLD41_8:
    case MVE_VLD42_8:
    case MVE_VLD43_8:
    case MVE_VLD40_16:
    case MVE_VLD41_16:
    case MVE_VLD42_16:
    case MVE_VLD43_16:
    case MVE_VLD40_32:
    case MVE_VLD41_32:
    case MVE_VLD42_32:
    case MVE_VLD43_32:
    case MVE_VLD40_8_wb:
    case MVE_VLD41_8_wb:
    case MVE_VLD42_8_wb:
    case MVE_VLD43_8_wb:
    case MVE_VLD40_16_wb:
    case MVE_VLD41_16_wb:
    case MVE_VLD42_16_wb:
    case MVE_VLD43_16_wb:
    case MVE_VLD40_32_wb:
    case MVE_VLD41_32_wb:
    case MVE_VLD42_32_wb:
    case MVE_VLD43_32_wb:
    case MVE_VLDRBS16:
    case MVE_VLDRBS16_post:
    case MVE_VLDRBS16_pre:
    case MVE_VLDRBS16_rq:
    case MVE_VLDRBS32:
    case MVE_VLDRBS32_post:
    case MVE_VLDRBS32_pre:
    case MVE_VLDRBS32_rq:
    case MVE_VLDRBU16:
    case MVE_VLDRBU16_post:
    case MVE_VLDRBU16_pre:
    case MVE_VLDRBU16_rq:
    case MVE_VLDRBU32:
    case MVE_VLDRBU32_post:
    case MVE_VLDRBU32_pre:
    case MVE_VLDRBU32_rq:
    case MVE_VLDRBU8:
    case MVE_VLDRBU8_post:
    case MVE_VLDRBU8_pre:
    case MVE_VLDRBU8_rq:
    case MVE_VLDRDU64_qi:
    case MVE_VLDRDU64_qi_pre:
    case MVE_VLDRDU64_rq:
    case MVE_VLDRDU64_rq_u:
    case MVE_VLDRHS32:
    case MVE_VLDRHS32_post:
    case MVE_VLDRHS32_pre:
    case MVE_VLDRHS32_rq:
    case MVE_VLDRHS32_rq_u:
    case MVE_VLDRHU16:
    case MVE_VLDRHU16_post:
    case MVE_VLDRHU16_pre:
    case MVE_VLDRHU16_rq:
    case MVE_VLDRHU16_rq_u:
    case MVE_VLDRHU32:
    case MVE_VLDRHU32_post:
    case MVE_VLDRHU32_pre:
    case MVE_VLDRHU32_rq:
    case MVE_VLDRHU32_rq_u:
    case MVE_VLDRWU32:
    case MVE_VLDRWU32_post:
    case MVE_VLDRWU32_pre:
    case MVE_VLDRWU32_qi:
    case MVE_VLDRWU32_qi_pre:
    case MVE_VLDRWU32_rq:
    case MVE_VLDRWU32_rq_u:
    case MVE_VMOVimmf32:
    case MVE_VMOVimmi16:
    case MVE_VMOVimmi32:
    case MVE_VMOVimmi64:
    case MVE_VMOVimmi8:
    case MVE_VMOVNi16bh:
    case MVE_VMOVNi16th:
    case MVE_VMOVNi32bh:
    case MVE_VMOVNi32th:
    case MVE_VMULLBp16:
    case MVE_VMULLBp8:
    case MVE_VMULLBs16:
    case MVE_VMULLBs32:
    case MVE_VMULLBs8:
    case MVE_VMULLBu16:
    case MVE_VMULLBu32:
    case MVE_VMULLBu8:
    case MVE_VMULLTp16:
    case MVE_VMULLTp8:
    case MVE_VMULLTs16:
    case MVE_VMULLTs32:
    case MVE_VMULLTs8:
    case MVE_VMULLTu16:
    case MVE_VMULLTu32:
    case MVE_VMULLTu8:
    case MVE_VMUL_qr_f16:
    case MVE_VMUL_qr_f32:
    case MVE_VMUL_qr_i16:
    case MVE_VMUL_qr_i32:
    case MVE_VMUL_qr_i8:
    case MVE_VMULf16:
    case MVE_VMULf32:
    case MVE_VMULi16:
    case MVE_VMULi8:
    case MVE_VMULi32:
    case MVE_VMULHs32:
    case MVE_VMULHs16:
    case MVE_VMULHs8:
    case MVE_VMULHu32:
    case MVE_VMULHu16:
    case MVE_VMULHu8:
    case MVE_VMVN:
    case MVE_VMVNimmi16:
    case MVE_VMVNimmi32:
    case MVE_VNEGf16:
    case MVE_VNEGf32:
    case MVE_VNEGs16:
    case MVE_VNEGs32:
    case MVE_VNEGs8:
    case MVE_VORN:
    case MVE_VORR:
    case MVE_VORRimmi16:
    case MVE_VORRimmi32:
    case MVE_VPST:
    case MVE_VPTv16i8:
    case MVE_VPTv8i16:
    case MVE_VPTv4i32:
    case MVE_VPTv16i8r:
    case MVE_VPTv8i16r:
    case MVE_VPTv4i32r:
    case MVE_VPTv16s8:
    case MVE_VPTv8s16:
    case MVE_VPTv4s32:
    case MVE_VPTv16s8r:
    case MVE_VPTv8s16r:
    case MVE_VPTv4s32r:
    case MVE_VPTv16u8:
    case MVE_VPTv8u16:
    case MVE_VPTv4u32:
    case MVE_VPTv16u8r:
    case MVE_VPTv8u16r:
    case MVE_VPTv4u32r:
    case MVE_VPTv8f16:
    case MVE_VPTv4f32:
    case MVE_VPTv8f16r:
    case MVE_VPTv4f32r:
    case MVE_VQABSs16:
    case MVE_VQABSs32:
    case MVE_VQABSs8:
    case MVE_VQADD_qr_s16:
    case MVE_VQADD_qr_s32:
    case MVE_VQADD_qr_s8:
    case MVE_VQADD_qr_u16:
    case MVE_VQADD_qr_u32:
    case MVE_VQADD_qr_u8:
    case MVE_VQADDs16:
    case MVE_VQADDs32:
    case MVE_VQADDs8:
    case MVE_VQADDu16:
    case MVE_VQADDu32:
    case MVE_VQADDu8:
    case MVE_VQDMULH_qr_s16:
    case MVE_VQDMULH_qr_s32:
    case MVE_VQDMULH_qr_s8:
    case MVE_VQDMULHi16:
    case MVE_VQDMULHi32:
    case MVE_VQDMULHi8:
    case MVE_VQDMULL_qr_s16bh:
    case MVE_VQDMULL_qr_s16th:
    case MVE_VQDMULL_qr_s32bh:
    case MVE_VQDMULL_qr_s32th:
    case MVE_VQDMULLs16bh:
    case MVE_VQDMULLs16th:
    case MVE_VQDMULLs32bh:
    case MVE_VQDMULLs32th:
    case MVE_VQRDMULH_qr_s16:
    case MVE_VQRDMULH_qr_s32:
    case MVE_VQRDMULH_qr_s8:
    case MVE_VQRDMULHi16:
    case MVE_VQRDMULHi32:
    case MVE_VQRDMULHi8:
    case MVE_VQNEGs16:
    case MVE_VQNEGs32:
    case MVE_VQNEGs8:
    case MVE_VQMOVNs16bh:
    case MVE_VQMOVNs16th:
    case MVE_VQMOVNs32bh:
    case MVE_VQMOVNs32th:
    case MVE_VQMOVNu16bh:
    case MVE_VQMOVNu16th:
    case MVE_VQMOVNu32bh:
    case MVE_VQMOVNu32th:
    case MVE_VQMOVUNs16bh:
    case MVE_VQMOVUNs16th:
    case MVE_VQMOVUNs32bh:
    case MVE_VQMOVUNs32th:
    case MVE_VQRSHL_by_vecs16:
    case MVE_VQRSHL_by_vecs32:
    case MVE_VQRSHL_by_vecs8:
    case MVE_VQRSHL_by_vecu16:
    case MVE_VQRSHL_by_vecu32:
    case MVE_VQRSHL_by_vecu8:
    case MVE_VQRSHL_qrs16:
    case MVE_VQRSHL_qrs32:
    case MVE_VQRSHL_qrs8:
    case MVE_VQRSHL_qru16:
    case MVE_VQRSHL_qru8:
    case MVE_VQRSHL_qru32:
    case MVE_VQSHLU_imms16:
    case MVE_VQSHLU_imms32:
    case MVE_VQSHLU_imms8:
    case MVE_VQSHLimms16:
    case MVE_VQSHLimms32:
    case MVE_VQSHLimms8:
    case MVE_VQSHLimmu16:
    case MVE_VQSHLimmu32:
    case MVE_VQSHLimmu8:
    case MVE_VQSHL_by_vecs16:
    case MVE_VQSHL_by_vecs32:
    case MVE_VQSHL_by_vecs8:
    case MVE_VQSHL_by_vecu16:
    case MVE_VQSHL_by_vecu32:
    case MVE_VQSHL_by_vecu8:
    case MVE_VQSHL_qrs16:
    case MVE_VQSHL_qrs32:
    case MVE_VQSHL_qrs8:
    case MVE_VQSHL_qru16:
    case MVE_VQSHL_qru32:
    case MVE_VQSHL_qru8:
    case MVE_VQRSHRNbhs16:
    case MVE_VQRSHRNbhs32:
    case MVE_VQRSHRNbhu16:
    case MVE_VQRSHRNbhu32:
    case MVE_VQRSHRNths16:
    case MVE_VQRSHRNths32:
    case MVE_VQRSHRNthu16:
    case MVE_VQRSHRNthu32:
    case MVE_VQRSHRUNs16bh:
    case MVE_VQRSHRUNs16th:
    case MVE_VQRSHRUNs32bh:
    case MVE_VQRSHRUNs32th:
    case MVE_VQSHRNbhs16:
    case MVE_VQSHRNbhs32:
    case MVE_VQSHRNbhu16:
    case MVE_VQSHRNbhu32:
    case MVE_VQSHRNths16:
    case MVE_VQSHRNths32:
    case MVE_VQSHRNthu16:
    case MVE_VQSHRNthu32:
    case MVE_VQSHRUNs16bh:
    case MVE_VQSHRUNs16th:
    case MVE_VQSHRUNs32bh:
    case MVE_VQSHRUNs32th:
    case MVE_VQSUB_qr_s16:
    case MVE_VQSUB_qr_s32:
    case MVE_VQSUB_qr_s8:
    case MVE_VQSUB_qr_u16:
    case MVE_VQSUB_qr_u32:
    case MVE_VQSUB_qr_u8:
    case MVE_VQSUBs16:
    case MVE_VQSUBs32:
    case MVE_VQSUBs8:
    case MVE_VQSUBu16:
    case MVE_VQSUBu32:
    case MVE_VQSUBu8:
    case MVE_VRHADDs16:
    case MVE_VRHADDs32:
    case MVE_VRHADDs8:
    case MVE_VRHADDu16:
    case MVE_VRHADDu32:
    case MVE_VRHADDu8:
    case MVE_VRINTf16A:
    case MVE_VRINTf16M:
    case MVE_VRINTf16N:
    case MVE_VRINTf16P:
    case MVE_VRINTf16X:
    case MVE_VRINTf16Z:
    case MVE_VRINTf32A:
    case MVE_VRINTf32M:
    case MVE_VRINTf32N:
    case MVE_VRINTf32P:
    case MVE_VRINTf32X:
    case MVE_VRINTf32Z:
    case MVE_VRMULHs32:
    case MVE_VRMULHs16:
    case MVE_VRMULHs8:
    case MVE_VRMULHu32:
    case MVE_VRMULHu16:
    case MVE_VRMULHu8:
    case MVE_VRSHL_by_vecs16:
    case MVE_VRSHL_by_vecs32:
    case MVE_VRSHL_by_vecs8:
    case MVE_VRSHL_by_vecu16:
    case MVE_VRSHL_by_vecu32:
    case MVE_VRSHL_by_vecu8:
    case MVE_VRSHL_qrs16:
    case MVE_VRSHL_qrs32:
    case MVE_VRSHL_qrs8:
    case MVE_VRSHL_qru16:
    case MVE_VRSHL_qru32:
    case MVE_VRSHL_qru8:
    case MVE_VRSHR_imms16:
    case MVE_VRSHR_imms32:
    case MVE_VRSHR_imms8:
    case MVE_VRSHR_immu16:
    case MVE_VRSHR_immu32:
    case MVE_VRSHR_immu8:
    case MVE_VRSHRNi16bh:
    case MVE_VRSHRNi16th:
    case MVE_VRSHRNi32bh:
    case MVE_VRSHRNi32th:
    case MVE_VSHL_by_vecs16:
    case MVE_VSHL_by_vecs32:
    case MVE_VSHL_by_vecs8:
    case MVE_VSHL_by_vecu16:
    case MVE_VSHL_by_vecu32:
    case MVE_VSHL_by_vecu8:
    case MVE_VSHL_immi16:
    case MVE_VSHL_immi32:
    case MVE_VSHL_immi8:
    case MVE_VSHL_qrs16:
    case MVE_VSHL_qrs32:
    case MVE_VSHL_qrs8:
    case MVE_VSHL_qru16:
    case MVE_VSHL_qru32:
    case MVE_VSHL_qru8:
    case MVE_VSHR_imms16:
    case MVE_VSHR_imms32:
    case MVE_VSHR_imms8:
    case MVE_VSHR_immu16:
    case MVE_VSHR_immu32:
    case MVE_VSHR_immu8:
    case MVE_VSHRNi16bh:
    case MVE_VSHRNi16th:
    case MVE_VSHRNi32bh:
    case MVE_VSHRNi32th:
    case MVE_VSLIimm16:
    case MVE_VSLIimm32:
    case MVE_VSLIimm8:
    case MVE_VSRIimm16:
    case MVE_VSRIimm32:
    case MVE_VSRIimm8:
    case MVE_VSTRB16:
    case MVE_VSTRB16_post:
    case MVE_VSTRB16_pre:
    case MVE_VSTRB16_rq:
    case MVE_VSTRB32:
    case MVE_VSTRB32_post:
    case MVE_VSTRB32_pre:
    case MVE_VSTRB32_rq:
    case MVE_VSTRB8_rq:
    case MVE_VSTRBU8:
    case MVE_VSTRBU8_post:
    case MVE_VSTRBU8_pre:
    case MVE_VSTRD64_qi:
    case MVE_VSTRD64_qi_pre:
    case MVE_VSTRD64_rq:
    case MVE_VSTRD64_rq_u:
    case MVE_VSTRH16_rq:
    case MVE_VSTRH16_rq_u:
    case MVE_VSTRH32:
    case MVE_VSTRH32_post:
    case MVE_VSTRH32_pre:
    case MVE_VSTRH32_rq:
    case MVE_VSTRH32_rq_u:
    case MVE_VSTRHU16:
    case MVE_VSTRHU16_post:
    case MVE_VSTRHU16_pre:
    case MVE_VSTRW32_qi:
    case MVE_VSTRW32_qi_pre:
    case MVE_VSTRW32_rq:
    case MVE_VSTRW32_rq_u:
    case MVE_VSTRWU32:
    case MVE_VSTRWU32_post:
    case MVE_VSTRWU32_pre:
    case MVE_VSUB_qr_f16:
    case MVE_VSUB_qr_f32:
    case MVE_VSUB_qr_i16:
    case MVE_VSUB_qr_i32:
    case MVE_VSUB_qr_i8:
    case MVE_VSUBf16:
    case MVE_VSUBf32:
    case MVE_VSUBi16:
    case MVE_VSUBi32:
    case MVE_VSUBi8:
    case VLDR_P0_off:
    case VLDR_P0_post:
    case VLDR_P0_pre:
    case VLDR_VPR_off:
    case VLDR_VPR_post:
    case VLDR_VPR_pre:
    case VSTR_P0_off:
    case VSTR_P0_post:
    case VSTR_P0_pre:
    case VSTR_VPR_off:
    case VSTR_VPR_post:
    case VSTR_VPR_pre:
    case VMRS_P0:
    case VMRS_VPR:
      return true;
    }
  };

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
    static_cast<LLVMTargetMachine*>(
      T->createTargetMachine(TT, "generic", "", Options, None, None,
                             CodeGenOpt::Default)));
  ARMSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                  std::string(TM->getTargetFeatureString()),
                  *static_cast<const ARMBaseTargetMachine *>(TM.get()), false);

  auto MII = TM->getMCInstrInfo();
  for (unsigned i = 0; i < ARM::INSTRUCTION_LIST_END; ++i) {
    uint64_t Flags = MII->get(i).TSFlags;
    if ((Flags & ARMII::DomainMask) != ARMII::DomainMVE)
      continue;
    bool Valid = (Flags & ARMII::ValidForTailPredication) != 0;
    ASSERT_EQ(IsValidTPOpcode(i), Valid)
              << MII->getName(i)
              << ": mismatched expectation for tail-predicated safety\n";
  }
}

TEST(MachineInstr, HasSideEffects) {
  using namespace ARM;
  std::set<unsigned> UnpredictableOpcodes = {
      // MVE Instructions
      MVE_VCTP8,
      MVE_VCTP16,
      MVE_VCTP32,
      MVE_VCTP64,
      MVE_VPST,
      MVE_VPTv16i8,
      MVE_VPTv8i16,
      MVE_VPTv4i32,
      MVE_VPTv16i8r,
      MVE_VPTv8i16r,
      MVE_VPTv4i32r,
      MVE_VPTv16s8,
      MVE_VPTv8s16,
      MVE_VPTv4s32,
      MVE_VPTv16s8r,
      MVE_VPTv8s16r,
      MVE_VPTv4s32r,
      MVE_VPTv16u8,
      MVE_VPTv8u16,
      MVE_VPTv4u32,
      MVE_VPTv16u8r,
      MVE_VPTv8u16r,
      MVE_VPTv4u32r,
      MVE_VPTv8f16,
      MVE_VPTv4f32,
      MVE_VPTv8f16r,
      MVE_VPTv4f32r,
      MVE_VADC,
      MVE_VADCI,
      MVE_VSBC,
      MVE_VSBCI,
      MVE_VSHLC,
      // FP Instructions
      FLDMXIA,
      FLDMXDB_UPD,
      FLDMXIA_UPD,
      FSTMXDB_UPD,
      FSTMXIA,
      FSTMXIA_UPD,
      VLDR_FPCXTNS_off,
      VLDR_FPCXTNS_off,
      VLDR_FPCXTNS_post,
      VLDR_FPCXTNS_pre,
      VLDR_FPCXTS_off,
      VLDR_FPCXTS_post,
      VLDR_FPCXTS_pre,
      VLDR_FPSCR_NZCVQC_off,
      VLDR_FPSCR_NZCVQC_post,
      VLDR_FPSCR_NZCVQC_pre,
      VLDR_FPSCR_off,
      VLDR_FPSCR_post,
      VLDR_FPSCR_pre,
      VLDR_P0_off,
      VLDR_P0_post,
      VLDR_P0_pre,
      VLDR_VPR_off,
      VLDR_VPR_post,
      VLDR_VPR_pre,
      VLLDM,
      VLSTM,
      VMRS,
      VMRS_FPCXTNS,
      VMRS_FPCXTS,
      VMRS_FPEXC,
      VMRS_FPINST,
      VMRS_FPINST2,
      VMRS_FPSCR_NZCVQC,
      VMRS_FPSID,
      VMRS_MVFR0,
      VMRS_MVFR1,
      VMRS_MVFR2,
      VMRS_P0,
      VMRS_VPR,
      VMSR,
      VMSR_FPCXTNS,
      VMSR_FPCXTS,
      VMSR_FPEXC,
      VMSR_FPINST,
      VMSR_FPINST2,
      VMSR_FPSCR_NZCVQC,
      VMSR_FPSID,
      VMSR_P0,
      VMSR_VPR,
      VSCCLRMD,
      VSCCLRMS,
      VSTR_FPCXTNS_off,
      VSTR_FPCXTNS_post,
      VSTR_FPCXTNS_pre,
      VSTR_FPCXTS_off,
      VSTR_FPCXTS_post,
      VSTR_FPCXTS_pre,
      VSTR_FPSCR_NZCVQC_off,
      VSTR_FPSCR_NZCVQC_post,
      VSTR_FPSCR_NZCVQC_pre,
      VSTR_FPSCR_off,
      VSTR_FPSCR_post,
      VSTR_FPSCR_pre,
      VSTR_P0_off,
      VSTR_P0_post,
      VSTR_P0_pre,
      VSTR_VPR_off,
      VSTR_VPR_post,
      VSTR_VPR_pre,
  };

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
  const ARMBaseInstrInfo *TII = ST.getInstrInfo();
  auto MII = TM->getMCInstrInfo();

  for (unsigned Op = 0; Op < ARM::INSTRUCTION_LIST_END; ++Op) {
    const MCInstrDesc &Desc = TII->get(Op);
    if ((Desc.TSFlags &
         (ARMII::DomainMVE | ARMII::DomainVFP | ARMII::DomainNEONA8)) == 0)
      continue;
    if (UnpredictableOpcodes.count(Op))
      continue;

    ASSERT_FALSE(Desc.hasUnmodeledSideEffects())
        << MII->getName(Op) << " has unexpected side effects";
  }
}

TEST(MachineInstr, MVEVecSize) {
  using namespace ARM;
  auto MVEVecSize = [](unsigned Opcode) {
    switch (Opcode) {
    default:
    dbgs() << Opcode << "\n";
      llvm_unreachable("Unexpected MVE instruction!");
    case MVE_ASRLi:
    case MVE_ASRLr:
    case MVE_LSLLi:
    case MVE_LSLLr:
    case MVE_LSRL:
    case MVE_SQRSHR:
    case MVE_SQRSHRL:
    case MVE_SQSHL:
    case MVE_SQSHLL:
    case MVE_SRSHR:
    case MVE_SRSHRL:
    case MVE_UQRSHL:
    case MVE_UQRSHLL:
    case MVE_UQSHL:
    case MVE_UQSHLL:
    case MVE_URSHR:
    case MVE_URSHRL:
    case MVE_VABAVs8:
    case MVE_VABAVu8:
    case MVE_VABDs8:
    case MVE_VABDu8:
    case MVE_VABSs8:
    case MVE_VADDVs8acc:
    case MVE_VADDVs8no_acc:
    case MVE_VADDVu8acc:
    case MVE_VADDVu8no_acc:
    case MVE_VADD_qr_i8:
    case MVE_VADDi8:
    case MVE_VBRSR8:
    case MVE_VCADDi8:
    case MVE_VCLSs8:
    case MVE_VCLZs8:
    case MVE_VCMPi8:
    case MVE_VCMPi8r:
    case MVE_VCMPs8:
    case MVE_VCMPs8r:
    case MVE_VCMPu8:
    case MVE_VCMPu8r:
    case MVE_VCTP8:
    case MVE_VDDUPu8:
    case MVE_VDUP8:
    case MVE_VDWDUPu8:
    case MVE_VHADD_qr_s8:
    case MVE_VHADD_qr_u8:
    case MVE_VHADDs8:
    case MVE_VHADDu8:
    case MVE_VHCADDs8:
    case MVE_VHSUB_qr_s8:
    case MVE_VHSUB_qr_u8:
    case MVE_VHSUBs8:
    case MVE_VHSUBu8:
    case MVE_VIDUPu8:
    case MVE_VIWDUPu8:
    case MVE_VLD20_8:
    case MVE_VLD20_8_wb:
    case MVE_VLD21_8:
    case MVE_VLD21_8_wb:
    case MVE_VLD40_8:
    case MVE_VLD40_8_wb:
    case MVE_VLD41_8:
    case MVE_VLD41_8_wb:
    case MVE_VLD42_8:
    case MVE_VLD42_8_wb:
    case MVE_VLD43_8:
    case MVE_VLD43_8_wb:
    case MVE_VLDRBU8:
    case MVE_VLDRBU8_post:
    case MVE_VLDRBU8_pre:
    case MVE_VLDRBU8_rq:
    case MVE_VMAXAVs8:
    case MVE_VMAXAs8:
    case MVE_VMAXVs8:
    case MVE_VMAXVu8:
    case MVE_VMAXs8:
    case MVE_VMAXu8:
    case MVE_VMINAVs8:
    case MVE_VMINAs8:
    case MVE_VMINVs8:
    case MVE_VMINVu8:
    case MVE_VMINs8:
    case MVE_VMINu8:
    case MVE_VMLADAVas8:
    case MVE_VMLADAVau8:
    case MVE_VMLADAVaxs8:
    case MVE_VMLADAVs8:
    case MVE_VMLADAVu8:
    case MVE_VMLADAVxs8:
    case MVE_VMLAS_qr_i8:
    case MVE_VMLA_qr_i8:
    case MVE_VMLSDAVas8:
    case MVE_VMLSDAVaxs8:
    case MVE_VMLSDAVs8:
    case MVE_VMLSDAVxs8:
    case MVE_VMOV_from_lane_s8:
    case MVE_VMOV_from_lane_u8:
    case MVE_VMOV_to_lane_8:
    case MVE_VMOVimmi8:
    case MVE_VMULHs8:
    case MVE_VMULHu8:
    case MVE_VMUL_qr_i8:
    case MVE_VMULi8:
    case MVE_VNEGs8:
    case MVE_VPTv16i8:
    case MVE_VPTv16i8r:
    case MVE_VPTv16s8:
    case MVE_VPTv16s8r:
    case MVE_VPTv16u8:
    case MVE_VPTv16u8r:
    case MVE_VQABSs8:
    case MVE_VQADD_qr_s8:
    case MVE_VQADD_qr_u8:
    case MVE_VQADDs8:
    case MVE_VQADDu8:
    case MVE_VQDMLADHXs8:
    case MVE_VQDMLADHs8:
    case MVE_VQDMLAH_qrs8:
    case MVE_VQDMLASH_qrs8:
    case MVE_VQDMLSDHXs8:
    case MVE_VQDMLSDHs8:
    case MVE_VQDMULH_qr_s8:
    case MVE_VQDMULHi8:
    case MVE_VQNEGs8:
    case MVE_VQRDMLADHXs8:
    case MVE_VQRDMLADHs8:
    case MVE_VQRDMLAH_qrs8:
    case MVE_VQRDMLASH_qrs8:
    case MVE_VQRDMLSDHXs8:
    case MVE_VQRDMLSDHs8:
    case MVE_VQRDMULH_qr_s8:
    case MVE_VQRDMULHi8:
    case MVE_VQRSHL_by_vecs8:
    case MVE_VQRSHL_by_vecu8:
    case MVE_VQRSHL_qrs8:
    case MVE_VQRSHL_qru8:
    case MVE_VQSHLU_imms8:
    case MVE_VQSHL_by_vecs8:
    case MVE_VQSHL_by_vecu8:
    case MVE_VQSHL_qrs8:
    case MVE_VQSHL_qru8:
    case MVE_VQSHLimms8:
    case MVE_VQSHLimmu8:
    case MVE_VQSUB_qr_s8:
    case MVE_VQSUB_qr_u8:
    case MVE_VQSUBs8:
    case MVE_VQSUBu8:
    case MVE_VRHADDs8:
    case MVE_VRHADDu8:
    case MVE_VRMULHs8:
    case MVE_VRMULHu8:
    case MVE_VRSHL_by_vecs8:
    case MVE_VRSHL_by_vecu8:
    case MVE_VRSHL_qrs8:
    case MVE_VRSHL_qru8:
    case MVE_VRSHR_imms8:
    case MVE_VRSHR_immu8:
    case MVE_VSHL_by_vecs8:
    case MVE_VSHL_by_vecu8:
    case MVE_VSHL_immi8:
    case MVE_VSHL_qru8:
    case MVE_VSHL_qrs8:
    case MVE_VSHR_imms8:
    case MVE_VSHR_immu8:
    case MVE_VSLIimm8:
    case MVE_VSRIimm8:
    case MVE_VST20_8:
    case MVE_VST20_8_wb:
    case MVE_VST21_8:
    case MVE_VST21_8_wb:
    case MVE_VST40_8:
    case MVE_VST40_8_wb:
    case MVE_VST41_8:
    case MVE_VST41_8_wb:
    case MVE_VST42_8:
    case MVE_VST42_8_wb:
    case MVE_VST43_8:
    case MVE_VST43_8_wb:
    case MVE_VSTRB8_rq:
    case MVE_VSTRBU8:
    case MVE_VSTRBU8_post:
    case MVE_VSTRBU8_pre:
    case MVE_VSUB_qr_i8:
    case MVE_VSUBi8:
    case MVE_VAND:
    case MVE_VBIC:
    case MVE_VEOR:
    case MVE_VMVN:
    case MVE_VORN:
    case MVE_VORR:
    case MVE_VPNOT:
    case MVE_VPSEL:
    case MVE_VPST:
    case MQPRCopy:
      return 0;
    case MVE_VABAVs16:
    case MVE_VABAVu16:
    case MVE_VABDf16:
    case MVE_VABDs16:
    case MVE_VABDu16:
    case MVE_VABSf16:
    case MVE_VABSs16:
    case MVE_VADDVs16acc:
    case MVE_VADDVs16no_acc:
    case MVE_VADDVu16acc:
    case MVE_VADDVu16no_acc:
    case MVE_VADD_qr_f16:
    case MVE_VADD_qr_i16:
    case MVE_VADDf16:
    case MVE_VADDi16:
    case MVE_VBICimmi16:
    case MVE_VBRSR16:
    case MVE_VCADDf16:
    case MVE_VCADDi16:
    case MVE_VCLSs16:
    case MVE_VCLZs16:
    case MVE_VCMLAf16:
    case MVE_VCMPf16:
    case MVE_VCMPf16r:
    case MVE_VCMPi16:
    case MVE_VCMPi16r:
    case MVE_VCMPs16:
    case MVE_VCMPs16r:
    case MVE_VCMPu16:
    case MVE_VCMPu16r:
    case MVE_VCMULf16:
    case MVE_VCTP16:
    case MVE_VCVTf16s16_fix:
    case MVE_VCVTf16s16n:
    case MVE_VCVTf16u16_fix:
    case MVE_VCVTf16u16n:
    case MVE_VCVTs16f16_fix:
    case MVE_VCVTs16f16a:
    case MVE_VCVTs16f16m:
    case MVE_VCVTs16f16n:
    case MVE_VCVTs16f16p:
    case MVE_VCVTs16f16z:
    case MVE_VCVTu16f16_fix:
    case MVE_VCVTu16f16a:
    case MVE_VCVTu16f16m:
    case MVE_VCVTu16f16n:
    case MVE_VCVTu16f16p:
    case MVE_VCVTu16f16z:
    case MVE_VDDUPu16:
    case MVE_VDUP16:
    case MVE_VDWDUPu16:
    case MVE_VFMA_qr_Sf16:
    case MVE_VFMA_qr_f16:
    case MVE_VFMAf16:
    case MVE_VFMSf16:
    case MVE_VHADD_qr_s16:
    case MVE_VHADD_qr_u16:
    case MVE_VHADDs16:
    case MVE_VHADDu16:
    case MVE_VHCADDs16:
    case MVE_VHSUB_qr_s16:
    case MVE_VHSUB_qr_u16:
    case MVE_VHSUBs16:
    case MVE_VHSUBu16:
    case MVE_VIDUPu16:
    case MVE_VIWDUPu16:
    case MVE_VLD20_16:
    case MVE_VLD20_16_wb:
    case MVE_VLD21_16:
    case MVE_VLD21_16_wb:
    case MVE_VLD40_16:
    case MVE_VLD40_16_wb:
    case MVE_VLD41_16:
    case MVE_VLD41_16_wb:
    case MVE_VLD42_16:
    case MVE_VLD42_16_wb:
    case MVE_VLD43_16:
    case MVE_VLD43_16_wb:
    case MVE_VLDRBS16:
    case MVE_VLDRBS16_post:
    case MVE_VLDRBS16_pre:
    case MVE_VLDRBS16_rq:
    case MVE_VLDRBU16:
    case MVE_VLDRBU16_post:
    case MVE_VLDRBU16_pre:
    case MVE_VLDRBU16_rq:
    case MVE_VLDRHU16:
    case MVE_VLDRHU16_post:
    case MVE_VLDRHU16_pre:
    case MVE_VLDRHU16_rq:
    case MVE_VLDRHU16_rq_u:
    case MVE_VMAXAVs16:
    case MVE_VMAXAs16:
    case MVE_VMAXNMAVf16:
    case MVE_VMAXNMAf16:
    case MVE_VMAXNMVf16:
    case MVE_VMAXNMf16:
    case MVE_VMAXVs16:
    case MVE_VMAXVu16:
    case MVE_VMAXs16:
    case MVE_VMAXu16:
    case MVE_VMINAVs16:
    case MVE_VMINAs16:
    case MVE_VMINNMAVf16:
    case MVE_VMINNMAf16:
    case MVE_VMINNMVf16:
    case MVE_VMINNMf16:
    case MVE_VMINVs16:
    case MVE_VMINVu16:
    case MVE_VMINs16:
    case MVE_VMINu16:
    case MVE_VMLADAVas16:
    case MVE_VMLADAVau16:
    case MVE_VMLADAVaxs16:
    case MVE_VMLADAVs16:
    case MVE_VMLADAVu16:
    case MVE_VMLADAVxs16:
    case MVE_VMLALDAVas16:
    case MVE_VMLALDAVau16:
    case MVE_VMLALDAVaxs16:
    case MVE_VMLALDAVs16:
    case MVE_VMLALDAVu16:
    case MVE_VMLALDAVxs16:
    case MVE_VMLAS_qr_i16:
    case MVE_VMLA_qr_i16:
    case MVE_VMLSDAVas16:
    case MVE_VMLSDAVaxs16:
    case MVE_VMLSDAVs16:
    case MVE_VMLSDAVxs16:
    case MVE_VMLSLDAVas16:
    case MVE_VMLSLDAVaxs16:
    case MVE_VMLSLDAVs16:
    case MVE_VMLSLDAVxs16:
    case MVE_VMOVNi16bh:
    case MVE_VMOVNi16th:
    case MVE_VMOV_from_lane_s16:
    case MVE_VMOV_from_lane_u16:
    case MVE_VMOV_to_lane_16:
    case MVE_VMOVimmi16:
    case MVE_VMOVLs8bh:
    case MVE_VMOVLs8th:
    case MVE_VMOVLu8bh:
    case MVE_VMOVLu8th:
    case MVE_VMULLBp8:
    case MVE_VMULLBs8:
    case MVE_VMULLBu8:
    case MVE_VMULLTp8:
    case MVE_VMULLTs8:
    case MVE_VMULLTu8:
    case MVE_VMULHs16:
    case MVE_VMULHu16:
    case MVE_VMUL_qr_f16:
    case MVE_VMUL_qr_i16:
    case MVE_VMULf16:
    case MVE_VMULi16:
    case MVE_VMVNimmi16:
    case MVE_VNEGf16:
    case MVE_VNEGs16:
    case MVE_VORRimmi16:
    case MVE_VPTv8f16:
    case MVE_VPTv8f16r:
    case MVE_VPTv8i16:
    case MVE_VPTv8i16r:
    case MVE_VPTv8s16:
    case MVE_VPTv8s16r:
    case MVE_VPTv8u16:
    case MVE_VPTv8u16r:
    case MVE_VQABSs16:
    case MVE_VQADD_qr_s16:
    case MVE_VQADD_qr_u16:
    case MVE_VQADDs16:
    case MVE_VQADDu16:
    case MVE_VQDMLADHXs16:
    case MVE_VQDMLADHs16:
    case MVE_VQDMLAH_qrs16:
    case MVE_VQDMLASH_qrs16:
    case MVE_VQDMLSDHXs16:
    case MVE_VQDMLSDHs16:
    case MVE_VQDMULH_qr_s16:
    case MVE_VQDMULHi16:
    case MVE_VQDMULL_qr_s16bh:
    case MVE_VQDMULL_qr_s16th:
    case MVE_VQDMULLs16bh:
    case MVE_VQDMULLs16th:
    case MVE_VQMOVNs16bh:
    case MVE_VQMOVNs16th:
    case MVE_VQMOVNu16bh:
    case MVE_VQMOVNu16th:
    case MVE_VQMOVUNs16bh:
    case MVE_VQMOVUNs16th:
    case MVE_VQNEGs16:
    case MVE_VQRDMLADHXs16:
    case MVE_VQRDMLADHs16:
    case MVE_VQRDMLAH_qrs16:
    case MVE_VQRDMLASH_qrs16:
    case MVE_VQRDMLSDHXs16:
    case MVE_VQRDMLSDHs16:
    case MVE_VQRDMULH_qr_s16:
    case MVE_VQRDMULHi16:
    case MVE_VQRSHL_by_vecs16:
    case MVE_VQRSHL_by_vecu16:
    case MVE_VQRSHL_qrs16:
    case MVE_VQRSHL_qru16:
    case MVE_VQRSHRNbhs16:
    case MVE_VQRSHRNbhu16:
    case MVE_VQRSHRNths16:
    case MVE_VQRSHRNthu16:
    case MVE_VQRSHRUNs16bh:
    case MVE_VQRSHRUNs16th:
    case MVE_VQSHLU_imms16:
    case MVE_VQSHL_by_vecs16:
    case MVE_VQSHL_by_vecu16:
    case MVE_VQSHL_qrs16:
    case MVE_VQSHL_qru16:
    case MVE_VQSHLimms16:
    case MVE_VQSHLimmu16:
    case MVE_VQSHRNbhs16:
    case MVE_VQSHRNbhu16:
    case MVE_VQSHRNths16:
    case MVE_VQSHRNthu16:
    case MVE_VQSHRUNs16bh:
    case MVE_VQSHRUNs16th:
    case MVE_VQSUB_qr_s16:
    case MVE_VQSUB_qr_u16:
    case MVE_VQSUBs16:
    case MVE_VQSUBu16:
    case MVE_VREV16_8:
    case MVE_VRHADDs16:
    case MVE_VRHADDu16:
    case MVE_VRINTf16A:
    case MVE_VRINTf16M:
    case MVE_VRINTf16N:
    case MVE_VRINTf16P:
    case MVE_VRINTf16X:
    case MVE_VRINTf16Z:
    case MVE_VRMULHs16:
    case MVE_VRMULHu16:
    case MVE_VRSHL_by_vecs16:
    case MVE_VRSHL_by_vecu16:
    case MVE_VRSHL_qrs16:
    case MVE_VRSHL_qru16:
    case MVE_VRSHRNi16bh:
    case MVE_VRSHRNi16th:
    case MVE_VRSHR_imms16:
    case MVE_VRSHR_immu16:
    case MVE_VSHLL_imms8bh:
    case MVE_VSHLL_imms8th:
    case MVE_VSHLL_immu8bh:
    case MVE_VSHLL_immu8th:
    case MVE_VSHLL_lws8bh:
    case MVE_VSHLL_lws8th:
    case MVE_VSHLL_lwu8bh:
    case MVE_VSHLL_lwu8th:
    case MVE_VSHL_by_vecs16:
    case MVE_VSHL_by_vecu16:
    case MVE_VSHL_immi16:
    case MVE_VSHL_qrs16:
    case MVE_VSHL_qru16:
    case MVE_VSHRNi16bh:
    case MVE_VSHRNi16th:
    case MVE_VSHR_imms16:
    case MVE_VSHR_immu16:
    case MVE_VSLIimm16:
    case MVE_VSRIimm16:
    case MVE_VST20_16:
    case MVE_VST20_16_wb:
    case MVE_VST21_16:
    case MVE_VST21_16_wb:
    case MVE_VST40_16:
    case MVE_VST40_16_wb:
    case MVE_VST41_16:
    case MVE_VST41_16_wb:
    case MVE_VST42_16:
    case MVE_VST42_16_wb:
    case MVE_VST43_16:
    case MVE_VST43_16_wb:
    case MVE_VSTRB16:
    case MVE_VSTRB16_post:
    case MVE_VSTRB16_pre:
    case MVE_VSTRB16_rq:
    case MVE_VSTRH16_rq:
    case MVE_VSTRH16_rq_u:
    case MVE_VSTRHU16:
    case MVE_VSTRHU16_post:
    case MVE_VSTRHU16_pre:
    case MVE_VSUB_qr_f16:
    case MVE_VSUB_qr_i16:
    case MVE_VSUBf16:
    case MVE_VSUBi16:
      return 1;
    case MVE_VABAVs32:
    case MVE_VABAVu32:
    case MVE_VABDf32:
    case MVE_VABDs32:
    case MVE_VABDu32:
    case MVE_VABSf32:
    case MVE_VABSs32:
    case MVE_VADC:
    case MVE_VADCI:
    case MVE_VADDLVs32acc:
    case MVE_VADDLVs32no_acc:
    case MVE_VADDLVu32acc:
    case MVE_VADDLVu32no_acc:
    case MVE_VADDVs32acc:
    case MVE_VADDVs32no_acc:
    case MVE_VADDVu32acc:
    case MVE_VADDVu32no_acc:
    case MVE_VADD_qr_f32:
    case MVE_VADD_qr_i32:
    case MVE_VADDf32:
    case MVE_VADDi32:
    case MVE_VBICimmi32:
    case MVE_VBRSR32:
    case MVE_VCADDf32:
    case MVE_VCADDi32:
    case MVE_VCLSs32:
    case MVE_VCLZs32:
    case MVE_VCMLAf32:
    case MVE_VCMPf32:
    case MVE_VCMPf32r:
    case MVE_VCMPi32:
    case MVE_VCMPi32r:
    case MVE_VCMPs32:
    case MVE_VCMPs32r:
    case MVE_VCMPu32:
    case MVE_VCMPu32r:
    case MVE_VCMULf32:
    case MVE_VCTP32:
    case MVE_VCVTf16f32bh:
    case MVE_VCVTf16f32th:
    case MVE_VCVTf32f16bh:
    case MVE_VCVTf32f16th:
    case MVE_VCVTf32s32_fix:
    case MVE_VCVTf32s32n:
    case MVE_VCVTf32u32_fix:
    case MVE_VCVTf32u32n:
    case MVE_VCVTs32f32_fix:
    case MVE_VCVTs32f32a:
    case MVE_VCVTs32f32m:
    case MVE_VCVTs32f32n:
    case MVE_VCVTs32f32p:
    case MVE_VCVTs32f32z:
    case MVE_VCVTu32f32_fix:
    case MVE_VCVTu32f32a:
    case MVE_VCVTu32f32m:
    case MVE_VCVTu32f32n:
    case MVE_VCVTu32f32p:
    case MVE_VCVTu32f32z:
    case MVE_VDDUPu32:
    case MVE_VDUP32:
    case MVE_VDWDUPu32:
    case MVE_VFMA_qr_Sf32:
    case MVE_VFMA_qr_f32:
    case MVE_VFMAf32:
    case MVE_VFMSf32:
    case MVE_VHADD_qr_s32:
    case MVE_VHADD_qr_u32:
    case MVE_VHADDs32:
    case MVE_VHADDu32:
    case MVE_VHCADDs32:
    case MVE_VHSUB_qr_s32:
    case MVE_VHSUB_qr_u32:
    case MVE_VHSUBs32:
    case MVE_VHSUBu32:
    case MVE_VIDUPu32:
    case MVE_VIWDUPu32:
    case MVE_VLD20_32:
    case MVE_VLD20_32_wb:
    case MVE_VLD21_32:
    case MVE_VLD21_32_wb:
    case MVE_VLD40_32:
    case MVE_VLD40_32_wb:
    case MVE_VLD41_32:
    case MVE_VLD41_32_wb:
    case MVE_VLD42_32:
    case MVE_VLD42_32_wb:
    case MVE_VLD43_32:
    case MVE_VLD43_32_wb:
    case MVE_VLDRBS32:
    case MVE_VLDRBS32_post:
    case MVE_VLDRBS32_pre:
    case MVE_VLDRBS32_rq:
    case MVE_VLDRBU32:
    case MVE_VLDRBU32_post:
    case MVE_VLDRBU32_pre:
    case MVE_VLDRBU32_rq:
    case MVE_VLDRHS32:
    case MVE_VLDRHS32_post:
    case MVE_VLDRHS32_pre:
    case MVE_VLDRHS32_rq:
    case MVE_VLDRHS32_rq_u:
    case MVE_VLDRHU32:
    case MVE_VLDRHU32_post:
    case MVE_VLDRHU32_pre:
    case MVE_VLDRHU32_rq:
    case MVE_VLDRHU32_rq_u:
    case MVE_VLDRWU32:
    case MVE_VLDRWU32_post:
    case MVE_VLDRWU32_pre:
    case MVE_VLDRWU32_qi:
    case MVE_VLDRWU32_qi_pre:
    case MVE_VLDRWU32_rq:
    case MVE_VLDRWU32_rq_u:
    case MVE_VMAXAVs32:
    case MVE_VMAXAs32:
    case MVE_VMAXNMAVf32:
    case MVE_VMAXNMAf32:
    case MVE_VMAXNMVf32:
    case MVE_VMAXNMf32:
    case MVE_VMAXVs32:
    case MVE_VMAXVu32:
    case MVE_VMAXs32:
    case MVE_VMAXu32:
    case MVE_VMINAVs32:
    case MVE_VMINAs32:
    case MVE_VMINNMAVf32:
    case MVE_VMINNMAf32:
    case MVE_VMINNMVf32:
    case MVE_VMINNMf32:
    case MVE_VMINVs32:
    case MVE_VMINVu32:
    case MVE_VMINs32:
    case MVE_VMINu32:
    case MVE_VMLADAVas32:
    case MVE_VMLADAVau32:
    case MVE_VMLADAVaxs32:
    case MVE_VMLADAVs32:
    case MVE_VMLADAVu32:
    case MVE_VMLADAVxs32:
    case MVE_VMLALDAVas32:
    case MVE_VMLALDAVau32:
    case MVE_VMLALDAVaxs32:
    case MVE_VMLALDAVs32:
    case MVE_VMLALDAVu32:
    case MVE_VMLALDAVxs32:
    case MVE_VMLAS_qr_i32:
    case MVE_VMLA_qr_i32:
    case MVE_VMLSDAVas32:
    case MVE_VMLSDAVaxs32:
    case MVE_VMLSDAVs32:
    case MVE_VMLSDAVxs32:
    case MVE_VMLSLDAVas32:
    case MVE_VMLSLDAVaxs32:
    case MVE_VMLSLDAVs32:
    case MVE_VMLSLDAVxs32:
    case MVE_VMOVNi32bh:
    case MVE_VMOVNi32th:
    case MVE_VMOV_from_lane_32:
    case MVE_VMOV_q_rr:
    case MVE_VMOV_rr_q:
    case MVE_VMOV_to_lane_32:
    case MVE_VMOVimmf32:
    case MVE_VMOVimmi32:
    case MVE_VMOVLs16bh:
    case MVE_VMOVLs16th:
    case MVE_VMOVLu16bh:
    case MVE_VMOVLu16th:
    case MVE_VMULHs32:
    case MVE_VMULHu32:
    case MVE_VMULLBp16:
    case MVE_VMULLBs16:
    case MVE_VMULLBu16:
    case MVE_VMULLTp16:
    case MVE_VMULLTs16:
    case MVE_VMULLTu16:
    case MVE_VMUL_qr_f32:
    case MVE_VMUL_qr_i32:
    case MVE_VMULf32:
    case MVE_VMULi32:
    case MVE_VMVNimmi32:
    case MVE_VNEGf32:
    case MVE_VNEGs32:
    case MVE_VORRimmi32:
    case MVE_VPTv4f32:
    case MVE_VPTv4f32r:
    case MVE_VPTv4i32:
    case MVE_VPTv4i32r:
    case MVE_VPTv4s32:
    case MVE_VPTv4s32r:
    case MVE_VPTv4u32:
    case MVE_VPTv4u32r:
    case MVE_VQABSs32:
    case MVE_VQADD_qr_s32:
    case MVE_VQADD_qr_u32:
    case MVE_VQADDs32:
    case MVE_VQADDu32:
    case MVE_VQDMLADHXs32:
    case MVE_VQDMLADHs32:
    case MVE_VQDMLAH_qrs32:
    case MVE_VQDMLASH_qrs32:
    case MVE_VQDMLSDHXs32:
    case MVE_VQDMLSDHs32:
    case MVE_VQDMULH_qr_s32:
    case MVE_VQDMULHi32:
    case MVE_VQDMULL_qr_s32bh:
    case MVE_VQDMULL_qr_s32th:
    case MVE_VQDMULLs32bh:
    case MVE_VQDMULLs32th:
    case MVE_VQMOVNs32bh:
    case MVE_VQMOVNs32th:
    case MVE_VQMOVNu32bh:
    case MVE_VQMOVNu32th:
    case MVE_VQMOVUNs32bh:
    case MVE_VQMOVUNs32th:
    case MVE_VQNEGs32:
    case MVE_VQRDMLADHXs32:
    case MVE_VQRDMLADHs32:
    case MVE_VQRDMLAH_qrs32:
    case MVE_VQRDMLASH_qrs32:
    case MVE_VQRDMLSDHXs32:
    case MVE_VQRDMLSDHs32:
    case MVE_VQRDMULH_qr_s32:
    case MVE_VQRDMULHi32:
    case MVE_VQRSHL_by_vecs32:
    case MVE_VQRSHL_by_vecu32:
    case MVE_VQRSHL_qrs32:
    case MVE_VQRSHL_qru32:
    case MVE_VQRSHRNbhs32:
    case MVE_VQRSHRNbhu32:
    case MVE_VQRSHRNths32:
    case MVE_VQRSHRNthu32:
    case MVE_VQRSHRUNs32bh:
    case MVE_VQRSHRUNs32th:
    case MVE_VQSHLU_imms32:
    case MVE_VQSHL_by_vecs32:
    case MVE_VQSHL_by_vecu32:
    case MVE_VQSHL_qrs32:
    case MVE_VQSHL_qru32:
    case MVE_VQSHLimms32:
    case MVE_VQSHLimmu32:
    case MVE_VQSHRNbhs32:
    case MVE_VQSHRNbhu32:
    case MVE_VQSHRNths32:
    case MVE_VQSHRNthu32:
    case MVE_VQSHRUNs32bh:
    case MVE_VQSHRUNs32th:
    case MVE_VQSUB_qr_s32:
    case MVE_VQSUB_qr_u32:
    case MVE_VQSUBs32:
    case MVE_VQSUBu32:
    case MVE_VREV32_16:
    case MVE_VREV32_8:
    case MVE_VRHADDs32:
    case MVE_VRHADDu32:
    case MVE_VRINTf32A:
    case MVE_VRINTf32M:
    case MVE_VRINTf32N:
    case MVE_VRINTf32P:
    case MVE_VRINTf32X:
    case MVE_VRINTf32Z:
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
    case MVE_VRMULHs32:
    case MVE_VRMULHu32:
    case MVE_VRSHL_by_vecs32:
    case MVE_VRSHL_by_vecu32:
    case MVE_VRSHL_qrs32:
    case MVE_VRSHL_qru32:
    case MVE_VRSHRNi32bh:
    case MVE_VRSHRNi32th:
    case MVE_VRSHR_imms32:
    case MVE_VRSHR_immu32:
    case MVE_VSBC:
    case MVE_VSBCI:
    case MVE_VSHLC:
    case MVE_VSHLL_imms16bh:
    case MVE_VSHLL_imms16th:
    case MVE_VSHLL_immu16bh:
    case MVE_VSHLL_immu16th:
    case MVE_VSHLL_lws16bh:
    case MVE_VSHLL_lws16th:
    case MVE_VSHLL_lwu16bh:
    case MVE_VSHLL_lwu16th:
    case MVE_VSHL_by_vecs32:
    case MVE_VSHL_by_vecu32:
    case MVE_VSHL_immi32:
    case MVE_VSHL_qrs32:
    case MVE_VSHL_qru32:
    case MVE_VSHRNi32bh:
    case MVE_VSHRNi32th:
    case MVE_VSHR_imms32:
    case MVE_VSHR_immu32:
    case MVE_VSLIimm32:
    case MVE_VSRIimm32:
    case MVE_VST20_32:
    case MVE_VST20_32_wb:
    case MVE_VST21_32:
    case MVE_VST21_32_wb:
    case MVE_VST40_32:
    case MVE_VST40_32_wb:
    case MVE_VST41_32:
    case MVE_VST41_32_wb:
    case MVE_VST42_32:
    case MVE_VST42_32_wb:
    case MVE_VST43_32:
    case MVE_VST43_32_wb:
    case MVE_VSTRB32:
    case MVE_VSTRB32_post:
    case MVE_VSTRB32_pre:
    case MVE_VSTRB32_rq:
    case MVE_VSTRH32:
    case MVE_VSTRH32_post:
    case MVE_VSTRH32_pre:
    case MVE_VSTRH32_rq:
    case MVE_VSTRH32_rq_u:
    case MVE_VSTRW32_qi:
    case MVE_VSTRW32_qi_pre:
    case MVE_VSTRW32_rq:
    case MVE_VSTRW32_rq_u:
    case MVE_VSTRWU32:
    case MVE_VSTRWU32_post:
    case MVE_VSTRWU32_pre:
    case MVE_VSUB_qr_f32:
    case MVE_VSUB_qr_i32:
    case MVE_VSUBf32:
    case MVE_VSUBi32:
      return 2;
    case MVE_VCTP64:
    case MVE_VLDRDU64_qi:
    case MVE_VLDRDU64_qi_pre:
    case MVE_VLDRDU64_rq:
    case MVE_VLDRDU64_rq_u:
    case MVE_VMULLBs32:
    case MVE_VMULLBu32:
    case MVE_VMULLTs32:
    case MVE_VMULLTu32:
    case MVE_VMOVimmi64:
    case MVE_VREV64_16:
    case MVE_VREV64_32:
    case MVE_VREV64_8:
    case MVE_VSTRD64_qi:
    case MVE_VSTRD64_qi_pre:
    case MVE_VSTRD64_rq:
    case MVE_VSTRD64_rq_u:
      return 3;
    }
  };
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
    static_cast<LLVMTargetMachine*>(
      T->createTargetMachine(TT, "generic", "", Options, None, None,
                             CodeGenOpt::Default)));
  ARMSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                  std::string(TM->getTargetFeatureString()),
                  *static_cast<const ARMBaseTargetMachine *>(TM.get()), false);

  auto MII = TM->getMCInstrInfo();
  for (unsigned i = 0; i < ARM::INSTRUCTION_LIST_END; ++i) {
    uint64_t Flags = MII->get(i).TSFlags;
    if ((Flags & ARMII::DomainMask) != ARMII::DomainMVE)
      continue;
    int Size = (Flags & ARMII::VecSize) >> ARMII::VecSizeShift;
    ASSERT_EQ(MVEVecSize(i), Size)
              << MII->getName(i)
              << ": mismatched expectation for MVE vec size\n";
  }
}
