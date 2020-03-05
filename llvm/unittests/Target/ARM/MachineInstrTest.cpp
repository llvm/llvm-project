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
// descriptions. Currently we, conservatively, disallow:
// - cross beat carries.
// - narrowing of results.
// - complex operations.
// - horizontal operations.
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
    case MVE_SQRSHR:
    case MVE_SQSHL:
    case MVE_SRSHR:
    case MVE_UQRSHL:
    case MVE_UQSHL:
    case MVE_URSHR:
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
    case MVE_VMAXAs16:
    case MVE_VMAXAs32:
    case MVE_VMAXAs8:
    case MVE_VMAXs16:
    case MVE_VMAXs32:
    case MVE_VMAXs8:
    case MVE_VMAXu16:
    case MVE_VMAXu32:
    case MVE_VMAXu8:
    case MVE_VMINAs16:
    case MVE_VMINAs32:
    case MVE_VMINAs8:
    case MVE_VMINs16:
    case MVE_VMINs32:
    case MVE_VMINs8:
    case MVE_VMINu16:
    case MVE_VMINu32:
    case MVE_VMINu8:
    case MVE_VMLAS_qr_s16:
    case MVE_VMLAS_qr_s32:
    case MVE_VMLAS_qr_s8:
    case MVE_VMLAS_qr_u16:
    case MVE_VMLAS_qr_u32:
    case MVE_VMLAS_qr_u8:
    case MVE_VMLA_qr_s16:
    case MVE_VMLA_qr_s32:
    case MVE_VMLA_qr_s8:
    case MVE_VMLA_qr_u16:
    case MVE_VMLA_qr_u32:
    case MVE_VMLA_qr_u8:
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
    case MVE_VQDMULL_qr_s16bh:
    case MVE_VQDMULL_qr_s16th:
    case MVE_VQDMULL_qr_s32bh:
    case MVE_VQDMULL_qr_s32th:
    case MVE_VQDMULLs16bh:
    case MVE_VQDMULLs16th:
    case MVE_VQDMULLs32bh:
    case MVE_VQDMULLs32th:
    case MVE_VQNEGs16:
    case MVE_VQNEGs32:
    case MVE_VQNEGs8:
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
      return true;
    }
  };

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
  ARMSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                  std::string(TM->getTargetFeatureString()),
                  *static_cast<const ARMBaseTargetMachine *>(TM.get()), false);
  const ARMBaseInstrInfo *TII = ST.getInstrInfo();
  auto MII = TM->getMCInstrInfo();

  for (unsigned i = 0; i < ARM::INSTRUCTION_LIST_END; ++i) {
    const MCInstrDesc &Desc = TII->get(i);

    for (auto &Op : Desc.operands()) {
      // Only check instructions that access the MQPR regs.
      if ((Op.OperandType & MCOI::OPERAND_REGISTER) == 0 ||
          Op.RegClass != ARM::MQPRRegClassID)
        continue;

      uint64_t Flags = MII->get(i).TSFlags;
      bool Valid = (Flags & ARMII::ValidForTailPredication) != 0;
      ASSERT_EQ(IsValidTPOpcode(i), Valid)
                << MII->getName(i)
                << ": mismatched expectation for tail-predicated safety\n";
      break;
    }
  }
}

TEST(MachineInstr, HasSideEffects) {
  using namespace ARM;
  unsigned Opcodes[] = {
      // MVE Loads/Stores
      MVE_VLDRBS16,        MVE_VLDRBS16_post,   MVE_VLDRBS16_pre,
      MVE_VLDRBS16_rq,     MVE_VLDRBS32,        MVE_VLDRBS32_post,
      MVE_VLDRBS32_pre,    MVE_VLDRBS32_rq,     MVE_VLDRBU16,
      MVE_VLDRBU16_post,   MVE_VLDRBU16_pre,    MVE_VLDRBU16_rq,
      MVE_VLDRBU32,        MVE_VLDRBU32_post,   MVE_VLDRBU32_pre,
      MVE_VLDRBU32_rq,     MVE_VLDRBU8,         MVE_VLDRBU8_post,
      MVE_VLDRBU8_pre,     MVE_VLDRBU8_rq,      MVE_VLDRDU64_qi,
      MVE_VLDRDU64_qi_pre, MVE_VLDRDU64_rq,     MVE_VLDRDU64_rq_u,
      MVE_VLDRHS32,        MVE_VLDRHS32_post,   MVE_VLDRHS32_pre,
      MVE_VLDRHS32_rq,     MVE_VLDRHS32_rq_u,   MVE_VLDRHU16,
      MVE_VLDRHU16_post,   MVE_VLDRHU16_pre,    MVE_VLDRHU16_rq,
      MVE_VLDRHU16_rq_u,   MVE_VLDRHU32,        MVE_VLDRHU32_post,
      MVE_VLDRHU32_pre,    MVE_VLDRHU32_rq,     MVE_VLDRHU32_rq_u,
      MVE_VLDRWU32,        MVE_VLDRWU32_post,   MVE_VLDRWU32_pre,
      MVE_VLDRWU32_qi,     MVE_VLDRWU32_qi_pre, MVE_VLDRWU32_rq,
      MVE_VLDRWU32_rq_u,   MVE_VLD20_16,        MVE_VLD20_16_wb,
      MVE_VLD20_32,        MVE_VLD20_32_wb,     MVE_VLD20_8,
      MVE_VLD20_8_wb,      MVE_VLD21_16,        MVE_VLD21_16_wb,
      MVE_VLD21_32,        MVE_VLD21_32_wb,     MVE_VLD21_8,
      MVE_VLD21_8_wb,      MVE_VLD40_16,        MVE_VLD40_16_wb,
      MVE_VLD40_32,        MVE_VLD40_32_wb,     MVE_VLD40_8,
      MVE_VLD40_8_wb,      MVE_VLD41_16,        MVE_VLD41_16_wb,
      MVE_VLD41_32,        MVE_VLD41_32_wb,     MVE_VLD41_8,
      MVE_VLD41_8_wb,      MVE_VLD42_16,        MVE_VLD42_16_wb,
      MVE_VLD42_32,        MVE_VLD42_32_wb,     MVE_VLD42_8,
      MVE_VLD42_8_wb,      MVE_VLD43_16,        MVE_VLD43_16_wb,
      MVE_VLD43_32,        MVE_VLD43_32_wb,     MVE_VLD43_8,
      MVE_VLD43_8_wb,      MVE_VSTRB16,         MVE_VSTRB16_post,
      MVE_VSTRB16_pre,     MVE_VSTRB16_rq,      MVE_VSTRB32,
      MVE_VSTRB32_post,    MVE_VSTRB32_pre,     MVE_VSTRB32_rq,
      MVE_VSTRB8_rq,       MVE_VSTRBU8,         MVE_VSTRBU8_post,
      MVE_VSTRBU8_pre,     MVE_VSTRD64_qi,      MVE_VSTRD64_qi_pre,
      MVE_VSTRD64_rq,      MVE_VSTRD64_rq_u,    MVE_VSTRH16_rq,
      MVE_VSTRH16_rq_u,    MVE_VSTRH32,         MVE_VSTRH32_post,
      MVE_VSTRH32_pre,     MVE_VSTRH32_rq,      MVE_VSTRH32_rq_u,
      MVE_VSTRHU16,        MVE_VSTRHU16_post,   MVE_VSTRHU16_pre,
      MVE_VSTRW32_qi,      MVE_VSTRW32_qi_pre,  MVE_VSTRW32_rq,
      MVE_VSTRW32_rq_u,    MVE_VSTRWU32,        MVE_VSTRWU32_post,
      MVE_VSTRWU32_pre,    MVE_VST20_16,        MVE_VST20_16_wb,
      MVE_VST20_32,        MVE_VST20_32_wb,     MVE_VST20_8,
      MVE_VST20_8_wb,      MVE_VST21_16,        MVE_VST21_16_wb,
      MVE_VST21_32,        MVE_VST21_32_wb,     MVE_VST21_8,
      MVE_VST21_8_wb,      MVE_VST40_16,        MVE_VST40_16_wb,
      MVE_VST40_32,        MVE_VST40_32_wb,     MVE_VST40_8,
      MVE_VST40_8_wb,      MVE_VST41_16,        MVE_VST41_16_wb,
      MVE_VST41_32,        MVE_VST41_32_wb,     MVE_VST41_8,
      MVE_VST41_8_wb,      MVE_VST42_16,        MVE_VST42_16_wb,
      MVE_VST42_32,        MVE_VST42_32_wb,     MVE_VST42_8,
      MVE_VST42_8_wb,      MVE_VST43_16,        MVE_VST43_16_wb,
      MVE_VST43_32,        MVE_VST43_32_wb,     MVE_VST43_8,
      MVE_VST43_8_wb,
  };

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
      static_cast<LLVMTargetMachine *>(T->createTargetMachine(
          TT, "generic", "", Options, None, None, CodeGenOpt::Default)));
  ARMSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                  std::string(TM->getTargetFeatureString()),
                  *static_cast<const ARMBaseTargetMachine *>(TM.get()), false);
  const ARMBaseInstrInfo *TII = ST.getInstrInfo();
  auto MII = TM->getMCInstrInfo();

  for (unsigned Op : Opcodes) {
    const MCInstrDesc &Desc = TII->get(Op);
    ASSERT_FALSE(Desc.hasUnmodeledSideEffects())
        << MII->getName(Op) << " has unexpected side effects";
  }
}
