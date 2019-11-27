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
// - top/bottom operations.
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
    case MVE_VBICIZ0v4i32:
    case MVE_VBICIZ0v8i16:
    case MVE_VBICIZ16v4i32:
    case MVE_VBICIZ24v4i32:
    case MVE_VBICIZ8v4i32:
    case MVE_VBICIZ8v8i16:
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
    case MVE_VORRIZ0v4i32:
    case MVE_VORRIZ0v8i16:
    case MVE_VORRIZ16v4i32:
    case MVE_VORRIZ24v4i32:
    case MVE_VORRIZ8v4i32:
    case MVE_VORRIZ8v8i16:
    case MVE_VPNOT:
    case MVE_VPSEL:
    case MVE_VPST:	
    case MVE_VPTv16i8:
    case MVE_VPTv16i8r:
    case MVE_VPTv16s8:
    case MVE_VPTv16s8r:
    case MVE_VPTv16u8:	
    case MVE_VPTv16u8r:
    case MVE_VPTv4f32:
    case MVE_VPTv4f32r:
    case MVE_VPTv4i32:
    case MVE_VPTv4i32r:
    case MVE_VPTv4s32:
    case MVE_VPTv4s32r:
    case MVE_VPTv4u32:
    case MVE_VPTv4u32r:
    case MVE_VPTv8f16:
    case MVE_VPTv8f16r:
    case MVE_VPTv8i16:
    case MVE_VPTv8i16r:
    case MVE_VPTv8s16:
    case MVE_VPTv8s16r:
    case MVE_VPTv8u16:
    case MVE_VPTv8u16r:
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
    case MVE_VSLIimm16:
    case MVE_VSLIimm32:
    case MVE_VSLIimm8:
    case MVE_VSLIimms16:	
    case MVE_VSLIimms32:
    case MVE_VSLIimms8:
    case MVE_VSLIimmu16:
    case MVE_VSLIimmu32:
    case MVE_VSLIimmu8:
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
  ARMSubtarget ST(TM->getTargetTriple(), TM->getTargetCPU(),
                  TM->getTargetFeatureString(),
                  *static_cast<const ARMBaseTargetMachine*>(TM.get()), false);
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
