//===-- AMDGPUAsmPrinter.cpp - AMDGPU Assebly printer  --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
///
/// The AMDGPUAsmPrinter is used to print both assembly string and also binary
/// code.  When passed an MCAsmStreamer it prints assembly and when passed
/// an MCObjectStreamer it outputs binary code.
//
//===----------------------------------------------------------------------===//
//

#include "AMDGPUAsmPrinter.h"
#include "MCTargetDesc/AMDGPUTargetStreamer.h"
#include "InstPrinter/AMDGPUInstPrinter.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "AMDGPU.h"
#include "AMDKernelCodeT.h"
#include "AMDGPUSubtarget.h"
#include "R600Defines.h"
#include "R600MachineFunctionInfo.h"
#include "R600RegisterInfo.h"
#include "SIDefines.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetLoweringObjectFile.h"

using namespace llvm;

// TODO: This should get the default rounding mode from the kernel. We just set
// the default here, but this could change if the OpenCL rounding mode pragmas
// are used.
//
// The denormal mode here should match what is reported by the OpenCL runtime
// for the CL_FP_DENORM bit from CL_DEVICE_{HALF|SINGLE|DOUBLE}_FP_CONFIG, but
// can also be override to flush with the -cl-denorms-are-zero compiler flag.
//
// AMD OpenCL only sets flush none and reports CL_FP_DENORM for double
// precision, and leaves single precision to flush all and does not report
// CL_FP_DENORM for CL_DEVICE_SINGLE_FP_CONFIG. Mesa's OpenCL currently reports
// CL_FP_DENORM for both.
//
// FIXME: It seems some instructions do not support single precision denormals
// regardless of the mode (exp_*_f32, rcp_*_f32, rsq_*_f32, rsq_*f32, sqrt_f32,
// and sin_f32, cos_f32 on most parts).

// We want to use these instructions, and using fp32 denormals also causes
// instructions to run at the double precision rate for the device so it's
// probably best to just report no single precision denormals.
static uint32_t getFPMode(const MachineFunction &F) {
  const AMDGPUSubtarget& ST = F.getSubtarget<AMDGPUSubtarget>();
  // TODO: Is there any real use for the flush in only / flush out only modes?

  uint32_t FP32Denormals =
    ST.hasFP32Denormals() ? FP_DENORM_FLUSH_NONE : FP_DENORM_FLUSH_IN_FLUSH_OUT;

  uint32_t FP64Denormals =
    ST.hasFP64Denormals() ? FP_DENORM_FLUSH_NONE : FP_DENORM_FLUSH_IN_FLUSH_OUT;

  return FP_ROUND_MODE_SP(FP_ROUND_ROUND_TO_NEAREST) |
         FP_ROUND_MODE_DP(FP_ROUND_ROUND_TO_NEAREST) |
         FP_DENORM_MODE_SP(FP32Denormals) |
         FP_DENORM_MODE_DP(FP64Denormals);
}

static AsmPrinter *
createAMDGPUAsmPrinterPass(TargetMachine &tm,
                           std::unique_ptr<MCStreamer> &&Streamer) {
  return new AMDGPUAsmPrinter(tm, std::move(Streamer));
}

extern "C" void LLVMInitializeAMDGPUAsmPrinter() {
  TargetRegistry::RegisterAsmPrinter(TheAMDGPUTarget, createAMDGPUAsmPrinterPass);
  TargetRegistry::RegisterAsmPrinter(TheGCNTarget, createAMDGPUAsmPrinterPass);
}

AMDGPUAsmPrinter::AMDGPUAsmPrinter(TargetMachine &TM,
                                   std::unique_ptr<MCStreamer> Streamer)
    : AsmPrinter(TM, std::move(Streamer)) {}

void AMDGPUAsmPrinter::EmitFunctionBodyStart() {
  const AMDGPUSubtarget &STM = MF->getSubtarget<AMDGPUSubtarget>();
  SIProgramInfo KernelInfo;
  if (STM.isAmdHsaOS()) {
    getSIProgramInfo(KernelInfo, *MF);
    EmitAmdKernelCodeT(*MF, KernelInfo);
  }
}

void AMDGPUAsmPrinter::EmitFunctionEntryLabel() {
  const SIMachineFunctionInfo *MFI = MF->getInfo<SIMachineFunctionInfo>();
  const AMDGPUSubtarget &STM = MF->getSubtarget<AMDGPUSubtarget>();
  if (MFI->isKernel() && STM.isAmdHsaOS()) {
    AMDGPUTargetStreamer *TS =
        static_cast<AMDGPUTargetStreamer *>(OutStreamer->getTargetStreamer());
    TS->EmitAMDGPUSymbolType(CurrentFnSym->getName(),
                             ELF::STT_AMDGPU_HSA_KERNEL);
  }

  AsmPrinter::EmitFunctionEntryLabel();
}

static bool isModuleLinkage(const GlobalValue *GV) {
  switch (GV->getLinkage()) {
  case GlobalValue::InternalLinkage:
  case GlobalValue::CommonLinkage:
   return true;
  case GlobalValue::ExternalLinkage:
   return false;
  default: llvm_unreachable("unknown linkage type");
  }
}

void AMDGPUAsmPrinter::EmitGlobalVariable(const GlobalVariable *GV) {

  if (TM.getTargetTriple().getOS() != Triple::AMDHSA) {
    AsmPrinter::EmitGlobalVariable(GV);
    return;
  }

  if (GV->isDeclaration() || GV->getLinkage() == GlobalValue::PrivateLinkage) {
    AsmPrinter::EmitGlobalVariable(GV);
    return;
  }

  // Group segment variables aren't emitted in HSA.
  if (AMDGPU::isGroupSegment(GV))
    return;

  AMDGPUTargetStreamer *TS =
      static_cast<AMDGPUTargetStreamer *>(OutStreamer->getTargetStreamer());
  if (isModuleLinkage(GV)) {
    TS->EmitAMDGPUHsaModuleScopeGlobal(GV->getName());
  } else {
    TS->EmitAMDGPUHsaProgramScopeGlobal(GV->getName());
  }

  const DataLayout &DL = getDataLayout();
  OutStreamer->PushSection();
  OutStreamer->SwitchSection(
      getObjFileLowering().SectionForGlobal(GV, *Mang, TM));
  MCSymbol *GVSym = getSymbol(GV);
  const Constant *C = GV->getInitializer();
  OutStreamer->EmitLabel(GVSym);
  EmitGlobalConstant(DL, C);
  OutStreamer->PopSection();
}

bool AMDGPUAsmPrinter::runOnMachineFunction(MachineFunction &MF) {

  // The starting address of all shader programs must be 256 bytes aligned.
  MF.setAlignment(8);

  SetupMachineFunction(MF);

  MCContext &Context = getObjFileLowering().getContext();
  MCSectionELF *ConfigSection =
      Context.getELFSection(".AMDGPU.config", ELF::SHT_PROGBITS, 0);
  OutStreamer->SwitchSection(ConfigSection);

  const AMDGPUSubtarget &STM = MF.getSubtarget<AMDGPUSubtarget>();
  SIProgramInfo KernelInfo;
  if (STM.getGeneration() >= AMDGPUSubtarget::SOUTHERN_ISLANDS) {
    getSIProgramInfo(KernelInfo, MF);
    if (!STM.isAmdHsaOS()) {
      EmitProgramInfoSI(MF, KernelInfo);
    }
    // Emit directives
    AMDGPUTargetStreamer *TS =
        static_cast<AMDGPUTargetStreamer *>(OutStreamer->getTargetStreamer());
    TS->EmitDirectiveHSACodeObjectVersion(1, 0);
    AMDGPU::IsaVersion ISA = STM.getIsaVersion();
    TS->EmitDirectiveHSACodeObjectISA(ISA.Major, ISA.Minor, ISA.Stepping,
                                      "AMD", "AMDGPU");
  } else {
    EmitProgramInfoR600(MF);
  }

  DisasmLines.clear();
  HexLines.clear();
  DisasmLineMaxLen = 0;

  EmitFunctionBody();

  if (isVerbose()) {
    MCSectionELF *CommentSection =
        Context.getELFSection(".AMDGPU.csdata", ELF::SHT_PROGBITS, 0);
    OutStreamer->SwitchSection(CommentSection);

    if (STM.getGeneration() >= AMDGPUSubtarget::SOUTHERN_ISLANDS) {
      OutStreamer->emitRawComment(" Kernel info:", false);
      OutStreamer->emitRawComment(" codeLenInByte = " + Twine(KernelInfo.CodeLen),
                                  false);
      OutStreamer->emitRawComment(" NumSgprs: " + Twine(KernelInfo.NumSGPR),
                                  false);
      OutStreamer->emitRawComment(" NumVgprs: " + Twine(KernelInfo.NumVGPR),
                                  false);
      OutStreamer->emitRawComment(" FloatMode: " + Twine(KernelInfo.FloatMode),
                                  false);
      OutStreamer->emitRawComment(" IeeeMode: " + Twine(KernelInfo.IEEEMode),
                                  false);
      OutStreamer->emitRawComment(" ScratchSize: " + Twine(KernelInfo.ScratchSize),
                                  false);

      OutStreamer->emitRawComment(" COMPUTE_PGM_RSRC2:USER_SGPR: " +
                                  Twine(G_00B84C_USER_SGPR(KernelInfo.ComputePGMRSrc2)),
                                  false);
      OutStreamer->emitRawComment(" COMPUTE_PGM_RSRC2:TGID_X_EN: " +
                                  Twine(G_00B84C_TGID_X_EN(KernelInfo.ComputePGMRSrc2)),
                                  false);
      OutStreamer->emitRawComment(" COMPUTE_PGM_RSRC2:TGID_Y_EN: " +
                                  Twine(G_00B84C_TGID_Y_EN(KernelInfo.ComputePGMRSrc2)),
                                  false);
      OutStreamer->emitRawComment(" COMPUTE_PGM_RSRC2:TGID_Z_EN: " +
                                  Twine(G_00B84C_TGID_Z_EN(KernelInfo.ComputePGMRSrc2)),
                                  false);
      OutStreamer->emitRawComment(" COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: " +
                                  Twine(G_00B84C_TIDIG_COMP_CNT(KernelInfo.ComputePGMRSrc2)),
                                  false);

    } else {
      R600MachineFunctionInfo *MFI = MF.getInfo<R600MachineFunctionInfo>();
      OutStreamer->emitRawComment(
        Twine("SQ_PGM_RESOURCES:STACK_SIZE = " + Twine(MFI->StackSize)));
    }
  }

  if (STM.dumpCode()) {

    OutStreamer->SwitchSection(
        Context.getELFSection(".AMDGPU.disasm", ELF::SHT_NOTE, 0));

    for (size_t i = 0; i < DisasmLines.size(); ++i) {
      std::string Comment(DisasmLineMaxLen - DisasmLines[i].size(), ' ');
      Comment += " ; " + HexLines[i] + "\n";

      OutStreamer->EmitBytes(StringRef(DisasmLines[i]));
      OutStreamer->EmitBytes(StringRef(Comment));
    }
  }

  return false;
}

void AMDGPUAsmPrinter::EmitProgramInfoR600(const MachineFunction &MF) {
  unsigned MaxGPR = 0;
  bool killPixel = false;
  const AMDGPUSubtarget &STM = MF.getSubtarget<AMDGPUSubtarget>();
  const R600RegisterInfo *RI =
      static_cast<const R600RegisterInfo *>(STM.getRegisterInfo());
  const R600MachineFunctionInfo *MFI = MF.getInfo<R600MachineFunctionInfo>();

  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      if (MI.getOpcode() == AMDGPU::KILLGT)
        killPixel = true;
      unsigned numOperands = MI.getNumOperands();
      for (unsigned op_idx = 0; op_idx < numOperands; op_idx++) {
        const MachineOperand &MO = MI.getOperand(op_idx);
        if (!MO.isReg())
          continue;
        unsigned HWReg = RI->getEncodingValue(MO.getReg()) & 0xff;

        // Register with value > 127 aren't GPR
        if (HWReg > 127)
          continue;
        MaxGPR = std::max(MaxGPR, HWReg);
      }
    }
  }

  unsigned RsrcReg;
  if (STM.getGeneration() >= AMDGPUSubtarget::EVERGREEN) {
    // Evergreen / Northern Islands
    switch (MFI->getShaderType()) {
    default: // Fall through
    case ShaderType::COMPUTE:  RsrcReg = R_0288D4_SQ_PGM_RESOURCES_LS; break;
    case ShaderType::GEOMETRY: RsrcReg = R_028878_SQ_PGM_RESOURCES_GS; break;
    case ShaderType::PIXEL:    RsrcReg = R_028844_SQ_PGM_RESOURCES_PS; break;
    case ShaderType::VERTEX:   RsrcReg = R_028860_SQ_PGM_RESOURCES_VS; break;
    }
  } else {
    // R600 / R700
    switch (MFI->getShaderType()) {
    default: // Fall through
    case ShaderType::GEOMETRY: // Fall through
    case ShaderType::COMPUTE:  // Fall through
    case ShaderType::VERTEX:   RsrcReg = R_028868_SQ_PGM_RESOURCES_VS; break;
    case ShaderType::PIXEL:    RsrcReg = R_028850_SQ_PGM_RESOURCES_PS; break;
    }
  }

  OutStreamer->EmitIntValue(RsrcReg, 4);
  OutStreamer->EmitIntValue(S_NUM_GPRS(MaxGPR + 1) |
                           S_STACK_SIZE(MFI->StackSize), 4);
  OutStreamer->EmitIntValue(R_02880C_DB_SHADER_CONTROL, 4);
  OutStreamer->EmitIntValue(S_02880C_KILL_ENABLE(killPixel), 4);

  if (MFI->getShaderType() == ShaderType::COMPUTE) {
    OutStreamer->EmitIntValue(R_0288E8_SQ_LDS_ALLOC, 4);
    OutStreamer->EmitIntValue(RoundUpToAlignment(MFI->LDSSize, 4) >> 2, 4);
  }
}

void AMDGPUAsmPrinter::getSIProgramInfo(SIProgramInfo &ProgInfo,
                                        const MachineFunction &MF) const {
  const AMDGPUSubtarget &STM = MF.getSubtarget<AMDGPUSubtarget>();
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  uint64_t CodeSize = 0;
  unsigned MaxSGPR = 0;
  unsigned MaxVGPR = 0;
  bool VCCUsed = false;
  bool FlatUsed = false;
  const SIRegisterInfo *RI =
      static_cast<const SIRegisterInfo *>(STM.getRegisterInfo());

  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      // TODO: CodeSize should account for multiple functions.

      // TODO: Should we count size of debug info?
      if (MI.isDebugValue())
        continue;

      // FIXME: This is reporting 0 for many instructions.
      CodeSize += MI.getDesc().Size;

      unsigned numOperands = MI.getNumOperands();
      for (unsigned op_idx = 0; op_idx < numOperands; op_idx++) {
        const MachineOperand &MO = MI.getOperand(op_idx);
        unsigned width = 0;
        bool isSGPR = false;

        if (!MO.isReg())
          continue;

        unsigned reg = MO.getReg();
        switch (reg) {
        case AMDGPU::EXEC:
        case AMDGPU::SCC:
        case AMDGPU::M0:
          continue;

        case AMDGPU::VCC:
        case AMDGPU::VCC_LO:
        case AMDGPU::VCC_HI:
          VCCUsed = true;
          continue;

        case AMDGPU::FLAT_SCR:
        case AMDGPU::FLAT_SCR_LO:
        case AMDGPU::FLAT_SCR_HI:
          FlatUsed = true;
          continue;

        default:
          break;
        }

        if (AMDGPU::SReg_32RegClass.contains(reg)) {
          isSGPR = true;
          width = 1;
        } else if (AMDGPU::VGPR_32RegClass.contains(reg)) {
          isSGPR = false;
          width = 1;
        } else if (AMDGPU::SReg_64RegClass.contains(reg)) {
          isSGPR = true;
          width = 2;
        } else if (AMDGPU::VReg_64RegClass.contains(reg)) {
          isSGPR = false;
          width = 2;
        } else if (AMDGPU::VReg_96RegClass.contains(reg)) {
          isSGPR = false;
          width = 3;
        } else if (AMDGPU::SReg_128RegClass.contains(reg)) {
          isSGPR = true;
          width = 4;
        } else if (AMDGPU::VReg_128RegClass.contains(reg)) {
          isSGPR = false;
          width = 4;
        } else if (AMDGPU::SReg_256RegClass.contains(reg)) {
          isSGPR = true;
          width = 8;
        } else if (AMDGPU::VReg_256RegClass.contains(reg)) {
          isSGPR = false;
          width = 8;
        } else if (AMDGPU::SReg_512RegClass.contains(reg)) {
          isSGPR = true;
          width = 16;
        } else if (AMDGPU::VReg_512RegClass.contains(reg)) {
          isSGPR = false;
          width = 16;
        } else {
          llvm_unreachable("Unknown register class");
        }
        unsigned hwReg = RI->getEncodingValue(reg) & 0xff;
        unsigned maxUsed = hwReg + width - 1;
        if (isSGPR) {
          MaxSGPR = maxUsed > MaxSGPR ? maxUsed : MaxSGPR;
        } else {
          MaxVGPR = maxUsed > MaxVGPR ? maxUsed : MaxVGPR;
        }
      }
    }
  }

  if (VCCUsed || FlatUsed || STM.isXNACKEnabled()) {
    MaxSGPR += 2;

    if (FlatUsed)
      MaxSGPR += 2;

    if (STM.isXNACKEnabled())
      MaxSGPR += 2;
  }

  // We found the maximum register index. They start at 0, so add one to get the
  // number of registers.
  ProgInfo.NumVGPR = MaxVGPR + 1;
  ProgInfo.NumSGPR = MaxSGPR + 1;

  if (STM.hasSGPRInitBug()) {
    if (ProgInfo.NumSGPR > AMDGPUSubtarget::FIXED_SGPR_COUNT_FOR_INIT_BUG) {
      LLVMContext &Ctx = MF.getFunction()->getContext();
      Ctx.emitError("too many SGPRs used with the SGPR init bug");
    }

    ProgInfo.NumSGPR = AMDGPUSubtarget::FIXED_SGPR_COUNT_FOR_INIT_BUG;
  }

  if (MFI->NumUserSGPRs > STM.getMaxNumUserSGPRs()) {
    LLVMContext &Ctx = MF.getFunction()->getContext();
    Ctx.emitError("too many user SGPRs used");
  }

  ProgInfo.VGPRBlocks = (ProgInfo.NumVGPR - 1) / 4;
  ProgInfo.SGPRBlocks = (ProgInfo.NumSGPR - 1) / 8;
  // Set the value to initialize FP_ROUND and FP_DENORM parts of the mode
  // register.
  ProgInfo.FloatMode = getFPMode(MF);

  // XXX: Not quite sure what this does, but sc seems to unset this.
  ProgInfo.IEEEMode = 0;

  // Do not clamp NAN to 0.
  ProgInfo.DX10Clamp = 0;

  const MachineFrameInfo *FrameInfo = MF.getFrameInfo();
  ProgInfo.ScratchSize = FrameInfo->estimateStackSize(MF);

  ProgInfo.FlatUsed = FlatUsed;
  ProgInfo.VCCUsed = VCCUsed;
  ProgInfo.CodeLen = CodeSize;

  unsigned LDSAlignShift;
  if (STM.getGeneration() < AMDGPUSubtarget::SEA_ISLANDS) {
    // LDS is allocated in 64 dword blocks.
    LDSAlignShift = 8;
  } else {
    // LDS is allocated in 128 dword blocks.
    LDSAlignShift = 9;
  }

  unsigned LDSSpillSize = MFI->LDSWaveSpillSize *
                          MFI->getMaximumWorkGroupSize(MF);

  ProgInfo.LDSSize = MFI->LDSSize + LDSSpillSize;
  ProgInfo.LDSBlocks =
     RoundUpToAlignment(ProgInfo.LDSSize, 1 << LDSAlignShift) >> LDSAlignShift;

  // Scratch is allocated in 256 dword blocks.
  unsigned ScratchAlignShift = 10;
  // We need to program the hardware with the amount of scratch memory that
  // is used by the entire wave.  ProgInfo.ScratchSize is the amount of
  // scratch memory used per thread.
  ProgInfo.ScratchBlocks =
    RoundUpToAlignment(ProgInfo.ScratchSize * STM.getWavefrontSize(),
                       1 << ScratchAlignShift) >> ScratchAlignShift;

  ProgInfo.ComputePGMRSrc1 =
      S_00B848_VGPRS(ProgInfo.VGPRBlocks) |
      S_00B848_SGPRS(ProgInfo.SGPRBlocks) |
      S_00B848_PRIORITY(ProgInfo.Priority) |
      S_00B848_FLOAT_MODE(ProgInfo.FloatMode) |
      S_00B848_PRIV(ProgInfo.Priv) |
      S_00B848_DX10_CLAMP(ProgInfo.DX10Clamp) |
      S_00B848_DEBUG_MODE(ProgInfo.DebugMode) |
      S_00B848_IEEE_MODE(ProgInfo.IEEEMode);

  // 0 = X, 1 = XY, 2 = XYZ
  unsigned TIDIGCompCnt = 0;
  if (MFI->hasWorkItemIDZ())
    TIDIGCompCnt = 2;
  else if (MFI->hasWorkItemIDY())
    TIDIGCompCnt = 1;

  ProgInfo.ComputePGMRSrc2 =
      S_00B84C_SCRATCH_EN(ProgInfo.ScratchBlocks > 0) |
      S_00B84C_USER_SGPR(MFI->getNumUserSGPRs()) |
      S_00B84C_TGID_X_EN(MFI->hasWorkGroupIDX()) |
      S_00B84C_TGID_Y_EN(MFI->hasWorkGroupIDY()) |
      S_00B84C_TGID_Z_EN(MFI->hasWorkGroupIDZ()) |
      S_00B84C_TG_SIZE_EN(MFI->hasWorkGroupInfo()) |
      S_00B84C_TIDIG_COMP_CNT(TIDIGCompCnt) |
      S_00B84C_EXCP_EN_MSB(0) |
      S_00B84C_LDS_SIZE(ProgInfo.LDSBlocks) |
      S_00B84C_EXCP_EN(0);
}

static unsigned getRsrcReg(unsigned ShaderType) {
  switch (ShaderType) {
  default: // Fall through
  case ShaderType::COMPUTE:  return R_00B848_COMPUTE_PGM_RSRC1;
  case ShaderType::GEOMETRY: return R_00B228_SPI_SHADER_PGM_RSRC1_GS;
  case ShaderType::PIXEL:    return R_00B028_SPI_SHADER_PGM_RSRC1_PS;
  case ShaderType::VERTEX:   return R_00B128_SPI_SHADER_PGM_RSRC1_VS;
  }
}

void AMDGPUAsmPrinter::EmitProgramInfoSI(const MachineFunction &MF,
                                         const SIProgramInfo &KernelInfo) {
  const AMDGPUSubtarget &STM = MF.getSubtarget<AMDGPUSubtarget>();
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  unsigned RsrcReg = getRsrcReg(MFI->getShaderType());

  if (MFI->getShaderType() == ShaderType::COMPUTE) {
    OutStreamer->EmitIntValue(R_00B848_COMPUTE_PGM_RSRC1, 4);

    OutStreamer->EmitIntValue(KernelInfo.ComputePGMRSrc1, 4);

    OutStreamer->EmitIntValue(R_00B84C_COMPUTE_PGM_RSRC2, 4);
    OutStreamer->EmitIntValue(KernelInfo.ComputePGMRSrc2, 4);

    OutStreamer->EmitIntValue(R_00B860_COMPUTE_TMPRING_SIZE, 4);
    OutStreamer->EmitIntValue(S_00B860_WAVESIZE(KernelInfo.ScratchBlocks), 4);

    // TODO: Should probably note flat usage somewhere. SC emits a "FlatPtr32 =
    // 0" comment but I don't see a corresponding field in the register spec.
  } else {
    OutStreamer->EmitIntValue(RsrcReg, 4);
    OutStreamer->EmitIntValue(S_00B028_VGPRS(KernelInfo.VGPRBlocks) |
                              S_00B028_SGPRS(KernelInfo.SGPRBlocks), 4);
    if (STM.isVGPRSpillingEnabled(MFI)) {
      OutStreamer->EmitIntValue(R_0286E8_SPI_TMPRING_SIZE, 4);
      OutStreamer->EmitIntValue(S_0286E8_WAVESIZE(KernelInfo.ScratchBlocks), 4);
    }
  }

  if (MFI->getShaderType() == ShaderType::PIXEL) {
    OutStreamer->EmitIntValue(R_00B02C_SPI_SHADER_PGM_RSRC2_PS, 4);
    OutStreamer->EmitIntValue(S_00B02C_EXTRA_LDS_SIZE(KernelInfo.LDSBlocks), 4);
    OutStreamer->EmitIntValue(R_0286CC_SPI_PS_INPUT_ENA, 4);
    OutStreamer->EmitIntValue(MFI->PSInputAddr, 4);
  }
}

void AMDGPUAsmPrinter::EmitAmdKernelCodeT(const MachineFunction &MF,
                                         const SIProgramInfo &KernelInfo) const {
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  const AMDGPUSubtarget &STM = MF.getSubtarget<AMDGPUSubtarget>();
  amd_kernel_code_t header;

  AMDGPU::initDefaultAMDKernelCodeT(header, STM.getFeatureBits());

  header.compute_pgm_resource_registers =
      KernelInfo.ComputePGMRSrc1 |
      (KernelInfo.ComputePGMRSrc2 << 32);
  header.code_properties = AMD_CODE_PROPERTY_IS_PTR64;

  if (MFI->hasPrivateSegmentBuffer()) {
    header.code_properties |=
      AMD_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER;
  }

  if (MFI->hasDispatchPtr())
    header.code_properties |= AMD_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR;

  if (MFI->hasQueuePtr())
    header.code_properties |= AMD_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR;

  if (MFI->hasKernargSegmentPtr())
    header.code_properties |= AMD_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR;

  if (MFI->hasDispatchID())
    header.code_properties |= AMD_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID;

  if (MFI->hasFlatScratchInit())
    header.code_properties |= AMD_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT;

  // TODO: Private segment size

  if (MFI->hasGridWorkgroupCountX()) {
    header.code_properties |=
      AMD_CODE_PROPERTY_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X;
  }

  if (MFI->hasGridWorkgroupCountY()) {
    header.code_properties |=
      AMD_CODE_PROPERTY_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y;
  }

  if (MFI->hasGridWorkgroupCountZ()) {
    header.code_properties |=
      AMD_CODE_PROPERTY_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z;
  }

  if (MFI->hasDispatchPtr())
    header.code_properties |= AMD_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR;

  if (STM.isXNACKEnabled())
    header.code_properties |= AMD_CODE_PROPERTY_IS_XNACK_SUPPORTED;

  header.kernarg_segment_byte_size = MFI->ABIArgOffset;
  header.wavefront_sgpr_count = KernelInfo.NumSGPR;
  header.workitem_vgpr_count = KernelInfo.NumVGPR;
  header.workitem_private_segment_byte_size = KernelInfo.ScratchSize;
  header.workgroup_group_segment_byte_size = KernelInfo.LDSSize;

  AMDGPUTargetStreamer *TS =
      static_cast<AMDGPUTargetStreamer *>(OutStreamer->getTargetStreamer());
  TS->EmitAMDKernelCodeT(header);
}

bool AMDGPUAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                       unsigned AsmVariant,
                                       const char *ExtraCode, raw_ostream &O) {
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0)
      return true; // Unknown modifier.

    switch (ExtraCode[0]) {
    default:
      // See if this is a generic print operand
      return AsmPrinter::PrintAsmOperand(MI, OpNo, AsmVariant, ExtraCode, O);
    case 'r':
      break;
    }
  }

  AMDGPUInstPrinter::printRegOperand(MI->getOperand(OpNo).getReg(), O,
                   *TM.getSubtargetImpl(*MF->getFunction())->getRegisterInfo());
  return false;
}
