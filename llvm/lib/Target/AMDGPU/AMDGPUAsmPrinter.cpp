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
#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "AMDGPUTargetMachine.h"
#include "InstPrinter/AMDGPUInstPrinter.h"
#include "MCTargetDesc/AMDGPUTargetStreamer.h"
#include "R600Defines.h"
#include "R600MachineFunctionInfo.h"
#include "R600RegisterInfo.h"
#include "SIDefines.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/TargetLoweringObjectFile.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/AMDGPUMetadata.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;
using namespace llvm::AMDGPU;

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
  const SISubtarget& ST = F.getSubtarget<SISubtarget>();
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
  TargetRegistry::RegisterAsmPrinter(getTheAMDGPUTarget(),
                                     createAMDGPUAsmPrinterPass);
  TargetRegistry::RegisterAsmPrinter(getTheGCNTarget(),
                                     createAMDGPUAsmPrinterPass);
}

AMDGPUAsmPrinter::AMDGPUAsmPrinter(TargetMachine &TM,
                                   std::unique_ptr<MCStreamer> Streamer)
  : AsmPrinter(TM, std::move(Streamer)) {
    AMDGPUASI = static_cast<AMDGPUTargetMachine*>(&TM)->getAMDGPUAS();
  }

StringRef AMDGPUAsmPrinter::getPassName() const {
  return "AMDGPU Assembly Printer";
}

const MCSubtargetInfo* AMDGPUAsmPrinter::getSTI() const {
  return TM.getMCSubtargetInfo();
}

AMDGPUTargetStreamer* AMDGPUAsmPrinter::getTargetStreamer() const {
  if (!OutStreamer)
    return nullptr;
  return static_cast<AMDGPUTargetStreamer*>(OutStreamer->getTargetStreamer());
}

void AMDGPUAsmPrinter::EmitStartOfAsmFile(Module &M) {
  if (TM.getTargetTriple().getArch() != Triple::amdgcn)
    return;

  if (TM.getTargetTriple().getOS() != Triple::AMDHSA &&
      TM.getTargetTriple().getOS() != Triple::AMDPAL)
    return;

  if (TM.getTargetTriple().getOS() == Triple::AMDHSA)
    HSAMetadataStream.begin(M);

  if (TM.getTargetTriple().getOS() == Triple::AMDPAL)
    readPALMetadata(M);

  // Deprecated notes are not emitted for code object v3.
  if (IsaInfo::hasCodeObjectV3(getSTI()->getFeatureBits()))
    return;

  // HSA emits NT_AMDGPU_HSA_CODE_OBJECT_VERSION for code objects v2.
  if (TM.getTargetTriple().getOS() == Triple::AMDHSA)
    getTargetStreamer()->EmitDirectiveHSACodeObjectVersion(2, 1);

  // HSA and PAL emit NT_AMDGPU_HSA_ISA for code objects v2.
  IsaInfo::IsaVersion ISA = IsaInfo::getIsaVersion(getSTI()->getFeatureBits());
  getTargetStreamer()->EmitDirectiveHSACodeObjectISA(
      ISA.Major, ISA.Minor, ISA.Stepping, "AMD", "AMDGPU");
}

void AMDGPUAsmPrinter::EmitEndOfAsmFile(Module &M) {
  if (TM.getTargetTriple().getArch() != Triple::amdgcn)
    return;

  // Following code requires TargetStreamer to be present.
  if (!getTargetStreamer())
    return;

  // Emit ISA Version (NT_AMD_AMDGPU_ISA).
  std::string ISAVersionString;
  raw_string_ostream ISAVersionStream(ISAVersionString);
  IsaInfo::streamIsaVersion(getSTI(), ISAVersionStream);
  getTargetStreamer()->EmitISAVersion(ISAVersionStream.str());

  // Emit HSA Metadata (NT_AMD_AMDGPU_HSA_METADATA).
  if (TM.getTargetTriple().getOS() == Triple::AMDHSA) {
    HSAMetadataStream.end();
    getTargetStreamer()->EmitHSAMetadata(HSAMetadataStream.getHSAMetadata());
  }

  // Emit PAL Metadata (NT_AMD_AMDGPU_PAL_METADATA).
  if (TM.getTargetTriple().getOS() == Triple::AMDPAL) {
    // Copy the PAL metadata from the map where we collected it into a vector,
    // then write it as a .note.
    PALMD::Metadata PALMetadataVector;
    for (auto i : PALMetadataMap) {
      PALMetadataVector.push_back(i.first);
      PALMetadataVector.push_back(i.second);
    }
    getTargetStreamer()->EmitPALMetadata(PALMetadataVector);
  }
}

bool AMDGPUAsmPrinter::isBlockOnlyReachableByFallthrough(
  const MachineBasicBlock *MBB) const {
  if (!AsmPrinter::isBlockOnlyReachableByFallthrough(MBB))
    return false;

  if (MBB->empty())
    return true;

  // If this is a block implementing a long branch, an expression relative to
  // the start of the block is needed.  to the start of the block.
  // XXX - Is there a smarter way to check this?
  return (MBB->back().getOpcode() != AMDGPU::S_SETPC_B64);
}

void AMDGPUAsmPrinter::EmitFunctionBodyStart() {
  const AMDGPUMachineFunction *MFI = MF->getInfo<AMDGPUMachineFunction>();
  if (!MFI->isEntryFunction())
    return;

  const AMDGPUSubtarget &STM = MF->getSubtarget<AMDGPUSubtarget>();
  amd_kernel_code_t KernelCode;
  if (STM.isAmdCodeObjectV2(*MF)) {
    getAmdKernelCode(KernelCode, CurrentProgramInfo, *MF);

    OutStreamer->SwitchSection(getObjFileLowering().getTextSection());
    getTargetStreamer()->EmitAMDKernelCodeT(KernelCode);
  }

  if (TM.getTargetTriple().getOS() != Triple::AMDHSA)
    return;

  HSAMetadataStream.emitKernel(MF->getFunction(),
                               getHSACodeProps(*MF, CurrentProgramInfo),
                               getHSADebugProps(*MF, CurrentProgramInfo));
}

void AMDGPUAsmPrinter::EmitFunctionEntryLabel() {
  const SIMachineFunctionInfo *MFI = MF->getInfo<SIMachineFunctionInfo>();
  const AMDGPUSubtarget &STM = MF->getSubtarget<AMDGPUSubtarget>();
  if (MFI->isEntryFunction() && STM.isAmdCodeObjectV2(*MF)) {
    SmallString<128> SymbolName;
    getNameWithPrefix(SymbolName, &MF->getFunction()),
    getTargetStreamer()->EmitAMDGPUSymbolType(
        SymbolName, ELF::STT_AMDGPU_HSA_KERNEL);
  }
  const AMDGPUSubtarget &STI = MF->getSubtarget<AMDGPUSubtarget>();
  if (STI.dumpCode()) {
    // Disassemble function name label to text.
    DisasmLines.push_back(MF->getName().str() + ":");
    DisasmLineMaxLen = std::max(DisasmLineMaxLen, DisasmLines.back().size());
    HexLines.push_back("");
  }

  AsmPrinter::EmitFunctionEntryLabel();
}

void AMDGPUAsmPrinter::EmitBasicBlockStart(const MachineBasicBlock &MBB) const {
  const AMDGPUSubtarget &STI = MBB.getParent()->getSubtarget<AMDGPUSubtarget>();
  if (STI.dumpCode() && !isBlockOnlyReachableByFallthrough(&MBB)) {
    // Write a line for the basic block label if it is not only fallthrough.
    DisasmLines.push_back(
        (Twine("BB") + Twine(getFunctionNumber())
         + "_" + Twine(MBB.getNumber()) + ":").str());
    DisasmLineMaxLen = std::max(DisasmLineMaxLen, DisasmLines.back().size());
    HexLines.push_back("");
  }
  AsmPrinter::EmitBasicBlockStart(MBB);
}

void AMDGPUAsmPrinter::EmitGlobalVariable(const GlobalVariable *GV) {

  // Group segment variables aren't emitted in HSA.
  if (AMDGPU::isGroupSegment(GV))
    return;

  AsmPrinter::EmitGlobalVariable(GV);
}

bool AMDGPUAsmPrinter::doFinalization(Module &M) {
  CallGraphResourceInfo.clear();
  return AsmPrinter::doFinalization(M);
}

// For the amdpal OS type, read the amdgpu.pal.metadata supplied by the
// frontend into our PALMetadataMap, ready for per-function modification.  It
// is a NamedMD containing an MDTuple containing a number of MDNodes each of
// which is an integer value, and each two integer values forms a key=value
// pair that we store as PALMetadataMap[key]=value in the map.
void AMDGPUAsmPrinter::readPALMetadata(Module &M) {
  auto NamedMD = M.getNamedMetadata("amdgpu.pal.metadata");
  if (!NamedMD || !NamedMD->getNumOperands())
    return;
  auto Tuple = dyn_cast<MDTuple>(NamedMD->getOperand(0));
  if (!Tuple)
    return;
  for (unsigned I = 0, E = Tuple->getNumOperands() & -2; I != E; I += 2) {
    auto Key = mdconst::dyn_extract<ConstantInt>(Tuple->getOperand(I));
    auto Val = mdconst::dyn_extract<ConstantInt>(Tuple->getOperand(I + 1));
    if (!Key || !Val)
      continue;
    PALMetadataMap[Key->getZExtValue()] = Val->getZExtValue();
  }
}

// Print comments that apply to both callable functions and entry points.
void AMDGPUAsmPrinter::emitCommonFunctionComments(
  uint32_t NumVGPR,
  uint32_t NumSGPR,
  uint64_t ScratchSize,
  uint64_t CodeSize) {
  OutStreamer->emitRawComment(" codeLenInByte = " + Twine(CodeSize), false);
  OutStreamer->emitRawComment(" NumSgprs: " + Twine(NumSGPR), false);
  OutStreamer->emitRawComment(" NumVgprs: " + Twine(NumVGPR), false);
  OutStreamer->emitRawComment(" ScratchSize: " + Twine(ScratchSize), false);
}

bool AMDGPUAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  CurrentProgramInfo = SIProgramInfo();

  const AMDGPUMachineFunction *MFI = MF.getInfo<AMDGPUMachineFunction>();

  // The starting address of all shader programs must be 256 bytes aligned.
  // Regular functions just need the basic required instruction alignment.
  MF.setAlignment(MFI->isEntryFunction() ? 8 : 2);

  SetupMachineFunction(MF);

  const AMDGPUSubtarget &STM = MF.getSubtarget<AMDGPUSubtarget>();
  MCContext &Context = getObjFileLowering().getContext();
  if (!STM.isAmdHsaOS()) {
    MCSectionELF *ConfigSection =
        Context.getELFSection(".AMDGPU.config", ELF::SHT_PROGBITS, 0);
    OutStreamer->SwitchSection(ConfigSection);
  }

  if (STM.getGeneration() >= AMDGPUSubtarget::SOUTHERN_ISLANDS) {
    if (MFI->isEntryFunction()) {
      getSIProgramInfo(CurrentProgramInfo, MF);
    } else {
      auto I = CallGraphResourceInfo.insert(
        std::make_pair(&MF.getFunction(), SIFunctionResourceInfo()));
      SIFunctionResourceInfo &Info = I.first->second;
      assert(I.second && "should only be called once per function");
      Info = analyzeResourceUsage(MF);
    }

    if (STM.isAmdPalOS())
      EmitPALMetadata(MF, CurrentProgramInfo);
    if (!STM.isAmdHsaOS()) {
      EmitProgramInfoSI(MF, CurrentProgramInfo);
    }
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
      if (!MFI->isEntryFunction()) {
        OutStreamer->emitRawComment(" Function info:", false);
        SIFunctionResourceInfo &Info = CallGraphResourceInfo[&MF.getFunction()];
        emitCommonFunctionComments(
          Info.NumVGPR,
          Info.getTotalNumSGPRs(MF.getSubtarget<SISubtarget>()),
          Info.PrivateSegmentSize,
          getFunctionCodeSize(MF));
        return false;
      }

      OutStreamer->emitRawComment(" Kernel info:", false);
      emitCommonFunctionComments(CurrentProgramInfo.NumVGPR,
                                 CurrentProgramInfo.NumSGPR,
                                 CurrentProgramInfo.ScratchSize,
                                 getFunctionCodeSize(MF));

      OutStreamer->emitRawComment(
        " FloatMode: " + Twine(CurrentProgramInfo.FloatMode), false);
      OutStreamer->emitRawComment(
        " IeeeMode: " + Twine(CurrentProgramInfo.IEEEMode), false);
      OutStreamer->emitRawComment(
        " LDSByteSize: " + Twine(CurrentProgramInfo.LDSSize) +
        " bytes/workgroup (compile time only)", false);

      OutStreamer->emitRawComment(
        " SGPRBlocks: " + Twine(CurrentProgramInfo.SGPRBlocks), false);
      OutStreamer->emitRawComment(
        " VGPRBlocks: " + Twine(CurrentProgramInfo.VGPRBlocks), false);

      OutStreamer->emitRawComment(
        " NumSGPRsForWavesPerEU: " +
        Twine(CurrentProgramInfo.NumSGPRsForWavesPerEU), false);
      OutStreamer->emitRawComment(
        " NumVGPRsForWavesPerEU: " +
        Twine(CurrentProgramInfo.NumVGPRsForWavesPerEU), false);

      OutStreamer->emitRawComment(
        " ReservedVGPRFirst: " + Twine(CurrentProgramInfo.ReservedVGPRFirst),
        false);
      OutStreamer->emitRawComment(
        " ReservedVGPRCount: " + Twine(CurrentProgramInfo.ReservedVGPRCount),
        false);

      if (MF.getSubtarget<SISubtarget>().debuggerEmitPrologue()) {
        OutStreamer->emitRawComment(
          " DebuggerWavefrontPrivateSegmentOffsetSGPR: s" +
          Twine(CurrentProgramInfo.DebuggerWavefrontPrivateSegmentOffsetSGPR), false);
        OutStreamer->emitRawComment(
          " DebuggerPrivateSegmentBufferSGPR: s" +
          Twine(CurrentProgramInfo.DebuggerPrivateSegmentBufferSGPR), false);
      }

      OutStreamer->emitRawComment(
        " COMPUTE_PGM_RSRC2:USER_SGPR: " +
        Twine(G_00B84C_USER_SGPR(CurrentProgramInfo.ComputePGMRSrc2)), false);
      OutStreamer->emitRawComment(
        " COMPUTE_PGM_RSRC2:TRAP_HANDLER: " +
        Twine(G_00B84C_TRAP_HANDLER(CurrentProgramInfo.ComputePGMRSrc2)), false);
      OutStreamer->emitRawComment(
        " COMPUTE_PGM_RSRC2:TGID_X_EN: " +
        Twine(G_00B84C_TGID_X_EN(CurrentProgramInfo.ComputePGMRSrc2)), false);
      OutStreamer->emitRawComment(
        " COMPUTE_PGM_RSRC2:TGID_Y_EN: " +
        Twine(G_00B84C_TGID_Y_EN(CurrentProgramInfo.ComputePGMRSrc2)), false);
      OutStreamer->emitRawComment(
        " COMPUTE_PGM_RSRC2:TGID_Z_EN: " +
        Twine(G_00B84C_TGID_Z_EN(CurrentProgramInfo.ComputePGMRSrc2)), false);
      OutStreamer->emitRawComment(
        " COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: " +
        Twine(G_00B84C_TIDIG_COMP_CNT(CurrentProgramInfo.ComputePGMRSrc2)),
        false);
    } else {
      R600MachineFunctionInfo *MFI = MF.getInfo<R600MachineFunctionInfo>();
      OutStreamer->emitRawComment(
        Twine("SQ_PGM_RESOURCES:STACK_SIZE = " + Twine(MFI->CFStackSize)));
    }
  }

  if (STM.dumpCode()) {

    OutStreamer->SwitchSection(
        Context.getELFSection(".AMDGPU.disasm", ELF::SHT_NOTE, 0));

    for (size_t i = 0; i < DisasmLines.size(); ++i) {
      std::string Comment = "\n";
      if (!HexLines[i].empty()) {
        Comment = std::string(DisasmLineMaxLen - DisasmLines[i].size(), ' ');
        Comment += " ; " + HexLines[i] + "\n";
      }

      OutStreamer->EmitBytes(StringRef(DisasmLines[i]));
      OutStreamer->EmitBytes(StringRef(Comment));
    }
  }

  return false;
}

void AMDGPUAsmPrinter::EmitProgramInfoR600(const MachineFunction &MF) {
  unsigned MaxGPR = 0;
  bool killPixel = false;
  const R600Subtarget &STM = MF.getSubtarget<R600Subtarget>();
  const R600RegisterInfo *RI = STM.getRegisterInfo();
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
        unsigned HWReg = RI->getHWRegIndex(MO.getReg());

        // Register with value > 127 aren't GPR
        if (HWReg > 127)
          continue;
        MaxGPR = std::max(MaxGPR, HWReg);
      }
    }
  }

  unsigned RsrcReg;
  if (STM.getGeneration() >= R600Subtarget::EVERGREEN) {
    // Evergreen / Northern Islands
    switch (MF.getFunction().getCallingConv()) {
    default: LLVM_FALLTHROUGH;
    case CallingConv::AMDGPU_CS: RsrcReg = R_0288D4_SQ_PGM_RESOURCES_LS; break;
    case CallingConv::AMDGPU_GS: RsrcReg = R_028878_SQ_PGM_RESOURCES_GS; break;
    case CallingConv::AMDGPU_PS: RsrcReg = R_028844_SQ_PGM_RESOURCES_PS; break;
    case CallingConv::AMDGPU_VS: RsrcReg = R_028860_SQ_PGM_RESOURCES_VS; break;
    }
  } else {
    // R600 / R700
    switch (MF.getFunction().getCallingConv()) {
    default: LLVM_FALLTHROUGH;
    case CallingConv::AMDGPU_GS: LLVM_FALLTHROUGH;
    case CallingConv::AMDGPU_CS: LLVM_FALLTHROUGH;
    case CallingConv::AMDGPU_VS: RsrcReg = R_028868_SQ_PGM_RESOURCES_VS; break;
    case CallingConv::AMDGPU_PS: RsrcReg = R_028850_SQ_PGM_RESOURCES_PS; break;
    }
  }

  OutStreamer->EmitIntValue(RsrcReg, 4);
  OutStreamer->EmitIntValue(S_NUM_GPRS(MaxGPR + 1) |
                           S_STACK_SIZE(MFI->CFStackSize), 4);
  OutStreamer->EmitIntValue(R_02880C_DB_SHADER_CONTROL, 4);
  OutStreamer->EmitIntValue(S_02880C_KILL_ENABLE(killPixel), 4);

  if (AMDGPU::isCompute(MF.getFunction().getCallingConv())) {
    OutStreamer->EmitIntValue(R_0288E8_SQ_LDS_ALLOC, 4);
    OutStreamer->EmitIntValue(alignTo(MFI->getLDSSize(), 4) >> 2, 4);
  }
}

uint64_t AMDGPUAsmPrinter::getFunctionCodeSize(const MachineFunction &MF) const {
  const SISubtarget &STM = MF.getSubtarget<SISubtarget>();
  const SIInstrInfo *TII = STM.getInstrInfo();

  uint64_t CodeSize = 0;

  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      // TODO: CodeSize should account for multiple functions.

      // TODO: Should we count size of debug info?
      if (MI.isDebugValue())
        continue;

      CodeSize += TII->getInstSizeInBytes(MI);
    }
  }

  return CodeSize;
}

static bool hasAnyNonFlatUseOfReg(const MachineRegisterInfo &MRI,
                                  const SIInstrInfo &TII,
                                  unsigned Reg) {
  for (const MachineOperand &UseOp : MRI.reg_operands(Reg)) {
    if (!UseOp.isImplicit() || !TII.isFLAT(*UseOp.getParent()))
      return true;
  }

  return false;
}

static unsigned getNumExtraSGPRs(const SISubtarget &ST,
                                 bool VCCUsed,
                                 bool FlatScrUsed) {
  unsigned ExtraSGPRs = 0;
  if (VCCUsed)
    ExtraSGPRs = 2;

  if (ST.getGeneration() < SISubtarget::VOLCANIC_ISLANDS) {
    if (FlatScrUsed)
      ExtraSGPRs = 4;
  } else {
    if (ST.isXNACKEnabled())
      ExtraSGPRs = 4;

    if (FlatScrUsed)
      ExtraSGPRs = 6;
  }

  return ExtraSGPRs;
}

int32_t AMDGPUAsmPrinter::SIFunctionResourceInfo::getTotalNumSGPRs(
  const SISubtarget &ST) const {
  return NumExplicitSGPR + getNumExtraSGPRs(ST, UsesVCC, UsesFlatScratch);
}

AMDGPUAsmPrinter::SIFunctionResourceInfo AMDGPUAsmPrinter::analyzeResourceUsage(
  const MachineFunction &MF) const {
  SIFunctionResourceInfo Info;

  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  const SISubtarget &ST = MF.getSubtarget<SISubtarget>();
  const MachineFrameInfo &FrameInfo = MF.getFrameInfo();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIInstrInfo *TII = ST.getInstrInfo();
  const SIRegisterInfo &TRI = TII->getRegisterInfo();

  Info.UsesFlatScratch = MRI.isPhysRegUsed(AMDGPU::FLAT_SCR_LO) ||
                         MRI.isPhysRegUsed(AMDGPU::FLAT_SCR_HI);

  // Even if FLAT_SCRATCH is implicitly used, it has no effect if flat
  // instructions aren't used to access the scratch buffer. Inline assembly may
  // need it though.
  //
  // If we only have implicit uses of flat_scr on flat instructions, it is not
  // really needed.
  if (Info.UsesFlatScratch && !MFI->hasFlatScratchInit() &&
      (!hasAnyNonFlatUseOfReg(MRI, *TII, AMDGPU::FLAT_SCR) &&
       !hasAnyNonFlatUseOfReg(MRI, *TII, AMDGPU::FLAT_SCR_LO) &&
       !hasAnyNonFlatUseOfReg(MRI, *TII, AMDGPU::FLAT_SCR_HI))) {
    Info.UsesFlatScratch = false;
  }

  Info.HasDynamicallySizedStack = FrameInfo.hasVarSizedObjects();
  Info.PrivateSegmentSize = FrameInfo.getStackSize();


  Info.UsesVCC = MRI.isPhysRegUsed(AMDGPU::VCC_LO) ||
                 MRI.isPhysRegUsed(AMDGPU::VCC_HI);

  // If there are no calls, MachineRegisterInfo can tell us the used register
  // count easily.
  // A tail call isn't considered a call for MachineFrameInfo's purposes.
  if (!FrameInfo.hasCalls() && !FrameInfo.hasTailCall()) {
    MCPhysReg HighestVGPRReg = AMDGPU::NoRegister;
    for (MCPhysReg Reg : reverse(AMDGPU::VGPR_32RegClass.getRegisters())) {
      if (MRI.isPhysRegUsed(Reg)) {
        HighestVGPRReg = Reg;
        break;
      }
    }

    MCPhysReg HighestSGPRReg = AMDGPU::NoRegister;
    for (MCPhysReg Reg : reverse(AMDGPU::SGPR_32RegClass.getRegisters())) {
      if (MRI.isPhysRegUsed(Reg)) {
        HighestSGPRReg = Reg;
        break;
      }
    }

    // We found the maximum register index. They start at 0, so add one to get the
    // number of registers.
    Info.NumVGPR = HighestVGPRReg == AMDGPU::NoRegister ? 0 :
      TRI.getHWRegIndex(HighestVGPRReg) + 1;
    Info.NumExplicitSGPR = HighestSGPRReg == AMDGPU::NoRegister ? 0 :
      TRI.getHWRegIndex(HighestSGPRReg) + 1;

    return Info;
  }

  int32_t MaxVGPR = -1;
  int32_t MaxSGPR = -1;
  uint64_t CalleeFrameSize = 0;

  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      // TODO: Check regmasks? Do they occur anywhere except calls?
      for (const MachineOperand &MO : MI.operands()) {
        unsigned Width = 0;
        bool IsSGPR = false;

        if (!MO.isReg())
          continue;

        unsigned Reg = MO.getReg();
        switch (Reg) {
        case AMDGPU::EXEC:
        case AMDGPU::EXEC_LO:
        case AMDGPU::EXEC_HI:
        case AMDGPU::SCC:
        case AMDGPU::M0:
        case AMDGPU::SRC_SHARED_BASE:
        case AMDGPU::SRC_SHARED_LIMIT:
        case AMDGPU::SRC_PRIVATE_BASE:
        case AMDGPU::SRC_PRIVATE_LIMIT:
          continue;

        case AMDGPU::NoRegister:
          assert(MI.isDebugValue());
          continue;

        case AMDGPU::VCC:
        case AMDGPU::VCC_LO:
        case AMDGPU::VCC_HI:
          Info.UsesVCC = true;
          continue;

        case AMDGPU::FLAT_SCR:
        case AMDGPU::FLAT_SCR_LO:
        case AMDGPU::FLAT_SCR_HI:
          continue;

        case AMDGPU::XNACK_MASK:
        case AMDGPU::XNACK_MASK_LO:
        case AMDGPU::XNACK_MASK_HI:
          llvm_unreachable("xnack_mask registers should not be used");

        case AMDGPU::TBA:
        case AMDGPU::TBA_LO:
        case AMDGPU::TBA_HI:
        case AMDGPU::TMA:
        case AMDGPU::TMA_LO:
        case AMDGPU::TMA_HI:
          llvm_unreachable("trap handler registers should not be used");

        default:
          break;
        }

        if (AMDGPU::SReg_32RegClass.contains(Reg)) {
          assert(!AMDGPU::TTMP_32RegClass.contains(Reg) &&
                 "trap handler registers should not be used");
          IsSGPR = true;
          Width = 1;
        } else if (AMDGPU::VGPR_32RegClass.contains(Reg)) {
          IsSGPR = false;
          Width = 1;
        } else if (AMDGPU::SReg_64RegClass.contains(Reg)) {
          assert(!AMDGPU::TTMP_64RegClass.contains(Reg) &&
                 "trap handler registers should not be used");
          IsSGPR = true;
          Width = 2;
        } else if (AMDGPU::VReg_64RegClass.contains(Reg)) {
          IsSGPR = false;
          Width = 2;
        } else if (AMDGPU::VReg_96RegClass.contains(Reg)) {
          IsSGPR = false;
          Width = 3;
        } else if (AMDGPU::SReg_128RegClass.contains(Reg)) {
          assert(!AMDGPU::TTMP_128RegClass.contains(Reg) &&
            "trap handler registers should not be used");
          IsSGPR = true;
          Width = 4;
        } else if (AMDGPU::VReg_128RegClass.contains(Reg)) {
          IsSGPR = false;
          Width = 4;
        } else if (AMDGPU::SReg_256RegClass.contains(Reg)) {
          assert(!AMDGPU::TTMP_256RegClass.contains(Reg) &&
            "trap handler registers should not be used");
          IsSGPR = true;
          Width = 8;
        } else if (AMDGPU::VReg_256RegClass.contains(Reg)) {
          IsSGPR = false;
          Width = 8;
        } else if (AMDGPU::SReg_512RegClass.contains(Reg)) {
          assert(!AMDGPU::TTMP_512RegClass.contains(Reg) &&
            "trap handler registers should not be used");
          IsSGPR = true;
          Width = 16;
        } else if (AMDGPU::VReg_512RegClass.contains(Reg)) {
          IsSGPR = false;
          Width = 16;
        } else {
          llvm_unreachable("Unknown register class");
        }
        unsigned HWReg = TRI.getHWRegIndex(Reg);
        int MaxUsed = HWReg + Width - 1;
        if (IsSGPR) {
          MaxSGPR = MaxUsed > MaxSGPR ? MaxUsed : MaxSGPR;
        } else {
          MaxVGPR = MaxUsed > MaxVGPR ? MaxUsed : MaxVGPR;
        }
      }

      if (MI.isCall()) {
        // Pseudo used just to encode the underlying global. Is there a better
        // way to track this?

        const MachineOperand *CalleeOp
          = TII->getNamedOperand(MI, AMDGPU::OpName::callee);
        const Function *Callee = cast<Function>(CalleeOp->getGlobal());
        if (Callee->isDeclaration()) {
          // If this is a call to an external function, we can't do much. Make
          // conservative guesses.

          // 48 SGPRs - vcc, - flat_scr, -xnack
          int MaxSGPRGuess = 47 - getNumExtraSGPRs(ST, true,
                                                   ST.hasFlatAddressSpace());
          MaxSGPR = std::max(MaxSGPR, MaxSGPRGuess);
          MaxVGPR = std::max(MaxVGPR, 23);

          CalleeFrameSize = std::max(CalleeFrameSize, UINT64_C(16384));
          Info.UsesVCC = true;
          Info.UsesFlatScratch = ST.hasFlatAddressSpace();
          Info.HasDynamicallySizedStack = true;
        } else {
          // We force CodeGen to run in SCC order, so the callee's register
          // usage etc. should be the cumulative usage of all callees.
          auto I = CallGraphResourceInfo.find(Callee);
          assert(I != CallGraphResourceInfo.end() &&
                 "callee should have been handled before caller");

          MaxSGPR = std::max(I->second.NumExplicitSGPR - 1, MaxSGPR);
          MaxVGPR = std::max(I->second.NumVGPR - 1, MaxVGPR);
          CalleeFrameSize
            = std::max(I->second.PrivateSegmentSize, CalleeFrameSize);
          Info.UsesVCC |= I->second.UsesVCC;
          Info.UsesFlatScratch |= I->second.UsesFlatScratch;
          Info.HasDynamicallySizedStack |= I->second.HasDynamicallySizedStack;
          Info.HasRecursion |= I->second.HasRecursion;
        }

        if (!Callee->doesNotRecurse())
          Info.HasRecursion = true;
      }
    }
  }

  Info.NumExplicitSGPR = MaxSGPR + 1;
  Info.NumVGPR = MaxVGPR + 1;
  Info.PrivateSegmentSize += CalleeFrameSize;

  return Info;
}

void AMDGPUAsmPrinter::getSIProgramInfo(SIProgramInfo &ProgInfo,
                                        const MachineFunction &MF) {
  SIFunctionResourceInfo Info = analyzeResourceUsage(MF);

  ProgInfo.NumVGPR = Info.NumVGPR;
  ProgInfo.NumSGPR = Info.NumExplicitSGPR;
  ProgInfo.ScratchSize = Info.PrivateSegmentSize;
  ProgInfo.VCCUsed = Info.UsesVCC;
  ProgInfo.FlatUsed = Info.UsesFlatScratch;
  ProgInfo.DynamicCallStack = Info.HasDynamicallySizedStack || Info.HasRecursion;

  if (!isUInt<32>(ProgInfo.ScratchSize)) {
    DiagnosticInfoStackSize DiagStackSize(MF.getFunction(),
                                          ProgInfo.ScratchSize, DS_Error);
    MF.getFunction().getContext().diagnose(DiagStackSize);
  }

  const SISubtarget &STM = MF.getSubtarget<SISubtarget>();
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  const SIInstrInfo *TII = STM.getInstrInfo();
  const SIRegisterInfo *RI = &TII->getRegisterInfo();

  unsigned ExtraSGPRs = getNumExtraSGPRs(STM,
                                         ProgInfo.VCCUsed,
                                         ProgInfo.FlatUsed);
  unsigned ExtraVGPRs = STM.getReservedNumVGPRs(MF);

  // Check the addressable register limit before we add ExtraSGPRs.
  if (STM.getGeneration() >= AMDGPUSubtarget::VOLCANIC_ISLANDS &&
      !STM.hasSGPRInitBug()) {
    unsigned MaxAddressableNumSGPRs = STM.getAddressableNumSGPRs();
    if (ProgInfo.NumSGPR > MaxAddressableNumSGPRs) {
      // This can happen due to a compiler bug or when using inline asm.
      LLVMContext &Ctx = MF.getFunction().getContext();
      DiagnosticInfoResourceLimit Diag(MF.getFunction(),
                                       "addressable scalar registers",
                                       ProgInfo.NumSGPR, DS_Error,
                                       DK_ResourceLimit,
                                       MaxAddressableNumSGPRs);
      Ctx.diagnose(Diag);
      ProgInfo.NumSGPR = MaxAddressableNumSGPRs - 1;
    }
  }

  // Account for extra SGPRs and VGPRs reserved for debugger use.
  ProgInfo.NumSGPR += ExtraSGPRs;
  ProgInfo.NumVGPR += ExtraVGPRs;

  // Adjust number of registers used to meet default/requested minimum/maximum
  // number of waves per execution unit request.
  ProgInfo.NumSGPRsForWavesPerEU = std::max(
    std::max(ProgInfo.NumSGPR, 1u), STM.getMinNumSGPRs(MFI->getMaxWavesPerEU()));
  ProgInfo.NumVGPRsForWavesPerEU = std::max(
    std::max(ProgInfo.NumVGPR, 1u), STM.getMinNumVGPRs(MFI->getMaxWavesPerEU()));

  if (STM.getGeneration() <= AMDGPUSubtarget::SEA_ISLANDS ||
      STM.hasSGPRInitBug()) {
    unsigned MaxAddressableNumSGPRs = STM.getAddressableNumSGPRs();
    if (ProgInfo.NumSGPR > MaxAddressableNumSGPRs) {
      // This can happen due to a compiler bug or when using inline asm to use
      // the registers which are usually reserved for vcc etc.
      LLVMContext &Ctx = MF.getFunction().getContext();
      DiagnosticInfoResourceLimit Diag(MF.getFunction(),
                                       "scalar registers",
                                       ProgInfo.NumSGPR, DS_Error,
                                       DK_ResourceLimit,
                                       MaxAddressableNumSGPRs);
      Ctx.diagnose(Diag);
      ProgInfo.NumSGPR = MaxAddressableNumSGPRs;
      ProgInfo.NumSGPRsForWavesPerEU = MaxAddressableNumSGPRs;
    }
  }

  if (STM.hasSGPRInitBug()) {
    ProgInfo.NumSGPR =
        AMDGPU::IsaInfo::FIXED_NUM_SGPRS_FOR_INIT_BUG;
    ProgInfo.NumSGPRsForWavesPerEU =
        AMDGPU::IsaInfo::FIXED_NUM_SGPRS_FOR_INIT_BUG;
  }

  if (MFI->getNumUserSGPRs() > STM.getMaxNumUserSGPRs()) {
    LLVMContext &Ctx = MF.getFunction().getContext();
    DiagnosticInfoResourceLimit Diag(MF.getFunction(), "user SGPRs",
                                     MFI->getNumUserSGPRs(), DS_Error);
    Ctx.diagnose(Diag);
  }

  if (MFI->getLDSSize() > static_cast<unsigned>(STM.getLocalMemorySize())) {
    LLVMContext &Ctx = MF.getFunction().getContext();
    DiagnosticInfoResourceLimit Diag(MF.getFunction(), "local memory",
                                     MFI->getLDSSize(), DS_Error);
    Ctx.diagnose(Diag);
  }

  // SGPRBlocks is actual number of SGPR blocks minus 1.
  ProgInfo.SGPRBlocks = alignTo(ProgInfo.NumSGPRsForWavesPerEU,
                                STM.getSGPREncodingGranule());
  ProgInfo.SGPRBlocks = ProgInfo.SGPRBlocks / STM.getSGPREncodingGranule() - 1;

  // VGPRBlocks is actual number of VGPR blocks minus 1.
  ProgInfo.VGPRBlocks = alignTo(ProgInfo.NumVGPRsForWavesPerEU,
                                STM.getVGPREncodingGranule());
  ProgInfo.VGPRBlocks = ProgInfo.VGPRBlocks / STM.getVGPREncodingGranule() - 1;

  // Record first reserved VGPR and number of reserved VGPRs.
  ProgInfo.ReservedVGPRFirst = STM.debuggerReserveRegs() ? ProgInfo.NumVGPR : 0;
  ProgInfo.ReservedVGPRCount = STM.getReservedNumVGPRs(MF);

  // Update DebuggerWavefrontPrivateSegmentOffsetSGPR and
  // DebuggerPrivateSegmentBufferSGPR fields if "amdgpu-debugger-emit-prologue"
  // attribute was requested.
  if (STM.debuggerEmitPrologue()) {
    ProgInfo.DebuggerWavefrontPrivateSegmentOffsetSGPR =
      RI->getHWRegIndex(MFI->getScratchWaveOffsetReg());
    ProgInfo.DebuggerPrivateSegmentBufferSGPR =
      RI->getHWRegIndex(MFI->getScratchRSrcReg());
  }

  // Set the value to initialize FP_ROUND and FP_DENORM parts of the mode
  // register.
  ProgInfo.FloatMode = getFPMode(MF);

  ProgInfo.IEEEMode = STM.enableIEEEBit(MF);

  // Make clamp modifier on NaN input returns 0.
  ProgInfo.DX10Clamp = STM.enableDX10Clamp();

  unsigned LDSAlignShift;
  if (STM.getGeneration() < SISubtarget::SEA_ISLANDS) {
    // LDS is allocated in 64 dword blocks.
    LDSAlignShift = 8;
  } else {
    // LDS is allocated in 128 dword blocks.
    LDSAlignShift = 9;
  }

  unsigned LDSSpillSize =
    MFI->getLDSWaveSpillSize() * MFI->getMaxFlatWorkGroupSize();

  ProgInfo.LDSSize = MFI->getLDSSize() + LDSSpillSize;
  ProgInfo.LDSBlocks =
      alignTo(ProgInfo.LDSSize, 1ULL << LDSAlignShift) >> LDSAlignShift;

  // Scratch is allocated in 256 dword blocks.
  unsigned ScratchAlignShift = 10;
  // We need to program the hardware with the amount of scratch memory that
  // is used by the entire wave.  ProgInfo.ScratchSize is the amount of
  // scratch memory used per thread.
  ProgInfo.ScratchBlocks =
      alignTo(ProgInfo.ScratchSize * STM.getWavefrontSize(),
              1ULL << ScratchAlignShift) >>
      ScratchAlignShift;

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
      S_00B84C_TRAP_HANDLER(STM.isTrapHandlerEnabled()) |
      S_00B84C_TGID_X_EN(MFI->hasWorkGroupIDX()) |
      S_00B84C_TGID_Y_EN(MFI->hasWorkGroupIDY()) |
      S_00B84C_TGID_Z_EN(MFI->hasWorkGroupIDZ()) |
      S_00B84C_TG_SIZE_EN(MFI->hasWorkGroupInfo()) |
      S_00B84C_TIDIG_COMP_CNT(TIDIGCompCnt) |
      S_00B84C_EXCP_EN_MSB(0) |
      // For AMDHSA, LDS_SIZE must be zero, as it is populated by the CP.
      S_00B84C_LDS_SIZE(STM.isAmdHsaOS() ? 0 : ProgInfo.LDSBlocks) |
      S_00B84C_EXCP_EN(0);
}

static unsigned getRsrcReg(CallingConv::ID CallConv) {
  switch (CallConv) {
  default: LLVM_FALLTHROUGH;
  case CallingConv::AMDGPU_CS: return R_00B848_COMPUTE_PGM_RSRC1;
  case CallingConv::AMDGPU_LS: return R_00B528_SPI_SHADER_PGM_RSRC1_LS;
  case CallingConv::AMDGPU_HS: return R_00B428_SPI_SHADER_PGM_RSRC1_HS;
  case CallingConv::AMDGPU_ES: return R_00B328_SPI_SHADER_PGM_RSRC1_ES;
  case CallingConv::AMDGPU_GS: return R_00B228_SPI_SHADER_PGM_RSRC1_GS;
  case CallingConv::AMDGPU_VS: return R_00B128_SPI_SHADER_PGM_RSRC1_VS;
  case CallingConv::AMDGPU_PS: return R_00B028_SPI_SHADER_PGM_RSRC1_PS;
  }
}

void AMDGPUAsmPrinter::EmitProgramInfoSI(const MachineFunction &MF,
                                         const SIProgramInfo &CurrentProgramInfo) {
  const SISubtarget &STM = MF.getSubtarget<SISubtarget>();
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  unsigned RsrcReg = getRsrcReg(MF.getFunction().getCallingConv());

  if (AMDGPU::isCompute(MF.getFunction().getCallingConv())) {
    OutStreamer->EmitIntValue(R_00B848_COMPUTE_PGM_RSRC1, 4);

    OutStreamer->EmitIntValue(CurrentProgramInfo.ComputePGMRSrc1, 4);

    OutStreamer->EmitIntValue(R_00B84C_COMPUTE_PGM_RSRC2, 4);
    OutStreamer->EmitIntValue(CurrentProgramInfo.ComputePGMRSrc2, 4);

    OutStreamer->EmitIntValue(R_00B860_COMPUTE_TMPRING_SIZE, 4);
    OutStreamer->EmitIntValue(S_00B860_WAVESIZE(CurrentProgramInfo.ScratchBlocks), 4);

    // TODO: Should probably note flat usage somewhere. SC emits a "FlatPtr32 =
    // 0" comment but I don't see a corresponding field in the register spec.
  } else {
    OutStreamer->EmitIntValue(RsrcReg, 4);
    OutStreamer->EmitIntValue(S_00B028_VGPRS(CurrentProgramInfo.VGPRBlocks) |
                              S_00B028_SGPRS(CurrentProgramInfo.SGPRBlocks), 4);
    unsigned Rsrc2Val = 0;
    if (STM.isVGPRSpillingEnabled(MF.getFunction())) {
      OutStreamer->EmitIntValue(R_0286E8_SPI_TMPRING_SIZE, 4);
      OutStreamer->EmitIntValue(S_0286E8_WAVESIZE(CurrentProgramInfo.ScratchBlocks), 4);
      if (TM.getTargetTriple().getOS() == Triple::AMDPAL)
        Rsrc2Val = S_00B84C_SCRATCH_EN(CurrentProgramInfo.ScratchBlocks > 0);
    }
    if (MF.getFunction().getCallingConv() == CallingConv::AMDGPU_PS) {
      OutStreamer->EmitIntValue(R_0286CC_SPI_PS_INPUT_ENA, 4);
      OutStreamer->EmitIntValue(MFI->getPSInputEnable(), 4);
      OutStreamer->EmitIntValue(R_0286D0_SPI_PS_INPUT_ADDR, 4);
      OutStreamer->EmitIntValue(MFI->getPSInputAddr(), 4);
      Rsrc2Val |= S_00B02C_EXTRA_LDS_SIZE(CurrentProgramInfo.LDSBlocks);
    }
    if (Rsrc2Val) {
      OutStreamer->EmitIntValue(RsrcReg + 4 /*rsrc2*/, 4);
      OutStreamer->EmitIntValue(Rsrc2Val, 4);
    }
  }

  OutStreamer->EmitIntValue(R_SPILLED_SGPRS, 4);
  OutStreamer->EmitIntValue(MFI->getNumSpilledSGPRs(), 4);
  OutStreamer->EmitIntValue(R_SPILLED_VGPRS, 4);
  OutStreamer->EmitIntValue(MFI->getNumSpilledVGPRs(), 4);
}

// This is the equivalent of EmitProgramInfoSI above, but for when the OS type
// is AMDPAL.  It stores each compute/SPI register setting and other PAL
// metadata items into the PALMetadataMap, combining with any provided by the
// frontend as LLVM metadata. Once all functions are written, PALMetadataMap is
// then written as a single block in the .note section.
void AMDGPUAsmPrinter::EmitPALMetadata(const MachineFunction &MF,
       const SIProgramInfo &CurrentProgramInfo) {
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  // Given the calling convention, calculate the register number for rsrc1. In
  // principle the register number could change in future hardware, but we know
  // it is the same for gfx6-9 (except that LS and ES don't exist on gfx9), so
  // we can use the same fixed value that .AMDGPU.config has for Mesa. Note
  // that we use a register number rather than a byte offset, so we need to
  // divide by 4.
  unsigned Rsrc1Reg = getRsrcReg(MF.getFunction().getCallingConv()) / 4;
  unsigned Rsrc2Reg = Rsrc1Reg + 1;
  // Also calculate the PAL metadata key for *S_SCRATCH_SIZE. It can be used
  // with a constant offset to access any non-register shader-specific PAL
  // metadata key.
  unsigned ScratchSizeKey = PALMD::Key::CS_SCRATCH_SIZE;
  switch (MF.getFunction().getCallingConv()) {
    case CallingConv::AMDGPU_PS:
      ScratchSizeKey = PALMD::Key::PS_SCRATCH_SIZE;
      break;
    case CallingConv::AMDGPU_VS:
      ScratchSizeKey = PALMD::Key::VS_SCRATCH_SIZE;
      break;
    case CallingConv::AMDGPU_GS:
      ScratchSizeKey = PALMD::Key::GS_SCRATCH_SIZE;
      break;
    case CallingConv::AMDGPU_ES:
      ScratchSizeKey = PALMD::Key::ES_SCRATCH_SIZE;
      break;
    case CallingConv::AMDGPU_HS:
      ScratchSizeKey = PALMD::Key::HS_SCRATCH_SIZE;
      break;
    case CallingConv::AMDGPU_LS:
      ScratchSizeKey = PALMD::Key::LS_SCRATCH_SIZE;
      break;
  }
  unsigned NumUsedVgprsKey = ScratchSizeKey +
      PALMD::Key::VS_NUM_USED_VGPRS - PALMD::Key::VS_SCRATCH_SIZE;
  unsigned NumUsedSgprsKey = ScratchSizeKey +
      PALMD::Key::VS_NUM_USED_SGPRS - PALMD::Key::VS_SCRATCH_SIZE;
  PALMetadataMap[NumUsedVgprsKey] = CurrentProgramInfo.NumVGPRsForWavesPerEU;
  PALMetadataMap[NumUsedSgprsKey] = CurrentProgramInfo.NumSGPRsForWavesPerEU;
  if (AMDGPU::isCompute(MF.getFunction().getCallingConv())) {
    PALMetadataMap[Rsrc1Reg] |= CurrentProgramInfo.ComputePGMRSrc1;
    PALMetadataMap[Rsrc2Reg] |= CurrentProgramInfo.ComputePGMRSrc2;
    // ScratchSize is in bytes, 16 aligned.
    PALMetadataMap[ScratchSizeKey] |=
        alignTo(CurrentProgramInfo.ScratchSize, 16);
  } else {
    PALMetadataMap[Rsrc1Reg] |= S_00B028_VGPRS(CurrentProgramInfo.VGPRBlocks) |
        S_00B028_SGPRS(CurrentProgramInfo.SGPRBlocks);
    if (CurrentProgramInfo.ScratchBlocks > 0)
      PALMetadataMap[Rsrc2Reg] |= S_00B84C_SCRATCH_EN(1);
    // ScratchSize is in bytes, 16 aligned.
    PALMetadataMap[ScratchSizeKey] |=
        alignTo(CurrentProgramInfo.ScratchSize, 16);
  }
  if (MF.getFunction().getCallingConv() == CallingConv::AMDGPU_PS) {
    PALMetadataMap[Rsrc2Reg] |=
        S_00B02C_EXTRA_LDS_SIZE(CurrentProgramInfo.LDSBlocks);
    PALMetadataMap[R_0286CC_SPI_PS_INPUT_ENA / 4] |= MFI->getPSInputEnable();
    PALMetadataMap[R_0286D0_SPI_PS_INPUT_ADDR / 4] |= MFI->getPSInputAddr();
  }
}

// This is supposed to be log2(Size)
static amd_element_byte_size_t getElementByteSizeValue(unsigned Size) {
  switch (Size) {
  case 4:
    return AMD_ELEMENT_4_BYTES;
  case 8:
    return AMD_ELEMENT_8_BYTES;
  case 16:
    return AMD_ELEMENT_16_BYTES;
  default:
    llvm_unreachable("invalid private_element_size");
  }
}

void AMDGPUAsmPrinter::getAmdKernelCode(amd_kernel_code_t &Out,
                                        const SIProgramInfo &CurrentProgramInfo,
                                        const MachineFunction &MF) const {
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  const SISubtarget &STM = MF.getSubtarget<SISubtarget>();

  AMDGPU::initDefaultAMDKernelCodeT(Out, STM.getFeatureBits());

  Out.compute_pgm_resource_registers =
      CurrentProgramInfo.ComputePGMRSrc1 |
      (CurrentProgramInfo.ComputePGMRSrc2 << 32);
  Out.code_properties = AMD_CODE_PROPERTY_IS_PTR64;

  if (CurrentProgramInfo.DynamicCallStack)
    Out.code_properties |= AMD_CODE_PROPERTY_IS_DYNAMIC_CALLSTACK;

  AMD_HSA_BITS_SET(Out.code_properties,
                   AMD_CODE_PROPERTY_PRIVATE_ELEMENT_SIZE,
                   getElementByteSizeValue(STM.getMaxPrivateElementSize()));

  if (MFI->hasPrivateSegmentBuffer()) {
    Out.code_properties |=
      AMD_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER;
  }

  if (MFI->hasDispatchPtr())
    Out.code_properties |= AMD_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR;

  if (MFI->hasQueuePtr())
    Out.code_properties |= AMD_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR;

  if (MFI->hasKernargSegmentPtr())
    Out.code_properties |= AMD_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR;

  if (MFI->hasDispatchID())
    Out.code_properties |= AMD_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID;

  if (MFI->hasFlatScratchInit())
    Out.code_properties |= AMD_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT;

  if (MFI->hasGridWorkgroupCountX()) {
    Out.code_properties |=
      AMD_CODE_PROPERTY_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X;
  }

  if (MFI->hasGridWorkgroupCountY()) {
    Out.code_properties |=
      AMD_CODE_PROPERTY_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y;
  }

  if (MFI->hasGridWorkgroupCountZ()) {
    Out.code_properties |=
      AMD_CODE_PROPERTY_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z;
  }

  if (MFI->hasDispatchPtr())
    Out.code_properties |= AMD_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR;

  if (STM.debuggerSupported())
    Out.code_properties |= AMD_CODE_PROPERTY_IS_DEBUG_SUPPORTED;

  if (STM.isXNACKEnabled())
    Out.code_properties |= AMD_CODE_PROPERTY_IS_XNACK_SUPPORTED;

  // FIXME: Should use getKernArgSize
  Out.kernarg_segment_byte_size =
    STM.getKernArgSegmentSize(MF, MFI->getABIArgOffset());
  Out.wavefront_sgpr_count = CurrentProgramInfo.NumSGPR;
  Out.workitem_vgpr_count = CurrentProgramInfo.NumVGPR;
  Out.workitem_private_segment_byte_size = CurrentProgramInfo.ScratchSize;
  Out.workgroup_group_segment_byte_size = CurrentProgramInfo.LDSSize;
  Out.reserved_vgpr_first = CurrentProgramInfo.ReservedVGPRFirst;
  Out.reserved_vgpr_count = CurrentProgramInfo.ReservedVGPRCount;

  // These alignment values are specified in powers of two, so alignment =
  // 2^n.  The minimum alignment is 2^4 = 16.
  Out.kernarg_segment_alignment = std::max((size_t)4,
      countTrailingZeros(MFI->getMaxKernArgAlign()));

  if (STM.debuggerEmitPrologue()) {
    Out.debug_wavefront_private_segment_offset_sgpr =
      CurrentProgramInfo.DebuggerWavefrontPrivateSegmentOffsetSGPR;
    Out.debug_private_segment_buffer_sgpr =
      CurrentProgramInfo.DebuggerPrivateSegmentBufferSGPR;
  }
}

AMDGPU::HSAMD::Kernel::CodeProps::Metadata AMDGPUAsmPrinter::getHSACodeProps(
    const MachineFunction &MF,
    const SIProgramInfo &ProgramInfo) const {
  const SISubtarget &STM = MF.getSubtarget<SISubtarget>();
  const SIMachineFunctionInfo &MFI = *MF.getInfo<SIMachineFunctionInfo>();
  HSAMD::Kernel::CodeProps::Metadata HSACodeProps;

  HSACodeProps.mKernargSegmentSize =
      STM.getKernArgSegmentSize(MF, MFI.getABIArgOffset());
  HSACodeProps.mGroupSegmentFixedSize = ProgramInfo.LDSSize;
  HSACodeProps.mPrivateSegmentFixedSize = ProgramInfo.ScratchSize;
  HSACodeProps.mKernargSegmentAlign =
      std::max(uint32_t(4), MFI.getMaxKernArgAlign());
  HSACodeProps.mWavefrontSize = STM.getWavefrontSize();
  HSACodeProps.mNumSGPRs = CurrentProgramInfo.NumSGPR;
  HSACodeProps.mNumVGPRs = CurrentProgramInfo.NumVGPR;
  HSACodeProps.mMaxFlatWorkGroupSize = MFI.getMaxFlatWorkGroupSize();
  HSACodeProps.mIsDynamicCallStack = ProgramInfo.DynamicCallStack;
  HSACodeProps.mIsXNACKEnabled = STM.isXNACKEnabled();
  HSACodeProps.mNumSpilledSGPRs = MFI.getNumSpilledSGPRs();
  HSACodeProps.mNumSpilledVGPRs = MFI.getNumSpilledVGPRs();

  return HSACodeProps;
}

AMDGPU::HSAMD::Kernel::DebugProps::Metadata AMDGPUAsmPrinter::getHSADebugProps(
    const MachineFunction &MF,
    const SIProgramInfo &ProgramInfo) const {
  const SISubtarget &STM = MF.getSubtarget<SISubtarget>();
  HSAMD::Kernel::DebugProps::Metadata HSADebugProps;

  if (!STM.debuggerSupported())
    return HSADebugProps;

  HSADebugProps.mDebuggerABIVersion.push_back(1);
  HSADebugProps.mDebuggerABIVersion.push_back(0);
  HSADebugProps.mReservedNumVGPRs = ProgramInfo.ReservedVGPRCount;
  HSADebugProps.mReservedFirstVGPR = ProgramInfo.ReservedVGPRFirst;

  if (STM.debuggerEmitPrologue()) {
    HSADebugProps.mPrivateSegmentBufferSGPR =
        ProgramInfo.DebuggerPrivateSegmentBufferSGPR;
    HSADebugProps.mWavefrontPrivateSegmentOffsetSGPR =
        ProgramInfo.DebuggerWavefrontPrivateSegmentOffsetSGPR;
  }

  return HSADebugProps;
}

bool AMDGPUAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                       unsigned AsmVariant,
                                       const char *ExtraCode, raw_ostream &O) {
  // First try the generic code, which knows about modifiers like 'c' and 'n'.
  if (!AsmPrinter::PrintAsmOperand(MI, OpNo, AsmVariant, ExtraCode, O))
    return false;

  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0)
      return true; // Unknown modifier.

    switch (ExtraCode[0]) {
    case 'r':
      break;
    default:
      return true;
    }
  }

  // TODO: Should be able to support other operand types like globals.
  const MachineOperand &MO = MI->getOperand(OpNo);
  if (MO.isReg()) {
    AMDGPUInstPrinter::printRegOperand(MO.getReg(), O,
                                       *MF->getSubtarget().getRegisterInfo());
    return false;
  }

  return true;
}
