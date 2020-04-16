//===-- AMDGPUAsmPrinter.cpp - AMDGPU assembly printer --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "MCTargetDesc/AMDGPUInstPrinter.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "MCTargetDesc/AMDGPUTargetStreamer.h"
#include "R600AsmPrinter.h"
#include "R600Defines.h"
#include "R600MachineFunctionInfo.h"
#include "R600RegisterInfo.h"
#include "SIDefines.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "TargetInfo/AMDGPUTargetInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/AMDGPUMetadata.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TargetParser.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetLoweringObjectFile.h"

using namespace llvm;
using namespace llvm::AMDGPU;
using namespace llvm::AMDGPU::HSAMD;

// This should get the default rounding mode from the kernel. We just set the
// default here, but this could change if the OpenCL rounding mode pragmas are
// used.
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
static uint32_t getFPMode(AMDGPU::SIModeRegisterDefaults Mode) {
  return FP_ROUND_MODE_SP(FP_ROUND_ROUND_TO_NEAREST) |
         FP_ROUND_MODE_DP(FP_ROUND_ROUND_TO_NEAREST) |
         FP_DENORM_MODE_SP(Mode.fpDenormModeSPValue()) |
         FP_DENORM_MODE_DP(Mode.fpDenormModeDPValue());
}

static AsmPrinter *
createAMDGPUAsmPrinterPass(TargetMachine &tm,
                           std::unique_ptr<MCStreamer> &&Streamer) {
  return new AMDGPUAsmPrinter(tm, std::move(Streamer));
}

extern "C" void LLVM_EXTERNAL_VISIBILITY LLVMInitializeAMDGPUAsmPrinter() {
  TargetRegistry::RegisterAsmPrinter(getTheAMDGPUTarget(),
                                     llvm::createR600AsmPrinterPass);
  TargetRegistry::RegisterAsmPrinter(getTheGCNTarget(),
                                     createAMDGPUAsmPrinterPass);
}

AMDGPUAsmPrinter::AMDGPUAsmPrinter(TargetMachine &TM,
                                   std::unique_ptr<MCStreamer> Streamer)
  : AsmPrinter(TM, std::move(Streamer)) {
    if (IsaInfo::hasCodeObjectV3(getGlobalSTI()))
      HSAMetadataStream.reset(new MetadataStreamerV3());
    else
      HSAMetadataStream.reset(new MetadataStreamerV2());
}

StringRef AMDGPUAsmPrinter::getPassName() const {
  return "AMDGPU Assembly Printer";
}

const MCSubtargetInfo *AMDGPUAsmPrinter::getGlobalSTI() const {
  return TM.getMCSubtargetInfo();
}

AMDGPUTargetStreamer* AMDGPUAsmPrinter::getTargetStreamer() const {
  if (!OutStreamer)
    return nullptr;
  return static_cast<AMDGPUTargetStreamer*>(OutStreamer->getTargetStreamer());
}

void AMDGPUAsmPrinter::emitStartOfAsmFile(Module &M) {
  if (IsaInfo::hasCodeObjectV3(getGlobalSTI())) {
    std::string ExpectedTarget;
    raw_string_ostream ExpectedTargetOS(ExpectedTarget);
    IsaInfo::streamIsaVersion(getGlobalSTI(), ExpectedTargetOS);

    getTargetStreamer()->EmitDirectiveAMDGCNTarget(ExpectedTarget);
  }

  if (TM.getTargetTriple().getOS() != Triple::AMDHSA &&
      TM.getTargetTriple().getOS() != Triple::AMDPAL)
    return;

  if (TM.getTargetTriple().getOS() == Triple::AMDHSA)
    HSAMetadataStream->begin(M);

  if (TM.getTargetTriple().getOS() == Triple::AMDPAL)
    getTargetStreamer()->getPALMetadata()->readFromIR(M);

  if (IsaInfo::hasCodeObjectV3(getGlobalSTI()))
    return;

  // HSA emits NT_AMDGPU_HSA_CODE_OBJECT_VERSION for code objects v2.
  if (TM.getTargetTriple().getOS() == Triple::AMDHSA)
    getTargetStreamer()->EmitDirectiveHSACodeObjectVersion(2, 1);

  // HSA and PAL emit NT_AMDGPU_HSA_ISA for code objects v2.
  IsaVersion Version = getIsaVersion(getGlobalSTI()->getCPU());
  getTargetStreamer()->EmitDirectiveHSACodeObjectISA(
      Version.Major, Version.Minor, Version.Stepping, "AMD", "AMDGPU");
}

void AMDGPUAsmPrinter::emitEndOfAsmFile(Module &M) {
  // Following code requires TargetStreamer to be present.
  if (!getTargetStreamer())
    return;

  if (!IsaInfo::hasCodeObjectV3(getGlobalSTI())) {
    // Emit ISA Version (NT_AMD_AMDGPU_ISA).
    std::string ISAVersionString;
    raw_string_ostream ISAVersionStream(ISAVersionString);
    IsaInfo::streamIsaVersion(getGlobalSTI(), ISAVersionStream);
    getTargetStreamer()->EmitISAVersion(ISAVersionStream.str());
  }

  // Emit HSA Metadata (NT_AMD_AMDGPU_HSA_METADATA).
  if (TM.getTargetTriple().getOS() == Triple::AMDHSA) {
    HSAMetadataStream->end();
    bool Success = HSAMetadataStream->emitTo(*getTargetStreamer());
    (void)Success;
    assert(Success && "Malformed HSA Metadata");
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

void AMDGPUAsmPrinter::emitFunctionBodyStart() {
  const SIMachineFunctionInfo &MFI = *MF->getInfo<SIMachineFunctionInfo>();
  if (!MFI.isEntryFunction())
    return;

  const GCNSubtarget &STM = MF->getSubtarget<GCNSubtarget>();
  const Function &F = MF->getFunction();
  if (!STM.hasCodeObjectV3() && STM.isAmdHsaOrMesa(F) &&
      (F.getCallingConv() == CallingConv::AMDGPU_KERNEL ||
       F.getCallingConv() == CallingConv::SPIR_KERNEL)) {
    amd_kernel_code_t KernelCode;
    getAmdKernelCode(KernelCode, CurrentProgramInfo, *MF);
    getTargetStreamer()->EmitAMDKernelCodeT(KernelCode);
  }

  if (STM.isAmdHsaOS())
    HSAMetadataStream->emitKernel(*MF, CurrentProgramInfo);
}

void AMDGPUAsmPrinter::emitFunctionBodyEnd() {
  const SIMachineFunctionInfo &MFI = *MF->getInfo<SIMachineFunctionInfo>();
  if (!MFI.isEntryFunction())
    return;

  if (!IsaInfo::hasCodeObjectV3(getGlobalSTI()) ||
      TM.getTargetTriple().getOS() != Triple::AMDHSA)
    return;

  auto &Streamer = getTargetStreamer()->getStreamer();
  auto &Context = Streamer.getContext();
  auto &ObjectFileInfo = *Context.getObjectFileInfo();
  auto &ReadOnlySection = *ObjectFileInfo.getReadOnlySection();

  Streamer.PushSection();
  Streamer.SwitchSection(&ReadOnlySection);

  // CP microcode requires the kernel descriptor to be allocated on 64 byte
  // alignment.
  Streamer.emitValueToAlignment(64, 0, 1, 0);
  if (ReadOnlySection.getAlignment() < 64)
    ReadOnlySection.setAlignment(Align(64));

  const MCSubtargetInfo &STI = MF->getSubtarget();

  SmallString<128> KernelName;
  getNameWithPrefix(KernelName, &MF->getFunction());
  getTargetStreamer()->EmitAmdhsaKernelDescriptor(
      STI, KernelName, getAmdhsaKernelDescriptor(*MF, CurrentProgramInfo),
      CurrentProgramInfo.NumVGPRsForWavesPerEU,
      CurrentProgramInfo.NumSGPRsForWavesPerEU -
          IsaInfo::getNumExtraSGPRs(&STI,
                                    CurrentProgramInfo.VCCUsed,
                                    CurrentProgramInfo.FlatUsed),
      CurrentProgramInfo.VCCUsed, CurrentProgramInfo.FlatUsed,
      hasXNACK(STI));

  Streamer.PopSection();
}

void AMDGPUAsmPrinter::emitFunctionEntryLabel() {
  if (IsaInfo::hasCodeObjectV3(getGlobalSTI()) &&
      TM.getTargetTriple().getOS() == Triple::AMDHSA) {
    AsmPrinter::emitFunctionEntryLabel();
    return;
  }

  const SIMachineFunctionInfo *MFI = MF->getInfo<SIMachineFunctionInfo>();
  const GCNSubtarget &STM = MF->getSubtarget<GCNSubtarget>();
  if (MFI->isEntryFunction() && STM.isAmdHsaOrMesa(MF->getFunction())) {
    SmallString<128> SymbolName;
    getNameWithPrefix(SymbolName, &MF->getFunction()),
    getTargetStreamer()->EmitAMDGPUSymbolType(
        SymbolName, ELF::STT_AMDGPU_HSA_KERNEL);
  }
  if (DumpCodeInstEmitter) {
    // Disassemble function name label to text.
    DisasmLines.push_back(MF->getName().str() + ":");
    DisasmLineMaxLen = std::max(DisasmLineMaxLen, DisasmLines.back().size());
    HexLines.push_back("");
  }

  AsmPrinter::emitFunctionEntryLabel();
}

void AMDGPUAsmPrinter::emitBasicBlockStart(const MachineBasicBlock &MBB) {
  if (DumpCodeInstEmitter && !isBlockOnlyReachableByFallthrough(&MBB)) {
    // Write a line for the basic block label if it is not only fallthrough.
    DisasmLines.push_back(
        (Twine("BB") + Twine(getFunctionNumber())
         + "_" + Twine(MBB.getNumber()) + ":").str());
    DisasmLineMaxLen = std::max(DisasmLineMaxLen, DisasmLines.back().size());
    HexLines.push_back("");
  }
  AsmPrinter::emitBasicBlockStart(MBB);
}

void AMDGPUAsmPrinter::emitGlobalVariable(const GlobalVariable *GV) {
  if (GV->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS) {
    if (GV->hasInitializer() && !isa<UndefValue>(GV->getInitializer())) {
      OutContext.reportError({},
                             Twine(GV->getName()) +
                                 ": unsupported initializer for address space");
      return;
    }

    // LDS variables aren't emitted in HSA or PAL yet.
    const Triple::OSType OS = TM.getTargetTriple().getOS();
    if (OS == Triple::AMDHSA || OS == Triple::AMDPAL)
      return;

    MCSymbol *GVSym = getSymbol(GV);

    GVSym->redefineIfPossible();
    if (GVSym->isDefined() || GVSym->isVariable())
      report_fatal_error("symbol '" + Twine(GVSym->getName()) +
                         "' is already defined");

    const DataLayout &DL = GV->getParent()->getDataLayout();
    uint64_t Size = DL.getTypeAllocSize(GV->getValueType());
    unsigned Align = GV->getAlignment();
    if (!Align)
      Align = 4;

    emitVisibility(GVSym, GV->getVisibility(), !GV->isDeclaration());
    emitLinkage(GV, GVSym);
    if (auto TS = getTargetStreamer())
      TS->emitAMDGPULDS(GVSym, Size, Align);
    return;
  }

  AsmPrinter::emitGlobalVariable(GV);
}

bool AMDGPUAsmPrinter::doFinalization(Module &M) {
  CallGraphResourceInfo.clear();

  // Pad with s_code_end to help tools and guard against instruction prefetch
  // causing stale data in caches. Arguably this should be done by the linker,
  // which is why this isn't done for Mesa.
  const MCSubtargetInfo &STI = *getGlobalSTI();
  if (AMDGPU::isGFX10(STI) &&
      (STI.getTargetTriple().getOS() == Triple::AMDHSA ||
       STI.getTargetTriple().getOS() == Triple::AMDPAL)) {
    OutStreamer->SwitchSection(getObjFileLowering().getTextSection());
    getTargetStreamer()->EmitCodeEnd();
  }

  return AsmPrinter::doFinalization(M);
}

// Print comments that apply to both callable functions and entry points.
void AMDGPUAsmPrinter::emitCommonFunctionComments(
  uint32_t NumVGPR,
  Optional<uint32_t> NumAGPR,
  uint32_t TotalNumVGPR,
  uint32_t NumSGPR,
  uint64_t ScratchSize,
  uint64_t CodeSize,
  const AMDGPUMachineFunction *MFI) {
  OutStreamer->emitRawComment(" codeLenInByte = " + Twine(CodeSize), false);
  OutStreamer->emitRawComment(" NumSgprs: " + Twine(NumSGPR), false);
  OutStreamer->emitRawComment(" NumVgprs: " + Twine(NumVGPR), false);
  if (NumAGPR) {
    OutStreamer->emitRawComment(" NumAgprs: " + Twine(*NumAGPR), false);
    OutStreamer->emitRawComment(" TotalNumVgprs: " + Twine(TotalNumVGPR),
                                false);
  }
  OutStreamer->emitRawComment(" ScratchSize: " + Twine(ScratchSize), false);
  OutStreamer->emitRawComment(" MemoryBound: " + Twine(MFI->isMemoryBound()),
                              false);
}

uint16_t AMDGPUAsmPrinter::getAmdhsaKernelCodeProperties(
    const MachineFunction &MF) const {
  const SIMachineFunctionInfo &MFI = *MF.getInfo<SIMachineFunctionInfo>();
  uint16_t KernelCodeProperties = 0;

  if (MFI.hasPrivateSegmentBuffer()) {
    KernelCodeProperties |=
        amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER;
  }
  if (MFI.hasDispatchPtr()) {
    KernelCodeProperties |=
        amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR;
  }
  if (MFI.hasQueuePtr()) {
    KernelCodeProperties |=
        amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR;
  }
  if (MFI.hasKernargSegmentPtr()) {
    KernelCodeProperties |=
        amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR;
  }
  if (MFI.hasDispatchID()) {
    KernelCodeProperties |=
        amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID;
  }
  if (MFI.hasFlatScratchInit()) {
    KernelCodeProperties |=
        amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT;
  }
  if (MF.getSubtarget<GCNSubtarget>().isWave32()) {
    KernelCodeProperties |=
        amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32;
  }

  return KernelCodeProperties;
}

amdhsa::kernel_descriptor_t AMDGPUAsmPrinter::getAmdhsaKernelDescriptor(
    const MachineFunction &MF,
    const SIProgramInfo &PI) const {
  amdhsa::kernel_descriptor_t KernelDescriptor;
  memset(&KernelDescriptor, 0x0, sizeof(KernelDescriptor));

  assert(isUInt<32>(PI.ScratchSize));
  assert(isUInt<32>(PI.ComputePGMRSrc1));
  assert(isUInt<32>(PI.ComputePGMRSrc2));

  KernelDescriptor.group_segment_fixed_size = PI.LDSSize;
  KernelDescriptor.private_segment_fixed_size = PI.ScratchSize;
  KernelDescriptor.compute_pgm_rsrc1 = PI.ComputePGMRSrc1;
  KernelDescriptor.compute_pgm_rsrc2 = PI.ComputePGMRSrc2;
  KernelDescriptor.kernel_code_properties = getAmdhsaKernelCodeProperties(MF);

  return KernelDescriptor;
}

bool AMDGPUAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  CurrentProgramInfo = SIProgramInfo();

  const AMDGPUMachineFunction *MFI = MF.getInfo<AMDGPUMachineFunction>();

  // The starting address of all shader programs must be 256 bytes aligned.
  // Regular functions just need the basic required instruction alignment.
  MF.setAlignment(MFI->isEntryFunction() ? Align(256) : Align(4));

  SetupMachineFunction(MF);

  const GCNSubtarget &STM = MF.getSubtarget<GCNSubtarget>();
  MCContext &Context = getObjFileLowering().getContext();
  // FIXME: This should be an explicit check for Mesa.
  if (!STM.isAmdHsaOS() && !STM.isAmdPalOS()) {
    MCSectionELF *ConfigSection =
        Context.getELFSection(".AMDGPU.config", ELF::SHT_PROGBITS, 0);
    OutStreamer->SwitchSection(ConfigSection);
  }

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
  else if (!STM.isAmdHsaOS()) {
    EmitProgramInfoSI(MF, CurrentProgramInfo);
  }

  DumpCodeInstEmitter = nullptr;
  if (STM.dumpCode()) {
    // For -dumpcode, get the assembler out of the streamer, even if it does
    // not really want to let us have it. This only works with -filetype=obj.
    bool SaveFlag = OutStreamer->getUseAssemblerInfoForParsing();
    OutStreamer->setUseAssemblerInfoForParsing(true);
    MCAssembler *Assembler = OutStreamer->getAssemblerPtr();
    OutStreamer->setUseAssemblerInfoForParsing(SaveFlag);
    if (Assembler)
      DumpCodeInstEmitter = Assembler->getEmitterPtr();
  }

  DisasmLines.clear();
  HexLines.clear();
  DisasmLineMaxLen = 0;

  emitFunctionBody();

  if (isVerbose()) {
    MCSectionELF *CommentSection =
        Context.getELFSection(".AMDGPU.csdata", ELF::SHT_PROGBITS, 0);
    OutStreamer->SwitchSection(CommentSection);

    if (!MFI->isEntryFunction()) {
      OutStreamer->emitRawComment(" Function info:", false);
      SIFunctionResourceInfo &Info = CallGraphResourceInfo[&MF.getFunction()];
      emitCommonFunctionComments(
        Info.NumVGPR,
        STM.hasMAIInsts() ? Info.NumAGPR : Optional<uint32_t>(),
        Info.getTotalNumVGPRs(STM),
        Info.getTotalNumSGPRs(MF.getSubtarget<GCNSubtarget>()),
        Info.PrivateSegmentSize,
        getFunctionCodeSize(MF), MFI);
      return false;
    }

    OutStreamer->emitRawComment(" Kernel info:", false);
    emitCommonFunctionComments(CurrentProgramInfo.NumArchVGPR,
                               STM.hasMAIInsts()
                                 ? CurrentProgramInfo.NumAccVGPR
                                 : Optional<uint32_t>(),
                               CurrentProgramInfo.NumVGPR,
                               CurrentProgramInfo.NumSGPR,
                               CurrentProgramInfo.ScratchSize,
                               getFunctionCodeSize(MF), MFI);

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
      " Occupancy: " +
      Twine(CurrentProgramInfo.Occupancy), false);

    OutStreamer->emitRawComment(
      " WaveLimiterHint : " + Twine(MFI->needsWaveLimiter()), false);

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
  }

  if (DumpCodeInstEmitter) {

    OutStreamer->SwitchSection(
        Context.getELFSection(".AMDGPU.disasm", ELF::SHT_PROGBITS, 0));

    for (size_t i = 0; i < DisasmLines.size(); ++i) {
      std::string Comment = "\n";
      if (!HexLines[i].empty()) {
        Comment = std::string(DisasmLineMaxLen - DisasmLines[i].size(), ' ');
        Comment += " ; " + HexLines[i] + "\n";
      }

      OutStreamer->emitBytes(StringRef(DisasmLines[i]));
      OutStreamer->emitBytes(StringRef(Comment));
    }
  }

  return false;
}

uint64_t AMDGPUAsmPrinter::getFunctionCodeSize(const MachineFunction &MF) const {
  const GCNSubtarget &STM = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = STM.getInstrInfo();

  uint64_t CodeSize = 0;

  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      // TODO: CodeSize should account for multiple functions.

      // TODO: Should we count size of debug info?
      if (MI.isDebugInstr())
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

int32_t AMDGPUAsmPrinter::SIFunctionResourceInfo::getTotalNumSGPRs(
  const GCNSubtarget &ST) const {
  return NumExplicitSGPR + IsaInfo::getNumExtraSGPRs(&ST,
                                                     UsesVCC, UsesFlatScratch);
}

int32_t AMDGPUAsmPrinter::SIFunctionResourceInfo::getTotalNumVGPRs(
  const GCNSubtarget &ST) const {
  return std::max(NumVGPR, NumAGPR);
}

static const Function *getCalleeFunction(const MachineOperand &Op) {
  if (Op.isImm()) {
    assert(Op.getImm() == 0);
    return nullptr;
  }

  return cast<Function>(Op.getGlobal());
}

AMDGPUAsmPrinter::SIFunctionResourceInfo AMDGPUAsmPrinter::analyzeResourceUsage(
  const MachineFunction &MF) const {
  SIFunctionResourceInfo Info;

  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
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
  if (MFI->isStackRealigned())
    Info.PrivateSegmentSize += FrameInfo.getMaxAlign().value();

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

    if (ST.hasMAIInsts()) {
      MCPhysReg HighestAGPRReg = AMDGPU::NoRegister;
      for (MCPhysReg Reg : reverse(AMDGPU::AGPR_32RegClass.getRegisters())) {
        if (MRI.isPhysRegUsed(Reg)) {
          HighestAGPRReg = Reg;
          break;
        }
      }
      Info.NumAGPR = HighestAGPRReg == AMDGPU::NoRegister ? 0 :
        TRI.getHWRegIndex(HighestAGPRReg) + 1;
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
  int32_t MaxAGPR = -1;
  int32_t MaxSGPR = -1;
  uint64_t CalleeFrameSize = 0;

  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      // TODO: Check regmasks? Do they occur anywhere except calls?
      for (const MachineOperand &MO : MI.operands()) {
        unsigned Width = 0;
        bool IsSGPR = false;
        bool IsAGPR = false;

        if (!MO.isReg())
          continue;

        Register Reg = MO.getReg();
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
        case AMDGPU::SGPR_NULL:
          continue;

        case AMDGPU::SRC_POPS_EXITING_WAVE_ID:
          llvm_unreachable("src_pops_exiting_wave_id should not be used");

        case AMDGPU::NoRegister:
          assert(MI.isDebugInstr());
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

        case AMDGPU::LDS_DIRECT:
          llvm_unreachable("lds_direct register should not be used");

        case AMDGPU::TBA:
        case AMDGPU::TBA_LO:
        case AMDGPU::TBA_HI:
        case AMDGPU::TMA:
        case AMDGPU::TMA_LO:
        case AMDGPU::TMA_HI:
          llvm_unreachable("trap handler registers should not be used");

        case AMDGPU::SRC_VCCZ:
          llvm_unreachable("src_vccz register should not be used");

        case AMDGPU::SRC_EXECZ:
          llvm_unreachable("src_execz register should not be used");

        case AMDGPU::SRC_SCC:
          llvm_unreachable("src_scc register should not be used");

        default:
          break;
        }

        if (AMDGPU::SReg_32RegClass.contains(Reg) ||
            AMDGPU::SGPR_LO16RegClass.contains(Reg) ||
            AMDGPU::SGPR_HI16RegClass.contains(Reg)) {
          assert(!AMDGPU::TTMP_32RegClass.contains(Reg) &&
                 "trap handler registers should not be used");
          IsSGPR = true;
          Width = 1;
        } else if (AMDGPU::VGPR_32RegClass.contains(Reg) ||
                   AMDGPU::VGPR_LO16RegClass.contains(Reg) ||
                   AMDGPU::VGPR_HI16RegClass.contains(Reg)) {
          IsSGPR = false;
          Width = 1;
        } else if (AMDGPU::AGPR_32RegClass.contains(Reg)) {
          IsSGPR = false;
          IsAGPR = true;
          Width = 1;
        } else if (AMDGPU::SReg_64RegClass.contains(Reg)) {
          assert(!AMDGPU::TTMP_64RegClass.contains(Reg) &&
                 "trap handler registers should not be used");
          IsSGPR = true;
          Width = 2;
        } else if (AMDGPU::VReg_64RegClass.contains(Reg)) {
          IsSGPR = false;
          Width = 2;
        } else if (AMDGPU::AReg_64RegClass.contains(Reg)) {
          IsSGPR = false;
          IsAGPR = true;
          Width = 2;
        } else if (AMDGPU::VReg_96RegClass.contains(Reg)) {
          IsSGPR = false;
          Width = 3;
        } else if (AMDGPU::SReg_96RegClass.contains(Reg)) {
          IsSGPR = true;
          Width = 3;
        } else if (AMDGPU::SReg_128RegClass.contains(Reg)) {
          assert(!AMDGPU::TTMP_128RegClass.contains(Reg) &&
            "trap handler registers should not be used");
          IsSGPR = true;
          Width = 4;
        } else if (AMDGPU::VReg_128RegClass.contains(Reg)) {
          IsSGPR = false;
          Width = 4;
        } else if (AMDGPU::AReg_128RegClass.contains(Reg)) {
          IsSGPR = false;
          IsAGPR = true;
          Width = 4;
        } else if (AMDGPU::VReg_160RegClass.contains(Reg)) {
          IsSGPR = false;
          Width = 5;
        } else if (AMDGPU::SReg_160RegClass.contains(Reg)) {
          IsSGPR = true;
          Width = 5;
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
        } else if (AMDGPU::AReg_512RegClass.contains(Reg)) {
          IsSGPR = false;
          IsAGPR = true;
          Width = 16;
        } else if (AMDGPU::SReg_1024RegClass.contains(Reg)) {
          IsSGPR = true;
          Width = 32;
        } else if (AMDGPU::VReg_1024RegClass.contains(Reg)) {
          IsSGPR = false;
          Width = 32;
        } else if (AMDGPU::AReg_1024RegClass.contains(Reg)) {
          IsSGPR = false;
          IsAGPR = true;
          Width = 32;
        } else {
          llvm_unreachable("Unknown register class");
        }
        unsigned HWReg = TRI.getHWRegIndex(Reg);
        int MaxUsed = HWReg + Width - 1;
        if (IsSGPR) {
          MaxSGPR = MaxUsed > MaxSGPR ? MaxUsed : MaxSGPR;
        } else if (IsAGPR) {
          MaxAGPR = MaxUsed > MaxAGPR ? MaxUsed : MaxAGPR;
        } else {
          MaxVGPR = MaxUsed > MaxVGPR ? MaxUsed : MaxVGPR;
        }
      }

      if (MI.isCall()) {
        // Pseudo used just to encode the underlying global. Is there a better
        // way to track this?

        const MachineOperand *CalleeOp
          = TII->getNamedOperand(MI, AMDGPU::OpName::callee);

        const Function *Callee = getCalleeFunction(*CalleeOp);
        if (!Callee || Callee->isDeclaration()) {
          // If this is a call to an external function, we can't do much. Make
          // conservative guesses.

          // 48 SGPRs - vcc, - flat_scr, -xnack
          int MaxSGPRGuess =
            47 - IsaInfo::getNumExtraSGPRs(&ST, true, ST.hasFlatAddressSpace());
          MaxSGPR = std::max(MaxSGPR, MaxSGPRGuess);
          MaxVGPR = std::max(MaxVGPR, 23);
          MaxAGPR = std::max(MaxAGPR, 23);

          CalleeFrameSize = std::max(CalleeFrameSize, UINT64_C(16384));
          Info.UsesVCC = true;
          Info.UsesFlatScratch = ST.hasFlatAddressSpace();
          Info.HasDynamicallySizedStack = true;
        } else {
          // We force CodeGen to run in SCC order, so the callee's register
          // usage etc. should be the cumulative usage of all callees.

          auto I = CallGraphResourceInfo.find(Callee);
          if (I == CallGraphResourceInfo.end()) {
            // Avoid crashing on undefined behavior with an illegal call to a
            // kernel. If a callsite's calling convention doesn't match the
            // function's, it's undefined behavior. If the callsite calling
            // convention does match, that would have errored earlier.
            // FIXME: The verifier shouldn't allow this.
            if (AMDGPU::isEntryFunctionCC(Callee->getCallingConv()))
              report_fatal_error("invalid call to entry function");

            llvm_unreachable("callee should have been handled before caller");
          }

          MaxSGPR = std::max(I->second.NumExplicitSGPR - 1, MaxSGPR);
          MaxVGPR = std::max(I->second.NumVGPR - 1, MaxVGPR);
          MaxAGPR = std::max(I->second.NumAGPR - 1, MaxAGPR);
          CalleeFrameSize
            = std::max(I->second.PrivateSegmentSize, CalleeFrameSize);
          Info.UsesVCC |= I->second.UsesVCC;
          Info.UsesFlatScratch |= I->second.UsesFlatScratch;
          Info.HasDynamicallySizedStack |= I->second.HasDynamicallySizedStack;
          Info.HasRecursion |= I->second.HasRecursion;
        }

        // FIXME: Call site could have norecurse on it
        if (!Callee || !Callee->doesNotRecurse())
          Info.HasRecursion = true;
      }
    }
  }

  Info.NumExplicitSGPR = MaxSGPR + 1;
  Info.NumVGPR = MaxVGPR + 1;
  Info.NumAGPR = MaxAGPR + 1;
  Info.PrivateSegmentSize += CalleeFrameSize;

  return Info;
}

void AMDGPUAsmPrinter::getSIProgramInfo(SIProgramInfo &ProgInfo,
                                        const MachineFunction &MF) {
  SIFunctionResourceInfo Info = analyzeResourceUsage(MF);
  const GCNSubtarget &STM = MF.getSubtarget<GCNSubtarget>();

  ProgInfo.NumArchVGPR = Info.NumVGPR;
  ProgInfo.NumAccVGPR = Info.NumAGPR;
  ProgInfo.NumVGPR = Info.getTotalNumVGPRs(STM);
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

  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

  // TODO(scott.linder): The calculations related to SGPR/VGPR blocks are
  // duplicated in part in AMDGPUAsmParser::calculateGPRBlocks, and could be
  // unified.
  unsigned ExtraSGPRs = IsaInfo::getNumExtraSGPRs(
      &STM, ProgInfo.VCCUsed, ProgInfo.FlatUsed);

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

  // Ensure there are enough SGPRs and VGPRs for wave dispatch, where wave
  // dispatch registers are function args.
  unsigned WaveDispatchNumSGPR = 0, WaveDispatchNumVGPR = 0;
  for (auto &Arg : MF.getFunction().args()) {
    unsigned NumRegs = (Arg.getType()->getPrimitiveSizeInBits() + 31) / 32;
    if (Arg.hasAttribute(Attribute::InReg))
      WaveDispatchNumSGPR += NumRegs;
    else
      WaveDispatchNumVGPR += NumRegs;
  }
  ProgInfo.NumSGPR = std::max(ProgInfo.NumSGPR, WaveDispatchNumSGPR);
  ProgInfo.NumVGPR = std::max(ProgInfo.NumVGPR, WaveDispatchNumVGPR);

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

  ProgInfo.SGPRBlocks = IsaInfo::getNumSGPRBlocks(
      &STM, ProgInfo.NumSGPRsForWavesPerEU);
  ProgInfo.VGPRBlocks = IsaInfo::getNumVGPRBlocks(
      &STM, ProgInfo.NumVGPRsForWavesPerEU);

  const SIModeRegisterDefaults Mode = MFI->getMode();

  // Set the value to initialize FP_ROUND and FP_DENORM parts of the mode
  // register.
  ProgInfo.FloatMode = getFPMode(Mode);

  ProgInfo.IEEEMode = Mode.IEEE;

  // Make clamp modifier on NaN input returns 0.
  ProgInfo.DX10Clamp = Mode.DX10Clamp;

  unsigned LDSAlignShift;
  if (STM.getGeneration() < AMDGPUSubtarget::SEA_ISLANDS) {
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

  if (getIsaVersion(getGlobalSTI()->getCPU()).Major >= 10) {
    ProgInfo.WgpMode = STM.isCuModeEnabled() ? 0 : 1;
    ProgInfo.MemOrdered = 1;
  }

  ProgInfo.ComputePGMRSrc1 =
      S_00B848_VGPRS(ProgInfo.VGPRBlocks) |
      S_00B848_SGPRS(ProgInfo.SGPRBlocks) |
      S_00B848_PRIORITY(ProgInfo.Priority) |
      S_00B848_FLOAT_MODE(ProgInfo.FloatMode) |
      S_00B848_PRIV(ProgInfo.Priv) |
      S_00B848_DX10_CLAMP(ProgInfo.DX10Clamp) |
      S_00B848_DEBUG_MODE(ProgInfo.DebugMode) |
      S_00B848_IEEE_MODE(ProgInfo.IEEEMode) |
      S_00B848_WGP_MODE(ProgInfo.WgpMode) |
      S_00B848_MEM_ORDERED(ProgInfo.MemOrdered);

  // 0 = X, 1 = XY, 2 = XYZ
  unsigned TIDIGCompCnt = 0;
  if (MFI->hasWorkItemIDZ())
    TIDIGCompCnt = 2;
  else if (MFI->hasWorkItemIDY())
    TIDIGCompCnt = 1;

  ProgInfo.ComputePGMRSrc2 =
      S_00B84C_SCRATCH_EN(ProgInfo.ScratchBlocks > 0) |
      S_00B84C_USER_SGPR(MFI->getNumUserSGPRs()) |
      // For AMDHSA, TRAP_HANDLER must be zero, as it is populated by the CP.
      S_00B84C_TRAP_HANDLER(STM.isAmdHsaOS() ? 0 : STM.isTrapHandlerEnabled()) |
      S_00B84C_TGID_X_EN(MFI->hasWorkGroupIDX()) |
      S_00B84C_TGID_Y_EN(MFI->hasWorkGroupIDY()) |
      S_00B84C_TGID_Z_EN(MFI->hasWorkGroupIDZ()) |
      S_00B84C_TG_SIZE_EN(MFI->hasWorkGroupInfo()) |
      S_00B84C_TIDIG_COMP_CNT(TIDIGCompCnt) |
      S_00B84C_EXCP_EN_MSB(0) |
      // For AMDHSA, LDS_SIZE must be zero, as it is populated by the CP.
      S_00B84C_LDS_SIZE(STM.isAmdHsaOS() ? 0 : ProgInfo.LDSBlocks) |
      S_00B84C_EXCP_EN(0);

  ProgInfo.Occupancy = STM.computeOccupancy(MF, ProgInfo.LDSSize,
                                            ProgInfo.NumSGPRsForWavesPerEU,
                                            ProgInfo.NumVGPRsForWavesPerEU);
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
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  unsigned RsrcReg = getRsrcReg(MF.getFunction().getCallingConv());

  if (AMDGPU::isCompute(MF.getFunction().getCallingConv())) {
    OutStreamer->emitInt32(R_00B848_COMPUTE_PGM_RSRC1);

    OutStreamer->emitInt32(CurrentProgramInfo.ComputePGMRSrc1);

    OutStreamer->emitInt32(R_00B84C_COMPUTE_PGM_RSRC2);
    OutStreamer->emitInt32(CurrentProgramInfo.ComputePGMRSrc2);

    OutStreamer->emitInt32(R_00B860_COMPUTE_TMPRING_SIZE);
    OutStreamer->emitInt32(S_00B860_WAVESIZE(CurrentProgramInfo.ScratchBlocks));

    // TODO: Should probably note flat usage somewhere. SC emits a "FlatPtr32 =
    // 0" comment but I don't see a corresponding field in the register spec.
  } else {
    OutStreamer->emitInt32(RsrcReg);
    OutStreamer->emitIntValue(S_00B028_VGPRS(CurrentProgramInfo.VGPRBlocks) |
                              S_00B028_SGPRS(CurrentProgramInfo.SGPRBlocks), 4);
    OutStreamer->emitInt32(R_0286E8_SPI_TMPRING_SIZE);
    OutStreamer->emitIntValue(
        S_0286E8_WAVESIZE(CurrentProgramInfo.ScratchBlocks), 4);
  }

  if (MF.getFunction().getCallingConv() == CallingConv::AMDGPU_PS) {
    OutStreamer->emitInt32(R_00B02C_SPI_SHADER_PGM_RSRC2_PS);
    OutStreamer->emitInt32(
        S_00B02C_EXTRA_LDS_SIZE(CurrentProgramInfo.LDSBlocks));
    OutStreamer->emitInt32(R_0286CC_SPI_PS_INPUT_ENA);
    OutStreamer->emitInt32(MFI->getPSInputEnable());
    OutStreamer->emitInt32(R_0286D0_SPI_PS_INPUT_ADDR);
    OutStreamer->emitInt32(MFI->getPSInputAddr());
  }

  OutStreamer->emitInt32(R_SPILLED_SGPRS);
  OutStreamer->emitInt32(MFI->getNumSpilledSGPRs());
  OutStreamer->emitInt32(R_SPILLED_VGPRS);
  OutStreamer->emitInt32(MFI->getNumSpilledVGPRs());
}

// This is the equivalent of EmitProgramInfoSI above, but for when the OS type
// is AMDPAL.  It stores each compute/SPI register setting and other PAL
// metadata items into the PALMD::Metadata, combining with any provided by the
// frontend as LLVM metadata. Once all functions are written, the PAL metadata
// is then written as a single block in the .note section.
void AMDGPUAsmPrinter::EmitPALMetadata(const MachineFunction &MF,
       const SIProgramInfo &CurrentProgramInfo) {
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  auto CC = MF.getFunction().getCallingConv();
  auto MD = getTargetStreamer()->getPALMetadata();

  MD->setEntryPoint(CC, MF.getFunction().getName());
  MD->setNumUsedVgprs(CC, CurrentProgramInfo.NumVGPRsForWavesPerEU);
  MD->setNumUsedSgprs(CC, CurrentProgramInfo.NumSGPRsForWavesPerEU);
  if (AMDGPU::isCompute(MF.getFunction().getCallingConv())) {
    MD->setRsrc1(CC, CurrentProgramInfo.ComputePGMRSrc1);
    MD->setRsrc2(CC, CurrentProgramInfo.ComputePGMRSrc2);
  } else {
    MD->setRsrc1(CC, S_00B028_VGPRS(CurrentProgramInfo.VGPRBlocks) |
        S_00B028_SGPRS(CurrentProgramInfo.SGPRBlocks));
    if (CurrentProgramInfo.ScratchBlocks > 0)
      MD->setRsrc2(CC, S_00B84C_SCRATCH_EN(1));
  }
  // ScratchSize is in bytes, 16 aligned.
  MD->setScratchSize(CC, alignTo(CurrentProgramInfo.ScratchSize, 16));
  if (MF.getFunction().getCallingConv() == CallingConv::AMDGPU_PS) {
    MD->setRsrc2(CC, S_00B02C_EXTRA_LDS_SIZE(CurrentProgramInfo.LDSBlocks));
    MD->setSpiPsInputEna(MFI->getPSInputEnable());
    MD->setSpiPsInputAddr(MFI->getPSInputAddr());
  }

  const GCNSubtarget &STM = MF.getSubtarget<GCNSubtarget>();
  if (STM.isWave32())
    MD->setWave32(MF.getFunction().getCallingConv());
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
  const Function &F = MF.getFunction();
  assert(F.getCallingConv() == CallingConv::AMDGPU_KERNEL ||
         F.getCallingConv() == CallingConv::SPIR_KERNEL);

  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  const GCNSubtarget &STM = MF.getSubtarget<GCNSubtarget>();

  AMDGPU::initDefaultAMDKernelCodeT(Out, &STM);

  Out.compute_pgm_resource_registers =
      CurrentProgramInfo.ComputePGMRSrc1 |
      (CurrentProgramInfo.ComputePGMRSrc2 << 32);
  Out.code_properties |= AMD_CODE_PROPERTY_IS_PTR64;

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

  if (MFI->hasDispatchPtr())
    Out.code_properties |= AMD_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR;

  if (STM.isXNACKEnabled())
    Out.code_properties |= AMD_CODE_PROPERTY_IS_XNACK_SUPPORTED;

  Align MaxKernArgAlign;
  Out.kernarg_segment_byte_size = STM.getKernArgSegmentSize(F, MaxKernArgAlign);
  Out.wavefront_sgpr_count = CurrentProgramInfo.NumSGPR;
  Out.workitem_vgpr_count = CurrentProgramInfo.NumVGPR;
  Out.workitem_private_segment_byte_size = CurrentProgramInfo.ScratchSize;
  Out.workgroup_group_segment_byte_size = CurrentProgramInfo.LDSSize;

  // kernarg_segment_alignment is specified as log of the alignment.
  // The minimum alignment is 16.
  Out.kernarg_segment_alignment = Log2(std::max(Align(16), MaxKernArgAlign));
}

bool AMDGPUAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                       const char *ExtraCode, raw_ostream &O) {
  // First try the generic code, which knows about modifiers like 'c' and 'n'.
  if (!AsmPrinter::PrintAsmOperand(MI, OpNo, ExtraCode, O))
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
