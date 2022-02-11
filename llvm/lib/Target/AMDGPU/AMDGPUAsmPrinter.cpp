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
#include "AMDGPUHSAMetadataStreamer.h"
#include "AMDGPUResourceUsageAnalysis.h"
#include "AMDKernelCodeT.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUInstPrinter.h"
#include "MCTargetDesc/AMDGPUTargetStreamer.h"
#include "R600AsmPrinter.h"
#include "SIMachineFunctionInfo.h"
#include "TargetInfo/AMDGPUTargetInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/TargetParser.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;
using namespace llvm::AMDGPU;

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
  if (TM.getTargetTriple().getOS() == Triple::AMDHSA) {
    if (isHsaAbiVersion2(getGlobalSTI())) {
      HSAMetadataStream.reset(new HSAMD::MetadataStreamerV2());
    } else if (isHsaAbiVersion3(getGlobalSTI())) {
      HSAMetadataStream.reset(new HSAMD::MetadataStreamerV3());
    } else if (isHsaAbiVersion5(getGlobalSTI())) {
      HSAMetadataStream.reset(new HSAMD::MetadataStreamerV5());
    } else {
      HSAMetadataStream.reset(new HSAMD::MetadataStreamerV4());
    }
  }
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
  IsTargetStreamerInitialized = false;
}

void AMDGPUAsmPrinter::initTargetStreamer(Module &M) {
  IsTargetStreamerInitialized = true;

  // TODO: Which one is called first, emitStartOfAsmFile or
  // emitFunctionBodyStart?
  if (getTargetStreamer() && !getTargetStreamer()->getTargetID())
    initializeTargetID(M);

  if (TM.getTargetTriple().getOS() != Triple::AMDHSA &&
      TM.getTargetTriple().getOS() != Triple::AMDPAL)
    return;

  if (isHsaAbiVersion3AndAbove(getGlobalSTI()))
    getTargetStreamer()->EmitDirectiveAMDGCNTarget();

  if (TM.getTargetTriple().getOS() == Triple::AMDHSA)
    HSAMetadataStream->begin(M, *getTargetStreamer()->getTargetID());

  if (TM.getTargetTriple().getOS() == Triple::AMDPAL)
    getTargetStreamer()->getPALMetadata()->readFromIR(M);

  if (isHsaAbiVersion3AndAbove(getGlobalSTI()))
    return;

  // HSA emits NT_AMD_HSA_CODE_OBJECT_VERSION for code objects v2.
  if (TM.getTargetTriple().getOS() == Triple::AMDHSA)
    getTargetStreamer()->EmitDirectiveHSACodeObjectVersion(2, 1);

  // HSA and PAL emit NT_AMD_HSA_ISA_VERSION for code objects v2.
  IsaVersion Version = getIsaVersion(getGlobalSTI()->getCPU());
  getTargetStreamer()->EmitDirectiveHSACodeObjectISAV2(
      Version.Major, Version.Minor, Version.Stepping, "AMD", "AMDGPU");
}

void AMDGPUAsmPrinter::emitEndOfAsmFile(Module &M) {
  // Init target streamer if it has not yet happened
  if (!IsTargetStreamerInitialized)
    initTargetStreamer(M);

  // Following code requires TargetStreamer to be present.
  if (!getTargetStreamer())
    return;

  if (TM.getTargetTriple().getOS() != Triple::AMDHSA ||
      isHsaAbiVersion2(getGlobalSTI()))
    getTargetStreamer()->EmitISAVersion();

  // Emit HSA Metadata (NT_AMD_AMDGPU_HSA_METADATA).
  // Emit HSA Metadata (NT_AMD_HSA_METADATA).
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
  const GCNSubtarget &STM = MF->getSubtarget<GCNSubtarget>();
  const Function &F = MF->getFunction();

  // TODO: Which one is called first, emitStartOfAsmFile or
  // emitFunctionBodyStart?
  if (getTargetStreamer() && !getTargetStreamer()->getTargetID())
    initializeTargetID(*F.getParent());

  const auto &FunctionTargetID = STM.getTargetID();
  // Make sure function's xnack settings are compatible with module's
  // xnack settings.
  if (FunctionTargetID.isXnackSupported() &&
      FunctionTargetID.getXnackSetting() != IsaInfo::TargetIDSetting::Any &&
      FunctionTargetID.getXnackSetting() != getTargetStreamer()->getTargetID()->getXnackSetting()) {
    OutContext.reportError({}, "xnack setting of '" + Twine(MF->getName()) +
                           "' function does not match module xnack setting");
    return;
  }
  // Make sure function's sramecc settings are compatible with module's
  // sramecc settings.
  if (FunctionTargetID.isSramEccSupported() &&
      FunctionTargetID.getSramEccSetting() != IsaInfo::TargetIDSetting::Any &&
      FunctionTargetID.getSramEccSetting() != getTargetStreamer()->getTargetID()->getSramEccSetting()) {
    OutContext.reportError({}, "sramecc setting of '" + Twine(MF->getName()) +
                           "' function does not match module sramecc setting");
    return;
  }

  if (!MFI.isEntryFunction())
    return;

  if ((STM.isMesaKernel(F) || isHsaAbiVersion2(getGlobalSTI())) &&
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

  if (TM.getTargetTriple().getOS() != Triple::AMDHSA ||
      isHsaAbiVersion2(getGlobalSTI()))
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

  const GCNSubtarget &STM = MF->getSubtarget<GCNSubtarget>();

  SmallString<128> KernelName;
  getNameWithPrefix(KernelName, &MF->getFunction());
  getTargetStreamer()->EmitAmdhsaKernelDescriptor(
      STM, KernelName, getAmdhsaKernelDescriptor(*MF, CurrentProgramInfo),
      CurrentProgramInfo.NumVGPRsForWavesPerEU,
      CurrentProgramInfo.NumSGPRsForWavesPerEU -
          IsaInfo::getNumExtraSGPRs(&STM,
                                    CurrentProgramInfo.VCCUsed,
                                    CurrentProgramInfo.FlatUsed),
      CurrentProgramInfo.VCCUsed, CurrentProgramInfo.FlatUsed);

  Streamer.PopSection();
}

void AMDGPUAsmPrinter::emitFunctionEntryLabel() {
  if (TM.getTargetTriple().getOS() == Triple::AMDHSA &&
      isHsaAbiVersion3AndAbove(getGlobalSTI())) {
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
    Align Alignment = GV->getAlign().getValueOr(Align(4));

    emitVisibility(GVSym, GV->getVisibility(), !GV->isDeclaration());
    emitLinkage(GV, GVSym);
    if (auto TS = getTargetStreamer())
      TS->emitAMDGPULDS(GVSym, Size, Alignment);
    return;
  }

  AsmPrinter::emitGlobalVariable(GV);
}

bool AMDGPUAsmPrinter::doFinalization(Module &M) {
  // Pad with s_code_end to help tools and guard against instruction prefetch
  // causing stale data in caches. Arguably this should be done by the linker,
  // which is why this isn't done for Mesa.
  const MCSubtargetInfo &STI = *getGlobalSTI();
  if ((AMDGPU::isGFX10Plus(STI) || AMDGPU::isGFX90A(STI)) &&
      (STI.getTargetTriple().getOS() == Triple::AMDHSA ||
       STI.getTargetTriple().getOS() == Triple::AMDPAL)) {
    OutStreamer->SwitchSection(getObjFileLowering().getTextSection());
    getTargetStreamer()->EmitCodeEnd(STI);
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
  const GCNSubtarget &STM = MF.getSubtarget<GCNSubtarget>();
  const Function &F = MF.getFunction();

  amdhsa::kernel_descriptor_t KernelDescriptor;
  memset(&KernelDescriptor, 0x0, sizeof(KernelDescriptor));

  assert(isUInt<32>(PI.ScratchSize));
  assert(isUInt<32>(PI.getComputePGMRSrc1()));
  assert(isUInt<32>(PI.ComputePGMRSrc2));

  KernelDescriptor.group_segment_fixed_size = PI.LDSSize;
  KernelDescriptor.private_segment_fixed_size = PI.ScratchSize;

  Align MaxKernArgAlign;
  KernelDescriptor.kernarg_size = STM.getKernArgSegmentSize(F, MaxKernArgAlign);

  KernelDescriptor.compute_pgm_rsrc1 = PI.getComputePGMRSrc1();
  KernelDescriptor.compute_pgm_rsrc2 = PI.ComputePGMRSrc2;
  KernelDescriptor.kernel_code_properties = getAmdhsaKernelCodeProperties(MF);

  assert(STM.hasGFX90AInsts() || CurrentProgramInfo.ComputePGMRSrc3GFX90A == 0);
  if (STM.hasGFX90AInsts())
    KernelDescriptor.compute_pgm_rsrc3 =
      CurrentProgramInfo.ComputePGMRSrc3GFX90A;

  return KernelDescriptor;
}

bool AMDGPUAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  // Init target streamer lazily on the first function so that previous passes
  // can set metadata.
  if (!IsTargetStreamerInitialized)
    initTargetStreamer(*MF.getFunction().getParent());

  ResourceUsage = &getAnalysis<AMDGPUResourceUsageAnalysis>();
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

  if (MFI->isModuleEntryFunction()) {
    getSIProgramInfo(CurrentProgramInfo, MF);
  }

  if (STM.isAmdPalOS()) {
    if (MFI->isEntryFunction())
      EmitPALMetadata(MF, CurrentProgramInfo);
    else if (MFI->isModuleEntryFunction())
      emitPALFunctionMetadata(MF);
  } else if (!STM.isAmdHsaOS()) {
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
      const AMDGPUResourceUsageAnalysis::SIFunctionResourceInfo &Info =
          ResourceUsage->getResourceInfo(&MF.getFunction());
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

    if (STM.hasGFX90AInsts())
      OutStreamer->emitRawComment(
        " AccumOffset: " +
        Twine((CurrentProgramInfo.AccumOffset + 1) * 4), false);

    OutStreamer->emitRawComment(
      " Occupancy: " +
      Twine(CurrentProgramInfo.Occupancy), false);

    OutStreamer->emitRawComment(
      " WaveLimiterHint : " + Twine(MFI->needsWaveLimiter()), false);

    OutStreamer->emitRawComment(
      " COMPUTE_PGM_RSRC2:SCRATCH_EN: " +
      Twine(G_00B84C_SCRATCH_EN(CurrentProgramInfo.ComputePGMRSrc2)), false);
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

    assert(STM.hasGFX90AInsts() ||
           CurrentProgramInfo.ComputePGMRSrc3GFX90A == 0);
    if (STM.hasGFX90AInsts()) {
      OutStreamer->emitRawComment(
        " COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: " +
        Twine((AMDHSA_BITS_GET(CurrentProgramInfo.ComputePGMRSrc3GFX90A,
                               amdhsa::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET))),
                               false);
      OutStreamer->emitRawComment(
        " COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: " +
        Twine((AMDHSA_BITS_GET(CurrentProgramInfo.ComputePGMRSrc3GFX90A,
                               amdhsa::COMPUTE_PGM_RSRC3_GFX90A_TG_SPLIT))),
                               false);
    }
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

// TODO: Fold this into emitFunctionBodyStart.
void AMDGPUAsmPrinter::initializeTargetID(const Module &M) {
  // In the beginning all features are either 'Any' or 'NotSupported',
  // depending on global target features. This will cover empty modules.
  getTargetStreamer()->initializeTargetID(
      *getGlobalSTI(), getGlobalSTI()->getFeatureString());

  // If module is empty, we are done.
  if (M.empty())
    return;

  // If module is not empty, need to find first 'Off' or 'On' feature
  // setting per feature from functions in module.
  for (auto &F : M) {
    auto &TSTargetID = getTargetStreamer()->getTargetID();
    if ((!TSTargetID->isXnackSupported() || TSTargetID->isXnackOnOrOff()) &&
        (!TSTargetID->isSramEccSupported() || TSTargetID->isSramEccOnOrOff()))
      break;

    const GCNSubtarget &STM = TM.getSubtarget<GCNSubtarget>(F);
    const IsaInfo::AMDGPUTargetID &STMTargetID = STM.getTargetID();
    if (TSTargetID->isXnackSupported())
      if (TSTargetID->getXnackSetting() == IsaInfo::TargetIDSetting::Any)
        TSTargetID->setXnackSetting(STMTargetID.getXnackSetting());
    if (TSTargetID->isSramEccSupported())
      if (TSTargetID->getSramEccSetting() == IsaInfo::TargetIDSetting::Any)
        TSTargetID->setSramEccSetting(STMTargetID.getSramEccSetting());
  }
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

void AMDGPUAsmPrinter::getSIProgramInfo(SIProgramInfo &ProgInfo,
                                        const MachineFunction &MF) {
  const AMDGPUResourceUsageAnalysis::SIFunctionResourceInfo &Info =
      ResourceUsage->getResourceInfo(&MF.getFunction());
  const GCNSubtarget &STM = MF.getSubtarget<GCNSubtarget>();

  ProgInfo.NumArchVGPR = Info.NumVGPR;
  ProgInfo.NumAccVGPR = Info.NumAGPR;
  ProgInfo.NumVGPR = Info.getTotalNumVGPRs(STM);
  ProgInfo.AccumOffset = alignTo(std::max(1, Info.NumVGPR), 4) / 4 - 1;
  ProgInfo.TgSplit = STM.isTgSplitEnabled();
  ProgInfo.NumSGPR = Info.NumExplicitSGPR;
  ProgInfo.ScratchSize = Info.PrivateSegmentSize;
  ProgInfo.VCCUsed = Info.UsesVCC;
  ProgInfo.FlatUsed = Info.UsesFlatScratch;
  ProgInfo.DynamicCallStack = Info.HasDynamicallySizedStack || Info.HasRecursion;

  const uint64_t MaxScratchPerWorkitem =
      GCNSubtarget::MaxWaveScratchSize / STM.getWavefrontSize();
  if (ProgInfo.ScratchSize > MaxScratchPerWorkitem) {
    DiagnosticInfoStackSize DiagStackSize(MF.getFunction(),
                                          ProgInfo.ScratchSize,
                                          MaxScratchPerWorkitem, DS_Error);
    MF.getFunction().getContext().diagnose(DiagStackSize);
  }

  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

  // The calculations related to SGPR/VGPR blocks are
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
      DiagnosticInfoResourceLimit Diag(
          MF.getFunction(), "addressable scalar registers", ProgInfo.NumSGPR,
          MaxAddressableNumSGPRs, DS_Error, DK_ResourceLimit);
      Ctx.diagnose(Diag);
      ProgInfo.NumSGPR = MaxAddressableNumSGPRs - 1;
    }
  }

  // Account for extra SGPRs and VGPRs reserved for debugger use.
  ProgInfo.NumSGPR += ExtraSGPRs;

  const Function &F = MF.getFunction();

  // Ensure there are enough SGPRs and VGPRs for wave dispatch, where wave
  // dispatch registers are function args.
  unsigned WaveDispatchNumSGPR = 0, WaveDispatchNumVGPR = 0;

  if (isShader(F.getCallingConv())) {
    bool IsPixelShader =
        F.getCallingConv() == CallingConv::AMDGPU_PS && !STM.isAmdHsaOS();

    // Calculate the number of VGPR registers based on the SPI input registers
    uint32_t InputEna = 0;
    uint32_t InputAddr = 0;
    unsigned LastEna = 0;

    if (IsPixelShader) {
      // Note for IsPixelShader:
      // By this stage, all enabled inputs are tagged in InputAddr as well.
      // We will use InputAddr to determine whether the input counts against the
      // vgpr total and only use the InputEnable to determine the last input
      // that is relevant - if extra arguments are used, then we have to honour
      // the InputAddr for any intermediate non-enabled inputs.
      InputEna = MFI->getPSInputEnable();
      InputAddr = MFI->getPSInputAddr();

      // We only need to consider input args up to the last used arg.
      assert((InputEna || InputAddr) &&
             "PSInputAddr and PSInputEnable should "
             "never both be 0 for AMDGPU_PS shaders");
      // There are some rare circumstances where InputAddr is non-zero and
      // InputEna can be set to 0. In this case we default to setting LastEna
      // to 1.
      LastEna = InputEna ? findLastSet(InputEna) + 1 : 1;
    }

    // FIXME: We should be using the number of registers determined during
    // calling convention lowering to legalize the types.
    const DataLayout &DL = F.getParent()->getDataLayout();
    unsigned PSArgCount = 0;
    unsigned IntermediateVGPR = 0;
    for (auto &Arg : F.args()) {
      unsigned NumRegs = (DL.getTypeSizeInBits(Arg.getType()) + 31) / 32;
      if (Arg.hasAttribute(Attribute::InReg)) {
        WaveDispatchNumSGPR += NumRegs;
      } else {
        // If this is a PS shader and we're processing the PS Input args (first
        // 16 VGPR), use the InputEna and InputAddr bits to define how many
        // VGPRs are actually used.
        // Any extra VGPR arguments are handled as normal arguments (and
        // contribute to the VGPR count whether they're used or not).
        if (IsPixelShader && PSArgCount < 16) {
          if ((1 << PSArgCount) & InputAddr) {
            if (PSArgCount < LastEna)
              WaveDispatchNumVGPR += NumRegs;
            else
              IntermediateVGPR += NumRegs;
          }
          PSArgCount++;
        } else {
          // If there are extra arguments we have to include the allocation for
          // the non-used (but enabled with InputAddr) input arguments
          if (IntermediateVGPR) {
            WaveDispatchNumVGPR += IntermediateVGPR;
            IntermediateVGPR = 0;
          }
          WaveDispatchNumVGPR += NumRegs;
        }
      }
    }
    ProgInfo.NumSGPR = std::max(ProgInfo.NumSGPR, WaveDispatchNumSGPR);
    ProgInfo.NumArchVGPR = std::max(ProgInfo.NumVGPR, WaveDispatchNumVGPR);
    ProgInfo.NumVGPR =
        Info.getTotalNumVGPRs(STM, Info.NumAGPR, ProgInfo.NumArchVGPR);
  }

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
      DiagnosticInfoResourceLimit Diag(MF.getFunction(), "scalar registers",
                                       ProgInfo.NumSGPR, MaxAddressableNumSGPRs,
                                       DS_Error, DK_ResourceLimit);
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
                                     MFI->getNumUserSGPRs(),
                                     STM.getMaxNumUserSGPRs(), DS_Error);
    Ctx.diagnose(Diag);
  }

  if (MFI->getLDSSize() > static_cast<unsigned>(STM.getLocalMemorySize())) {
    LLVMContext &Ctx = MF.getFunction().getContext();
    DiagnosticInfoResourceLimit Diag(MF.getFunction(), "local memory",
                                     MFI->getLDSSize(),
                                     STM.getLocalMemorySize(), DS_Error);
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

  if (STM.hasGFX90AInsts()) {
    AMDHSA_BITS_SET(ProgInfo.ComputePGMRSrc3GFX90A,
                    amdhsa::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET,
                    ProgInfo.AccumOffset);
    AMDHSA_BITS_SET(ProgInfo.ComputePGMRSrc3GFX90A,
                    amdhsa::COMPUTE_PGM_RSRC3_GFX90A_TG_SPLIT,
                    ProgInfo.TgSplit);
  }

  ProgInfo.Occupancy = STM.computeOccupancy(MF.getFunction(), ProgInfo.LDSSize,
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

    OutStreamer->emitInt32(CurrentProgramInfo.getComputePGMRSrc1());

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
  MD->setRsrc1(CC, CurrentProgramInfo.getPGMRSrc1(CC));
  if (AMDGPU::isCompute(CC)) {
    MD->setRsrc2(CC, CurrentProgramInfo.ComputePGMRSrc2);
  } else {
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

void AMDGPUAsmPrinter::emitPALFunctionMetadata(const MachineFunction &MF) {
  auto *MD = getTargetStreamer()->getPALMetadata();
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  MD->setFunctionScratchSize(MF, MFI.getStackSize());

  // Set compute registers
  MD->setRsrc1(CallingConv::AMDGPU_CS,
               CurrentProgramInfo.getPGMRSrc1(CallingConv::AMDGPU_CS));
  MD->setRsrc2(CallingConv::AMDGPU_CS, CurrentProgramInfo.ComputePGMRSrc2);

  // Set optional info
  MD->setFunctionLdsSize(MF, CurrentProgramInfo.LDSSize);
  MD->setFunctionNumUsedVgprs(MF, CurrentProgramInfo.NumVGPRsForWavesPerEU);
  MD->setFunctionNumUsedSgprs(MF, CurrentProgramInfo.NumSGPRsForWavesPerEU);
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
      CurrentProgramInfo.getComputePGMRSrc1() |
      (CurrentProgramInfo.ComputePGMRSrc2 << 32);
  Out.code_properties |= AMD_CODE_PROPERTY_IS_PTR64;

  if (CurrentProgramInfo.DynamicCallStack)
    Out.code_properties |= AMD_CODE_PROPERTY_IS_DYNAMIC_CALLSTACK;

  AMD_HSA_BITS_SET(Out.code_properties,
                   AMD_CODE_PROPERTY_PRIVATE_ELEMENT_SIZE,
                   getElementByteSizeValue(STM.getMaxPrivateElementSize(true)));

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
  // FIXME: The metadata treats the minimum as 4?
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
  } else if (MO.isImm()) {
    int64_t Val = MO.getImm();
    if (AMDGPU::isInlinableIntLiteral(Val)) {
      O << Val;
    } else if (isUInt<16>(Val)) {
      O << format("0x%" PRIx16, static_cast<uint16_t>(Val));
    } else if (isUInt<32>(Val)) {
      O << format("0x%" PRIx32, static_cast<uint32_t>(Val));
    } else {
      O << format("0x%" PRIx64, static_cast<uint64_t>(Val));
    }
    return false;
  }
  return true;
}

void AMDGPUAsmPrinter::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<AMDGPUResourceUsageAnalysis>();
  AU.addPreserved<AMDGPUResourceUsageAnalysis>();
  AsmPrinter::getAnalysisUsage(AU);
}
