//===- AMDGPUBaseInfo.cpp - AMDGPU Base encoding information --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUBaseInfo.h"
#include "AMDGPU.h"
#include "AMDGPUAsmUtils.h"
#include "AMDKernelCodeT.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsR600.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetParser.h"

#define GET_INSTRINFO_NAMED_OPS
#define GET_INSTRMAP_INFO
#include "AMDGPUGenInstrInfo.inc"

static llvm::cl::opt<unsigned>
    AmdhsaCodeObjectVersion("amdhsa-code-object-version", llvm::cl::Hidden,
                            llvm::cl::desc("AMDHSA Code Object Version"),
                            llvm::cl::init(4));

// TODO-GFX11: Remove this when full 16-bit codegen is implemented.
static llvm::cl::opt<bool>
    LimitTo128VGPRs("amdgpu-limit-to-128-vgprs", llvm::cl::Hidden,
                    llvm::cl::desc("Never use more than 128 VGPRs"));

namespace {

/// \returns Bit mask for given bit \p Shift and bit \p Width.
unsigned getBitMask(unsigned Shift, unsigned Width) {
  return ((1 << Width) - 1) << Shift;
}

/// Packs \p Src into \p Dst for given bit \p Shift and bit \p Width.
///
/// \returns Packed \p Dst.
unsigned packBits(unsigned Src, unsigned Dst, unsigned Shift, unsigned Width) {
  unsigned Mask = getBitMask(Shift, Width);
  return ((Src << Shift) & Mask) | (Dst & ~Mask);
}

/// Unpacks bits from \p Src for given bit \p Shift and bit \p Width.
///
/// \returns Unpacked bits.
unsigned unpackBits(unsigned Src, unsigned Shift, unsigned Width) {
  return (Src & getBitMask(Shift, Width)) >> Shift;
}

/// \returns Vmcnt bit shift (lower bits).
unsigned getVmcntBitShiftLo(unsigned VersionMajor) {
  return VersionMajor >= 11 ? 10 : 0;
}

/// \returns Vmcnt bit width (lower bits).
unsigned getVmcntBitWidthLo(unsigned VersionMajor) {
  return VersionMajor >= 11 ? 6 : 4;
}

/// \returns Expcnt bit shift.
unsigned getExpcntBitShift(unsigned VersionMajor) {
  return VersionMajor >= 11 ? 0 : 4;
}

/// \returns Expcnt bit width.
unsigned getExpcntBitWidth(unsigned VersionMajor) { return 3; }

/// \returns Lgkmcnt bit shift.
unsigned getLgkmcntBitShift(unsigned VersionMajor) {
  return VersionMajor >= 11 ? 4 : 8;
}

/// \returns Lgkmcnt bit width.
unsigned getLgkmcntBitWidth(unsigned VersionMajor) {
  return VersionMajor >= 10 ? 6 : 4;
}

/// \returns Vmcnt bit shift (higher bits).
unsigned getVmcntBitShiftHi(unsigned VersionMajor) { return 14; }

/// \returns Vmcnt bit width (higher bits).
unsigned getVmcntBitWidthHi(unsigned VersionMajor) {
  return (VersionMajor == 9 || VersionMajor == 10) ? 2 : 0;
}

} // end namespace anonymous

namespace llvm {

namespace AMDGPU {

Optional<uint8_t> getHsaAbiVersion(const MCSubtargetInfo *STI) {
  if (STI && STI->getTargetTriple().getOS() != Triple::AMDHSA)
    return None;

  switch (AmdhsaCodeObjectVersion) {
  case 2:
    return ELF::ELFABIVERSION_AMDGPU_HSA_V2;
  case 3:
    return ELF::ELFABIVERSION_AMDGPU_HSA_V3;
  case 4:
    return ELF::ELFABIVERSION_AMDGPU_HSA_V4;
  case 5:
    return ELF::ELFABIVERSION_AMDGPU_HSA_V5;
  default:
    report_fatal_error(Twine("Unsupported AMDHSA Code Object Version ") +
                       Twine(AmdhsaCodeObjectVersion));
  }
}

bool isHsaAbiVersion2(const MCSubtargetInfo *STI) {
  if (Optional<uint8_t> HsaAbiVer = getHsaAbiVersion(STI))
    return *HsaAbiVer == ELF::ELFABIVERSION_AMDGPU_HSA_V2;
  return false;
}

bool isHsaAbiVersion3(const MCSubtargetInfo *STI) {
  if (Optional<uint8_t> HsaAbiVer = getHsaAbiVersion(STI))
    return *HsaAbiVer == ELF::ELFABIVERSION_AMDGPU_HSA_V3;
  return false;
}

bool isHsaAbiVersion4(const MCSubtargetInfo *STI) {
  if (Optional<uint8_t> HsaAbiVer = getHsaAbiVersion(STI))
    return *HsaAbiVer == ELF::ELFABIVERSION_AMDGPU_HSA_V4;
  return false;
}

bool isHsaAbiVersion5(const MCSubtargetInfo *STI) {
  if (Optional<uint8_t> HsaAbiVer = getHsaAbiVersion(STI))
    return *HsaAbiVer == ELF::ELFABIVERSION_AMDGPU_HSA_V5;
  return false;
}

bool isHsaAbiVersion3AndAbove(const MCSubtargetInfo *STI) {
  return isHsaAbiVersion3(STI) || isHsaAbiVersion4(STI) ||
         isHsaAbiVersion5(STI);
}

unsigned getAmdhsaCodeObjectVersion() {
  return AmdhsaCodeObjectVersion;
}

unsigned getMultigridSyncArgImplicitArgPosition() {
  switch (AmdhsaCodeObjectVersion) {
  case 2:
  case 3:
  case 4:
    return 48;
  case 5:
    return AMDGPU::ImplicitArg::MULTIGRID_SYNC_ARG_OFFSET;
  default:
    llvm_unreachable("Unexpected code object version");
    return 0;
  }
}


// FIXME: All such magic numbers about the ABI should be in a
// central TD file.
unsigned getHostcallImplicitArgPosition() {
  switch (AmdhsaCodeObjectVersion) {
  case 2:
  case 3:
  case 4:
    return 24;
  case 5:
    return AMDGPU::ImplicitArg::HOSTCALL_PTR_OFFSET;
  default:
    llvm_unreachable("Unexpected code object version");
    return 0;
  }
}

#define GET_MIMGBaseOpcodesTable_IMPL
#define GET_MIMGDimInfoTable_IMPL
#define GET_MIMGInfoTable_IMPL
#define GET_MIMGLZMappingTable_IMPL
#define GET_MIMGMIPMappingTable_IMPL
#define GET_MIMGBiasMappingTable_IMPL
#define GET_MIMGOffsetMappingTable_IMPL
#define GET_MIMGG16MappingTable_IMPL
#define GET_MAIInstInfoTable_IMPL
#include "AMDGPUGenSearchableTables.inc"

int getMIMGOpcode(unsigned BaseOpcode, unsigned MIMGEncoding,
                  unsigned VDataDwords, unsigned VAddrDwords) {
  const MIMGInfo *Info = getMIMGOpcodeHelper(BaseOpcode, MIMGEncoding,
                                             VDataDwords, VAddrDwords);
  return Info ? Info->Opcode : -1;
}

const MIMGBaseOpcodeInfo *getMIMGBaseOpcode(unsigned Opc) {
  const MIMGInfo *Info = getMIMGInfo(Opc);
  return Info ? getMIMGBaseOpcodeInfo(Info->BaseOpcode) : nullptr;
}

int getMaskedMIMGOp(unsigned Opc, unsigned NewChannels) {
  const MIMGInfo *OrigInfo = getMIMGInfo(Opc);
  const MIMGInfo *NewInfo =
      getMIMGOpcodeHelper(OrigInfo->BaseOpcode, OrigInfo->MIMGEncoding,
                          NewChannels, OrigInfo->VAddrDwords);
  return NewInfo ? NewInfo->Opcode : -1;
}

unsigned getAddrSizeMIMGOp(const MIMGBaseOpcodeInfo *BaseOpcode,
                           const MIMGDimInfo *Dim, bool IsA16,
                           bool IsG16Supported) {
  unsigned AddrWords = BaseOpcode->NumExtraArgs;
  unsigned AddrComponents = (BaseOpcode->Coordinates ? Dim->NumCoords : 0) +
                            (BaseOpcode->LodOrClampOrMip ? 1 : 0);
  if (IsA16)
    AddrWords += divideCeil(AddrComponents, 2);
  else
    AddrWords += AddrComponents;

  // Note: For subtargets that support A16 but not G16, enabling A16 also
  // enables 16 bit gradients.
  // For subtargets that support A16 (operand) and G16 (done with a different
  // instruction encoding), they are independent.

  if (BaseOpcode->Gradients) {
    if ((IsA16 && !IsG16Supported) || BaseOpcode->G16)
      // There are two gradients per coordinate, we pack them separately.
      // For the 3d case,
      // we get (dy/du, dx/du) (-, dz/du) (dy/dv, dx/dv) (-, dz/dv)
      AddrWords += alignTo<2>(Dim->NumGradients / 2);
    else
      AddrWords += Dim->NumGradients;
  }
  return AddrWords;
}

struct MUBUFInfo {
  uint16_t Opcode;
  uint16_t BaseOpcode;
  uint8_t elements;
  bool has_vaddr;
  bool has_srsrc;
  bool has_soffset;
  bool IsBufferInv;
};

struct MTBUFInfo {
  uint16_t Opcode;
  uint16_t BaseOpcode;
  uint8_t elements;
  bool has_vaddr;
  bool has_srsrc;
  bool has_soffset;
};

struct SMInfo {
  uint16_t Opcode;
  bool IsBuffer;
};

struct VOPInfo {
  uint16_t Opcode;
  bool IsSingle;
};

struct VOPC64DPPInfo {
  uint16_t Opcode;
};

#define GET_MTBUFInfoTable_DECL
#define GET_MTBUFInfoTable_IMPL
#define GET_MUBUFInfoTable_DECL
#define GET_MUBUFInfoTable_IMPL
#define GET_SMInfoTable_DECL
#define GET_SMInfoTable_IMPL
#define GET_VOP1InfoTable_DECL
#define GET_VOP1InfoTable_IMPL
#define GET_VOP2InfoTable_DECL
#define GET_VOP2InfoTable_IMPL
#define GET_VOP3InfoTable_DECL
#define GET_VOP3InfoTable_IMPL
#define GET_VOPC64DPPTable_DECL
#define GET_VOPC64DPPTable_IMPL
#define GET_VOPC64DPP8Table_DECL
#define GET_VOPC64DPP8Table_IMPL
#define GET_WMMAOpcode2AddrMappingTable_DECL
#define GET_WMMAOpcode2AddrMappingTable_IMPL
#define GET_WMMAOpcode3AddrMappingTable_DECL
#define GET_WMMAOpcode3AddrMappingTable_IMPL
#include "AMDGPUGenSearchableTables.inc"

int getMTBUFBaseOpcode(unsigned Opc) {
  const MTBUFInfo *Info = getMTBUFInfoFromOpcode(Opc);
  return Info ? Info->BaseOpcode : -1;
}

int getMTBUFOpcode(unsigned BaseOpc, unsigned Elements) {
  const MTBUFInfo *Info = getMTBUFInfoFromBaseOpcodeAndElements(BaseOpc, Elements);
  return Info ? Info->Opcode : -1;
}

int getMTBUFElements(unsigned Opc) {
  const MTBUFInfo *Info = getMTBUFOpcodeHelper(Opc);
  return Info ? Info->elements : 0;
}

bool getMTBUFHasVAddr(unsigned Opc) {
  const MTBUFInfo *Info = getMTBUFOpcodeHelper(Opc);
  return Info ? Info->has_vaddr : false;
}

bool getMTBUFHasSrsrc(unsigned Opc) {
  const MTBUFInfo *Info = getMTBUFOpcodeHelper(Opc);
  return Info ? Info->has_srsrc : false;
}

bool getMTBUFHasSoffset(unsigned Opc) {
  const MTBUFInfo *Info = getMTBUFOpcodeHelper(Opc);
  return Info ? Info->has_soffset : false;
}

int getMUBUFBaseOpcode(unsigned Opc) {
  const MUBUFInfo *Info = getMUBUFInfoFromOpcode(Opc);
  return Info ? Info->BaseOpcode : -1;
}

int getMUBUFOpcode(unsigned BaseOpc, unsigned Elements) {
  const MUBUFInfo *Info = getMUBUFInfoFromBaseOpcodeAndElements(BaseOpc, Elements);
  return Info ? Info->Opcode : -1;
}

int getMUBUFElements(unsigned Opc) {
  const MUBUFInfo *Info = getMUBUFOpcodeHelper(Opc);
  return Info ? Info->elements : 0;
}

bool getMUBUFHasVAddr(unsigned Opc) {
  const MUBUFInfo *Info = getMUBUFOpcodeHelper(Opc);
  return Info ? Info->has_vaddr : false;
}

bool getMUBUFHasSrsrc(unsigned Opc) {
  const MUBUFInfo *Info = getMUBUFOpcodeHelper(Opc);
  return Info ? Info->has_srsrc : false;
}

bool getMUBUFHasSoffset(unsigned Opc) {
  const MUBUFInfo *Info = getMUBUFOpcodeHelper(Opc);
  return Info ? Info->has_soffset : false;
}

bool getMUBUFIsBufferInv(unsigned Opc) {
  const MUBUFInfo *Info = getMUBUFOpcodeHelper(Opc);
  return Info ? Info->IsBufferInv : false;
}

bool getSMEMIsBuffer(unsigned Opc) {
  const SMInfo *Info = getSMEMOpcodeHelper(Opc);
  return Info ? Info->IsBuffer : false;
}

bool getVOP1IsSingle(unsigned Opc) {
  const VOPInfo *Info = getVOP1OpcodeHelper(Opc);
  return Info ? Info->IsSingle : false;
}

bool getVOP2IsSingle(unsigned Opc) {
  const VOPInfo *Info = getVOP2OpcodeHelper(Opc);
  return Info ? Info->IsSingle : false;
}

bool getVOP3IsSingle(unsigned Opc) {
  const VOPInfo *Info = getVOP3OpcodeHelper(Opc);
  return Info ? Info->IsSingle : false;
}

bool isVOPC64DPP(unsigned Opc) {
  return isVOPC64DPPOpcodeHelper(Opc) || isVOPC64DPP8OpcodeHelper(Opc);
}

bool getMAIIsDGEMM(unsigned Opc) {
  const MAIInstInfo *Info = getMAIInstInfoHelper(Opc);
  return Info ? Info->is_dgemm : false;
}

bool getMAIIsGFX940XDL(unsigned Opc) {
  const MAIInstInfo *Info = getMAIInstInfoHelper(Opc);
  return Info ? Info->is_gfx940_xdl : false;
}

unsigned mapWMMA2AddrTo3AddrOpcode(unsigned Opc) {
  const WMMAOpcodeMappingInfo *Info = getWMMAMappingInfoFrom2AddrOpcode(Opc);
  return Info ? Info->Opcode3Addr : ~0u;
}

unsigned mapWMMA3AddrTo2AddrOpcode(unsigned Opc) {
  const WMMAOpcodeMappingInfo *Info = getWMMAMappingInfoFrom3AddrOpcode(Opc);
  return Info ? Info->Opcode2Addr : ~0u;
}

// Wrapper for Tablegen'd function.  enum Subtarget is not defined in any
// header files, so we need to wrap it in a function that takes unsigned
// instead.
int getMCOpcode(uint16_t Opcode, unsigned Gen) {
  return getMCOpcodeGen(Opcode, static_cast<Subtarget>(Gen));
}

namespace IsaInfo {

AMDGPUTargetID::AMDGPUTargetID(const MCSubtargetInfo &STI)
    : STI(STI), XnackSetting(TargetIDSetting::Any),
      SramEccSetting(TargetIDSetting::Any) {
  if (!STI.getFeatureBits().test(FeatureSupportsXNACK))
    XnackSetting = TargetIDSetting::Unsupported;
  if (!STI.getFeatureBits().test(FeatureSupportsSRAMECC))
    SramEccSetting = TargetIDSetting::Unsupported;
}

void AMDGPUTargetID::setTargetIDFromFeaturesString(StringRef FS) {
  // Check if xnack or sramecc is explicitly enabled or disabled.  In the
  // absence of the target features we assume we must generate code that can run
  // in any environment.
  SubtargetFeatures Features(FS);
  Optional<bool> XnackRequested;
  Optional<bool> SramEccRequested;

  for (const std::string &Feature : Features.getFeatures()) {
    if (Feature == "+xnack")
      XnackRequested = true;
    else if (Feature == "-xnack")
      XnackRequested = false;
    else if (Feature == "+sramecc")
      SramEccRequested = true;
    else if (Feature == "-sramecc")
      SramEccRequested = false;
  }

  bool XnackSupported = isXnackSupported();
  bool SramEccSupported = isSramEccSupported();

  if (XnackRequested) {
    if (XnackSupported) {
      XnackSetting =
          *XnackRequested ? TargetIDSetting::On : TargetIDSetting::Off;
    } else {
      // If a specific xnack setting was requested and this GPU does not support
      // xnack emit a warning. Setting will remain set to "Unsupported".
      if (*XnackRequested) {
        errs() << "warning: xnack 'On' was requested for a processor that does "
                  "not support it!\n";
      } else {
        errs() << "warning: xnack 'Off' was requested for a processor that "
                  "does not support it!\n";
      }
    }
  }

  if (SramEccRequested) {
    if (SramEccSupported) {
      SramEccSetting =
          *SramEccRequested ? TargetIDSetting::On : TargetIDSetting::Off;
    } else {
      // If a specific sramecc setting was requested and this GPU does not
      // support sramecc emit a warning. Setting will remain set to
      // "Unsupported".
      if (*SramEccRequested) {
        errs() << "warning: sramecc 'On' was requested for a processor that "
                  "does not support it!\n";
      } else {
        errs() << "warning: sramecc 'Off' was requested for a processor that "
                  "does not support it!\n";
      }
    }
  }
}

static TargetIDSetting
getTargetIDSettingFromFeatureString(StringRef FeatureString) {
  if (FeatureString.endswith("-"))
    return TargetIDSetting::Off;
  if (FeatureString.endswith("+"))
    return TargetIDSetting::On;

  llvm_unreachable("Malformed feature string");
}

void AMDGPUTargetID::setTargetIDFromTargetIDStream(StringRef TargetID) {
  SmallVector<StringRef, 3> TargetIDSplit;
  TargetID.split(TargetIDSplit, ':');

  for (const auto &FeatureString : TargetIDSplit) {
    if (FeatureString.startswith("xnack"))
      XnackSetting = getTargetIDSettingFromFeatureString(FeatureString);
    if (FeatureString.startswith("sramecc"))
      SramEccSetting = getTargetIDSettingFromFeatureString(FeatureString);
  }
}

std::string AMDGPUTargetID::toString() const {
  std::string StringRep;
  raw_string_ostream StreamRep(StringRep);

  auto TargetTriple = STI.getTargetTriple();
  auto Version = getIsaVersion(STI.getCPU());

  StreamRep << TargetTriple.getArchName() << '-'
            << TargetTriple.getVendorName() << '-'
            << TargetTriple.getOSName() << '-'
            << TargetTriple.getEnvironmentName() << '-';

  std::string Processor;
  // TODO: Following else statement is present here because we used various
  // alias names for GPUs up until GFX9 (e.g. 'fiji' is same as 'gfx803').
  // Remove once all aliases are removed from GCNProcessors.td.
  if (Version.Major >= 9)
    Processor = STI.getCPU().str();
  else
    Processor = (Twine("gfx") + Twine(Version.Major) + Twine(Version.Minor) +
                 Twine(Version.Stepping))
                    .str();

  std::string Features;
  if (Optional<uint8_t> HsaAbiVersion = getHsaAbiVersion(&STI)) {
    switch (*HsaAbiVersion) {
    case ELF::ELFABIVERSION_AMDGPU_HSA_V2:
      // Code object V2 only supported specific processors and had fixed
      // settings for the XNACK.
      if (Processor == "gfx600") {
      } else if (Processor == "gfx601") {
      } else if (Processor == "gfx602") {
      } else if (Processor == "gfx700") {
      } else if (Processor == "gfx701") {
      } else if (Processor == "gfx702") {
      } else if (Processor == "gfx703") {
      } else if (Processor == "gfx704") {
      } else if (Processor == "gfx705") {
      } else if (Processor == "gfx801") {
        if (!isXnackOnOrAny())
          report_fatal_error(
              "AMD GPU code object V2 does not support processor " +
              Twine(Processor) + " without XNACK");
      } else if (Processor == "gfx802") {
      } else if (Processor == "gfx803") {
      } else if (Processor == "gfx805") {
      } else if (Processor == "gfx810") {
        if (!isXnackOnOrAny())
          report_fatal_error(
              "AMD GPU code object V2 does not support processor " +
              Twine(Processor) + " without XNACK");
      } else if (Processor == "gfx900") {
        if (isXnackOnOrAny())
          Processor = "gfx901";
      } else if (Processor == "gfx902") {
        if (isXnackOnOrAny())
          Processor = "gfx903";
      } else if (Processor == "gfx904") {
        if (isXnackOnOrAny())
          Processor = "gfx905";
      } else if (Processor == "gfx906") {
        if (isXnackOnOrAny())
          Processor = "gfx907";
      } else if (Processor == "gfx90c") {
        if (isXnackOnOrAny())
          report_fatal_error(
              "AMD GPU code object V2 does not support processor " +
              Twine(Processor) + " with XNACK being ON or ANY");
      } else {
        report_fatal_error(
            "AMD GPU code object V2 does not support processor " +
            Twine(Processor));
      }
      break;
    case ELF::ELFABIVERSION_AMDGPU_HSA_V3:
      // xnack.
      if (isXnackOnOrAny())
        Features += "+xnack";
      // In code object v2 and v3, "sramecc" feature was spelled with a
      // hyphen ("sram-ecc").
      if (isSramEccOnOrAny())
        Features += "+sram-ecc";
      break;
    case ELF::ELFABIVERSION_AMDGPU_HSA_V4:
    case ELF::ELFABIVERSION_AMDGPU_HSA_V5:
      // sramecc.
      if (getSramEccSetting() == TargetIDSetting::Off)
        Features += ":sramecc-";
      else if (getSramEccSetting() == TargetIDSetting::On)
        Features += ":sramecc+";
      // xnack.
      if (getXnackSetting() == TargetIDSetting::Off)
        Features += ":xnack-";
      else if (getXnackSetting() == TargetIDSetting::On)
        Features += ":xnack+";
      break;
    default:
      break;
    }
  }

  StreamRep << Processor << Features;

  StreamRep.flush();
  return StringRep;
}

unsigned getWavefrontSize(const MCSubtargetInfo *STI) {
  if (STI->getFeatureBits().test(FeatureWavefrontSize16))
    return 16;
  if (STI->getFeatureBits().test(FeatureWavefrontSize32))
    return 32;

  return 64;
}

unsigned getLocalMemorySize(const MCSubtargetInfo *STI) {
  if (STI->getFeatureBits().test(FeatureLocalMemorySize32768))
    return 32768;
  if (STI->getFeatureBits().test(FeatureLocalMemorySize65536))
    return 65536;

  return 0;
}

unsigned getEUsPerCU(const MCSubtargetInfo *STI) {
  // "Per CU" really means "per whatever functional block the waves of a
  // workgroup must share". For gfx10 in CU mode this is the CU, which contains
  // two SIMDs.
  if (isGFX10Plus(*STI) && STI->getFeatureBits().test(FeatureCuMode))
    return 2;
  // Pre-gfx10 a CU contains four SIMDs. For gfx10 in WGP mode the WGP contains
  // two CUs, so a total of four SIMDs.
  return 4;
}

unsigned getMaxWorkGroupsPerCU(const MCSubtargetInfo *STI,
                               unsigned FlatWorkGroupSize) {
  assert(FlatWorkGroupSize != 0);
  if (STI->getTargetTriple().getArch() != Triple::amdgcn)
    return 8;
  unsigned N = getWavesPerWorkGroup(STI, FlatWorkGroupSize);
  if (N == 1)
    return 40;
  N = 40 / N;
  return std::min(N, 16u);
}

unsigned getMinWavesPerEU(const MCSubtargetInfo *STI) {
  return 1;
}

unsigned getMaxWavesPerEU(const MCSubtargetInfo *STI) {
  // FIXME: Need to take scratch memory into account.
  if (isGFX90A(*STI))
    return 8;
  if (!isGFX10Plus(*STI))
    return 10;
  return hasGFX10_3Insts(*STI) ? 16 : 20;
}

unsigned getWavesPerEUForWorkGroup(const MCSubtargetInfo *STI,
                                   unsigned FlatWorkGroupSize) {
  return divideCeil(getWavesPerWorkGroup(STI, FlatWorkGroupSize),
                    getEUsPerCU(STI));
}

unsigned getMinFlatWorkGroupSize(const MCSubtargetInfo *STI) {
  return 1;
}

unsigned getMaxFlatWorkGroupSize(const MCSubtargetInfo *STI) {
  // Some subtargets allow encoding 2048, but this isn't tested or supported.
  return 1024;
}

unsigned getWavesPerWorkGroup(const MCSubtargetInfo *STI,
                              unsigned FlatWorkGroupSize) {
  return divideCeil(FlatWorkGroupSize, getWavefrontSize(STI));
}

unsigned getSGPRAllocGranule(const MCSubtargetInfo *STI) {
  IsaVersion Version = getIsaVersion(STI->getCPU());
  if (Version.Major >= 10)
    return getAddressableNumSGPRs(STI);
  if (Version.Major >= 8)
    return 16;
  return 8;
}

unsigned getSGPREncodingGranule(const MCSubtargetInfo *STI) {
  return 8;
}

unsigned getTotalNumSGPRs(const MCSubtargetInfo *STI) {
  IsaVersion Version = getIsaVersion(STI->getCPU());
  if (Version.Major >= 8)
    return 800;
  return 512;
}

unsigned getAddressableNumSGPRs(const MCSubtargetInfo *STI) {
  if (STI->getFeatureBits().test(FeatureSGPRInitBug))
    return FIXED_NUM_SGPRS_FOR_INIT_BUG;

  IsaVersion Version = getIsaVersion(STI->getCPU());
  if (Version.Major >= 10)
    return 106;
  if (Version.Major >= 8)
    return 102;
  return 104;
}

unsigned getMinNumSGPRs(const MCSubtargetInfo *STI, unsigned WavesPerEU) {
  assert(WavesPerEU != 0);

  IsaVersion Version = getIsaVersion(STI->getCPU());
  if (Version.Major >= 10)
    return 0;

  if (WavesPerEU >= getMaxWavesPerEU(STI))
    return 0;

  unsigned MinNumSGPRs = getTotalNumSGPRs(STI) / (WavesPerEU + 1);
  if (STI->getFeatureBits().test(FeatureTrapHandler))
    MinNumSGPRs -= std::min(MinNumSGPRs, (unsigned)TRAP_NUM_SGPRS);
  MinNumSGPRs = alignDown(MinNumSGPRs, getSGPRAllocGranule(STI)) + 1;
  return std::min(MinNumSGPRs, getAddressableNumSGPRs(STI));
}

unsigned getMaxNumSGPRs(const MCSubtargetInfo *STI, unsigned WavesPerEU,
                        bool Addressable) {
  assert(WavesPerEU != 0);

  unsigned AddressableNumSGPRs = getAddressableNumSGPRs(STI);
  IsaVersion Version = getIsaVersion(STI->getCPU());
  if (Version.Major >= 10)
    return Addressable ? AddressableNumSGPRs : 108;
  if (Version.Major >= 8 && !Addressable)
    AddressableNumSGPRs = 112;
  unsigned MaxNumSGPRs = getTotalNumSGPRs(STI) / WavesPerEU;
  if (STI->getFeatureBits().test(FeatureTrapHandler))
    MaxNumSGPRs -= std::min(MaxNumSGPRs, (unsigned)TRAP_NUM_SGPRS);
  MaxNumSGPRs = alignDown(MaxNumSGPRs, getSGPRAllocGranule(STI));
  return std::min(MaxNumSGPRs, AddressableNumSGPRs);
}

unsigned getNumExtraSGPRs(const MCSubtargetInfo *STI, bool VCCUsed,
                          bool FlatScrUsed, bool XNACKUsed) {
  unsigned ExtraSGPRs = 0;
  if (VCCUsed)
    ExtraSGPRs = 2;

  IsaVersion Version = getIsaVersion(STI->getCPU());
  if (Version.Major >= 10)
    return ExtraSGPRs;

  if (Version.Major < 8) {
    if (FlatScrUsed)
      ExtraSGPRs = 4;
  } else {
    if (XNACKUsed)
      ExtraSGPRs = 4;

    if (FlatScrUsed ||
        STI->getFeatureBits().test(AMDGPU::FeatureArchitectedFlatScratch))
      ExtraSGPRs = 6;
  }

  return ExtraSGPRs;
}

unsigned getNumExtraSGPRs(const MCSubtargetInfo *STI, bool VCCUsed,
                          bool FlatScrUsed) {
  return getNumExtraSGPRs(STI, VCCUsed, FlatScrUsed,
                          STI->getFeatureBits().test(AMDGPU::FeatureXNACK));
}

unsigned getNumSGPRBlocks(const MCSubtargetInfo *STI, unsigned NumSGPRs) {
  NumSGPRs = alignTo(std::max(1u, NumSGPRs), getSGPREncodingGranule(STI));
  // SGPRBlocks is actual number of SGPR blocks minus 1.
  return NumSGPRs / getSGPREncodingGranule(STI) - 1;
}

unsigned getVGPRAllocGranule(const MCSubtargetInfo *STI,
                             Optional<bool> EnableWavefrontSize32) {
  if (STI->getFeatureBits().test(FeatureGFX90AInsts))
    return 8;

  bool IsWave32 = EnableWavefrontSize32 ?
      *EnableWavefrontSize32 :
      STI->getFeatureBits().test(FeatureWavefrontSize32);

  if (hasGFX10_3Insts(*STI))
    return IsWave32 ? 16 : 8;

  return IsWave32 ? 8 : 4;
}

unsigned getVGPREncodingGranule(const MCSubtargetInfo *STI,
                                Optional<bool> EnableWavefrontSize32) {
  if (STI->getFeatureBits().test(FeatureGFX90AInsts))
    return 8;

  bool IsWave32 = EnableWavefrontSize32 ?
      *EnableWavefrontSize32 :
      STI->getFeatureBits().test(FeatureWavefrontSize32);

  return IsWave32 ? 8 : 4;
}

unsigned getTotalNumVGPRs(const MCSubtargetInfo *STI) {
  if (STI->getFeatureBits().test(FeatureGFX90AInsts))
    return 512;
  if (!isGFX10Plus(*STI))
    return 256;
  return STI->getFeatureBits().test(FeatureWavefrontSize32) ? 1024 : 512;
}

unsigned getAddressableNumVGPRs(const MCSubtargetInfo *STI) {
  if (LimitTo128VGPRs.getNumOccurrences() ? LimitTo128VGPRs
                                          : isGFX11Plus(*STI)) {
    // GFX11 changes the encoding of 16-bit operands in VOP1/2/C instructions
    // such that values 128..255 no longer mean v128..v255, they mean
    // v0.hi..v127.hi instead. Until the compiler understands this, it is not
    // safe to use v128..v255.
    // TODO-GFX11: Remove this when full 16-bit codegen is implemented.
    return 128;
  }
  if (STI->getFeatureBits().test(FeatureGFX90AInsts))
    return 512;
  return 256;
}

unsigned getMinNumVGPRs(const MCSubtargetInfo *STI, unsigned WavesPerEU) {
  assert(WavesPerEU != 0);

  if (WavesPerEU >= getMaxWavesPerEU(STI))
    return 0;
  unsigned MinNumVGPRs =
      alignDown(getTotalNumVGPRs(STI) / (WavesPerEU + 1),
                getVGPRAllocGranule(STI)) + 1;
  return std::min(MinNumVGPRs, getAddressableNumVGPRs(STI));
}

unsigned getMaxNumVGPRs(const MCSubtargetInfo *STI, unsigned WavesPerEU) {
  assert(WavesPerEU != 0);

  unsigned MaxNumVGPRs = alignDown(getTotalNumVGPRs(STI) / WavesPerEU,
                                   getVGPRAllocGranule(STI));
  unsigned AddressableNumVGPRs = getAddressableNumVGPRs(STI);
  return std::min(MaxNumVGPRs, AddressableNumVGPRs);
}

unsigned getNumVGPRBlocks(const MCSubtargetInfo *STI, unsigned NumVGPRs,
                          Optional<bool> EnableWavefrontSize32) {
  NumVGPRs = alignTo(std::max(1u, NumVGPRs),
                     getVGPREncodingGranule(STI, EnableWavefrontSize32));
  // VGPRBlocks is actual number of VGPR blocks minus 1.
  return NumVGPRs / getVGPREncodingGranule(STI, EnableWavefrontSize32) - 1;
}

} // end namespace IsaInfo

void initDefaultAMDKernelCodeT(amd_kernel_code_t &Header,
                               const MCSubtargetInfo *STI) {
  IsaVersion Version = getIsaVersion(STI->getCPU());

  memset(&Header, 0, sizeof(Header));

  Header.amd_kernel_code_version_major = 1;
  Header.amd_kernel_code_version_minor = 2;
  Header.amd_machine_kind = 1; // AMD_MACHINE_KIND_AMDGPU
  Header.amd_machine_version_major = Version.Major;
  Header.amd_machine_version_minor = Version.Minor;
  Header.amd_machine_version_stepping = Version.Stepping;
  Header.kernel_code_entry_byte_offset = sizeof(Header);
  Header.wavefront_size = 6;

  // If the code object does not support indirect functions, then the value must
  // be 0xffffffff.
  Header.call_convention = -1;

  // These alignment values are specified in powers of two, so alignment =
  // 2^n.  The minimum alignment is 2^4 = 16.
  Header.kernarg_segment_alignment = 4;
  Header.group_segment_alignment = 4;
  Header.private_segment_alignment = 4;

  if (Version.Major >= 10) {
    if (STI->getFeatureBits().test(FeatureWavefrontSize32)) {
      Header.wavefront_size = 5;
      Header.code_properties |= AMD_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32;
    }
    Header.compute_pgm_resource_registers |=
      S_00B848_WGP_MODE(STI->getFeatureBits().test(FeatureCuMode) ? 0 : 1) |
      S_00B848_MEM_ORDERED(1);
  }
}

amdhsa::kernel_descriptor_t getDefaultAmdhsaKernelDescriptor(
    const MCSubtargetInfo *STI) {
  IsaVersion Version = getIsaVersion(STI->getCPU());

  amdhsa::kernel_descriptor_t KD;
  memset(&KD, 0, sizeof(KD));

  AMDHSA_BITS_SET(KD.compute_pgm_rsrc1,
                  amdhsa::COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_16_64,
                  amdhsa::FLOAT_DENORM_MODE_FLUSH_NONE);
  AMDHSA_BITS_SET(KD.compute_pgm_rsrc1,
                  amdhsa::COMPUTE_PGM_RSRC1_ENABLE_DX10_CLAMP, 1);
  AMDHSA_BITS_SET(KD.compute_pgm_rsrc1,
                  amdhsa::COMPUTE_PGM_RSRC1_ENABLE_IEEE_MODE, 1);
  AMDHSA_BITS_SET(KD.compute_pgm_rsrc2,
                  amdhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X, 1);
  if (Version.Major >= 10) {
    AMDHSA_BITS_SET(KD.kernel_code_properties,
                    amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32,
                    STI->getFeatureBits().test(FeatureWavefrontSize32) ? 1 : 0);
    AMDHSA_BITS_SET(KD.compute_pgm_rsrc1,
                    amdhsa::COMPUTE_PGM_RSRC1_WGP_MODE,
                    STI->getFeatureBits().test(FeatureCuMode) ? 0 : 1);
    AMDHSA_BITS_SET(KD.compute_pgm_rsrc1,
                    amdhsa::COMPUTE_PGM_RSRC1_MEM_ORDERED, 1);
  }
  if (AMDGPU::isGFX90A(*STI)) {
    AMDHSA_BITS_SET(KD.compute_pgm_rsrc3,
                    amdhsa::COMPUTE_PGM_RSRC3_GFX90A_TG_SPLIT,
                    STI->getFeatureBits().test(FeatureTgSplit) ? 1 : 0);
  }
  return KD;
}

bool isGroupSegment(const GlobalValue *GV) {
  return GV->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS;
}

bool isGlobalSegment(const GlobalValue *GV) {
  return GV->getAddressSpace() == AMDGPUAS::GLOBAL_ADDRESS;
}

bool isReadOnlySegment(const GlobalValue *GV) {
  unsigned AS = GV->getAddressSpace();
  return AS == AMDGPUAS::CONSTANT_ADDRESS ||
         AS == AMDGPUAS::CONSTANT_ADDRESS_32BIT;
}

bool shouldEmitConstantsToTextSection(const Triple &TT) {
  return TT.getArch() == Triple::r600;
}

int getIntegerAttribute(const Function &F, StringRef Name, int Default) {
  Attribute A = F.getFnAttribute(Name);
  int Result = Default;

  if (A.isStringAttribute()) {
    StringRef Str = A.getValueAsString();
    if (Str.getAsInteger(0, Result)) {
      LLVMContext &Ctx = F.getContext();
      Ctx.emitError("can't parse integer attribute " + Name);
    }
  }

  return Result;
}

std::pair<int, int> getIntegerPairAttribute(const Function &F,
                                            StringRef Name,
                                            std::pair<int, int> Default,
                                            bool OnlyFirstRequired) {
  Attribute A = F.getFnAttribute(Name);
  if (!A.isStringAttribute())
    return Default;

  LLVMContext &Ctx = F.getContext();
  std::pair<int, int> Ints = Default;
  std::pair<StringRef, StringRef> Strs = A.getValueAsString().split(',');
  if (Strs.first.trim().getAsInteger(0, Ints.first)) {
    Ctx.emitError("can't parse first integer attribute " + Name);
    return Default;
  }
  if (Strs.second.trim().getAsInteger(0, Ints.second)) {
    if (!OnlyFirstRequired || !Strs.second.trim().empty()) {
      Ctx.emitError("can't parse second integer attribute " + Name);
      return Default;
    }
  }

  return Ints;
}

unsigned getVmcntBitMask(const IsaVersion &Version) {
  return (1 << (getVmcntBitWidthLo(Version.Major) +
                getVmcntBitWidthHi(Version.Major))) -
         1;
}

unsigned getExpcntBitMask(const IsaVersion &Version) {
  return (1 << getExpcntBitWidth(Version.Major)) - 1;
}

unsigned getLgkmcntBitMask(const IsaVersion &Version) {
  return (1 << getLgkmcntBitWidth(Version.Major)) - 1;
}

unsigned getWaitcntBitMask(const IsaVersion &Version) {
  unsigned VmcntLo = getBitMask(getVmcntBitShiftLo(Version.Major),
                                getVmcntBitWidthLo(Version.Major));
  unsigned Expcnt = getBitMask(getExpcntBitShift(Version.Major),
                               getExpcntBitWidth(Version.Major));
  unsigned Lgkmcnt = getBitMask(getLgkmcntBitShift(Version.Major),
                                getLgkmcntBitWidth(Version.Major));
  unsigned VmcntHi = getBitMask(getVmcntBitShiftHi(Version.Major),
                                getVmcntBitWidthHi(Version.Major));
  return VmcntLo | Expcnt | Lgkmcnt | VmcntHi;
}

unsigned decodeVmcnt(const IsaVersion &Version, unsigned Waitcnt) {
  unsigned VmcntLo = unpackBits(Waitcnt, getVmcntBitShiftLo(Version.Major),
                                getVmcntBitWidthLo(Version.Major));
  unsigned VmcntHi = unpackBits(Waitcnt, getVmcntBitShiftHi(Version.Major),
                                getVmcntBitWidthHi(Version.Major));
  return VmcntLo | VmcntHi << getVmcntBitWidthLo(Version.Major);
}

unsigned decodeExpcnt(const IsaVersion &Version, unsigned Waitcnt) {
  return unpackBits(Waitcnt, getExpcntBitShift(Version.Major),
                    getExpcntBitWidth(Version.Major));
}

unsigned decodeLgkmcnt(const IsaVersion &Version, unsigned Waitcnt) {
  return unpackBits(Waitcnt, getLgkmcntBitShift(Version.Major),
                    getLgkmcntBitWidth(Version.Major));
}

void decodeWaitcnt(const IsaVersion &Version, unsigned Waitcnt,
                   unsigned &Vmcnt, unsigned &Expcnt, unsigned &Lgkmcnt) {
  Vmcnt = decodeVmcnt(Version, Waitcnt);
  Expcnt = decodeExpcnt(Version, Waitcnt);
  Lgkmcnt = decodeLgkmcnt(Version, Waitcnt);
}

Waitcnt decodeWaitcnt(const IsaVersion &Version, unsigned Encoded) {
  Waitcnt Decoded;
  Decoded.VmCnt = decodeVmcnt(Version, Encoded);
  Decoded.ExpCnt = decodeExpcnt(Version, Encoded);
  Decoded.LgkmCnt = decodeLgkmcnt(Version, Encoded);
  return Decoded;
}

unsigned encodeVmcnt(const IsaVersion &Version, unsigned Waitcnt,
                     unsigned Vmcnt) {
  Waitcnt = packBits(Vmcnt, Waitcnt, getVmcntBitShiftLo(Version.Major),
                     getVmcntBitWidthLo(Version.Major));
  return packBits(Vmcnt >> getVmcntBitWidthLo(Version.Major), Waitcnt,
                  getVmcntBitShiftHi(Version.Major),
                  getVmcntBitWidthHi(Version.Major));
}

unsigned encodeExpcnt(const IsaVersion &Version, unsigned Waitcnt,
                      unsigned Expcnt) {
  return packBits(Expcnt, Waitcnt, getExpcntBitShift(Version.Major),
                  getExpcntBitWidth(Version.Major));
}

unsigned encodeLgkmcnt(const IsaVersion &Version, unsigned Waitcnt,
                       unsigned Lgkmcnt) {
  return packBits(Lgkmcnt, Waitcnt, getLgkmcntBitShift(Version.Major),
                  getLgkmcntBitWidth(Version.Major));
}

unsigned encodeWaitcnt(const IsaVersion &Version,
                       unsigned Vmcnt, unsigned Expcnt, unsigned Lgkmcnt) {
  unsigned Waitcnt = getWaitcntBitMask(Version);
  Waitcnt = encodeVmcnt(Version, Waitcnt, Vmcnt);
  Waitcnt = encodeExpcnt(Version, Waitcnt, Expcnt);
  Waitcnt = encodeLgkmcnt(Version, Waitcnt, Lgkmcnt);
  return Waitcnt;
}

unsigned encodeWaitcnt(const IsaVersion &Version, const Waitcnt &Decoded) {
  return encodeWaitcnt(Version, Decoded.VmCnt, Decoded.ExpCnt, Decoded.LgkmCnt);
}

//===----------------------------------------------------------------------===//
// Custom Operands.
//
// A table of custom operands shall describe "primary" operand names
// first followed by aliases if any. It is not required but recommended
// to arrange operands so that operand encoding match operand position
// in the table. This will make disassembly a bit more efficient.
// Unused slots in the table shall have an empty name.
//
//===----------------------------------------------------------------------===//

template <class T>
static bool isValidOpr(int Idx, const CustomOperand<T> OpInfo[], int OpInfoSize,
                       T Context) {
  return 0 <= Idx && Idx < OpInfoSize && !OpInfo[Idx].Name.empty() &&
         (!OpInfo[Idx].Cond || OpInfo[Idx].Cond(Context));
}

template <class T>
static int getOprIdx(std::function<bool(const CustomOperand<T> &)> Test,
                     const CustomOperand<T> OpInfo[], int OpInfoSize,
                     T Context) {
  int InvalidIdx = OPR_ID_UNKNOWN;
  for (int Idx = 0; Idx < OpInfoSize; ++Idx) {
    if (Test(OpInfo[Idx])) {
      if (!OpInfo[Idx].Cond || OpInfo[Idx].Cond(Context))
        return Idx;
      InvalidIdx = OPR_ID_UNSUPPORTED;
    }
  }
  return InvalidIdx;
}

template <class T>
static int getOprIdx(const StringRef Name, const CustomOperand<T> OpInfo[],
                     int OpInfoSize, T Context) {
  auto Test = [=](const CustomOperand<T> &Op) { return Op.Name == Name; };
  return getOprIdx<T>(Test, OpInfo, OpInfoSize, Context);
}

template <class T>
static int getOprIdx(int Id, const CustomOperand<T> OpInfo[], int OpInfoSize,
                     T Context, bool QuickCheck = true) {
  auto Test = [=](const CustomOperand<T> &Op) {
    return Op.Encoding == Id && !Op.Name.empty();
  };
  // This is an optimization that should work in most cases.
  // As a side effect, it may cause selection of an alias
  // instead of a primary operand name in case of sparse tables.
  if (QuickCheck && isValidOpr<T>(Id, OpInfo, OpInfoSize, Context) &&
      OpInfo[Id].Encoding == Id) {
    return Id;
  }
  return getOprIdx<T>(Test, OpInfo, OpInfoSize, Context);
}

//===----------------------------------------------------------------------===//
// Custom Operand Values
//===----------------------------------------------------------------------===//

static unsigned getDefaultCustomOperandEncoding(const CustomOperandVal *Opr,
                                                int Size,
                                                const MCSubtargetInfo &STI) {
  unsigned Enc = 0;
  for (int Idx = 0; Idx < Size; ++Idx) {
    const auto &Op = Opr[Idx];
    if (Op.isSupported(STI))
      Enc |= Op.encode(Op.Default);
  }
  return Enc;
}

static bool isSymbolicCustomOperandEncoding(const CustomOperandVal *Opr,
                                            int Size, unsigned Code,
                                            bool &HasNonDefaultVal,
                                            const MCSubtargetInfo &STI) {
  unsigned UsedOprMask = 0;
  HasNonDefaultVal = false;
  for (int Idx = 0; Idx < Size; ++Idx) {
    const auto &Op = Opr[Idx];
    if (!Op.isSupported(STI))
      continue;
    UsedOprMask |= Op.getMask();
    unsigned Val = Op.decode(Code);
    if (!Op.isValid(Val))
      return false;
    HasNonDefaultVal |= (Val != Op.Default);
  }
  return (Code & ~UsedOprMask) == 0;
}

static bool decodeCustomOperand(const CustomOperandVal *Opr, int Size,
                                unsigned Code, int &Idx, StringRef &Name,
                                unsigned &Val, bool &IsDefault,
                                const MCSubtargetInfo &STI) {
  while (Idx < Size) {
    const auto &Op = Opr[Idx++];
    if (Op.isSupported(STI)) {
      Name = Op.Name;
      Val = Op.decode(Code);
      IsDefault = (Val == Op.Default);
      return true;
    }
  }

  return false;
}

static int encodeCustomOperandVal(const CustomOperandVal &Op,
                                  int64_t InputVal) {
  if (InputVal < 0 || InputVal > Op.Max)
    return OPR_VAL_INVALID;
  return Op.encode(InputVal);
}

static int encodeCustomOperand(const CustomOperandVal *Opr, int Size,
                               const StringRef Name, int64_t InputVal,
                               unsigned &UsedOprMask,
                               const MCSubtargetInfo &STI) {
  int InvalidId = OPR_ID_UNKNOWN;
  for (int Idx = 0; Idx < Size; ++Idx) {
    const auto &Op = Opr[Idx];
    if (Op.Name == Name) {
      if (!Op.isSupported(STI)) {
        InvalidId = OPR_ID_UNSUPPORTED;
        continue;
      }
      auto OprMask = Op.getMask();
      if (OprMask & UsedOprMask)
        return OPR_ID_DUPLICATE;
      UsedOprMask |= OprMask;
      return encodeCustomOperandVal(Op, InputVal);
    }
  }
  return InvalidId;
}

//===----------------------------------------------------------------------===//
// DepCtr
//===----------------------------------------------------------------------===//

namespace DepCtr {

int getDefaultDepCtrEncoding(const MCSubtargetInfo &STI) {
  static int Default = -1;
  if (Default == -1)
    Default = getDefaultCustomOperandEncoding(DepCtrInfo, DEP_CTR_SIZE, STI);
  return Default;
}

bool isSymbolicDepCtrEncoding(unsigned Code, bool &HasNonDefaultVal,
                              const MCSubtargetInfo &STI) {
  return isSymbolicCustomOperandEncoding(DepCtrInfo, DEP_CTR_SIZE, Code,
                                         HasNonDefaultVal, STI);
}

bool decodeDepCtr(unsigned Code, int &Id, StringRef &Name, unsigned &Val,
                  bool &IsDefault, const MCSubtargetInfo &STI) {
  return decodeCustomOperand(DepCtrInfo, DEP_CTR_SIZE, Code, Id, Name, Val,
                             IsDefault, STI);
}

int encodeDepCtr(const StringRef Name, int64_t Val, unsigned &UsedOprMask,
                 const MCSubtargetInfo &STI) {
  return encodeCustomOperand(DepCtrInfo, DEP_CTR_SIZE, Name, Val, UsedOprMask,
                             STI);
}

} // namespace DepCtr

//===----------------------------------------------------------------------===//
// hwreg
//===----------------------------------------------------------------------===//

namespace Hwreg {

int64_t getHwregId(const StringRef Name, const MCSubtargetInfo &STI) {
  int Idx = getOprIdx<const MCSubtargetInfo &>(Name, Opr, OPR_SIZE, STI);
  return (Idx < 0) ? Idx : Opr[Idx].Encoding;
}

bool isValidHwreg(int64_t Id) {
  return 0 <= Id && isUInt<ID_WIDTH_>(Id);
}

bool isValidHwregOffset(int64_t Offset) {
  return 0 <= Offset && isUInt<OFFSET_WIDTH_>(Offset);
}

bool isValidHwregWidth(int64_t Width) {
  return 0 <= (Width - 1) && isUInt<WIDTH_M1_WIDTH_>(Width - 1);
}

uint64_t encodeHwreg(uint64_t Id, uint64_t Offset, uint64_t Width) {
  return (Id << ID_SHIFT_) |
         (Offset << OFFSET_SHIFT_) |
         ((Width - 1) << WIDTH_M1_SHIFT_);
}

StringRef getHwreg(unsigned Id, const MCSubtargetInfo &STI) {
  int Idx = getOprIdx<const MCSubtargetInfo &>(Id, Opr, OPR_SIZE, STI);
  return (Idx < 0) ? "" : Opr[Idx].Name;
}

void decodeHwreg(unsigned Val, unsigned &Id, unsigned &Offset, unsigned &Width) {
  Id = (Val & ID_MASK_) >> ID_SHIFT_;
  Offset = (Val & OFFSET_MASK_) >> OFFSET_SHIFT_;
  Width = ((Val & WIDTH_M1_MASK_) >> WIDTH_M1_SHIFT_) + 1;
}

} // namespace Hwreg

//===----------------------------------------------------------------------===//
// exp tgt
//===----------------------------------------------------------------------===//

namespace Exp {

struct ExpTgt {
  StringLiteral Name;
  unsigned Tgt;
  unsigned MaxIndex;
};

static constexpr ExpTgt ExpTgtInfo[] = {
  {{"null"},           ET_NULL,            ET_NULL_MAX_IDX},
  {{"mrtz"},           ET_MRTZ,            ET_MRTZ_MAX_IDX},
  {{"prim"},           ET_PRIM,            ET_PRIM_MAX_IDX},
  {{"mrt"},            ET_MRT0,            ET_MRT_MAX_IDX},
  {{"pos"},            ET_POS0,            ET_POS_MAX_IDX},
  {{"dual_src_blend"}, ET_DUAL_SRC_BLEND0, ET_DUAL_SRC_BLEND_MAX_IDX},
  {{"param"},          ET_PARAM0,          ET_PARAM_MAX_IDX},
};

bool getTgtName(unsigned Id, StringRef &Name, int &Index) {
  for (const ExpTgt &Val : ExpTgtInfo) {
    if (Val.Tgt <= Id && Id <= Val.Tgt + Val.MaxIndex) {
      Index = (Val.MaxIndex == 0) ? -1 : (Id - Val.Tgt);
      Name = Val.Name;
      return true;
    }
  }
  return false;
}

unsigned getTgtId(const StringRef Name) {

  for (const ExpTgt &Val : ExpTgtInfo) {
    if (Val.MaxIndex == 0 && Name == Val.Name)
      return Val.Tgt;

    if (Val.MaxIndex > 0 && Name.startswith(Val.Name)) {
      StringRef Suffix = Name.drop_front(Val.Name.size());

      unsigned Id;
      if (Suffix.getAsInteger(10, Id) || Id > Val.MaxIndex)
        return ET_INVALID;

      // Disable leading zeroes
      if (Suffix.size() > 1 && Suffix[0] == '0')
        return ET_INVALID;

      return Val.Tgt + Id;
    }
  }
  return ET_INVALID;
}

bool isSupportedTgtId(unsigned Id, const MCSubtargetInfo &STI) {
  switch (Id) {
  case ET_NULL:
    return !isGFX11Plus(STI);
  case ET_POS4:
  case ET_PRIM:
    return isGFX10Plus(STI);
  case ET_DUAL_SRC_BLEND0:
  case ET_DUAL_SRC_BLEND1:
    return isGFX11Plus(STI);
  default:
    if (Id >= ET_PARAM0 && Id <= ET_PARAM31)
      return !isGFX11Plus(STI);
    return true;
  }
}

} // namespace Exp

//===----------------------------------------------------------------------===//
// MTBUF Format
//===----------------------------------------------------------------------===//

namespace MTBUFFormat {

int64_t getDfmt(const StringRef Name) {
  for (int Id = DFMT_MIN; Id <= DFMT_MAX; ++Id) {
    if (Name == DfmtSymbolic[Id])
      return Id;
  }
  return DFMT_UNDEF;
}

StringRef getDfmtName(unsigned Id) {
  assert(Id <= DFMT_MAX);
  return DfmtSymbolic[Id];
}

static StringLiteral const *getNfmtLookupTable(const MCSubtargetInfo &STI) {
  if (isSI(STI) || isCI(STI))
    return NfmtSymbolicSICI;
  if (isVI(STI) || isGFX9(STI))
    return NfmtSymbolicVI;
  return NfmtSymbolicGFX10;
}

int64_t getNfmt(const StringRef Name, const MCSubtargetInfo &STI) {
  auto lookupTable = getNfmtLookupTable(STI);
  for (int Id = NFMT_MIN; Id <= NFMT_MAX; ++Id) {
    if (Name == lookupTable[Id])
      return Id;
  }
  return NFMT_UNDEF;
}

StringRef getNfmtName(unsigned Id, const MCSubtargetInfo &STI) {
  assert(Id <= NFMT_MAX);
  return getNfmtLookupTable(STI)[Id];
}

bool isValidDfmtNfmt(unsigned Id, const MCSubtargetInfo &STI) {
  unsigned Dfmt;
  unsigned Nfmt;
  decodeDfmtNfmt(Id, Dfmt, Nfmt);
  return isValidNfmt(Nfmt, STI);
}

bool isValidNfmt(unsigned Id, const MCSubtargetInfo &STI) {
  return !getNfmtName(Id, STI).empty();
}

int64_t encodeDfmtNfmt(unsigned Dfmt, unsigned Nfmt) {
  return (Dfmt << DFMT_SHIFT) | (Nfmt << NFMT_SHIFT);
}

void decodeDfmtNfmt(unsigned Format, unsigned &Dfmt, unsigned &Nfmt) {
  Dfmt = (Format >> DFMT_SHIFT) & DFMT_MASK;
  Nfmt = (Format >> NFMT_SHIFT) & NFMT_MASK;
}

int64_t getUnifiedFormat(const StringRef Name, const MCSubtargetInfo &STI) {
  if (isGFX11Plus(STI)) {
    for (int Id = UfmtGFX11::UFMT_FIRST; Id <= UfmtGFX11::UFMT_LAST; ++Id) {
      if (Name == UfmtSymbolicGFX11[Id])
        return Id;
    }
  } else {
    for (int Id = UfmtGFX10::UFMT_FIRST; Id <= UfmtGFX10::UFMT_LAST; ++Id) {
      if (Name == UfmtSymbolicGFX10[Id])
        return Id;
    }
  }
  return UFMT_UNDEF;
}

StringRef getUnifiedFormatName(unsigned Id, const MCSubtargetInfo &STI) {
  if(isValidUnifiedFormat(Id, STI))
    return isGFX10(STI) ? UfmtSymbolicGFX10[Id] : UfmtSymbolicGFX11[Id];
  return "";
}

bool isValidUnifiedFormat(unsigned Id, const MCSubtargetInfo &STI) {
  return isGFX10(STI) ? Id <= UfmtGFX10::UFMT_LAST : Id <= UfmtGFX11::UFMT_LAST;
}

int64_t convertDfmtNfmt2Ufmt(unsigned Dfmt, unsigned Nfmt,
                             const MCSubtargetInfo &STI) {
  int64_t Fmt = encodeDfmtNfmt(Dfmt, Nfmt);
  if (isGFX11Plus(STI)) {
    for (int Id = UfmtGFX11::UFMT_FIRST; Id <= UfmtGFX11::UFMT_LAST; ++Id) {
      if (Fmt == DfmtNfmt2UFmtGFX11[Id])
        return Id;
    }
  } else {
    for (int Id = UfmtGFX10::UFMT_FIRST; Id <= UfmtGFX10::UFMT_LAST; ++Id) {
      if (Fmt == DfmtNfmt2UFmtGFX10[Id])
        return Id;
    }
  }
  return UFMT_UNDEF;
}

bool isValidFormatEncoding(unsigned Val, const MCSubtargetInfo &STI) {
  return isGFX10Plus(STI) ? (Val <= UFMT_MAX) : (Val <= DFMT_NFMT_MAX);
}

unsigned getDefaultFormatEncoding(const MCSubtargetInfo &STI) {
  if (isGFX10Plus(STI))
    return UFMT_DEFAULT;
  return DFMT_NFMT_DEFAULT;
}

} // namespace MTBUFFormat

//===----------------------------------------------------------------------===//
// SendMsg
//===----------------------------------------------------------------------===//

namespace SendMsg {

static uint64_t getMsgIdMask(const MCSubtargetInfo &STI) {
  return isGFX11Plus(STI) ? ID_MASK_GFX11Plus_ : ID_MASK_PreGFX11_;
}

int64_t getMsgId(const StringRef Name, const MCSubtargetInfo &STI) {
  int Idx = getOprIdx<const MCSubtargetInfo &>(Name, Msg, MSG_SIZE, STI);
  return (Idx < 0) ? Idx : Msg[Idx].Encoding;
}

bool isValidMsgId(int64_t MsgId, const MCSubtargetInfo &STI) {
  return (MsgId & ~(getMsgIdMask(STI))) == 0;
}

StringRef getMsgName(int64_t MsgId, const MCSubtargetInfo &STI) {
  int Idx = getOprIdx<const MCSubtargetInfo &>(MsgId, Msg, MSG_SIZE, STI);
  return (Idx < 0) ? "" : Msg[Idx].Name;
}

int64_t getMsgOpId(int64_t MsgId, const StringRef Name) {
  const char* const *S = (MsgId == ID_SYSMSG) ? OpSysSymbolic : OpGsSymbolic;
  const int F = (MsgId == ID_SYSMSG) ? OP_SYS_FIRST_ : OP_GS_FIRST_;
  const int L = (MsgId == ID_SYSMSG) ? OP_SYS_LAST_ : OP_GS_LAST_;
  for (int i = F; i < L; ++i) {
    if (Name == S[i]) {
      return i;
    }
  }
  return OP_UNKNOWN_;
}

bool isValidMsgOp(int64_t MsgId, int64_t OpId, const MCSubtargetInfo &STI,
                  bool Strict) {
  assert(isValidMsgId(MsgId, STI));

  if (!Strict)
    return 0 <= OpId && isUInt<OP_WIDTH_>(OpId);

  if (MsgId == ID_SYSMSG)
    return OP_SYS_FIRST_ <= OpId && OpId < OP_SYS_LAST_;
  if (!isGFX11Plus(STI)) {
    switch (MsgId) {
    case ID_GS_PreGFX11:
      return (OP_GS_FIRST_ <= OpId && OpId < OP_GS_LAST_) && OpId != OP_GS_NOP;
    case ID_GS_DONE_PreGFX11:
      return OP_GS_FIRST_ <= OpId && OpId < OP_GS_LAST_;
    }
  }
  return OpId == OP_NONE_;
}

StringRef getMsgOpName(int64_t MsgId, int64_t OpId,
                       const MCSubtargetInfo &STI) {
  assert(msgRequiresOp(MsgId, STI));
  return (MsgId == ID_SYSMSG)? OpSysSymbolic[OpId] : OpGsSymbolic[OpId];
}

bool isValidMsgStream(int64_t MsgId, int64_t OpId, int64_t StreamId,
                      const MCSubtargetInfo &STI, bool Strict) {
  assert(isValidMsgOp(MsgId, OpId, STI, Strict));

  if (!Strict)
    return 0 <= StreamId && isUInt<STREAM_ID_WIDTH_>(StreamId);

  if (!isGFX11Plus(STI)) {
    switch (MsgId) {
    case ID_GS_PreGFX11:
      return STREAM_ID_FIRST_ <= StreamId && StreamId < STREAM_ID_LAST_;
    case ID_GS_DONE_PreGFX11:
      return (OpId == OP_GS_NOP) ?
          (StreamId == STREAM_ID_NONE_) :
          (STREAM_ID_FIRST_ <= StreamId && StreamId < STREAM_ID_LAST_);
    }
  }
  return StreamId == STREAM_ID_NONE_;
}

bool msgRequiresOp(int64_t MsgId, const MCSubtargetInfo &STI) {
  return MsgId == ID_SYSMSG ||
      (!isGFX11Plus(STI) &&
       (MsgId == ID_GS_PreGFX11 || MsgId == ID_GS_DONE_PreGFX11));
}

bool msgSupportsStream(int64_t MsgId, int64_t OpId,
                       const MCSubtargetInfo &STI) {
  return !isGFX11Plus(STI) &&
      (MsgId == ID_GS_PreGFX11 || MsgId == ID_GS_DONE_PreGFX11) &&
      OpId != OP_GS_NOP;
}

void decodeMsg(unsigned Val, uint16_t &MsgId, uint16_t &OpId,
               uint16_t &StreamId, const MCSubtargetInfo &STI) {
  MsgId = Val & getMsgIdMask(STI);
  if (isGFX11Plus(STI)) {
    OpId = 0;
    StreamId = 0;
  } else {
    OpId = (Val & OP_MASK_) >> OP_SHIFT_;
    StreamId = (Val & STREAM_ID_MASK_) >> STREAM_ID_SHIFT_;
  }
}

uint64_t encodeMsg(uint64_t MsgId,
                   uint64_t OpId,
                   uint64_t StreamId) {
  return MsgId | (OpId << OP_SHIFT_) | (StreamId << STREAM_ID_SHIFT_);
}

} // namespace SendMsg

//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

unsigned getInitialPSInputAddr(const Function &F) {
  return getIntegerAttribute(F, "InitialPSInputAddr", 0);
}

bool getHasColorExport(const Function &F) {
  // As a safe default always respond as if PS has color exports.
  return getIntegerAttribute(
             F, "amdgpu-color-export",
             F.getCallingConv() == CallingConv::AMDGPU_PS ? 1 : 0) != 0;
}

bool getHasDepthExport(const Function &F) {
  return getIntegerAttribute(F, "amdgpu-depth-export", 0) != 0;
}

bool isShader(CallingConv::ID cc) {
  switch(cc) {
    case CallingConv::AMDGPU_VS:
    case CallingConv::AMDGPU_LS:
    case CallingConv::AMDGPU_HS:
    case CallingConv::AMDGPU_ES:
    case CallingConv::AMDGPU_GS:
    case CallingConv::AMDGPU_PS:
    case CallingConv::AMDGPU_CS:
      return true;
    default:
      return false;
  }
}

bool isGraphics(CallingConv::ID cc) {
  return isShader(cc) || cc == CallingConv::AMDGPU_Gfx;
}

bool isCompute(CallingConv::ID cc) {
  return !isGraphics(cc) || cc == CallingConv::AMDGPU_CS;
}

bool isEntryFunctionCC(CallingConv::ID CC) {
  switch (CC) {
  case CallingConv::AMDGPU_KERNEL:
  case CallingConv::SPIR_KERNEL:
  case CallingConv::AMDGPU_VS:
  case CallingConv::AMDGPU_GS:
  case CallingConv::AMDGPU_PS:
  case CallingConv::AMDGPU_CS:
  case CallingConv::AMDGPU_ES:
  case CallingConv::AMDGPU_HS:
  case CallingConv::AMDGPU_LS:
    return true;
  default:
    return false;
  }
}

bool isModuleEntryFunctionCC(CallingConv::ID CC) {
  switch (CC) {
  case CallingConv::AMDGPU_Gfx:
    return true;
  default:
    return isEntryFunctionCC(CC);
  }
}

bool isKernelCC(const Function *Func) {
  return AMDGPU::isModuleEntryFunctionCC(Func->getCallingConv());
}

bool hasXNACK(const MCSubtargetInfo &STI) {
  return STI.getFeatureBits()[AMDGPU::FeatureXNACK];
}

bool hasSRAMECC(const MCSubtargetInfo &STI) {
  return STI.getFeatureBits()[AMDGPU::FeatureSRAMECC];
}

bool hasMIMG_R128(const MCSubtargetInfo &STI) {
  return STI.getFeatureBits()[AMDGPU::FeatureMIMG_R128] && !STI.getFeatureBits()[AMDGPU::FeatureR128A16];
}

bool hasGFX10A16(const MCSubtargetInfo &STI) {
  return STI.getFeatureBits()[AMDGPU::FeatureGFX10A16];
}

bool hasG16(const MCSubtargetInfo &STI) {
  return STI.getFeatureBits()[AMDGPU::FeatureG16];
}

bool hasPackedD16(const MCSubtargetInfo &STI) {
  return !STI.getFeatureBits()[AMDGPU::FeatureUnpackedD16VMem] && !isCI(STI) &&
         !isSI(STI);
}

bool isSI(const MCSubtargetInfo &STI) {
  return STI.getFeatureBits()[AMDGPU::FeatureSouthernIslands];
}

bool isCI(const MCSubtargetInfo &STI) {
  return STI.getFeatureBits()[AMDGPU::FeatureSeaIslands];
}

bool isVI(const MCSubtargetInfo &STI) {
  return STI.getFeatureBits()[AMDGPU::FeatureVolcanicIslands];
}

bool isGFX9(const MCSubtargetInfo &STI) {
  return STI.getFeatureBits()[AMDGPU::FeatureGFX9];
}

bool isGFX9_GFX10(const MCSubtargetInfo &STI) {
  return isGFX9(STI) || isGFX10(STI);
}

bool isGFX8_GFX9_GFX10(const MCSubtargetInfo &STI) {
  return isVI(STI) || isGFX9(STI) || isGFX10(STI);
}

bool isGFX8Plus(const MCSubtargetInfo &STI) {
  return isVI(STI) || isGFX9Plus(STI);
}

bool isGFX9Plus(const MCSubtargetInfo &STI) {
  return isGFX9(STI) || isGFX10Plus(STI);
}

bool isGFX10(const MCSubtargetInfo &STI) {
  return STI.getFeatureBits()[AMDGPU::FeatureGFX10];
}

bool isGFX10Plus(const MCSubtargetInfo &STI) {
  return isGFX10(STI) || isGFX11Plus(STI);
}

bool isGFX11(const MCSubtargetInfo &STI) {
  return STI.getFeatureBits()[AMDGPU::FeatureGFX11];
}

bool isGFX11Plus(const MCSubtargetInfo &STI) {
  return isGFX11(STI);
}

bool isNotGFX11Plus(const MCSubtargetInfo &STI) {
  return !isGFX11Plus(STI);
}

bool isNotGFX10Plus(const MCSubtargetInfo &STI) {
  return isSI(STI) || isCI(STI) || isVI(STI) || isGFX9(STI);
}

bool isGFX10Before1030(const MCSubtargetInfo &STI) {
  return isGFX10(STI) && !AMDGPU::isGFX10_BEncoding(STI);
}

bool isGCN3Encoding(const MCSubtargetInfo &STI) {
  return STI.getFeatureBits()[AMDGPU::FeatureGCN3Encoding];
}

bool isGFX10_AEncoding(const MCSubtargetInfo &STI) {
  return STI.getFeatureBits()[AMDGPU::FeatureGFX10_AEncoding];
}

bool isGFX10_BEncoding(const MCSubtargetInfo &STI) {
  return STI.getFeatureBits()[AMDGPU::FeatureGFX10_BEncoding];
}

bool hasGFX10_3Insts(const MCSubtargetInfo &STI) {
  return STI.getFeatureBits()[AMDGPU::FeatureGFX10_3Insts];
}

bool isGFX90A(const MCSubtargetInfo &STI) {
  return STI.getFeatureBits()[AMDGPU::FeatureGFX90AInsts];
}

bool isGFX940(const MCSubtargetInfo &STI) {
  return STI.getFeatureBits()[AMDGPU::FeatureGFX940Insts];
}

bool hasArchitectedFlatScratch(const MCSubtargetInfo &STI) {
  return STI.getFeatureBits()[AMDGPU::FeatureArchitectedFlatScratch];
}

bool hasMAIInsts(const MCSubtargetInfo &STI) {
  return STI.getFeatureBits()[AMDGPU::FeatureMAIInsts];
}

bool hasVOPD(const MCSubtargetInfo &STI) {
  return STI.getFeatureBits()[AMDGPU::FeatureVOPD];
}

int32_t getTotalNumVGPRs(bool has90AInsts, int32_t ArgNumAGPR,
                         int32_t ArgNumVGPR) {
  if (has90AInsts && ArgNumAGPR)
    return alignTo(ArgNumVGPR, 4) + ArgNumAGPR;
  return std::max(ArgNumVGPR, ArgNumAGPR);
}

bool isSGPR(unsigned Reg, const MCRegisterInfo* TRI) {
  const MCRegisterClass SGPRClass = TRI->getRegClass(AMDGPU::SReg_32RegClassID);
  const unsigned FirstSubReg = TRI->getSubReg(Reg, AMDGPU::sub0);
  return SGPRClass.contains(FirstSubReg != 0 ? FirstSubReg : Reg) ||
    Reg == AMDGPU::SCC;
}

#define MAP_REG2REG \
  using namespace AMDGPU; \
  switch(Reg) { \
  default: return Reg; \
  CASE_CI_VI(FLAT_SCR) \
  CASE_CI_VI(FLAT_SCR_LO) \
  CASE_CI_VI(FLAT_SCR_HI) \
  CASE_VI_GFX9PLUS(TTMP0) \
  CASE_VI_GFX9PLUS(TTMP1) \
  CASE_VI_GFX9PLUS(TTMP2) \
  CASE_VI_GFX9PLUS(TTMP3) \
  CASE_VI_GFX9PLUS(TTMP4) \
  CASE_VI_GFX9PLUS(TTMP5) \
  CASE_VI_GFX9PLUS(TTMP6) \
  CASE_VI_GFX9PLUS(TTMP7) \
  CASE_VI_GFX9PLUS(TTMP8) \
  CASE_VI_GFX9PLUS(TTMP9) \
  CASE_VI_GFX9PLUS(TTMP10) \
  CASE_VI_GFX9PLUS(TTMP11) \
  CASE_VI_GFX9PLUS(TTMP12) \
  CASE_VI_GFX9PLUS(TTMP13) \
  CASE_VI_GFX9PLUS(TTMP14) \
  CASE_VI_GFX9PLUS(TTMP15) \
  CASE_VI_GFX9PLUS(TTMP0_TTMP1) \
  CASE_VI_GFX9PLUS(TTMP2_TTMP3) \
  CASE_VI_GFX9PLUS(TTMP4_TTMP5) \
  CASE_VI_GFX9PLUS(TTMP6_TTMP7) \
  CASE_VI_GFX9PLUS(TTMP8_TTMP9) \
  CASE_VI_GFX9PLUS(TTMP10_TTMP11) \
  CASE_VI_GFX9PLUS(TTMP12_TTMP13) \
  CASE_VI_GFX9PLUS(TTMP14_TTMP15) \
  CASE_VI_GFX9PLUS(TTMP0_TTMP1_TTMP2_TTMP3) \
  CASE_VI_GFX9PLUS(TTMP4_TTMP5_TTMP6_TTMP7) \
  CASE_VI_GFX9PLUS(TTMP8_TTMP9_TTMP10_TTMP11) \
  CASE_VI_GFX9PLUS(TTMP12_TTMP13_TTMP14_TTMP15) \
  CASE_VI_GFX9PLUS(TTMP0_TTMP1_TTMP2_TTMP3_TTMP4_TTMP5_TTMP6_TTMP7) \
  CASE_VI_GFX9PLUS(TTMP4_TTMP5_TTMP6_TTMP7_TTMP8_TTMP9_TTMP10_TTMP11) \
  CASE_VI_GFX9PLUS(TTMP8_TTMP9_TTMP10_TTMP11_TTMP12_TTMP13_TTMP14_TTMP15) \
  CASE_VI_GFX9PLUS(TTMP0_TTMP1_TTMP2_TTMP3_TTMP4_TTMP5_TTMP6_TTMP7_TTMP8_TTMP9_TTMP10_TTMP11_TTMP12_TTMP13_TTMP14_TTMP15) \
  CASE_GFXPRE11_GFX11PLUS(M0) \
  CASE_GFXPRE11_GFX11PLUS(SGPR_NULL) \
  CASE_GFXPRE11_GFX11PLUS_TO(SGPR_NULL64, SGPR_NULL) \
  }

#define CASE_CI_VI(node) \
  assert(!isSI(STI)); \
  case node: return isCI(STI) ? node##_ci : node##_vi;

#define CASE_VI_GFX9PLUS(node) \
  case node: return isGFX9Plus(STI) ? node##_gfx9plus : node##_vi;

#define CASE_GFXPRE11_GFX11PLUS(node) \
  case node: return isGFX11Plus(STI) ? node##_gfx11plus : node##_gfxpre11;

#define CASE_GFXPRE11_GFX11PLUS_TO(node, result) \
  case node: return isGFX11Plus(STI) ? result##_gfx11plus : result##_gfxpre11;

unsigned getMCReg(unsigned Reg, const MCSubtargetInfo &STI) {
  if (STI.getTargetTriple().getArch() == Triple::r600)
    return Reg;
  MAP_REG2REG
}

#undef CASE_CI_VI
#undef CASE_VI_GFX9PLUS
#undef CASE_GFXPRE11_GFX11PLUS
#undef CASE_GFXPRE11_GFX11PLUS_TO

#define CASE_CI_VI(node)   case node##_ci: case node##_vi:   return node;
#define CASE_VI_GFX9PLUS(node) case node##_vi: case node##_gfx9plus: return node;
#define CASE_GFXPRE11_GFX11PLUS(node) case node##_gfx11plus: case node##_gfxpre11: return node;
#define CASE_GFXPRE11_GFX11PLUS_TO(node, result)

unsigned mc2PseudoReg(unsigned Reg) {
  MAP_REG2REG
}

#undef CASE_CI_VI
#undef CASE_VI_GFX9PLUS
#undef CASE_GFXPRE11_GFX11PLUS
#undef CASE_GFXPRE11_GFX11PLUS_TO
#undef MAP_REG2REG

bool isSISrcOperand(const MCInstrDesc &Desc, unsigned OpNo) {
  assert(OpNo < Desc.NumOperands);
  unsigned OpType = Desc.OpInfo[OpNo].OperandType;
  return OpType >= AMDGPU::OPERAND_SRC_FIRST &&
         OpType <= AMDGPU::OPERAND_SRC_LAST;
}

bool isSISrcFPOperand(const MCInstrDesc &Desc, unsigned OpNo) {
  assert(OpNo < Desc.NumOperands);
  unsigned OpType = Desc.OpInfo[OpNo].OperandType;
  switch (OpType) {
  case AMDGPU::OPERAND_REG_IMM_FP32:
  case AMDGPU::OPERAND_REG_IMM_FP32_DEFERRED:
  case AMDGPU::OPERAND_REG_IMM_FP64:
  case AMDGPU::OPERAND_REG_IMM_FP16:
  case AMDGPU::OPERAND_REG_IMM_FP16_DEFERRED:
  case AMDGPU::OPERAND_REG_IMM_V2FP16:
  case AMDGPU::OPERAND_REG_IMM_V2INT16:
  case AMDGPU::OPERAND_REG_INLINE_C_FP32:
  case AMDGPU::OPERAND_REG_INLINE_C_FP64:
  case AMDGPU::OPERAND_REG_INLINE_C_FP16:
  case AMDGPU::OPERAND_REG_INLINE_C_V2FP16:
  case AMDGPU::OPERAND_REG_INLINE_C_V2INT16:
  case AMDGPU::OPERAND_REG_INLINE_AC_FP32:
  case AMDGPU::OPERAND_REG_INLINE_AC_FP16:
  case AMDGPU::OPERAND_REG_INLINE_AC_V2FP16:
  case AMDGPU::OPERAND_REG_INLINE_AC_V2INT16:
  case AMDGPU::OPERAND_REG_IMM_V2FP32:
  case AMDGPU::OPERAND_REG_INLINE_C_V2FP32:
  case AMDGPU::OPERAND_REG_INLINE_AC_FP64:
    return true;
  default:
    return false;
  }
}

bool isSISrcInlinableOperand(const MCInstrDesc &Desc, unsigned OpNo) {
  assert(OpNo < Desc.NumOperands);
  unsigned OpType = Desc.OpInfo[OpNo].OperandType;
  return OpType >= AMDGPU::OPERAND_REG_INLINE_C_FIRST &&
         OpType <= AMDGPU::OPERAND_REG_INLINE_C_LAST;
}

// Avoid using MCRegisterClass::getSize, since that function will go away
// (move from MC* level to Target* level). Return size in bits.
unsigned getRegBitWidth(unsigned RCID) {
  switch (RCID) {
  case AMDGPU::VGPR_LO16RegClassID:
  case AMDGPU::VGPR_HI16RegClassID:
  case AMDGPU::SGPR_LO16RegClassID:
  case AMDGPU::AGPR_LO16RegClassID:
    return 16;
  case AMDGPU::SGPR_32RegClassID:
  case AMDGPU::VGPR_32RegClassID:
  case AMDGPU::VRegOrLds_32RegClassID:
  case AMDGPU::AGPR_32RegClassID:
  case AMDGPU::VS_32RegClassID:
  case AMDGPU::AV_32RegClassID:
  case AMDGPU::SReg_32RegClassID:
  case AMDGPU::SReg_32_XM0RegClassID:
  case AMDGPU::SRegOrLds_32RegClassID:
    return 32;
  case AMDGPU::SGPR_64RegClassID:
  case AMDGPU::VS_64RegClassID:
  case AMDGPU::SReg_64RegClassID:
  case AMDGPU::VReg_64RegClassID:
  case AMDGPU::AReg_64RegClassID:
  case AMDGPU::SReg_64_XEXECRegClassID:
  case AMDGPU::VReg_64_Align2RegClassID:
  case AMDGPU::AReg_64_Align2RegClassID:
  case AMDGPU::AV_64RegClassID:
  case AMDGPU::AV_64_Align2RegClassID:
    return 64;
  case AMDGPU::SGPR_96RegClassID:
  case AMDGPU::SReg_96RegClassID:
  case AMDGPU::VReg_96RegClassID:
  case AMDGPU::AReg_96RegClassID:
  case AMDGPU::VReg_96_Align2RegClassID:
  case AMDGPU::AReg_96_Align2RegClassID:
  case AMDGPU::AV_96RegClassID:
  case AMDGPU::AV_96_Align2RegClassID:
    return 96;
  case AMDGPU::SGPR_128RegClassID:
  case AMDGPU::SReg_128RegClassID:
  case AMDGPU::VReg_128RegClassID:
  case AMDGPU::AReg_128RegClassID:
  case AMDGPU::VReg_128_Align2RegClassID:
  case AMDGPU::AReg_128_Align2RegClassID:
  case AMDGPU::AV_128RegClassID:
  case AMDGPU::AV_128_Align2RegClassID:
    return 128;
  case AMDGPU::SGPR_160RegClassID:
  case AMDGPU::SReg_160RegClassID:
  case AMDGPU::VReg_160RegClassID:
  case AMDGPU::AReg_160RegClassID:
  case AMDGPU::VReg_160_Align2RegClassID:
  case AMDGPU::AReg_160_Align2RegClassID:
  case AMDGPU::AV_160RegClassID:
  case AMDGPU::AV_160_Align2RegClassID:
    return 160;
  case AMDGPU::SGPR_192RegClassID:
  case AMDGPU::SReg_192RegClassID:
  case AMDGPU::VReg_192RegClassID:
  case AMDGPU::AReg_192RegClassID:
  case AMDGPU::VReg_192_Align2RegClassID:
  case AMDGPU::AReg_192_Align2RegClassID:
  case AMDGPU::AV_192RegClassID:
  case AMDGPU::AV_192_Align2RegClassID:
    return 192;
  case AMDGPU::SGPR_224RegClassID:
  case AMDGPU::SReg_224RegClassID:
  case AMDGPU::VReg_224RegClassID:
  case AMDGPU::AReg_224RegClassID:
  case AMDGPU::VReg_224_Align2RegClassID:
  case AMDGPU::AReg_224_Align2RegClassID:
  case AMDGPU::AV_224RegClassID:
  case AMDGPU::AV_224_Align2RegClassID:
    return 224;
  case AMDGPU::SGPR_256RegClassID:
  case AMDGPU::SReg_256RegClassID:
  case AMDGPU::VReg_256RegClassID:
  case AMDGPU::AReg_256RegClassID:
  case AMDGPU::VReg_256_Align2RegClassID:
  case AMDGPU::AReg_256_Align2RegClassID:
  case AMDGPU::AV_256RegClassID:
  case AMDGPU::AV_256_Align2RegClassID:
    return 256;
  case AMDGPU::SGPR_512RegClassID:
  case AMDGPU::SReg_512RegClassID:
  case AMDGPU::VReg_512RegClassID:
  case AMDGPU::AReg_512RegClassID:
  case AMDGPU::VReg_512_Align2RegClassID:
  case AMDGPU::AReg_512_Align2RegClassID:
  case AMDGPU::AV_512RegClassID:
  case AMDGPU::AV_512_Align2RegClassID:
    return 512;
  case AMDGPU::SGPR_1024RegClassID:
  case AMDGPU::SReg_1024RegClassID:
  case AMDGPU::VReg_1024RegClassID:
  case AMDGPU::AReg_1024RegClassID:
  case AMDGPU::VReg_1024_Align2RegClassID:
  case AMDGPU::AReg_1024_Align2RegClassID:
  case AMDGPU::AV_1024RegClassID:
  case AMDGPU::AV_1024_Align2RegClassID:
    return 1024;
  default:
    llvm_unreachable("Unexpected register class");
  }
}

unsigned getRegBitWidth(const MCRegisterClass &RC) {
  return getRegBitWidth(RC.getID());
}

unsigned getRegOperandSize(const MCRegisterInfo *MRI, const MCInstrDesc &Desc,
                           unsigned OpNo) {
  assert(OpNo < Desc.NumOperands);
  unsigned RCID = Desc.OpInfo[OpNo].RegClass;
  return getRegBitWidth(MRI->getRegClass(RCID)) / 8;
}

bool isInlinableLiteral64(int64_t Literal, bool HasInv2Pi) {
  if (isInlinableIntLiteral(Literal))
    return true;

  uint64_t Val = static_cast<uint64_t>(Literal);
  return (Val == DoubleToBits(0.0)) ||
         (Val == DoubleToBits(1.0)) ||
         (Val == DoubleToBits(-1.0)) ||
         (Val == DoubleToBits(0.5)) ||
         (Val == DoubleToBits(-0.5)) ||
         (Val == DoubleToBits(2.0)) ||
         (Val == DoubleToBits(-2.0)) ||
         (Val == DoubleToBits(4.0)) ||
         (Val == DoubleToBits(-4.0)) ||
         (Val == 0x3fc45f306dc9c882 && HasInv2Pi);
}

bool isInlinableLiteral32(int32_t Literal, bool HasInv2Pi) {
  if (isInlinableIntLiteral(Literal))
    return true;

  // The actual type of the operand does not seem to matter as long
  // as the bits match one of the inline immediate values.  For example:
  //
  // -nan has the hexadecimal encoding of 0xfffffffe which is -2 in decimal,
  // so it is a legal inline immediate.
  //
  // 1065353216 has the hexadecimal encoding 0x3f800000 which is 1.0f in
  // floating-point, so it is a legal inline immediate.

  uint32_t Val = static_cast<uint32_t>(Literal);
  return (Val == FloatToBits(0.0f)) ||
         (Val == FloatToBits(1.0f)) ||
         (Val == FloatToBits(-1.0f)) ||
         (Val == FloatToBits(0.5f)) ||
         (Val == FloatToBits(-0.5f)) ||
         (Val == FloatToBits(2.0f)) ||
         (Val == FloatToBits(-2.0f)) ||
         (Val == FloatToBits(4.0f)) ||
         (Val == FloatToBits(-4.0f)) ||
         (Val == 0x3e22f983 && HasInv2Pi);
}

bool isInlinableLiteral16(int16_t Literal, bool HasInv2Pi) {
  if (!HasInv2Pi)
    return false;

  if (isInlinableIntLiteral(Literal))
    return true;

  uint16_t Val = static_cast<uint16_t>(Literal);
  return Val == 0x3C00 || // 1.0
         Val == 0xBC00 || // -1.0
         Val == 0x3800 || // 0.5
         Val == 0xB800 || // -0.5
         Val == 0x4000 || // 2.0
         Val == 0xC000 || // -2.0
         Val == 0x4400 || // 4.0
         Val == 0xC400 || // -4.0
         Val == 0x3118;   // 1/2pi
}

bool isInlinableLiteralV216(int32_t Literal, bool HasInv2Pi) {
  assert(HasInv2Pi);

  if (isInt<16>(Literal) || isUInt<16>(Literal)) {
    int16_t Trunc = static_cast<int16_t>(Literal);
    return AMDGPU::isInlinableLiteral16(Trunc, HasInv2Pi);
  }
  if (!(Literal & 0xffff))
    return AMDGPU::isInlinableLiteral16(Literal >> 16, HasInv2Pi);

  int16_t Lo16 = static_cast<int16_t>(Literal);
  int16_t Hi16 = static_cast<int16_t>(Literal >> 16);
  return Lo16 == Hi16 && isInlinableLiteral16(Lo16, HasInv2Pi);
}

bool isInlinableIntLiteralV216(int32_t Literal) {
  int16_t Lo16 = static_cast<int16_t>(Literal);
  if (isInt<16>(Literal) || isUInt<16>(Literal))
    return isInlinableIntLiteral(Lo16);

  int16_t Hi16 = static_cast<int16_t>(Literal >> 16);
  if (!(Literal & 0xffff))
    return isInlinableIntLiteral(Hi16);
  return Lo16 == Hi16 && isInlinableIntLiteral(Lo16);
}

bool isFoldableLiteralV216(int32_t Literal, bool HasInv2Pi) {
  assert(HasInv2Pi);

  int16_t Lo16 = static_cast<int16_t>(Literal);
  if (isInt<16>(Literal) || isUInt<16>(Literal))
    return true;

  int16_t Hi16 = static_cast<int16_t>(Literal >> 16);
  if (!(Literal & 0xffff))
    return true;
  return Lo16 == Hi16;
}

bool isArgPassedInSGPR(const Argument *A) {
  const Function *F = A->getParent();

  // Arguments to compute shaders are never a source of divergence.
  CallingConv::ID CC = F->getCallingConv();
  switch (CC) {
  case CallingConv::AMDGPU_KERNEL:
  case CallingConv::SPIR_KERNEL:
    return true;
  case CallingConv::AMDGPU_VS:
  case CallingConv::AMDGPU_LS:
  case CallingConv::AMDGPU_HS:
  case CallingConv::AMDGPU_ES:
  case CallingConv::AMDGPU_GS:
  case CallingConv::AMDGPU_PS:
  case CallingConv::AMDGPU_CS:
  case CallingConv::AMDGPU_Gfx:
    // For non-compute shaders, SGPR inputs are marked with either inreg or byval.
    // Everything else is in VGPRs.
    return F->getAttributes().hasParamAttr(A->getArgNo(), Attribute::InReg) ||
           F->getAttributes().hasParamAttr(A->getArgNo(), Attribute::ByVal);
  default:
    // TODO: Should calls support inreg for SGPR inputs?
    return false;
  }
}

static bool hasSMEMByteOffset(const MCSubtargetInfo &ST) {
  return isGCN3Encoding(ST) || isGFX10Plus(ST);
}

static bool hasSMRDSignedImmOffset(const MCSubtargetInfo &ST) {
  return isGFX9Plus(ST);
}

bool isLegalSMRDEncodedUnsignedOffset(const MCSubtargetInfo &ST,
                                      int64_t EncodedOffset) {
  return hasSMEMByteOffset(ST) ? isUInt<20>(EncodedOffset)
                               : isUInt<8>(EncodedOffset);
}

bool isLegalSMRDEncodedSignedOffset(const MCSubtargetInfo &ST,
                                    int64_t EncodedOffset,
                                    bool IsBuffer) {
  return !IsBuffer &&
         hasSMRDSignedImmOffset(ST) &&
         isInt<21>(EncodedOffset);
}

static bool isDwordAligned(uint64_t ByteOffset) {
  return (ByteOffset & 3) == 0;
}

uint64_t convertSMRDOffsetUnits(const MCSubtargetInfo &ST,
                                uint64_t ByteOffset) {
  if (hasSMEMByteOffset(ST))
    return ByteOffset;

  assert(isDwordAligned(ByteOffset));
  return ByteOffset >> 2;
}

Optional<int64_t> getSMRDEncodedOffset(const MCSubtargetInfo &ST,
                                       int64_t ByteOffset, bool IsBuffer) {
  // The signed version is always a byte offset.
  if (!IsBuffer && hasSMRDSignedImmOffset(ST)) {
    assert(hasSMEMByteOffset(ST));
    return isInt<20>(ByteOffset) ? Optional<int64_t>(ByteOffset) : None;
  }

  if (!isDwordAligned(ByteOffset) && !hasSMEMByteOffset(ST))
    return None;

  int64_t EncodedOffset = convertSMRDOffsetUnits(ST, ByteOffset);
  return isLegalSMRDEncodedUnsignedOffset(ST, EncodedOffset)
             ? Optional<int64_t>(EncodedOffset)
             : None;
}

Optional<int64_t> getSMRDEncodedLiteralOffset32(const MCSubtargetInfo &ST,
                                                int64_t ByteOffset) {
  if (!isCI(ST) || !isDwordAligned(ByteOffset))
    return None;

  int64_t EncodedOffset = convertSMRDOffsetUnits(ST, ByteOffset);
  return isUInt<32>(EncodedOffset) ? Optional<int64_t>(EncodedOffset) : None;
}

unsigned getNumFlatOffsetBits(const MCSubtargetInfo &ST, bool Signed) {
  // Address offset is 12-bit signed for GFX10, 13-bit for GFX9 and GFX11+.
  if (AMDGPU::isGFX10(ST))
    return Signed ? 12 : 11;

  return Signed ? 13 : 12;
}

// Given Imm, split it into the values to put into the SOffset and ImmOffset
// fields in an MUBUF instruction. Return false if it is not possible (due to a
// hardware bug needing a workaround).
//
// The required alignment ensures that individual address components remain
// aligned if they are aligned to begin with. It also ensures that additional
// offsets within the given alignment can be added to the resulting ImmOffset.
bool splitMUBUFOffset(uint32_t Imm, uint32_t &SOffset, uint32_t &ImmOffset,
                      const GCNSubtarget *Subtarget, Align Alignment) {
  const uint32_t MaxImm = alignDown(4095, Alignment.value());
  uint32_t Overflow = 0;

  if (Imm > MaxImm) {
    if (Imm <= MaxImm + 64) {
      // Use an SOffset inline constant for 4..64
      Overflow = Imm - MaxImm;
      Imm = MaxImm;
    } else {
      // Try to keep the same value in SOffset for adjacent loads, so that
      // the corresponding register contents can be re-used.
      //
      // Load values with all low-bits (except for alignment bits) set into
      // SOffset, so that a larger range of values can be covered using
      // s_movk_i32.
      //
      // Atomic operations fail to work correctly when individual address
      // components are unaligned, even if their sum is aligned.
      uint32_t High = (Imm + Alignment.value()) & ~4095;
      uint32_t Low = (Imm + Alignment.value()) & 4095;
      Imm = Low;
      Overflow = High - Alignment.value();
    }
  }

  // There is a hardware bug in SI and CI which prevents address clamping in
  // MUBUF instructions from working correctly with SOffsets. The immediate
  // offset is unaffected.
  if (Overflow > 0 &&
      Subtarget->getGeneration() <= AMDGPUSubtarget::SEA_ISLANDS)
    return false;

  ImmOffset = Imm;
  SOffset = Overflow;
  return true;
}

SIModeRegisterDefaults::SIModeRegisterDefaults(const Function &F) {
  *this = getDefaultForCallingConv(F.getCallingConv());

  StringRef IEEEAttr = F.getFnAttribute("amdgpu-ieee").getValueAsString();
  if (!IEEEAttr.empty())
    IEEE = IEEEAttr == "true";

  StringRef DX10ClampAttr
    = F.getFnAttribute("amdgpu-dx10-clamp").getValueAsString();
  if (!DX10ClampAttr.empty())
    DX10Clamp = DX10ClampAttr == "true";

  StringRef DenormF32Attr = F.getFnAttribute("denormal-fp-math-f32").getValueAsString();
  if (!DenormF32Attr.empty()) {
    DenormalMode DenormMode = parseDenormalFPAttribute(DenormF32Attr);
    FP32InputDenormals = DenormMode.Input == DenormalMode::IEEE;
    FP32OutputDenormals = DenormMode.Output == DenormalMode::IEEE;
  }

  StringRef DenormAttr = F.getFnAttribute("denormal-fp-math").getValueAsString();
  if (!DenormAttr.empty()) {
    DenormalMode DenormMode = parseDenormalFPAttribute(DenormAttr);

    if (DenormF32Attr.empty()) {
      FP32InputDenormals = DenormMode.Input == DenormalMode::IEEE;
      FP32OutputDenormals = DenormMode.Output == DenormalMode::IEEE;
    }

    FP64FP16InputDenormals = DenormMode.Input == DenormalMode::IEEE;
    FP64FP16OutputDenormals = DenormMode.Output == DenormalMode::IEEE;
  }
}

namespace {

struct SourceOfDivergence {
  unsigned Intr;
};
const SourceOfDivergence *lookupSourceOfDivergence(unsigned Intr);

#define GET_SourcesOfDivergence_IMPL
#define GET_Gfx9BufferFormat_IMPL
#define GET_Gfx10BufferFormat_IMPL
#define GET_Gfx11PlusBufferFormat_IMPL
#include "AMDGPUGenSearchableTables.inc"

} // end anonymous namespace

bool isIntrinsicSourceOfDivergence(unsigned IntrID) {
  return lookupSourceOfDivergence(IntrID);
}

const GcnBufferFormatInfo *getGcnBufferFormatInfo(uint8_t BitsPerComp,
                                                  uint8_t NumComponents,
                                                  uint8_t NumFormat,
                                                  const MCSubtargetInfo &STI) {
  return isGFX11Plus(STI)
             ? getGfx11PlusBufferFormatInfo(BitsPerComp, NumComponents,
                                            NumFormat)
             : isGFX10(STI) ? getGfx10BufferFormatInfo(BitsPerComp,
                                                       NumComponents, NumFormat)
                            : getGfx9BufferFormatInfo(BitsPerComp,
                                                      NumComponents, NumFormat);
}

const GcnBufferFormatInfo *getGcnBufferFormatInfo(uint8_t Format,
                                                  const MCSubtargetInfo &STI) {
  return isGFX11Plus(STI) ? getGfx11PlusBufferFormatInfo(Format)
                          : isGFX10(STI) ? getGfx10BufferFormatInfo(Format)
                                         : getGfx9BufferFormatInfo(Format);
}

} // namespace AMDGPU

raw_ostream &operator<<(raw_ostream &OS,
                        const AMDGPU::IsaInfo::TargetIDSetting S) {
  switch (S) {
  case (AMDGPU::IsaInfo::TargetIDSetting::Unsupported):
    OS << "Unsupported";
    break;
  case (AMDGPU::IsaInfo::TargetIDSetting::Any):
    OS << "Any";
    break;
  case (AMDGPU::IsaInfo::TargetIDSetting::Off):
    OS << "Off";
    break;
  case (AMDGPU::IsaInfo::TargetIDSetting::On):
    OS << "On";
    break;
  }
  return OS;
}

} // namespace llvm
