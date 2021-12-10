//===-- TargetParser - Parser for target features ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a target parser to recognise hardware features such as
// FPU/CPU/ARCH names as well as specific support such as HDIV, etc.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/TargetParser.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ARMBuildAttributes.h"

using namespace llvm;
using namespace AMDGPU;

namespace {

struct GPUInfo {
  StringLiteral Name;
  StringLiteral CanonicalName;
  AMDGPU::GPUKind Kind;
  unsigned Features;
};

constexpr GPUInfo R600GPUs[] = {
  // Name       Canonical    Kind        Features
  //            Name
  {{"r600"},    {"r600"},    GK_R600,    FEATURE_NONE },
  {{"rv630"},   {"r600"},    GK_R600,    FEATURE_NONE },
  {{"rv635"},   {"r600"},    GK_R600,    FEATURE_NONE },
  {{"r630"},    {"r630"},    GK_R630,    FEATURE_NONE },
  {{"rs780"},   {"rs880"},   GK_RS880,   FEATURE_NONE },
  {{"rs880"},   {"rs880"},   GK_RS880,   FEATURE_NONE },
  {{"rv610"},   {"rs880"},   GK_RS880,   FEATURE_NONE },
  {{"rv620"},   {"rs880"},   GK_RS880,   FEATURE_NONE },
  {{"rv670"},   {"rv670"},   GK_RV670,   FEATURE_NONE },
  {{"rv710"},   {"rv710"},   GK_RV710,   FEATURE_NONE },
  {{"rv730"},   {"rv730"},   GK_RV730,   FEATURE_NONE },
  {{"rv740"},   {"rv770"},   GK_RV770,   FEATURE_NONE },
  {{"rv770"},   {"rv770"},   GK_RV770,   FEATURE_NONE },
  {{"cedar"},   {"cedar"},   GK_CEDAR,   FEATURE_NONE },
  {{"palm"},    {"cedar"},   GK_CEDAR,   FEATURE_NONE },
  {{"cypress"}, {"cypress"}, GK_CYPRESS, FEATURE_FMA  },
  {{"hemlock"}, {"cypress"}, GK_CYPRESS, FEATURE_FMA  },
  {{"juniper"}, {"juniper"}, GK_JUNIPER, FEATURE_NONE },
  {{"redwood"}, {"redwood"}, GK_REDWOOD, FEATURE_NONE },
  {{"sumo"},    {"sumo"},    GK_SUMO,    FEATURE_NONE },
  {{"sumo2"},   {"sumo"},    GK_SUMO,    FEATURE_NONE },
  {{"barts"},   {"barts"},   GK_BARTS,   FEATURE_NONE },
  {{"caicos"},  {"caicos"},  GK_CAICOS,  FEATURE_NONE },
  {{"aruba"},   {"cayman"},  GK_CAYMAN,  FEATURE_FMA  },
  {{"cayman"},  {"cayman"},  GK_CAYMAN,  FEATURE_FMA  },
  {{"turks"},   {"turks"},   GK_TURKS,   FEATURE_NONE }
};

// This table should be sorted by the value of GPUKind
// Don't bother listing the implicitly true features
constexpr GPUInfo AMDGCNGPUs[] = {
  // Name         Canonical    Kind        Features
  //              Name
  {{"gfx600"},    {"gfx600"},  GK_GFX600,  FEATURE_FAST_FMA_F32},
  {{"tahiti"},    {"gfx600"},  GK_GFX600,  FEATURE_FAST_FMA_F32},
  {{"gfx601"},    {"gfx601"},  GK_GFX601,  FEATURE_NONE},
  {{"pitcairn"},  {"gfx601"},  GK_GFX601,  FEATURE_NONE},
  {{"verde"},     {"gfx601"},  GK_GFX601,  FEATURE_NONE},
  {{"gfx602"},    {"gfx602"},  GK_GFX602,  FEATURE_NONE},
  {{"hainan"},    {"gfx602"},  GK_GFX602,  FEATURE_NONE},
  {{"oland"},     {"gfx602"},  GK_GFX602,  FEATURE_NONE},
  {{"gfx700"},    {"gfx700"},  GK_GFX700,  FEATURE_NONE},
  {{"kaveri"},    {"gfx700"},  GK_GFX700,  FEATURE_NONE},
  {{"gfx701"},    {"gfx701"},  GK_GFX701,  FEATURE_FAST_FMA_F32},
  {{"hawaii"},    {"gfx701"},  GK_GFX701,  FEATURE_FAST_FMA_F32},
  {{"gfx702"},    {"gfx702"},  GK_GFX702,  FEATURE_FAST_FMA_F32},
  {{"gfx703"},    {"gfx703"},  GK_GFX703,  FEATURE_NONE},
  {{"kabini"},    {"gfx703"},  GK_GFX703,  FEATURE_NONE},
  {{"mullins"},   {"gfx703"},  GK_GFX703,  FEATURE_NONE},
  {{"gfx704"},    {"gfx704"},  GK_GFX704,  FEATURE_NONE},
  {{"bonaire"},   {"gfx704"},  GK_GFX704,  FEATURE_NONE},
  {{"gfx705"},    {"gfx705"},  GK_GFX705,  FEATURE_NONE},
  {{"gfx801"},    {"gfx801"},  GK_GFX801,  FEATURE_FAST_FMA_F32|FEATURE_FAST_DENORMAL_F32|FEATURE_XNACK},
  {{"carrizo"},   {"gfx801"},  GK_GFX801,  FEATURE_FAST_FMA_F32|FEATURE_FAST_DENORMAL_F32|FEATURE_XNACK},
  {{"gfx802"},    {"gfx802"},  GK_GFX802,  FEATURE_FAST_DENORMAL_F32},
  {{"iceland"},   {"gfx802"},  GK_GFX802,  FEATURE_FAST_DENORMAL_F32},
  {{"tonga"},     {"gfx802"},  GK_GFX802,  FEATURE_FAST_DENORMAL_F32},
  {{"gfx803"},    {"gfx803"},  GK_GFX803,  FEATURE_FAST_DENORMAL_F32},
  {{"fiji"},      {"gfx803"},  GK_GFX803,  FEATURE_FAST_DENORMAL_F32},
  {{"polaris10"}, {"gfx803"},  GK_GFX803,  FEATURE_FAST_DENORMAL_F32},
  {{"polaris11"}, {"gfx803"},  GK_GFX803,  FEATURE_FAST_DENORMAL_F32},
  {{"gfx805"},    {"gfx805"},  GK_GFX805,  FEATURE_FAST_DENORMAL_F32},
  {{"tongapro"},  {"gfx805"},  GK_GFX805,  FEATURE_FAST_DENORMAL_F32},
  {{"gfx810"},    {"gfx810"},  GK_GFX810,  FEATURE_FAST_DENORMAL_F32|FEATURE_XNACK},
  {{"stoney"},    {"gfx810"},  GK_GFX810,  FEATURE_FAST_DENORMAL_F32|FEATURE_XNACK},
  {{"gfx900"},    {"gfx900"},  GK_GFX900,  FEATURE_FAST_FMA_F32|FEATURE_FAST_DENORMAL_F32|FEATURE_XNACK},
  {{"gfx902"},    {"gfx902"},  GK_GFX902,  FEATURE_FAST_FMA_F32|FEATURE_FAST_DENORMAL_F32|FEATURE_XNACK},
  {{"gfx904"},    {"gfx904"},  GK_GFX904,  FEATURE_FAST_FMA_F32|FEATURE_FAST_DENORMAL_F32|FEATURE_XNACK},
  {{"gfx906"},    {"gfx906"},  GK_GFX906,  FEATURE_FAST_FMA_F32|FEATURE_FAST_DENORMAL_F32|FEATURE_XNACK|FEATURE_SRAMECC},
  {{"gfx908"},    {"gfx908"},  GK_GFX908,  FEATURE_FAST_FMA_F32|FEATURE_FAST_DENORMAL_F32|FEATURE_XNACK|FEATURE_SRAMECC},
  {{"gfx909"},    {"gfx909"},  GK_GFX909,  FEATURE_FAST_FMA_F32|FEATURE_FAST_DENORMAL_F32|FEATURE_XNACK},
  {{"gfx90a"},    {"gfx90a"},  GK_GFX90A,  FEATURE_FAST_FMA_F32|FEATURE_FAST_DENORMAL_F32|FEATURE_XNACK|FEATURE_SRAMECC},
  {{"gfx90c"},    {"gfx90c"},  GK_GFX90C,  FEATURE_FAST_FMA_F32|FEATURE_FAST_DENORMAL_F32|FEATURE_XNACK},
  {{"gfx1010"},   {"gfx1010"}, GK_GFX1010, FEATURE_FAST_FMA_F32|FEATURE_FAST_DENORMAL_F32|FEATURE_WAVE32|FEATURE_XNACK},
  {{"gfx1011"},   {"gfx1011"}, GK_GFX1011, FEATURE_FAST_FMA_F32|FEATURE_FAST_DENORMAL_F32|FEATURE_WAVE32|FEATURE_XNACK},
  {{"gfx1012"},   {"gfx1012"}, GK_GFX1012, FEATURE_FAST_FMA_F32|FEATURE_FAST_DENORMAL_F32|FEATURE_WAVE32|FEATURE_XNACK},
  {{"gfx1013"},   {"gfx1013"}, GK_GFX1013, FEATURE_FAST_FMA_F32|FEATURE_FAST_DENORMAL_F32|FEATURE_WAVE32|FEATURE_XNACK},
  {{"gfx1030"},   {"gfx1030"}, GK_GFX1030, FEATURE_FAST_FMA_F32|FEATURE_FAST_DENORMAL_F32|FEATURE_WAVE32},
  {{"gfx1031"},   {"gfx1031"}, GK_GFX1031, FEATURE_FAST_FMA_F32|FEATURE_FAST_DENORMAL_F32|FEATURE_WAVE32},
  {{"gfx1032"},   {"gfx1032"}, GK_GFX1032, FEATURE_FAST_FMA_F32|FEATURE_FAST_DENORMAL_F32|FEATURE_WAVE32},
  {{"gfx1033"},   {"gfx1033"}, GK_GFX1033, FEATURE_FAST_FMA_F32|FEATURE_FAST_DENORMAL_F32|FEATURE_WAVE32},
  {{"gfx1034"},   {"gfx1034"}, GK_GFX1034, FEATURE_FAST_FMA_F32|FEATURE_FAST_DENORMAL_F32|FEATURE_WAVE32},
  {{"gfx1035"},   {"gfx1035"}, GK_GFX1035, FEATURE_FAST_FMA_F32|FEATURE_FAST_DENORMAL_F32|FEATURE_WAVE32},
};

const GPUInfo *getArchEntry(AMDGPU::GPUKind AK, ArrayRef<GPUInfo> Table) {
  GPUInfo Search = { {""}, {""}, AK, AMDGPU::FEATURE_NONE };

  auto I =
      llvm::lower_bound(Table, Search, [](const GPUInfo &A, const GPUInfo &B) {
        return A.Kind < B.Kind;
      });

  if (I == Table.end())
    return nullptr;
  return I;
}

} // namespace

StringRef llvm::AMDGPU::getArchNameAMDGCN(GPUKind AK) {
  if (const auto *Entry = getArchEntry(AK, AMDGCNGPUs))
    return Entry->CanonicalName;
  return "";
}

StringRef llvm::AMDGPU::getArchNameR600(GPUKind AK) {
  if (const auto *Entry = getArchEntry(AK, R600GPUs))
    return Entry->CanonicalName;
  return "";
}

AMDGPU::GPUKind llvm::AMDGPU::parseArchAMDGCN(StringRef CPU) {
  for (const auto &C : AMDGCNGPUs) {
    if (CPU == C.Name)
      return C.Kind;
  }

  return AMDGPU::GPUKind::GK_NONE;
}

AMDGPU::GPUKind llvm::AMDGPU::parseArchR600(StringRef CPU) {
  for (const auto &C : R600GPUs) {
    if (CPU == C.Name)
      return C.Kind;
  }

  return AMDGPU::GPUKind::GK_NONE;
}

unsigned AMDGPU::getArchAttrAMDGCN(GPUKind AK) {
  if (const auto *Entry = getArchEntry(AK, AMDGCNGPUs))
    return Entry->Features;
  return FEATURE_NONE;
}

unsigned AMDGPU::getArchAttrR600(GPUKind AK) {
  if (const auto *Entry = getArchEntry(AK, R600GPUs))
    return Entry->Features;
  return FEATURE_NONE;
}

void AMDGPU::fillValidArchListAMDGCN(SmallVectorImpl<StringRef> &Values) {
  // XXX: Should this only report unique canonical names?
  for (const auto &C : AMDGCNGPUs)
    Values.push_back(C.Name);
}

void AMDGPU::fillValidArchListR600(SmallVectorImpl<StringRef> &Values) {
  for (const auto &C : R600GPUs)
    Values.push_back(C.Name);
}

AMDGPU::IsaVersion AMDGPU::getIsaVersion(StringRef GPU) {
  AMDGPU::GPUKind AK = parseArchAMDGCN(GPU);
  if (AK == AMDGPU::GPUKind::GK_NONE) {
    if (GPU == "generic-hsa")
      return {7, 0, 0};
    if (GPU == "generic")
      return {6, 0, 0};
    return {0, 0, 0};
  }

  switch (AK) {
  case GK_GFX600:  return {6, 0, 0};
  case GK_GFX601:  return {6, 0, 1};
  case GK_GFX602:  return {6, 0, 2};
  case GK_GFX700:  return {7, 0, 0};
  case GK_GFX701:  return {7, 0, 1};
  case GK_GFX702:  return {7, 0, 2};
  case GK_GFX703:  return {7, 0, 3};
  case GK_GFX704:  return {7, 0, 4};
  case GK_GFX705:  return {7, 0, 5};
  case GK_GFX801:  return {8, 0, 1};
  case GK_GFX802:  return {8, 0, 2};
  case GK_GFX803:  return {8, 0, 3};
  case GK_GFX805:  return {8, 0, 5};
  case GK_GFX810:  return {8, 1, 0};
  case GK_GFX900:  return {9, 0, 0};
  case GK_GFX902:  return {9, 0, 2};
  case GK_GFX904:  return {9, 0, 4};
  case GK_GFX906:  return {9, 0, 6};
  case GK_GFX908:  return {9, 0, 8};
  case GK_GFX909:  return {9, 0, 9};
  case GK_GFX90A:  return {9, 0, 10};
  case GK_GFX90C:  return {9, 0, 12};
  case GK_GFX1010: return {10, 1, 0};
  case GK_GFX1011: return {10, 1, 1};
  case GK_GFX1012: return {10, 1, 2};
  case GK_GFX1013: return {10, 1, 3};
  case GK_GFX1030: return {10, 3, 0};
  case GK_GFX1031: return {10, 3, 1};
  case GK_GFX1032: return {10, 3, 2};
  case GK_GFX1033: return {10, 3, 3};
  case GK_GFX1034: return {10, 3, 4};
  case GK_GFX1035: return {10, 3, 5};
  default:         return {0, 0, 0};
  }
}

StringRef AMDGPU::getCanonicalArchName(const Triple &T, StringRef Arch) {
  assert(T.isAMDGPU());
  auto ProcKind = T.isAMDGCN() ? parseArchAMDGCN(Arch) : parseArchR600(Arch);
  if (ProcKind == GK_NONE)
    return StringRef();

  return T.isAMDGCN() ? getArchNameAMDGCN(ProcKind) : getArchNameR600(ProcKind);
}

namespace llvm {
namespace RISCV {

struct CPUInfo {
  StringLiteral Name;
  CPUKind Kind;
  unsigned Features;
  StringLiteral DefaultMarch;
  bool is64Bit() const { return (Features & FK_64BIT); }
};

constexpr CPUInfo RISCVCPUInfo[] = {
#define PROC(ENUM, NAME, FEATURES, DEFAULT_MARCH)                              \
  {NAME, CK_##ENUM, FEATURES, DEFAULT_MARCH},
#include "llvm/Support/RISCVTargetParser.def"
};

bool checkCPUKind(CPUKind Kind, bool IsRV64) {
  if (Kind == CK_INVALID)
    return false;
  return RISCVCPUInfo[static_cast<unsigned>(Kind)].is64Bit() == IsRV64;
}

bool checkTuneCPUKind(CPUKind Kind, bool IsRV64) {
  if (Kind == CK_INVALID)
    return false;
  return RISCVCPUInfo[static_cast<unsigned>(Kind)].is64Bit() == IsRV64;
}

CPUKind parseCPUKind(StringRef CPU) {
  return llvm::StringSwitch<CPUKind>(CPU)
#define PROC(ENUM, NAME, FEATURES, DEFAULT_MARCH) .Case(NAME, CK_##ENUM)
#include "llvm/Support/RISCVTargetParser.def"
      .Default(CK_INVALID);
}

StringRef resolveTuneCPUAlias(StringRef TuneCPU, bool IsRV64) {
  return llvm::StringSwitch<StringRef>(TuneCPU)
#define PROC_ALIAS(NAME, RV32, RV64) .Case(NAME, IsRV64 ? StringRef(RV64) : StringRef(RV32))
#include "llvm/Support/RISCVTargetParser.def"
      .Default(TuneCPU);
}

CPUKind parseTuneCPUKind(StringRef TuneCPU, bool IsRV64) {
  TuneCPU = resolveTuneCPUAlias(TuneCPU, IsRV64);

  return llvm::StringSwitch<CPUKind>(TuneCPU)
#define PROC(ENUM, NAME, FEATURES, DEFAULT_MARCH) .Case(NAME, CK_##ENUM)
#include "llvm/Support/RISCVTargetParser.def"
      .Default(CK_INVALID);
}

StringRef getMArchFromMcpu(StringRef CPU) {
  CPUKind Kind = parseCPUKind(CPU);
  return RISCVCPUInfo[static_cast<unsigned>(Kind)].DefaultMarch;
}

void fillValidCPUArchList(SmallVectorImpl<StringRef> &Values, bool IsRV64) {
  for (const auto &C : RISCVCPUInfo) {
    if (C.Kind != CK_INVALID && IsRV64 == C.is64Bit())
      Values.emplace_back(C.Name);
  }
}

void fillValidTuneCPUArchList(SmallVectorImpl<StringRef> &Values, bool IsRV64) {
  for (const auto &C : RISCVCPUInfo) {
    if (C.Kind != CK_INVALID && IsRV64 == C.is64Bit())
      Values.emplace_back(C.Name);
  }
#define PROC_ALIAS(NAME, RV32, RV64) Values.emplace_back(StringRef(NAME));
#include "llvm/Support/RISCVTargetParser.def"
}

// Get all features except standard extension feature
bool getCPUFeaturesExceptStdExt(CPUKind Kind,
                                std::vector<StringRef> &Features) {
  unsigned CPUFeatures = RISCVCPUInfo[static_cast<unsigned>(Kind)].Features;

  if (CPUFeatures == FK_INVALID)
    return false;

  if (CPUFeatures & FK_64BIT)
    Features.push_back("+64bit");
  else
    Features.push_back("-64bit");

  return true;
}

StringRef computeDefaultABIFromArch(const llvm::RISCVISAInfo &ISAInfo) {
  if (ISAInfo.getXLen() == 32) {
    if (ISAInfo.hasExtension("d"))
      return "ilp32d";
    if (ISAInfo.hasExtension("e"))
      return "ilp32e";
    return "ilp32";
  } else if (ISAInfo.getXLen() == 64) {
    if (ISAInfo.hasExtension("d"))
      return "lp64d";
    return "lp64";
  }
  llvm_unreachable("Invalid XLEN");
}

} // namespace RISCV
} // namespace llvm

// Parse a branch protection specification, which has the form
//   standard | none | [bti,pac-ret[+b-key,+leaf]*]
// Returns true on success, with individual elements of the specification
// returned in `PBP`. Returns false in error, with `Err` containing
// an erroneous part of the spec.
bool ARM::parseBranchProtection(StringRef Spec, ParsedBranchProtection &PBP,
                                StringRef &Err) {
  PBP = {"none", "a_key", false};
  if (Spec == "none")
    return true; // defaults are ok

  if (Spec == "standard") {
    PBP.Scope = "non-leaf";
    PBP.BranchTargetEnforcement = true;
    return true;
  }

  SmallVector<StringRef, 4> Opts;
  Spec.split(Opts, "+");
  for (int I = 0, E = Opts.size(); I != E; ++I) {
    StringRef Opt = Opts[I].trim();
    if (Opt == "bti") {
      PBP.BranchTargetEnforcement = true;
      continue;
    }
    if (Opt == "pac-ret") {
      PBP.Scope = "non-leaf";
      for (; I + 1 != E; ++I) {
        StringRef PACOpt = Opts[I + 1].trim();
        if (PACOpt == "leaf")
          PBP.Scope = "all";
        else if (PACOpt == "b-key")
          PBP.Key = "b_key";
        else
          break;
      }
      continue;
    }
    if (Opt == "")
      Err = "<empty>";
    else
      Err = Opt;
    return false;
  }

  return true;
}
