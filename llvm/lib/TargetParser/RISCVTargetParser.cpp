//===-- RISCVTargetParser.cpp - Parser for target features ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a target parser to recognise hardware features
// for RISC-V CPUs.
//
//===----------------------------------------------------------------------===//

#include "llvm/TargetParser/RISCVTargetParser.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/StringTable.h"
#include "llvm/TargetParser/RISCVISAInfo.h"

namespace llvm {
namespace RISCV {

char ParserError::ID = 0;
char ParserWarning::ID = 0;

enum CPUKind : unsigned {
#define PROC(ENUM, NAME, DEFAULT_MARCH, FAST_SCALAR_UNALIGN,                   \
             FAST_VECTOR_UNALIGN, MVENDORID, MARCHID, MIMPID)                  \
  CK_##ENUM,
#define TUNE_PROC(ENUM, NAME) CK_##ENUM,
#include "llvm/TargetParser/RISCVTargetParserDef.inc"
};

constexpr CPUInfo RISCVCPUInfo[] = {
#define PROC(ENUM, NAME, DEFAULT_MARCH, FAST_SCALAR_UNALIGN,                   \
             FAST_VECTOR_UNALIGN, MVENDORID, MARCHID, MIMPID)                  \
  {                                                                            \
      NAME,                                                                    \
      DEFAULT_MARCH,                                                           \
      FAST_SCALAR_UNALIGN,                                                     \
      FAST_VECTOR_UNALIGN,                                                     \
      {MVENDORID, MARCHID, MIMPID},                                            \
  },
#include "llvm/TargetParser/RISCVTargetParserDef.inc"
};

static const CPUInfo *getCPUInfoByName(StringRef CPU) {
  for (auto &C : RISCVCPUInfo)
    if (C.Name == CPU)
      return &C;
  return nullptr;
}

bool hasFastScalarUnalignedAccess(StringRef CPU) {
  const CPUInfo *Info = getCPUInfoByName(CPU);
  return Info && Info->FastScalarUnalignedAccess;
}

bool hasFastVectorUnalignedAccess(StringRef CPU) {
  const CPUInfo *Info = getCPUInfoByName(CPU);
  return Info && Info->FastVectorUnalignedAccess;
}

bool hasValidCPUModel(StringRef CPU) { return getCPUModel(CPU).isValid(); }

CPUModel getCPUModel(StringRef CPU) {
  const CPUInfo *Info = getCPUInfoByName(CPU);
  if (!Info)
    return {0, 0, 0};
  return Info->Model;
}

StringRef getCPUNameFromCPUModel(const CPUModel &Model) {
  if (!Model.isValid())
    return "";

  for (auto &C : RISCVCPUInfo)
    if (C.Model == Model)
      return C.Name;
  return "";
}

bool parseCPU(StringRef CPU, bool IsRV64) {
  const CPUInfo *Info = getCPUInfoByName(CPU);

  if (!Info)
    return false;
  return Info->is64Bit() == IsRV64;
}

bool parseTuneCPU(StringRef TuneCPU, bool IsRV64) {
  std::optional<CPUKind> Kind =
      llvm::StringSwitch<std::optional<CPUKind>>(TuneCPU)
#define TUNE_PROC(ENUM, NAME) .Case(NAME, CK_##ENUM)
  #include "llvm/TargetParser/RISCVTargetParserDef.inc"
      .Default(std::nullopt);

  if (Kind.has_value())
    return true;

  // Fallback to parsing as a CPU.
  return parseCPU(TuneCPU, IsRV64);
}

StringRef getMArchFromMcpu(StringRef CPU) {
  const CPUInfo *Info = getCPUInfoByName(CPU);
  if (!Info)
    return "";
  return Info->DefaultMarch;
}

void fillValidCPUArchList(SmallVectorImpl<StringRef> &Values, bool IsRV64) {
  for (const auto &C : RISCVCPUInfo) {
    if (IsRV64 == C.is64Bit())
      Values.emplace_back(C.Name);
  }
}

void fillValidTuneCPUArchList(SmallVectorImpl<StringRef> &Values, bool IsRV64) {
  for (const auto &C : RISCVCPUInfo) {
    if (IsRV64 == C.is64Bit())
      Values.emplace_back(C.Name);
  }
#define TUNE_PROC(ENUM, NAME) Values.emplace_back(StringRef(NAME));
#include "llvm/TargetParser/RISCVTargetParserDef.inc"
}

// This function is currently used by IREE, so it's not dead code.
void getFeaturesForCPU(StringRef CPU,
                       SmallVectorImpl<std::string> &EnabledFeatures,
                       bool NeedPlus) {
  StringRef MarchFromCPU = llvm::RISCV::getMArchFromMcpu(CPU);
  if (MarchFromCPU == "")
    return;

  EnabledFeatures.clear();
  auto RII = RISCVISAInfo::parseArchString(
      MarchFromCPU, /* EnableExperimentalExtension */ true);

  if (llvm::errorToBool(RII.takeError()))
    return;

  std::vector<std::string> FeatStrings =
      (*RII)->toFeatures(/* AddAllExtensions */ false);
  for (const auto &F : FeatStrings)
    if (NeedPlus)
      EnabledFeatures.push_back(F);
    else
      EnabledFeatures.push_back(F.substr(1));
}

class RISCVTuneFeatureLookupTable {
  struct RISCVTuneFeature {
    unsigned PosIdx;
    unsigned NegIdx;
    unsigned FeatureIdx;
  };

  struct RISCVImpliedTuneFeature {
    unsigned FeatureIdx;
    unsigned ImpliedFeatureIdx;
  };

#define GET_TUNE_FEATURES
#include "llvm/TargetParser/RISCVTargetParserDef.inc"

  // Positive directive name -> Feature name
  StringMap<StringRef> PositiveMap;
  // Negative directive name -> Feature name
  StringMap<StringRef> NegativeMap;

  StringMap<SmallVector<StringRef>> ImpliedFeatureMap;
  StringMap<SmallVector<StringRef>> InvImpliedFeatureMap;

public:
  static void getAllTuneFeatures(SmallVectorImpl<StringRef> &Features) {
    for (const auto &TuneFeature : TuneFeatures)
      Features.push_back(TuneFeatureStrings[TuneFeature.FeatureIdx]);
  }

  RISCVTuneFeatureLookupTable() {
    for (const auto &TuneFeature : TuneFeatures) {
      StringRef PosDirective = TuneFeatureStrings[TuneFeature.PosIdx];
      StringRef NegDirective = TuneFeatureStrings[TuneFeature.NegIdx];
      StringRef FeatureName = TuneFeatureStrings[TuneFeature.FeatureIdx];
      PositiveMap[PosDirective] = FeatureName;
      NegativeMap[NegDirective] = FeatureName;
    }

    for (const auto &Imp : ImpliedTuneFeatures) {
      StringRef Feature = TuneFeatureStrings[Imp.FeatureIdx];
      StringRef ImpliedFeature = TuneFeatureStrings[Imp.ImpliedFeatureIdx];
      ImpliedFeatureMap[Feature].push_back(ImpliedFeature);
      InvImpliedFeatureMap[ImpliedFeature].push_back(Feature);
    }
  }

  /// Returns {Feature name, Is positive or not}, or empty feature name
  /// if not found.
  std::pair<StringRef, bool> getFeature(StringRef DirectiveName) const {
    auto It = PositiveMap.find(DirectiveName);
    if (It != PositiveMap.end())
      return {It->getValue(), /*IsPositive=*/true};

    return {NegativeMap.lookup(DirectiveName), /*IsPositive=*/false};
  }

  /// Returns the implied features, or empty ArrayRef if not found. Note:
  /// ImpliedFeatureMap / InvImpliedFeatureMap are the owner of these implied
  /// feature list, so we can just return the ArrayRef.
  ArrayRef<StringRef> featureImplies(StringRef FeatureName,
                                     bool Inverse = false) const {
    const auto &Map = Inverse ? InvImpliedFeatureMap : ImpliedFeatureMap;
    auto It = Map.find(FeatureName);
    if (It == Map.end())
      return {};
    return It->second;
  }
};

void getAllTuneFeatures(SmallVectorImpl<StringRef> &Features) {
  RISCVTuneFeatureLookupTable::getAllTuneFeatures(Features);
}

Error parseTuneFeatureString(StringRef TFString,
                             SmallVectorImpl<std::string> &ResFeatures) {
  using SmallStringSet = SmallSet<StringRef, 4>;
  RISCVTuneFeatureLookupTable TFLookup;

  // Do not create ParserWarning right away. Instead, we store the warning
  // message until the last moment.
  std::string WarningMsg;

  TFString = TFString.trim();
  // Note: StringSet is not really ergnomic to use in this case here.
  SmallStringSet PositiveFeatures;
  SmallStringSet NegativeFeatures;
  // Phase 1: Collect explicit features.
  StringRef DirectiveStr;
  do {
    std::tie(DirectiveStr, TFString) = TFString.split(",");
    auto [FeatureName, IsPositive] = TFLookup.getFeature(DirectiveStr);
    if (FeatureName.empty()) {
      raw_string_ostream SS(WarningMsg);
      SS << "unrecognized tune feature directive '" << DirectiveStr << "'";
      continue;
    }

    auto &Features = IsPositive ? PositiveFeatures : NegativeFeatures;
    if (!Features.insert(FeatureName).second)
      return make_error<ParserError>(
          "cannot specify more than one instance of '" + Twine(DirectiveStr) +
          "'");
  } while (!TFString.empty());

  auto Intersection =
      llvm::set_intersection(PositiveFeatures, NegativeFeatures);
  if (!Intersection.empty()) {
    std::string IntersectedStr = join(Intersection, "', '");
    return make_error<ParserError>("Feature(s) '" + Twine(IntersectedStr) +
                                   "' cannot appear in both "
                                   "positive and negative directives");
  }

  // Phase 2: Derive implied features.
  SmallStringSet DerivedPosFeatures;
  SmallStringSet DerivedNegFeatures;
  for (StringRef PF : PositiveFeatures) {
    if (auto FeatureList = TFLookup.featureImplies(PF); !FeatureList.empty())
      DerivedPosFeatures.insert_range(FeatureList);
  }
  for (StringRef NF : NegativeFeatures) {
    if (auto FeatureList = TFLookup.featureImplies(NF, /*Inverse=*/true);
        !FeatureList.empty())
      DerivedNegFeatures.insert_range(FeatureList);
  }
  PositiveFeatures.insert_range(DerivedPosFeatures);
  NegativeFeatures.insert_range(DerivedNegFeatures);

  Intersection = llvm::set_intersection(PositiveFeatures, NegativeFeatures);
  if (!Intersection.empty()) {
    std::string IntersectedStr = join(Intersection, "', '");
    return make_error<ParserError>("Feature(s) '" + Twine(IntersectedStr) +
                                   "' were implied by both "
                                   "positive and negative directives");
  }

  // Export the result.
  const std::string PosPrefix("+");
  const std::string NegPrefix("-");
  for (StringRef PF : PositiveFeatures)
    ResFeatures.emplace_back(PosPrefix + PF.str());
  for (StringRef NF : NegativeFeatures)
    ResFeatures.emplace_back(NegPrefix + NF.str());

  if (WarningMsg.empty())
    return Error::success();
  else
    return make_error<ParserWarning>(WarningMsg);
}
} // namespace RISCV

namespace RISCVVType {
// Encode VTYPE into the binary format used by the the VSETVLI instruction which
// is used by our MC layer representation.
//
// Bits | Name       | Description
// -----+------------+------------------------------------------------
// 8    | altfmt     | Alternative format for bf16/ofp8
// 7    | vma        | Vector mask agnostic
// 6    | vta        | Vector tail agnostic
// 5:3  | vsew[2:0]  | Standard element width (SEW) setting
// 2:0  | vlmul[2:0] | Vector register group multiplier (LMUL) setting
unsigned encodeVTYPE(VLMUL VLMul, unsigned SEW, bool TailAgnostic,
                     bool MaskAgnostic, bool AltFmt) {
  assert(isValidSEW(SEW) && "Invalid SEW");
  unsigned VLMulBits = static_cast<unsigned>(VLMul);
  unsigned VSEWBits = encodeSEW(SEW);
  unsigned VTypeI = (VSEWBits << 3) | (VLMulBits & 0x7);
  if (TailAgnostic)
    VTypeI |= 0x40;
  if (MaskAgnostic)
    VTypeI |= 0x80;
  if (AltFmt)
    VTypeI |= 0x100;

  return VTypeI;
}

unsigned encodeXSfmmVType(unsigned SEW, unsigned Widen, bool AltFmt) {
  assert(isValidSEW(SEW) && "Invalid SEW");
  assert((Widen == 1 || Widen == 2 || Widen == 4) && "Invalid Widen");
  unsigned VSEWBits = encodeSEW(SEW);
  unsigned TWiden = Log2_32(Widen) + 1;
  unsigned VTypeI = (VSEWBits << 3) | AltFmt << 8 | TWiden << 9;
  return VTypeI;
}

std::pair<unsigned, bool> decodeVLMUL(VLMUL VLMul) {
  switch (VLMul) {
  default:
    llvm_unreachable("Unexpected LMUL value!");
  case LMUL_1:
  case LMUL_2:
  case LMUL_4:
  case LMUL_8:
    return std::make_pair(1 << static_cast<unsigned>(VLMul), false);
  case LMUL_F2:
  case LMUL_F4:
  case LMUL_F8:
    return std::make_pair(1 << (8 - static_cast<unsigned>(VLMul)), true);
  }
}

void printVType(unsigned VType, raw_ostream &OS) {
  unsigned Sew = getSEW(VType);
  OS << "e" << Sew;

  bool AltFmt = RISCVVType::isAltFmt(VType);
  if (AltFmt)
    OS << "alt";

  unsigned LMul;
  bool Fractional;
  std::tie(LMul, Fractional) = decodeVLMUL(getVLMUL(VType));

  if (Fractional)
    OS << ", mf";
  else
    OS << ", m";
  OS << LMul;

  if (isTailAgnostic(VType))
    OS << ", ta";
  else
    OS << ", tu";

  if (isMaskAgnostic(VType))
    OS << ", ma";
  else
    OS << ", mu";
}

void printXSfmmVType(unsigned VType, raw_ostream &OS) {
  OS << "e" << getSEW(VType) << ", w" << getXSfmmWiden(VType);
}

unsigned getSEWLMULRatio(unsigned SEW, VLMUL VLMul) {
  unsigned LMul;
  bool Fractional;
  std::tie(LMul, Fractional) = decodeVLMUL(VLMul);

  // Convert LMul to a fixed point value with 3 fractional bits.
  LMul = Fractional ? (8 / LMul) : (LMul * 8);

  assert(SEW >= 8 && "Unexpected SEW value");
  return (SEW * 8) / LMul;
}

std::optional<VLMUL> getSameRatioLMUL(unsigned SEW, VLMUL VLMul, unsigned EEW) {
  unsigned Ratio = RISCVVType::getSEWLMULRatio(SEW, VLMul);
  unsigned EMULFixedPoint = (EEW * 8) / Ratio;
  bool Fractional = EMULFixedPoint < 8;
  unsigned EMUL = Fractional ? 8 / EMULFixedPoint : EMULFixedPoint / 8;
  if (!isValidLMUL(EMUL, Fractional))
    return std::nullopt;
  return RISCVVType::encodeLMUL(EMUL, Fractional);
}

} // namespace RISCVVType

} // namespace llvm
