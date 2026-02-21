//===- NVPTXSubtarget.cpp - NVPTX Subtarget Information -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the NVPTX specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "NVPTXSubtarget.h"
#include "NVPTXSelectionDAGInfo.h"
#include "NVPTXTargetMachine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"

using namespace llvm;

#define DEBUG_TYPE "nvptx-subtarget"

#define GET_SUBTARGETINFO_ENUM
#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "NVPTXGenSubtargetInfo.inc"

static cl::opt<bool>
    NoF16Math("nvptx-no-f16-math", cl::Hidden,
              cl::desc("NVPTX Specific: Disable generation of f16 math ops."),
              cl::init(false));

static cl::opt<bool> NoF32x2("nvptx-no-f32x2", cl::Hidden,
                             cl::desc("NVPTX Specific: Disable generation of "
                                      "f32x2 instructions and registers."),
                             cl::init(false));

// FullSmVersion encoding helpers: SM * 10 + suffix offset
// (0 = base, 2 = 'f', 3 = 'a').
static constexpr unsigned SM(unsigned Version) { return Version * 10; }
static constexpr unsigned SMF(unsigned Version) { return SM(Version) + 2; }
static constexpr unsigned SMA(unsigned Version) { return SM(Version) + 3; }

// Pin the vtable to this file.
void NVPTXSubtarget::anchor() {}

// Returns the minimum PTX version required for a given SM target.
// This must be kept in sync with the "Supported Targets" column of the
// "PTX Release History" table in the PTX ISA documentation:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#release-notes-ptx-release-history
//
// Note: LLVM's minimum supported PTX version is 3.2 (see FeaturePTX in
// NVPTX.td), so older SMs that supported earlier PTX versions instead use 3.2
// as their effective minimum.
static unsigned getMinPTXVersionForSM(unsigned FullSmVersion) {
  switch (FullSmVersion) {
  case SM(20):
  case SM(21):
  case SM(30):
  case SM(35):
    return 32;
  case SM(32):
  case SM(50):
    return 40;
  case SM(37):
  case SM(52):
    return 41;
  case SM(53):
    return 42;
  case SM(60):
  case SM(61):
  case SM(62):
    return 50;
  case SM(70):
    return 60;
  case SM(72):
    return 61;
  case SM(75):
    return 63;
  case SM(80):
    return 70;
  case SM(86):
    return 71;
  case SM(87):
    return 74;
  case SM(89):
  case SM(90):
    return 78;
  case SMA(90):
    return 80;
  case SM(100):
  case SMA(100):
  case SM(101):
  case SMA(101):
    return 86;
  case SM(120):
  case SMA(120):
    return 87;
  case SMF(100):
  case SMF(101):
  case SM(103):
  case SMF(103):
  case SMA(103):
  case SMF(120):
  case SM(121):
  case SMF(121):
  case SMA(121):
    return 88;
  case SM(88):
  case SM(110):
  case SMF(110):
  case SMA(110):
    return 90;
  default:
    llvm_unreachable("Unknown SM version");
  }
}

NVPTXSubtarget &NVPTXSubtarget::initializeSubtargetDependencies(StringRef CPU,
                                                                StringRef FS) {
  TargetName = std::string(CPU);

  ParseSubtargetFeatures(getTargetName(), /*TuneCPU=*/getTargetName(), FS);

  // Re-map SM version numbers, SmVersion carries the regular SMs which do
  // have relative order, while FullSmVersion allows distinguishing sm_90 from
  // sm_90a, which would *not* be a subset of sm_91.
  SmVersion = getSmVersion();

  unsigned MinPTX = getMinPTXVersionForSM(FullSmVersion);

  if (PTXVersion == 0) {
    // User didn't request a specific PTX version; use the minimum for this SM.
    PTXVersion = MinPTX;
  } else if (PTXVersion < MinPTX) {
    // User explicitly requested an insufficient PTX version.
    reportFatalUsageError(
        formatv("PTX version {0}.{1} does not support target '{2}'. "
                "Minimum required PTX version is {3}.{4}. "
                "Either remove the PTX version to use the default, "
                "or increase it to at least {3}.{4}.",
                PTXVersion / 10, PTXVersion % 10, getTargetName(), MinPTX / 10,
                MinPTX % 10));
  }

  return *this;
}

NVPTXSubtarget::NVPTXSubtarget(const Triple &TT, const std::string &CPU,
                               const std::string &FS,
                               const NVPTXTargetMachine &TM)
    : NVPTXGenSubtargetInfo(TT, CPU, /*TuneCPU*/ CPU, FS), PTXVersion(0),
      FullSmVersion(200), SmVersion(getSmVersion()),
      InstrInfo(initializeSubtargetDependencies(CPU, FS)), TLInfo(TM, *this) {
  TSInfo = std::make_unique<NVPTXSelectionDAGInfo>();
}

NVPTXSubtarget::~NVPTXSubtarget() = default;

const SelectionDAGTargetInfo *NVPTXSubtarget::getSelectionDAGInfo() const {
  return TSInfo.get();
}

bool NVPTXSubtarget::hasPTXWithFamilySMs(unsigned PTXVersion,
                                         ArrayRef<unsigned> SMVersions) const {
  unsigned PTXVer = getPTXVersion();
  if (!hasFamilySpecificFeatures() || PTXVer < PTXVersion)
    return false;

  unsigned SMVer = getSmVersion();
  return llvm::any_of(SMVersions, [&](unsigned SM) {
    // sm_101 is a different family, never group it with sm_10x.
    if (SMVer == 101 || SM == 101)
      return SMVer == SM &&
             // PTX 9.0 and later renamed sm_101 to sm_110, so sm_101 is not
             // supported.
             !(PTXVer >= 90 && SMVer == 101);

    return getSmFamilyVersion() == SM / 10 && SMVer >= SM;
  });
}

bool NVPTXSubtarget::hasPTXWithAccelSMs(unsigned PTXVersion,
                                        ArrayRef<unsigned> SMVersions) const {
  unsigned PTXVer = getPTXVersion();
  if (!hasArchAccelFeatures() || PTXVer < PTXVersion)
    return false;

  unsigned SMVer = getSmVersion();
  return llvm::any_of(SMVersions, [&](unsigned SM) {
    return SMVer == SM &&
           // PTX 9.0 and later renamed sm_101 to sm_110, so sm_101 is not
           // supported.
           !(PTXVer >= 90 && SMVer == 101);
  });
}

bool NVPTXSubtarget::allowFP16Math() const {
  return hasFP16Math() && NoF16Math == false;
}

bool NVPTXSubtarget::hasF32x2Instructions() const {
  return SmVersion >= 100 && PTXVersion >= 86 && !NoF32x2;
}

bool NVPTXSubtarget::hasNativeBF16Support(int Opcode) const {
  if (!hasBF16Math())
    return false;

  switch (Opcode) {
  // Several BF16 instructions are available on sm_90 only.
  case ISD::FADD:
  case ISD::FMUL:
  case ISD::FSUB:
  case ISD::SELECT:
  case ISD::SELECT_CC:
  case ISD::SETCC:
  case ISD::FEXP2:
  case ISD::FCEIL:
  case ISD::FFLOOR:
  case ISD::FNEARBYINT:
  case ISD::FRINT:
  case ISD::FROUNDEVEN:
  case ISD::FTRUNC:
    return getSmVersion() >= 90 && getPTXVersion() >= 78;
  // Several BF16 instructions are available on sm_80 only.
  case ISD::FMINNUM:
  case ISD::FMAXNUM:
  case ISD::FMAXNUM_IEEE:
  case ISD::FMINNUM_IEEE:
  case ISD::FMAXIMUM:
  case ISD::FMINIMUM:
    return getSmVersion() >= 80 && getPTXVersion() >= 70;
  }
  return true;
}

void NVPTXSubtarget::failIfClustersUnsupported(
    std::string const &FailureMessage) const {
  if (hasClusters())
    return;

  report_fatal_error(formatv(
      "NVPTX SM architecture \"{}\" and PTX version \"{}\" do not support {}. "
      "Requires SM >= 90 and PTX >= 78.",
      getFullSmVersion(), PTXVersion, FailureMessage));
}
