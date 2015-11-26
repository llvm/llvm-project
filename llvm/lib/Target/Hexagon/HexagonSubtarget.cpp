//===-- HexagonSubtarget.cpp - Hexagon Subtarget Information --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Hexagon specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "HexagonSubtarget.h"
#include "Hexagon.h"
#include "HexagonRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include <map>

using namespace llvm;

#define DEBUG_TYPE "hexagon-subtarget"

#define GET_SUBTARGETINFO_CTOR
#define GET_SUBTARGETINFO_TARGET_DESC
#include "HexagonGenSubtargetInfo.inc"

static cl::opt<bool> EnableMemOps("enable-hexagon-memops",
  cl::Hidden, cl::ZeroOrMore, cl::ValueDisallowed, cl::init(true),
  cl::desc("Generate V4 MEMOP in code generation for Hexagon target"));

static cl::opt<bool> DisableMemOps("disable-hexagon-memops",
  cl::Hidden, cl::ZeroOrMore, cl::ValueDisallowed, cl::init(false),
  cl::desc("Do not generate V4 MEMOP in code generation for Hexagon target"));

static cl::opt<bool> EnableIEEERndNear("enable-hexagon-ieee-rnd-near",
  cl::Hidden, cl::ZeroOrMore, cl::init(false),
  cl::desc("Generate non-chopped conversion from fp to int."));

static cl::opt<bool> EnableBSBSched("enable-bsb-sched",
  cl::Hidden, cl::ZeroOrMore, cl::init(true));

static cl::opt<bool> EnableHexagonHVXDouble("enable-hexagon-hvx-double",
  cl::Hidden, cl::ZeroOrMore, cl::init(false),
  cl::desc("Enable Hexagon Double Vector eXtensions"));

static cl::opt<bool> EnableHexagonHVX("enable-hexagon-hvx",
  cl::Hidden, cl::ZeroOrMore, cl::init(false),
  cl::desc("Enable Hexagon Vector eXtensions"));

static cl::opt<bool> DisableHexagonMISched("disable-hexagon-misched",
  cl::Hidden, cl::ZeroOrMore, cl::init(false),
  cl::desc("Disable Hexagon MI Scheduling"));

void HexagonSubtarget::initializeEnvironment() {
  UseMemOps = false;
  ModeIEEERndNear = false;
  UseBSBScheduling = false;
}

HexagonSubtarget &
HexagonSubtarget::initializeSubtargetDependencies(StringRef CPU, StringRef FS) {
  // Default architecture.
  if (CPUString.empty())
    CPUString = "hexagonv60";

  static std::map<StringRef, HexagonArchEnum> CpuTable {
    { "hexagonv4", V4 },
    { "hexagonv5", V5 },
    { "hexagonv55", V55 },
    { "hexagonv60", V60 },
  };

  auto foundIt = CpuTable.find(CPUString);
  if (foundIt != CpuTable.end())
    HexagonArchVersion = foundIt->second;
  else
    llvm_unreachable("Unrecognized Hexagon processor version");

  UseHVXOps = false;
  UseHVXDblOps = false;
  ParseSubtargetFeatures(CPUString, FS);

  if (EnableHexagonHVX.getPosition())
    UseHVXOps = EnableHexagonHVX;
  if (EnableHexagonHVXDouble.getPosition())
    UseHVXDblOps = EnableHexagonHVXDouble;

  return *this;
}

HexagonSubtarget::HexagonSubtarget(const Triple &TT, StringRef CPU,
                                   StringRef FS, const TargetMachine &TM)
    : HexagonGenSubtargetInfo(TT, CPU, FS), CPUString(CPU),
      InstrInfo(initializeSubtargetDependencies(CPU, FS)), TLInfo(TM, *this),
      FrameLowering() {

  initializeEnvironment();

  // Initialize scheduling itinerary for the specified CPU.
  InstrItins = getInstrItineraryForCPU(CPUString);

  // UseMemOps on by default unless disabled explicitly
  if (DisableMemOps)
    UseMemOps = false;
  else if (EnableMemOps)
    UseMemOps = true;
  else
    UseMemOps = false;

  if (EnableIEEERndNear)
    ModeIEEERndNear = true;
  else
    ModeIEEERndNear = false;

  UseBSBScheduling = hasV60TOps() && EnableBSBSched;
}

// Pin the vtable to this file.
void HexagonSubtarget::anchor() {}

bool HexagonSubtarget::enableMachineScheduler() const {
  if (DisableHexagonMISched.getNumOccurrences())
    return !DisableHexagonMISched;
  return true;
}
