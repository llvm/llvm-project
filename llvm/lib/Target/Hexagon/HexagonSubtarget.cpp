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
using namespace llvm;

#define DEBUG_TYPE "hexagon-subtarget"

#define GET_SUBTARGETINFO_CTOR
#define GET_SUBTARGETINFO_TARGET_DESC
#include "HexagonGenSubtargetInfo.inc"

static cl::opt<bool>
EnableV3("enable-hexagon-v3", cl::Hidden,
         cl::desc("Enable Hexagon V3 instructions."));

static cl::opt<bool>
EnableMemOps(
    "enable-hexagon-memops",
    cl::Hidden, cl::ZeroOrMore, cl::ValueDisallowed, cl::init(true),
    cl::desc(
      "Generate V4 MEMOP in code generation for Hexagon target"));

static cl::opt<bool>
DisableMemOps(
    "disable-hexagon-memops",
    cl::Hidden, cl::ZeroOrMore, cl::ValueDisallowed, cl::init(false),
    cl::desc(
      "Do not generate V4 MEMOP in code generation for Hexagon target"));

static cl::opt<bool>
EnableIEEERndNear(
    "enable-hexagon-ieee-rnd-near",
    cl::Hidden, cl::ZeroOrMore, cl::init(false),
    cl::desc("Generate non-chopped conversion from fp to int."));

static cl::opt<bool> DisableHexagonMISched("disable-hexagon-misched",
      cl::Hidden, cl::ZeroOrMore, cl::init(false),
      cl::desc("Disable Hexagon MI Scheduling"));

HexagonSubtarget &
HexagonSubtarget::initializeSubtargetDependencies(StringRef CPU, StringRef FS) {
  // If the programmer has not specified a Hexagon version, default to -mv4.
  if (CPUString.empty())
    CPUString = "hexagonv4";

  if (CPUString == "hexagonv4") {
    HexagonArchVersion = V4;
  } else if (CPUString == "hexagonv5") {
    HexagonArchVersion = V5;
  } else {
    llvm_unreachable("Unrecognized Hexagon processor version");
  }

  ParseSubtargetFeatures(CPUString, FS);
  return *this;
}

HexagonSubtarget::HexagonSubtarget(const Triple &TT, StringRef CPU,
                                   StringRef FS, const TargetMachine &TM)
    : HexagonGenSubtargetInfo(TT, CPU, FS), CPUString(CPU),
      InstrInfo(initializeSubtargetDependencies(CPU, FS)), TLInfo(TM, *this),
      FrameLowering() {

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
}

// Pin the vtable to this file.
void HexagonSubtarget::anchor() {}

bool HexagonSubtarget::enableMachineScheduler() const {
  if (DisableHexagonMISched.getNumOccurrences())
    return !DisableHexagonMISched;
  return true;
}
