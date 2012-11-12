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

#define GET_SUBTARGETINFO_CTOR
#define GET_SUBTARGETINFO_TARGET_DESC
#include "HexagonGenSubtargetInfo.inc"

static cl::opt<bool>
EnableV3("enable-hexagon-v3", cl::Hidden,
         cl::desc("Enable Hexagon V3 instructions."));

static cl::opt<bool>
EnableMemOps(
    "enable-hexagon-memops",
    cl::Hidden, cl::ZeroOrMore, cl::ValueDisallowed,
    cl::desc("Generate V4 memop instructions."));

static cl::opt<bool>
EnableIEEERndNear(
    "enable-hexagon-ieee-rnd-near",
    cl::Hidden, cl::ZeroOrMore, cl::init(false),
    cl::desc("Generate non-chopped conversion from fp to int."));

HexagonSubtarget::HexagonSubtarget(StringRef TT, StringRef CPU, StringRef FS):
  HexagonGenSubtargetInfo(TT, CPU, FS),
  CPUString(CPU.str()) {

  // If the programmer has not specified a Hexagon version, default to -mv4.
  if (CPUString.empty())
    CPUString = "hexagonv4";

  if (CPUString == "hexagonv2") {
    HexagonArchVersion = V2;
  } else if (CPUString == "hexagonv3") {
    EnableV3 = true;
    HexagonArchVersion = V3;
  } else if (CPUString == "hexagonv4") {
    HexagonArchVersion = V4;
  } else if (CPUString == "hexagonv5") {
    HexagonArchVersion = V5;
  } else {
    llvm_unreachable("Unrecognized Hexagon processor version");
  }

  ParseSubtargetFeatures(CPUString, FS);

  // Initialize scheduling itinerary for the specified CPU.
  InstrItins = getInstrItineraryForCPU(CPUString);

  if (EnableMemOps)
    UseMemOps = true;
  else
    UseMemOps = false;

  if (EnableIEEERndNear)
    ModeIEEERndNear = true;
  else
    ModeIEEERndNear = false;
}

