//===-- SystemZSubtarget.cpp - SystemZ subtarget information --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SystemZSubtarget.h"
#include "MCTargetDesc/SystemZMCTargetDesc.h"
#include "llvm/IR/GlobalValue.h"

using namespace llvm;

#define DEBUG_TYPE "systemz-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "SystemZGenSubtargetInfo.inc"

// Pin the vtable to this file.
void SystemZSubtarget::anchor() {}

SystemZSubtarget &
SystemZSubtarget::initializeSubtargetDependencies(StringRef CPU, StringRef FS) {
  std::string CPUName = CPU;
  if (CPUName.empty())
    CPUName = "generic";
  // Parse features string.
  ParseSubtargetFeatures(CPUName, FS);
  return *this;
}

SystemZSubtarget::SystemZSubtarget(const Triple &TT, const std::string &CPU,
                                   const std::string &FS,
                                   const TargetMachine &TM)
    : SystemZGenSubtargetInfo(TT, CPU, FS), HasDistinctOps(false),
      HasLoadStoreOnCond(false), HasHighWord(false), HasFPExtension(false),
      HasPopulationCount(false), HasMessageSecurityAssist4(false),
      HasFastSerialization(false), HasInterlockedAccess1(false),
      HasMiscellaneousExtensions(false),
      HasExecutionHint(false), HasLoadAndTrap(false),
      HasTransactionalExecution(false), HasProcessorAssist(false),
      HasVector(false), HasLoadStoreOnCond2(false),
      HasLoadAndZeroRightmostByte(false), HasMessageSecurityAssist5(false),
      TargetTriple(TT), InstrInfo(initializeSubtargetDependencies(CPU, FS)),
      TLInfo(TM, *this), TSInfo(), FrameLowering() {}

bool SystemZSubtarget::isPC32DBLSymbol(const GlobalValue *GV,
                                       CodeModel::Model CM) const {
  // PC32DBL accesses require the low bit to be clear.  Note that a zero
  // value selects the default alignment and is therefore OK.
  if (GV->getAlignment() == 1)
    return false;

  // For the small model, all locally-binding symbols are in range.
  if (CM == CodeModel::Small)
    return TLInfo.getTargetMachine().shouldAssumeDSOLocal(*GV->getParent(), GV);

  // For Medium and above, assume that the symbol is not within the 4GB range.
  // Taking the address of locally-defined text would be OK, but that
  // case isn't easy to detect.
  return false;
}
