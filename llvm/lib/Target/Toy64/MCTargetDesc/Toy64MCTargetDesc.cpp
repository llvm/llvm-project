//===-- Toy64MCTargetDesc.cpp - Toy64 Target Descriptions -----------------===//

#include "Toy64MCTargetDesc.h"
#include "Toy64InstPrinter.h"
#include "Toy64MCAsmInfo.h"

#include "TargetInfo/Toy64TargetInfo.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"
#include "llvm/TargetParser/Triple.h"

#include <string>

// Generate the implementation of InitToy64MCInstrInfo().
#define GET_INSTRINFO_MC_DESC
#include "Toy64GenInstrInfo.inc"

// Generate the implementation of createToy64MCSubtargetInfoImpl().
#define GET_SUBTARGETINFO_MC_DESC
#include "Toy64GenSubtargetInfo.inc"

// Generate the implementation of InitToy64MCRegisterInfo().
#define GET_REGINFO_MC_DESC
#include "Toy64GenRegisterInfo.inc"

using namespace llvm;

//===----------------------------------------------------------------------===//
// MC instruction information
//===----------------------------------------------------------------------===//

static MCInstrInfo *createToy64MCInstrInfo() {
  auto *X = new MCInstrInfo();
  InitToy64MCInstrInfo(X);
  return X;
}

//===----------------------------------------------------------------------===//
// MC register information
//===----------------------------------------------------------------------===//

static MCRegisterInfo *createToy64MCRegisterInfo(const Triple &TT) {
  auto *X = new MCRegisterInfo();

  // X1 is Toy64's return-address register.
  InitToy64MCRegisterInfo(X, Toy64::X1);

  return X;
}

//===----------------------------------------------------------------------===//
// MC subtarget information
//===----------------------------------------------------------------------===//

static MCSubtargetInfo *
createToy64MCSubtargetInfo(const Triple &TT, StringRef CPU, StringRef FS) {
  std::string CPUName = CPU.str();

  if (CPUName.empty())
    CPUName = "generic";

  return createToy64MCSubtargetInfoImpl(
      TT,
      CPUName,
      /*TuneCPU=*/CPUName,
      FS);
}

//===----------------------------------------------------------------------===//
// MC instruction printer
//===----------------------------------------------------------------------===//

static MCInstPrinter *
createToy64MCInstPrinter(
    const Triple &TT,
    unsigned SyntaxVariant,
    const MCAsmInfo &MAI,
    const MCInstrInfo &MII,
    const MCRegisterInfo &MRI) {

  // Toy64 currently supports one assembly syntax.
  if (SyntaxVariant != 0)
    return nullptr;

  return new Toy64InstPrinter(
      MAI,
      MII,
      MRI);
}


//===----------------------------------------------------------------------===//
// Target MC registration
//===----------------------------------------------------------------------===//

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeToy64TargetMC() {
  Target &T = getTheToy64Target();

  RegisterMCAsmInfo<Toy64MCAsmInfo> X(T);

  TargetRegistry::RegisterMCInstrInfo(
      T, createToy64MCInstrInfo);

  TargetRegistry::RegisterMCRegInfo(
      T, createToy64MCRegisterInfo);

  TargetRegistry::RegisterMCSubtargetInfo(
      T, createToy64MCSubtargetInfo);

  TargetRegistry::RegisterMCInstPrinter(
      T, createToy64MCInstPrinter);
}