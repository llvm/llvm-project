//===-- MipsMachineFunctionInfo.cpp - Private data used for Mips ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/MipsBaseInfo.h"
#include "MipsInstrInfo.h"
#include "MipsMachineFunction.h"
#include "MipsSubtarget.h"
#include "MipsTargetMachine.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<bool>
FixGlobalBaseReg("mips-fix-global-base-reg", cl::Hidden, cl::init(true),
                 cl::desc("Always use $gp as the global base register."));

MipsFunctionInfo::~MipsFunctionInfo() {}

bool MipsFunctionInfo::globalBaseRegSet() const {
  return GlobalBaseReg;
}

unsigned MipsFunctionInfo::getGlobalBaseReg() {
  // Return if it has already been initialized.
  if (GlobalBaseReg)
    return GlobalBaseReg;

  MipsSubtarget const &STI =
      static_cast<const MipsSubtarget &>(MF.getSubtarget());

  const TargetRegisterClass *RC =
      STI.inMips16Mode()
          ? &Mips::CPU16RegsRegClass
          : STI.inMicroMipsMode()
                ? STI.hasMips64()
                      ? &Mips::GPRMM16_64RegClass
                      : &Mips::GPRMM16RegClass
                : static_cast<const MipsTargetMachine &>(MF.getTarget())
                          .getABI()
                          .IsN64()
                      ? &Mips::GPR64RegClass
                      : &Mips::GPR32RegClass;
  return GlobalBaseReg = MF.getRegInfo().createVirtualRegister(RC);
}

void MipsFunctionInfo::createEhDataRegsFI() {
  for (int I = 0; I < 4; ++I) {
    const TargetRegisterClass *RC =
        static_cast<const MipsTargetMachine &>(MF.getTarget()).getABI().IsN64()
            ? &Mips::GPR64RegClass
            : &Mips::GPR32RegClass;

    EhDataRegFI[I] = MF.getFrameInfo()->CreateStackObject(RC->getSize(),
        RC->getAlignment(), false);
  }
}

void MipsFunctionInfo::createISRRegFI() {
  // ISRs require spill slots for Status & ErrorPC Coprocessor 0 registers.
  // The current implementation only supports Mips32r2+ not Mips64rX. Status
  // is always 32 bits, ErrorPC is 32 or 64 bits dependant on architecture,
  // however Mips32r2+ is the supported architecture.
  const TargetRegisterClass *RC = &Mips::GPR32RegClass;

  for (int I = 0; I < 2; ++I)
    ISRDataRegFI[I] = MF.getFrameInfo()->CreateStackObject(
        RC->getSize(), RC->getAlignment(), false);
}

bool MipsFunctionInfo::isEhDataRegFI(int FI) const {
  return CallsEhReturn && (FI == EhDataRegFI[0] || FI == EhDataRegFI[1]
                        || FI == EhDataRegFI[2] || FI == EhDataRegFI[3]);
}

bool MipsFunctionInfo::isISRRegFI(int FI) const {
  return IsISR && (FI == ISRDataRegFI[0] || FI == ISRDataRegFI[1]);
}
MachinePointerInfo MipsFunctionInfo::callPtrInfo(const char *ES) {
  return MachinePointerInfo(MF.getPSVManager().getExternalSymbolCallEntry(ES));
}

MachinePointerInfo MipsFunctionInfo::callPtrInfo(const GlobalValue *GV) {
  return MachinePointerInfo(MF.getPSVManager().getGlobalValueCallEntry(GV));
}

int MipsFunctionInfo::getMoveF64ViaSpillFI(const TargetRegisterClass *RC) {
  if (MoveF64ViaSpillFI == -1) {
    MoveF64ViaSpillFI = MF.getFrameInfo()->CreateStackObject(
        RC->getSize(), RC->getAlignment(), false);
  }
  return MoveF64ViaSpillFI;
}

void MipsFunctionInfo::anchor() { }
