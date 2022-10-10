//===-- LlvmState.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LlvmState.h"
#include "Target.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

namespace llvm {
namespace exegesis {

Expected<LLVMState> LLVMState::Create(std::string Triple, std::string CpuName,
                                      const StringRef Features) {
  if (Triple.empty())
    Triple = sys::getProcessTriple();
  if (CpuName.empty())
    CpuName = sys::getHostCPUName().str();
  std::string Error;
  const Target *const TheTarget = TargetRegistry::lookupTarget(Triple, Error);
  if (!TheTarget) {
    return llvm::make_error<llvm::StringError>(
        "no LLVM target for triple " + Triple, llvm::inconvertibleErrorCode());
  }
  const TargetOptions Options;
  std::unique_ptr<const TargetMachine> TM(
      static_cast<LLVMTargetMachine *>(TheTarget->createTargetMachine(
          Triple, CpuName, Features, Options, Reloc::Model::Static)));
  if (!TM) {
    return llvm::make_error<llvm::StringError>(
        "unable to create target machine", llvm::inconvertibleErrorCode());
  }

  const ExegesisTarget *ET =
      Triple.empty() ? &ExegesisTarget::getDefault()
                     : ExegesisTarget::lookup(TM->getTargetTriple());
  if (!ET) {
    return llvm::make_error<llvm::StringError>(
        "no Exegesis target for triple " + Triple,
        llvm::inconvertibleErrorCode());
  }
  return LLVMState(std::move(TM), ET, CpuName);
}

LLVMState::LLVMState(std::unique_ptr<const TargetMachine> TM,
                     const ExegesisTarget *ET, const StringRef CpuName)
    : TheExegesisTarget(ET), TheTargetMachine(std::move(TM)) {
  PfmCounters = &TheExegesisTarget->getPfmCounters(CpuName);

  BitVector ReservedRegs = getFunctionReservedRegs(getTargetMachine());
  for (const unsigned Reg : TheExegesisTarget->getUnavailableRegisters())
    ReservedRegs.set(Reg);
  RATC.reset(
      new RegisterAliasingTrackerCache(getRegInfo(), std::move(ReservedRegs)));
  IC.reset(new InstructionsCache(getInstrInfo(), getRATC()));
}

std::unique_ptr<LLVMTargetMachine> LLVMState::createTargetMachine() const {
  return std::unique_ptr<LLVMTargetMachine>(static_cast<LLVMTargetMachine *>(
      TheTargetMachine->getTarget().createTargetMachine(
          TheTargetMachine->getTargetTriple().normalize(),
          TheTargetMachine->getTargetCPU(),
          TheTargetMachine->getTargetFeatureString(), TheTargetMachine->Options,
          Reloc::Model::Static)));
}

bool LLVMState::canAssemble(const MCInst &Inst) const {
  MCContext Context(TheTargetMachine->getTargetTriple(),
                    TheTargetMachine->getMCAsmInfo(),
                    TheTargetMachine->getMCRegisterInfo(),
                    TheTargetMachine->getMCSubtargetInfo());
  std::unique_ptr<const MCCodeEmitter> CodeEmitter(
      TheTargetMachine->getTarget().createMCCodeEmitter(
          *TheTargetMachine->getMCInstrInfo(), Context));
  assert(CodeEmitter && "unable to create code emitter");
  SmallVector<char, 16> Tmp;
  raw_svector_ostream OS(Tmp);
  SmallVector<MCFixup, 4> Fixups;
  CodeEmitter->encodeInstruction(Inst, OS, Fixups,
                                 *TheTargetMachine->getMCSubtargetInfo());
  return Tmp.size() > 0;
}

} // namespace exegesis
} // namespace llvm
