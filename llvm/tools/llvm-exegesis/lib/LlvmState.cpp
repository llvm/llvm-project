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
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

namespace llvm {
namespace exegesis {

LLVMState::LLVMState(const std::string &Triple, const std::string &CpuName,
                     const std::string &Features) {
  std::string Error;
  const Target *const TheTarget = TargetRegistry::lookupTarget(Triple, Error);
  assert(TheTarget && "unknown target for host");
  const TargetOptions Options;
  TheTargetMachine.reset(
      static_cast<LLVMTargetMachine *>(TheTarget->createTargetMachine(
          Triple, CpuName, Features, Options, Reloc::Model::Static)));
  assert(TheTargetMachine && "unable to create target machine");
  TheExegesisTarget = ExegesisTarget::lookup(TheTargetMachine->getTargetTriple());
  if (!TheExegesisTarget) {
    errs() << "no exegesis target for " << Triple << ", using default\n";
    TheExegesisTarget = &ExegesisTarget::getDefault();
  }
  PfmCounters = &TheExegesisTarget->getPfmCounters(CpuName);

  BitVector ReservedRegs = getFunctionReservedRegs(getTargetMachine());
  for (const unsigned Reg : TheExegesisTarget->getUnavailableRegisters())
    ReservedRegs.set(Reg);
  RATC.reset(
      new RegisterAliasingTrackerCache(getRegInfo(), std::move(ReservedRegs)));
  IC.reset(new InstructionsCache(getInstrInfo(), getRATC()));
}

LLVMState::LLVMState(const std::string &CpuName)
    : LLVMState(sys::getProcessTriple(),
                CpuName.empty() ? sys::getHostCPUName().str() : CpuName, "") {}

std::unique_ptr<LLVMTargetMachine> LLVMState::createTargetMachine() const {
  return std::unique_ptr<LLVMTargetMachine>(static_cast<LLVMTargetMachine *>(
      TheTargetMachine->getTarget().createTargetMachine(
          TheTargetMachine->getTargetTriple().normalize(),
          TheTargetMachine->getTargetCPU(),
          TheTargetMachine->getTargetFeatureString(), TheTargetMachine->Options,
          Reloc::Model::Static)));
}

bool LLVMState::canAssemble(const MCInst &Inst) const {
  MCObjectFileInfo ObjectFileInfo;
  MCContext Context(TheTargetMachine->getMCAsmInfo(),
                    TheTargetMachine->getMCRegisterInfo(), &ObjectFileInfo);
  std::unique_ptr<const MCCodeEmitter> CodeEmitter(
      TheTargetMachine->getTarget().createMCCodeEmitter(
          *TheTargetMachine->getMCInstrInfo(), *TheTargetMachine->getMCRegisterInfo(),
          Context));
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
