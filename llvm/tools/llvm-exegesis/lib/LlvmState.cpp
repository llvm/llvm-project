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
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

namespace llvm {
namespace exegesis {

LLVMState::LLVMState(const std::string &Triple, const std::string &CpuName,
                     const std::string &Features) {
  std::string Error;
  const llvm::Target *const TheTarget =
      llvm::TargetRegistry::lookupTarget(Triple, Error);
  assert(TheTarget && "unknown target for host");
  const llvm::TargetOptions Options;
  TargetMachine.reset(
      static_cast<llvm::LLVMTargetMachine *>(TheTarget->createTargetMachine(
          Triple, CpuName, Features, Options, llvm::Reloc::Model::Static)));
  TheExegesisTarget = ExegesisTarget::lookup(TargetMachine->getTargetTriple());
  if (!TheExegesisTarget) {
    llvm::errs() << "no exegesis target for " << Triple << ", using default\n";
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
    : LLVMState(llvm::sys::getProcessTriple(),
                CpuName.empty() ? llvm::sys::getHostCPUName().str() : CpuName,
                "") {}

std::unique_ptr<llvm::LLVMTargetMachine>
LLVMState::createTargetMachine() const {
  return std::unique_ptr<llvm::LLVMTargetMachine>(
      static_cast<llvm::LLVMTargetMachine *>(
          TargetMachine->getTarget().createTargetMachine(
              TargetMachine->getTargetTriple().normalize(),
              TargetMachine->getTargetCPU(),
              TargetMachine->getTargetFeatureString(), TargetMachine->Options,
              llvm::Reloc::Model::Static)));
}

bool LLVMState::canAssemble(const llvm::MCInst &Inst) const {
  llvm::MCObjectFileInfo ObjectFileInfo;
  llvm::MCContext Context(TargetMachine->getMCAsmInfo(),
                          TargetMachine->getMCRegisterInfo(), &ObjectFileInfo);
  std::unique_ptr<const llvm::MCCodeEmitter> CodeEmitter(
      TargetMachine->getTarget().createMCCodeEmitter(
          *TargetMachine->getMCInstrInfo(), *TargetMachine->getMCRegisterInfo(),
          Context));
  llvm::SmallVector<char, 16> Tmp;
  llvm::raw_svector_ostream OS(Tmp);
  llvm::SmallVector<llvm::MCFixup, 4> Fixups;
  CodeEmitter->encodeInstruction(Inst, OS, Fixups,
                                 *TargetMachine->getMCSubtargetInfo());
  return Tmp.size() > 0;
}

} // namespace exegesis
} // namespace llvm
