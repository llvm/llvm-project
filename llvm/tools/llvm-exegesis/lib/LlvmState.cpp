//===-- LlvmState.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "LlvmState.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

namespace exegesis {

LLVMState::LLVMState(const std::string &Triple, const std::string &CpuName) {
  std::string Error;
  const llvm::Target *const TheTarget =
      llvm::TargetRegistry::lookupTarget(Triple, Error);
  assert(TheTarget && "unknown target for host");
  const llvm::TargetOptions Options;
  TargetMachine.reset(static_cast<llvm::LLVMTargetMachine *>(
      TheTarget->createTargetMachine(Triple, CpuName, /*Features*/ "", Options,
                                     llvm::Reloc::Model::Static)));
  TheExegesisTarget = ExegesisTarget::lookup(TargetMachine->getTargetTriple());
}

LLVMState::LLVMState()
    : LLVMState(llvm::sys::getProcessTriple(),
                llvm::sys::getHostCPUName().str()) {}

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
