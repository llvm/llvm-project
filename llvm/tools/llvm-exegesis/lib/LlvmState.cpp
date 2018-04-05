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

LLVMState::LLVMState()
    : TheTriple(llvm::sys::getProcessTriple()),
      CpuName(llvm::sys::getHostCPUName().str()) {
  std::string Error;
  TheTarget = llvm::TargetRegistry::lookupTarget(TheTriple, Error);
  assert(TheTarget && "unknown target for host");
  SubtargetInfo.reset(
      TheTarget->createMCSubtargetInfo(TheTriple, CpuName, Features));
  InstrInfo.reset(TheTarget->createMCInstrInfo());
  RegInfo.reset(TheTarget->createMCRegInfo(TheTriple));
  AsmInfo.reset(TheTarget->createMCAsmInfo(*RegInfo, TheTriple));
}

std::unique_ptr<llvm::LLVMTargetMachine>
LLVMState::createTargetMachine() const {
  const llvm::TargetOptions Options;
  return std::unique_ptr<llvm::LLVMTargetMachine>(
      static_cast<llvm::LLVMTargetMachine *>(TheTarget->createTargetMachine(
          TheTriple, CpuName, Features, Options, llvm::Reloc::Model::Static)));
}

bool LLVMState::canAssemble(const llvm::MCInst &Inst) const {
  llvm::MCObjectFileInfo ObjectFileInfo;
  llvm::MCContext Context(AsmInfo.get(), RegInfo.get(), &ObjectFileInfo);
  std::unique_ptr<const llvm::MCCodeEmitter> CodeEmitter(
      TheTarget->createMCCodeEmitter(*InstrInfo, *RegInfo, Context));
  llvm::SmallVector<char, 16> Tmp;
  llvm::raw_svector_ostream OS(Tmp);
  llvm::SmallVector<llvm::MCFixup, 4> Fixups;
  CodeEmitter->encodeInstruction(Inst, OS, Fixups, *SubtargetInfo);
  return Tmp.size() > 0;
}

} // namespace exegesis
