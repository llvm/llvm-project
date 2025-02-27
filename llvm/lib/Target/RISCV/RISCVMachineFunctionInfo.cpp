//=- RISCVMachineFunctionInfo.cpp - RISC-V machine function info --*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares RISCV-specific per-machine-function information.
//
//===----------------------------------------------------------------------===//

#include "RISCVMachineFunctionInfo.h"
#include "llvm/IR/Module.h"

using namespace llvm;

yaml::RISCVMachineFunctionInfo::RISCVMachineFunctionInfo(
    const llvm::RISCVMachineFunctionInfo &MFI)
    : VarArgsFrameIndex(MFI.getVarArgsFrameIndex()),
      VarArgsSaveSize(MFI.getVarArgsSaveSize()) {}

MachineFunctionInfo *RISCVMachineFunctionInfo::clone(
    BumpPtrAllocator &Allocator, MachineFunction &DestMF,
    const DenseMap<MachineBasicBlock *, MachineBasicBlock *> &Src2DstMBB)
    const {
  return DestMF.cloneInfo<RISCVMachineFunctionInfo>(*this);
}

RISCVMachineFunctionInfo::RISCVMachineFunctionInfo(const Function &F,
                                                   const RISCVSubtarget *STI) {

  // The default stack probe size is 4096 if the function has no
  // stack-probe-size attribute. This is a safe default because it is the
  // smallest possible guard page size.
  uint64_t ProbeSize = 4096;
  if (F.hasFnAttribute("stack-probe-size"))
    ProbeSize = F.getFnAttributeAsParsedInteger("stack-probe-size");
  else if (const auto *PS = mdconst::extract_or_null<ConstantInt>(
               F.getParent()->getModuleFlag("stack-probe-size")))
    ProbeSize = PS->getZExtValue();
  assert(int64_t(ProbeSize) > 0 && "Invalid stack probe size");

  // Round down to the stack alignment.
  uint64_t StackAlign =
      STI->getFrameLowering()->getTransientStackAlign().value();
  ProbeSize = std::max(StackAlign, alignDown(ProbeSize, StackAlign));
  StringRef ProbeKind;
  if (F.hasFnAttribute("probe-stack"))
    ProbeKind = F.getFnAttribute("probe-stack").getValueAsString();
  else if (const auto *PS = dyn_cast_or_null<MDString>(
               F.getParent()->getModuleFlag("probe-stack")))
    ProbeKind = PS->getString();
  if (ProbeKind.size()) {
    StackProbeSize = ProbeSize;
  }
}

void yaml::RISCVMachineFunctionInfo::mappingImpl(yaml::IO &YamlIO) {
  MappingTraits<RISCVMachineFunctionInfo>::mapping(YamlIO, *this);
}

void RISCVMachineFunctionInfo::initializeBaseYamlFields(
    const yaml::RISCVMachineFunctionInfo &YamlMFI) {
  VarArgsFrameIndex = YamlMFI.VarArgsFrameIndex;
  VarArgsSaveSize = YamlMFI.VarArgsSaveSize;
}

void RISCVMachineFunctionInfo::addSExt32Register(Register Reg) {
  SExt32Registers.push_back(Reg);
}

bool RISCVMachineFunctionInfo::isSExt32Register(Register Reg) const {
  return is_contained(SExt32Registers, Reg);
}
