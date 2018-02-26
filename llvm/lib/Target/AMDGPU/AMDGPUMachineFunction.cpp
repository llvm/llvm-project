//===-- AMDGPUMachineFunctionInfo.cpp ---------------------------------------=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUMachineFunction.h"
#include "AMDGPUSubtarget.h"

using namespace llvm;

AMDGPUMachineFunction::AMDGPUMachineFunction(const MachineFunction &MF) :
  MachineFunctionInfo(),
  LocalMemoryObjects(),
  KernArgSize(0),
  MaxKernArgAlign(0),
  LDSSize(0),
  ABIArgOffset(0),
  IsEntryFunction(AMDGPU::isEntryFunctionCC(MF.getFunction().getCallingConv())),
  NoSignedZerosFPMath(MF.getTarget().Options.NoSignedZerosFPMath) {
  // FIXME: Should initialize KernArgSize based on ExplicitKernelArgOffset,
  // except reserved size is not correctly aligned.
}

unsigned AMDGPUMachineFunction::allocateLDSGlobal(const DataLayout &DL,
                                                  const GlobalValue &GV) {
  auto Entry = LocalMemoryObjects.insert(std::make_pair(&GV, 0));
  if (!Entry.second)
    return Entry.first->second;

  unsigned Align = GV.getAlignment();
  if (Align == 0)
    Align = DL.getABITypeAlignment(GV.getValueType());

  /// TODO: We should sort these to minimize wasted space due to alignment
  /// padding. Currently the padding is decided by the first encountered use
  /// during lowering.
  unsigned Offset = LDSSize = alignTo(LDSSize, Align);

  Entry.first->second = Offset;
  LDSSize += DL.getTypeAllocSize(GV.getValueType());

  return Offset;
}
