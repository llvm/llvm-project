#include "AMDGPUMachineFunction.h"

using namespace llvm;

// Pin the vtable to this file.
void AMDGPUMachineFunction::anchor() {}

AMDGPUMachineFunction::AMDGPUMachineFunction(const MachineFunction &MF) :
  MachineFunctionInfo(),
  KernArgSize(0),
  MaxKernArgAlign(0),
  LDSSize(0),
  ABIArgOffset(0),
  ScratchSize(0),
  IsKernel(MF.getFunction()->getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL ||
           MF.getFunction()->getCallingConv() == llvm::CallingConv::SPIR_KERNEL)
{
}

bool AMDGPUMachineFunction::isKernel() const
{
  return IsKernel;
}
