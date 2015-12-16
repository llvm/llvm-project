#include "AMDGPUMachineFunction.h"
#include "AMDGPU.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
using namespace llvm;

// Pin the vtable to this file.
void AMDGPUMachineFunction::anchor() {}

AMDGPUMachineFunction::AMDGPUMachineFunction(const MachineFunction &MF) :
  MachineFunctionInfo(),
  ShaderType(ShaderType::COMPUTE),
  LDSSize(0),
  ABIArgOffset(0),
  ScratchSize(0),
  IsKernel(true) {

  ShaderType = AMDGPU::getShaderType(*MF.getFunction());
}
