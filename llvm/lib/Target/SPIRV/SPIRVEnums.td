//===-- SPIRVEnums.td - Describe SPIRV Enum Operands -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// All SPIRV enums defined in SPIRVBaseInfo.h should have a corresponding enum
// operand here. This enables the correct PrintMethod to be defined so
// its name or mask bits can be automatically printed in SPIRVInstPrinter
// when referred to in SPIRVInstrInfo.td.
//
//===----------------------------------------------------------------------===//

class EnumOperand<string Name> : Operand<i32>{
  let PrintMethod = "print"#Name;
}

def ExtInst : EnumOperand<"ExtInst">;

def Capability : EnumOperand<"Capability">;
def SourceLanguage : EnumOperand<"SourceLanguage">;
def ExecutionModel : EnumOperand<"ExecutionModel">;
def AddressingModel : EnumOperand<"AddressingModel">;
def MemoryModel : EnumOperand<"MemoryModel">;
def ExecutionMode : EnumOperand<"ExecutionMode">;
def StorageClass : EnumOperand<"StorageClass">;
def Dim : EnumOperand<"Dim">;
def SamplerAddressingMode : EnumOperand<"SamplerAddressingMode">;
def SamplerFilterMode : EnumOperand<"SamplerFilterMode">;
def ImageFormat : EnumOperand<"ImageFormat">;
def ImageChannelOrder : EnumOperand<"ImageChannelOrder">;
def ImageChannelDataType : EnumOperand<"ImageChannelDataType">;
def ImageOperand : EnumOperand<"ImageOperand">;
def FPFastMathMode : EnumOperand<"FPFastMathMode">;
def FProundingMode : EnumOperand<"FPRoundingMode">;
def LinkageType : EnumOperand<"LinkageType">;
def AccessQualifier : EnumOperand<"AccessQualifier">;
def FunctionParameterAttribute : EnumOperand<"FunctionParameterAttribute">;
def Decoration : EnumOperand<"Decoration">;
def Builtin : EnumOperand<"Builtin">;
def SelectionControl: EnumOperand<"SelectionControl">;
def LoopControl: EnumOperand<"LoopControl">;
def FunctionControl : EnumOperand<"FunctionControl">;
def MemorySemantics : EnumOperand<"MemorySemantics">;
def MemoryOperand : EnumOperand<"MemoryOperand">;
def Scope : EnumOperand<"Scope">;
def GroupOperation : EnumOperand<"GroupOperation">;
def KernelEnqueueFlags : EnumOperand<"KernelEnqueueFlags">;
def KernelProfilingInfo : EnumOperand<"KernelProfilingInfo">;
