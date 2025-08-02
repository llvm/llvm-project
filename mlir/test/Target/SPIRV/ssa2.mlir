# RUN: split-file %s %t
# RUN: spirv-as --target-env spv1.0 %t/spv.spvasm -o %t.spv
# RUN: mlir-translate --deserialize-spirv %t.spv -o - | FileCheck %s
// CHECK: module
// CHECK: spirv.func @main
// CHECK: spirv.Variable
// CHECK: spirv.Return
//--- spv.spvasm
; SPIR-V
; Version: 1.0
; Generator: Khronos SPIR-V Tools Assembler; 0
; Bound: 20
; Schema: 0
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginUpperLeft
       %void = OpTypeVoid
      %float = OpTypeFloat 32
    %ptr_f = OpTypePointer Function %float
         %fn = OpTypeFunction %void
    %float_1 = OpConstant %float 1.0
       %bool = OpTypeBool
      %true = OpConstantTrue %bool
       %main = OpFunction %void None %fn
      %entry = OpLabel
         %var = OpVariable %ptr_f Function
               OpSelectionMerge %merge None
               OpBranchConditional %true %then %merge
       %then = OpLabel
               OpStore %var %float_1
               OpBranch %merge
      %merge = OpLabel
         %load = OpLoad %float %var
               OpReturn
               OpFunctionEnd