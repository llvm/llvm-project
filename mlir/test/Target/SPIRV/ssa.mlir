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
    %v3float = OpTypeVector %float 3
    %ptr_v3f = OpTypePointer Function %v3float
         %fn = OpTypeFunction %void
    %float_0 = OpConstant %float 0
    %float_1 = OpConstant %float 1
       %bool = OpTypeBool
      %true = OpConstantTrue %bool
    %v3_zero = OpConstantComposite %v3float %float_0 %float_0 %float_0
       %main = OpFunction %void None %fn
      %entry = OpLabel
         %var = OpVariable %ptr_v3f Function
               OpSelectionMerge %merge None
               OpBranchConditional %true %then %merge
       %then = OpLabel
               OpStore %var %v3_zero
               OpBranch %merge
      %merge = OpLabel
         %load = OpLoad %v3float %var
               OpReturn
               OpFunctionEnd