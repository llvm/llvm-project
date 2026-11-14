; RUN: llc -O0 -mtriple=spirv-unknown-vulkan-compute -spirv-ext=+SPV_EXT_long_vector %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute -spirv-ext=+SPV_EXT_long_vector %s -o - -filetype=obj | spirv-val %}

; The <1 x float> result of @getv makes the module emit the constant 1 as the
; component count of OpTypeVectorIdEXT, which is why the extension is enabled
; here. An ordinary OpTypeVector takes a literal count and emits no constant.
; Case value 1 then reaches the pre-legalizer as that SPIR-V constant, and case
; value 2 as a G_CONSTANT, so both decode paths run.

target triple = "spirv-unknown-vulkan-compute"

; CHECK: %[[#one:]] = OpConstant %[[#]] 1
; CHECK: OpTypeVectorIdEXT %[[#]] %[[#one]]
; CHECK: OpSwitch %[[#]] %[[#]] 1 %[[#]] 2 %[[#]]

define void @main() #0 {
entry:
  %t0 = tail call token @llvm.experimental.convergence.entry()
  %v = call <1 x float> @getv() [ "convergencectrl"(token %t0) ]
  %i = call i32 @get() [ "convergencectrl"(token %t0) ]
  switch i32 %i, label %exit [
    i32 1, label %c1
    i32 2, label %c2
  ]

c1:
  call void @put(i32 1) [ "convergencectrl"(token %t0) ]
  br label %exit

c2:
  call void @put(i32 2) [ "convergencectrl"(token %t0) ]
  br label %exit

exit:
  ret void
}

declare token @llvm.experimental.convergence.entry() #1
declare <1 x float> @getv() #2
declare i32 @get() #2
declare void @put(i32) #2

attributes #0 = { convergent noinline norecurse nounwind "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
attributes #1 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { convergent }
