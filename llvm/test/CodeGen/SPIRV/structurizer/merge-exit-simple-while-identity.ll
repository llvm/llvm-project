; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}
; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan-compute"

define internal spir_func void @main() #0 {

; CHECK:   %[[#entry:]] = OpLabel
; CHECK:                  OpBranch %[[#while_cond:]]
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %idx = alloca i32, align 4
  store i32 -1, ptr %idx, align 4
  br label %while.cond

; CHECK:   %[[#while_cond]] = OpLabel
; CHECK:                      OpBranchConditional %[[#cond:]] %[[#while_body:]] %[[#while_end:]]
while.cond:
  %1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  %2 = load i32, ptr %idx, align 4
  %cmp = icmp ne i32 %2, 0
  br i1 %cmp, label %while.body, label %while.end

; CHECK:   %[[#while_body]] = OpLabel
; CHECK:                      OpBranch %[[#while_cond]]
while.body:
  %3 = call i32 @__hlsl_wave_get_lane_index() [ "convergencectrl"(token %1) ]
  store i32 %3, ptr %idx, align 4
  br label %while.cond

; CHECK:        %[[#while_end]] = OpLabel
; CHECK-NEXT:                     OpReturn
while.end:
  ret void
}

declare token @llvm.experimental.convergence.entry() #2
declare token @llvm.experimental.convergence.loop() #2
declare i32 @__hlsl_wave_get_lane_index() #3

attributes #0 = { convergent noinline norecurse nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent norecurse "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #3 = { convergent }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"dx.disable_optimizations", i32 1}
