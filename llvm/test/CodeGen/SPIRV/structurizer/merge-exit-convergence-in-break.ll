; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s --match-full-lines

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan-compute"

define internal spir_func void @main() #0 {

; CHECK:                      OpDecorate %[[#builtin:]] BuiltIn SubgroupLocalInvocationId
; CHECK-DAG:  %[[#int_ty:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#pint_ty:]] = OpTypePointer Function %[[#int_ty]]
; CHECK-DAG: %[[#bool_ty:]] = OpTypeBool
; CHECK-DAG:   %[[#int_0:]] = OpConstant %[[#int_ty]] 0
; CHECK-DAG:   %[[#int_1:]] = OpConstant %[[#int_ty]] 1
; CHECK-DAG:  %[[#int_10:]] = OpConstant %[[#int_ty]] 10

; CHECK:   %[[#entry:]] = OpLabel
; CHECK:     %[[#idx:]] = OpVariable %[[#pint_ty]] Function
; CHECK:                  OpStore %[[#idx]] %[[#int_0]] Aligned 4
; CHECK:                  OpBranch %[[#while_cond:]]
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %idx = alloca i32, align 4
  store i32 0, ptr %idx, align 4
  br label %while.cond

; CHECK:   %[[#while_cond]] = OpLabel
; CHECK:         %[[#tmp:]] = OpLoad %[[#int_ty]] %[[#idx]] Aligned 4
; CHECK:         %[[#cmp:]] = OpINotEqual %[[#bool_ty]] %[[#tmp]] %[[#int_10]]
; CHECK:                      OpBranchConditional %[[#cmp]] %[[#while_body:]] %[[#new_end:]]
while.cond:
  %1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  %2 = load i32, ptr %idx, align 4
  %cmp = icmp ne i32 %2, 10
  br i1 %cmp, label %while.body, label %while.end

; CHECK:   %[[#while_body]] = OpLabel
; CHECK-NEXT:    %[[#tmp:]] = OpLoad %[[#int_ty]] %[[#builtin]] Aligned 1
; CHECK-NEXT:                 OpStore %[[#idx]] %[[#tmp]] Aligned 4
; CHECK-NEXT:    %[[#tmp:]] = OpLoad %[[#int_ty]] %[[#idx]] Aligned 4
; CHECK-NEXT:   %[[#cmp1:]] = OpIEqual %[[#bool_ty]] %[[#tmp]] %[[#int_0]]
; CHECK:                      OpBranchConditional %[[#cmp1]] %[[#if_then:]] %[[#if_end:]]
while.body:
  %3 = call i32 @__hlsl_wave_get_lane_index() [ "convergencectrl"(token %1) ]
  store i32 %3, ptr %idx, align 4
  %4 = load i32, ptr %idx, align 4
  %cmp1 = icmp eq i32 %4, 0
  br i1 %cmp1, label %if.then, label %if.end

; CHECK:        %[[#if_then:]] = OpLabel
; CHECK-NEXT:                    OpBranch %[[#tail:]]
if.then:
  br label %tail

; CHECK:        %[[#tail:]] = OpLabel
; CHECK-NEXT:       %[[#tmp:]] = OpLoad %[[#int_ty]] %[[#builtin]] Aligned 1
; CHECK-NEXT:                    OpStore %[[#idx]] %[[#tmp]] Aligned 4
; CHECK:                         OpBranch %[[#new_end:]]
tail:
  %5 = call i32 @__hlsl_wave_get_lane_index() [ "convergencectrl"(token %1) ]
  store i32 %5, ptr %idx, align 4
  br label %while.end

; CHECK:   %[[#if_end]] = OpLabel
; CHECK:                  OpBranch %[[#while_cond]]
if.end:
  br label %while.cond

; CHECK:   %[[#while_end_loopexit:]] = OpLabel
; CHECK:                               OpBranch %[[#while_end:]]

; CHECK:   %[[#while_end]] = OpLabel
; CHECK:                     OpReturn
while.end:
  ret void

; CHECK:   %[[#new_end]] = OpLabel
; CHECK:    %[[#route:]] = OpPhi %[[#int_ty]] %[[#int_0]] %[[#while_cond]] %[[#int_1]] %[[#tail]]
; CHECK:                   OpSwitch %[[#route]] %[[#while_end_loopexit]] 1 %[[#while_end]]
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

