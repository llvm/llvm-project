; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s --match-full-lines
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan-compute"

define internal spir_func void @main() #0 {

; CHECK-DAG:                  OpName %[[#idx:]] "idx"
; CHECK-DAG:                  OpName %[[#reg_0:]] "reg"
; CHECK-DAG:                  OpDecorate %[[#builtin:]] BuiltIn SubgroupLocalInvocationId
; CHECK-DAG:  %[[#int_ty:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#bool_ty:]] = OpTypeBool
; CHECK-DAG:   %[[#int_0:]] = OpConstant %[[#int_ty]] 0
; CHECK-DAG:   %[[#int_1:]] = OpConstant %[[#int_ty]] 1
; CHECK-DAG:   %[[#int_2:]] = OpConstant %[[#int_ty]] 2
; CHECK-DAG:  %[[#int_10:]] = OpConstant %[[#int_ty]] 10

; CHECK:   %[[#entry:]] = OpLabel
; CHECK:      %[[#idx]] = OpVariable %[[#]] Function
; CHECK:                  OpStore %[[#idx]] %[[#int_0]] Aligned 4
; CHECK:                  OpBranch %[[#while_cond:]]
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %idx = alloca i32, align 4
  store i32 0, ptr %idx, align 4
  br label %while.cond

; CHECK:   %[[#while_cond]] = OpLabel
; CHECK:                      OpStore %[[#reg_0]] %[[#]] Aligned 4
; CHECK:         %[[#tmp:]] = OpLoad %[[#int_ty]] %[[#idx]] Aligned 4
; CHECK:         %[[#cmp:]] = OpINotEqual %[[#bool_ty]] %[[#tmp]] %[[#int_10]]
; CHECK:                      OpLoopMerge %[[#new_end:]] %[[#if_end2:]] None
; CHECK:                      OpBranchConditional %[[#cmp]] %[[#while_body:]] %[[#new_end]]
while.cond:
  %1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  %2 = load i32, ptr %idx, align 4
  %cmp = icmp ne i32 %2, 10
  br i1 %cmp, label %while.body, label %while.end

; CHECK:   %[[#while_body]] = OpLabel
; CHECK:                      OpStore %[[#reg_0]] %[[#]] Aligned 4
; CHECK:         %[[#tmp:]] = OpLoad %[[#]] %[[#builtin]] Aligned 1
; CHECK:                      OpStore %[[#idx]] %[[#tmp]] Aligned 4
; CHECK:         %[[#tmp:]] = OpLoad %[[#int_ty]] %[[#idx]] Aligned 4
; CHECK:        %[[#cmp1:]] = OpIEqual %[[#bool_ty]] %[[#tmp]] %[[#int_0]]
; CHECK:                      OpBranchConditional %[[#cmp1]] %[[#new_end]] %[[#if_end:]]
while.body:
  %3 = call i32 @__hlsl_wave_get_lane_index() [ "convergencectrl"(token %1) ]
  store i32 %3, ptr %idx, align 4
  %4 = load i32, ptr %idx, align 4
  %cmp1 = icmp eq i32 %4, 0
  br i1 %cmp1, label %if.then, label %if.end

; CHECK:               %[[#if_end]] = OpLabel
; CHECK:                              OpStore %[[#reg_0]] %[[#]] Aligned 4
; CHECK:                 %[[#tmp:]] = OpLoad %[[#int_ty]] %[[#builtin]] Aligned 1
; CHECK:                              OpStore %[[#idx]] %[[#tmp]] Aligned 4
; CHECK:                 %[[#tmp:]] = OpLoad %[[#int_ty]] %[[#idx]] Aligned 4
; CHECK:                %[[#cmp2:]] = OpIEqual %[[#bool_ty]] %[[#tmp]] %[[#int_0]]
; CHECK:                              OpBranchConditional %[[#cmp2]] %[[#new_end]] %[[#if_end2]]
if.end:
  %5 = call i32 @__hlsl_wave_get_lane_index() [ "convergencectrl"(token %1) ]
  store i32 %5, ptr %idx, align 4
  %6 = load i32, ptr %idx, align 4
  %cmp2 = icmp eq i32 %6, 0
  br i1 %cmp2, label %if.then2, label %if.end2

; CHECK:   %[[#if_end2]] = OpLabel
; CHECK:                   OpBranch %[[#while_cond]]

; TODO: this OpSwitch is useless. Improve the "remove useless branches" step of the structurizer to
;       cleanup those.
; CHECK:   %[[#new_end]] = OpLabel
; CHECK:    %[[#route:]] = OpLoad %[[#]] %[[#reg_0]] Aligned 4
; CHECK:                   OpSwitch %[[#route]] %[[#while_end:]] 1 %[[#while_end:]] 2 %[[#while_end:]]

if.end2:
  br label %while.cond

; CHECK: %[[#while_end]] = OpLabel
; CHECK:                   OpReturn
while.end:
  ret void

; Those blocks are removed by the structurizer.
if.then:
  br label %while.end

if.then2:
  br label %while.end
}

declare token @llvm.experimental.convergence.entry() #1
declare token @llvm.experimental.convergence.loop() #1
declare i32 @__hlsl_wave_get_lane_index() convergent

attributes #0 = { convergent noinline norecurse nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"dx.disable_optimizations", i32 1}
