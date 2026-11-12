; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}
; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s

; The goal of this test is to voluntarily create 2 overlapping convergence
; structures: the loop, and the inner condition.
; Here, the condition header also branches to 2 internal nodes, which are not
; directly a merge/exits.
; This will require a proper header-split.
; In addition, splitting the header makes the continue the merge of the inner
; condition, so we need to properly split the continue block to create a
; valid inner merge, in the correct order.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

; CHECK-DAG:    OpName %[[#switch_0:]] "reg1"
; CHECK-DAG:    OpName %[[#variable:]] "var"

; CHECK-DAG:    %[[#int_0:]] = OpConstant %[[#]] 0
; CHECK-DAG:    %[[#int_1:]] = OpConstant %[[#]] 1
; CHECK-DAG:    %[[#int_2:]] = OpConstant %[[#]] 2
; CHECK-DAG:    %[[#int_3:]] = OpConstant %[[#]] 3
; CHECK-DAG:    %[[#int_4:]] = OpConstant %[[#]] 4

define internal spir_func void @main() #1 {
; CHECK:      %[[#entry:]] = OpLabel
; CHECK:    %[[#switch_0]] = OpVariable %[[#]] Function
; CHECK:    %[[#variable]] = OpVariable %[[#]] Function
; CHECK:                     OpBranch %[[#header:]]
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %var = alloca i32, align 4
  br label %header

; CHECK: %[[#header]] = OpLabel
; CHECK:                OpLoopMerge %[[#merge:]] %[[#continue:]] None
; CHECK:                OpBranch %[[#split_header:]]

; CHECK: %[[#split_header]] = OpLabel
; CHECK:                      OpSelectionMerge %[[#inner_merge:]] None
; CHECK:                      OpBranchConditional %[[#]] %[[#left:]] %[[#right:]]
header:
  %2 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  br i1 true, label %left, label %right

; CHECK:     %[[#right]] = OpLabel
; CHECK-DAG:               OpStore %[[#switch_0]] %[[#int_0]]
; CHECK-DAG:               OpStore %[[#variable]] %[[#int_2]]
; CHECK:                   OpBranchConditional %[[#]] %[[#inner_merge]] %[[#right_next:]]
right:
  store i32 2, ptr %var
  br i1 true, label %merge, label %right_next

; CHECK:     %[[#right_next]] = OpLabel
; CHECK-DAG:                    OpStore %[[#switch_0]] %[[#int_1]]
; CHECK-DAG:                    OpStore %[[#variable]] %[[#int_4]]
; CHECK:                        OpBranch %[[#inner_merge]]
right_next:
  store i32 4, ptr %var
  br label %continue

; CHECK:     %[[#left]] = OpLabel
; CHECK-DAG:              OpStore %[[#switch_0]] %[[#int_0]]
; CHECK-DAG:              OpStore %[[#variable]] %[[#int_1]]
; CHECK:                  OpBranchConditional %[[#]] %[[#inner_merge]] %[[#left_next:]]
left:
  store i32 1, ptr %var
  br i1 true, label %merge, label %left_next

; CHECK:     %[[#left_next]] = OpLabel
; CHECK-DAG:                   OpStore %[[#switch_0]] %[[#int_1]]
; CHECK-DAG:                   OpStore %[[#variable]] %[[#int_3]]
; CHECK:                       OpBranch %[[#inner_merge]]
left_next:
  store i32 3, ptr %var
  br label %continue

; CHECK: %[[#inner_merge]] = OpLabel
; CHECK:        %[[#tmp:]] = OpLoad %[[#]] %[[#switch_0]]
; CHECK:       %[[#cond:]] = OpIEqual %[[#]] %[[#int_0]] %[[#tmp]]
; CHECK:                     OpBranchConditional %[[#cond]] %[[#merge]] %[[#continue]]

; CHECK: %[[#continue]] = OpLabel
; CHECK:                  OpBranch %[[#header]]
continue:
  br label %header

; CHECK: %[[#merge]] = OpLabel
; CHECK:               OpReturn
merge:
  ret void
}


declare token @llvm.experimental.convergence.entry() #0
declare token @llvm.experimental.convergence.loop() #0

attributes #0 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #1 = { convergent noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"dx.disable_optimizations", i32 1}
