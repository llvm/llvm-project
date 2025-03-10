; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan-compute"

; Function Attrs: convergent noinline nounwind optnone
define spir_func noundef i32 @fn() #4 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  ret i32 1
}

; Function Attrs: convergent noinline nounwind optnone
define spir_func noundef i32 @fn1() #4 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  ret i32 0
}

; Function Attrs: convergent noinline nounwind optnone
define spir_func noundef i32 @fn2() #4 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  ret i32 1
}


; CHECK-DAG:             OpName %[[#reg_0:]] "cond.reg2mem"
; CHECK-DAG:             OpName %[[#reg_1:]] "cond9.reg2mem"

define internal spir_func void @main() #0 {
; CHECK:                  OpSelectionMerge %[[#cond1_merge:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#cond1_true:]] %[[#cond1_false:]]
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  br i1 true, label %cond1_true, label %cond1_false

; CHECK:  %[[#cond1_false]] = OpLabel
; CHECK:                      OpStore %[[#reg_0]] %[[#]]
; CHECK:                      OpBranch %[[#cond1_merge]]
cond1_false:
  %2 = load i32, ptr %b, align 4
  br label %cond1_merge

; CHECK:  %[[#cond1_true]] = OpLabel
; CHECK:                     OpStore %[[#reg_0]] %[[#]]
; CHECK:                     OpBranch %[[#cond1_merge]]
cond1_true:
  %3 = load i32, ptr %a, align 4
  br label %cond1_merge

; CHECK: %[[#cond1_merge]] = OpLabel
; CHECK:        %[[#tmp:]] = OpLoad %[[#]] %[[#reg_0]]
; CHECK:       %[[#cond:]] = OpINotEqual %[[#]] %[[#tmp]] %[[#]]
; CHECK:                     OpSelectionMerge %[[#cond2_merge:]] None
; CHECK:                     OpBranchConditional %[[#cond]] %[[#cond2_true:]] %[[#cond2_merge]]
cond1_merge:
  %cond = phi i32 [ %3, %cond1_true ], [ %2, %cond1_false ]
  %tobool1 = icmp ne i32 %cond, 0
  br i1 %tobool1, label %cond2_true, label %cond2_merge

; CHECK:  %[[#cond2_true]] = OpLabel
; CHECK:                     OpBranch %[[#cond2_merge]]
cond2_true:
  store i32 0, ptr %a
  br label %cond2_merge

; CHECK:    %[[#cond2_merge]] = OpLabel
; CHECK:                        OpFunctionCall
; CHECK:                        OpSelectionMerge %[[#cond3_merge:]] None
; CHECK:                        OpBranchConditional %[[#]] %[[#cond3_true:]] %[[#cond3_false:]]
cond2_merge:
  %call2 = call spir_func noundef i32 @fn() #4 [ "convergencectrl"(token %0) ]
  br i1 true, label %cond3_true, label %cond3_false

; CHECK:  %[[#cond3_false]] = OpLabel
; CHECK:                      OpFunctionCall
; CHECK:                      OpStore %[[#reg_1]] %[[#]]
; CHECK:                      OpBranch %[[#cond3_merge]]
cond3_false:
  %call7 = call spir_func noundef i32 @fn2() #4 [ "convergencectrl"(token %0) ]
  br label %cond3_merge

; CHECK:  %[[#cond3_true]] = OpLabel
; CHECK:                     OpFunctionCall
; CHECK:                     OpStore %[[#reg_1]] %[[#]]
; CHECK:                     OpBranch %[[#cond3_merge]]
cond3_true:
  %call5 = call spir_func noundef i32 @fn1() #4 [ "convergencectrl"(token %0) ]
  br label %cond3_merge

; CHECK:  %[[#cond3_merge]] = OpLabel
; CHECK:         %[[#tmp:]] = OpLoad %[[#]] %[[#reg_1]]
; CHECK:       %[[#cond:]] = OpINotEqual %[[#]] %[[#tmp]] %[[#]]
; CHECK:                      OpSelectionMerge %[[#cond4_merge:]] None
; CHECK:                      OpBranchConditional %[[#cond]] %[[#cond4_true:]] %[[#cond4_merge]]
cond3_merge:
  %cond9 = phi i32 [ %call5, %cond3_true ], [ %call7, %cond3_false ]
  %tobool10 = icmp ne i32 %cond9, 0
  br i1 %tobool10, label %cond4_true, label %cond4_merge

; CHECK:  %[[#cond4_true]] = OpLabel
; CHECK:                     OpBranch %[[#cond4_merge]]
cond4_true:
  store i32 0, ptr %a
  br label %cond4_merge

; CHECK:  %[[#cond4_merge]] = OpLabel
; CHECK:                  OpReturn
cond4_merge:
  ret void
}

declare token @llvm.experimental.convergence.entry() #2

attributes #0 = { convergent noinline norecurse nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent norecurse "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #3 = { convergent }
attributes #4 = { convergent noinline nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"dx.disable_optimizations", i32 1}
