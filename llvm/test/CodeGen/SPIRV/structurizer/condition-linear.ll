; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}
; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s --match-full-lines

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

define internal spir_func void @main() #0 {
; CHECK:    %[[#cond:]] = OpINotEqual %[[#bool_ty:]] %[[#a:]] %[[#b:]]
; CHECK:                  OpSelectionMerge %[[#cond_end:]] None
; CHECK:                  OpBranchConditional %[[#cond]] %[[#cond_true:]] %[[#cond_false:]]
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %c = alloca i32, align 4
  %val = alloca i32, align 4
  store i32 0, ptr %val, align 4
  %1 = load i32, ptr %a, align 4
  %tobool = icmp ne i32 %1, 0
  br i1 %tobool, label %cond.true, label %cond.false

; CHECK:  %[[#cond_true]] = OpLabel
; CHECK:                    OpBranch %[[#cond_end]]
cond.true:
  %2 = load i32, ptr %b, align 4
  br label %cond.end

; CHECK:  %[[#cond_false]] = OpLabel
; CHECK:                     OpBranch %[[#cond_end]]
cond.false:
  %3 = load i32, ptr %c, align 4
  br label %cond.end

; CHECK:  %[[#cond_end]] = OpLabel
; CHECK:     %[[#tmp:]]  = OpPhi %[[#int_ty:]] %[[#load_cond_true:]] %[[#cond_true]] %[[#load_cond_false:]] %[[#cond_false:]]
; CHECK:     %[[#cond:]] = OpINotEqual %[[#bool_ty]] %[[#tmp]] %[[#int_0:]]
; CHECK:                   OpSelectionMerge %[[#if_end:]] None
; CHECK:                   OpBranchConditional %[[#cond]] %[[#if_then:]] %[[#if_end]]
cond.end:
  %cond = phi i32 [ %2, %cond.true ], [ %3, %cond.false ]
  %tobool1 = icmp ne i32 %cond, 0
  br i1 %tobool1, label %if.then, label %if.end

; CHECK:  %[[#if_then]] = OpLabel
; CHECK:                  OpBranch %[[#if_end]]
if.then:
  %4 = load i32, ptr %val, align 4
  %inc = add nsw i32 %4, 1
  store i32 %inc, ptr %val, align 4
  br label %if.end

; CHECK:    %[[#if_end]] = OpLabel
; CHECK:                   OpSelectionMerge %[[#cond_end8:]] None
; CHECK:                   OpBranchConditional %[[#tmp:]] %[[#cond4_true:]] %[[#cond_false6:]]
if.end:
  %call2 = call spir_func noundef i32 @fn() #4 [ "convergencectrl"(token %0) ]
  %tobool3 = icmp ne i32 %call2, 0
  br i1 %tobool3, label %cond.true4, label %cond.false6

; CHECK:  %[[#cond4_true]] = OpLabel
; CHECK:                     OpBranch %[[#cond_end8]]
cond.true4:
  %call5 = call spir_func noundef i32 @fn1() #4 [ "convergencectrl"(token %0) ]
  br label %cond.end8

; CHECK:  %[[#cond_false6]] = OpLabel
; CHECK:                      OpBranch %[[#cond_end8]]
cond.false6:
  %call7 = call spir_func noundef i32 @fn2() #4 [ "convergencectrl"(token %0) ]
  br label %cond.end8

; CHECK:  %[[#cond_end8]] = OpLabel
; CHECK:                      OpSelectionMerge %[[#if_end13:]] None
; CHECK:                      OpBranchConditional %[[#tmp:]] %[[#if_then11:]] %[[#if_end13]]
cond.end8:
  %cond9 = phi i32 [ %call5, %cond.true4 ], [ %call7, %cond.false6 ]
  %tobool10 = icmp ne i32 %cond9, 0
  br i1 %tobool10, label %if.then11, label %if.end13

; CHECK:  %[[#if_then11]] = OpLabel
; CHECK:                    OpBranch %[[#if_end13]]
if.then11:
  %5 = load i32, ptr %val, align 4
  %inc12 = add nsw i32 %5, 1
  store i32 %inc12, ptr %val, align 4
  br label %if.end13

; CHECK:  %[[#if_end13]] = OpLabel
; CHECK:                  OpReturn
if.end13:
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
