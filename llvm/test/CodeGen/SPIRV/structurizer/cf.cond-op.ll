; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

; CHECK-DAG: OpName %[[#process:]] "_Z7processv"
; CHECK-DAG: OpName %[[#fn:]] "_Z2fnv"
; CHECK-DAG: OpName %[[#fn1:]] "_Z3fn1v"
; CHECK-DAG: OpName %[[#fn2:]] "_Z3fn2v"
; CHECK-DAG: OpName %[[#val:]] "val"
; CHECK-DAG: OpName %[[#a:]] "a"
; CHECK-DAG: OpName %[[#b:]] "b"
; CHECK-DAG: OpName %[[#c:]] "c"

; CHECK-DAG: %[[#int_ty:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#bool_ty:]] = OpTypeBool
; CHECK-DAG: %[[#int_pfty:]] = OpTypePointer Function %[[#int_ty]]

; CHECK-DAG: %[[#int_0:]] = OpConstant %[[#int_ty]] 0

declare token @llvm.experimental.convergence.entry() #1

; CHECK: %[[#fn]] = OpFunction %[[#int_ty]]
define spir_func noundef i32 @_Z2fnv() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  ret i32 1
}

; CHECK: %[[#fn1]] = OpFunction %[[#int_ty]]
define spir_func noundef i32 @_Z3fn1v() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  ret i32 0
}

; CHECK: %[[#fn2]] = OpFunction %[[#int_ty]]
define spir_func noundef i32 @_Z3fn2v() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  ret i32 1
}

; CHECK: %[[#process]] = OpFunction %[[#int_ty]]
define spir_func noundef i32 @_Z7processv() #0 {
entry:
  ; CHECK:     %[[#entry:]] = OpLabel
  %0 = call token @llvm.experimental.convergence.entry()
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %c = alloca i32, align 4
  %val = alloca i32, align 4
  store i32 0, ptr %a, align 4
  store i32 1, ptr %b, align 4
  store i32 2, ptr %c, align 4
  store i32 0, ptr %val, align 4
  ; CHECK-DAG:      %[[#a]] = OpVariable %[[#int_pfty]] Function
  ; CHECK-DAG:      %[[#b]] = OpVariable %[[#int_pfty]] Function
  ; CHECK-DAG:      %[[#c]] = OpVariable %[[#int_pfty]] Function
  ; CHECK-DAG:    %[[#val]] = OpVariable %[[#int_pfty]] Function
  %1 = load i32, ptr %a, align 4
  %tobool = icmp ne i32 %1, 0
  br i1 %tobool, label %cond.true, label %cond.false
  ; CHECK:        %[[#tmp:]] = OpLoad %[[#int_ty]] %[[#a]]
  ; CHECK:       %[[#cond:]] = OpINotEqual %[[#bool_ty]] %[[#tmp]] %[[#int_0]]
  ; CHECK:                     OpSelectionMerge %[[#cond_end:]]
  ; CHECK:                     OpBranchConditional %[[#cond]] %[[#cond_true:]] %[[#cond_false:]]

cond.true:                                        ; preds = %entry
  %2 = load i32, ptr %b, align 4
  br label %cond.end
  ; CHECK: %[[#cond_true]] = OpLabel
  ; CHECK:                   OpBranch %[[#cond_end]]

cond.false:                                       ; preds = %entry
  %3 = load i32, ptr %c, align 4
  br label %cond.end
  ; CHECK: %[[#cond_false]] = OpLabel
  ; CHECK:    %[[#load_c:]] = OpLoad %[[#]] %[[#c]]
  ; CHECK:                    OpBranch %[[#cond_end]]

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %2, %cond.true ], [ %3, %cond.false ]
  %tobool1 = icmp ne i32 %cond, 0
  br i1 %tobool1, label %if.then, label %if.end
  ; CHECK: %[[#cond_end]] = OpLabel
  ; CHECK:     %[[#tmp:]] = OpPhi %[[#int_ty]] %[[#load_b:]] %[[#cond_true]] %[[#load_c]] %[[#cond_false]]
  ; CHECK:                  OpSelectionMerge %[[#if_end:]]
  ; CHECK:                  OpBranchConditional %[[#]] %[[#if_then:]] %[[#if_end]]

if.then:                                          ; preds = %cond.end
  %4 = load i32, ptr %val, align 4
  %inc = add nsw i32 %4, 1
  store i32 %inc, ptr %val, align 4
  br label %if.end
  ; CHECK: %[[#if_then]] = OpLabel
  ; CHECK:                 OpBranch %[[#if_end]]

if.end:                                           ; preds = %if.then, %cond.end
  %call2 = call spir_func noundef i32 @_Z2fnv() #4 [ "convergencectrl"(token %0) ]
  %tobool3 = icmp ne i32 %call2, 0
  br i1 %tobool3, label %cond.true4, label %cond.false6
  ; CHECK: %[[#if_end]] = OpLabel
  ; CHECK:                OpSelectionMerge %[[#cond_end8:]]
  ; CHECK:                OpBranchConditional %[[#]] %[[#cond_true4:]] %[[#cond_false6:]]

cond.true4:                                       ; preds = %if.end
  %call5 = call spir_func noundef i32 @_Z3fn1v() #4 [ "convergencectrl"(token %0) ]
  br label %cond.end8
  ; CHECK: %[[#cond_true4]] = OpLabel
  ; CHECK:                   OpBranch %[[#cond_end8]]

cond.false6:                                      ; preds = %if.end
  %call7 = call spir_func noundef i32 @_Z3fn2v() #4 [ "convergencectrl"(token %0) ]
  br label %cond.end8
  ; CHECK: %[[#cond_false6]] = OpLabel
  ; CHECK:                     OpBranch %[[#cond_end8]]

cond.end8:                                        ; preds = %cond.false6, %cond.true4
  %cond9 = phi i32 [ %call5, %cond.true4 ], [ %call7, %cond.false6 ]
  %tobool10 = icmp ne i32 %cond9, 0
  br i1 %tobool10, label %if.then11, label %if.end13
  ; CHECK: %[[#cond_end8]] = OpLabel
  ; CHECK:                   OpSelectionMerge %[[#if_end13:]]
  ; CHECK:                   OpBranchConditional %[[#]] %[[#if_then11:]] %[[#if_end13]]

if.then11:                                        ; preds = %cond.end8
  %5 = load i32, ptr %val, align 4
  %inc12 = add nsw i32 %5, 1
  store i32 %inc12, ptr %val, align 4
  br label %if.end13
  ; CHECK: %[[#if_then11]] = OpLabel
  ; CHECK:                   OpBranch %[[#if_end13]]

if.end13:                                         ; preds = %if.then11, %cond.end8
  %6 = load i32, ptr %val, align 4
  ret i32 %6
  ; CHECK: %[[#if_end13]] = OpLabel
  ; CHECK:                  OpReturnValue
}

; Function Attrs: convergent noinline norecurse nounwind optnone
define internal spir_func void @main() #2 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %call1 = call spir_func noundef i32 @_Z7processv() #4 [ "convergencectrl"(token %0) ]
  ret void
}

; Function Attrs: convergent norecurse
define void @main.1() #3 {
entry:
  call void @main()
  ret void
}

attributes #0 = { convergent noinline nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { convergent noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { convergent norecurse "frame-pointer"="all" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #4 = { convergent }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"dx.disable_optimizations", i32 1}
!2 = !{i32 7, !"frame-pointer", i32 2}
