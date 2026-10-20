; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

;
; int foo() { return true; }
;
; [numthreads(1, 1, 1)]
; void main() {
;   int val = 0;
;   int i = 0;
;
;   while (i < 10) {
;     val = i;
;     if (val > 5) {
;       break;
;     }
;
;     if (val > 6) {
;       break;
;       break;       // No SPIR-V should be emitted for this statement.
;       val++;       // No SPIR-V should be emitted for this statement.
;       while(true); // No SPIR-V should be emitted for this statement.
;       --i;         // No SPIR-V should be emitted for this statement.
;     }
;   }
;
;   ////////////////////////////////////////////////////////////////////////////////
;   // Nested while loops with break statements                                   //
;   // Each break statement should branch to the corresponding loop's break block //
;   ////////////////////////////////////////////////////////////////////////////////
;
;   while (true) {
;     i++;
;     while(i<20) {
;       val = i;
;       {{break;}}
;     }
;     --i;
;     break;
;   }
; }

; CHECK: %[[#func_16:]] = OpFunction %[[#uint:]] DontInline %[[#]]
; CHECK:    %[[#bb44:]] = OpLabel
; CHECK:                  OpReturnValue %[[#]]
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_19:]] = OpFunction %[[#void:]] DontInline %[[#]]
; CHECK:    %[[#bb45:]] = OpLabel
; CHECK:                  OpBranch %[[#bb46:]]
; CHECK:    %[[#bb46:]] = OpLabel
; CHECK:                  OpLoopMerge %[[#bb47:]] %[[#bb48:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb49:]] %[[#bb47:]]
; CHECK:    %[[#bb49:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb47:]] %[[#bb50:]]
; CHECK:    %[[#bb50:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb47:]] %[[#bb48:]]
; CHECK:    %[[#bb48:]] = OpLabel
; CHECK:                  OpBranch %[[#bb46:]]
; CHECK:    %[[#bb47:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb51:]] None
; CHECK:                  OpSwitch %[[#]] %[[#bb51:]] 1 %[[#bb51:]] 2 %[[#bb51:]]
; CHECK:    %[[#bb51:]] = OpLabel
; CHECK:                  OpBranch %[[#bb52:]]
; CHECK:    %[[#bb52:]] = OpLabel
; CHECK:                  OpBranch %[[#bb53:]]
; CHECK:    %[[#bb53:]] = OpLabel
; CHECK:                  OpBranch %[[#bb54:]]
; CHECK:    %[[#bb54:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb55:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb56:]] %[[#bb55:]]
; CHECK:    %[[#bb56:]] = OpLabel
; CHECK:                  OpBranch %[[#bb55:]]
; CHECK:    %[[#bb55:]] = OpLabel
; CHECK:                  OpBranch %[[#bb57:]]
; CHECK:    %[[#bb57:]] = OpLabel
; CHECK:                  OpReturn
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_40:]] = OpFunction %[[#void:]] None %[[#]]
; CHECK:    %[[#bb58:]] = OpLabel
; CHECK:                  OpReturn
; CHECK:                  OpFunctionEnd

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

; Function Attrs: convergent noinline norecurse nounwind optnone
define spir_func noundef i32 @_Z3foov() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  ret i32 1
}

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare token @llvm.experimental.convergence.entry() #1

; Function Attrs: convergent noinline norecurse nounwind optnone
define internal spir_func void @main() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %val = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 0, ptr %val, align 4
  store i32 0, ptr %i, align 4
  br label %while.cond

while.cond:                                       ; preds = %if.end4, %entry
  %1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  %2 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %2, 10
  br i1 %cmp, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  %3 = load i32, ptr %i, align 4
  store i32 %3, ptr %val, align 4
  %4 = load i32, ptr %val, align 4
  %cmp1 = icmp sgt i32 %4, 5
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %while.body
  br label %while.end

if.end:                                           ; preds = %while.body
  %5 = load i32, ptr %val, align 4
  %cmp2 = icmp sgt i32 %5, 6
  br i1 %cmp2, label %if.then3, label %if.end4

if.then3:                                         ; preds = %if.end
  br label %while.end

if.end4:                                          ; preds = %if.end
  br label %while.cond

while.end:                                        ; preds = %if.then3, %if.then, %while.cond
  br label %while.cond5

while.cond5:                                      ; preds = %while.end
  %6 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  br label %while.body6

while.body6:                                      ; preds = %while.cond5
  %7 = load i32, ptr %i, align 4
  %inc = add nsw i32 %7, 1
  store i32 %inc, ptr %i, align 4
  br label %while.cond7

while.cond7:                                      ; preds = %while.body6
  %8 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %6) ]
  %9 = load i32, ptr %i, align 4
  %cmp8 = icmp slt i32 %9, 20
  br i1 %cmp8, label %while.body9, label %while.end10

while.body9:                                      ; preds = %while.cond7
  %10 = load i32, ptr %i, align 4
  store i32 %10, ptr %val, align 4
  br label %while.end10

while.end10:                                      ; preds = %while.body9, %while.cond7
  %11 = load i32, ptr %i, align 4
  %dec = add nsw i32 %11, -1
  store i32 %dec, ptr %i, align 4
  br label %while.end11

while.end11:                                      ; preds = %while.end10
  ret void
}

; Function Attrs: convergent norecurse
define void @main.1() #2 {
entry:
  call void @main()
  ret void
}

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare token @llvm.experimental.convergence.loop() #1

attributes #0 = { convergent noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { convergent norecurse "frame-pointer"="all" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.module.flags = !{!0, !1, !2}


!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"dx.disable_optimizations", i32 1}
!2 = !{i32 7, !"frame-pointer", i32 2}


