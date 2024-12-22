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
;   //////////////////////////
;   //// Basic while loop ////
;   //////////////////////////
;   while (i < 10) {
;       val = i;
;   }
;
;   //////////////////////////
;   ////  infinite loop   ////
;   //////////////////////////
;   while (true) {
;       val = 0;
;   }
;
;   //////////////////////////
;   ////    Null Body     ////
;   //////////////////////////
;   while (val < 20)
;     ;
;
;   ////////////////////////////////////////////////////////////////
;   //// Condition variable has VarDecl                         ////
;   //// foo() returns an integer which must be cast to boolean ////
;   ////////////////////////////////////////////////////////////////
;   while (int a = foo()) {
;     val = a;
;   }
;
; }

; CHECK: %[[#func_11:]] = OpFunction %[[#uint:]] DontInline %[[#]]
; CHECK:    %[[#bb20:]] = OpLabel
; CHECK:                  OpReturnValue %[[#]]
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_12:]] = OpFunction %[[#void:]] DontInline %[[#]]
; CHECK:    %[[#bb21:]] = OpLabel
; CHECK:                  OpBranch %[[#bb22:]]
; CHECK:    %[[#bb22:]] = OpLabel
; CHECK:                  OpLoopMerge %[[#bb23:]] %[[#bb24:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb24:]] %[[#bb23:]]
; CHECK:    %[[#bb24:]] = OpLabel
; CHECK:                  OpBranch %[[#bb22:]]
; CHECK:    %[[#bb23:]] = OpLabel
; CHECK:                  OpBranch %[[#bb25:]]
; CHECK:    %[[#bb25:]] = OpLabel
; CHECK:                  OpLoopMerge %[[#bb26:]] %[[#bb27:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb26:]] %[[#bb27:]]
; CHECK:    %[[#bb27:]] = OpLabel
; CHECK:                  OpBranch %[[#bb25:]]
; CHECK:    %[[#bb26:]] = OpLabel
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_18:]] = OpFunction %[[#void:]] None %[[#]]
; CHECK:    %[[#bb28:]] = OpLabel
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

while.cond:                                       ; preds = %while.body, %entry
  %1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  %2 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %2, 10
  br i1 %cmp, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  %3 = load i32, ptr %i, align 4
  store i32 %3, ptr %val, align 4
  br label %while.cond

while.end:                                        ; preds = %while.cond
  br label %while.cond1

while.cond1:                                      ; preds = %while.body2, %while.end
  %4 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  br label %while.body2

while.body2:                                      ; preds = %while.cond1
  store i32 0, ptr %val, align 4
  br label %while.cond1
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


