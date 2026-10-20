; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

;
; int process() {
;   int color = 0;
;
;   int val = 0;
;
;   if (color < 0) {
;     val = 1;
;   }
;
;   // for-stmt following if-stmt
;   for (int i = 0; i < 10; ++i) {
;     if (color < 0) { // if-stmt nested in for-stmt
;       val = val + 1;
;       for (int j = 0; j < 15; ++j) { // for-stmt deeply nested in if-then
;         val = val * 2;
;       } // end for (int j
;       val = val + 3;
;     }
;
;     if (color < 1) { // if-stmt following if-stmt
;       val = val * 4;
;     } else {
;       for (int k = 0; k < 20; ++k) { // for-stmt deeply nested in if-else
;         val = val - 5;
;         if (val < 0) { // deeply nested if-stmt
;           val = val + 100;
;         }
;       } // end for (int k
;     } // end elsek
;   } // end for (int i
;
;   // if-stmt following for-stmt
;   if (color < 2) {
;     val = val + 6;
;   }
;
;   return val;
; }
;
; [numthreads(1, 1, 1)]
; void main() {
;   process();
; }

; CHECK: %[[#func_18:]] = OpFunction %[[#uint:]] DontInline %[[#]]
; CHECK:    %[[#bb65:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb66:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb67:]] %[[#bb66]]
; CHECK:     %[[#bb67]] = OpLabel
; CHECK:                  OpBranch %[[#bb66]]
; CHECK:     %[[#bb66]] = OpLabel
; CHECK:                  OpBranch %[[#bb68:]]
; CHECK:     %[[#bb68]] = OpLabel
; CHECK:                  OpLoopMerge %[[#bb69:]] %[[#bb70:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb71:]] %[[#bb69]]
; CHECK:     %[[#bb69]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb72:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb73:]] %[[#bb72]]
; CHECK:     %[[#bb73]] = OpLabel
; CHECK:                  OpBranch %[[#bb72]]
; CHECK:     %[[#bb72]] = OpLabel
; CHECK:                  OpReturnValue %[[#]]
; CHECK:     %[[#bb71]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb74:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb75:]] %[[#bb74]]
; CHECK:     %[[#bb75]] = OpLabel
; CHECK:                  OpBranch %[[#bb76:]]
; CHECK:     %[[#bb76]] = OpLabel
; CHECK:                  OpLoopMerge %[[#bb77:]] %[[#bb78:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb79:]] %[[#bb77]]
; CHECK:     %[[#bb77]] = OpLabel
; CHECK:                  OpBranch %[[#bb74]]
; CHECK:     %[[#bb74]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb80:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb81:]] %[[#bb82:]]
; CHECK:     %[[#bb82]] = OpLabel
; CHECK:                  OpBranch %[[#bb83:]]
; CHECK:     %[[#bb83]] = OpLabel
; CHECK:                  OpLoopMerge %[[#bb84:]] %[[#bb85:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb86:]] %[[#bb84]]
; CHECK:     %[[#bb84]] = OpLabel
; CHECK:                  OpBranch %[[#bb80]]
; CHECK:     %[[#bb86]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb87:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb88:]] %[[#bb87]]
; CHECK:     %[[#bb88]] = OpLabel
; CHECK:                  OpBranch %[[#bb87]]
; CHECK:     %[[#bb87]] = OpLabel
; CHECK:                  OpBranch %[[#bb85]]
; CHECK:     %[[#bb85]] = OpLabel
; CHECK:                  OpBranch %[[#bb83]]
; CHECK:     %[[#bb81]] = OpLabel
; CHECK:                  OpBranch %[[#bb80]]
; CHECK:     %[[#bb80]] = OpLabel
; CHECK:                  OpBranch %[[#bb70]]
; CHECK:     %[[#bb70]] = OpLabel
; CHECK:                  OpBranch %[[#bb68]]
; CHECK:     %[[#bb79]] = OpLabel
; CHECK:                  OpBranch %[[#bb78]]
; CHECK:     %[[#bb78]] = OpLabel
; CHECK:                  OpBranch %[[#bb76]]
; CHECK:                  OpFunctionEnd

; CHECK: %[[#func_61:]] = OpFunction %[[#void:]] DontInline %[[#]]
; CHECK:    %[[#bb89:]] = OpLabel
; CHECK:                  OpReturn
; CHECK:                  OpFunctionEnd

; CHECK: %[[#func_63:]] = OpFunction %[[#void]] None %[[#]]
; CHECK:    %[[#bb90:]] = OpLabel
; CHECK:                  OpReturn
; CHECK:                  OpFunctionEnd


target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

; Function Attrs: convergent noinline norecurse nounwind optnone
define spir_func noundef i32 @_Z7processv() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %color = alloca i32, align 4
  %val = alloca i32, align 4
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %k = alloca i32, align 4
  store i32 0, ptr %color, align 4
  store i32 0, ptr %val, align 4
  %1 = load i32, ptr %color, align 4
  %cmp = icmp slt i32 %1, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 1, ptr %val, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc23, %if.end
  %2 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  %3 = load i32, ptr %i, align 4
  %cmp1 = icmp slt i32 %3, 10
  br i1 %cmp1, label %for.body, label %for.end25

for.body:                                         ; preds = %for.cond
  %4 = load i32, ptr %color, align 4
  %cmp2 = icmp slt i32 %4, 0
  br i1 %cmp2, label %if.then3, label %if.end8

if.then3:                                         ; preds = %for.body
  %5 = load i32, ptr %val, align 4
  %add = add nsw i32 %5, 1
  store i32 %add, ptr %val, align 4
  store i32 0, ptr %j, align 4
  br label %for.cond4

for.cond4:                                        ; preds = %for.inc, %if.then3
  %6 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %2) ]
  %7 = load i32, ptr %j, align 4
  %cmp5 = icmp slt i32 %7, 15
  br i1 %cmp5, label %for.body6, label %for.end

for.body6:                                        ; preds = %for.cond4
  %8 = load i32, ptr %val, align 4
  %mul = mul nsw i32 %8, 2
  store i32 %mul, ptr %val, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body6
  %9 = load i32, ptr %j, align 4
  %inc = add nsw i32 %9, 1
  store i32 %inc, ptr %j, align 4
  br label %for.cond4

for.end:                                          ; preds = %for.cond4
  %10 = load i32, ptr %val, align 4
  %add7 = add nsw i32 %10, 3
  store i32 %add7, ptr %val, align 4
  br label %if.end8

if.end8:                                          ; preds = %for.end, %for.body
  %11 = load i32, ptr %color, align 4
  %cmp9 = icmp slt i32 %11, 1
  br i1 %cmp9, label %if.then10, label %if.else

if.then10:                                        ; preds = %if.end8
  %12 = load i32, ptr %val, align 4
  %mul11 = mul nsw i32 %12, 4
  store i32 %mul11, ptr %val, align 4
  br label %if.end22

if.else:                                          ; preds = %if.end8
  store i32 0, ptr %k, align 4
  br label %for.cond12

for.cond12:                                       ; preds = %for.inc19, %if.else
  %13 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %2) ]
  %14 = load i32, ptr %k, align 4
  %cmp13 = icmp slt i32 %14, 20
  br i1 %cmp13, label %for.body14, label %for.end21

for.body14:                                       ; preds = %for.cond12
  %15 = load i32, ptr %val, align 4
  %sub = sub nsw i32 %15, 5
  store i32 %sub, ptr %val, align 4
  %16 = load i32, ptr %val, align 4
  %cmp15 = icmp slt i32 %16, 0
  br i1 %cmp15, label %if.then16, label %if.end18

if.then16:                                        ; preds = %for.body14
  %17 = load i32, ptr %val, align 4
  %add17 = add nsw i32 %17, 100
  store i32 %add17, ptr %val, align 4
  br label %if.end18

if.end18:                                         ; preds = %if.then16, %for.body14
  br label %for.inc19

for.inc19:                                        ; preds = %if.end18
  %18 = load i32, ptr %k, align 4
  %inc20 = add nsw i32 %18, 1
  store i32 %inc20, ptr %k, align 4
  br label %for.cond12

for.end21:                                        ; preds = %for.cond12
  br label %if.end22

if.end22:                                         ; preds = %for.end21, %if.then10
  br label %for.inc23

for.inc23:                                        ; preds = %if.end22
  %19 = load i32, ptr %i, align 4
  %inc24 = add nsw i32 %19, 1
  store i32 %inc24, ptr %i, align 4
  br label %for.cond

for.end25:                                        ; preds = %for.cond
  %20 = load i32, ptr %color, align 4
  %cmp26 = icmp slt i32 %20, 2
  br i1 %cmp26, label %if.then27, label %if.end29

if.then27:                                        ; preds = %for.end25
  %21 = load i32, ptr %val, align 4
  %add28 = add nsw i32 %21, 6
  store i32 %add28, ptr %val, align 4
  br label %if.end29

if.end29:                                         ; preds = %if.then27, %for.end25
  %22 = load i32, ptr %val, align 4
  ret i32 %22
}

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare token @llvm.experimental.convergence.entry() #1

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare token @llvm.experimental.convergence.loop() #1

; Function Attrs: convergent noinline norecurse nounwind optnone
define internal spir_func void @main() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %call1 = call spir_func noundef i32 @_Z7processv() #3 [ "convergencectrl"(token %0) ]
  ret void
}

; Function Attrs: convergent norecurse
define void @main.1() #2 {
entry:
  call void @main()
  ret void
}

attributes #0 = { convergent noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { convergent norecurse "frame-pointer"="all" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { convergent }

!llvm.module.flags = !{!0, !1, !2}


!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"dx.disable_optimizations", i32 1}
!2 = !{i32 7, !"frame-pointer", i32 2}


