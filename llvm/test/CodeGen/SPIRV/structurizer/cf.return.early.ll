; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

;
; int process() {
;   int cond = 1;
;   int value = 0;
;
;   while(value < 10) {
;     switch(value) {
;       case 1:
;         value = 1;
;         return value;
;       case 2: {
;         value = 3;
;         {return value;}   // Return from function.
;         value = 4;      // No SPIR-V should be emitted for this statement.
;         break;      // No SPIR-V should be emitted for this statement.
;       }
;       case 5 : {
;         value = 5;
;         {{return value;}} // Return from function.
;         value = 6;      // No SPIR-V should be emitted for this statement.
;       }
;       default:
;         for (int i=0; i<10; ++i) {
;           if (cond) {
;             return value;    // Return from function.
;             return value;    // No SPIR-V should be emitted for this statement.
;             continue;  // No SPIR-V should be emitted for this statement.
;             break;     // No SPIR-V should be emitted for this statement.
;             ++value;       // No SPIR-V should be emitted for this statement.
;           } else {
;             return value;   // Return from function
;             continue; // No SPIR-V should be emitted for this statement.
;             break;    // No SPIR-V should be emitted for this statement.
;             ++value;      // No SPIR-V should be emitted for this statement.
;           }
;         }
;
;         // Return from function.
;         // Even though this statement will never be executed [because both "if" and "else" above have return statements],
;         // SPIR-V code should be emitted for it as we do not analyze the logic.
;         return value;
;     }
;
;     // Return from function.
;     // Even though this statement will never be executed [because all "case" statements above contain a return statement],
;     // SPIR-V code should be emitted for it as we do not analyze the logic.
;     return value;
;   }
;
;   return value;
; }
;
; [numthreads(1, 1, 1)]
; void main() {
;   process();
; }

; CHECK: %[[#func_13:]] = OpFunction %[[#uint:]] DontInline %[[#]]
; CHECK:    %[[#bb46:]] = OpLabel
; CHECK:                  OpBranch %[[#bb47:]]
; CHECK:     %[[#bb47]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb48:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb49:]] %[[#bb50:]]
; CHECK:     %[[#bb50]] = OpLabel
; CHECK:                  OpBranch %[[#bb48]]
; CHECK:     %[[#bb49]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb51:]] None
; CHECK:                  OpSwitch %[[#]] %[[#bb51]] 1 %[[#bb52:]] 2 %[[#bb53:]] 5 %[[#bb54:]]
; CHECK:     %[[#bb54]] = OpLabel
; CHECK:                  OpBranch %[[#bb51]]
; CHECK:     %[[#bb53]] = OpLabel
; CHECK:                  OpBranch %[[#bb51]]
; CHECK:     %[[#bb52]] = OpLabel
; CHECK:                  OpBranch %[[#bb51]]
; CHECK:     %[[#bb51]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb55:]] %[[#bb48]]
; CHECK:     %[[#bb55]] = OpLabel
; CHECK:                  OpBranch %[[#bb56:]]
; CHECK:     %[[#bb56]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb57:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb57]] %[[#bb58:]]
; CHECK:     %[[#bb58]] = OpLabel
; CHECK:                  OpBranch %[[#bb57]]
; CHECK:     %[[#bb57]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb59:]] %[[#bb48]]
; CHECK:     %[[#bb59]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb60:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb60]] %[[#bb61:]]
; CHECK:     %[[#bb61]] = OpLabel
; CHECK:                  OpBranch %[[#bb60]]
; CHECK:     %[[#bb60]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb62:]] %[[#bb48]]
; CHECK:     %[[#bb62]] = OpLabel
; CHECK:                  OpBranch %[[#bb48]]
; CHECK:     %[[#bb48]] = OpLabel
; CHECK:                  OpReturnValue %[[#]]
; CHECK:                  OpFunctionEnd

; CHECK: %[[#func_42:]] = OpFunction %[[#void:]] DontInline %[[#]]
; CHECK:    %[[#bb63:]] = OpLabel
; CHECK:                  OpReturn
; CHECK:                  OpFunctionEnd

; CHECK: %[[#func_44:]] = OpFunction %[[#void]] None %[[#]]
; CHECK:    %[[#bb64:]] = OpLabel
; CHECK:                  OpReturn
; CHECK:                  OpFunctionEnd

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

; Function Attrs: convergent noinline norecurse nounwind optnone
define spir_func noundef i32 @_Z7processv() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %retval = alloca i32, align 4
  %cond = alloca i32, align 4
  %value = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 1, ptr %cond, align 4
  store i32 0, ptr %value, align 4
  br label %while.cond

while.cond:                                       ; preds = %entry
  %1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  %2 = load i32, ptr %value, align 4
  %cmp = icmp slt i32 %2, 10
  br i1 %cmp, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  %3 = load i32, ptr %value, align 4
  switch i32 %3, label %sw.default [
    i32 1, label %sw.bb
    i32 2, label %sw.bb1
    i32 5, label %sw.bb2
  ]

sw.bb:                                            ; preds = %while.body
  store i32 1, ptr %value, align 4
  %4 = load i32, ptr %value, align 4
  store i32 %4, ptr %retval, align 4
  br label %return

sw.bb1:                                           ; preds = %while.body
  store i32 3, ptr %value, align 4
  %5 = load i32, ptr %value, align 4
  store i32 %5, ptr %retval, align 4
  br label %return

sw.bb2:                                           ; preds = %while.body
  store i32 5, ptr %value, align 4
  %6 = load i32, ptr %value, align 4
  store i32 %6, ptr %retval, align 4
  br label %return

sw.default:                                       ; preds = %while.body
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %sw.default
  %7 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %1) ]
  %8 = load i32, ptr %i, align 4
  %cmp3 = icmp slt i32 %8, 10
  br i1 %cmp3, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %9 = load i32, ptr %cond, align 4
  %tobool = icmp ne i32 %9, 0
  br i1 %tobool, label %if.then, label %if.else

if.then:                                          ; preds = %for.body
  %10 = load i32, ptr %value, align 4
  store i32 %10, ptr %retval, align 4
  br label %return

if.else:                                          ; preds = %for.body
  %11 = load i32, ptr %value, align 4
  store i32 %11, ptr %retval, align 4
  br label %return

for.inc:                                          ; No predecessors!
  %12 = load i32, ptr %i, align 4
  %inc = add nsw i32 %12, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %13 = load i32, ptr %value, align 4
  store i32 %13, ptr %retval, align 4
  br label %return

while.end:                                        ; preds = %while.cond
  %14 = load i32, ptr %value, align 4
  store i32 %14, ptr %retval, align 4
  br label %return

return:                                           ; preds = %while.end, %for.end, %if.else, %if.then, %sw.bb2, %sw.bb1, %sw.bb
  %15 = load i32, ptr %retval, align 4
  ret i32 %15
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


