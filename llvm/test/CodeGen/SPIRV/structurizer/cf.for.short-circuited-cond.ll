; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}
; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | spirv-sim --function=_Z7processv --wave=1 --expects=9

;
; int process() {
;   int a = 0;
;   int b = 1;
;   int val = 0;
;
;   for (int i = 0; a && b; ++i) {
;     val += 1;
;   }
;
;   for (int i = 0; a || b; ++i) {
;     val += 1;
;     b = 0;
;   }
;
;   b = 1;
;   for (int i = 0; a && ((a || b) && b); ++i) {
;     val += 4;
;     b = 0;
;   }
;
;   b = 1;
;   for (int i = 0; a ? a : b; ++i) {
;     val += 8;
;     b = 0;
;   }
;
;   int x = 0;
;   int y = 0;
;   for (int i = 0; x + (x && y); ++i) {
;     val += 16;
;   }
;
;   return val;
; }
;
; [numthreads(1, 1, 1)]
; void main() {
;   process();
; }

; CHECK: %[[#func_14:]] = OpFunction %[[#uint:]] DontInline %[[#]]
; CHECK:    %[[#bb87:]] = OpLabel
; CHECK:                  OpBranch %[[#bb88:]]
; CHECK:    %[[#bb88:]] = OpLabel
; CHECK:                  OpLoopMerge %[[#bb89:]] %[[#bb90:]] None
; CHECK:                  OpBranch %[[#bb91:]]
; CHECK:    %[[#bb91:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb92:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb93:]] %[[#bb92:]]
; CHECK:    %[[#bb93:]] = OpLabel
; CHECK:                  OpBranch %[[#bb92:]]
; CHECK:    %[[#bb92:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb94:]] %[[#bb89:]]
; CHECK:    %[[#bb94:]] = OpLabel
; CHECK:                  OpBranch %[[#bb90:]]
; CHECK:    %[[#bb89:]] = OpLabel
; CHECK:                  OpBranch %[[#bb95:]]
; CHECK:    %[[#bb90:]] = OpLabel
; CHECK:                  OpBranch %[[#bb88:]]
; CHECK:    %[[#bb95:]] = OpLabel
; CHECK:                  OpLoopMerge %[[#bb96:]] %[[#bb97:]] None
; CHECK:                  OpBranch %[[#bb98:]]
; CHECK:    %[[#bb98:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb99:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb99:]] %[[#bb100:]]
; CHECK:   %[[#bb100:]] = OpLabel
; CHECK:                  OpBranch %[[#bb99:]]
; CHECK:    %[[#bb99:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb101:]] %[[#bb96:]]
; CHECK:   %[[#bb101:]] = OpLabel
; CHECK:                  OpBranch %[[#bb97:]]
; CHECK:    %[[#bb96:]] = OpLabel
; CHECK:                  OpBranch %[[#bb102:]]
; CHECK:    %[[#bb97:]] = OpLabel
; CHECK:                  OpBranch %[[#bb95:]]
; CHECK:   %[[#bb102:]] = OpLabel
; CHECK:                  OpLoopMerge %[[#bb103:]] %[[#bb104:]] None
; CHECK:                  OpBranch %[[#bb105:]]
; CHECK:   %[[#bb105:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb106:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb107:]] %[[#bb106:]]
; CHECK:   %[[#bb107:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb108:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb109:]] %[[#bb110:]]
; CHECK:   %[[#bb109:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb111:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb111:]] %[[#bb112:]]
; CHECK:   %[[#bb110:]] = OpLabel
; CHECK:   %[[#bb112:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb111:]] %[[#bb113:]]
; CHECK:   %[[#bb113:]] = OpLabel
; CHECK:                  OpBranch %[[#bb111:]]
; CHECK:   %[[#bb111:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb114:]] %[[#bb108:]]
; CHECK:   %[[#bb114:]] = OpLabel
; CHECK:                  OpBranch %[[#bb108:]]
; CHECK:   %[[#bb108:]] = OpLabel
; CHECK:                  OpBranch %[[#bb106:]]
; CHECK:   %[[#bb106:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb115:]] %[[#bb103:]]
; CHECK:   %[[#bb115:]] = OpLabel
; CHECK:                  OpBranch %[[#bb104:]]
; CHECK:   %[[#bb103:]] = OpLabel
; CHECK:                  OpBranch %[[#bb116:]]
; CHECK:   %[[#bb104:]] = OpLabel
; CHECK:                  OpBranch %[[#bb102:]]
; CHECK:   %[[#bb116:]] = OpLabel
; CHECK:                  OpLoopMerge %[[#bb117:]] %[[#bb118:]] None
; CHECK:                  OpBranch %[[#bb119:]]
; CHECK:   %[[#bb119:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb120:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb121:]] %[[#bb122:]]
; CHECK:   %[[#bb121:]] = OpLabel
; CHECK:                  OpBranch %[[#bb120:]]
; CHECK:   %[[#bb122:]] = OpLabel
; CHECK:                  OpBranch %[[#bb120:]]
; CHECK:   %[[#bb120:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb123:]] %[[#bb117:]]
; CHECK:   %[[#bb123:]] = OpLabel
; CHECK:                  OpBranch %[[#bb118:]]
; CHECK:   %[[#bb117:]] = OpLabel
; CHECK:                  OpBranch %[[#bb124:]]
; CHECK:   %[[#bb118:]] = OpLabel
; CHECK:                  OpBranch %[[#bb116:]]
; CHECK:   %[[#bb124:]] = OpLabel
; CHECK:                  OpLoopMerge %[[#bb125:]] %[[#bb126:]] None
; CHECK:                  OpBranch %[[#bb127:]]
; CHECK:   %[[#bb127:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb128:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb129:]] %[[#bb128:]]
; CHECK:   %[[#bb129:]] = OpLabel
; CHECK:                  OpBranch %[[#bb128:]]
; CHECK:   %[[#bb128:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb130:]] %[[#bb125:]]
; CHECK:   %[[#bb130:]] = OpLabel
; CHECK:                  OpBranch %[[#bb126:]]
; CHECK:   %[[#bb125:]] = OpLabel
; CHECK:                  OpReturnValue %[[#]]
; CHECK:   %[[#bb126:]] = OpLabel
; CHECK:                  OpBranch %[[#bb124:]]
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_83:]] = OpFunction %[[#void:]] DontInline %[[#]]
; CHECK:   %[[#bb131:]] = OpLabel
; CHECK:                  OpReturn
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_85:]] = OpFunction %[[#void:]] None %[[#]]
; CHECK:   %[[#bb132:]] = OpLabel
; CHECK:                  OpReturn
; CHECK:                  OpFunctionEnd

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

; Function Attrs: convergent noinline norecurse nounwind optnone
define spir_func noundef i32 @_Z7processv() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %val = alloca i32, align 4
  %i = alloca i32, align 4
  %i2 = alloca i32, align 4
  %i11 = alloca i32, align 4
  %i26 = alloca i32, align 4
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  %i35 = alloca i32, align 4
  store i32 0, ptr %a, align 4
  store i32 1, ptr %b, align 4
  store i32 0, ptr %val, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  %2 = load i32, ptr %a, align 4
  %tobool = icmp ne i32 %2, 0
  br i1 %tobool, label %land.rhs, label %land.end

land.rhs:                                         ; preds = %for.cond
  %3 = load i32, ptr %b, align 4
  %tobool1 = icmp ne i32 %3, 0
  br label %land.end

land.end:                                         ; preds = %land.rhs, %for.cond
  %4 = phi i1 [ false, %for.cond ], [ %tobool1, %land.rhs ]
  br i1 %4, label %for.body, label %for.end

for.body:                                         ; preds = %land.end
  %5 = load i32, ptr %val, align 4
  %add = add nsw i32 %5, 1
  store i32 %add, ptr %val, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %6 = load i32, ptr %i, align 4
  %inc = add nsw i32 %6, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond

for.end:                                          ; preds = %land.end
  store i32 0, ptr %i2, align 4
  br label %for.cond3

for.cond3:                                        ; preds = %for.inc8, %for.end
  %7 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  %8 = load i32, ptr %a, align 4
  %tobool4 = icmp ne i32 %8, 0
  br i1 %tobool4, label %lor.end, label %lor.rhs

lor.rhs:                                          ; preds = %for.cond3
  %9 = load i32, ptr %b, align 4
  %tobool5 = icmp ne i32 %9, 0
  br label %lor.end

lor.end:                                          ; preds = %lor.rhs, %for.cond3
  %10 = phi i1 [ true, %for.cond3 ], [ %tobool5, %lor.rhs ]
  br i1 %10, label %for.body6, label %for.end10

for.body6:                                        ; preds = %lor.end
  %11 = load i32, ptr %val, align 4
  %add7 = add nsw i32 %11, 1
  store i32 %add7, ptr %val, align 4
  store i32 0, ptr %b, align 4
  br label %for.inc8

for.inc8:                                         ; preds = %for.body6
  %12 = load i32, ptr %i2, align 4
  %inc9 = add nsw i32 %12, 1
  store i32 %inc9, ptr %i2, align 4
  br label %for.cond3

for.end10:                                        ; preds = %lor.end
  store i32 1, ptr %b, align 4
  store i32 0, ptr %i11, align 4
  br label %for.cond12

for.cond12:                                       ; preds = %for.inc23, %for.end10
  %13 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  %14 = load i32, ptr %a, align 4
  %tobool13 = icmp ne i32 %14, 0
  br i1 %tobool13, label %land.rhs14, label %land.end20

land.rhs14:                                       ; preds = %for.cond12
  %15 = load i32, ptr %a, align 4
  %tobool15 = icmp ne i32 %15, 0
  br i1 %tobool15, label %land.rhs17, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %land.rhs14
  %16 = load i32, ptr %b, align 4
  %tobool16 = icmp ne i32 %16, 0
  br i1 %tobool16, label %land.rhs17, label %land.end19

land.rhs17:                                       ; preds = %lor.lhs.false, %land.rhs14
  %17 = load i32, ptr %b, align 4
  %tobool18 = icmp ne i32 %17, 0
  br label %land.end19

land.end19:                                       ; preds = %land.rhs17, %lor.lhs.false
  %18 = phi i1 [ false, %lor.lhs.false ], [ %tobool18, %land.rhs17 ]
  br label %land.end20

land.end20:                                       ; preds = %land.end19, %for.cond12
  %19 = phi i1 [ false, %for.cond12 ], [ %18, %land.end19 ]
  br i1 %19, label %for.body21, label %for.end25

for.body21:                                       ; preds = %land.end20
  %20 = load i32, ptr %val, align 4
  %add22 = add nsw i32 %20, 4
  store i32 %add22, ptr %val, align 4
  store i32 0, ptr %b, align 4
  br label %for.inc23

for.inc23:                                        ; preds = %for.body21
  %21 = load i32, ptr %i11, align 4
  %inc24 = add nsw i32 %21, 1
  store i32 %inc24, ptr %i11, align 4
  br label %for.cond12

for.end25:                                        ; preds = %land.end20
  store i32 1, ptr %b, align 4
  store i32 0, ptr %i26, align 4
  br label %for.cond27

for.cond27:                                       ; preds = %for.inc32, %for.end25
  %22 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  %23 = load i32, ptr %a, align 4
  %tobool28 = icmp ne i32 %23, 0
  br i1 %tobool28, label %cond.true, label %cond.false

cond.true:                                        ; preds = %for.cond27
  %24 = load i32, ptr %a, align 4
  br label %cond.end

cond.false:                                       ; preds = %for.cond27
  %25 = load i32, ptr %b, align 4
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %24, %cond.true ], [ %25, %cond.false ]
  %tobool29 = icmp ne i32 %cond, 0
  br i1 %tobool29, label %for.body30, label %for.end34

for.body30:                                       ; preds = %cond.end
  %26 = load i32, ptr %val, align 4
  %add31 = add nsw i32 %26, 8
  store i32 %add31, ptr %val, align 4
  store i32 0, ptr %b, align 4
  br label %for.inc32

for.inc32:                                        ; preds = %for.body30
  %27 = load i32, ptr %i26, align 4
  %inc33 = add nsw i32 %27, 1
  store i32 %inc33, ptr %i26, align 4
  br label %for.cond27

for.end34:                                        ; preds = %cond.end
  store i32 0, ptr %x, align 4
  store i32 0, ptr %y, align 4
  store i32 0, ptr %i35, align 4
  br label %for.cond36

for.cond36:                                       ; preds = %for.inc45, %for.end34
  %28 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  %29 = load i32, ptr %x, align 4
  %30 = load i32, ptr %x, align 4
  %tobool37 = icmp ne i32 %30, 0
  br i1 %tobool37, label %land.rhs38, label %land.end40

land.rhs38:                                       ; preds = %for.cond36
  %31 = load i32, ptr %y, align 4
  %tobool39 = icmp ne i32 %31, 0
  br label %land.end40

land.end40:                                       ; preds = %land.rhs38, %for.cond36
  %32 = phi i1 [ false, %for.cond36 ], [ %tobool39, %land.rhs38 ]
  %conv = zext i1 %32 to i32
  %add41 = add nsw i32 %29, %conv
  %tobool42 = icmp ne i32 %add41, 0
  br i1 %tobool42, label %for.body43, label %for.end47

for.body43:                                       ; preds = %land.end40
  %33 = load i32, ptr %val, align 4
  %add44 = add nsw i32 %33, 16
  store i32 %add44, ptr %val, align 4
  br label %for.inc45

for.inc45:                                        ; preds = %for.body43
  %34 = load i32, ptr %i35, align 4
  %inc46 = add nsw i32 %34, 1
  store i32 %inc46, ptr %i35, align 4
  br label %for.cond36

for.end47:                                        ; preds = %land.end40
  %35 = load i32, ptr %val, align 4
  ret i32 %35
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


