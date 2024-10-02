; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}
; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | spirv-sim --function=_Z7processv --wave=1 --expects=308

;
; int foo() { return 200; }
;
; int process() {
;   int a = 0;
;   int b = 0;
;   int c = 0;
;   const int r = 20;
;   const int s = 40;
;   const int t = 3*r+2*s;
;
;
;   ////////////////////////////////////////
;   // DefaultStmt is the first statement //
;   ////////////////////////////////////////
;   switch(a) {
;     default:
;       b += 0;
;     case 1:
;       b += 1;
;       break;
;     case 2:
;       b += 2;
;   }
;
;
;   //////////////////////////////////////////////
;   // DefaultStmt in the middle of other cases //
;   //////////////////////////////////////////////
;   switch(a) {
;     case 10:
;       b += 1;
;     default:
;       b += 0;
;     case 20:
;       b += 2;
;       break;
;   }
;
;   ///////////////////////////////////////////////
;   // Various CaseStmt and BreakStmt topologies //
;   // DefaultStmt is the last statement         //
;   ///////////////////////////////////////////////
;   switch(int d = 5) {
;     case 1:
;       b += 1;
;       c += foo();
;     case 2:
;       b += 2;
;       break;
;     case 3:
;     {
;       b += 3;
;       break;
;     }
;     case t:
;       b += t;
;     case 4:
;     case 5:
;       b += 5;
;       break;
;     case 6: {
;     case 7:
;       break;}
;     default:
;       break;
;   }
;
;
;   //////////////////////////
;   // No Default statement //
;   //////////////////////////
;   switch(a) {
;     case 100:
;       b += 100;
;       break;
;   }
;
;
;   /////////////////////////////////////////////////////////
;   // No cases. Only a default                            //
;   // This means the default body will always be executed //
;   /////////////////////////////////////////////////////////
;   switch(a) {
;     default:
;       b += 100;
;       c += 200;
;       break;
;   }
;
;
;   ////////////////////////////////////////////////////////////
;   // Nested Switch with branching                           //
;   // The two inner switch statements should be executed for //
;   // both cases of the outer switch (case 300 and case 400) //
;   ////////////////////////////////////////////////////////////
;   switch(a) {
;     case 300:
;       b += 300;
;     case 400:
;       switch(c) {
;         case 500:
;           b += 500;
;           break;
;         case 600:
;           switch(b) {
;             default:
;             a += 600;
;             b += 600;
;           }
;       }
;   }
;
;   return a + b + c;
; }
;
; [numthreads(1, 1, 1)]
; void main() {
;   process();
; }

; CHECK: %[[#func_22:]] = OpFunction %[[#uint:]] DontInline %[[#]]
; CHECK:    %[[#bb94:]] = OpLabel
; CHECK:                  OpReturnValue %[[#]]
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_23:]] = OpFunction %[[#uint:]] DontInline %[[#]]
; CHECK:    %[[#bb95:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb96:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb97:]] %[[#bb98:]]
; CHECK:    %[[#bb97:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb99:]] None
; CHECK:                  OpSwitch %[[#]] %[[#bb100:]] 1 %[[#bb99:]] 2 %[[#bb101:]]
; CHECK:    %[[#bb98:]] = OpLabel
; CHECK:   %[[#bb100:]] = OpLabel
; CHECK:                  OpBranch %[[#bb99:]]
; CHECK:   %[[#bb101:]] = OpLabel
; CHECK:                  OpBranch %[[#bb99:]]
; CHECK:    %[[#bb99:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb102:]] %[[#bb96:]]
; CHECK:   %[[#bb102:]] = OpLabel
; CHECK:                  OpBranch %[[#bb96:]]
; CHECK:    %[[#bb96:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb103:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb104:]] %[[#bb105:]]
; CHECK:   %[[#bb104:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb106:]] None
; CHECK:                  OpSwitch %[[#]] %[[#bb106:]] 10 %[[#bb107:]] 20 %[[#bb108:]]
; CHECK:   %[[#bb105:]] = OpLabel
; CHECK:   %[[#bb107:]] = OpLabel
; CHECK:                  OpBranch %[[#bb106:]]
; CHECK:   %[[#bb108:]] = OpLabel
; CHECK:                  OpBranch %[[#bb106:]]
; CHECK:   %[[#bb106:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb109:]] %[[#bb103:]]
; CHECK:   %[[#bb109:]] = OpLabel
; CHECK:                  OpBranch %[[#bb103:]]
; CHECK:   %[[#bb103:]] = OpLabel
; CHECK:                  OpBranch %[[#bb110:]]
; CHECK:   %[[#bb110:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb111:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb112:]] %[[#bb113:]]
; CHECK:   %[[#bb112:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb114:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb115:]] %[[#bb116:]]
; CHECK:   %[[#bb113:]] = OpLabel
; CHECK:   %[[#bb115:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb117:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb118:]] %[[#bb119:]]
; CHECK:   %[[#bb116:]] = OpLabel
; CHECK:   %[[#bb118:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb120:]] None
; CHECK:                  OpSwitch %[[#]] %[[#bb121:]] 1 %[[#bb122:]] 2 %[[#bb120:]] 3 %[[#bb123:]] 140 %[[#bb124:]] 4 %[[#bb125:]] 5 %[[#bb126:]] 6 %[[#bb127:]] 7 %[[#bb128:]]
; CHECK:   %[[#bb119:]] = OpLabel
; CHECK:   %[[#bb121:]] = OpLabel
; CHECK:                  OpBranch %[[#bb120:]]
; CHECK:   %[[#bb122:]] = OpLabel
; CHECK:                  OpBranch %[[#bb120:]]
; CHECK:   %[[#bb123:]] = OpLabel
; CHECK:                  OpBranch %[[#bb120:]]
; CHECK:   %[[#bb124:]] = OpLabel
; CHECK:                  OpBranch %[[#bb120:]]
; CHECK:   %[[#bb125:]] = OpLabel
; CHECK:                  OpBranch %[[#bb120:]]
; CHECK:   %[[#bb126:]] = OpLabel
; CHECK:                  OpBranch %[[#bb120:]]
; CHECK:   %[[#bb127:]] = OpLabel
; CHECK:                  OpBranch %[[#bb120:]]
; CHECK:   %[[#bb128:]] = OpLabel
; CHECK:                  OpBranch %[[#bb120:]]
; CHECK:   %[[#bb120:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb129:]] None
; CHECK:                  OpSwitch %[[#]] %[[#bb130:]] 1 %[[#bb129:]] 2 %[[#bb131:]] 3 %[[#bb132:]]
; CHECK:   %[[#bb130:]] = OpLabel
; CHECK:                  OpBranch %[[#bb129:]]
; CHECK:   %[[#bb131:]] = OpLabel
; CHECK:                  OpBranch %[[#bb129:]]
; CHECK:   %[[#bb132:]] = OpLabel
; CHECK:                  OpBranch %[[#bb129:]]
; CHECK:   %[[#bb129:]] = OpLabel
; CHECK:                  OpBranch %[[#bb117:]]
; CHECK:   %[[#bb117:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb133:]] None
; CHECK:                  OpSwitch %[[#]] %[[#bb134:]] 1 %[[#bb133:]] 2 %[[#bb135:]]
; CHECK:   %[[#bb134:]] = OpLabel
; CHECK:                  OpBranch %[[#bb133:]]
; CHECK:   %[[#bb135:]] = OpLabel
; CHECK:                  OpBranch %[[#bb133:]]
; CHECK:   %[[#bb133:]] = OpLabel
; CHECK:                  OpBranch %[[#bb114:]]
; CHECK:   %[[#bb114:]] = OpLabel
; CHECK:                  OpBranch %[[#bb111:]]
; CHECK:   %[[#bb111:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb136:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb137:]] %[[#bb136:]]
; CHECK:   %[[#bb137:]] = OpLabel
; CHECK:                  OpBranch %[[#bb136:]]
; CHECK:   %[[#bb136:]] = OpLabel
; CHECK:                  OpBranch %[[#bb138:]]
; CHECK:   %[[#bb138:]] = OpLabel
; CHECK:                  OpBranch %[[#bb139:]]
; CHECK:   %[[#bb139:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb140:]] None
; CHECK:                  OpBranchConditional %[[#]] %[[#bb141:]] %[[#bb142:]]
; CHECK:   %[[#bb141:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb143:]] None
; CHECK:                  OpSwitch %[[#]] %[[#bb143:]] 300 %[[#bb144:]] 400 %[[#bb145:]]
; CHECK:   %[[#bb142:]] = OpLabel
; CHECK:   %[[#bb144:]] = OpLabel
; CHECK:                  OpBranch %[[#bb143:]]
; CHECK:   %[[#bb145:]] = OpLabel
; CHECK:                  OpBranch %[[#bb143:]]
; CHECK:   %[[#bb143:]] = OpLabel
; CHECK:                  OpBranchConditional %[[#]] %[[#bb140:]] %[[#bb146:]]
; CHECK:   %[[#bb146:]] = OpLabel
; CHECK:                  OpSelectionMerge %[[#bb147:]] None
; CHECK:                  OpSwitch %[[#]] %[[#bb147:]] 500 %[[#bb148:]] 600 %[[#bb149:]]
; CHECK:   %[[#bb148:]] = OpLabel
; CHECK:                  OpBranch %[[#bb147:]]
; CHECK:   %[[#bb149:]] = OpLabel
; CHECK:                  OpBranch %[[#bb150:]]
; CHECK:   %[[#bb150:]] = OpLabel
; CHECK:                  OpBranch %[[#bb147:]]
; CHECK:   %[[#bb147:]] = OpLabel
; CHECK:                  OpBranch %[[#bb140:]]
; CHECK:   %[[#bb140:]] = OpLabel
; CHECK:                  OpReturnValue %[[#]]
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_90:]] = OpFunction %[[#void:]] DontInline %[[#]]
; CHECK:   %[[#bb151:]] = OpLabel
; CHECK:                  OpReturn
; CHECK:                  OpFunctionEnd
; CHECK: %[[#func_92:]] = OpFunction %[[#void:]] None %[[#]]
; CHECK:   %[[#bb152:]] = OpLabel
; CHECK:                  OpReturn
; CHECK:                  OpFunctionEnd



target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

; Function Attrs: convergent noinline norecurse nounwind optnone
define spir_func noundef i32 @_Z3foov() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  ret i32 200
}

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare token @llvm.experimental.convergence.entry() #1

; Function Attrs: convergent noinline norecurse nounwind optnone
define spir_func noundef i32 @_Z7processv() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %c = alloca i32, align 4
  %r = alloca i32, align 4
  %s = alloca i32, align 4
  %t = alloca i32, align 4
  %d = alloca i32, align 4
  store i32 0, ptr %a, align 4
  store i32 0, ptr %b, align 4
  store i32 0, ptr %c, align 4
  store i32 20, ptr %r, align 4
  store i32 40, ptr %s, align 4
  store i32 140, ptr %t, align 4
  %1 = load i32, ptr %a, align 4
  switch i32 %1, label %sw.default [
    i32 1, label %sw.bb
    i32 2, label %sw.bb2
  ]

sw.default:                                       ; preds = %entry
  %2 = load i32, ptr %b, align 4
  %add = add nsw i32 %2, 0
  store i32 %add, ptr %b, align 4
  br label %sw.bb

sw.bb:                                            ; preds = %entry, %sw.default
  %3 = load i32, ptr %b, align 4
  %add1 = add nsw i32 %3, 1
  store i32 %add1, ptr %b, align 4
  br label %sw.epilog

sw.bb2:                                           ; preds = %entry
  %4 = load i32, ptr %b, align 4
  %add3 = add nsw i32 %4, 2
  store i32 %add3, ptr %b, align 4
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.bb2, %sw.bb
  %5 = load i32, ptr %a, align 4
  switch i32 %5, label %sw.default6 [
    i32 10, label %sw.bb4
    i32 20, label %sw.bb8
  ]

sw.bb4:                                           ; preds = %sw.epilog
  %6 = load i32, ptr %b, align 4
  %add5 = add nsw i32 %6, 1
  store i32 %add5, ptr %b, align 4
  br label %sw.default6

sw.default6:                                      ; preds = %sw.epilog, %sw.bb4
  %7 = load i32, ptr %b, align 4
  %add7 = add nsw i32 %7, 0
  store i32 %add7, ptr %b, align 4
  br label %sw.bb8

sw.bb8:                                           ; preds = %sw.epilog, %sw.default6
  %8 = load i32, ptr %b, align 4
  %add9 = add nsw i32 %8, 2
  store i32 %add9, ptr %b, align 4
  br label %sw.epilog10

sw.epilog10:                                      ; preds = %sw.bb8
  store i32 5, ptr %d, align 4
  %9 = load i32, ptr %d, align 4
  switch i32 %9, label %sw.default25 [
    i32 1, label %sw.bb11
    i32 2, label %sw.bb15
    i32 3, label %sw.bb17
    i32 140, label %sw.bb19
    i32 4, label %sw.bb21
    i32 5, label %sw.bb21
    i32 6, label %sw.bb23
    i32 7, label %sw.bb24
  ]

sw.bb11:                                          ; preds = %sw.epilog10
  %10 = load i32, ptr %b, align 4
  %add12 = add nsw i32 %10, 1
  store i32 %add12, ptr %b, align 4
  %call13 = call spir_func noundef i32 @_Z3foov() #3 [ "convergencectrl"(token %0) ]
  %11 = load i32, ptr %c, align 4
  %add14 = add nsw i32 %11, %call13
  store i32 %add14, ptr %c, align 4
  br label %sw.bb15

sw.bb15:                                          ; preds = %sw.epilog10, %sw.bb11
  %12 = load i32, ptr %b, align 4
  %add16 = add nsw i32 %12, 2
  store i32 %add16, ptr %b, align 4
  br label %sw.epilog26

sw.bb17:                                          ; preds = %sw.epilog10
  %13 = load i32, ptr %b, align 4
  %add18 = add nsw i32 %13, 3
  store i32 %add18, ptr %b, align 4
  br label %sw.epilog26

sw.bb19:                                          ; preds = %sw.epilog10
  %14 = load i32, ptr %b, align 4
  %add20 = add nsw i32 %14, 140
  store i32 %add20, ptr %b, align 4
  br label %sw.bb21

sw.bb21:                                          ; preds = %sw.epilog10, %sw.epilog10, %sw.bb19
  %15 = load i32, ptr %b, align 4
  %add22 = add nsw i32 %15, 5
  store i32 %add22, ptr %b, align 4
  br label %sw.epilog26

sw.bb23:                                          ; preds = %sw.epilog10
  br label %sw.bb24

sw.bb24:                                          ; preds = %sw.epilog10, %sw.bb23
  br label %sw.epilog26

sw.default25:                                     ; preds = %sw.epilog10
  br label %sw.epilog26

sw.epilog26:                                      ; preds = %sw.default25, %sw.bb24, %sw.bb21, %sw.bb17, %sw.bb15
  %16 = load i32, ptr %a, align 4
  switch i32 %16, label %sw.epilog29 [
    i32 100, label %sw.bb27
  ]

sw.bb27:                                          ; preds = %sw.epilog26
  %17 = load i32, ptr %b, align 4
  %add28 = add nsw i32 %17, 100
  store i32 %add28, ptr %b, align 4
  br label %sw.epilog29

sw.epilog29:                                      ; preds = %sw.epilog26, %sw.bb27
  %18 = load i32, ptr %a, align 4
  switch i32 %18, label %sw.default30 [
  ]

sw.default30:                                     ; preds = %sw.epilog29
  %19 = load i32, ptr %b, align 4
  %add31 = add nsw i32 %19, 100
  store i32 %add31, ptr %b, align 4
  %20 = load i32, ptr %c, align 4
  %add32 = add nsw i32 %20, 200
  store i32 %add32, ptr %c, align 4
  br label %sw.epilog33

sw.epilog33:                                      ; preds = %sw.default30
  %21 = load i32, ptr %a, align 4
  switch i32 %21, label %sw.epilog45 [
    i32 300, label %sw.bb34
    i32 400, label %sw.bb36
  ]

sw.bb34:                                          ; preds = %sw.epilog33
  %22 = load i32, ptr %b, align 4
  %add35 = add nsw i32 %22, 300
  store i32 %add35, ptr %b, align 4
  br label %sw.bb36

sw.bb36:                                          ; preds = %sw.epilog33, %sw.bb34
  %23 = load i32, ptr %c, align 4
  switch i32 %23, label %sw.epilog44 [
    i32 500, label %sw.bb37
    i32 600, label %sw.bb39
  ]

sw.bb37:                                          ; preds = %sw.bb36
  %24 = load i32, ptr %b, align 4
  %add38 = add nsw i32 %24, 500
  store i32 %add38, ptr %b, align 4
  br label %sw.epilog44

sw.bb39:                                          ; preds = %sw.bb36
  %25 = load i32, ptr %b, align 4
  switch i32 %25, label %sw.default40 [
  ]

sw.default40:                                     ; preds = %sw.bb39
  %26 = load i32, ptr %a, align 4
  %add41 = add nsw i32 %26, 600
  store i32 %add41, ptr %a, align 4
  %27 = load i32, ptr %b, align 4
  %add42 = add nsw i32 %27, 600
  store i32 %add42, ptr %b, align 4
  br label %sw.epilog43

sw.epilog43:                                      ; preds = %sw.default40
  br label %sw.epilog44

sw.epilog44:                                      ; preds = %sw.epilog43, %sw.bb36, %sw.bb37
  br label %sw.epilog45

sw.epilog45:                                      ; preds = %sw.epilog44, %sw.epilog33
  %28 = load i32, ptr %a, align 4
  %29 = load i32, ptr %b, align 4
  %add46 = add nsw i32 %28, %29
  %30 = load i32, ptr %c, align 4
  %add47 = add nsw i32 %add46, %30
  ret i32 %add47
}

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


