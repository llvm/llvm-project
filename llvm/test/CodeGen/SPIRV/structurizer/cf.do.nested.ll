; RUN: %if spirv-tools %{ llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | spirv-as --preserve-numeric-ids - -o - | spirv-val %}
;
; Source HLSL:
;
; int process() {
;   int val=0, i=0, j=0, k=0;
;
;   do {
;     val = val + i;
;
;     do {
;       do {
;         ++k;
;       } while (k < 30);
;
;       ++j;
;     } while (j < 20);
;
;     ++i;
;
;   } while (i < 10);
;
;   return val + i + j + k;
; }
;
; [numthreads(1, 1, 1)]
; void main() {
;   process();
; }

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

; Function Attrs: convergent noinline nounwind optnone
define spir_func noundef i32 @_Z7processv() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %val = alloca i32, align 4
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %k = alloca i32, align 4
  store i32 0, ptr %val, align 4
  store i32 0, ptr %i, align 4
  store i32 0, ptr %j, align 4
  store i32 0, ptr %k, align 4
  br label %do.body

do.body:                                          ; preds = %do.cond8, %entry
  %1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  %2 = load i32, ptr %val, align 4
  %3 = load i32, ptr %i, align 4
  %add = add nsw i32 %2, %3
  store i32 %add, ptr %val, align 4
  br label %do.body1

do.body1:                                         ; preds = %do.cond4, %do.body
  %4 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %1) ]
  br label %do.body2

do.body2:                                         ; preds = %do.cond, %do.body1
  %5 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %4) ]
  %6 = load i32, ptr %k, align 4
  %inc = add nsw i32 %6, 1
  store i32 %inc, ptr %k, align 4
  br label %do.cond

do.cond:                                          ; preds = %do.body2
  %7 = load i32, ptr %k, align 4
  %cmp = icmp slt i32 %7, 30
  br i1 %cmp, label %do.body2, label %do.end

do.end:                                           ; preds = %do.cond
  %8 = load i32, ptr %j, align 4
  %inc3 = add nsw i32 %8, 1
  store i32 %inc3, ptr %j, align 4
  br label %do.cond4

do.cond4:                                         ; preds = %do.end
  %9 = load i32, ptr %j, align 4
  %cmp5 = icmp slt i32 %9, 20
  br i1 %cmp5, label %do.body1, label %do.end6

do.end6:                                          ; preds = %do.cond4
  %10 = load i32, ptr %i, align 4
  %inc7 = add nsw i32 %10, 1
  store i32 %inc7, ptr %i, align 4
  br label %do.cond8

do.cond8:                                         ; preds = %do.end6
  %11 = load i32, ptr %i, align 4
  %cmp9 = icmp slt i32 %11, 10
  br i1 %cmp9, label %do.body, label %do.end10

do.end10:                                         ; preds = %do.cond8
  %12 = load i32, ptr %val, align 4
  %13 = load i32, ptr %i, align 4
  %add11 = add nsw i32 %12, %13
  %14 = load i32, ptr %j, align 4
  %add12 = add nsw i32 %add11, %14
  %15 = load i32, ptr %k, align 4
  %add13 = add nsw i32 %add12, %15
  ret i32 %add13
}

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare token @llvm.experimental.convergence.entry() #1

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare token @llvm.experimental.convergence.loop() #1

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

