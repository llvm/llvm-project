; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | spirv-sim --function=_Z7processv --wave=1 --expects=10
; RUN: %if spirv-tools %{ llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | spirv-as --preserve-numeric-ids - -o - | spirv-val %}
;
; Source HLSL:
;
; int foo() { return true; }
;
; int process() {
;   int val = 0;
;   int i = 0;
;
;   do {
;     ++i;
;     if (i > 5) {
;       {
;         {
;           continue;
;         }
;       }
;       val = i;     // No SPIR-V should be emitted for this statement.
;       while(true); // No SPIR-V should be emitted for this statement.
;     }
;     val = i;
;     continue;
;     val = val * 2; // No SPIR-V should be emitted for this statement.
;     continue;      // No SPIR-V should be emitted for this statement.
;
;   } while (i < 10);
;
;   //////////////////////////////////////////////////////////////////////////////////////
;   // Nested do-while loops with continue statements                                   //
;   // Each continue statement should branch to the corresponding loop's continue block //
;   //////////////////////////////////////////////////////////////////////////////////////
;
;   do {
;     ++i;
;     do {
;       ++val;
;       continue;
;     } while (i < 10);
;
;     --i;
;     continue;
;     continue;  // No SPIR-V should be emitted for this statement.
;
;   } while(val < 10);
;
;   return val;
; }
;
; [numthreads(1, 1, 1)]
; void main() {
;   process();
; }

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan1.3-compute"

; Function Attrs: convergent noinline nounwind optnone
define spir_func noundef i32 @_Z3foov() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  ret i32 1
}

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare token @llvm.experimental.convergence.entry() #1

; Function Attrs: convergent noinline nounwind optnone
define spir_func noundef i32 @_Z7processv() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %val = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 0, ptr %val, align 4
  store i32 0, ptr %i, align 4
  br label %do.body

do.body:                                          ; preds = %do.cond, %entry
  %1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  %2 = load i32, ptr %i, align 4
  %inc = add nsw i32 %2, 1
  store i32 %inc, ptr %i, align 4
  %3 = load i32, ptr %i, align 4
  %cmp = icmp sgt i32 %3, 5
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %do.body
  br label %do.cond

if.end:                                           ; preds = %do.body
  %4 = load i32, ptr %i, align 4
  store i32 %4, ptr %val, align 4
  br label %do.cond

do.cond:                                          ; preds = %if.end, %if.then
  %5 = load i32, ptr %i, align 4
  %cmp1 = icmp slt i32 %5, 10
  br i1 %cmp1, label %do.body, label %do.end

do.end:                                           ; preds = %do.cond
  br label %do.body2

do.body2:                                         ; preds = %do.cond9, %do.end
  %6 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  %7 = load i32, ptr %i, align 4
  %inc3 = add nsw i32 %7, 1
  store i32 %inc3, ptr %i, align 4
  br label %do.body4

do.body4:                                         ; preds = %do.cond6, %do.body2
  %8 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %6) ]
  %9 = load i32, ptr %val, align 4
  %inc5 = add nsw i32 %9, 1
  store i32 %inc5, ptr %val, align 4
  br label %do.cond6

do.cond6:                                         ; preds = %do.body4
  %10 = load i32, ptr %i, align 4
  %cmp7 = icmp slt i32 %10, 10
  br i1 %cmp7, label %do.body4, label %do.end8

do.end8:                                          ; preds = %do.cond6
  %11 = load i32, ptr %i, align 4
  %dec = add nsw i32 %11, -1
  store i32 %dec, ptr %i, align 4
  br label %do.cond9

do.cond9:                                         ; preds = %do.end8
  %12 = load i32, ptr %val, align 4
  %cmp10 = icmp slt i32 %12, 10
  br i1 %cmp10, label %do.body2, label %do.end11

do.end11:                                         ; preds = %do.cond9
  %13 = load i32, ptr %val, align 4
  ret i32 %13
}

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
