; RUN: %if spirv-tools %{ llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | spirv-as --preserve-numeric-ids - -o - | spirv-val %}
;
; Source HLSL:
;
; int process() {
;   int val = 0;
;
;   for (int i = 0; i < 10; ++i) {
;     if (i < 5) {
;       break;
;     }
;     val = i;
;     {
;       break;
;     }
;     break;       // No SPIR-V should be emitted for this statement.
;     val++;       // No SPIR-V should be emitted for this statement.
;     while(true); // No SPIR-V should be emitted for this statement.
;   }
;
;   ////////////////////////////////////////////////////////////////////////////////
;   // Nested for loops with break statements                                     //
;   // Each break statement should branch to the corresponding loop's break block //
;   ////////////////////////////////////////////////////////////////////////////////
;   for (int j = 0; j < 10; ++j) {
;     val = j+5;
;     for ( ; val < 20; ++val) {
;       int k = val + j;
;       {
;         {
;           break;
;         }
;       }
;       k++; // No SPIR-V should be emitted for this statement.
;     }
;
;     // FIXME: val--; generates add nsw i32 ..., -1
;     // This is compiled into a OpIAdd %val, MAX_INT.
;     val -= 1;
;     break;
;     break;        // No SPIR-V should be emitted for this statement.
;     val = val*10; // No SPIR-V should be emitted for this statement.
;   }
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
define spir_func noundef i32 @_Z7processv() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %val = alloca i32, align 4
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %k = alloca i32, align 4
  store i32 0, ptr %val, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  %2 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %2, 10
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %3 = load i32, ptr %i, align 4
  %cmp1 = icmp slt i32 %3, 5
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  br label %for.end

if.end:                                           ; preds = %for.body
  %4 = load i32, ptr %i, align 4
  store i32 %4, ptr %val, align 4
  br label %for.end

for.inc:                                          ; No predecessors!
  %5 = load i32, ptr %i, align 4
  %inc = add nsw i32 %5, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond

for.end:                                          ; preds = %if.end, %if.then, %for.cond
  store i32 0, ptr %j, align 4
  br label %for.cond2

for.cond2:                                        ; preds = %for.inc12, %for.end
  %6 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  %7 = load i32, ptr %j, align 4
  %cmp3 = icmp slt i32 %7, 10
  br i1 %cmp3, label %for.body4, label %for.end14

for.body4:                                        ; preds = %for.cond2
  %8 = load i32, ptr %j, align 4
  %add = add nsw i32 %8, 5
  store i32 %add, ptr %val, align 4
  br label %for.cond5

for.cond5:                                        ; preds = %for.inc9, %for.body4
  %9 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %6) ]
  %10 = load i32, ptr %val, align 4
  %cmp6 = icmp slt i32 %10, 20
  br i1 %cmp6, label %for.body7, label %for.end11

for.body7:                                        ; preds = %for.cond5
  %11 = load i32, ptr %val, align 4
  %12 = load i32, ptr %j, align 4
  %add8 = add nsw i32 %11, %12
  store i32 %add8, ptr %k, align 4
  br label %for.end11

for.inc9:                                         ; No predecessors!
  %13 = load i32, ptr %val, align 4
  %inc10 = add nsw i32 %13, 1
  store i32 %inc10, ptr %val, align 4
  br label %for.cond5

for.end11:                                        ; preds = %for.body7, %for.cond5
  %14 = load i32, ptr %val, align 4
  %sub = sub nsw i32 %14, 1
  store i32 %sub, ptr %val, align 4
  br label %for.end14

for.inc12:                                        ; No predecessors!
  %15 = load i32, ptr %j, align 4
  %inc13 = add nsw i32 %15, 1
  store i32 %inc13, ptr %j, align 4
  br label %for.cond2

for.end14:                                        ; preds = %for.end11, %for.cond2
  %16 = load i32, ptr %val, align 4
  ret i32 %16
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

