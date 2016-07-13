; RUN: llc %s -O0 -march=sparc -mcpu=ut699 -o - | FileCheck %s -check-prefix=NO_ROUND_FUNC
; RUN: llc %s -O0 -march=sparc -mcpu=leon3 -mattr=+prvntroundchange -o - | FileCheck %s -check-prefix=NO_ROUND_FUNC

; RUN: llc %s -O0 -march=sparc -mcpu=ut699 -mattr=-prvntroundchange -o - | FileCheck %s -check-prefix=ROUND_FUNC
; RUN: llc %s -O0 -march=sparc -mcpu=leon3  -o - | FileCheck %s -check-prefix=ROUND_FUNC


; NO_ROUND_FUNC-LABEL: test_round_change
; NO_ROUND_FUNC-NOT: fesetround

; ROUND_FUNC-LABEL: test_round_change
; ROUND_FUNC: fesetround

; ModuleID = '<stdin>'
target datalayout = "E-m:e-p:32:32-i64:64-f128:64-n32-S64"
target triple = "sparc-unknown--eabi"

@.str = private unnamed_addr constant [17 x i8] c"-((-a)*b) != a*b\00", align 1
@.str.1 = private unnamed_addr constant [7 x i8] c"test.c\00", align 1
@__PRETTY_FUNCTION__.mult = private unnamed_addr constant [12 x i8] c"void mult()\00", align 1

; Function Attrs: nounwind
define void @test_round_change() #0 {
entry:
  %a = alloca double, align 8
  %b = alloca double, align 8
  %x = alloca float, align 4
  store double 1.100000e+00, double* %a, align 8
  store double 1.010000e+01, double* %b, align 8
  store float 0x400921FA00000000, float* %x, align 4
  %call = call i32 @fesetround(i32 2048) #2
  %0 = load double, double* %a, align 8
  %sub = fsub double -0.000000e+00, %0
  %1 = load double, double* %b, align 8
  %mul = fmul double %sub, %1
  %sub1 = fsub double -0.000000e+00, %mul
  %2 = load double, double* %a, align 8
  %3 = load double, double* %b, align 8
  %mul2 = fmul double %2, %3
  %cmp = fcmp une double %sub1, %mul2
  br i1 %cmp, label %cond.true, label %cond.false

cond.true:                                        ; preds = %entry
  br label %cond.end

cond.false:                                       ; preds = %entry
  call void @__assert_fail(i8* getelementptr inbounds ([17 x i8], [17 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.1, i32 0, i32 0), i32 10, i8* getelementptr inbounds ([12 x i8], [12 x i8]* @__PRETTY_FUNCTION__.mult, i32 0, i32 0)) #3
  unreachable
                                                  ; No predecessors!
  br label %cond.end

cond.end:                                         ; preds = %4, %cond.true
  ret void
}

; Function Attrs: nounwind
declare i32 @fesetround(i32) #0

; Function Attrs: noreturn nounwind
declare void @__assert_fail(i8*, i8*, i32, i8*) #1

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noreturn nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
attributes #3 = { noreturn nounwind }