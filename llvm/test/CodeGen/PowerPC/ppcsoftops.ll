; RUN: llc -verify-machineinstrs  -mtriple=powerpc-unknown-linux-gnu -O0 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs  -mtriple=powerpc64-unknown-linux-gnu -O0 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs  -mtriple=powerpc64le-unknown-linux-gnu -O0 < %s | FileCheck %s

; Testing operations in soft-float mode
define double @foo() #0 {
entry:
  %a = alloca double, align 8
  %b = alloca double, align 8
  %0 = load double, ptr %a, align 8
  %1 = load double, ptr %b, align 8
  %add = fadd double %0, %1
  ret double %add

  ; CHECK-LABEL:      __adddf3
}

define double @foo1() #0 {
entry:
  %a = alloca double, align 8
  %b = alloca double, align 8
  %0 = load double, ptr %a, align 8
  %1 = load double, ptr %b, align 8
  %mul = fmul double %0, %1
  ret double %mul

  ; CHECK-LABEL:      __muldf3
}

define double @foo2() #0 {
entry:
  %a = alloca double, align 8
  %b = alloca double, align 8
  %0 = load double, ptr %a, align 8
  %1 = load double, ptr %b, align 8
  %sub = fsub double %0, %1
  ret double %sub

  ; CHECK-LABEL:      __subdf3
}

define double @foo3() #0 {
entry:
  %a = alloca double, align 8
  %b = alloca double, align 8
  %0 = load double, ptr %a, align 8
  %1 = load double, ptr %b, align 8
  %div = fdiv double %0, %1
  ret double %div

  ; CHECK-LABEL:      __divdf3
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local zeroext i32 @func(double noundef %0, double noundef %1) #0 {
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  store double %0, ptr %3, align 8
  store double %1, ptr %4, align 8
  %5 = load double, ptr %3, align 8
  %6 = load double, ptr %4, align 8
  %7 = fneg double %6
  %8 = call double @llvm.fmuladd.f64(double %7, double 0x41F0000000000000, double %5)
  %9 = fptoui double %8 to i32
  ret i32 %9

  ; CHECK-LABEL:      __muldf3
  ; CHECK-LABEL:      __adddf3
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #1

attributes #0 = {"use-soft-float"="true" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
