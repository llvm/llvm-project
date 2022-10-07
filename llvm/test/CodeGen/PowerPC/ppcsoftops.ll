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

attributes #0 = {"use-soft-float"="true" }
