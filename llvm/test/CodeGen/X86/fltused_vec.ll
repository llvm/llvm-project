; The purpose of this test to verify that the fltused symbol is
; not emitted when purely vector floating point operations are used on Windows.

; RUN: llc < %s -mtriple i686-pc-win32 | FileCheck %s
; RUN: llc < %s -mtriple x86_64-pc-win32 | FileCheck %s

@foo = external dso_local global [4 x float], align 16

; Function Attrs: noinline nounwind optnone sspstrong uwtable
define dso_local <4 x float> @func() #0 {
entry:
  %__p.addr.i = alloca ptr, align 8
  %vector1 = alloca <4 x float>, align 16
  store ptr @foo, ptr %__p.addr.i, align 8
  %0 = load ptr, ptr %__p.addr.i, align 8
  %1 = load <4 x float>, ptr %0, align 16
  store <4 x float> %1, ptr %vector1, align 16
  %2 = load <4 x float>, ptr %vector1, align 16
  ret <4 x float> %2
}

define <4 x float> @mul_vectors(<4 x float> %a, <4 x float> %b) {
entry:
  %result = fmul <4 x float> %a, %b
  ret <4 x float> %result
}

; _fltused is determined at a module level
; CHECK-NOT: .globl {{_?}}_fltused
