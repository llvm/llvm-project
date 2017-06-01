; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 -mattr=+vsx < %s | FileCheck %s --implicit-check-not lxsiwzx

declare void @bar(double)

define void @foo1(i8* %p) {
entry:
  %0 = load i8, i8* %p, align 1
  %conv = uitofp i8 %0 to double
  call void @bar(double %conv)
  ret void

; CHECK-LABEL: @foo1
; CHECK:     mtvsrwz
}

define void @foo2(i16* %p) {
entry:
  %0 = load i16, i16* %p, align 2
  %conv = uitofp i16 %0 to double
  call void @bar(double %conv)
  ret void

; CHECK-LABEL: @foo2
; CHECK:       mtvsrwz
}

