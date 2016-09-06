; RUN: llc < %s -mtriple=i386-apple-darwin9 -mattr=+sse2  | FileCheck %s --check-prefix=X32
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mattr=+sse2 | FileCheck %s --check-prefix=X64
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mattr=-sse3 | FileCheck %s --check-prefix=X64_BAD

; Sibcall optimization of expanded libcalls.
; rdar://8707777

define double @foo(double %a) nounwind readonly ssp {
entry:
; X32-LABEL: foo:
; X32: jmp _sin

; X64-LABEL: foo:
; X64: jmp _sin
  %0 = tail call double @sin(double %a) nounwind readonly
  ret double %0
}

define float @bar(float %a) nounwind readonly ssp {
; X32-LABEL: bar:
; X32: jmp _sinf

; X64-LABEL: bar:
; X64: jmp _sinf
entry:
  %0 = tail call float @sinf(float %a) nounwind readonly
  ret float %0
}


declare float @sinf(float) nounwind readonly

declare double @sin(double) nounwind readonly

; rdar://10930395
%0 = type opaque

@"\01L_OBJC_SELECTOR_REFERENCES_2" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"

define hidden { double, double } @foo2(%0* %self, i8* nocapture %_cmd) uwtable optsize ssp {
; X64_BAD: foo
; X64_BAD: call
; X64_BAD: call
; X64_BAD: call
  %1 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_2", align 8, !invariant.load !0
  %2 = bitcast %0* %self to i8*
  %3 = tail call { double, double } bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to { double, double } (i8*, i8*)*)(i8* %2, i8* %1) optsize
  %4 = extractvalue { double, double } %3, 0
  %5 = extractvalue { double, double } %3, 1
  %6 = tail call double @floor(double %4) optsize
  %7 = tail call double @floor(double %5) optsize
  %insert.i.i = insertvalue { double, double } undef, double %6, 0
  %insert5.i.i = insertvalue { double, double } %insert.i.i, double %7, 1
  ret { double, double } %insert5.i.i
}

declare i8* @objc_msgSend(i8*, i8*, ...)

declare double @floor(double) optsize

!0 = !{}
