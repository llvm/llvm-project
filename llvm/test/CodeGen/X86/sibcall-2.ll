; RUN: llc -verify-machineinstrs < %s -mtriple=i386-apple-darwin   -frame-pointer=all | FileCheck %s -check-prefix=32
; RUN: llc -verify-machineinstrs < %s -mtriple=x86_64-apple-darwin -frame-pointer=all | FileCheck %s -check-prefix=64

; Tail call should not use ebp / rbp after it's popped. Use esp / rsp.

define void @t1(i8* nocapture %value) nounwind {
entry:
; 32-LABEL: t1:
; 32: jmpl *4(%esp)

; 64-LABEL: t1:
; 64: jmpq *%rdi
  %0 = bitcast i8* %value to void ()*
  tail call void %0() nounwind
  ret void
}

define void @t2(i32 %a, i8* nocapture %value) nounwind {
entry:
; 32-LABEL: t2:
; 32: jmpl *8(%esp)

; 64-LABEL: t2:
; 64: jmpq *%rsi
  %0 = bitcast i8* %value to void ()*
  tail call void %0() nounwind
  ret void
}

define void @t3(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f, i8* nocapture %value) nounwind {
entry:
; 32-LABEL: t3:
; 32: jmpl *28(%esp)

; 64-LABEL: t3:
; 64: jmpq *8(%rsp)
  %0 = bitcast i8* %value to void ()*
  tail call void %0() nounwind
  ret void
}

define void @t4(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f, i32 %g, i8* nocapture %value) nounwind {
entry:
; 32-LABEL: t4:
; 32: jmpl *32(%esp)

; 64-LABEL: t4:
; 64: jmpq *16(%rsp)
  %0 = bitcast i8* %value to void ()*
  tail call void %0() nounwind
  ret void
}
