; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s -check-prefix=X64
; X64:      mov
; X64-NEXT: movb %ah, (%rsi)
; X64:      mov
; X64-NEXT: movb %ah, (%rsi)
; X64:      mov
; X64-NEXT: movb %ah, (%rsi)
; X64-NOT:      mov

; RUN: llc < %s -mtriple=x86_64-win32 | FileCheck %s -check-prefix=W64
; W64-NOT:      mov
; W64:      movb %ch, (%rdx)
; W64-NOT:      mov
; W64:      movb %ch, (%rdx)
; W64-NOT:      mov
; W64:      movb %ch, (%rdx)
; W64-NOT:      mov

; RUN: llc < %s -march=x86 | FileCheck %s -check-prefix=X32
; X32-NOT:      mov
; X32:      movb %ah, (%e
; X32-NOT:      mov
; X32:      movb %ah, (%e
; X32-NOT:      mov
; X32:      movb %ah, (%e
; X32-NOT:      mov

; Use h-register extract and store.

define void @foo16(i16 inreg %p, i8* inreg %z) nounwind {
  %q = lshr i16 %p, 8
  %t = trunc i16 %q to i8
  store i8 %t, i8* %z
  ret void
}
define void @foo32(i32 inreg %p, i8* inreg %z) nounwind {
  %q = lshr i32 %p, 8
  %t = trunc i32 %q to i8
  store i8 %t, i8* %z
  ret void
}
define void @foo64(i64 inreg %p, i8* inreg %z) nounwind {
  %q = lshr i64 %p, 8
  %t = trunc i64 %q to i8
  store i8 %t, i8* %z
  ret void
}
