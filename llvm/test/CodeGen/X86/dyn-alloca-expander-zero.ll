; RUN: llc < %s -mtriple=i686-pc-win32 | FileCheck %s --check-prefix=X86
; RUN: llc < %s -mtriple=x86_64-pc-windows-msvc | FileCheck %s --check-prefix=X64

; A zero-sized dynamic alloca (materialized as the MOV32r0 idiom, widened via
; SUBREG_TO_REG on 64-bit) should be folded and elided, not probed.
define void @zero_const(i32 %n) {
; X86-LABEL: zero_const:
; X86-NOT: __chkstk
; X64-LABEL: zero_const:
; X64-NOT: __chkstk
entry:
  br label %b
b:
  %p = alloca i8, i32 0
  call void @use(ptr %p)
  ret void
}

; A dynamic size that folds to zero reaches the pass the same way.
define void @zero_folded(i32 %n) {
; X86-LABEL: zero_folded:
; X86-NOT: __chkstk
; X64-LABEL: zero_folded:
; X64-NOT: __chkstk
entry:
  %z = sub i32 %n, %n
  %p = alloca i8, i32 %z
  call void @use(ptr %p)
  ret void
}

declare void @use(ptr)
