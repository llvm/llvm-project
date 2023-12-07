; RUN: llc < %s -mtriple=thumb-apple-darwin | FileCheck %s
; rdar://7268481

define void @t(ptr %a, ...) nounwind {
; CHECK-LABEL:      t:
; CHECK:      pop {[[POP_REG:r[0-3]]]}
; CHECK-NEXT: add sp, #12
; CHECK-NEXT: bx [[POP_REG]]
entry:
  %a.addr = alloca i8, i32 4
  call void @llvm.va_start(ptr %a.addr)
  ret void
}

declare void @llvm.va_start(ptr) nounwind
