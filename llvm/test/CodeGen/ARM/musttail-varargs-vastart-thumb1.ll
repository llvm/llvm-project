; RUN: llc -verify-machineinstrs -mtriple=thumbv8m.base-none-eabi %s -o - | FileCheck %s

;; Thumb1 releases the save area in its own epilogue code, before the tail
;; branch too.

declare void @llvm.va_start(ptr)
declare void @llvm.va_end(ptr)

define void @fwd_vastart(ptr %pfn, ...) nounwind {
; CHECK-LABEL: fwd_vastart:
; CHECK:       sub sp, #12
; CHECK:       stm r{{[0-7]}}!, {r1, r2, r3}
; CHECK:       pop {r4}
; CHECK-NEXT:  add sp, #16
; CHECK-NEXT:  bx r0
entry:
  %ap = alloca ptr, align 4
  call void @llvm.va_start(ptr %ap)
  call void @llvm.va_end(ptr %ap)
  musttail call void (ptr, ...) %pfn(ptr %pfn, ...)
  ret void
}
