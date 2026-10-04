; RUN: llc < %s -mtriple=s390x-ibm-zos | FileCheck %s
;
; @@ALCAXP moves the frame header (save area at SP + 2048) down with the stack
; pointer. A stack restore must copy the current header back to the restored
; stack pointer before the frame is used again, from the highest word down.

declare void @use(ptr)

; CHECK-LABEL: f DS 0H
; CHECK:      lgr 11,4
; CHECK:      basr 7,6
; CHECK:      la 1,2240(4)
; CHECK:      basr 7,6
; CHECK:      lg 0,2168(4)
; CHECK-NEXT: stg 0,2168(11)
; CHECK-NEXT: lg 0,2160(4)
; CHECK-NEXT: stg 0,2160(11)
; CHECK:      lg 0,2056(4)
; CHECK-NEXT: stg 0,2056(11)
; CHECK-NEXT: lg 0,2048(4)
; CHECK-NEXT: stg 0,2048(11)
; CHECK-NEXT: lgr 4,11
; CHECK:      basr 7,6
; CHECK:      lmg 4,11,2048(4)
define void @f(i64 %n) {
  %sp = call ptr @llvm.stacksave.p0()
  %p = alloca i8, i64 %n, align 8
  call void @use(ptr %p)
  call void @llvm.stackrestore.p0(ptr %sp)
  call void @use(ptr null)
  ret void
}
