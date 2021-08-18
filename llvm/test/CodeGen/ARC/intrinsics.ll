; RUN: llc -march=arc < %s | FileCheck %s

target triple = "arc"

declare i32 @llvm.ctlz.i32(i32, i1)
declare i32 @llvm.cttz.i32(i32, i1)

; CHECK-LABEL: clz32:
; CHECK:       fls.f   %r0, %r0
; CHECK-NEXT:  mov.eq  %r0, 32
; CHECK-NEXT:  rsub.ne %r0, %r0, 31
define i32 @clz32(i32 %x) {
  %a = call i32 @llvm.ctlz.i32(i32 %x, i1 false)
  ret i32 %a
}

; CHECK-LABEL: ctz32:
; CHECK:       ffs.f   %r0, %r0
; CHECK-NEXT:  mov.eq  %r0, 32
define i32 @ctz32(i32 %x) {
  %a = call i32 @llvm.cttz.i32(i32 %x, i1 false)
  ret i32 %a
}
