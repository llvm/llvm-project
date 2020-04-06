; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define i64 @func1(i64 %p) {
; CHECK-LABEL: func1:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ldz %s0, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = tail call i64 @llvm.ctlz.i64(i64 %p, i1 true)
  ret i64 %r
}

declare i64 @llvm.ctlz.i64(i64, i1)

define i32 @func2(i32 %p) {
; CHECK-LABEL: func2:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    # kill: def $sw0 killed $sw0 def $sx0
; CHECK-NEXT:    sll %s0, %s0, 32
; CHECK-NEXT:    ldz %s0, %s0
; CHECK-NEXT:    # kill: def $sw0 killed $sw0 killed $sx0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = tail call i32 @llvm.ctlz.i32(i32 %p, i1 true)
  ret i32 %r
}

declare i32 @llvm.ctlz.i32(i32, i1)

define i16 @func3(i16 %p) {
; CHECK-LABEL: func3:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    and %s0, %s0, (48)0
; CHECK-NEXT:    sll %s0, %s0, 32
; CHECK-NEXT:    ldz %s0, %s0
; CHECK-NEXT:    adds.w.sx %s0, -16, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = tail call i16 @llvm.ctlz.i16(i16 %p, i1 true)
  ret i16 %r
}

declare i16 @llvm.ctlz.i16(i16, i1)

define i8 @func4(i8 %p) {
; CHECK-LABEL: func4:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    and %s0, %s0, (56)0
; CHECK-NEXT:    sll %s0, %s0, 32
; CHECK-NEXT:    ldz %s0, %s0
; CHECK-NEXT:    adds.w.sx %s0, -24, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = tail call i8 @llvm.ctlz.i8(i8 %p, i1 true)
  ret i8 %r
}

declare i8 @llvm.ctlz.i8(i8, i1)
