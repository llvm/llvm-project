; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define i64 @func1(i64 %p) {
; CHECK-LABEL: func1:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    brv %s0, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = tail call i64 @llvm.bitreverse.i64(i64 %p)
  ret i64 %r
}

declare i64 @llvm.bitreverse.i64(i64)

define i32 @func2(i32 %p) {
; CHECK-LABEL: func2:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    # kill: def $sw0 killed $sw0 def $sx0
; CHECK-NEXT:    brv %s0, %s0
; CHECK-NEXT:    srl %s0, %s0, 32
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = tail call i32 @llvm.bitreverse.i32(i32 %p)
  ret i32 %r
}

declare i32 @llvm.bitreverse.i32(i32)

define signext i16 @func3(i16 signext %p) {
; CHECK-LABEL: func3:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    # kill: def $sw0 killed $sw0 def $sx0
; CHECK-NEXT:    brv %s0, %s0
; CHECK-NEXT:    sra.l %s0, %s0, 48
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = tail call i16 @llvm.bitreverse.i16(i16 %p)
  ret i16 %r
}

declare i16 @llvm.bitreverse.i16(i16)

define signext i8 @func4(i8 signext %p) {
; CHECK-LABEL: func4:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    # kill: def $sw0 killed $sw0 def $sx0
; CHECK-NEXT:    brv %s0, %s0
; CHECK-NEXT:    sra.l %s0, %s0, 56
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = tail call i8 @llvm.bitreverse.i8(i8 %p)
  ret i8 %r
}

declare i8 @llvm.bitreverse.i8(i8)

define i64 @func5(i64 %p) {
; CHECK-LABEL: func5:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    brv %s0, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = tail call i64 @llvm.bitreverse.i64(i64 %p)
  ret i64 %r
}

define i32 @func6(i32 %p) {
; CHECK-LABEL: func6:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    # kill: def $sw0 killed $sw0 def $sx0
; CHECK-NEXT:    brv %s0, %s0
; CHECK-NEXT:    srl %s0, %s0, 32
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = tail call i32 @llvm.bitreverse.i32(i32 %p)
  ret i32 %r
}

define zeroext i16 @func7(i16 zeroext %p) {
; CHECK-LABEL: func7:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    # kill: def $sw0 killed $sw0 def $sx0
; CHECK-NEXT:    brv %s0, %s0
; CHECK-NEXT:    srl %s0, %s0, 48
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = tail call i16 @llvm.bitreverse.i16(i16 %p)
  ret i16 %r
}

define zeroext i8 @func8(i8 zeroext %p) {
; CHECK-LABEL: func8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    # kill: def $sw0 killed $sw0 def $sx0
; CHECK-NEXT:    brv %s0, %s0
; CHECK-NEXT:    srl %s0, %s0, 56
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = tail call i8 @llvm.bitreverse.i8(i8 %p)
  ret i8 %r
}

