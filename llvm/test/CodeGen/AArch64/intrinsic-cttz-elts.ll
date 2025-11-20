; RUN: llc -mtriple=aarch64-linux-gnu < %s | FileCheck %s

; FIXED WIDTH

define i8 @ctz_v8i1(<8 x i1> %a) {
; CHECK-LABEL: .LCPI0_0:
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .byte 7
; CHECK-NEXT:   .byte 6
; CHECK-NEXT:   .byte 5
; CHECK-NEXT:   .byte 4
; CHECK-NEXT:   .byte 3
; CHECK-NEXT:   .byte 2
; CHECK-NEXT:   .byte 1
; CHECK-LABEL: ctz_v8i1:
; CHECK:       // %bb.0:
; CHECK-NEXT:    shl v0.8b, v0.8b, #7
; CHECK-NEXT:    adrp x8, .LCPI0_0
; CHECK-NEXT:    mov w9, #8 // =0x8
; CHECK-NEXT:    ldr d1, [x8, :lo12:.LCPI0_0]
; CHECK-NEXT:    cmlt v0.8b, v0.8b, #0
; CHECK-NEXT:    and v0.8b, v0.8b, v1.8b
; CHECK-NEXT:    umaxv b0, v0.8b
; CHECK-NEXT:    fmov w8, s0
; CHECK-NEXT:    sub w0, w9, w8
; CHECK-NEXT:    ret
  %res = call i8 @llvm.experimental.cttz.elts.i8.v8i1(<8 x i1> %a, i1 0)
  ret i8 %res
}

define i32 @ctz_v16i1(<16 x i1> %a) {
; CHECK-LABEL: .LCPI1_0:
; CHECK-NEXT:   .byte 16
; CHECK-NEXT:   .byte 15
; CHECK-NEXT:   .byte 14
; CHECK-NEXT:   .byte 13
; CHECK-NEXT:   .byte 12
; CHECK-NEXT:   .byte 11
; CHECK-NEXT:   .byte 10
; CHECK-NEXT:   .byte 9
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .byte 7
; CHECK-NEXT:   .byte 6
; CHECK-NEXT:   .byte 5
; CHECK-NEXT:   .byte 4
; CHECK-NEXT:   .byte 3
; CHECK-NEXT:   .byte 2
; CHECK-NEXT:   .byte 1
; CHECK-LABEL: ctz_v16i1:
; CHECK:       // %bb.0:
; CHECK-NEXT:    shl v0.16b, v0.16b, #7
; CHECK-NEXT:    adrp x8, .LCPI1_0
; CHECK-NEXT:    mov w9, #16 // =0x10
; CHECK-NEXT:    ldr q1, [x8, :lo12:.LCPI1_0]
; CHECK-NEXT:    cmlt v0.16b, v0.16b, #0
; CHECK-NEXT:    and v0.16b, v0.16b, v1.16b
; CHECK-NEXT:    umaxv b0, v0.16b
; CHECK-NEXT:    fmov w8, s0
; CHECK-NEXT:    sub w8, w9, w8
; CHECK-NEXT:    and w0, w8, #0xff
; CHECK-NEXT:    ret
  %res = call i32 @llvm.experimental.cttz.elts.i32.v16i1(<16 x i1> %a, i1 0)
  ret i32 %res
}

define i16 @ctz_v4i32(<4 x i32> %a) {
; CHECK-LABEL: .LCPI2_0:
; CHECK-NEXT:   .hword 4
; CHECK-NEXT:   .hword 3
; CHECK-NEXT:   .hword 2
; CHECK-NEXT:   .hword 1
; CHECK-LABEL: ctz_v4i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    cmtst v0.4s, v0.4s, v0.4s
; CHECK-NEXT:    adrp x8, .LCPI2_0
; CHECK-NEXT:    mov w9, #4 // =0x4
; CHECK-NEXT:    ldr d1, [x8, :lo12:.LCPI2_0]
; CHECK-NEXT:    xtn v0.4h, v0.4s
; CHECK-NEXT:    and v0.8b, v0.8b, v1.8b
; CHECK-NEXT:    umaxv h0, v0.4h
; CHECK-NEXT:    fmov w8, s0
; CHECK-NEXT:    sub w8, w9, w8
; CHECK-NEXT:    and w0, w8, #0xff
; CHECK-NEXT:    ret
  %res = call i16 @llvm.experimental.cttz.elts.i16.v4i32(<4 x i32> %a, i1 0)
  ret i16 %res
}

define i7 @ctz_i7_v8i1(<8 x i1> %a) {
; CHECK-LABEL: .LCPI3_0:
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .byte 7
; CHECK-NEXT:   .byte 6
; CHECK-NEXT:   .byte 5
; CHECK-NEXT:   .byte 4
; CHECK-NEXT:   .byte 3
; CHECK-NEXT:   .byte 2
; CHECK-NEXT:   .byte 1
; CHECK-LABEL: ctz_i7_v8i1:
; CHECK:       // %bb.0:
; CHECK-NEXT:    shl v0.8b, v0.8b, #7
; CHECK-NEXT:    adrp x8, .LCPI3_0
; CHECK-NEXT:    mov w9, #8 // =0x8
; CHECK-NEXT:    ldr d1, [x8, :lo12:.LCPI3_0]
; CHECK-NEXT:    cmlt v0.8b, v0.8b, #0
; CHECK-NEXT:    and v0.8b, v0.8b, v1.8b
; CHECK-NEXT:    umaxv b0, v0.8b
; CHECK-NEXT:    fmov w8, s0
; CHECK-NEXT:    sub w0, w9, w8
; CHECK-NEXT:    ret
  %res = call i7 @llvm.experimental.cttz.elts.i7.v8i1(<8 x i1> %a, i1 0)
  ret i7 %res
}

; ZERO IS POISON

define i8 @ctz_v8i1_poison(<8 x i1> %a) {
; CHECK-LABEL: .LCPI4_0:
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .byte 7
; CHECK-NEXT:   .byte 6
; CHECK-NEXT:   .byte 5
; CHECK-NEXT:   .byte 4
; CHECK-NEXT:   .byte 3
; CHECK-NEXT:   .byte 2
; CHECK-NEXT:   .byte 1
; CHECK-LABEL: ctz_v8i1_poison:
; CHECK:       // %bb.0:
; CHECK-NEXT:    shl v0.8b, v0.8b, #7
; CHECK-NEXT:    adrp x8, .LCPI4_0
; CHECK-NEXT:    mov w9, #8 // =0x8
; CHECK-NEXT:    ldr d1, [x8, :lo12:.LCPI4_0]
; CHECK-NEXT:    cmlt v0.8b, v0.8b, #0
; CHECK-NEXT:    and v0.8b, v0.8b, v1.8b
; CHECK-NEXT:    umaxv b0, v0.8b
; CHECK-NEXT:    fmov w8, s0
; CHECK-NEXT:    sub w0, w9, w8
; CHECK-NEXT:    ret
  %res = call i8 @llvm.experimental.cttz.elts.i8.v8i1(<8 x i1> %a, i1 1)
  ret i8 %res
}

declare i8 @llvm.experimental.cttz.elts.i8.v8i1(<8 x i1>, i1)
declare i7 @llvm.experimental.cttz.elts.i7.v8i1(<8 x i1>, i1)
declare i32 @llvm.experimental.cttz.elts.i32.v16i1(<16 x i1>, i1)
declare i16 @llvm.experimental.cttz.elts.i16.v4i32(<4 x i32>, i1)
