; RUN: llc -mtriple=aarch64 < %s | FileCheck %s

; CHECK-LABEL: fn1_vector:
; CHECK:      adrp x[[BASE:[0-9]+]], .LCP
; CHECK-NEXT: ldr q[[NUM:[0-9]+]], [x[[BASE]],
; CHECK-NEXT: mul v0.16b, v0.16b, v[[NUM]].16b
; CHECK-NEXT: ret
define <16 x i8> @fn1_vector(<16 x i8> %arg) {
entry:
  %shl = shl <16 x i8> %arg, <i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7>
  %mul = mul <16 x i8> %shl, <i8 0, i8 1, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>
  ret <16 x i8> %mul
}

; CHECK-LABEL: fn2_vector:
; CHECK:      adrp x[[BASE:[0-9]+]], .LCP
; CHECK-NEXT: ldr q[[NUM:[0-9]+]], [x[[BASE]],
; CHECK-NEXT: mul v0.16b, v0.16b, v[[NUM]].16b
; CHECK-NEXT: ret
define <16 x i8> @fn2_vector(<16 x i8> %arg) {
entry:
  %mul = mul <16 x i8> %arg, <i8 0, i8 1, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>
  %shl = shl <16 x i8> %mul, <i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7>
  ret <16 x i8> %shl
}

; CHECK-LABEL: fn1_vector_undef:
; CHECK:      adrp x[[BASE:[0-9]+]], .LCP
; CHECK-NEXT: ldr q[[NUM:[0-9]+]], [x[[BASE]],
; CHECK-NEXT: mul v0.16b, v0.16b, v[[NUM]].16b
; CHECK-NEXT: ret
define <16 x i8> @fn1_vector_undef(<16 x i8> %arg) {
entry:
  %shl = shl <16 x i8> %arg, <i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7>
  %mul = mul <16 x i8> %shl, <i8 undef, i8 1, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>
  ret <16 x i8> %mul
}

; CHECK-LABEL: fn2_vector_undef:
; CHECK:      adrp x[[BASE:[0-9]+]], .LCP
; CHECK-NEXT: ldr q[[NUM:[0-9]+]], [x[[BASE]],
; CHECK-NEXT: mul v0.16b, v0.16b, v[[NUM]].16b
; CHECK-NEXT: ret
define <16 x i8> @fn2_vector_undef(<16 x i8> %arg) {
entry:
  %mul = mul <16 x i8> %arg, <i8 undef, i8 1, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>
  %shl = shl <16 x i8> %mul, <i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7>
  ret <16 x i8> %shl
}

; CHECK-LABEL: fn1_scalar:
; CHECK:      mov w[[REG:[0-9]+]], #1664
; CHECK-NEXT: mul w0, w0, w[[REG]]
; CHECK-NEXT: ret
define i32 @fn1_scalar(i32 %arg) {
entry:
  %shl = shl i32 %arg, 7
  %mul = mul i32 %shl, 13
  ret i32 %mul
}

; CHECK-LABEL: fn2_scalar:
; CHECK:      mov w[[REG:[0-9]+]], #1664
; CHECK-NEXT: mul w0, w0, w[[REG]]
; CHECK-NEXT: ret
define i32 @fn2_scalar(i32 %arg) {
entry:
  %mul = mul i32 %arg, 13
  %shl = shl i32 %mul, 7
  ret i32 %shl
}

; CHECK-LABEL: fn1_scalar_undef:
; CHECK:      mov w0
; CHECK-NEXT: ret
define i32 @fn1_scalar_undef(i32 %arg) {
entry:
  %shl = shl i32 %arg, 7
  %mul = mul i32 %shl, undef
  ret i32 %mul
}

; CHECK-LABEL: fn2_scalar_undef:
; CHECK:      mov w0
; CHECK-NEXT: ret
define i32 @fn2_scalar_undef(i32 %arg) {
entry:
  %mul = mul i32 %arg, undef
  %shl = shl i32 %mul, 7
  ret i32 %shl
}

; CHECK-LABEL: fn1_scalar_opaque:
; CHECK:      mov w[[REG:[0-9]+]], #13
; CHECK-NEXT: mul w[[REG]], w0, w[[REG]]
; CHECK-NEXT: lsl w0, w[[REG]], #7
; CHECK-NEXT: ret
define i32 @fn1_scalar_opaque(i32 %arg) {
entry:
  %bitcast = bitcast i32 13 to i32
  %shl = shl i32 %arg, 7
  %mul = mul i32 %shl, %bitcast
  ret i32 %mul
}

; CHECK-LABEL: fn2_scalar_opaque:
; CHECK:      mov w[[REG:[0-9]+]], #13
; CHECK-NEXT: mul w[[REG]], w0, w[[REG]]
; CHECK-NEXT: lsl w0, w[[REG]], #7
; CHECK-NEXT: ret
define i32 @fn2_scalar_opaque(i32 %arg) {
entry:
  %bitcast = bitcast i32 13 to i32
  %mul = mul i32 %arg, %bitcast
  %shl = shl i32 %mul, 7
  ret i32 %shl
}
