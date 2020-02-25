; RUN: llc -mtriple=aarch64--linux-gnu -mattr=+sve --asm-verbose=false < %s |FileCheck %s

declare i32 @llvm.vscale.i32()
declare i64 @llvm.vscale.i64()

; Fold (add (vscale * C0), (vscale * C1)) to (vscale * (C0 + C1)).
define i64 @combine_add_vscale_i64() nounwind {
; CHECK-LABEL: combine_add_vscale_i64:
; CHECK-NOT:   add
; CHECK-NEXT:  cntd  x0
; CHECK-NEXT:  ret
 %vscale = call i64 @llvm.vscale.i64()
 %add = add i64 %vscale, %vscale
 ret i64 %add
}

define i32 @combine_add_vscale_i32() nounwind {
; CHECK-LABEL: combine_add_vscale_i32:
; CHECK-NOT:   add
; CHECK-NEXT:  cntd  x0
; CHECK-NEXT:  ret
 %vscale = call i32 @llvm.vscale.i32()
 %add = add i32 %vscale, %vscale
 ret i32 %add
}

; Fold (mul (vscale * C0), C1) to (vscale * (C0 * C1)).
; In this test, C0 = 1, C1 = 32.
define i64 @combine_mul_vscale_i64() nounwind {
; CHECK-LABEL: combine_mul_vscale_i64:
; CHECK-NOT:   mul
; CHECK-NEXT:  rdvl  x0, #2
; CHECK-NEXT:  ret
 %vscale = call i64 @llvm.vscale.i64()
 %mul = mul i64 %vscale, 32
 ret i64 %mul
}

define i32 @combine_mul_vscale_i32() nounwind {
; CHECK-LABEL: combine_mul_vscale_i32:
; CHECK-NOT:   mul
; CHECK-NEXT:  rdvl  x0, #3
; CHECK-NEXT:  ret
 %vscale = call i32 @llvm.vscale.i32()
 %mul = mul i32 %vscale, 48
 ret i32 %mul
}

; Canonicalize (sub X, (vscale * C)) to (add X,  (vscale * -C))
define i64 @combine_sub_vscale_i64(i64 %in) nounwind {
; CHECK-LABEL: combine_sub_vscale_i64:
; CHECK-NOT:   sub
; CHECK-NEXT:  rdvl  x8, #-1
; CHECK-NEXT:  asr   x8, x8, #4
; CHECK-NEXT:  add   x0, x0, x8
; CHECK-NEXT:  ret
 %vscale = call i64 @llvm.vscale.i64()
 %sub = sub i64 %in,  %vscale
 ret i64 %sub
}

define i32 @combine_sub_vscale_i32(i32 %in) nounwind {
; CHECK-LABEL: combine_sub_vscale_i32:
; CHECK-NOT:   sub
; CHECK-NEXT:  rdvl  x8, #-1
; CHECK-NEXT:  asr   x8, x8, #4
; CHECK-NEXT:  add   w0, w0, w8
; CHECK-NEXT:  ret
 %vscale = call i32 @llvm.vscale.i32()
 %sub = sub i32 %in, %vscale
 ret i32 %sub
}

; Fold (shl (vscale * C0), C1) to (vscale * (C0 << C1)).
; C0 = 1 , C1 = 4
; At IR level,  %shl = 2^4 * VSCALE.
; At Assembly level, the output of RDVL is also 2^4 * VSCALE.
; Hence, the immediate for RDVL is #1.
define i64 @combine_shl_vscale_i64() nounwind {
; CHECK-LABEL: combine_shl_vscale_i64:
; CHECK-NOT:   shl
; CHECK-NEXT:  rdvl  x0, #1
; CHECK-NEXT:  ret
 %vscale = call i64 @llvm.vscale.i64()
 %shl = shl i64 %vscale, 4
 ret i64 %shl
}

define i32 @combine_shl_vscale_i32() nounwind {
; CHECK-LABEL: combine_shl_vscale_i32:
; CHECK-NOT:   shl
; CHECK-NEXT:  rdvl  x0, #1
; CHECK-NEXT:  ret
 %vscale = call i32 @llvm.vscale.i32()
 %shl = shl i32 %vscale, 4
 ret i32 %shl
}
