; RUN: llc -mtriple=mips64el-linux-gnu64abi -mcpu=mips64r5 -mattr=+msa < %s \
; RUN:   | FileCheck %s --check-prefixes=MIPS64R5

@v4f32 = global <4 x float> <float 0.0, float 0.0, float 0.0, float 0.0>

define void @nonsplatvalue_v4f32(float %a, float %b, float %c, float %d) nounwind {
; MIPS64R5-LABEL: nonsplatvalue_v4f32:
; MIPS64R5:       # %bb.0:
; MIPS64R5-NEXT:    # kill: def $f15 killed $f15 def $w15
; MIPS64R5-NEXT:    # kill: def $f14 killed $f14 def $w14
; MIPS64R5-NEXT:    # kill: def $f13 killed $f13 def $w13
; MIPS64R5-NEXT:    # kill: def $f12 killed $f12 def $w12
; MIPS64R5-NEXT:    insve.w $w0[0], $w12[0]
; MIPS64R5-NEXT:    insve.w $w0[1], $w13[0]
; MIPS64R5-NEXT:    insve.w $w0[2], $w14[0]
; MIPS64R5-NEXT:    insve.w $w0[3], $w15[0]
; MIPS64R5-NEXT:    fmax_a.w $w0, $w0, $w0
; MIPS64R5-NEXT:    lui $1, %highest(v4f32)
; MIPS64R5-NEXT:    daddiu $1, $1, %higher(v4f32)
; MIPS64R5-NEXT:    dsll $1, $1, 16
; MIPS64R5-NEXT:    daddiu $1, $1, %hi(v4f32)
; MIPS64R5-NEXT:    dsll $1, $1, 16
; MIPS64R5-NEXT:    daddiu $1, $1, %lo(v4f32)
; MIPS64R5-NEXT:    jr $ra
; MIPS64R5-NEXT:    st.w $w0, 0($1)
  %v0 = insertelement <4 x float> poison, float %a, i64 0
  %v1 = insertelement <4 x float> %v0, float %b, i32 1
  %v2 = insertelement <4 x float> %v1, float %c, i32 2
  %v3 = insertelement <4 x float> %v2, float %d, i32 3

  %fabs = call <4 x float> @llvm.fabs.v4f32(<4 x float> %v3)
  store volatile <4 x float> %fabs, ptr @v4f32

  ret void
}

declare <4 x float> @llvm.fabs.v4f32(<4 x float>)
