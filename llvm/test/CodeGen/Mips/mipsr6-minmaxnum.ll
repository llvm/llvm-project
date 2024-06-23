; RUN: llc %s -mtriple=mipsisa32r6el-linux-gnu -o - | \
; RUN:     FileCheck %s --check-prefix=MIPS32R6EL
; RUN: llc %s -mtriple=mipsisa64r6el-linux-gnuabi64 -o - | \
; RUN:     FileCheck %s --check-prefix=MIPS64R6EL

define float @mins(float %x, float %y) {
; MIPS32R6EL-LABEL:	mins
; MIPS32R6EL:		# %bb.0:
; MIPS32R6EL-NEXT:	min.s	$f0, $f14, $f14
; MIPS32R6EL-NEXT:	min.s	$f1, $f12, $f12
; MIPS32R6EL-NEXT:	jr	$ra
; MIPS32R6EL-NEXT:	min.s	$f0, $f1, $f0
;
; MIPS64R6EL-LABEL:	mins
; MIPS64R6EL:		# %bb.0:
; MIPS64R6EL-NEXT:	min.s	$f0, $f13, $f13
; MIPS64R6EL-NEXT:	min.s	$f1, $f12, $f12
; MIPS64R6EL-NEXT:	jr	$ra
; MIPS64R6EL-NEXT:	min.s	$f0, $f1, $f0

  %r = tail call float @llvm.minnum.f32(float %x, float %y)
  ret float %r
}

define float @maxs(float %x, float %y) {
; MIPS32R6EL-LABEL:	maxs
; MIPS32R6EL:		# %bb.0:
; MIPS32R6EL-NEXT:	min.s	$f0, $f14, $f14
; MIPS32R6EL-NEXT:	min.s	$f1, $f12, $f12
; MIPS32R6EL-NEXT:	jr	$ra
; MIPS32R6EL-NEXT:	max.s	$f0, $f1, $f0
;
; MIPS64R6EL-LABEL:	maxs
; MIPS64R6EL:		# %bb.0:
; MIPS64R6EL-NEXT:	min.s	$f0, $f13, $f13
; MIPS64R6EL-NEXT:	min.s	$f1, $f12, $f12
; MIPS64R6EL-NEXT:	jr	$ra
; MIPS64R6EL-NEXT:	max.s	$f0, $f1, $f0

  %r = tail call float @llvm.maxnum.f32(float %x, float %y)
  ret float %r
}

define double @mind(double %x, double %y) {
; MIPS32R6EL-LABEL:	mind
; MIPS32R6EL:		# %bb.0:
; MIPS32R6EL-NEXT:	min.d	$f0, $f14, $f14
; MIPS32R6EL-NEXT:	min.d	$f1, $f12, $f12
; MIPS32R6EL-NEXT:	jr	$ra
; MIPS32R6EL-NEXT:	min.d	$f0, $f1, $f0
;
; MIPS64R6EL-LABEL:	mind
; MIPS64R6EL:		# %bb.0:
; MIPS64R6EL-NEXT:	min.d	$f0, $f13, $f13
; MIPS64R6EL-NEXT:	min.d	$f1, $f12, $f12
; MIPS64R6EL-NEXT:	jr	$ra
; MIPS64R6EL-NEXT:	min.d	$f0, $f1, $f0

  %r = tail call double @llvm.minnum.f64(double %x, double %y)
  ret double %r
}

define double @maxd(double %x, double %y) {
; MIPS32R6EL-LABEL:	maxd
; MIPS32R6EL:		# %bb.0:
; MIPS32R6EL-NEXT:	min.d	$f0, $f14, $f14
; MIPS32R6EL-NEXT:	min.d	$f1, $f12, $f12
; MIPS32R6EL-NEXT:	jr	$ra
; MIPS32R6EL-NEXT:	max.d	$f0, $f1, $f0
;
; MIPS64R6EL-LABEL:	maxd
; MIPS64R6EL:		# %bb.0:
; MIPS64R6EL-NEXT:	min.d	$f0, $f13, $f13
; MIPS64R6EL-NEXT:	min.d	$f1, $f12, $f12
; MIPS64R6EL-NEXT:	jr	$ra
; MIPS64R6EL-NEXT:	max.d	$f0, $f1, $f0

  %r = tail call double @llvm.maxnum.f64(double %x, double %y)
  ret double %r
}

declare float @llvm.minnum.f32(float, float)
declare float @llvm.maxnum.f32(float, float)
declare double @llvm.minnum.f64(double, double)
declare double @llvm.maxnum.f64(double, double)
