; RUN: llc %s -mtriple=mipsel-linux-gnu -o - | \
; RUN:     FileCheck %s --check-prefix=MIPS32R5EL
; RUN: llc %s -mtriple=mipsisa32r6el-linux-gnu -o - | \
; RUN:     FileCheck %s --check-prefix=MIPS32R6EL
; RUN: llc %s -mtriple=mips64el-linux-gnuabi64 -o - | \
; RUN:     FileCheck %s --check-prefix=MIPS64R5EL
; RUN: llc %s -mtriple=mipsisa64r6el-linux-gnuabi64 -o - | \
; RUN:     FileCheck %s --check-prefix=MIPS64R6EL

define float @maxs(float %x, float %y) unnamed_addr {
start:
; MIPS32R5EL-LABEL: 	maxs
; MIPS32R5EL:		# %bb.0:
; MIPS32R5EL-NEXT:	mfc1	$1, $f12
; MIPS32R5EL-NEXT:	slti	$1, $1, 0
; MIPS32R5EL-NEXT:	mov.s	$f1, $f12
; MIPS32R5EL-NEXT:	movn.s	$f1, $f14, $1
; MIPS32R5EL-NEXT:	c.ule.s	$f12, $f14
; MIPS32R5EL-NEXT:	mov.s	$f0, $f14
; MIPS32R5EL-NEXT:	movf.s	$f0, $f12, $fcc0
; MIPS32R5EL-NEXT:	lui	$1, %hi($CPI0_0)
; MIPS32R5EL-NEXT:	lwc1	$f2, %lo($CPI0_0)($1)
; MIPS32R5EL-NEXT:	c.un.s	$f12, $f14
; MIPS32R5EL-NEXT:	movt.s	$f0, $f2, $fcc0
; MIPS32R5EL-NEXT:	add.s	$f2, $f12, $f14
; MIPS32R5EL-NEXT:	mtc1	$zero, $f3
; MIPS32R5EL-NEXT:	c.eq.s	$f2, $f3
; MIPS32R5EL-NEXT:	jr	$ra
; MIPS32R5EL-NEXT:	movt.s	$f0, $f1, $fcc0
;
; MIPS64R5EL-LABEL:	maxs
; MIPS64R5EL:		# %bb.0:
; MIPS64R5EL-NEXT:	mfc1	$1, $f12
; MIPS64R5EL-NEXT:	slti	$1, $1, 0
; MIPS64R5EL-NEXT:	mov.s	$f1, $f12
; MIPS64R5EL-NEXT:	movn.s	$f1, $f13, $1
; MIPS64R5EL-NEXT:	c.ule.s	$f12, $f13
; MIPS64R5EL-NEXT:	mov.s	$f0, $f13
; MIPS64R5EL-NEXT:	movf.s	$f0, $f12, $fcc0
; MIPS64R5EL-NEXT:	lui	$1, %highest(.LCPI0_0)
; MIPS64R5EL-NEXT:	daddiu	$1, $1, %higher(.LCPI0_0)
; MIPS64R5EL-NEXT:	dsll	$1, $1, 16
; MIPS64R5EL-NEXT:	daddiu	$1, $1, %hi(.LCPI0_0)
; MIPS64R5EL-NEXT:	dsll	$1, $1, 16
; MIPS64R5EL-NEXT:	lwc1	$f2, %lo(.LCPI0_0)($1)
; MIPS64R5EL-NEXT:	c.un.s	$f12, $f13
; MIPS64R5EL-NEXT:	movt.s	$f0, $f2, $fcc0
; MIPS64R5EL-NEXT:	add.s	$f2, $f12, $f13
; MIPS64R5EL-NEXT:	mtc1	$zero, $f3
; MIPS64R5EL-NEXT:	c.eq.s	$f2, $f3
; MIPS64R5EL-NEXT:	jr	$ra
; MIPS64R5EL-NEXT:	movt.s	$f0, $f1, $fcc0
;
; MIPS32R6EL-LABEL:	maxs
; MIPS32R6EL:		# %bb.0:
; MIPS32R6EL-NEXT:	max.s	$f0, $f12, $f14
; MIPS32R6EL-NEXT:	cmp.un.s	$f1, $f12, $f14
; MIPS32R6EL-NEXT:	lui	$1, %hi($CPI0_0)
; MIPS32R6EL-NEXT:	lwc1	$f2, %lo($CPI0_0)($1)
; MIPS32R6EL-NEXT:	sel.s	$f1, $f0, $f2
; MIPS32R6EL-NEXT:	mfc1	$1, $f12
; MIPS32R6EL-NEXT:	slti	$1, $1, 0
; MIPS32R6EL-NEXT:	mtc1	$1, $f2
; MIPS32R6EL-NEXT:	sel.s	$f2, $f12, $f14
; MIPS32R6EL-NEXT:	add.s	$f0, $f12, $f14
; MIPS32R6EL-NEXT:	mtc1	$zero, $f3
; MIPS32R6EL-NEXT:	cmp.eq.s	$f0, $f0, $f3
; MIPS32R6EL-NEXT:	jr	$ra
; MIPS32R6EL-NEXT:	sel.s	$f0, $f1, $f2
;
; MIPS64R6EL-LABEL:	maxs
; MIPS64R6EL:		# %bb.0:
; MIPS64R6EL-NEXT:	mfc1	$1, $f12
; MIPS64R6EL-NEXT:	slti	$1, $1, 0
; MIPS64R6EL-NEXT:	mtc1	$1, $f1
; MIPS64R6EL-NEXT:	sel.s	$f1, $f12, $f13
; MIPS64R6EL-NEXT:	max.s	$f0, $f12, $f13
; MIPS64R6EL-NEXT:	cmp.un.s	$f2, $f12, $f13
; MIPS64R6EL-NEXT:	lui	$1, %highest(.LCPI0_0)
; MIPS64R6EL-NEXT:	daddiu	$1, $1, %higher(.LCPI0_0)
; MIPS64R6EL-NEXT:	dsll	$1, $1, 16
; MIPS64R6EL-NEXT:	daddiu	$1, $1, %hi(.LCPI0_0)
; MIPS64R6EL-NEXT:	dsll	$1, $1, 16
; MIPS64R6EL-NEXT:	lwc1	$f3, %lo(.LCPI0_0)($1)
; MIPS64R6EL-NEXT:	sel.s	$f2, $f0, $f3
; MIPS64R6EL-NEXT:	add.s	$f0, $f12, $f13
; MIPS64R6EL-NEXT:	mtc1	$zero, $f3
; MIPS64R6EL-NEXT:	cmp.eq.s	$f0, $f0, $f3
; MIPS64R6EL-NEXT:	jr	$ra
; MIPS64R6EL-NEXT:	sel.s	$f0, $f2, $f1

  %0 = tail call float @llvm.maximum.f32(float %x, float %y)
  ret float %0
}

define float @mins(float %x, float %y) unnamed_addr {
start:
; MIPS32R5EL-LABEL:	mins
; MIPS32R5EL:		# %bb.0:
; MIPS32R5EL-NEXT:	mfc1	$1, $f12
; MIPS32R5EL-NEXT:	slti	$1, $1, 0
; MIPS32R5EL-NEXT:	mov.s	$f1, $f14
; MIPS32R5EL-NEXT:	movn.s	$f1, $f12, $1
; MIPS32R5EL-NEXT:	c.olt.s	$f12, $f14
; MIPS32R5EL-NEXT:	mov.s	$f0, $f14
; MIPS32R5EL-NEXT:	movt.s	$f0, $f12, $fcc0
; MIPS32R5EL-NEXT:	lui	$1, %hi($CPI1_0)
; MIPS32R5EL-NEXT:	lwc1	$f2, %lo($CPI1_0)($1)
; MIPS32R5EL-NEXT:	c.un.s	$f12, $f14
; MIPS32R5EL-NEXT:	movt.s	$f0, $f2, $fcc0
; MIPS32R5EL-NEXT:	add.s	$f2, $f12, $f14
; MIPS32R5EL-NEXT:	mtc1	$zero, $f3
; MIPS32R5EL-NEXT:	c.eq.s	$f2, $f3
; MIPS32R5EL-NEXT:	jr	$ra
; MIPS32R5EL-NEXT:	movt.s	$f0, $f1, $fcc0
;
; MIPS64R5EL-LABEL:	mins
; MIPS64R5EL:		# %bb.0:
; MIPS64R5EL-NEXT:	mfc1	$1, $f12
; MIPS64R5EL-NEXT:	slti	$1, $1, 0
; MIPS64R5EL-NEXT:	mov.s	$f1, $f13
; MIPS64R5EL-NEXT:	movn.s	$f1, $f12, $1
; MIPS64R5EL-NEXT:	c.olt.s	$f12, $f13
; MIPS64R5EL-NEXT:	mov.s	$f0, $f13
; MIPS64R5EL-NEXT:	movt.s	$f0, $f12, $fcc0
; MIPS64R5EL-NEXT:	lui	$1, %highest(.LCPI1_0)
; MIPS64R5EL-NEXT:	daddiu	$1, $1, %higher(.LCPI1_0)
; MIPS64R5EL-NEXT:	dsll	$1, $1, 16
; MIPS64R5EL-NEXT:	daddiu	$1, $1, %hi(.LCPI1_0)
; MIPS64R5EL-NEXT:	dsll	$1, $1, 16
; MIPS64R5EL-NEXT:	lwc1	$f2, %lo(.LCPI1_0)($1)
; MIPS64R5EL-NEXT:	c.un.s	$f12, $f13
; MIPS64R5EL-NEXT:	movt.s	$f0, $f2, $fcc0
; MIPS64R5EL-NEXT:	add.s	$f2, $f12, $f13
; MIPS64R5EL-NEXT:	mtc1	$zero, $f3
; MIPS64R5EL-NEXT:	c.eq.s	$f2, $f3
; MIPS64R5EL-NEXT:	jr	$ra
; MIPS64R5EL-NEXT:	movt.s	$f0, $f1, $fcc0
;
; MIPS32R6EL-LABEL:	mins
; MIPS32R6EL:		# %bb.0:
; MIPS32R6EL-NEXT:	min.s	$f0, $f12, $f14
; MIPS32R6EL-NEXT:	cmp.un.s	$f1, $f12, $f14
; MIPS32R6EL-NEXT:	lui	$1, %hi($CPI1_0)
; MIPS32R6EL-NEXT:	lwc1	$f2, %lo($CPI1_0)($1)
; MIPS32R6EL-NEXT:	sel.s	$f1, $f0, $f2
; MIPS32R6EL-NEXT:	mfc1	$1, $f12
; MIPS32R6EL-NEXT:	slti	$1, $1, 0
; MIPS32R6EL-NEXT:	mtc1	$1, $f2
; MIPS32R6EL-NEXT:	sel.s	$f2, $f14, $f12
; MIPS32R6EL-NEXT:	add.s	$f0, $f12, $f14
; MIPS32R6EL-NEXT:	mtc1	$zero, $f3
; MIPS32R6EL-NEXT:	cmp.eq.s	$f0, $f0, $f3
; MIPS32R6EL-NEXT:	jr	$ra
; MIPS32R6EL-NEXT:	sel.s	$f0, $f1, $f2
;
; MIPS64R6EL-LABEL:	mins
; MIPS64R6EL:		# %bb.0:
; MIPS64R6EL-NEXT:	mfc1	$1, $f12
; MIPS64R6EL-NEXT:	slti	$1, $1, 0
; MIPS64R6EL-NEXT:	mtc1	$1, $f1
; MIPS64R6EL-NEXT:	sel.s	$f1, $f13, $f12
; MIPS64R6EL-NEXT:	min.s	$f0, $f12, $f13
; MIPS64R6EL-NEXT:	cmp.un.s	$f2, $f12, $f13
; MIPS64R6EL-NEXT:	lui	$1, %highest(.LCPI1_0)
; MIPS64R6EL-NEXT:	daddiu	$1, $1, %higher(.LCPI1_0)
; MIPS64R6EL-NEXT:	dsll	$1, $1, 16
; MIPS64R6EL-NEXT:	daddiu	$1, $1, %hi(.LCPI1_0)
; MIPS64R6EL-NEXT:	dsll	$1, $1, 16
; MIPS64R6EL-NEXT:	lwc1	$f3, %lo(.LCPI1_0)($1)
; MIPS64R6EL-NEXT:	sel.s	$f2, $f0, $f3
; MIPS64R6EL-NEXT:	add.s	$f0, $f12, $f13
; MIPS64R6EL-NEXT:	mtc1	$zero, $f3
; MIPS64R6EL-NEXT:	cmp.eq.s	$f0, $f0, $f3
; MIPS64R6EL-NEXT:	jr	$ra
; MIPS64R6EL-NEXT:	sel.s	$f0, $f2, $f1

  %0 = tail call float @llvm.minimum.f32(float %x, float %y)
  ret float %0
}

define double @maxd(double %x, double %y) unnamed_addr {
start:
; MIPS32R5EL-LABEL:	maxd
; MIPS32R5EL:		# %bb.0
; MIPS32R5EL-NEXT:	mfc1	$1, $f13
; MIPS32R5EL-NEXT:	slti	$1, $1, 0
; MIPS32R5EL-NEXT:	mov.d	$f2, $f12
; MIPS32R5EL-NEXT:	movn.d	$f2, $f14, $1
; MIPS32R5EL-NEXT:	c.ule.d	$f12, $f14
; MIPS32R5EL-NEXT:	mov.d	$f0, $f14
; MIPS32R5EL-NEXT:	movf.d	$f0, $f12, $fcc0
; MIPS32R5EL-NEXT:	lui	$1, %hi($CPI2_0)
; MIPS32R5EL-NEXT:	ldc1	$f4, %lo($CPI2_0)($1)
; MIPS32R5EL-NEXT:	c.un.d	$f12, $f14
; MIPS32R5EL-NEXT:	movt.d	$f0, $f4, $fcc0
; MIPS32R5EL-NEXT:	add.d	$f4, $f12, $f14
; MIPS32R5EL-NEXT:	mtc1	$zero, $f6
; MIPS32R5EL-NEXT:	mtc1	$zero, $f7
; MIPS32R5EL-NEXT:	c.eq.d	$f4, $f6
; MIPS32R5EL-NEXT:	jr	$ra
; MIPS32R5EL-NEXT:	movt.d	$f0, $f2, $fcc0
;
; MIPS64R5EL-LABEL:	maxd
; MIPS64R5EL:		# %bb.0:
; MIPS64R5EL-NEXT:	dmfc1	$1, $f12
; MIPS64R5EL-NEXT:	slti	$1, $1, 0
; MIPS64R5EL-NEXT:	mov.d	$f1, $f12
; MIPS64R5EL-NEXT:	movn.d	$f1, $f13, $1
; MIPS64R5EL-NEXT:	c.ule.d	$f12, $f13
; MIPS64R5EL-NEXT:	mov.d	$f0, $f13
; MIPS64R5EL-NEXT:	movf.d	$f0, $f12, $fcc0
; MIPS64R5EL-NEXT:	lui	$1, %highest(.LCPI2_0)
; MIPS64R5EL-NEXT:	daddiu	$1, $1, %higher(.LCPI2_0)
; MIPS64R5EL-NEXT:	dsll	$1, $1, 16
; MIPS64R5EL-NEXT:	daddiu	$1, $1, %hi(.LCPI2_0)
; MIPS64R5EL-NEXT:	dsll	$1, $1, 16
; MIPS64R5EL-NEXT:	ldc1	$f2, %lo(.LCPI2_0)($1)
; MIPS64R5EL-NEXT:	c.un.d	$f12, $f13
; MIPS64R5EL-NEXT:	movt.d	$f0, $f2, $fcc0
; MIPS64R5EL-NEXT:	add.d	$f2, $f12, $f13
; MIPS64R5EL-NEXT:	dmtc1	$zero, $f3
; MIPS64R5EL-NEXT:	c.eq.d	$f2, $f3
; MIPS64R5EL-NEXT:	jr	$ra
; MIPS64R5EL-NEXT:	movt.d	$f0, $f1, $fcc0
;
; MIPS32R6EL-LABEL:	maxd
; MIPS32R6EL:		# %bb.0:
; MIPS32R6EL-NEXT:	max.d	$f0, $f12, $f14
; MIPS32R6EL-NEXT:	cmp.un.d	$f1, $f12, $f14
; MIPS32R6EL-NEXT:	mfc1	$1, $f1
; MIPS32R6EL-NEXT:	mtc1	$1, $f1
; MIPS32R6EL-NEXT:	mfhc1	$1, $f12
; MIPS32R6EL-NEXT:	slti	$1, $1, 0
; MIPS32R6EL-NEXT:	lui	$2, %hi($CPI2_0)
; MIPS32R6EL-NEXT:	ldc1	$f2, %lo($CPI2_0)($2)
; MIPS32R6EL-NEXT:	sel.d	$f1, $f0, $f2
; MIPS32R6EL-NEXT:	mtc1	$1, $f2
; MIPS32R6EL-NEXT:	sel.d	$f2, $f12, $f14
; MIPS32R6EL-NEXT:	add.d	$f0, $f12, $f14
; MIPS32R6EL-NEXT:	mtc1	$zero, $f3
; MIPS32R6EL-NEXT:	mthc1	$zero, $f3
; MIPS32R6EL-NEXT:	cmp.eq.d	$f0, $f0, $f3
; MIPS32R6EL-NEXT:	mfc1	$1, $f0
; MIPS32R6EL-NEXT:	mtc1	$1, $f0
; MIPS32R6EL-NEXT:	jr	$ra
; MIPS32R6EL-NEXT:	sel.d	$f0, $f1, $f2
;
; MIPS64R6EL-LABEL:	maxd
; MIPS64R6EL:		# %bb.0:
; MIPS64R6EL-NEXT:	cmp.un.d	$f0, $f12, $f13
; MIPS64R6EL-NEXT:	dmfc1	$1, $f12
; MIPS64R6EL-NEXT:	slti	$1, $1, 0
; MIPS64R6EL-NEXT:	mtc1	$1, $f1
; MIPS64R6EL-NEXT:	sel.d	$f1, $f12, $f13
; MIPS64R6EL-NEXT:	max.d	$f2, $f12, $f13
; MIPS64R6EL-NEXT:	mfc1	$1, $f0
; MIPS64R6EL-NEXT:	mtc1	$1, $f3
; MIPS64R6EL-NEXT:	lui	$1, %highest(.LCPI2_0)
; MIPS64R6EL-NEXT:	daddiu	$1, $1, %higher(.LCPI2_0)
; MIPS64R6EL-NEXT:	dsll	$1, $1, 16
; MIPS64R6EL-NEXT:	daddiu	$1, $1, %hi(.LCPI2_0)
; MIPS64R6EL-NEXT:	dsll	$1, $1, 16
; MIPS64R6EL-NEXT:	ldc1	$f0, %lo(.LCPI2_0)($1)
; MIPS64R6EL-NEXT:	sel.d	$f3, $f2, $f0
; MIPS64R6EL-NEXT:	add.d	$f0, $f12, $f13
; MIPS64R6EL-NEXT:	dmtc1	$zero, $f2
; MIPS64R6EL-NEXT:	cmp.eq.d	$f0, $f0, $f2
; MIPS64R6EL-NEXT:	mfc1	$1, $f0
; MIPS64R6EL-NEXT:	mtc1	$1, $f0
; MIPS64R6EL-NEXT:	jr	$ra
; MIPS64R6EL-NEXT:	sel.d	$f0, $f3, $f1

  %0 = tail call double @llvm.maximum.f64(double %x, double %y)
  ret double %0
}

define double @mind(double %x, double %y) unnamed_addr {
start:
; MIPS32R5EL-LABEL:	mind
; MIPS32R5EL:		# %bb.0:
; MIPS32R5EL-NEXT:	mfc1	$1, $f13
; MIPS32R5EL-NEXT:	slti	$1, $1, 0
; MIPS32R5EL-NEXT:	mov.d	$f2, $f14
; MIPS32R5EL-NEXT:	movn.d	$f2, $f12, $1
; MIPS32R5EL-NEXT:	c.olt.d	$f12, $f14
; MIPS32R5EL-NEXT:	mov.d	$f0, $f14
; MIPS32R5EL-NEXT:	movt.d	$f0, $f12, $fcc0
; MIPS32R5EL-NEXT:	lui	$1, %hi($CPI3_0)
; MIPS32R5EL-NEXT:	ldc1	$f4, %lo($CPI3_0)($1)
; MIPS32R5EL-NEXT:	c.un.d	$f12, $f14
; MIPS32R5EL-NEXT:	movt.d	$f0, $f4, $fcc0
; MIPS32R5EL-NEXT:	add.d	$f4, $f12, $f14
; MIPS32R5EL-NEXT:	mtc1	$zero, $f6
; MIPS32R5EL-NEXT:	mtc1	$zero, $f7
; MIPS32R5EL-NEXT:	c.eq.d	$f4, $f6
; MIPS32R5EL-NEXT:	jr	$ra
; MIPS32R5EL-NEXT:	movt.d	$f0, $f2, $fcc0
;
; MIPS64R5EL-LABEL:	mind
; MIPS64R5EL:		# %bb.0:
; MIPS64R5EL-NEXT:	dmfc1	$1, $f12
; MIPS64R5EL-NEXT:	slti	$1, $1, 0
; MIPS64R5EL-NEXT:	mov.d	$f1, $f13
; MIPS64R5EL-NEXT:	movn.d	$f1, $f12, $1
; MIPS64R5EL-NEXT:	c.olt.d	$f12, $f13
; MIPS64R5EL-NEXT:	mov.d	$f0, $f13
; MIPS64R5EL-NEXT:	movt.d	$f0, $f12, $fcc0
; MIPS64R5EL-NEXT:	lui	$1, %highest(.LCPI3_0)
; MIPS64R5EL-NEXT:	daddiu	$1, $1, %higher(.LCPI3_0)
; MIPS64R5EL-NEXT:	dsll	$1, $1, 16
; MIPS64R5EL-NEXT:	daddiu	$1, $1, %hi(.LCPI3_0)
; MIPS64R5EL-NEXT:	dsll	$1, $1, 16
; MIPS64R5EL-NEXT:	ldc1	$f2, %lo(.LCPI3_0)($1)
; MIPS64R5EL-NEXT:	c.un.d	$f12, $f13
; MIPS64R5EL-NEXT:	movt.d	$f0, $f2, $fcc0
; MIPS64R5EL-NEXT:	add.d	$f2, $f12, $f13
; MIPS64R5EL-NEXT:	dmtc1	$zero, $f3
; MIPS64R5EL-NEXT:	c.eq.d	$f2, $f3
; MIPS64R5EL-NEXT:	jr	$ra
; MIPS64R5EL-NEXT:	movt.d	$f0, $f1, $fcc0
;
; MIPS32R6EL-LABEL:	mind
; MIPS32R6EL:		# %bb.0:
; MIPS32R6EL-NEXT:	min.d	$f0, $f12, $f14
; MIPS32R6EL-NEXT:	cmp.un.d	$f1, $f12, $f14
; MIPS32R6EL-NEXT:	mfc1	$1, $f1
; MIPS32R6EL-NEXT:	mtc1	$1, $f1
; MIPS32R6EL-NEXT:	mfhc1	$1, $f12
; MIPS32R6EL-NEXT:	slti	$1, $1, 0
; MIPS32R6EL-NEXT:	lui	$2, %hi($CPI3_0)
; MIPS32R6EL-NEXT:	ldc1	$f2, %lo($CPI3_0)($2)
; MIPS32R6EL-NEXT:	sel.d	$f1, $f0, $f2
; MIPS32R6EL-NEXT:	mtc1	$1, $f2
; MIPS32R6EL-NEXT:	sel.d	$f2, $f14, $f12
; MIPS32R6EL-NEXT:	add.d	$f0, $f12, $f14
; MIPS32R6EL-NEXT:	mtc1	$zero, $f3
; MIPS32R6EL-NEXT:	mthc1	$zero, $f3
; MIPS32R6EL-NEXT:	cmp.eq.d	$f0, $f0, $f3
; MIPS32R6EL-NEXT:	mfc1	$1, $f0
; MIPS32R6EL-NEXT:	mtc1	$1, $f0
; MIPS32R6EL-NEXT:	jr	$ra
; MIPS32R6EL-NEXT:	sel.d	$f0, $f1, $f2
;
; MIPS64R6EL-LABEL:	mind
; MIPS64R6EL:		# %bb.0:
; MIPS64R6EL-NEXT:	cmp.un.d	$f0, $f12, $f13
; MIPS64R6EL-NEXT:	dmfc1	$1, $f12
; MIPS64R6EL-NEXT:	slti	$1, $1, 0
; MIPS64R6EL-NEXT:	mtc1	$1, $f1
; MIPS64R6EL-NEXT:	sel.d	$f1, $f13, $f12
; MIPS64R6EL-NEXT:	min.d	$f2, $f12, $f13
; MIPS64R6EL-NEXT:	mfc1	$1, $f0
; MIPS64R6EL-NEXT:	mtc1	$1, $f3
; MIPS64R6EL-NEXT:	lui	$1, %highest(.LCPI3_0)
; MIPS64R6EL-NEXT:	daddiu	$1, $1, %higher(.LCPI3_0)
; MIPS64R6EL-NEXT:	dsll	$1, $1, 16
; MIPS64R6EL-NEXT:	daddiu	$1, $1, %hi(.LCPI3_0)
; MIPS64R6EL-NEXT:	dsll	$1, $1, 16
; MIPS64R6EL-NEXT:	ldc1	$f0, %lo(.LCPI3_0)($1)
; MIPS64R6EL-NEXT:	sel.d	$f3, $f2, $f0
; MIPS64R6EL-NEXT:	add.d	$f0, $f12, $f13
; MIPS64R6EL-NEXT:	dmtc1	$zero, $f2
; MIPS64R6EL-NEXT:	cmp.eq.d	$f0, $f0, $f2
; MIPS64R6EL-NEXT:	mfc1	$1, $f0
; MIPS64R6EL-NEXT:	mtc1	$1, $f0
; MIPS64R6EL-NEXT:	jr	$ra
; MIPS64R6EL-NEXT:	sel.d	$f0, $f3, $f1

  %0 = tail call double @llvm.minimum.f64(double %x, double %y)
  ret double %0
}

declare float @llvm.minimum.f32(float, float)
declare float @llvm.maximum.f32(float, float)
declare double @llvm.minimum.f64(double, double)
declare double @llvm.maximum.f64(double, double)
