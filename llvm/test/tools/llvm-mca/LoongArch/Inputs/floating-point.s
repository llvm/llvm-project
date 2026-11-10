fadd.d $fa0, $fa1, $fa2
fmul.d $fa0, $fa1, $fa2
fmadd.d $fa0, $fa1, $fa2, $fa3
fmax.d $fa0, $fa1, $fa2
fabs.d $fa0, $fa1
fcopysign.d $fa0, $fa1, $fa2
fclass.d $fa0, $fa1
fscaleb.d $fa0, $fa1, $fa2
flogb.d $fa0, $fa1
fdiv.s $fa0, $fa1, $fa2
fdiv.d $fa0, $fa1, $fa2
fsqrt.s $fa0, $fa1
fsqrt.d $fa0, $fa1
frecip.d $fa0, $fa1
frsqrt.d $fa0, $fa1
fcmp.ceq.d $fcc0, $fa1, $fa2
fsel $fa0, $fa1, $fa2, $fcc0
fcvt.s.d $fa0, $fa1
ftint.w.d $fa0, $fa1
ffint.d.l $fa0, $fa1
frint.d $fa0, $fa1
fmov.d $fa0, $fa1
movgr2fr.d $fa0, $a1
movfr2gr.d $a0, $fa1
movgr2cf $fcc0, $a0
movcf2gr $a0, $fcc0
bcnez $fcc0, 8
