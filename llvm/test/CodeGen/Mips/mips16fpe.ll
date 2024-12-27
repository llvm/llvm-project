; RUN: llc -mtriple=mipsel -mattr=mips16 -relocation-model=pic -O3 \
; RUN:     -verify-machineinstrs < %s | FileCheck %s -check-prefix=16hf

@x = global float 5.000000e+00, align 4
@y = global float 1.500000e+01, align 4
@xd = global double 6.000000e+00, align 8
@yd = global double 1.800000e+01, align 8
@two = global i32 2, align 4
@addsf3_result = common global float 0.000000e+00, align 4
@adddf3_result = common global double 0.000000e+00, align 8
@subsf3_result = common global float 0.000000e+00, align 4
@subdf3_result = common global double 0.000000e+00, align 8
@mulsf3_result = common global float 0.000000e+00, align 4
@muldf3_result = common global double 0.000000e+00, align 8
@divsf3_result = common global float 0.000000e+00, align 4
@divdf3_result = common global double 0.000000e+00, align 8
@extendsfdf2_result = common global double 0.000000e+00, align 8
@xd2 = global double 0x40147E6B74B4CF6A, align 8
@truncdfsf2_result = common global float 0.000000e+00, align 4
@fix_truncsfsi_result = common global i32 0, align 4
@fix_truncdfsi_result = common global i32 0, align 4
@si = global i32 -9, align 4
@ui = global i32 9, align 4
@floatsisf_result = common global float 0.000000e+00, align 4
@floatsidf_result = common global double 0.000000e+00, align 8
@floatunsisf_result = common global float 0.000000e+00, align 4
@floatunsidf_result = common global double 0.000000e+00, align 8
@xx = global float 5.000000e+00, align 4
@eqsf2_result = common global i32 0, align 4
@xxd = global double 6.000000e+00, align 8
@eqdf2_result = common global i32 0, align 4
@nesf2_result = common global i32 0, align 4
@nedf2_result = common global i32 0, align 4
@gesf2_result = common global i32 0, align 4
@gedf2_result = common global i32 0, align 4
@ltsf2_result = common global i32 0, align 4
@ltdf2_result = common global i32 0, align 4
@lesf2_result = common global i32 0, align 4
@ledf2_result = common global i32 0, align 4
@gtsf2_result = common global i32 0, align 4
@gtdf2_result = common global i32 0, align 4

define void @test_addsf3() nounwind {
entry:
;16hf-LABEL: test_addsf3:
  %0 = load float, ptr @x, align 4
  %1 = load float, ptr @y, align 4
  %add = fadd float %0, %1
  store float %add, ptr @addsf3_result, align 4
;16hf:  lw	${{[0-9]+}}, %call16(__mips16_addsf3)(${{[0-9]+}})
  ret void
}

define void @test_adddf3() nounwind {
entry:
;16hf-LABEL: test_adddf3:
  %0 = load double, ptr @xd, align 8
  %1 = load double, ptr @yd, align 8
  %add = fadd double %0, %1
  store double %add, ptr @adddf3_result, align 8
;16hf:  lw	${{[0-9]+}}, %call16(__mips16_adddf3)(${{[0-9]+}})
  ret void
}

define void @test_subsf3() nounwind {
entry:
;16hf-LABEL: test_subsf3:
  %0 = load float, ptr @x, align 4
  %1 = load float, ptr @y, align 4
  %sub = fsub float %0, %1
  store float %sub, ptr @subsf3_result, align 4
;16hf:  lw	${{[0-9]+}}, %call16(__mips16_subsf3)(${{[0-9]+}})
  ret void
}

define void @test_subdf3() nounwind {
entry:
;16hf-LABEL: test_subdf3:
  %0 = load double, ptr @xd, align 8
  %1 = load double, ptr @yd, align 8
  %sub = fsub double %0, %1
  store double %sub, ptr @subdf3_result, align 8
;16hf:  lw	${{[0-9]+}}, %call16(__mips16_subdf3)(${{[0-9]+}})
  ret void
}

define void @test_mulsf3() nounwind {
entry:
;16hf-LABEL: test_mulsf3:
  %0 = load float, ptr @x, align 4
  %1 = load float, ptr @y, align 4
  %mul = fmul float %0, %1
  store float %mul, ptr @mulsf3_result, align 4
;16hf:  lw	${{[0-9]+}}, %call16(__mips16_mulsf3)(${{[0-9]+}})
  ret void
}

define void @test_muldf3() nounwind {
entry:
;16hf-LABEL: test_muldf3:
  %0 = load double, ptr @xd, align 8
  %1 = load double, ptr @yd, align 8
  %mul = fmul double %0, %1
  store double %mul, ptr @muldf3_result, align 8
;16hf:  lw	${{[0-9]+}}, %call16(__mips16_muldf3)(${{[0-9]+}})
  ret void
}

define void @test_divsf3() nounwind {
entry:
;16hf-LABEL: test_divsf3:
  %0 = load float, ptr @y, align 4
  %1 = load float, ptr @x, align 4
  %div = fdiv float %0, %1
  store float %div, ptr @divsf3_result, align 4
;16hf:  lw	${{[0-9]+}}, %call16(__mips16_divsf3)(${{[0-9]+}})
  ret void
}

define void @test_divdf3() nounwind {
entry:
;16hf-LABEL: test_divdf3:
  %0 = load double, ptr @yd, align 8
  %mul = fmul double %0, 2.000000e+00
  %1 = load double, ptr @xd, align 8
  %div = fdiv double %mul, %1
  store double %div, ptr @divdf3_result, align 8
;16hf:  lw	${{[0-9]+}}, %call16(__mips16_divdf3)(${{[0-9]+}})
  ret void
}

define void @test_extendsfdf2() nounwind {
entry:
;16hf-LABEL: test_extendsfdf2:
  %0 = load float, ptr @x, align 4
  %conv = fpext float %0 to double
  store double %conv, ptr @extendsfdf2_result, align 8
;16hf:  lw	${{[0-9]+}}, %call16(__mips16_extendsfdf2)(${{[0-9]+}})
  ret void
}

define void @test_truncdfsf2() nounwind {
entry:
;16hf-LABEL: test_truncdfsf2:
  %0 = load double, ptr @xd2, align 8
  %conv = fptrunc double %0 to float
  store float %conv, ptr @truncdfsf2_result, align 4
;16hf:  lw	${{[0-9]+}}, %call16(__mips16_truncdfsf2)(${{[0-9]+}})
  ret void
}

define void @test_fix_truncsfsi() nounwind {
entry:
;16hf-LABEL: test_fix_truncsfsi:
  %0 = load float, ptr @x, align 4
  %conv = fptosi float %0 to i32
  store i32 %conv, ptr @fix_truncsfsi_result, align 4
;16hf:  lw	${{[0-9]+}}, %call16(__mips16_fix_truncsfsi)(${{[0-9]+}})
  ret void
}

define void @test_fix_truncdfsi() nounwind {
entry:
;16hf-LABEL: test_fix_truncdfsi:
  %0 = load double, ptr @xd, align 8
  %conv = fptosi double %0 to i32
  store i32 %conv, ptr @fix_truncdfsi_result, align 4
;16hf:  lw	${{[0-9]+}}, %call16(__mips16_fix_truncdfsi)(${{[0-9]+}})
  ret void
}

define void @test_floatsisf() nounwind {
entry:
;16hf-LABEL: test_floatsisf:
  %0 = load i32, ptr @si, align 4
  %conv = sitofp i32 %0 to float
  store float %conv, ptr @floatsisf_result, align 4
;16hf:  lw	${{[0-9]+}}, %call16(__mips16_floatsisf)(${{[0-9]+}})
  ret void
}

define void @test_floatsidf() nounwind {
entry:
;16hf-LABEL: test_floatsidf:
  %0 = load i32, ptr @si, align 4
  %conv = sitofp i32 %0 to double
  store double %conv, ptr @floatsidf_result, align 8
;16hf:  lw	${{[0-9]+}}, %call16(__mips16_floatsidf)(${{[0-9]+}})
  ret void
}

define void @test_floatunsisf() nounwind {
entry:
;16hf-LABEL: test_floatunsisf:
  %0 = load i32, ptr @ui, align 4
  %conv = uitofp i32 %0 to float
  store float %conv, ptr @floatunsisf_result, align 4
;16hf:  lw	${{[0-9]+}}, %call16(__mips16_floatunsisf)(${{[0-9]+}})
  ret void
}

define void @test_floatunsidf() nounwind {
entry:
;16hf-LABEL: test_floatunsidf:
  %0 = load i32, ptr @ui, align 4
  %conv = uitofp i32 %0 to double
  store double %conv, ptr @floatunsidf_result, align 8
;16hf:  lw	${{[0-9]+}}, %call16(__mips16_floatunsidf)(${{[0-9]+}})
  ret void
}

define void @test_eqsf2() nounwind {
entry:
;16hf-LABEL: test_eqsf2:
  %0 = load float, ptr @x, align 4
  %1 = load float, ptr @xx, align 4
  %cmp = fcmp oeq float %0, %1
  %conv = zext i1 %cmp to i32
  store i32 %conv, ptr @eqsf2_result, align 4
;16hf:  lw	${{[0-9]+}}, %call16(__mips16_eqsf2)(${{[0-9]+}})
  ret void
}

define void @test_eqdf2() nounwind {
entry:
;16hf-LABEL: test_eqdf2:
  %0 = load double, ptr @xd, align 8
  %1 = load double, ptr @xxd, align 8
  %cmp = fcmp oeq double %0, %1
  %conv = zext i1 %cmp to i32
  store i32 %conv, ptr @eqdf2_result, align 4
;16hf:  lw	${{[0-9]+}}, %call16(__mips16_eqdf2)(${{[0-9]+}})
  ret void
}

define void @test_nesf2() nounwind {
entry:
;16hf-LABEL: test_nesf2:
  %0 = load float, ptr @x, align 4
  %1 = load float, ptr @y, align 4
  %cmp = fcmp une float %0, %1
  %conv = zext i1 %cmp to i32
  store i32 %conv, ptr @nesf2_result, align 4
;16hf:  lw	${{[0-9]+}}, %call16(__mips16_nesf2)(${{[0-9]+}})
  ret void
}

define void @test_nedf2() nounwind {
entry:
;16hf-LABEL: test_nedf2:
  %0 = load double, ptr @xd, align 8
  %1 = load double, ptr @yd, align 8
  %cmp = fcmp une double %0, %1
  %conv = zext i1 %cmp to i32
  store i32 %conv, ptr @nedf2_result, align 4
;16hf:  lw	${{[0-9]+}}, %call16(__mips16_nedf2)(${{[0-9]+}})
  ret void
}

define void @test_gesf2() nounwind {
entry:
;16hf-LABEL: test_gesf2:
  %0 = load float, ptr @x, align 4
  %1 = load float, ptr @xx, align 4
  %cmp = fcmp oge float %0, %1
  %2 = load float, ptr @y, align 4
  %cmp1 = fcmp oge float %2, %0
  %and3 = and i1 %cmp, %cmp1
  %and = zext i1 %and3 to i32
  store i32 %and, ptr @gesf2_result, align 4
;16hf:  lw	${{[0-9]+}}, %call16(__mips16_gesf2)(${{[0-9]+}})
  ret void
}

define void @test_gedf2() nounwind {
entry:
;16hf-LABEL: test_gedf2:
  %0 = load double, ptr @xd, align 8
  %1 = load double, ptr @xxd, align 8
  %cmp = fcmp oge double %0, %1
  %2 = load double, ptr @yd, align 8
  %cmp1 = fcmp oge double %2, %0
  %and3 = and i1 %cmp, %cmp1
  %and = zext i1 %and3 to i32
  store i32 %and, ptr @gedf2_result, align 4
;16hf:  lw	${{[0-9]+}}, %call16(__mips16_gedf2)(${{[0-9]+}})
  ret void
}

define void @test_ltsf2() nounwind {
entry:
;16hf-LABEL: test_ltsf2:
  %0 = load float, ptr @x, align 4
  %1 = load float, ptr @xx, align 4
  %lnot = fcmp uge float %0, %1
  %2 = load float, ptr @y, align 4
  %cmp1 = fcmp olt float %0, %2
  %and2 = and i1 %lnot, %cmp1
  %and = zext i1 %and2 to i32
  store i32 %and, ptr @ltsf2_result, align 4
;16hf:  lw	${{[0-9]+}}, %call16(__mips16_ltsf2)(${{[0-9]+}})
;16hf:  lw	${{[0-9]+}}, %call16(__mips16_ltsf2)(${{[0-9]+}})
  ret void
}

define void @test_ltdf2() nounwind {
entry:
;16hf-LABEL: test_ltdf2:
  %0 = load double, ptr @xd, align 8
  %1 = load double, ptr @xxd, align 8
  %lnot = fcmp uge double %0, %1
  %2 = load double, ptr @yd, align 8
  %cmp1 = fcmp olt double %0, %2
  %and2 = and i1 %lnot, %cmp1
  %and = zext i1 %and2 to i32
  store i32 %and, ptr @ltdf2_result, align 4
;16hf:  lw	${{[0-9]+}}, %call16(__mips16_ltdf2)(${{[0-9]+}})
;16hf:  lw	${{[0-9]+}}, %call16(__mips16_ltdf2)(${{[0-9]+}})
  ret void
}

define void @test_lesf2() nounwind {
entry:
;16hf-LABEL: test_lesf2:
  %0 = load float, ptr @x, align 4
  %1 = load float, ptr @xx, align 4
  %cmp = fcmp ole float %0, %1
  %2 = load float, ptr @y, align 4
  %cmp1 = fcmp ole float %0, %2
  %and3 = and i1 %cmp, %cmp1
  %and = zext i1 %and3 to i32
  store i32 %and, ptr @lesf2_result, align 4
;16hf:  lw	${{[0-9]+}}, %call16(__mips16_lesf2)(${{[0-9]+}})
  ret void
}

define void @test_ledf2() nounwind {
entry:
;16hf-LABEL: test_ledf2:
  %0 = load double, ptr @xd, align 8
  %1 = load double, ptr @xxd, align 8
  %cmp = fcmp ole double %0, %1
  %2 = load double, ptr @yd, align 8
  %cmp1 = fcmp ole double %0, %2
  %and3 = and i1 %cmp, %cmp1
  %and = zext i1 %and3 to i32
  store i32 %and, ptr @ledf2_result, align 4
;16hf:  lw	${{[0-9]+}}, %call16(__mips16_ledf2)(${{[0-9]+}})
  ret void
}

define void @test_gtsf2() nounwind {
entry:
;16hf-LABEL: test_gtsf2:
  %0 = load float, ptr @x, align 4
  %1 = load float, ptr @xx, align 4
  %lnot = fcmp ule float %0, %1
  %2 = load float, ptr @y, align 4
  %cmp1 = fcmp ogt float %2, %0
  %and2 = and i1 %lnot, %cmp1
  %and = zext i1 %and2 to i32
  store i32 %and, ptr @gtsf2_result, align 4
;16hf:  lw	${{[0-9]+}}, %call16(__mips16_gtsf2)(${{[0-9]+}})
  ret void
}

define void @test_gtdf2() nounwind {
entry:
;16hf-LABEL: test_gtdf2:
  %0 = load double, ptr @xd, align 8
  %1 = load double, ptr @xxd, align 8
  %lnot = fcmp ule double %0, %1
  %2 = load double, ptr @yd, align 8
  %cmp1 = fcmp ogt double %2, %0
  %and2 = and i1 %lnot, %cmp1
  %and = zext i1 %and2 to i32
  store i32 %and, ptr @gtdf2_result, align 4
;16hf:  lw	${{[0-9]+}}, %call16(__mips16_gtdf2)(${{[0-9]+}})
  ret void
}


