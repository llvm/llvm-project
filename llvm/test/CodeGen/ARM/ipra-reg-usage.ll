; RUN: llc -mcpu=cortex-a8 -enable-ipra -print-regusage -o /dev/null 2>&1 < %s | FileCheck %s
; ipra used to wrongly assumed that registers without subregisters were saved.
; In that example, r0 wouldn't be in the list of clobbered registers.
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "armv7-eabi"

declare void @bar1()
define void @foo()#0 {
; CHECK: foo Clobbered Registers: $apsr $apsr_nzcv $cpsr $fpcxtns $fpcxts $fpexc $fpinst $fpscr $fpscr_nzcv $fpscr_nzcvqc $fpscr_rm $fpsid $itstate $pc $ra_auth_code $sp $spsr $vpr $zr $d0 $d1 $d2 $d3 $d4 $d5 $d6 $d7 $d16 $d17 $d18 $d19 $d20 $d21 $d22 $d23 $d24 $d25 $d26 $d27 $d28 $d29 $d30 $d31 $fpinst2 $mvfr0 $mvfr1 $mvfr2 $p0 $q0 $q1 $q2 $q3 $q8 $q9 $q10 $q11 $q12 $q13 $q14 $q15 $r0 $r1 $r2 $r3 $r12 $s0 $s1 $s2 $s3 $s4 $s5 $s6 $s7 $s8 $s9 $s10 $s11 $s12 $s13 $s14 $s15 $d0_d2 $d1_d3 $d2_d4 $d3_d5 $d4_d6 $d5_d7 $d6_d8 $d7_d9 $d14_d16 $d15_d17 $d16_d18 $d17_d19 $d18_d20 $d19_d21 $d20_d22 $d21_d23 $d22_d24 $d23_d25 $d24_d26 $d25_d27 $d26_d28 $d27_d29 $d28_d30 $d29_d31 $q0_q1 $q1_q2 $q2_q3 $q3_q4 $q7_q8 $q8_q9 $q9_q10 $q10_q11 $q11_q12 $q12_q13 $q13_q14 $q14_q15 $q0_q1_q2_q3 $q1_q2_q3_q4 $q2_q3_q4_q5 $q3_q4_q5_q6 $q5_q6_q7_q8 $q6_q7_q8_q9 $q7_q8_q9_q10 $q8_q9_q10_q11 $q9_q10_q11_q12 $q10_q11_q12_q13 $q11_q12_q13_q14 $q12_q13_q14_q15 $r0_r1 $r2_r3 $r12_sp $d0_d1_d2 $d1_d2_d3 $d2_d3_d4 $d3_d4_d5 $d4_d5_d6 $d5_d6_d7 $d6_d7_d8 $d7_d8_d9 $d14_d15_d16 $d15_d16_d17 $d16_d17_d18 $d17_d18_d19 $d18_d19_d20 $d19_d20_d21 $d20_d21_d22 $d21_d22_d23 $d22_d23_d24 $d23_d24_d25 $d24_d25_d26 $d25_d26_d27 $d26_d27_d28 $d27_d28_d29 $d28_d29_d30 $d29_d30_d31 $d0_d2_d4 $d1_d3_d5 $d2_d4_d6 $d3_d5_d7 $d4_d6_d8 $d5_d7_d9 $d6_d8_d10 $d7_d9_d11 $d12_d14_d16 $d13_d15_d17 $d14_d16_d18 $d15_d17_d19 $d16_d18_d20 $d17_d19_d21 $d18_d20_d22 $d19_d21_d23 $d20_d22_d24 $d21_d23_d25 $d22_d24_d26 $d23_d25_d27 $d24_d26_d28 $d25_d27_d29 $d26_d28_d30 $d27_d29_d31 $d0_d2_d4_d6 $d1_d3_d5_d7 $d2_d4_d6_d8 $d3_d5_d7_d9 $d4_d6_d8_d10 $d5_d7_d9_d11 $d6_d8_d10_d12 $d7_d9_d11_d13 $d10_d12_d14_d16 $d11_d13_d15_d17 $d12_d14_d16_d18 $d13_d15_d17_d19 $d14_d16_d18_d20 $d15_d17_d19_d21 $d16_d18_d20_d22 $d17_d19_d21_d23 $d18_d20_d22_d24 $d19_d21_d23_d25 $d20_d22_d24_d26 $d21_d23_d25_d27 $d22_d24_d26_d28 $d23_d25_d27_d29 $d24_d26_d28_d30 $d25_d27_d29_d31 $d1_d2 $d3_d4 $d5_d6 $d7_d8 $d15_d16 $d17_d18 $d19_d20 $d21_d22 $d23_d24 $d25_d26 $d27_d28 $d29_d30 $d1_d2_d3_d4 $d3_d4_d5_d6 $d5_d6_d7_d8 $d7_d8_d9_d10 $d13_d14_d15_d16 $d15_d16_d17_d18 $d17_d18_d19_d20 $d19_d20_d21_d22 $d21_d22_d23_d24 $d23_d24_d25_d26 $d25_d26_d27_d28 $d27_d28_d29_d30
  call void @bar1()
  call void @bar2()
  ret void
}

@llvm.used = appending global [1 x ptr] [ptr @foo]

declare void @bar2()
attributes #0 = {nounwind}
