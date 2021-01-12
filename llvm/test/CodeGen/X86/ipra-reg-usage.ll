; RUN: llc -enable-ipra -print-regusage -o /dev/null 2>&1 < %s | FileCheck %s

target triple = "x86_64-unknown-unknown"
declare void @bar1()
define preserve_allcc void @foo()#0 {
; CHECK: foo Clobbered Registers: $cs $df $ds $eflags $eip $eiz $es $esp $fpcw $fpsw $fs $gs $hip $hsp $ip $mxcsr $rip $riz $rsp $sp $sph $spl $ss $ssp $tmmcfg $bnd0 $bnd1 $bnd2 $bnd3 $cr0 $cr1 $cr2 $cr3 $cr4 $cr5 $cr6 $cr7 $cr8 $cr9 $cr10 $cr11 $cr12 $cr13 $cr14 $cr15 $dr0 $dr1 $dr2 $dr3 $dr4 $dr5 $dr6 $dr7 $dr8 $dr9 $dr10 $dr11 $dr12 $dr13 $dr14 $dr15 $fp0 $fp1 $fp2 $fp3 $fp4 $fp5 $fp6 $fp7 $k0 $k1 $k2 $k3 $k4 $k5 $k6 $k7 $mm0 $mm1 $mm2 $mm3 $mm4 $mm5 $mm6 $mm7 $r11 $st0 $st1 $st2 $st3 $st4 $st5 $st6 $st7 $tmm0 $tmm1 $tmm2 $tmm3 $tmm4 $tmm5 $tmm6 $tmm7 $xmm16 $xmm17 $xmm18 $xmm19 $xmm20 $xmm21 $xmm22 $xmm23 $xmm24 $xmm25 $xmm26 $xmm27 $xmm28 $xmm29 $xmm30 $xmm31 $ymm0 $ymm1 $ymm2 $ymm3 $ymm4 $ymm5 $ymm6 $ymm7 $ymm8 $ymm9 $ymm10 $ymm11 $ymm12 $ymm13 $ymm14 $ymm15 $ymm16 $ymm17 $ymm18 $ymm19 $ymm20 $ymm21 $ymm22 $ymm23 $ymm24 $ymm25 $ymm26 $ymm27 $ymm28 $ymm29 $ymm30 $ymm31 $zmm0 $zmm1 $zmm2 $zmm3 $zmm4 $zmm5 $zmm6 $zmm7 $zmm8 $zmm9 $zmm10 $zmm11 $zmm12 $zmm13 $zmm14 $zmm15 $zmm16 $zmm17 $zmm18 $zmm19 $zmm20 $zmm21 $zmm22 $zmm23 $zmm24 $zmm25 $zmm26 $zmm27 $zmm28 $zmm29 $zmm30 $zmm31 $r11b $r11bh $r11d $r11w $r11wh $k0_k1 $k2_k3 $k4_k5 $k6_k7
  call void @bar1()
  call void @bar2()
  ret void
}
declare void @bar2()

@llvm.used = appending global [1 x i8*] [i8* bitcast (void ()* @foo to i8*)]

attributes #0 = {nounwind}
