; RUN: llc -mtriple=i686-- -mcpu pentium4 < %s | FileCheck %s -check-prefix=SSE
; RUN: llc -mtriple=i686-- -mcpu pentium3 < %s | FileCheck %s -check-prefix=NOSSE2
; RUN: llc -mtriple=i686-- -mcpu pentium2 < %s | FileCheck %s -check-prefix=NOSSE1
; RUN: llc -mtriple=i686-- -mcpu pentium < %s | FileCheck %s -check-prefix=NOCMOV
; PR14035

define double @test1(i32 %a, i32 %b, double %x) nounwind {
  %cmp = icmp ugt i32 %a, %b
  %sel = select i1 %cmp, double 99.0, double %x
  ret double %sel

; SSE-LABEL: test1:
; SSE: movsd

; NOSSE2-LABEL: test1:
; NOSSE2: fcmovnbe

; NOSSE1-LABEL: test1:
; NOSSE1: fcmovnbe

; NOCMOV-LABEL: test1:
; NOCMOV: fstp

}

define double @test2(i32 %a, i32 %b, double %x) nounwind {
  %cmp = icmp uge i32 %a, %b
  %sel = select i1 %cmp, double 99.0, double %x
  ret double %sel

; SSE-LABEL: test2:
; SSE: movsd

; NOSSE2-LABEL: test2:
; NOSSE2: fcmovnb

; NOSSE1-LABEL: test2:
; NOSSE1: fcmovnb

; NOCMOV-LABEL: test2:
; NOCMOV: fstp
}

define double @test3(i32 %a, i32 %b, double %x) nounwind {
  %cmp = icmp ult i32 %a, %b
  %sel = select i1 %cmp, double 99.0, double %x
  ret double %sel

; SSE-LABEL: test3:
; SSE: movsd

; NOSSE2-LABEL: test3:
; NOSSE2: fcmovb

; NOSSE1-LABEL: test3:
; NOSSE1: fcmovb

; NOCMOV-LABEL: test3:
; NOCMOV: fstp
}

define double @test4(i32 %a, i32 %b, double %x) nounwind {
  %cmp = icmp ule i32 %a, %b
  %sel = select i1 %cmp, double 99.0, double %x
  ret double %sel

; SSE-LABEL: test4:
; SSE: movsd

; NOSSE2-LABEL: test4:
; NOSSE2: fcmovbe

; NOSSE1-LABEL: test4:
; NOSSE1: fcmovbe

; NOCMOV-LABEL: test4:
; NOCMOV: fstp
}

define double @test5(i32 %a, i32 %b, double %x) nounwind {
  %cmp = icmp sgt i32 %a, %b
  %sel = select i1 %cmp, double 99.0, double %x
  ret double %sel

; SSE-LABEL: test5:
; SSE: movsd

; NOSSE2-LABEL: test5:
; NOSSE2: fstp

; NOSSE1-LABEL: test5:
; NOSSE1: fstp

; NOCMOV-LABEL: test5:
; NOCMOV: fstp
}

define double @test6(i32 %a, i32 %b, double %x) nounwind {
  %cmp = icmp sge i32 %a, %b
  %sel = select i1 %cmp, double 99.0, double %x
  ret double %sel

; SSE-LABEL: test6:
; SSE: movsd

; NOSSE2-LABEL: test6:
; NOSSE2: fstp

; NOSSE1-LABEL: test6:
; NOSSE1: fstp

; NOCMOV-LABEL: test6:
; NOCMOV: fstp
}

define double @test7(i32 %a, i32 %b, double %x) nounwind {
  %cmp = icmp slt i32 %a, %b
  %sel = select i1 %cmp, double 99.0, double %x
  ret double %sel

; SSE-LABEL: test7:
; SSE: movsd

; NOSSE2-LABEL: test7:
; NOSSE2: fstp

; NOSSE1-LABEL: test7:
; NOSSE1: fstp

; NOCMOV-LABEL: test7:
; NOCMOV: fstp
}

define double @test8(i32 %a, i32 %b, double %x) nounwind {
  %cmp = icmp sle i32 %a, %b
  %sel = select i1 %cmp, double 99.0, double %x
  ret double %sel

; SSE-LABEL: test8:
; SSE: movsd

; NOSSE2-LABEL: test8:
; NOSSE2: fstp

; NOSSE1-LABEL: test8:
; NOSSE1: fstp

; NOCMOV-LABEL: test8:
; NOCMOV: fstp
}

define float @test9(i32 %a, i32 %b, float %x) nounwind {
  %cmp = icmp ugt i32 %a, %b
  %sel = select i1 %cmp, float 99.0, float %x
  ret float %sel

; SSE-LABEL: test9:
; SSE: movss

; NOSSE2-LABEL: test9:
; NOSSE2: movss

; NOSSE1-LABEL: test9:
; NOSSE1: fcmovnbe

; NOCMOV-LABEL: test9:
; NOCMOV: fstp
}

define float @test10(i32 %a, i32 %b, float %x) nounwind {
  %cmp = icmp uge i32 %a, %b
  %sel = select i1 %cmp, float 99.0, float %x
  ret float %sel

; SSE-LABEL: test10:
; SSE: movss

; NOSSE2-LABEL: test10:
; NOSSE2: movss

; NOSSE1-LABEL: test10:
; NOSSE1: fcmovnb

; NOCMOV-LABEL: test10:
; NOCMOV: fstp
}

define float @test11(i32 %a, i32 %b, float %x) nounwind {
  %cmp = icmp ult i32 %a, %b
  %sel = select i1 %cmp, float 99.0, float %x
  ret float %sel

; SSE-LABEL: test11:
; SSE: movss

; NOSSE2-LABEL: test11:
; NOSSE2: movss

; NOSSE1-LABEL: test11:
; NOSSE1: fcmovb

; NOCMOV-LABEL: test11:
; NOCMOV: fstp
}

define float @test12(i32 %a, i32 %b, float %x) nounwind {
  %cmp = icmp ule i32 %a, %b
  %sel = select i1 %cmp, float 99.0, float %x
  ret float %sel

; SSE-LABEL: test12:
; SSE: movss

; NOSSE2-LABEL: test12:
; NOSSE2: movss

; NOSSE1-LABEL: test12:
; NOSSE1: fcmovbe

; NOCMOV-LABEL: test12:
; NOCMOV: fstp
}

define float @test13(i32 %a, i32 %b, float %x) nounwind {
  %cmp = icmp sgt i32 %a, %b
  %sel = select i1 %cmp, float 99.0, float %x
  ret float %sel

; SSE-LABEL: test13:
; SSE: movss

; NOSSE2-LABEL: test13:
; NOSSE2: movss

; NOSSE1-LABEL: test13:
; NOSSE1: fstp

; NOCMOV-LABEL: test13:
; NOCMOV: fstp
}

define float @test14(i32 %a, i32 %b, float %x) nounwind {
  %cmp = icmp sge i32 %a, %b
  %sel = select i1 %cmp, float 99.0, float %x
  ret float %sel

; SSE-LABEL: test14:
; SSE: movss

; NOSSE2-LABEL: test14:
; NOSSE2: movss

; NOSSE1-LABEL: test14:
; NOSSE1: fstp

; NOCMOV-LABEL: test14:
; NOCMOV: fstp
}

define float @test15(i32 %a, i32 %b, float %x) nounwind {
  %cmp = icmp slt i32 %a, %b
  %sel = select i1 %cmp, float 99.0, float %x
  ret float %sel

; SSE-LABEL: test15:
; SSE: movss

; NOSSE2-LABEL: test15:
; NOSSE2: movss

; NOSSE1-LABEL: test15:
; NOSSE1: fstp

; NOCMOV-LABEL: test15:
; NOCMOV: fstp
}

define float @test16(i32 %a, i32 %b, float %x) nounwind {
  %cmp = icmp sle i32 %a, %b
  %sel = select i1 %cmp, float 99.0, float %x
  ret float %sel

; SSE-LABEL: test16:
; SSE: movss

; NOSSE2-LABEL: test16:
; NOSSE2: movss

; NOSSE1-LABEL: test16:
; NOSSE1: fstp

; NOCMOV-LABEL: test16:
; NOCMOV: fstp
}

define x86_fp80 @test17(i32 %a, i32 %b, x86_fp80 %x) nounwind {
  %cmp = icmp ugt i32 %a, %b
  %sel = select i1 %cmp, x86_fp80 0xK4005C600000000000000, x86_fp80 %x
  ret x86_fp80 %sel

; SSE-LABEL: test17:
; SSE: fcmovnbe

; NOSSE2-LABEL: test17:
; NOSSE2: fcmovnbe

; NOSSE1-LABEL: test17:
; NOSSE1: fcmovnbe

; NOCMOV-LABEL: test17:
; NOCMOV: fstp
}

define x86_fp80 @test18(i32 %a, i32 %b, x86_fp80 %x) nounwind {
  %cmp = icmp uge i32 %a, %b
  %sel = select i1 %cmp, x86_fp80 0xK4005C600000000000000, x86_fp80 %x
  ret x86_fp80 %sel

; SSE-LABEL: test18:
; SSE: fcmovnb

; NOSSE2-LABEL: test18:
; NOSSE2: fcmovnb

; NOSSE1-LABEL: test18:
; NOSSE1: fcmovnb

; NOCMOV-LABEL: test18:
; NOCMOV: fstp
}

define x86_fp80 @test19(i32 %a, i32 %b, x86_fp80 %x) nounwind {
  %cmp = icmp ult i32 %a, %b
  %sel = select i1 %cmp, x86_fp80 0xK4005C600000000000000, x86_fp80 %x
  ret x86_fp80 %sel

; SSE-LABEL: test19:
; SSE: fcmovb

; NOSSE2-LABEL: test19:
; NOSSE2: fcmovb

; NOSSE1-LABEL: test19:
; NOSSE1: fcmovb

; NOCMOV-LABEL: test19:
; NOCMOV: fstp
}

define x86_fp80 @test20(i32 %a, i32 %b, x86_fp80 %x) nounwind {
  %cmp = icmp ule i32 %a, %b
  %sel = select i1 %cmp, x86_fp80 0xK4005C600000000000000, x86_fp80 %x
  ret x86_fp80 %sel

; SSE-LABEL: test20:
; SSE: fcmovbe

; NOSSE2-LABEL: test20:
; NOSSE2: fcmovbe

; NOSSE1-LABEL: test20:
; NOSSE1: fcmovbe

; NOCMOV-LABEL: test20:
; NOCMOV: fstp
}

define x86_fp80 @test21(i32 %a, i32 %b, x86_fp80 %x) nounwind {
  %cmp = icmp sgt i32 %a, %b
  %sel = select i1 %cmp, x86_fp80 0xK4005C600000000000000, x86_fp80 %x
  ret x86_fp80 %sel

; We don't emit a branch for fp80, why?
; SSE-LABEL: test21:
; SSE: testb
; SSE: fcmovne

; NOSSE2-LABEL: test21:
; NOSSE2: testb
; NOSSE2: fcmovne

; NOSSE1-LABEL: test21:
; NOSSE1: testb
; NOSSE1: fcmovne

; NOCMOV-LABEL: test21:
; NOCMOV: fstp
}

define x86_fp80 @test22(i32 %a, i32 %b, x86_fp80 %x) nounwind {
  %cmp = icmp sge i32 %a, %b
  %sel = select i1 %cmp, x86_fp80 0xK4005C600000000000000, x86_fp80 %x
  ret x86_fp80 %sel

; SSE-LABEL: test22:
; SSE: testb
; SSE: fcmovne

; NOSSE2-LABEL: test22:
; NOSSE2: testb
; NOSSE2: fcmovne

; NOSSE1-LABEL: test22:
; NOSSE1: testb
; NOSSE1: fcmovne

; NOCMOV-LABEL: test22:
; NOCMOV: fstp
}

define x86_fp80 @test23(i32 %a, i32 %b, x86_fp80 %x) nounwind {
  %cmp = icmp slt i32 %a, %b
  %sel = select i1 %cmp, x86_fp80 0xK4005C600000000000000, x86_fp80 %x
  ret x86_fp80 %sel

; SSE-LABEL: test23:
; SSE: testb
; SSE: fcmovne

; NOSSE2-LABEL: test23:
; NOSSE2: testb
; NOSSE2: fcmovne

; NOSSE1-LABEL: test23:
; NOSSE1: testb
; NOSSE1: fcmovne

; NOCMOV-LABEL: test23:
; NOCMOV: fstp
}

define x86_fp80 @test24(i32 %a, i32 %b, x86_fp80 %x) nounwind {
  %cmp = icmp sle i32 %a, %b
  %sel = select i1 %cmp, x86_fp80 0xK4005C600000000000000, x86_fp80 %x
  ret x86_fp80 %sel

; SSE-LABEL: test24:
; SSE: testb
; SSE: fcmovne

; NOSSE2-LABEL: test24:
; NOSSE2: testb
; NOSSE2: fcmovne

; NOSSE1-LABEL: test24:
; NOSSE1: testb
; NOSSE1: fcmovne

; NOCMOV-LABEL: test24:
; NOCMOV: fstp
}
