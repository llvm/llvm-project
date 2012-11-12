; RUN: llc -march x86 -mcpu pentium4 < %s | FileCheck %s -check-prefix=SSE
; RUN: llc -march x86 -mcpu pentium3 < %s | FileCheck %s -check-prefix=NOSSE2
; RUN: llc -march x86 -mcpu pentium2 < %s | FileCheck %s -check-prefix=NOSSE1
; RUN: llc -march x86 -mcpu pentium < %s | FileCheck %s -check-prefix=NOCMOV
; PR14035

define double @test1(i32 %a, i32 %b, double %x) nounwind {
  %cmp = icmp ugt i32 %a, %b
  %sel = select i1 %cmp, double 99.0, double %x
  ret double %sel

; SSE: test1:
; SSE: movsd

; NOSSE2: test1:
; NOSSE2: fcmovnbe

; NOSSE1: test1:
; NOSSE1: fcmovnbe

; NOCMOV: test1:
; NOCMOV: fstp

}

define double @test2(i32 %a, i32 %b, double %x) nounwind {
  %cmp = icmp uge i32 %a, %b
  %sel = select i1 %cmp, double 99.0, double %x
  ret double %sel

; SSE: test2:
; SSE: movsd

; NOSSE2: test2:
; NOSSE2: fcmovnb

; NOSSE1: test2:
; NOSSE1: fcmovnb

; NOCMOV: test2:
; NOCMOV: fstp
}

define double @test3(i32 %a, i32 %b, double %x) nounwind {
  %cmp = icmp ult i32 %a, %b
  %sel = select i1 %cmp, double 99.0, double %x
  ret double %sel

; SSE: test3:
; SSE: movsd

; NOSSE2: test3:
; NOSSE2: fcmovb

; NOSSE1: test3:
; NOSSE1: fcmovb

; NOCMOV: test3:
; NOCMOV: fstp
}

define double @test4(i32 %a, i32 %b, double %x) nounwind {
  %cmp = icmp ule i32 %a, %b
  %sel = select i1 %cmp, double 99.0, double %x
  ret double %sel

; SSE: test4:
; SSE: movsd

; NOSSE2: test4:
; NOSSE2: fcmovbe

; NOSSE1: test4:
; NOSSE1: fcmovbe

; NOCMOV: test4:
; NOCMOV: fstp
}

define double @test5(i32 %a, i32 %b, double %x) nounwind {
  %cmp = icmp sgt i32 %a, %b
  %sel = select i1 %cmp, double 99.0, double %x
  ret double %sel

; SSE: test5:
; SSE: movsd

; NOSSE2: test5:
; NOSSE2: fstp

; NOSSE1: test5:
; NOSSE1: fstp

; NOCMOV: test5:
; NOCMOV: fstp
}

define double @test6(i32 %a, i32 %b, double %x) nounwind {
  %cmp = icmp sge i32 %a, %b
  %sel = select i1 %cmp, double 99.0, double %x
  ret double %sel

; SSE: test6:
; SSE: movsd

; NOSSE2: test6:
; NOSSE2: fstp

; NOSSE1: test6:
; NOSSE1: fstp

; NOCMOV: test6:
; NOCMOV: fstp
}

define double @test7(i32 %a, i32 %b, double %x) nounwind {
  %cmp = icmp slt i32 %a, %b
  %sel = select i1 %cmp, double 99.0, double %x
  ret double %sel

; SSE: test7:
; SSE: movsd

; NOSSE2: test7:
; NOSSE2: fstp

; NOSSE1: test7:
; NOSSE1: fstp

; NOCMOV: test7:
; NOCMOV: fstp
}

define double @test8(i32 %a, i32 %b, double %x) nounwind {
  %cmp = icmp sle i32 %a, %b
  %sel = select i1 %cmp, double 99.0, double %x
  ret double %sel

; SSE: test8:
; SSE: movsd

; NOSSE2: test8:
; NOSSE2: fstp

; NOSSE1: test8:
; NOSSE1: fstp

; NOCMOV: test8:
; NOCMOV: fstp
}

define float @test9(i32 %a, i32 %b, float %x) nounwind {
  %cmp = icmp ugt i32 %a, %b
  %sel = select i1 %cmp, float 99.0, float %x
  ret float %sel

; SSE: test9:
; SSE: movss

; NOSSE2: test9:
; NOSSE2: movss

; NOSSE1: test9:
; NOSSE1: fcmovnbe

; NOCMOV: test9:
; NOCMOV: fstp
}

define float @test10(i32 %a, i32 %b, float %x) nounwind {
  %cmp = icmp uge i32 %a, %b
  %sel = select i1 %cmp, float 99.0, float %x
  ret float %sel

; SSE: test10:
; SSE: movss

; NOSSE2: test10:
; NOSSE2: movss

; NOSSE1: test10:
; NOSSE1: fcmovnb

; NOCMOV: test10:
; NOCMOV: fstp
}

define float @test11(i32 %a, i32 %b, float %x) nounwind {
  %cmp = icmp ult i32 %a, %b
  %sel = select i1 %cmp, float 99.0, float %x
  ret float %sel

; SSE: test11:
; SSE: movss

; NOSSE2: test11:
; NOSSE2: movss

; NOSSE1: test11:
; NOSSE1: fcmovb

; NOCMOV: test11:
; NOCMOV: fstp
}

define float @test12(i32 %a, i32 %b, float %x) nounwind {
  %cmp = icmp ule i32 %a, %b
  %sel = select i1 %cmp, float 99.0, float %x
  ret float %sel

; SSE: test12:
; SSE: movss

; NOSSE2: test12:
; NOSSE2: movss

; NOSSE1: test12:
; NOSSE1: fcmovbe

; NOCMOV: test12:
; NOCMOV: fstp
}

define float @test13(i32 %a, i32 %b, float %x) nounwind {
  %cmp = icmp sgt i32 %a, %b
  %sel = select i1 %cmp, float 99.0, float %x
  ret float %sel

; SSE: test13:
; SSE: movss

; NOSSE2: test13:
; NOSSE2: movss

; NOSSE1: test13:
; NOSSE1: fstp

; NOCMOV: test13:
; NOCMOV: fstp
}

define float @test14(i32 %a, i32 %b, float %x) nounwind {
  %cmp = icmp sge i32 %a, %b
  %sel = select i1 %cmp, float 99.0, float %x
  ret float %sel

; SSE: test14:
; SSE: movss

; NOSSE2: test14:
; NOSSE2: movss

; NOSSE1: test14:
; NOSSE1: fstp

; NOCMOV: test14:
; NOCMOV: fstp
}

define float @test15(i32 %a, i32 %b, float %x) nounwind {
  %cmp = icmp slt i32 %a, %b
  %sel = select i1 %cmp, float 99.0, float %x
  ret float %sel

; SSE: test15:
; SSE: movss

; NOSSE2: test15:
; NOSSE2: movss

; NOSSE1: test15:
; NOSSE1: fstp

; NOCMOV: test15:
; NOCMOV: fstp
}

define float @test16(i32 %a, i32 %b, float %x) nounwind {
  %cmp = icmp sle i32 %a, %b
  %sel = select i1 %cmp, float 99.0, float %x
  ret float %sel

; SSE: test16:
; SSE: movss

; NOSSE2: test16:
; NOSSE2: movss

; NOSSE1: test16:
; NOSSE1: fstp

; NOCMOV: test16:
; NOCMOV: fstp
}

define x86_fp80 @test17(i32 %a, i32 %b, x86_fp80 %x) nounwind {
  %cmp = icmp ugt i32 %a, %b
  %sel = select i1 %cmp, x86_fp80 0xK4005C600000000000000, x86_fp80 %x
  ret x86_fp80 %sel

; SSE: test17:
; SSE: fcmovnbe

; NOSSE2: test17:
; NOSSE2: fcmovnbe

; NOSSE1: test17:
; NOSSE1: fcmovnbe

; NOCMOV: test17:
; NOCMOV: fstp
}

define x86_fp80 @test18(i32 %a, i32 %b, x86_fp80 %x) nounwind {
  %cmp = icmp uge i32 %a, %b
  %sel = select i1 %cmp, x86_fp80 0xK4005C600000000000000, x86_fp80 %x
  ret x86_fp80 %sel

; SSE: test18:
; SSE: fcmovnb

; NOSSE2: test18:
; NOSSE2: fcmovnb

; NOSSE1: test18:
; NOSSE1: fcmovnb

; NOCMOV: test18:
; NOCMOV: fstp
}

define x86_fp80 @test19(i32 %a, i32 %b, x86_fp80 %x) nounwind {
  %cmp = icmp ult i32 %a, %b
  %sel = select i1 %cmp, x86_fp80 0xK4005C600000000000000, x86_fp80 %x
  ret x86_fp80 %sel

; SSE: test19:
; SSE: fcmovb

; NOSSE2: test19:
; NOSSE2: fcmovb

; NOSSE1: test19:
; NOSSE1: fcmovb

; NOCMOV: test19:
; NOCMOV: fstp
}

define x86_fp80 @test20(i32 %a, i32 %b, x86_fp80 %x) nounwind {
  %cmp = icmp ule i32 %a, %b
  %sel = select i1 %cmp, x86_fp80 0xK4005C600000000000000, x86_fp80 %x
  ret x86_fp80 %sel

; SSE: test20:
; SSE: fcmovbe

; NOSSE2: test20:
; NOSSE2: fcmovbe

; NOSSE1: test20:
; NOSSE1: fcmovbe

; NOCMOV: test20:
; NOCMOV: fstp
}

define x86_fp80 @test21(i32 %a, i32 %b, x86_fp80 %x) nounwind {
  %cmp = icmp sgt i32 %a, %b
  %sel = select i1 %cmp, x86_fp80 0xK4005C600000000000000, x86_fp80 %x
  ret x86_fp80 %sel

; We don't emit a branch for fp80, why?
; SSE: test21:
; SSE: testb
; SSE: fcmovne

; NOSSE2: test21:
; NOSSE2: testb
; NOSSE2: fcmovne

; NOSSE1: test21:
; NOSSE1: testb
; NOSSE1: fcmovne

; NOCMOV: test21:
; NOCMOV: fstp
}

define x86_fp80 @test22(i32 %a, i32 %b, x86_fp80 %x) nounwind {
  %cmp = icmp sge i32 %a, %b
  %sel = select i1 %cmp, x86_fp80 0xK4005C600000000000000, x86_fp80 %x
  ret x86_fp80 %sel

; SSE: test22:
; SSE: testb
; SSE: fcmovne

; NOSSE2: test22:
; NOSSE2: testb
; NOSSE2: fcmovne

; NOSSE1: test22:
; NOSSE1: testb
; NOSSE1: fcmovne

; NOCMOV: test22:
; NOCMOV: fstp
}

define x86_fp80 @test23(i32 %a, i32 %b, x86_fp80 %x) nounwind {
  %cmp = icmp slt i32 %a, %b
  %sel = select i1 %cmp, x86_fp80 0xK4005C600000000000000, x86_fp80 %x
  ret x86_fp80 %sel

; SSE: test23:
; SSE: testb
; SSE: fcmovne

; NOSSE2: test23:
; NOSSE2: testb
; NOSSE2: fcmovne

; NOSSE1: test23:
; NOSSE1: testb
; NOSSE1: fcmovne

; NOCMOV: test23:
; NOCMOV: fstp
}

define x86_fp80 @test24(i32 %a, i32 %b, x86_fp80 %x) nounwind {
  %cmp = icmp sle i32 %a, %b
  %sel = select i1 %cmp, x86_fp80 0xK4005C600000000000000, x86_fp80 %x
  ret x86_fp80 %sel

; SSE: test24:
; SSE: testb
; SSE: fcmovne

; NOSSE2: test24:
; NOSSE2: testb
; NOSSE2: fcmovne

; NOSSE1: test24:
; NOSSE1: testb
; NOSSE1: fcmovne

; NOCMOV: test24:
; NOCMOV: fstp
}
