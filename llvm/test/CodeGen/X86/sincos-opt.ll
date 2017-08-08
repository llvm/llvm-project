; RUN: llc < %s -mtriple=x86_64-apple-macosx10.9.0 -mcpu=core2 | FileCheck %s --check-prefix=OSX_SINCOS
; RUN: llc < %s -mtriple=x86_64-apple-macosx10.8.0 -mcpu=core2 | FileCheck %s --check-prefix=OSX_NOOPT
; RUN: llc < %s -mtriple=x86_64-pc-linux-gnu -mcpu=core2 | FileCheck %s --check-prefix=GNU_SINCOS
; RUN: llc < %s -mtriple=x86_64-pc-linux-gnu -mcpu=core2 -enable-unsafe-fp-math | FileCheck %s --check-prefix=GNU_SINCOS_FASTMATH
; RUN: llc < %s -mtriple=x86_64-pc-linux-gnux32 -mcpu=core2 -enable-unsafe-fp-math | FileCheck %s --check-prefix=GNU_SINCOS_FASTMATH

; Combine sin / cos into a single call unless they may write errno (as
; captured by readnone attrbiute, controlled by clang -fmath-errno
; setting).
; rdar://13087969
; rdar://13599493

define float @test1(float %x) nounwind {
entry:
; GNU_SINCOS-LABEL: test1:
; GNU_SINCOS: callq sincosf
; GNU_SINCOS: movss 4(%rsp), %xmm0
; GNU_SINCOS: addss (%rsp), %xmm0

; GNU_SINCOS_FASTMATH-LABEL: test1:
; GNU_SINCOS_FASTMATH: callq sincosf
; GNU_SINCOS_FASTMATH: movss 4(%{{[re]}}sp), %xmm0
; GNU_SINCOS_FASTMATH: addss (%{{[re]}}sp), %xmm0

; OSX_SINCOS-LABEL: test1:
; OSX_SINCOS: callq ___sincosf_stret
; OSX_SINCOS: movshdup {{.*}} xmm1 = xmm0[1,1,3,3]
; OSX_SINCOS: addss %xmm1, %xmm0

; OSX_NOOPT-LABEL: test1:
; OSX_NOOPT: callq _sinf
; OSX_NOOPT: callq _cosf
  %call = tail call float @sinf(float %x) readnone
  %call1 = tail call float @cosf(float %x) readnone
  %add = fadd float %call, %call1
  ret float %add
}

define float @test1_errno(float %x) nounwind {
entry:
; GNU_SINCOS-LABEL: test1_errno:
; GNU_SINCOS: callq sinf
; GNU_SINCOS: callq cosf

; GNU_SINCOS_FASTMATH-LABEL: test1_errno:
; GNU_SINCOS_FASTMATH: callq sinf
; GNU_SINCOS_FASTMATH: callq cosf

; OSX_SINCOS-LABEL: test1_errno:
; OSX_SINCOS: callq _sinf
; OSX_SINCOS: callq _cosf

; OSX_NOOPT-LABEL: test1_errno:
; OSX_NOOPT: callq _sinf
; OSX_NOOPT: callq _cosf
  %call = tail call float @sinf(float %x)
  %call1 = tail call float @cosf(float %x)
  %add = fadd float %call, %call1
  ret float %add
}

define double @test2(double %x) nounwind {
entry:
; GNU_SINCOS-LABEL: test2:
; GNU_SINCOS: callq sincos
; GNU_SINCOS: movsd 16(%rsp), %xmm0
; GNU_SINCOS: addsd 8(%rsp), %xmm0

; GNU_SINCOS_FASTMATH-LABEL: test2:
; GNU_SINCOS_FASTMATH: callq sincos
; GNU_SINCOS_FASTMATH: movsd 16(%{{[re]}}sp), %xmm0
; GNU_SINCOS_FASTMATH: addsd 8(%{{[re]}}sp), %xmm0

; OSX_SINCOS-LABEL: test2:
; OSX_SINCOS: callq ___sincos_stret
; OSX_SINCOS: addsd %xmm1, %xmm0

; OSX_NOOPT-LABEL: test2:
; OSX_NOOPT: callq _sin
; OSX_NOOPT: callq _cos
  %call = tail call double @sin(double %x) readnone
  %call1 = tail call double @cos(double %x) readnone
  %add = fadd double %call, %call1
  ret double %add
}

define double @test2_errno(double %x) nounwind {
entry:
; GNU_SINCOS-LABEL: test2_errno:
; GNU_SINCOS: callq sin
; GNU_SINCOS: callq cos

; GNU_SINCOS_FASTMATH-LABEL: test2_errno:
; GNU_SINCOS_FASTMATH: callq sin
; GNU_SINCOS_FASTMATH: callq cos

; OSX_SINCOS-LABEL: test2_errno:
; OSX_SINCOS: callq _sin
; OSX_SINCOS: callq _cos

; OSX_NOOPT-LABEL: test2_errno:
; OSX_NOOPT: callq _sin
; OSX_NOOPT: callq _cos
  %call = tail call double @sin(double %x)
  %call1 = tail call double @cos(double %x)
  %add = fadd double %call, %call1
  ret double %add
}

define x86_fp80 @test3(x86_fp80 %x) nounwind {
entry:
; GNU_SINCOS-LABEL: test3:
; GNU_SINCOS: callq sincosl
; GNU_SINCOS: fldt 16(%rsp)
; GNU_SINCOS: fldt 32(%rsp)
; GNU_SINCOS: faddp %st(1)

; GNU_SINCOS_FASTMATH-LABEL: test3:
; GNU_SINCOS_FASTMATH: fsin
; GNU_SINCOS_FASTMATH: fcos
; GNU_SINCOS_FASTMATH: faddp %st(1)
; GNU_SINCOS_FASTMATH: ret
  %call = tail call x86_fp80 @sinl(x86_fp80 %x) readnone
  %call1 = tail call x86_fp80 @cosl(x86_fp80 %x) readnone
  %add = fadd x86_fp80 %call, %call1
  ret x86_fp80 %add
}

define x86_fp80 @test3_errno(x86_fp80 %x) nounwind {
entry:
; GNU_SINCOS-LABEL: test3_errno:
; GNU_SINCOS: callq sinl
; GNU_SINCOS: callq cosl

; GNU_SINCOS_FASTMATH-LABEL: test3_errno:
; GNU_SINCOS_FASTMATH: callq sinl
; GNU_SINCOS_FASTMATH: callq cosl
  %call = tail call x86_fp80 @sinl(x86_fp80 %x)
  %call1 = tail call x86_fp80 @cosl(x86_fp80 %x)
  %add = fadd x86_fp80 %call, %call1
  ret x86_fp80 %add
}

declare float  @sinf(float)
declare double @sin(double)
declare float @cosf(float)
declare double @cos(double)
declare x86_fp80 @sinl(x86_fp80)
declare x86_fp80 @cosl(x86_fp80)
