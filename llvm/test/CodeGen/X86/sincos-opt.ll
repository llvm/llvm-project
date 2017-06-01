; RUN: llc < %s -mtriple=x86_64-apple-macosx10.9.0 -mcpu=core2 | FileCheck %s --check-prefix=OSX_SINCOS
; RUN: llc < %s -mtriple=x86_64-apple-macosx10.8.0 -mcpu=core2 | FileCheck %s --check-prefix=OSX_NOOPT
; RUN: llc < %s -mtriple=x86_64-pc-linux-gnu -mcpu=core2 | FileCheck %s --check-prefix=GNU_NOOPT
; RUN: llc < %s -mtriple=x86_64-pc-linux-gnu -mcpu=core2 -enable-unsafe-fp-math | FileCheck %s --check-prefix=GNU_SINCOS
; RUN: llc < %s -mtriple=x86_64-pc-linux-gnux32 -mcpu=core2 -enable-unsafe-fp-math | FileCheck %s --check-prefix=GNUX32_SINCOS

; Combine sin / cos into a single call.
; rdar://13087969
; rdar://13599493

define float @test1(float %x) nounwind {
entry:
; GNU_SINCOS-LABEL: test1:
; GNU_SINCOS: callq sincosf
; GNU_SINCOS: movss 4(%rsp), %xmm0
; GNU_SINCOS: addss (%rsp), %xmm0

; GNUX32_SINCOS-LABEL: test1:
; GNUX32_SINCOS: callq sincosf
; GNUX32_SINCOS: movss 4(%esp), %xmm0
; GNUX32_SINCOS: addss (%esp), %xmm0

; GNU_NOOPT: test1
; GNU_NOOPT: callq sinf
; GNU_NOOPT: callq cosf

; OSX_SINCOS-LABEL: test1:
; OSX_SINCOS: callq ___sincosf_stret
; OSX_SINCOS: movshdup {{.*}} xmm1 = xmm0[1,1,3,3]
; OSX_SINCOS: addss %xmm1, %xmm0

; OSX_NOOPT: test1
; OSX_NOOPT: callq _sinf
; OSX_NOOPT: callq _cosf
  %call = tail call float @sinf(float %x) nounwind readnone
  %call1 = tail call float @cosf(float %x) nounwind readnone
  %add = fadd float %call, %call1
  ret float %add
}

define double @test2(double %x) nounwind {
entry:
; GNU_SINCOS-LABEL: test2:
; GNU_SINCOS: callq sincos
; GNU_SINCOS: movsd 16(%rsp), %xmm0
; GNU_SINCOS: addsd 8(%rsp), %xmm0

; GNUX32_SINCOS-LABEL: test2:
; GNUX32_SINCOS: callq sincos
; GNUX32_SINCOS: movsd 16(%esp), %xmm0
; GNUX32_SINCOS: addsd 8(%esp), %xmm0

; GNU_NOOPT: test2:
; GNU_NOOPT: callq sin
; GNU_NOOPT: callq cos

; OSX_SINCOS-LABEL: test2:
; OSX_SINCOS: callq ___sincos_stret
; OSX_SINCOS: addsd %xmm1, %xmm0

; OSX_NOOPT: test2
; OSX_NOOPT: callq _sin
; OSX_NOOPT: callq _cos
  %call = tail call double @sin(double %x) nounwind readnone
  %call1 = tail call double @cos(double %x) nounwind readnone
  %add = fadd double %call, %call1
  ret double %add
}

define x86_fp80 @test3(x86_fp80 %x) nounwind {
entry:
; GNU_SINCOS-LABEL: test3:
; GNU_SINCOS: callq sinl
; GNU_SINCOS: callq cosl
; GNU_SINCOS: ret

; GNUX32_SINCOS-LABEL: test3:
; GNUX32_SINCOS: callq sinl
; GNUX32_SINCOS: callq cosl
; GNUX32_SINCOS: ret

; GNU_NOOPT: test3:
; GNU_NOOPT: callq sinl
; GNU_NOOPT: callq cosl

  %call = tail call x86_fp80 @sinl(x86_fp80 %x) nounwind
  %call1 = tail call x86_fp80 @cosl(x86_fp80 %x) nounwind
  %add = fadd x86_fp80 %call, %call1
  ret x86_fp80 %add
}

declare float  @sinf(float) readonly
declare double @sin(double) readonly
declare float @cosf(float) readonly
declare double @cos(double) readonly

declare x86_fp80 @sinl(x86_fp80)
declare x86_fp80 @cosl(x86_fp80)
