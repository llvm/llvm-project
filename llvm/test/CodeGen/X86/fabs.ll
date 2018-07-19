; Make sure this testcase codegens to the fabs instruction, not a call to fabsf
; RUN: llc < %s -mtriple=i686-apple-macosx -mattr=-sse2,-sse3,-sse | FileCheck %s
; RUN: llc < %s -mtriple=i686-apple-macosx -mattr=-sse,-sse2,-sse3 -enable-unsafe-fp-math -enable-no-nans-fp-math | FileCheck %s --check-prefix=UNSAFE
; RUN: llc < %s -mtriple=x86_64-apple-macosx -O0 | FileCheck %s --check-prefix=NOOPT

declare float @fabsf(float)

declare x86_fp80 @fabsl(x86_fp80)

; CHECK-LABEL:  test1:
; UNSAFE-LABEL: test1:
; NOOPT-LABEL:  test1:
define float @test1(float %X) {
        %Y = call float @fabsf(float %X) readnone
        ret float %Y
}
; CHECK:  {{^[ \t]+fabs$}}
; UNSAFE: {{^[ \t]+fabs$}}

; CHECK-NOT:  fabs
; UNSAFE-NOT: fabs
; NOOPT-NOT:  fabsf

; CHECK-LABEL:  test2:
; UNSAFE-LABEL: test2:
; NOOPT-LABEL:  test2:
define double @test2(double %X) {
        %Y = fcmp oge double %X, -0.0
        %Z = fsub double -0.0, %X
        %Q = select i1 %Y, double %X, double %Z
        ret double %Q
}
; fabs is not used here.
; CHECK-NOT:  fabs
; NOOPT-NOT:  fabs

; UNSAFE: {{^[ \t]+fabs$}}

; UNSAFE-NOT: fabs

; CHECK-LABEL:  test3:
; UNSAFE-LABEL: test3:
; NOOPT-LABEL:  test3:
define x86_fp80 @test3(x86_fp80 %X) {
        %Y = call x86_fp80 @fabsl(x86_fp80 %X) readnone
        ret x86_fp80 %Y
}
; CHECK:  {{^[ \t]+fabs$}}
; UNSAFE: {{^[ \t]+fabs$}}
; NOOPT:  {{^[ \t]+fabs$}}

; CHECK-NOT:  fabs
; UNSAFE-NOT: fabs
; NOOPT-NOT:  fabs
