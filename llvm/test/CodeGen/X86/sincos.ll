; Make sure this testcase codegens to the sin and cos instructions, not calls
; RUN: llc < %s -mtriple=i686-apple-macosx -mattr=-sse,-sse2,-sse3 -enable-unsafe-fp-math  | FileCheck %s --check-prefix=SIN
; RUN: llc < %s -mtriple=i686-apple-macosx -mattr=-sse,-sse2,-sse3 -enable-unsafe-fp-math  | FileCheck %s --check-prefix=COS
; RUN: llc < %s -mtriple=i686-apple-macosx -mattr=-sse,-sse2,-sse3 | FileCheck %s --check-prefix=SAFE

declare float  @sinf(float) readonly

declare double @sin(double) readonly

declare x86_fp80 @sinl(x86_fp80) readonly

; SIN: test1:
define float @test1(float %X) {
        %Y = call float @sinf(float %X) readonly
        ret float %Y
}
; SIN: {{^[ \t]*fsin$}}

; SIN-NOT: fsin

; SAFE: test1
; SAFE-NOT: fsin

; SIN: test2:
define double @test2(double %X) {
        %Y = call double @sin(double %X) readonly
        ret double %Y
}
; SIN: {{^[ \t]*fsin$}}

; SIN-NOT: fsin

; SAFE: test2
; SAFE-NOT: fsin

; SIN: test3:
define x86_fp80 @test3(x86_fp80 %X) {
        %Y = call x86_fp80 @sinl(x86_fp80 %X) readonly
        ret x86_fp80 %Y
}
; SIN: {{^[ \t]*fsin$}}

; SIN-NOT: fsin
; COS-NOT: fcos
declare float @cosf(float) readonly

declare double @cos(double) readonly

declare x86_fp80 @cosl(x86_fp80) readonly


; SIN: test4:
; COS: test3:
define float @test4(float %X) {
        %Y = call float @cosf(float %X) readonly
        ret float %Y
}
; COS: {{^[ \t]*fcos}}

; SAFE: test4
; SAFE-NOT: fcos

define double @test5(double %X) {
        %Y = call double @cos(double %X) readonly
        ret double %Y
}
; COS: {{^[ \t]*fcos}}

; SAFE: test5
; SAFE-NOT: fcos

define x86_fp80 @test6(x86_fp80 %X) {
        %Y = call x86_fp80 @cosl(x86_fp80 %X) readonly
        ret x86_fp80 %Y
}
; COS: {{^[ \t]*fcos}}

; SIN-NOT: fsin
; COS-NOT: fcos
