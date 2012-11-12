; RUN: llc < %s -relocation-model=static -realign-stack=1 -mcpu=yonah | FileCheck %s

; The double argument is at 4(esp) which is 16-byte aligned, allowing us to
; fold the load into the andpd.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin8"
@G = external global double

define void @test({ double, double }* byval  %z, double* %P) nounwind {
entry:
	%tmp3 = load double* @G, align 16		; <double> [#uses=1]
	%tmp4 = tail call double @fabs( double %tmp3 ) readnone	; <double> [#uses=1]
        store volatile double %tmp4, double* %P
	%tmp = getelementptr { double, double }* %z, i32 0, i32 0		; <double*> [#uses=1]
	%tmp1 = load volatile double* %tmp, align 8		; <double> [#uses=1]
	%tmp2 = tail call double @fabs( double %tmp1 ) readnone	; <double> [#uses=1]
    ; CHECK: andpd{{.*}}4(%esp), %xmm
	%tmp6 = fadd double %tmp4, %tmp2		; <double> [#uses=1]
	store volatile double %tmp6, double* %P, align 8
	ret void
}

define void @test2() alignstack(16) nounwind {
entry:
    ; CHECK: andl{{.*}}$-16, %esp
    ret void
}

; Use a call to force a spill.
define <2 x double> @test3(<2 x double> %x, <2 x double> %y) alignstack(32) nounwind {
entry:
    ; CHECK: andl{{.*}}$-32, %esp
    call void @test2()
    %A = fmul <2 x double> %x, %y
    ret <2 x double> %A
}

declare double @fabs(double)

; The pointer is already known aligned, so and x,-16 is eliminable.
define i32 @test4() nounwind {
entry:
  %buffer = alloca [2048 x i8], align 16
  %0 = ptrtoint [2048 x i8]* %buffer to i32
  %and = and i32 %0, -16
  ret i32 %and
; CHECK: test4:
; CHECK-NOT: and
; CHECK: ret
}
