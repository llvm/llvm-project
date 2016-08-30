; RUN: llc < %s -march=x86 -mattr=+sse2 | FileCheck %s
; PR2656

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin9.4.0"
	%struct.anon = type <{ float, float }>
@.str = internal constant [17 x i8] c"pt: %.0f, %.0f\0A\00\00"		; <[17 x i8]*> [#uses=1]

; We can not fold either stack load into an 'xor' instruction because that
; would change what should be a 4-byte load into a 16-byte load.
; We can fold the 16-byte constant load into either 'xor' instruction,
; but we do not. It has more than one use, so it gets loaded into a register.

define void @foo(%struct.anon* byval %p) nounwind {
; CHECK-LABEL: foo:
; CHECK:         movss {{.*#+}} xmm0 = mem[0],zero,zero,zero
; CHECK-NEXT:    movss {{.*#+}} xmm1 = mem[0],zero,zero,zero
; CHECK-NEXT:    movaps {{.*#+}} xmm2 = [-0.000000e+00,-0.000000e+00,-0.000000e+00,-0.000000e+00]
; CHECK-NEXT:    xorps %xmm2, %xmm0
; CHECK-NEXT:    cvtss2sd %xmm0, %xmm0
; CHECK-NEXT:    xorps %xmm2, %xmm1
entry:
	%tmp = getelementptr %struct.anon, %struct.anon* %p, i32 0, i32 0		; <float*> [#uses=1]
	%tmp1 = load float, float* %tmp		; <float> [#uses=1]
	%tmp2 = getelementptr %struct.anon, %struct.anon* %p, i32 0, i32 1		; <float*> [#uses=1]
	%tmp3 = load float, float* %tmp2		; <float> [#uses=1]
	%neg = fsub float -0.000000e+00, %tmp1		; <float> [#uses=1]
	%conv = fpext float %neg to double		; <double> [#uses=1]
	%neg4 = fsub float -0.000000e+00, %tmp3		; <float> [#uses=1]
	%conv5 = fpext float %neg4 to double		; <double> [#uses=1]
	%call = call i32 (...) @printf( i8* getelementptr ([17 x i8], [17 x i8]* @.str, i32 0, i32 0), double %conv, double %conv5 )		; <i32> [#uses=0]
	ret void
}

declare i32 @printf(...)

; We can not fold the load from the stack into the 'and' instruction because
; that changes an 8-byte load into a 16-byte load (illegal memory access).
; We can fold the load of the constant because it is a 16-byte vector constant.

define double @PR22371(double %x) {
; CHECK-LABEL: PR22371:
; CHECK:       movsd  16(%esp), %xmm0
; CHECK-NEXT:  andpd  LCPI1_0, %xmm0
; CHECK-NEXT:  movlpd  %xmm0, (%esp)
  %call = tail call double @fabs(double %x) #0
  ret double %call
}

declare double @fabs(double) #0
attributes #0 = { readnone }

