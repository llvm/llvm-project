; RUN: llc < %s -mtriple=i386-apple-darwin10 -mcpu=penryn | FileCheck %s

; rdar://7475489

define i32 @test1(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK-LABEL: test1:
; CHECK: xorb
; CHECK-NOT: andb
; CHECK-NOT: shrb
; CHECK: testb $64
  %0 = and i32 %a, 16384
  %1 = icmp ne i32 %0, 0
  %2 = and i32 %b, 16384
  %3 = icmp ne i32 %2, 0
  %4 = xor i1 %1, %3
  br i1 %4, label %bb1, label %bb

bb:                                               ; preds = %entry
  %5 = tail call i32 (...) @foo() nounwind       ; <i32> [#uses=1]
  ret i32 %5

bb1:                                              ; preds = %entry
  %6 = tail call i32 (...) @bar() nounwind       ; <i32> [#uses=1]
  ret i32 %6
}

declare i32 @foo(...)

declare i32 @bar(...)


; <rdar://problem/7598384>:
;
;    jCC  L1
;    jmp  L2
; L1:
;   ...
; L2:
;   ...
;
; to:
; 
;    jnCC L2
; L1:
;   ...
; L2:
;   ...
define float @test4(float %x, float %y) nounwind readnone optsize ssp {
entry:
  %0 = fpext float %x to double                   ; <double> [#uses=1]
  %1 = fpext float %y to double                   ; <double> [#uses=1]
  %2 = fmul double %0, %1                         ; <double> [#uses=3]
  %3 = fcmp oeq double %2, 0.000000e+00           ; <i1> [#uses=1]
  br i1 %3, label %bb2, label %bb1

; CHECK:      jne
; CHECK-NEXT: jnp
; CHECK-NOT:  jmp
; CHECK:      LBB

bb1:                                              ; preds = %entry
  %4 = fadd double %2, -1.000000e+00              ; <double> [#uses=1]
  br label %bb2

bb2:                                              ; preds = %entry, %bb1
  %.0.in = phi double [ %4, %bb1 ], [ %2, %entry ] ; <double> [#uses=1]
  %.0 = fptrunc double %.0.in to float            ; <float> [#uses=1]
  ret float %.0
}

declare i32 @llvm.x86.sse41.ptestz(<4 x float> %p1, <4 x float> %p2) nounwind
declare i32 @llvm.x86.sse41.ptestc(<4 x float> %p1, <4 x float> %p2) nounwind

define <4 x float> @test5(<4 x float> %a, <4 x float> %b) nounwind {
entry:
; CHECK-LABEL: test5:
; CHECK: ptest
; CHECK-NEXT:	jne
; CHECK: ret

  %res = call i32 @llvm.x86.sse41.ptestz(<4 x float> %a, <4 x float> %a) nounwind 
  %one = icmp ne i32 %res, 0 
  br i1 %one, label %bb1, label %bb2

bb1:
  %c = fadd <4 x float> %b, < float 1.000000e+002, float 2.000000e+002, float 3.000000e+002, float 4.000000e+002 >
  br label %return

bb2:
	%d = fdiv <4 x float> %b, < float 1.000000e+002, float 2.000000e+002, float 3.000000e+002, float 4.000000e+002 >
	br label %return

return:
  %e = phi <4 x float> [%c, %bb1], [%d, %bb2]
  ret <4 x float> %e
}

define <4 x float> @test7(<4 x float> %a, <4 x float> %b) nounwind {
entry:
; CHECK-LABEL: test7:
; CHECK: ptest
; CHECK-NEXT:	jne
; CHECK: ret

  %res = call i32 @llvm.x86.sse41.ptestz(<4 x float> %a, <4 x float> %a) nounwind 
  %one = trunc i32 %res to i1 
  br i1 %one, label %bb1, label %bb2

bb1:
  %c = fadd <4 x float> %b, < float 1.000000e+002, float 2.000000e+002, float 3.000000e+002, float 4.000000e+002 >
  br label %return

bb2:
	%d = fdiv <4 x float> %b, < float 1.000000e+002, float 2.000000e+002, float 3.000000e+002, float 4.000000e+002 >
	br label %return

return:
  %e = phi <4 x float> [%c, %bb1], [%d, %bb2]
  ret <4 x float> %e
}

define <4 x float> @test8(<4 x float> %a, <4 x float> %b) nounwind {
entry:
; CHECK-LABEL: test8:
; CHECK: ptest
; CHECK-NEXT:	jae
; CHECK: ret

  %res = call i32 @llvm.x86.sse41.ptestc(<4 x float> %a, <4 x float> %a) nounwind 
  %one = icmp ne i32 %res, 0 
  br i1 %one, label %bb1, label %bb2

bb1:
  %c = fadd <4 x float> %b, < float 1.000000e+002, float 2.000000e+002, float 3.000000e+002, float 4.000000e+002 >
  br label %return

bb2:
	%d = fdiv <4 x float> %b, < float 1.000000e+002, float 2.000000e+002, float 3.000000e+002, float 4.000000e+002 >
	br label %return

return:
  %e = phi <4 x float> [%c, %bb1], [%d, %bb2]
  ret <4 x float> %e
}

define <4 x float> @test10(<4 x float> %a, <4 x float> %b) nounwind {
entry:
; CHECK-LABEL: test10:
; CHECK: ptest
; CHECK-NEXT:	jae
; CHECK: ret

  %res = call i32 @llvm.x86.sse41.ptestc(<4 x float> %a, <4 x float> %a) nounwind 
  %one = trunc i32 %res to i1 
  br i1 %one, label %bb1, label %bb2

bb1:
  %c = fadd <4 x float> %b, < float 1.000000e+002, float 2.000000e+002, float 3.000000e+002, float 4.000000e+002 >
  br label %return

bb2:
	%d = fdiv <4 x float> %b, < float 1.000000e+002, float 2.000000e+002, float 3.000000e+002, float 4.000000e+002 >
	br label %return

return:
  %e = phi <4 x float> [%c, %bb1], [%d, %bb2]
  ret <4 x float> %e
}

define <4 x float> @test11(<4 x float> %a, <4 x float> %b) nounwind {
entry:
; CHECK-LABEL: test11:
; CHECK: ptest
; CHECK-NEXT:	jne
; CHECK: ret

  %res = call i32 @llvm.x86.sse41.ptestz(<4 x float> %a, <4 x float> %a) nounwind 
  %one = icmp eq i32 %res, 1 
  br i1 %one, label %bb1, label %bb2

bb1:
  %c = fadd <4 x float> %b, < float 1.000000e+002, float 2.000000e+002, float 3.000000e+002, float 4.000000e+002 >
  br label %return

bb2:
	%d = fdiv <4 x float> %b, < float 1.000000e+002, float 2.000000e+002, float 3.000000e+002, float 4.000000e+002 >
	br label %return

return:
  %e = phi <4 x float> [%c, %bb1], [%d, %bb2]
  ret <4 x float> %e
}

define <4 x float> @test12(<4 x float> %a, <4 x float> %b) nounwind {
entry:
; CHECK-LABEL: test12:
; CHECK: ptest
; CHECK-NEXT:	je
; CHECK: ret

  %res = call i32 @llvm.x86.sse41.ptestz(<4 x float> %a, <4 x float> %a) nounwind 
  %one = icmp ne i32 %res, 1 
  br i1 %one, label %bb1, label %bb2

bb1:
  %c = fadd <4 x float> %b, < float 1.000000e+002, float 2.000000e+002, float 3.000000e+002, float 4.000000e+002 >
  br label %return

bb2:
	%d = fdiv <4 x float> %b, < float 1.000000e+002, float 2.000000e+002, float 3.000000e+002, float 4.000000e+002 >
	br label %return

return:
  %e = phi <4 x float> [%c, %bb1], [%d, %bb2]
  ret <4 x float> %e
}

