; RUN: llc < %s -mtriple=i386-apple-darwin10 -mcpu=penryn | FileCheck %s

; rdar://7475489

define i32 @test1(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK: test1:
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
  %5 = tail call i32 (...)* @foo() nounwind       ; <i32> [#uses=1]
  ret i32 %5

bb1:                                              ; preds = %entry
  %6 = tail call i32 (...)* @bar() nounwind       ; <i32> [#uses=1]
  ret i32 %6
}

declare i32 @foo(...)

declare i32 @bar(...)



; PR3351 - (P == 0) & (Q == 0) -> (P|Q) == 0
define i32 @test2(i32* %P, i32* %Q) nounwind ssp {
entry:
  %a = icmp eq i32* %P, null                    ; <i1> [#uses=1]
  %b = icmp eq i32* %Q, null                    ; <i1> [#uses=1]
  %c = and i1 %a, %b
  br i1 %c, label %bb1, label %return

bb1:                                              ; preds = %entry
  ret i32 4

return:                                           ; preds = %entry
  ret i32 192
; CHECK: test2:
; CHECK:	movl	4(%esp), %eax
; CHECK-NEXT:	orl	8(%esp), %eax
; CHECK-NEXT:	jne	LBB1_2
}

; PR3351 - (P != 0) | (Q != 0) -> (P|Q) != 0
define i32 @test3(i32* %P, i32* %Q) nounwind ssp {
entry:
  %a = icmp ne i32* %P, null                    ; <i1> [#uses=1]
  %b = icmp ne i32* %Q, null                    ; <i1> [#uses=1]
  %c = or i1 %a, %b
  br i1 %c, label %bb1, label %return

bb1:                                              ; preds = %entry
  ret i32 4

return:                                           ; preds = %entry
  ret i32 192
; CHECK: test3:
; CHECK:	movl	4(%esp), %eax
; CHECK-NEXT:	orl	8(%esp), %eax
; CHECK-NEXT:	je	LBB2_2
}

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

