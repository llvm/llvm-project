; RUN: llc < %s -march=lanai | FileCheck %s

; Test the alu setcc combiner.

; TODO: Enhance combiner to handle this case. This expands into:
;   sub     %r7, %r6, %r3
;   sub.f   %r7, %r6, %r0
;   sel.eq %r18, %r3, %rv
; This is different from the pattern currently matched. If the lowered form had
; been sub.f %r3, 0, %r0 then it would have matched.

; Function Attrs: norecurse nounwind readnone
define i32 @test0a(i32 inreg %a, i32 inreg %b, i32 inreg %c, i32 inreg %d) #0 {
entry:
  %sub = sub i32 %b, %a
  %cmp = icmp eq i32 %sub, 0
  %cond = select i1 %cmp, i32 %c, i32 %sub
  ret i32 %cond
}
; CHECK-LABEL: test0a
; CHECK: sub.f %r7
; CHECK: sel.eq

; Function Attrs: norecurse nounwind readnone
define i32 @test0b(i32 inreg %a, i32 inreg %b, i32 inreg %c, i32 inreg %d) #0 {
entry:
  %cmp = icmp eq i32 %b, %a
  %cond = select i1 %cmp, i32 %c, i32 %b
  ret i32 %cond
}
; CHECK-LABEL: test0b
; CHECK: sub.f %r7, %r6, %r0
; CHECK-NEXT: sel.eq

; Function Attrs: norecurse nounwind readnone
define i32 @test1a(i32 inreg %a, i32 inreg %b, i32 inreg %c, i32 inreg %d) #0 {
entry:
  %sub = sub i32 %b, %a
  %cmp = icmp slt i32 %sub, 0
  %cond = select i1 %cmp, i32 %c, i32 %d
  ret i32 %cond
}
; CHECK-LABEL: test1a
; CHECK: sub.f %r7, %r6
; CHECK-NEXT: sel.mi

; Function Attrs: norecurse nounwind readnone
define i32 @test1b(i32 inreg %a, i32 inreg %b, i32 inreg %c, i32 inreg %d) #0 {
entry:
  %sub = sub i32 %b, %a
  %cmp = icmp slt i32 %sub, 0
  %cond = select i1 %cmp, i32 %c, i32 %d
  ret i32 %cond
}
; CHECK-LABEL: test1b
; CHECK: sub.f %r7, %r6
; CHECK-NEXT: sel.mi

; Function Attrs: norecurse nounwind readnone
define i32 @test2a(i32 inreg %a, i32 inreg %b, i32 inreg %c, i32 inreg %d) #0 {
entry:
  %sub = sub i32 %b, %a
  %cmp = icmp sgt i32 %sub, -1
  %cond = select i1 %cmp, i32 %c, i32 %d
  ret i32 %cond
}
; CHECK-LABEL: test2a
; CHECK: sub.f %r7, %r6
; CHECK: sel.pl

; Function Attrs: norecurse nounwind readnone
define i32 @test2b(i32 inreg %a, i32 inreg %b, i32 inreg %c, i32 inreg %d) #0 {
entry:
  %sub = sub i32 %b, %a
  %cmp = icmp sgt i32 %sub, -1
  %cond = select i1 %cmp, i32 %c, i32 %d
  ret i32 %cond
}
; CHECK-LABEL: test2b
; CHECK: sub.f %r7, %r6
; CHECK: sel.pl

; Function Attrs: norecurse nounwind readnone
define i32 @test3(i32 inreg %a, i32 inreg %b, i32 inreg %c, i32 inreg %d) #0 {
entry:
  %sub = sub i32 %b, %a
  %cmp = icmp slt i32 %sub, 1
  %cond = select i1 %cmp, i32 %c, i32 %d
  ret i32 %cond
}

; Function Attrs: norecurse nounwind readnone
define i32 @test4(i32 inreg %a, i32 inreg %b, i32 inreg %c, i32 inreg %d) #0 {
entry:
  %cmp = icmp ne i32 %a, 0
  %cmp1 = icmp ult i32 %a, %b
  %or.cond = and i1 %cmp, %cmp1
  br i1 %or.cond, label %return, label %if.end

if.end:                                           ; preds = %entry
  %cmp2 = icmp ne i32 %b, 0
  %cmp4 = icmp ult i32 %b, %c
  %or.cond29 = and i1 %cmp2, %cmp4
  br i1 %or.cond29, label %return, label %if.end6

if.end6:                                          ; preds = %if.end
  %cmp7 = icmp ne i32 %c, 0
  %cmp9 = icmp ult i32 %c, %d
  %or.cond30 = and i1 %cmp7, %cmp9
  br i1 %or.cond30, label %return, label %if.end11

if.end11:                                         ; preds = %if.end6
  %cmp12 = icmp ne i32 %d, 0
  %cmp14 = icmp ult i32 %d, %a
  %or.cond31 = and i1 %cmp12, %cmp14
  %b. = select i1 %or.cond31, i32 %b, i32 21
  ret i32 %b.

return:                                           ; preds = %if.end6, %if.end, %entry
  %retval.0 = phi i32 [ %c, %entry ], [ %d, %if.end ], [ %a, %if.end6 ]
  ret i32 %retval.0
}
; CHECK-LABEL: test4
; TODO: Re-enable test. This test is disabled post making the combiner more
; conservative.
; DISABLED_CHECK: and.f

; Test to avoid incorrect fusing that spans across basic blocks
@a = global i32 -1, align 4
@b = global i32 0, align 4

; Function Attrs: nounwind
define void @testBB() {
entry:
  %0 = load i32, i32* @a, align 4, !tbaa !1
  %1 = load i32, i32* @b, align 4, !tbaa !1
  %sub.i = sub i32 %1, %0
  %tobool = icmp sgt i32 %sub.i, -1
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %call1 = tail call i32 bitcast (i32 (...)* @g to i32 ()*)()
  br label %while.body

while.body:                                       ; preds = %if.then, %while.body
  br label %while.body

if.end:                                           ; preds = %entry
  %cmp.i = icmp slt i32 %sub.i, 1
  br i1 %cmp.i, label %if.then4, label %if.end7

if.then4:                                         ; preds = %if.end
  %call5 = tail call i32 bitcast (i32 (...)* @g to i32 ()*)()
  br label %while.body6

while.body6:                                      ; preds = %if.then4, %while.body6
  br label %while.body6

if.end7:                                          ; preds = %if.end
  ret void
}

declare i32 @g(...)
; CHECK-LABEL: testBB
; CHECK: sub.f {{.*}}, %r0

!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
