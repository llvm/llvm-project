; RUN: opt < %s -simplifycfg -S | FileCheck %s

define zeroext i1 @test1(i1 zeroext %flag, i32 %blksA, i32 %blksB, i32 %nblks) {
entry:
  br i1 %flag, label %if.then, label %if.else

; CHECK-LABEL: test1
; CHECK: add
; CHECK: select
; CHECK: icmp
; CHECK-NOT: br
if.then:
  %cmp = icmp uge i32 %blksA, %nblks
  %frombool1 = zext i1 %cmp to i8
  br label %if.end

if.else:
  %add = add i32 %nblks, %blksB
  %cmp2 = icmp ule i32 %add, %blksA
  %frombool3 = zext i1 %cmp2 to i8
  br label %if.end

if.end:
  %obeys.0 = phi i8 [ %frombool1, %if.then ], [ %frombool3, %if.else ]
  %tobool4 = icmp ne i8 %obeys.0, 0
  ret i1 %tobool4
}

define zeroext i1 @test2(i1 zeroext %flag, i32 %blksA, i32 %blksB, i32 %nblks) {
entry:
  br i1 %flag, label %if.then, label %if.else

; CHECK-LABEL: test2
; CHECK: add
; CHECK: select
; CHECK: icmp
; CHECK-NOT: br
if.then:
  %cmp = icmp uge i32 %blksA, %nblks
  %frombool1 = zext i1 %cmp to i8
  br label %if.end

if.else:
  %add = add i32 %nblks, %blksB
  %cmp2 = icmp uge i32 %blksA, %add
  %frombool3 = zext i1 %cmp2 to i8
  br label %if.end

if.end:
  %obeys.0 = phi i8 [ %frombool1, %if.then ], [ %frombool3, %if.else ]
  %tobool4 = icmp ne i8 %obeys.0, 0
  ret i1 %tobool4
}

declare i32 @foo(i32, i32) nounwind readnone

define i32 @test3(i1 zeroext %flag, i32 %x, i32 %y) {
entry:
  br i1 %flag, label %if.then, label %if.else

if.then:
  %x0 = call i32 @foo(i32 %x, i32 0) nounwind readnone
  %y0 = call i32 @foo(i32 %x, i32 1) nounwind readnone
  br label %if.end

if.else:
  %x1 = call i32 @foo(i32 %y, i32 0) nounwind readnone
  %y1 = call i32 @foo(i32 %y, i32 1) nounwind readnone
  br label %if.end

if.end:
  %xx = phi i32 [ %x0, %if.then ], [ %x1, %if.else ]
  %yy = phi i32 [ %y0, %if.then ], [ %y1, %if.else ]
  %ret = add i32 %xx, %yy
  ret i32 %ret
}

; CHECK-LABEL: test3
; CHECK: select
; CHECK: call
; CHECK: call
; CHECK: add
; CHECK-NOT: br
