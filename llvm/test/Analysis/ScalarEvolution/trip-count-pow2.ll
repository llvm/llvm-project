; RUN: opt < %s -scalar-evolution -analyze | FileCheck %s

define void @test1(i32 %n) {
entry:
  %s = mul i32 %n, 96
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %i.next = add i32 %i, 32
  %t = icmp ne i32 %i.next, %s
  br i1 %t, label %loop, label %exit
exit:
  ret void

; CHECK-LABEL: @test1
; CHECK: Loop %loop: backedge-taken count is ((-32 + (96 * %n)) /u 32)
; CHECK: Loop %loop: max backedge-taken count is 134217727
}

; PR19183
define i32 @test2(i32 %n) {
entry:
  %s = and i32 %n, -32
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %i.next = add i32 %i, 32
  %t = icmp ne i32 %i.next, %s
  br i1 %t, label %loop, label %exit
exit:
  ret i32 %i

; CHECK-LABEL: @test2
; CHECK: Loop %loop: backedge-taken count is ((-32 + (32 * (%n /u 32))<nuw>) /u 32)
; CHECK: Loop %loop: max backedge-taken count is 134217727
}

define void @test3(i32 %n) {
entry:
  %s = mul i32 %n, 96
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %i.next = add i32 %i, 96
  %t = icmp ne i32 %i.next, %s
  br i1 %t, label %loop, label %exit
exit:
  ret void

; CHECK-LABEL: @test3
; CHECK: Loop %loop: backedge-taken count is ((-32 + (32 * %n)) /u 32)
; CHECK: Loop %loop: max backedge-taken count is 134217727
}

define void @test4(i32 %n) {
entry:
  %s = mul i32 %n, 4
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %i.next = add i32 %i, 12
  %t = icmp ne i32 %i.next, %s
  br i1 %t, label %loop, label %exit
exit:
  ret void

; CHECK-LABEL: @test4
; CHECK: Loop %loop: backedge-taken count is ((-4 + (-1431655764 * %n)) /u 4)
; CHECK: Loop %loop: max backedge-taken count is 1073741823
}

define void @test5(i32 %n) {
entry:
  %s = mul i32 %n, 4
  br label %loop
loop:
  %i = phi i32 [ %s, %entry ], [ %i.next, %loop ]
  %i.next = add i32 %i, -4
  %t = icmp ne i32 %i.next, 0
  br i1 %t, label %loop, label %exit
exit:
  ret void

; CHECK-LABEL: @test5
; CHECK: Loop %loop: backedge-taken count is ((-4 + (4 * %n)) /u 4)
; CHECK: Loop %loop: max backedge-taken count is 1073741823
}
