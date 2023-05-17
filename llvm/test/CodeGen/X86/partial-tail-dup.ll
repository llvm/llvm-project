; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s


@gvar = external global i32

; dupbb has two predecessors, p1 and p2. p1 is hot, p2 is cold. So dupbb
; should be placed after p1, and not duplicated into p2.
;
; CHECK-LABEL: test1
; CHECK:       %p1
; CHECK:       .LBB0_4: # %dupbb
; CHECK:       %p2
; CHECK:       jmp .LBB0_4

define void @test1(ptr %p) !prof !1 {
entry:
  br label %header

header:
  %call = call zeroext i1 @a()
  br i1 %call, label %p1, label %p2, !prof !2

p1:
  call void @b()
  br label %dupbb

p2:
  call void @c()
  br label %dupbb

dupbb:
  %cond = icmp eq ptr @gvar, %p
  br i1 %cond, label %header, label %latch, !prof !3

latch:
  %call3 = call zeroext i1 @a()
  br i1 %call3, label %header, label %end, !prof !2

end:
  ret void
}


; dupbb has four predecessors p1, p2, p3 and p4. p1 and p2 are hot, p3 and  p4
; are cold. So dupbb should be placed after p1, duplicated into p2. p3 and p4
; should jump to dupbb.
;
; CHECK-LABEL: test2
; CHECK:       %p1
; CHECK:       .LBB1_8: # %dupbb
;
; CHECK:       %p2
; CHECK:       callq c
; CHECK-NEXT:  cmpq
; CHECK-NEXT:  je
; CHECK-NEXT:  jmp
;
; CHECK:       %p3
; CHECK:       jmp .LBB1_8
; CHECK:       %p4
; CHECK:       jmp .LBB1_8

define void @test2(ptr %p) !prof !1 {
entry:
  br label %header

header:
  %call = call zeroext i1 @a()
  br i1 %call, label %bb1, label %bb2, !prof !2

bb1:
  %call1 = call zeroext i1 @a()
  br i1 %call1, label %p1, label %p2, !prof !4

bb2:
  %call2 = call zeroext i1 @a()
  br i1 %call2, label %p3, label %p4, !prof !4

p1:
  call void @b()
  br label %dupbb

p2:
  call void @c()
  br label %dupbb

p3:
  call void @d()
  br label %dupbb

p4:
  call void @e()
  br label %dupbb

dupbb:
  %cond = icmp eq ptr @gvar, %p
  br i1 %cond, label %bb3, label %bb4, !prof !4

bb3:
  call void @b()
  br label %bb4

bb4:
  %call4 = call zeroext i1 @a()
  br i1 %call4, label %header, label %latch, !prof !3

latch:
  %call3 = call zeroext i1 @a()
  br i1 %call3, label %header, label %end, !prof !2

end:
  ret void
}


; dupbb has three predecessors p1, p2 and p3. p3 has two successors, so dupbb
; can't be duplicated into p3, but it should not block it to be duplicated into
; other predecessors.
;
; CHECK-LABEL: test3
; CHECK:       %p1
; CHECK:       .LBB2_6: # %dupbb
;
; CHECK:       %p2
; CHECK:       callq c
; CHECK:       cmpq
; CHECK-NEXT:  je
; CHECK-NEXT:  jmp
;
; CHECK:       %p3
; CHECK:       jne .LBB2_6

define void @test3(ptr %p) !prof !1 {
entry:
  br label %header

header:
  %call = call zeroext i1 @a()
  br i1 %call, label %bb1, label %p3, !prof !2

bb1:
  %call1 = call zeroext i1 @a()
  br i1 %call1, label %p1, label %p2, !prof !4

p1:
  call void @b()
  br label %dupbb

p2:
  call void @c()
  br label %dupbb

p3:
  %call2 = call zeroext i1 @a()
  br i1 %call2, label %dupbb, label %bb4, !prof !4

dupbb:
  %cond = icmp eq ptr @gvar, %p
  br i1 %cond, label %bb3, label %bb4, !prof !4

bb3:
  call void @b()
  br label %bb4

bb4:
  %call4 = call zeroext i1 @a()
  br i1 %call4, label %header, label %latch, !prof !3

latch:
  %call3 = call zeroext i1 @a()
  br i1 %call3, label %header, label %end, !prof !2

end:
  ret void
}

declare zeroext i1 @a()
declare void @b()
declare void @c()
declare void @d()
declare void @e()
declare void @f()

!1 = !{!"function_entry_count", i64 1000}
!2 = !{!"branch_weights", i32 100, i32 1}
!3 = !{!"branch_weights", i32 1, i32 100}
!4 = !{!"branch_weights", i32 60, i32 40}
