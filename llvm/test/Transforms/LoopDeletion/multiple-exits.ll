; Checks whether dead loops with multiple exits can be eliminated.
; Note that we loop simplify and LCSSA over the test cases to make sure the
; critical components remain after those passes and are visible to the loop
; deletion pass.
;
; RUN: opt < %s -passes=loop-simplify,lcssa -S | FileCheck %s --check-prefixes=CHECK,BEFORE
; RUN: opt < %s -passes=no-op-loop -S | FileCheck %s --check-prefixes=CHECK,BEFORE
; RUN: opt < %s -passes=loop-deletion -S | FileCheck %s --check-prefixes=CHECK,AFTER


define void @foo(i64 %n, i64 %m) nounwind {
; CHECK-LABEL: @foo(

entry:
  br label %bb
; CHECK:       entry:
; BEFORE-NEXT:   br label %bb
; AFTER-NEXT:    br label %return

bb:
  %x.0 = phi i64 [ 0, %entry ], [ %t0, %bb2 ]
  %t0 = add i64 %x.0, 1
  %t1 = icmp slt i64 %x.0, %n
  br i1 %t1, label %bb2, label %return
; BEFORE:      bb:
; BEFORE:        br i1 {{.*}}, label %bb2, label %return
; AFTER-NOT:   bb:
; AFTER-NOT:     br

bb2:
  %t2 = icmp slt i64 %x.0, %m
  br i1 %t1, label %bb, label %return
; BEFORE:      bb2:
; BEFORE:        br i1 {{.*}}, label %bb, label %return
; AFTER-NOT:   bb2:
; AFTER-NOT:     br

return:
  ret void
; CHECK:       return:
; CHECK-NEXT:    ret void
}

define i64 @bar(i64 %n, i64 %m, i64 %maybe_zero) nounwind {
; CHECK-LABEL: @bar(

entry:
  br label %bb
; CHECK:       entry:
; BEFORE-NEXT:   br label %bb
; AFTER-NEXT:    br label %return

bb:
  %x.0 = phi i64 [ 0, %entry ], [ %t0, %bb3 ]
  %t0 = add i64 %x.0, 1
  %t1 = icmp slt i64 %x.0, %n
  br i1 %t1, label %bb2, label %return
; BEFORE:      bb:
; BEFORE:        br i1 {{.*}}, label %bb2, label %return
; AFTER-NOT:   bb:
; AFTER-NOT:     br

bb2:
  %t2 = icmp slt i64 %x.0, %m
  ; This unused division prevents unifying this loop exit path with others
  ; because it can be deleted but cannot be hoisted.
  %unused1 = udiv i64 42, %maybe_zero
  br i1 %t2, label %bb3, label %return
; BEFORE:      bb2:
; BEFORE:        br i1 {{.*}}, label %bb3, label %return
; AFTER-NOT:   bb2:
; AFTER-NOT:     br

bb3:
  %t3 = icmp slt i64 %x.0, %m
  ; This unused division prevents unifying this loop exit path with others
  ; because it can be deleted but cannot be hoisted.
  %unused2 = sdiv i64 42, %maybe_zero
  br i1 %t3, label %bb, label %return
; BEFORE:      bb3:
; BEFORE:        br i1 {{.*}}, label %bb, label %return
; AFTER-NOT:   bb3:
; AFTER-NOT:     br

return:
  %x.lcssa = phi i64 [ 10, %bb ], [ 10, %bb2 ], [ 10, %bb3 ]
  ret i64 %x.lcssa
; CHECK:       return:
; BEFORE-NEXT:   %[[X:.*]] = phi i64 [ 10, %bb ], [ 10, %bb2 ], [ 10, %bb3 ]
; AFTER-NEXT:    %[[X:.*]] = phi i64 [ 10, %entry ]
; CHECK-NEXT:    ret i64 %[[X]]
}

; This function has a loop which looks like @bar's but that cannot be deleted
; because which path we exit through determines which value is selected.
define i64 @baz(i64 %n, i64 %m, i64 %maybe_zero) nounwind {
; CHECK-LABEL:  @baz(

entry:
  br label %bb
; CHECK:       entry:
; CHECK-NEXT:    br label %bb

bb:
  %x.0 = phi i64 [ 0, %entry ], [ %t0, %bb3 ]
  %t0 = add i64 %x.0, 1
  %t1 = icmp slt i64 %x.0, %n
  br i1 %t1, label %bb2, label %return
; CHECK:       bb:
; CHECK:         br i1 {{.*}}, label %bb2, label %return

bb2:
  %t2 = icmp slt i64 %x.0, %m
  ; This unused division prevents unifying this loop exit path with others
  ; because it can be deleted but cannot be hoisted.
  %unused1 = udiv i64 42, %maybe_zero
  br i1 %t2, label %bb3, label %return
; CHECK:       bb2:
; CHECK:         br i1 {{.*}}, label %bb3, label %return

bb3:
  %t3 = icmp slt i64 %x.0, %m
  ; This unused division prevents unifying this loop exit path with others
  ; because it can be deleted but cannot be hoisted.
  %unused2 = sdiv i64 42, %maybe_zero
  br i1 %t3, label %bb, label %return
; CHECK:       bb3:
; CHECK:         br i1 {{.*}}, label %bb, label %return

return:
  %x.lcssa = phi i64 [ 12, %bb ], [ 10, %bb2 ], [ 10, %bb3 ]
  ret i64 %x.lcssa
; CHECK: return:
; CHECK-NEXT:  %[[X:.*]] = phi i64 [ 12, %bb ], [ 10, %bb2 ], [ 10, %bb3 ]
; CHECK-NEXT:  ret i64 %[[X]]
}
