; RUN: opt -loop-unswitch -loop-unswitch-threshold=10 -verify-dom-info -verify-memoryssa -S -enable-new-pm=0 %s | FileCheck %s

declare void @clobber()

; Test cases for partial unswitching, where the regular cost-model overestimates
; the cost of unswitching, because it misses the fact that the unswitched paths
; are no-ops.


define i32 @no_partial_unswitch_size_too_large_no_mustprogress(i32* %ptr, i32 %N) {
; CHECK-LABEL: @no_partial_unswitch_size_too_large_no_mustprogress
; CHECK-LABEL: entry:
; CHECK-NEXT:   br label %loop.header
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %lv = load i32, i32* %ptr
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  br label %loop.latch

clobber:
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret i32 10
}

define i32 @partial_unswitch_shortcut_mustprogress(i32* %ptr, i32 %N) mustprogress {
; CHECK-LABEL: @partial_unswitch_shortcut_mustprogress
; CHECK-LABEL: entry:
; CHECK-NEXT:   [[LV:%[0-9]+]] = load i32, i32* %ptr, align 4
; CHECK-NEXT:   [[C:%[0-9]+]] = icmp eq i32 [[LV]], 100
; CHECK-NEXT:   br i1 [[C]], label %[[CRIT_TO_EXIT:[a-z._]+]], label %[[CRIT_TO_HEADER:[a-z._]+]]
;
; CHECK:      [[CRIT_TO_HEADER]]:
; CHECK-NEXT:   br label %loop.header
;
; CHECK:      [[CRIT_TO_EXIT]]:
; CHECK-NEXT:   br label %exit
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %lv = load i32, i32* %ptr
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  br label %loop.latch

clobber:
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret i32 10
}

define i32 @partial_unswitch_shortcut_mustprogress_single_exit_on_path(i32* %ptr, i32 %N) mustprogress {
; CHECK-LABEL: @partial_unswitch_shortcut_mustprogress_single_exit_on_path
; CHECK-LABEL: entry:
; CHECK-NEXT:   [[LV:%[0-9]+]] = load i32, i32* %ptr, align 4
; CHECK-NEXT:   [[C:%[0-9]+]] = icmp eq i32 [[LV]], 100
; CHECK-NEXT:   br i1 [[C]], label %[[CRIT_TO_EXIT:.+]], label %[[CRIT_TO_HEADER:[a-z._]+]]
;
; CHECK:      [[CRIT_TO_HEADER]]:
; CHECK-NEXT:   br label %loop.header
;
; CHECK:      [[CRIT_TO_EXIT]]:
; CHECK-NEXT:   br label %exit
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %lv = load i32, i32* %ptr
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  %c.1 = icmp ult i32 %iv, 123
  br i1 %c.1, label %loop.latch, label %exit.1

clobber:
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  %c = icmp ult i32 %iv, %N
  br i1 %c, label %loop.latch, label %exit.2

loop.latch:
  %iv.next = add i32 %iv, 1
  br label %loop.header

exit.1:
  ret i32 10

exit.2:
  ret i32 10
}

define i32 @no_partial_unswitch_shortcut_mustprogress_no_exit_on_path(i32* %ptr, i32 %N) mustprogress {
; CHECK-LABEL: @no_partial_unswitch_shortcut_mustprogress_no_exit_on_path
; CHECK-LABEL: entry:
; CHECK-NEXT:   br label %loop.header
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %lv = load i32, i32* %ptr
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  br label %loop.latch

clobber:
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  %c = icmp ult i32 %iv, %N
  br i1 %c, label %loop.latch, label %exit

loop.latch:
  %iv.next = add i32 %iv, 1
  br label %loop.header

exit:
  ret i32 10
}

define i32 @no_partial_unswitch_shortcut_mustprogress_exit_value_used(i32* %ptr, i32 %N) mustprogress {
; CHECK-LABEL: @no_partial_unswitch_shortcut_mustprogress_exit_value_use
; CHECK-LABEL: entry:
; CHECK-NEXT:   br label %loop.header
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %red = phi i32 [ 0, %entry ], [ %red.next, %loop.latch ]
  %lv = load i32, i32* %ptr
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  %red.add = add i32 %red, %lv
  br label %loop.latch

clobber:
  %red.mul = mul i32 %red, %lv
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  br label %loop.latch

loop.latch:
  %red.next = phi i32 [ %red.add, %noclobber ], [ %red.mul, %clobber ]
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret i32 %red.next
}

define i32 @partial_unswitch_shortcut_multiple_exiting_blocks(i32* %ptr, i32 %N, i1 %ec.1) mustprogress {
; CHECK-LABEL: @partial_unswitch_shortcut_multiple_exiting_blocks
; CHECK-LABEL: entry:
; CHECK-NEXT:   [[LV:%[0-9]+]] = load i32, i32* %ptr, align 4
; CHECK-NEXT:   [[C:%[0-9]+]] = icmp eq i32 [[LV]], 100
; CHECK-NEXT:   br i1 [[C]], label %[[CRIT_TO_EXIT:.+]], label %[[CRIT_TO_HEADER:[a-z._]+]]
;
; CHECK:      [[CRIT_TO_HEADER]]:
; CHECK-NEXT:   br label %loop.header
;
; CHECK:      [[CRIT_TO_EXIT]]:
; CHECK-NEXT:   br label %exit
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %lv = load i32, i32* %ptr
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  br i1 %ec.1, label %loop.latch, label %exit

clobber:
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret i32 10
}

define i32 @no_partial_unswitch_shortcut_multiple_exit_blocks(i32* %ptr, i32 %N, i1 %ec.1) mustprogress {
; CHECK-LABEL: @no_partial_unswitch_shortcut_multiple_exit_blocks
; CHECK-LABEL: entry:
; CHECK-NEXT:   br label %loop.header
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %lv = load i32, i32* %ptr
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  br i1 %ec.1, label %loop.latch, label %exit.2

clobber:
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit.1

exit.1:
  ret i32 10

exit.2:
  ret i32 20
}

define i32 @no_partial_unswitch_shortcut_mustprogress_store(i32* %ptr, i32* noalias %dst, i32 %N) mustprogress {
; CHECK-LABEL: @no_partial_unswitch_shortcut_mustprogress_store
; CHECK-LABEL: entry:
; CHECK-NEXT:   br label %loop.header
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %lv = load i32, i32* %ptr
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  store i32 0, i32* %dst
  br label %loop.latch

clobber:
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret i32 10
}

define i32 @no_partial_unswitch_shortcut_mustprogress_store2(i32* %ptr, i32* noalias %dst, i32 %N) mustprogress {
; CHECK-LABEL: @no_partial_unswitch_shortcut_mustprogress_store
; CHECK-LABEL: entry:
; CHECK-NEXT:   br label %loop.header
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %lv = load i32, i32* %ptr
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  br label %loop.latch

clobber:
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  br label %loop.latch

loop.latch:
  store i32 0, i32* %dst
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret i32 10
}

define i32 @no_partial_unswitch_shortcut_mustprogress_store3(i32* %ptr, i32* noalias %dst, i32 %N) mustprogress {
; CHECK-LABEL: @no_partial_unswitch_shortcut_mustprogress_store
; CHECK-LABEL: entry:
; CHECK-NEXT:   br label %loop.header
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  store i32 0, i32* %dst
  %lv = load i32, i32* %ptr
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  br label %loop.latch

clobber:
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  call void @clobber()
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret i32 10
}
