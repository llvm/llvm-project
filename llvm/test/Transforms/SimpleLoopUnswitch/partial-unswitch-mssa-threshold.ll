; RUN: opt -simple-loop-unswitch-memoryssa-threshold=0 -memssa-check-limit=1 -passes='loop-mssa(simple-loop-unswitch<nontrivial>),verify<loops>' -S < %s | FileCheck --check-prefix=THRESHOLD-0 %s
; RUN: opt -memssa-check-limit=1 -passes='loop-mssa(simple-loop-unswitch<nontrivial>),verify<loops>' -S < %s | FileCheck --check-prefix=THRESHOLD-DEFAULT %s

; Make sure -loop-unswitch-memoryssa-threshold works. The test uses
; -memssa-check-limit=1 to effectively disable any MemorySSA optimizations
; on construction, so the test can be kept simple.

declare void @clobber()

; Partial unswitching is possible, because the store in %noclobber does not
; alias the load of the condition.
define i32 @partial_unswitch_true_successor_noclobber(ptr noalias %ptr.1, ptr noalias %ptr.2, i32 %N) {
; THRESHOLD-0-LABEL: @partial_unswitch_true_successor
; THRESHOLD-0: entry:
; THRESHOLD-0: br label %loop.header
;
; THRESHOLD-DEFAULT-LABEL: @partial_unswitch_true_successor
; THRESHOLD-DEFAULT-NEXT:  entry:
; THRESHOLD-DEFAULT-NEXT:   [[LV:%[0-9]+]] = load i32, ptr %ptr.1, align 4
; THRESHOLD-DEFAULT-NEXT:   [[C:%[0-9]+]] = icmp eq i32 [[LV]], 100
; THRESHOLD-DEFAULT-NEXT:   br i1 [[C]]
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %lv = load i32, ptr %ptr.1
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  %gep.1 = getelementptr i32, ptr %ptr.2, i32 %iv
  store i32 %lv, ptr %gep.1
  br label %loop.latch

clobber:
  call void @clobber()
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret i32 10
}
