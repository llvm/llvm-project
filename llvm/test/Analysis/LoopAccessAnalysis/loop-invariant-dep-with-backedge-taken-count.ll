; RUN: opt -passes='print<access-info>' -disable-output %s 2>&1 | FileCheck %s

; Test cases for using the backedge-taken-count to rule out dependencies between
; an invariant and strided accesses.

define void @test_distance_greater_than_BTC_100(ptr %a) {
; CHECK-LABEL: Loop access info in function 'test_distance_greater_than_BTC_100':
; CHECK-NEXT:   loop:
; CHECK-NEXT:     Report: unsafe dependent memory operations in loop.
; CHECK-NEXT: Unknown data dependence.
; CHECK-NEXT:     Dependences:
; CHECK-NEXT:       Unknown:
; CHECK-NEXT:           %l = load i32, ptr %gep.x, align 4 ->
; CHECK-NEXT:           store i32 %l, ptr %gep, align 4
; CHECK-EMPTY:
; CHECK-NEXT:     Run-time memory checks:
; CHECK-NEXT:     Grouped accesses:
; CHECK-EMPTY:
;
entry:
  %gep.x = getelementptr i32, ptr %a, i32 100
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %gep = getelementptr i32, ptr %a, i32 %iv
  %l = load i32, ptr %gep.x
  store i32 %l, ptr %gep
  %iv.next = add i32 %iv, 1
  %ec = icmp eq i32 %iv.next, 100
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

define void @test_distance_much_greater_than_BTC_100(ptr %a) {
; CHECK-LABEL: Loop access info in function 'test_distance_much_greater_than_BTC_100':
; CHECK-NEXT:   loop:
; CHECK-NEXT:     Report: unsafe dependent memory operations in loop.
; CHECK-NEXT: Unknown data dependence.
; CHECK-NEXT:     Dependences:
; CHECK-NEXT:       Unknown:
; CHECK-NEXT:           %l = load i32, ptr %gep.x, align 4 ->
; CHECK-NEXT:           store i32 %l, ptr %gep, align 4
; CHECK-EMPTY:
; CHECK-NEXT:     Run-time memory checks:
; CHECK-NEXT:     Grouped accesses:
; CHECK-EMPTY:
;
entry:
  %gep.x = getelementptr i32, ptr %a, i32 200
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %gep = getelementptr i32, ptr %a, i32 %iv
  %l = load i32, ptr %gep.x
  store i32 %l, ptr %gep
  %iv.next = add i32 %iv, 1
  %ec = icmp eq i32 %iv.next, 100
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

define void @test_distance_equal_BTC_100(ptr %a) {
; CHECK-LABEL: Loop access info in function 'test_distance_equal_BTC_100':
; CHECK-NEXT:   loop:
; CHECK-NEXT:     Report: unsafe dependent memory operations in loop.
; CHECK-NEXT: Unknown data dependence.
; CHECK-NEXT:     Dependences:
; CHECK-NEXT:       Unknown:
; CHECK-NEXT:           %l = load i32, ptr %gep.x, align 4 ->
; CHECK-NEXT:           store i32 %l, ptr %gep, align 4
; CHECK-EMPTY:
; CHECK-NEXT:     Run-time memory checks:
; CHECK-NEXT:     Grouped accesses:
; CHECK-EMPTY:
;
entry:
  %gep.x = getelementptr i32, ptr %a, i32 99
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %gep = getelementptr i32, ptr %a, i32 %iv
  %l = load i32, ptr %gep.x
  store i32 %l, ptr %gep
  %iv.next = add i32 %iv, 1
  %ec = icmp eq i32 %iv.next, 100
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

define void @test_distance_greater_than_BTC_10000(ptr %a) {
; CHECK-LABEL: Loop access info in function 'test_distance_greater_than_BTC_10000':
; CHECK-NEXT:   loop:
; CHECK-NEXT:     Report: unsafe dependent memory operations in loop.
; CHECK-NEXT: Unknown data dependence.
; CHECK-NEXT:     Dependences:
; CHECK-NEXT:       Unknown:
; CHECK-NEXT:           %l = load i32, ptr %gep.x, align 4 ->
; CHECK-NEXT:           store i32 %l, ptr %gep, align 4
; CHECK-EMPTY:
; CHECK-NEXT:     Run-time memory checks:
; CHECK-NEXT:     Grouped accesses:
; CHECK-EMPTY:
;
entry:
  %gep.x = getelementptr i32, ptr %a, i32 10000
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %gep = getelementptr i32, ptr %a, i32 %iv
  %l = load i32, ptr %gep.x
  store i32 %l, ptr %gep
  %iv.next = add i32 %iv, 1
  %ec = icmp eq i32 %iv.next, 10000
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

define void @test_distance_equal_to_BTC_10000(ptr %a) {
; CHECK-LABEL: Loop access info in function 'test_distance_equal_to_BTC_10000':
; CHECK-NEXT:   loop:
; CHECK-NEXT:     Report: unsafe dependent memory operations in loop.
; CHECK-NEXT: Unknown data dependence.
; CHECK-NEXT:     Dependences:
; CHECK-NEXT:       Unknown:
; CHECK-NEXT:           %l = load i32, ptr %gep.x, align 4 ->
; CHECK-NEXT:           store i32 %l, ptr %gep, align 4
; CHECK-EMPTY:
; CHECK-NEXT:     Run-time memory checks:
; CHECK-NEXT:     Grouped accesses:
; CHECK-EMPTY:
;
entry:
  %gep.x = getelementptr i32, ptr %a, i32 9999
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %gep = getelementptr i32, ptr %a, i32 %iv
  %l = load i32, ptr %gep.x
  store i32 %l, ptr %gep
  %iv.next = add i32 %iv, 1
  %ec = icmp eq i32 %iv.next, 100000
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

define void @test_btc_is_unknown_value(ptr %a, i32 %N) {
; CHECK-LABEL: Loop access info in function 'test_btc_is_unknown_value':
; CHECK-NEXT:   loop:
; CHECK-NEXT:     Report: unsafe dependent memory operations in loop.
; CHECK-NEXT: Unknown data dependence.
; CHECK-NEXT:     Dependences:
; CHECK-NEXT:       Unknown:
; CHECK-NEXT:           %l = load i32, ptr %gep.x, align 4 ->
; CHECK-NEXT:           store i32 %l, ptr %gep, align 4
; CHECK-EMPTY:
; CHECK-NEXT:     Run-time memory checks:
; CHECK-NEXT:     Grouped accesses:
;
entry:
  %gep.x = getelementptr i32, ptr %a, i32 100
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %gep = getelementptr i32, ptr %a, i32 %iv
  %l = load i32, ptr %gep.x
  store i32 %l, ptr %gep
  %iv.next = add i32 %iv, 1
  %ec = icmp eq i32 %iv.next, %N
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}
