; RUN: opt -passes=loop-vectorize -force-vector-width=2 -force-vector-interleave=1 \
; RUN:     -vplan-print-after=recordExecutionFrequencies -disable-output %s 2>&1 \
; RUN:   | FileCheck --check-prefix=VPLAN0 %s
; RUN: opt -passes=loop-vectorize -force-vector-width=2 -force-vector-interleave=1 \
; RUN:     -vplan-print-after=introduceMasksAndLinearize -disable-output %s 2>&1 \
; RUN:   | FileCheck --check-prefix=MASKED %s

; The recipes of %if.then all execute with the block's frequency of 1/4, the
; ones of %loop and %latch always execute.
define void @if_then(ptr noalias %a, ptr noalias %idx) {
; VPLAN0-LABEL: VPlan for loop in 'if_then'
; VPLAN0:         loop:
; VPLAN0-NEXT:      EMIT-SCALAR ir<%iv> = phi [ ir<0>, vector.ph ], [ ir<%iv.next>, latch ]
; VPLAN0-NEXT:      EMIT ir<%gep.idx> = getelementptr inbounds ir<%idx>, ir<%iv>
; VPLAN0-NEXT:      EMIT-SCALAR ir<%i> = load ir<%gep.idx>
; VPLAN0-NEXT:      EMIT ir<%c> = icmp sgt ir<%i>, ir<0>
; VPLAN0-NEXT:      EMIT branch-on-cond ir<%c> (!prof {1, 3})
; VPLAN0-NEXT:    Successor(s): if.then, latch
; VPLAN0-EMPTY:
; VPLAN0-NEXT:    if.then:
; VPLAN0-NEXT:      EMIT ir<%add> = add ir<%i>, ir<10> (!vplan.execution.frequency 2305843009213693952 (25%))
; VPLAN0-NEXT:      EMIT ir<%gep.a> = getelementptr inbounds ir<%a>, ir<%iv> (!vplan.execution.frequency 2305843009213693952 (25%))
; VPLAN0-NEXT:      EMIT store ir<%add>, ir<%gep.a> (!vplan.execution.frequency 2305843009213693952 (25%))
; VPLAN0-NEXT:    Successor(s): latch
; VPLAN0-EMPTY:
; VPLAN0-NEXT:    latch:
; VPLAN0-NEXT:      EMIT ir<%iv.next> = add ir<%iv>, ir<1>
; VPLAN0-NEXT:      EMIT ir<%ec> = icmp eq ir<%iv.next>, ir<1024>
;
; After predication, %gep.a is executed unconditionally.
;
; MASKED-LABEL: VPlan for loop in 'if_then'
; MASKED:         if.then:
; MASKED-NEXT:      EMIT ir<%add> = add ir<%i>, ir<10>, ir<%c> (!vplan.execution.frequency 2305843009213693952 (25%))
; MASKED-NEXT:      EMIT ir<%gep.a> = getelementptr inbounds ir<%a>, ir<%iv>{{$}}
; MASKED-NEXT:      EMIT store ir<%add>, ir<%gep.a>, ir<%c> (!vplan.execution.frequency 2305843009213693952 (25%))
; MASKED-NEXT:    Successor(s): latch
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %gep.idx = getelementptr inbounds i32, ptr %idx, i64 %iv
  %i = load i32, ptr %gep.idx, align 4
  %c = icmp sgt i32 %i, 0
  br i1 %c, label %if.then, label %latch, !prof !0

if.then:
  %add = add i32 %i, 10
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 %add, ptr %gep.a, align 4
  br label %latch

latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1024
  br i1 %ec, label %exit, label %loop, !prof !1

exit:
  ret void
}

; Both arms of the diamond are conditionally executed, with the frequencies of
; 1/4 and 3/4 recorded on their respective recipes.
define void @if_else(ptr noalias %a, ptr noalias %b, ptr noalias %idx) {
; VPLAN0-LABEL: VPlan for loop in 'if_else'
; VPLAN0:         loop:
; VPLAN0-NEXT:      EMIT-SCALAR ir<%iv> = phi [ ir<0>, vector.ph ], [ ir<%iv.next>, latch ]
; VPLAN0-NEXT:      EMIT ir<%gep.idx> = getelementptr inbounds ir<%idx>, ir<%iv>
; VPLAN0-NEXT:      EMIT-SCALAR ir<%i> = load ir<%gep.idx>
; VPLAN0-NEXT:      EMIT ir<%c> = icmp sgt ir<%i>, ir<0>
; VPLAN0-NEXT:      EMIT branch-on-cond ir<%c> (!prof {1, 3})
; VPLAN0-NEXT:    Successor(s): then, else
; VPLAN0-EMPTY:
; VPLAN0-NEXT:    else:
; VPLAN0-NEXT:      EMIT ir<%gep.b> = getelementptr inbounds ir<%b>, ir<%iv> (!vplan.execution.frequency 6917529027641081856 (75%))
; VPLAN0-NEXT:      EMIT store ir<%i>, ir<%gep.b> (!vplan.execution.frequency 6917529027641081856 (75%))
; VPLAN0-NEXT:    Successor(s): latch
; VPLAN0-EMPTY:
; VPLAN0-NEXT:    then:
; VPLAN0-NEXT:      EMIT ir<%gep.a> = getelementptr inbounds ir<%a>, ir<%iv> (!vplan.execution.frequency 2305843009213693952 (25%))
; VPLAN0-NEXT:      EMIT store ir<%i>, ir<%gep.a> (!vplan.execution.frequency 2305843009213693952 (25%))
; VPLAN0-NEXT:    Successor(s): latch
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %gep.idx = getelementptr inbounds i32, ptr %idx, i64 %iv
  %i = load i32, ptr %gep.idx, align 4
  %c = icmp sgt i32 %i, 0
  br i1 %c, label %then, label %else, !prof !0

then:
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 %i, ptr %gep.a, align 4
  br label %latch

else:
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv
  store i32 %i, ptr %gep.b, align 4
  br label %latch

latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1024
  br i1 %ec, label %exit, label %loop, !prof !1

exit:
  ret void
}

!0 = !{!"branch_weights", i32 1, i32 3}
!1 = !{!"branch_weights", i32 1, i32 999}
