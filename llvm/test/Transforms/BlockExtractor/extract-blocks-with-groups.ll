; Extract the 'if', 'then', and 'else' blocks into the same function.
; RUN: echo 'foo if;then;else' > %t
; Make sure we can still extract a single basic block
; RUN: echo 'foo end' >> %t
; RUN: echo 'bar bb14;bb20' >> %t
; RUN: opt -S -extract-blocks -extract-blocks-file=%t %s | FileCheck %s

; CHECK-LABEL: foo
;
; CHECK: if:
; The diamond will produce an i32 value, make sure we account for that.
; CHECK: [[RES_VAL_ADDR:%[^ ]*]] = alloca i32
; CHECK-NEXT: br label %[[FOO_DIAMOND_LABEL:.*$]]
;
; The if-then-else blocks should be in just one function.
; CHECK: [[FOO_DIAMOND_LABEL]]:
; CHECK: call void [[FOO_DIAMOND:@[^(]*]](i32 %arg1, i32 %arg, ptr [[RES_VAL_ADDR]])
; CHECK-NEXT: [[RES_VAL:%[^ ]*]] = load i32, ptr [[RES_VAL_ADDR]]
; Then it should directly jump to end.
; CHECK: br label %[[FOO_END_LABEL:.*$]]
;
; End should have been extracted into its own function.
; CHECK: [[FOO_END_LABEL]]:
; CHECK: [[CMP:%[^ ]*]] = call i1 [[FOO_END:@[^(]*]](i32 [[RES_VAL]], i32 %arg)
; CHECK-NEXT: br i1 [[CMP]], label %ret0, label %ret1
define i32 @foo(i32 %arg, i32 %arg1) {
if:
  %tmp5 = icmp sgt i32 %arg, 0
  %tmp8 = icmp sgt i32 %arg1, 0
  %or.cond = and i1 %tmp5, %tmp8
  br i1 %or.cond, label %then, label %else

then:
  %tmp12 = shl i32 %arg1, 2
  %tmp13 = add nsw i32 %tmp12, %arg
  br label %end

else:
  %tmp22 = mul nsw i32 %arg, 3
  %tmp24 = sdiv i32 %arg1, 6
  %tmp25 = add nsw i32 %tmp24, %tmp22
  br label %end

end:
  %tmp.0 = phi i32 [ %tmp13, %then ], [ %tmp25, %else ]
  %and0 = and i32 %tmp.0, %arg
  %cmp1 = icmp slt i32 %and0, 0
  br i1 %cmp1, label %ret0, label %ret1

ret0:
  ret i32 0

ret1:
  ret i32 1
}

; CHECK-LABEL: bar
;
; Check that we extracted bb14 and bb20 in their own (shared) function.
; CHECK: bb
; CHECK: br i1 %or.cond, label %bb9, label %[[BAR_DIAMOND_LABEL:.*$]]
;
; CHECK: [[BAR_DIAMOND_LABEL]]:
; CHECK: [[CMP:%[^ ]*]] = call i1 [[BAR_DIAMOND:@[^(]*]](i32 %arg1, i32 %arg, ptr
; CHECK: br i1 [[CMP]], label %bb26, label %bb30
define i32 @bar(i32 %arg, i32 %arg1) {
bb:
  %tmp5 = icmp sgt i32 %arg, 0
  %tmp8 = icmp sgt i32 %arg1, 0
  %or.cond = and i1 %tmp5, %tmp8
  br i1 %or.cond, label %bb9, label %bb14

bb9:                                              ; preds = %bb
  %tmp12 = shl i32 %arg1, 2
  %tmp13 = add nsw i32 %tmp12, %arg
  br label %bb30

bb14:                                             ; preds = %bb
  %0 = and i32 %arg1, %arg
  %1 = icmp slt i32 %0, 0
  br i1 %1, label %bb20, label %bb26

bb20:                                             ; preds = %bb14
  %tmp22 = mul nsw i32 %arg, 3
  %tmp24 = sdiv i32 %arg1, 6
  %tmp25 = add nsw i32 %tmp24, %tmp22
  br label %bb30

bb26:                                             ; preds = %bb14
  %tmp29 = sub nsw i32 %arg, %arg1
  br label %bb30

bb30:                                             ; preds = %bb26, %bb20, %bb9
  %tmp.0 = phi i32 [ %tmp13, %bb9 ], [ %tmp25, %bb20 ], [ %tmp29, %bb26 ]
  ret i32 %tmp.0
}

; Check that we extracted the three asked basic blocks.
; CHECK: [[FOO_DIAMOND]]
; CHECK: then:
; Make sure the function doesn't end in the middle of the
; list of the blocks we are checking.
; CHECK-NOT: }
; CHECK: else:
; CHECK-NOT: }
; The name of the if block is weird because it had to be split
; since it was the name of the entry of the original function.
; CHECK: if.split:
; CHECK: }

; CHECK: [[FOO_END]]
; CHECK-NOT: }
; CHECK: end:
; CHECK: }

; Check that we extracted the two asked basic blocks.
; CHECK: [[BAR_DIAMOND]]
; CHECK-NOT: }
; CHECK: bb14:
; CHECK-NOT: }
; CHECK: bb20:
; CHECK: }
