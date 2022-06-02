; RUN: llvm-extract -bb 'foo:if;then;else' -bb 'bar:bb14;bb20' -S %s  | FileCheck %s
; Extract two groups of basic blocks in two different functions.


; The first extracted function is the region composed by the
; blocks if, then, and else from foo.
; CHECK: define dso_local void @foo.if.split(i32 %arg1, i32 %arg, ptr %tmp.0.ce.out) {
; CHECK: newFuncRoot:
; CHECK:   br label %if.split
;
; CHECK: then:                                             ; preds = %if.split
; CHECK:   %tmp12 = shl i32 %arg1, 2
; CHECK:   %tmp13 = add nsw i32 %tmp12, %arg
; CHECK:   br label %end.split
;
; CHECK: else:                                             ; preds = %if.split
; CHECK:   %tmp22 = mul nsw i32 %arg, 3
; CHECK:   %tmp24 = sdiv i32 %arg1, 6
; CHECK:   %tmp25 = add nsw i32 %tmp24, %tmp22
; CHECK:   br label %end.split
;
; CHECK: if.split:                                         ; preds = %newFuncRoot
; CHECK:   %tmp5 = icmp sgt i32 %arg, 0
; CHECK:   %tmp8 = icmp sgt i32 %arg1, 0
; CHECK:   %or.cond = and i1 %tmp5, %tmp8
; CHECK:   br i1 %or.cond, label %then, label %else
;
; CHECK: end.split:                                        ; preds = %then, %else
; CHECK:   %tmp.0.ce = phi i32 [ %tmp13, %then ], [ %tmp25, %else ]
; CHECK:   store i32 %tmp.0.ce, ptr %tmp.0.ce.out
; CHECK:   br label %end.exitStub
;
; CHECK: end.exitStub:                                     ; preds = %end.split
; CHECK:   ret void
; CHECK: }

; The second extracted function is the region composed by the blocks
; bb14 and bb20 from bar.
; CHECK: define dso_local i1 @bar.bb14(i32 %arg1, i32 %arg, ptr %tmp25.out) {
; CHECK: newFuncRoot:
; CHECK:   br label %bb14
;
; CHECK: bb14:                                             ; preds = %newFuncRoot
; CHECK:   %tmp0 = and i32 %arg1, %arg
; CHECK:   %tmp1 = icmp slt i32 %tmp0, 0
; CHECK:   br i1 %tmp1, label %bb20, label %bb26.exitStub
;
; CHECK: bb20:                                             ; preds = %bb14
; CHECK:   %tmp22 = mul nsw i32 %arg, 3
; CHECK:   %tmp24 = sdiv i32 %arg1, 6
; CHECK:   %tmp25 = add nsw i32 %tmp24, %tmp22
; CHECK:   store i32 %tmp25, ptr %tmp25.out
; CHECK:   br label %bb30.exitStub
;
; CHECK: bb26.exitStub:                                    ; preds = %bb14
; CHECK:   ret i1 true
;
; CHECK: bb30.exitStub:                                    ; preds = %bb20
; CHECK:   ret i1 false
; CHECK: }

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
  %tmp0 = and i32 %arg1, %arg
  %tmp1 = icmp slt i32 %tmp0, 0
  br i1 %tmp1, label %bb20, label %bb26

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

