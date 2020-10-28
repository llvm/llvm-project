; RUN: opt < %s -instcombine -S | FileCheck %s
;
; This test ensures that InstCombine does not distribute And over Xor
; using simplifications involving undef.

define zeroext i1 @foo(i32 %arg) {
; CHECK-LABEL: @foo(

entry:
  %cmp1 = icmp eq i32 %arg, 37
  br i1 %cmp1, label %bb_then, label %bb_else

bb_then:
  call void @bar()
  br label %bb_exit

bb_else:
  %cmp2 = icmp slt i32 %arg, 17
  br label %bb_exit

; CHECK:       bb_exit:
; CHECK-NEXT:    [[PHI1:%.*]] = phi i1 [ [[CMP2:%.*]], [[BB_ELSE:%.*]] ], [ undef, [[BB_THEN:%.*]] ]
; CHECK-NEXT:    [[XOR1:%.*]] = xor i1 [[CMP1:%.*]], true
; CHECK-NEXT:    [[AND1:%.*]] = and i1 [[PHI1]], [[XOR1]]
; CHECK-NEXT:    ret i1 [[AND1]]
bb_exit:
  %phi1 = phi i1 [ %cmp2, %bb_else ], [ undef, %bb_then ]
  %xor1 = xor i1 %cmp1, true
  %and1 = and i1 %phi1, %xor1
  ret i1 %and1
}

declare void @bar()
