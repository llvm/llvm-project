; RUN: opt < %s -licm -loop-unroll -disable-output
; Test triggered an assertion in doFinalization() because loop unroll was deleting
; the inner loop which caused the loop to not get removed from the
; LoopToAliasSetMap.
; Test case taken from test/Transforms/LoopUnroll/unloop.ll.

declare i1 @check() nounwind
define void @skiplevelexit() nounwind {
entry:
  br label %outer

outer:
  br label %inner

inner:
  %iv = phi i32 [ 0, %outer ], [ %inc, %tail ]
  %inc = add i32 %iv, 1
  call zeroext i1 @check()
  br i1 true, label %outer.backedge, label %tail

tail:
  br i1 false, label %inner, label %exit

outer.backedge:
  br label %outer

exit:
  ret void
}

