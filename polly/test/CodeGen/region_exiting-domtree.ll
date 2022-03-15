; RUN: opt %loadPolly -polly-codegen -verify-dom-info -disable-output < %s

; Verify that the DominatorTree is preserved correctly for the inserted
; %polly.stmt.exit.exit block, which serves as new exit block for the generated
; subregion. In particulat, it must be dominated by %polly.stmt.subregion.enter,
; the generated subregion's entry block.

define void @func(i32 %n, i32* noalias nonnull %A) {
entry:
  br label %loop

loop:
  %i = phi i32 [0, %entry], [%i.inc, %loop.inc]
  %i.cmp = icmp slt i32 %i, %n
  br i1 %i.cmp, label %body, label %return

body:
  %skipcond = icmp slt i32 %i, 5
  br i1 %skipcond, label %subregion.enter, label %subregion.skip

subregion.skip:
  br label %exit

subregion.enter:
  %sqr = mul i32 %i, %i
  %cond = icmp eq i32 %sqr, 0
  store i32 %i, i32* %A
  br i1 %cond, label %subregion.true, label %subregion.false

subregion.true:
  br label %exit

subregion.false:
  br label %exit

exit:
  br label %loop.inc

loop.inc:
  %i.inc = add nuw nsw i32 %i, 1
  br label %loop

return:
  ret void
}
