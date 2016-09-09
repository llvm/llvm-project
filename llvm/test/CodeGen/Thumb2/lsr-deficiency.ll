; RUN: llc < %s -mtriple=thumbv7-apple-darwin10 -relocation-model=pic | FileCheck %s
; rdar://7387640

; This now reduces to a single induction variable.

@G = external global i32                          ; <i32*> [#uses=2]
@array = external global i32*                     ; <i32**> [#uses=1]

define void @t() nounwind optsize {
; CHECK-LABEL: t:
; CHECK: mov{{.*}}, #1000
entry:
  %.pre = load i32, i32* @G, align 4                   ; <i32> [#uses=1]
  br label %bb

bb:                                               ; preds = %bb, %entry
; CHECK: LBB0_1:
; CHECK: subs [[R2:r[0-9]+]], #1
; CHECK: adds {{.*}}, [[R2]], #1
; CHECK: bne LBB0_1

  %0 = phi i32 [ %.pre, %entry ], [ %3, %bb ]     ; <i32> [#uses=1]
  %indvar = phi i32 [ 0, %entry ], [ %indvar.next, %bb ] ; <i32> [#uses=2]
  %tmp5 = sub i32 1000, %indvar                   ; <i32> [#uses=1]
  %1 = load i32*, i32** @array, align 4                 ; <i32*> [#uses=1]
  %scevgep = getelementptr i32, i32* %1, i32 %tmp5     ; <i32*> [#uses=1]
  %2 = load i32, i32* %scevgep, align 4                ; <i32> [#uses=1]
  %3 = add nsw i32 %2, %0                         ; <i32> [#uses=2]
  store i32 %3, i32* @G, align 4
  %indvar.next = add i32 %indvar, 1               ; <i32> [#uses=2]
  %exitcond = icmp eq i32 %indvar.next, 1001      ; <i1> [#uses=1]
  br i1 %exitcond, label %return, label %bb

return:                                           ; preds = %bb
  ret void
}
