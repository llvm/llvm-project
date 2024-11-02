; RUN: opt < %s -passes=gvn,jump-threading -enable-pre -S | FileCheck %s

@H = common global i32 0
@G = common global i32 0

define i32 @test(i1 %cond, i32 %v) nounwind {
; CHECK-LABEL: @test
entry:
  br i1 %cond, label %bb, label %bb1

bb:
; CHECK: store
; CHECK-NOT: br label %return
  %add.1 = add nuw nsw i32 %v, -1
  store i32 %add.1, i32* @G, align 4
  br label %merge

bb1:
  br label %merge

merge:
  %add.2 = add i32 %v, -1
  %cmp = icmp sgt i32 %add.2, 0
  br i1 %cmp, label %action, label %return

action:
; CHECK: store
; CHECK-NEXT: br label %return
  store i32 %add.2, i32* @H, align 4
  br label %return

return:
  %p = phi i32 [0, %merge], [1, %action]
  ret i32 %p
}

