; RUN: opt -simplifycfg -S  < %s | FileCheck %s

declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32) #1

define i32 @f(i32 %a, i32 %b) #0 {
entry:
  %uadd = tail call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %a, i32 %b)
  %cmp = extractvalue { i32, i1 } %uadd, 1
  br i1 %cmp, label %return, label %if.end

if.end:                                           ; preds = %entry
  %0 = extractvalue { i32, i1 } %uadd, 0
  br label %return

return:                                           ; preds = %entry, %if.end
  %retval.0 = phi i32 [ %0, %if.end ], [ 0, %entry ]
  ret i32 %retval.0

; CHECK-LABEL: @f(
; CHECK-NOT: phi
; CHECK: select
}
