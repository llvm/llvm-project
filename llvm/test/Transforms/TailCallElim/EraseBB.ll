; RUN: opt -tailcallelim -verify-dom-info -S < %s 2>&1 | FileCheck %s

; CHECK: add nsw i32
; CHECK-NEXT: br label
; CHECK: add nsw i32
; CHECK-NEXT: br label
; CHECK-NOT: Uses remain when a value is destroyed
define i32 @test(i32 %n) {
entry:
  %cmp = icmp slt i32 %n, 2
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %v1 = add nsw i32 %n, -2
  %call1 = tail call i32 @test(i32 %v1)
  br label %return

if.else:                                          ; preds = %entry
  %v2 = add nsw i32 %n, 4
  %call2 = tail call i32 @test(i32 %v2)
  br label %return

return:                                           ; preds = %if.end, %if.else
  %retval = phi i32 [ %call1, %if.then ], [ %call2, %if.else ]
  ret i32 %retval
}
