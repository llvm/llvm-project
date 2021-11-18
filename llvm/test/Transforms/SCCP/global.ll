; RUN: opt < %s -S -passes=ipsccp | FileCheck %s

@_ZL6test1g = internal global i32 42, align 4

define void @_Z7test1f1v() nounwind {
entry:
  %tmp = load i32, i32* @_ZL6test1g, align 4
  %cmp = icmp eq i32 %tmp, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 0, i32* @_ZL6test1g, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; CHECK: @_Z7test1f2v()
; CHECK: entry:
; CHECK-NEXT: ret i32 42
define i32 @_Z7test1f2v() nounwind {
entry:
  %tmp = load i32, i32* @_ZL6test1g, align 4
  ret i32 %tmp
}
