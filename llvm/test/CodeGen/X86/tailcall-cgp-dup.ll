; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s

; Teach CGP to dup returns to enable tail call optimization.
; rdar://9147433

define i32 @foo(i32 %x) nounwind ssp {
; CHECK: foo:
entry:
  switch i32 %x, label %return [
    i32 1, label %sw.bb
    i32 2, label %sw.bb1
    i32 3, label %sw.bb3
    i32 4, label %sw.bb5
    i32 5, label %sw.bb7
    i32 6, label %sw.bb9
  ]

sw.bb:                                            ; preds = %entry
; CHECK: jmp _f1
  %call = tail call i32 @f1() nounwind
  br label %return

sw.bb1:                                           ; preds = %entry
; CHECK: jmp _f2
  %call2 = tail call i32 @f2() nounwind
  br label %return

sw.bb3:                                           ; preds = %entry
; CHECK: jmp _f3
  %call4 = tail call i32 @f3() nounwind
  br label %return

sw.bb5:                                           ; preds = %entry
; CHECK: jmp _f4
  %call6 = tail call i32 @f4() nounwind
  br label %return

sw.bb7:                                           ; preds = %entry
; CHECK: jmp _f5
  %call8 = tail call i32 @f5() nounwind
  br label %return

sw.bb9:                                           ; preds = %entry
; CHECK: jmp _f6
  %call10 = tail call i32 @f6() nounwind
  br label %return

return:                                           ; preds = %entry, %sw.bb9, %sw.bb7, %sw.bb5, %sw.bb3, %sw.bb1, %sw.bb
  %retval.0 = phi i32 [ %call10, %sw.bb9 ], [ %call8, %sw.bb7 ], [ %call6, %sw.bb5 ], [ %call4, %sw.bb3 ], [ %call2, %sw.bb1 ], [ %call, %sw.bb ], [ 0, %entry ]
  ret i32 %retval.0
}

declare i32 @f1()

declare i32 @f2()

declare i32 @f3()

declare i32 @f4()

declare i32 @f5()

declare i32 @f6()

; rdar://11958338
%0 = type opaque

declare i8* @bar(i8*) uwtable optsize noinline ssp

define hidden %0* @thingWithValue(i8* %self) uwtable ssp {
entry:
; CHECK: thingWithValue:
; CHECK: jmp _bar
  br i1 undef, label %if.then.i, label %if.else.i

if.then.i:                                        ; preds = %entry
  br label %someThingWithValue.exit

if.else.i:                                        ; preds = %entry
  %call4.i = tail call i8* @bar(i8* undef) optsize
  br label %someThingWithValue.exit

someThingWithValue.exit:                          ; preds = %if.else.i, %if.then.i
  %retval.0.in.i = phi i8* [ undef, %if.then.i ], [ %call4.i, %if.else.i ]
  %retval.0.i = bitcast i8* %retval.0.in.i to %0*
  ret %0* %retval.0.i
}
