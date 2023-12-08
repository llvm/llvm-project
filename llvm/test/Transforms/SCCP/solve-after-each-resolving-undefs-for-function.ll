; RUN: opt < %s -passes=ipsccp -S | FileCheck %s

; CHECK-LABEL: @testf(
; CHECK:         ret i32 undef
;
define internal i32 @testf(i1 %c) {
entry:
  br i1 %c, label %if.cond, label %if.end

if.cond:
  br i1 undef, label %if.then, label %if.end

if.then:                                          ; preds = %entry, %if.then
  ret i32 11

if.end:                                          ; preds = %if.then1, %entry
  ret i32 10
}

; CHECK-LABEL: @test1(
; CHECK:         ret i32 undef
;
define internal i32 @test1(i1 %c) {
entry:
  br label %if.then

if.then:                                          ; preds = %entry, %if.then
  %call = call i32 @testf(i1 %c)
  %res = icmp eq i32 %call, 10
  br i1 %res, label %ret1, label %ret2

ret1:                                           ; preds = %if.then, %entry
  ret i32 99

ret2:                                           ; preds = %if.then, %entry
  ret i32 0
}

; CHECK-LABEL: @main(
; CHECK-NEXT:    %res = call i32 @test1(
; CHECK-NEXT:    ret i32 99
;
define i32 @main(i1 %c) {
  %res = call i32 @test1(i1 %c)
  ret i32 %res
}
