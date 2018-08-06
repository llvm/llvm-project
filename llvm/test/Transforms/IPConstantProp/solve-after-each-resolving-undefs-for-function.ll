; RUN: opt < %s -ipsccp -S | FileCheck %s

; CHECK-LABEL: @testf(
; CHECK:         ret i32 undef
;
define internal i32 @testf() {
entry:
  br i1 undef, label %if.then, label %if.end

if.then:                                          ; preds = %entry, %if.then
  br label %if.end

if.end:                                          ; preds = %if.then1, %entry
  ret i32 10
}

; CHECK-LABEL: @test1(
; CHECK:         ret i32 undef
;
define internal i32 @test1() {
entry:
  br label %if.then

if.then:                                          ; preds = %entry, %if.then
  %call = call i32 @testf()
  %res = icmp eq i32 %call, 10
  br i1 %res, label %ret1, label %ret2

ret1:                                           ; preds = %if.then, %entry
  ret i32 99

ret2:                                           ; preds = %if.then, %entry
  ret i32 0
}

; CHECK-LABEL: @main(
; CHECK-NEXT:    %res = call i32 @test1()
; CHECK-NEXT:    ret i32 99
;
define i32 @main() {
  %res = call i32 @test1()
  ret i32 %res
}
