; RUN: opt < %s -mtriple=powerpc64le-unknown-linux-gnu -S -passes=inline | FileCheck %s
; Check that we only inline when we have compatible target features.

target datalayout = "e-m:e-Fn32-i64:64-n32:64-S128-v256:256:256-v512:512:512"
target triple = "powerpc64le-unknown-linux-gnu"

define i32 @f1() #0 {
; CHECK-LABEL: define i32 @f1(
; CHECK-NEXT:    [[CALL:%.*]] = call i32 (...) @f0()
; CHECK-NEXT:    ret i32 [[CALL]]
;
  %call = call i32 (...) @f0()
  ret i32 %call
}

define i32 @f2() #1 {
; CHECK-LABEL: define i32 @f2(
; CHECK-NEXT:    [[CALL_I:%.*]] = call i32 (...) @f0()
; CHECK-NEXT:    ret i32 [[CALL_I]]
;
  %call = call i32 @f1()
  ret i32 %call
}

define i32 @f3() #0 {
; CHECK-LABEL: define i32 @f3(
; CHECK-NEXT:    [[CALL:%.*]] = call i32 @f2()
; CHECK-NEXT:    ret i32 [[CALL]]
;
  %call = call i32 @f2()
  ret i32 %call
}

declare i32 @f0(...) #0

attributes #0 = { "target-cpu"="pwr7" "target-features"="-crbits,-crypto,-direct-move,-isa-v207-instructions,-power8-vector" }
attributes #1 = { "target-cpu"="pwr8" "target-features"="+crbits,+crypto,+direct-move,+isa-v207-instructions,+power8-vector" }
