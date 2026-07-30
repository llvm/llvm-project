; RUN: llc -O3 -o - %s | FileCheck %s
; Test case from PR16882.
target triple = "thumbv7a-none-eabi"

define i32 @test1() {
; CHECK-LABEL: @test1
; CHECK-NOT: push
entry:
  tail call void @overflow() #0
  unreachable
}

; Function Attrs: noreturn nounwind
declare void @overflow() #0

define i32 @test2(i32 %x, i32 %y) {
; CHECK-LABEL: @test2
; CHECK-NOT: push
; CHECK-NOT: pop
entry:
  %conv = sext i32 %x to i64
  %conv1 = sext i32 %y to i64
  %mul = mul nsw i64 %conv1, %conv
  %conv2 = trunc i64 %mul to i32
  %conv3 = sext i32 %conv2 to i64
  %cmp = icmp eq i64 %mul, %conv3
  br i1 %cmp, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  tail call void @overflow() #0
  unreachable

if.end:                                           ; preds = %entry
  ret i32 %conv2
}

; Test case for PR17825.
define i32 @test3() {
; CHECK-LABEL: @test3
; CHECK: push
entry:
  tail call void @overflow_with_unwind() #1
  unreachable
}

; Test case for uwtable
define i32 @test4() uwtable {
; CHECK-LABEL: @test4
; CHECK: push
entry:
  tail call void @overflow() #0
  unreachable
}

define i32 @test5() uwtable {
; CHECK-LABEL: @test5
; CHECK: push
entry:
  tail call void @overflow_with_unwind() #1
  unreachable
}


define i32 @test1_nofpelim() "frame-pointer"="all" {
; CHECK-LABEL: @test1_nofpelim
; CHECK: push
entry:
  tail call void @overflow() #0
  unreachable
}

define i32 @test2_nofpelim(i32 %x, i32 %y) "frame-pointer"="all" {
; CHECK-LABEL: @test2_nofpelim
; CHECK: push
entry:
  %conv = sext i32 %x to i64
  %conv1 = sext i32 %y to i64
  %mul = mul nsw i64 %conv1, %conv
  %conv2 = trunc i64 %mul to i32
  %conv3 = sext i32 %conv2 to i64
  %cmp = icmp eq i64 %mul, %conv3
  br i1 %cmp, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  tail call void @overflow() #0
  unreachable

if.end:                                           ; preds = %entry
  ret i32 %conv2
}

; Test case for PR17825.
define i32 @test3_nofpelim() "frame-pointer"="all" {
; CHECK-LABEL: @test3_nofpelim
; CHECK: push
entry:
  tail call void @overflow_with_unwind() #1
  unreachable
}

; Test case for uwtable
define i32 @test4_nofpelim() uwtable "frame-pointer"="all" {
; CHECK-LABEL: @test4_nofpelim
; CHECK: push
entry:
  tail call void @overflow() #0
  unreachable
}

define i32 @test5_nofpelim() uwtable "frame-pointer"="all" {
; CHECK-LABEL: @test5_nofpelim
; CHECK: push
entry:
  tail call void @overflow_with_unwind() #1
  unreachable
}

; Function Attrs: noreturn
declare void @overflow_with_unwind() #1

attributes #0 = { noreturn nounwind }
attributes #1 = { noreturn }
