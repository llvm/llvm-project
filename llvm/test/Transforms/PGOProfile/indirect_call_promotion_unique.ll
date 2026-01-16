; RUN: opt < %s -passes=pgo-icall-prom -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@foo = common global ptr null, align 8

; The names on the IR and in the profile are both "func1".
define i32 @func1() {
entry:
  ret i32 1
}

define i32 @bar1() {
entry:
  %tmp1 = load ptr, ptr @foo, align 8
; CHECK: icmp eq ptr %tmp1, @func1
  %call = call i32 %tmp1(), !prof !1
  ret i32 %call
}

; The name on the IR has ".llvm." suffix: "func2.llvm.10895321227755557127".
; The name in the profile has no ".llvm." suffix: "func2".
define i32 @func2.llvm.10895321227755557127() {
entry:
  ret i32 1
}

define i32 @bar2() {
entry:
  %tmp2 = load ptr, ptr @foo, align 8
; CHECK: icmp eq ptr %tmp2, @func2.llvm.10895321227755557127
  %call = call i32 %tmp2(), !prof !2
  ret i32 %call
}

; The names on the IR and in the profile are both
; "func3.__uniq.258901567653530696343884446915951489119".
define i32 @func3.__uniq.258901567653530696343884446915951489119() {
entry:
  ret i32 2
}

define i32 @bar3() {
entry:
  %tmp3 = load ptr, ptr @foo, align 8
; CHECK: icmp eq ptr %tmp3, @func3.__uniq.258901567653530696343884446915951489119
  %call = call i32 %tmp3(), !prof !3
  ret i32 %call
}

; The name on the IR has ".__uniq." and ".llvm." suffix:
; "func4.__uniq.140291095734751150107370763113257199296.llvm.10650195578168450516".
; The name in the profile has ".__uniq." but no ".llvm." suffix:
; "func4.__uniq.140291095734751150107370763113257199296".
define i32 @func4.__uniq.140291095734751150107370763113257199296.llvm.10650195578168450516() {
entry:
  ret i32 3
}

define i32 @bar4() {
entry:
  %tmp4 = load ptr, ptr @foo, align 8
; CHECK: icmp eq ptr %tmp4, @func4.__uniq.140291095734751150107370763113257199296.llvm.10650195578168450516
  %call = call i32 %tmp4(), !prof !4
  ret i32 %call
}

; The name on the IR has ".__uniq.", ".part." and ".llvm." suffix:
; "func4.__uniq.127882361580787111523790444488985774976.part.818292359123831.llvm.10650195578168450516".
; The name in the profile has ".__uniq." but no ".llvm." and no ".part." suffix:
; "func4.__uniq.127882361580787111523790444488985774976".
define i32 @func5.__uniq.127882361580787111523790444488985774976.part.818292359123831.llvm.10650195578168450516() {
entry:
  ret i32 3
}

define i32 @bar5() {
entry:
  %tmp5 = load ptr, ptr @foo, align 8
; CHECK: icmp eq ptr %tmp5, @func5.__uniq.127882361580787111523790444488985774976.part.818292359123831.llvm.10650195578168450516
  %call = call i32 %tmp5(), !prof !5
  ret i32 %call
}

; Check symbol with '.' in the front is handled properly.
define i32 @..func6() {
entry:
  ret i32 3
}

; IR has coro function "func7" and await suspend wrapper "func7.__await_suspend_wrapper__init",
; Profile only has the coro function name func7.
; func7.__await_suspend_wrapper__init is defined before func7 to be matched first.
; Note: ignore the incomplete function body, we only care about the names for test purpose.
define i32 @func7.__await_suspend_wrapper__init() {
entry:
  ret i32 3
}

define i32 @func7() {
entry:
  ret i32 3
}

define i32 @bar7() {
entry:
  %tmp5 = load ptr, ptr @foo, align 8
; CHECK: icmp eq ptr %tmp5, @func7{{$}}
  %call = call i32 %tmp5(), !prof !6
  ret i32 %call
}

; GUID of "func1" is -2545542355363006406.
; GUID of "func2" is -4377547752858689819.
; GUID of "func3.__uniq.258901567653530696343884446915951489119" is 8271224222042874235.
; GUID of "func4.__uniq.140291095734751150107370763113257199296" is 1491826207425861106.
; GUID of "func5.__uniq.127882361580787111523790444488985774976" is -4238550483433487304.
; GUID of "func7" is 4143431755475214830.
!1 = !{!"VP", i32 0, i64 1600, i64 -2545542355363006406, i64 1600}
!2 = !{!"VP", i32 0, i64 1600, i64 -4377547752858689819, i64 1600}
!3 = !{!"VP", i32 0, i64 1600, i64 8271224222042874235, i64 1600}
!4 = !{!"VP", i32 0, i64 1600, i64 1491826207425861106, i64 1600}
!5 = !{!"VP", i32 0, i64 1600, i64 -4238550483433487304, i64 1600}
!6 = !{!"VP", i32 0, i64 1600, i64 4143431755475214830, i64 1600}
