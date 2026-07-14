; RUN: opt < %s -passes=tailcallelim -disable-tail-call-elim-for-cold-calls=true -S | FileCheck %s --check-prefixes=CHECK,DISABLED
; RUN: opt < %s -passes=tailcallelim -disable-tail-call-elim-for-cold-calls=false -S | FileCheck %s --check-prefixes=CHECK,ENABLED

declare void @cold_callee() cold
declare void @normal_callee()
declare coldcc void @coldcc_callee()

; Check that a call to a cold callee is not marked as tail when the flag is enabled.
define void @test_cold_callee() {
; CHECK-LABEL: @test_cold_callee(
; DISABLED: call void @cold_callee()
; ENABLED: tail call void @cold_callee()
  call void @cold_callee()
  ret void
}

; Check that a call to a callee with coldcc is not marked as tail when the flag is enabled.
define void @test_coldcc_callee() {
; CHECK-LABEL: @test_coldcc_callee(
; DISABLED: call coldcc void @coldcc_callee()
; ENABLED: tail call coldcc void @coldcc_callee()
  call coldcc void @coldcc_callee()
  ret void
}

; Check that a call inside a cold enclosing function is not marked as tail when the flag is enabled.
define void @test_cold_caller() cold {
; CHECK-LABEL: @test_cold_caller(
; DISABLED: call void @normal_callee()
; ENABLED: tail call void @normal_callee()
  call void @normal_callee()
  ret void
}

; Check that a callsite with coldcc is not marked as tail when the flag is enabled.
define void @test_coldcc_callsite() {
; CHECK-LABEL: @test_coldcc_callsite(
; DISABLED: call coldcc void @normal_callee()
; ENABLED: tail call coldcc void @normal_callee()
  call coldcc void @normal_callee()
  ret void
}

; Check that mandatory musttail calls are never disabled even when calling a cold function and the flag is true.
define void @test_musttail_cold_callee() {
; CHECK-LABEL: @test_musttail_cold_callee(
; CHECK: musttail call void @cold_callee()
  musttail call void @cold_callee()
  ret void
}

; Check that tail recursion elimination to a loop header is disabled in a cold function when the flag is enabled.
define i32 @test_recursive_cold_caller(i32 %X) cold {
; CHECK-LABEL: @test_recursive_cold_caller(
; DISABLED: call i32 @test_recursive_cold_caller(
; DISABLED-NOT: br label %tailrecurse
; ENABLED: tailrecurse:
; ENABLED-NOT: call i32 @test_recursive_cold_caller(
entry:
  %cmp = icmp eq i32 %X, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:
  ret i32 0

if.else:
  %dec = sub i32 %X, 1
  %call = call i32 @test_recursive_cold_caller(i32 %dec)
  ret i32 %call
}
