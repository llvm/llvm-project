; RUN: opt -S -passes='assign-guid,wholeprogramdevirt' -devirtualize-speculatively %s | FileCheck %s
; RUN: opt -S -passes='assign-guid,wholeprogramdevirt' -whole-program-visibility -wholeprogramdevirt-check=fallback %s | FileCheck %s

@vt = constant [1 x ptr] [ptr @func1], !type !0

define i32 @func1(ptr %this) {
  ret i32 1
}

; CHECK-LABEL: define i32 @call(
; CHECK: br i1 {{.*}}, label %if.true.direct_targ, label %if.false.orig_indirect, !prof ![[WEIGHTS:[0-9]+]]
; CHECK: if.true.direct_targ:
; CHECK-NEXT: call i32 @func1(
; CHECK: if.false.orig_indirect:
; CHECK-NEXT: call i32 %fptr(
define i32 @call(ptr %obj) !prof !3 {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  %result = call i32 %fptr(ptr %obj), !prof !1
  ret i32 %result
}

; No value profile on the call: the check stays likely.
; CHECK-LABEL: define i32 @call_no_profile(
; CHECK: br i1 {{.*}}, label %if.true.direct_targ, label %if.false.orig_indirect, !prof ![[NOPROF_WEIGHTS:[0-9]+]]
; CHECK: if.true.direct_targ:
; CHECK-NEXT: call i32 @func1(
; CHECK: if.false.orig_indirect:
; CHECK-NEXT: call i32 %fptr(
define i32 @call_no_profile(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  %result = call i32 %fptr(ptr %obj)
  ret i32 %result
}

; A value profile exists but never names func1: stays likely.
; CHECK-LABEL: define i32 @call_other_target(
; CHECK: br i1 {{.*}}, label %if.true.direct_targ, label %if.false.orig_indirect, !prof ![[NOPROF_WEIGHTS]]
; CHECK: if.true.direct_targ:
; CHECK-NEXT: call i32 @func1(
; CHECK: if.false.orig_indirect:
; CHECK-NEXT: call i32 %fptr(
define i32 @call_other_target(ptr %obj) !prof !3 {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  %result = call i32 %fptr(ptr %obj), !prof !2
  ret i32 %result
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

; CHECK: ![[WEIGHTS]] = !{!"branch_weights", i32 40, i32 60}
; CHECK: ![[NOPROF_WEIGHTS]] = !{!"branch_weights", i32 1048575, i32 1}
!0 = !{i32 0, !"typeid"}
!1 = !{!"VP", i32 0, i64 100, i64 -2545542355363006406, i64 40}
!2 = !{!"VP", i32 0, i64 100, i64 123456789, i64 60}
!3 = !{!"function_entry_count", i64 100}
