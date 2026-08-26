; RUN: opt -passes=correlated-propagation -S < %s | FileCheck %s

; When a call argument is a constant C and a function parameter is known to
; equal C at the call site (due to a guarding icmp eq), replace C with the
; parameter so the backend can reuse the already-loaded register.
; Issue: https://github.com/llvm/llvm-project/issues/195907

; Basic case: i32 0 replaced with %a (same type, known equal).
; i64 0 is NOT replaced because %a is i32 (type mismatch).
define void @test_basic(i32 %a) {
; CHECK-LABEL: @test_basic(
; CHECK:       if.then:
; CHECK-NEXT:    call void @fdecl_0(i32 %a, i64 0)
entry:
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %if.then, label %if.end
if.then:
  call void @fdecl_0(i32 0, i64 0)
  br label %if.end
if.end:
  ret void
}

; Non-zero constant: %a == 5, so 5 is replaced with %a.
define void @test_nonzero(i32 %a) {
; CHECK-LABEL: @test_nonzero(
; CHECK:       if.then:
; CHECK-NEXT:    call void @fdecl_i32(i32 %a)
entry:
  %cmp = icmp eq i32 %a, 5
  br i1 %cmp, label %if.then, label %if.end
if.then:
  call void @fdecl_i32(i32 5)
  br label %if.end
if.end:
  ret void
}

; Negative: no condition guards the constant -- leave it alone.
define void @test_no_condition(i32 %a) {
; CHECK-LABEL: @test_no_condition(
; CHECK:    call void @fdecl_i32(i32 0)
  call void @fdecl_i32(i32 0)
  ret void
}

; Negative: type mismatch -- %a is i32 but call arg is i64.
define void @test_type_mismatch(i32 %a) {
; CHECK-LABEL: @test_type_mismatch(
; CHECK:       if.then:
; CHECK-NEXT:    call void @fdecl_i64(i64 0)
entry:
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %if.then, label %if.end
if.then:
  call void @fdecl_i64(i64 0)
  br label %if.end
if.end:
  ret void
}

; Two params: both replaceable when the branch depends on both being equal.
define void @test_two_params(i32 %a, i32 %b) {
; CHECK-LABEL: @test_two_params(
; CHECK:       if.then:
; CHECK-NEXT:    call void @fdecl_two_i32(i32 %a, i32 %b)
entry:
  %cmp_a = icmp eq i32 %a, 1
  %cmp_b = icmp eq i32 %b, 2
  %and = and i1 %cmp_a, %cmp_b
  br i1 %and, label %if.then, label %if.end
if.then:
  call void @fdecl_two_i32(i32 1, i32 2)
  br label %if.end
if.end:
  ret void
}

declare void @fdecl_0(i32, i64)
declare void @fdecl_i32(i32)
declare void @fdecl_i64(i64)
declare void @fdecl_two_i32(i32, i32)