; RUN: opt -passes=always-inline -S < %s | FileCheck %s
; RUN: opt -passes=always-inline -pass-remarks-missed=inline -S < %s 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: opt -passes=inline -S < %s | FileCheck %s
; RUN: opt -passes='cgscc(inline<only-mandatory>)' -S < %s | FileCheck %s

; Test that the flatten attribute recursively inlines all calls.

; Multiple levels are inlined.
define internal i32 @leaf() {
  ret i32 42
}

define internal i32 @middle() {
  %r = call i32 @leaf()
  ret i32 %r
}

define i32 @test_multilevel() flatten {
; CHECK-LABEL: @test_multilevel(
; CHECK-NOT: call i32 @middle
; CHECK-NOT: call i32 @leaf
; CHECK: ret i32 42
  %r = call i32 @middle()
  ret i32 %r
}

; Functions with invoke are inlined.
declare i32 @__gxx_personality_v0(...)
declare void @may_throw()

define internal i32 @callee_with_invoke() personality ptr @__gxx_personality_v0 {
entry:
  invoke void @may_throw() to label %cont unwind label %lpad
cont:
  ret i32 100
lpad:
  %lp = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %lp
}

define i32 @test_invoke() flatten personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: @test_invoke(
; CHECK-NOT: call i32 @callee_with_invoke
; CHECK: invoke void @may_throw()
; CHECK: ret i32 100
entry:
  %r = call i32 @callee_with_invoke()
  ret i32 %r
}

; Declaration without definition is not inlined.
declare i32 @external_func()

define i32 @test_declaration() flatten {
; CHECK-LABEL: @test_declaration(
; CHECK: call i32 @external_func()
; CHECK: ret i32
  %r = call i32 @external_func()
  ret i32 %r
}

; Inlined callee that calls a declaration - the declaration should remain after flattening.
define internal i32 @calls_external() {
  %r = call i32 @external_func()
  ret i32 %r
}

define i32 @test_inline_then_declaration() flatten {
; CHECK-LABEL: @test_inline_then_declaration(
; CHECK-NOT: call i32 @calls_external()
; CHECK: call i32 @external_func()
; CHECK: ret i32
  %r = call i32 @calls_external()
  ret i32 %r
}

; Indirect calls are not inlined.
define internal i32 @target_func() {
  ret i32 99
}

define i32 @test_indirect(ptr %func_ptr) flatten {
; CHECK-LABEL: @test_indirect(
; CHECK: call i32 %func_ptr()
; CHECK: ret i32
  %r = call i32 %func_ptr()
  ret i32 %r
}

; Direct recursion back to flattened function.
; The callee calls the flattened function - should not cause infinite inlining.
define internal i32 @calls_flattened_func() {
  %r = call i32 @test_direct_recursion()
  ret i32 %r
}

define i32 @test_direct_recursion() flatten {
; CHECK-LABEL: @test_direct_recursion(
; The call to calls_flattened_func should be inlined, but the recursive call back
; to test_direct_recursion should remain.
; CHECK-NOT: call i32 @calls_flattened_func()
; CHECK: call i32 @test_direct_recursion()
; CHECK: ret i32
  %r = call i32 @calls_flattened_func()
  ret i32 %r
}

; Mutual recursion (A calls B, B calls A).
; Should inline once but not infinitely.
define internal i32 @mutual_a() {
  %r = call i32 @mutual_b()
  ret i32 %r
}

define internal i32 @mutual_b() {
  %r = call i32 @mutual_a()
  ret i32 %r
}

define i32 @test_mutual_recursion() flatten {
; CHECK-LABEL: @test_mutual_recursion(
; After inlining mutual_a, we get call to mutual_b.
; After inlining mutual_b, we get call to mutual_a which should remain (skipped due to recursion).
; CHECK-NOT: call i32 @mutual_b()
; CHECK: call i32 @mutual_a()
; CHECK: ret i32
  %r = call i32 @mutual_a()
  ret i32 %r
}

; Recursive callee via indirection.
; A function that is part of a recursive cycle should be inlined once but not infinitely.
; Note: Direct self-recursive functions (f calls f) are not inlineable in LLVM.
; So we test with mutual recursion pattern where each function individually is viable.
define internal i32 @recursive_a(i32 %n) {
  %r = call i32 @recursive_b(i32 %n)
  ret i32 %r
}

define internal i32 @recursive_b(i32 %n) {
  %r = call i32 @recursive_a(i32 %n)
  ret i32 %r
}

define i32 @test_self_recursion() flatten {
; CHECK-LABEL: @test_self_recursion(
; After inlining recursive_a (produces call to recursive_b with the original arg)
; After inlining recursive_b (produces call to recursive_a - skipped due to history)
; Both recursive_a and recursive_b should be inlined (CHECK-NOT matches any call to them)
; The remaining call is to recursive_a with the propagated constant.
; CHECK-NOT: call i32 @recursive_b
; CHECK: call i32 @recursive_a(i32 5)
; CHECK: ret i32
  %r = call i32 @recursive_a(i32 5)
  ret i32 %r
}

; Check that optimization remark is emitted for recursive calls during flattening.
; REMARK: remark: {{.*}} 'test_direct_recursion' is not inlined into 'test_direct_recursion': recursive call during flattening
