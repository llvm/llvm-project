; RUN: opt < %s -passes=inline -S | FileCheck %s
; RUN: opt < %s -passes='cgscc(inline,function(instcombine))' -S | FileCheck %s

declare void @ext_method(ptr, i32)
declare signext i16 @vararg_fn(...) #0
declare "cc 9" void @vararg_fn_cc9(ptr %p, ...)

define linkonce_odr void @thunk(ptr %this, ...) {
  %this_adj = getelementptr i8, ptr %this, i32 4
  musttail call void (ptr, ...) @ext_method(ptr nonnull %this_adj, ...)
  ret void
}

define void @thunk_caller(ptr %p) {
  call void (ptr, ...) @thunk(ptr %p, i32 42)
  ret void
}
; CHECK-LABEL: define void @thunk_caller(ptr %p)
; CHECK: call void (ptr, ...) @ext_method(ptr nonnull %this_adj.i, i32 42)

define signext i16 @test_callee_2(...) {
  %res = musttail call signext i16 (...) @vararg_fn(...) #0
  ret i16 %res
}

define void @test_caller_2(ptr %p, ptr %q, i16 %r) {
  call signext i16 (...) @test_callee_2(ptr %p, ptr byval(i8) %q, i16 signext %r)
  ret void
}
; CHECK-LABEL: define void @test_caller_2
; CHECK: call signext i16 (...) @vararg_fn(ptr %p, ptr byval(i8) %q, i16 signext %r) [[FN_ATTRS:#[0-9]+]]

define void @test_callee_3(ptr %p, ...) {
  call signext i16 (...) @vararg_fn()
  ret void
}

define void @test_caller_3(ptr %p, ptr %q) {
  call void (ptr, ...) @test_callee_3(ptr nonnull %p, ptr %q)
  ret void
}
; CHECK-LABEL: define void @test_caller_3
; CHECK: call signext i16 (...) @vararg_fn()

define void @test_preserve_cc(ptr %p, ...) {
  musttail call "cc 9" void (ptr, ...) @vararg_fn_cc9(ptr %p, ...)
  ret void
}

define void @test_caller_preserve_cc(ptr %p, ptr %q) {
  call void (ptr, ...) @test_preserve_cc(ptr %p, ptr %q)
  ret void
}
; CHECK-LABEL: define void @test_caller_preserve_cc
; CHECK: call "cc 9" void (ptr, ...) @vararg_fn_cc9(ptr %p, ptr %q)

define internal i32 @varg_accessed(...) {
entry:
  %vargs = alloca ptr, align 8
  call void @llvm.va_start(ptr %vargs)
  %va1 = va_arg ptr %vargs, i32
  call void @llvm.va_end(ptr %vargs)
  ret i32 %va1
}

define internal i32 @varg_accessed_alwaysinline(...) alwaysinline {
entry:
  %vargs = alloca ptr, align 8
  call void @llvm.va_start(ptr %vargs)
  %va1 = va_arg ptr %vargs, i32
  call void @llvm.va_end(ptr %vargs)
  ret i32 %va1
}

define i32 @call_vargs() {
  %res1 = call i32 (...) @varg_accessed(i32 10)
  %res2 = call i32 (...) @varg_accessed_alwaysinline(i32 15)
  %res = add i32 %res1, %res2
  ret i32 %res
}
; CHECK-LABEL: @call_vargs
; CHECK: %res1 = call i32 (...) @varg_accessed(i32 10)
; CHECK-NEXT: %res2 = call i32 (...) @varg_accessed_alwaysinline(i32 15)

define void @caller_with_vastart(ptr noalias nocapture readnone %args, ...) {
entry:
  %ap = alloca ptr, align 4
  %ap2 = alloca ptr, align 4
  call void @llvm.va_start(ptr nonnull %ap)
  call fastcc void @callee_with_vaend(ptr nonnull %ap)
  call void @llvm.va_start(ptr nonnull %ap)
  call fastcc void @callee_with_vaend_alwaysinline(ptr nonnull %ap)
  ret void
}

define internal fastcc void @callee_with_vaend_alwaysinline(ptr %a) alwaysinline {
entry:
  tail call void @llvm.va_end(ptr %a)
  ret void
}

define internal fastcc void @callee_with_vaend(ptr %a) {
entry:
  tail call void @llvm.va_end(ptr %a)
  ret void
}

; CHECK-LABEL: @caller_with_vastart
; CHECK-NOT: @callee_with_vaend
; CHECK-NOT: @callee_with_vaend_alwaysinline

declare void @llvm.va_start(ptr)
declare void @llvm.va_end(ptr)

; CHECK: attributes [[FN_ATTRS]] = { "foo"="bar" }
attributes #0 = { "foo"="bar" }
