; RUN: opt < %s -inline -S | FileCheck %s
; RUN: opt < %s -passes='cgscc(inline,function(instcombine))' -S | FileCheck %s

declare void @ext_method(i8*, i32)
declare signext i16 @vararg_fn(...) #0
declare "cc 9" void @vararg_fn_cc9(i8* %p, ...)

define linkonce_odr void @thunk(i8* %this, ...) {
  %this_adj = getelementptr i8, i8* %this, i32 4
  musttail call void (i8*, ...) bitcast (void (i8*, i32)* @ext_method to void (i8*, ...)*)(i8* nonnull %this_adj, ...)
  ret void
}

define void @thunk_caller(i8* %p) {
  call void (i8*, ...) @thunk(i8* %p, i32 42)
  ret void
}
; CHECK-LABEL: define void @thunk_caller(i8* %p)
; CHECK: call void (i8*, ...) bitcast (void (i8*, i32)* @ext_method to void (i8*, ...)*)(i8* nonnull %this_adj.i, i32 42)

define signext i16 @test_callee_2(...) {
  %res = musttail call signext i16 (...) @vararg_fn(...) #0
  ret i16 %res
}

define void @test_caller_2(i8* %p, i8* %q, i16 %r) {
  call signext i16 (...) @test_callee_2(i8* %p, i8* byval %q, i16 signext %r)
  ret void
}
; CHECK-LABEL: define void @test_caller_2
; CHECK: call signext i16 (...) @vararg_fn(i8* %p, i8* byval %q, i16 signext %r) [[FN_ATTRS:#[0-9]+]]

define void @test_callee_3(i8* %p, ...) {
  call signext i16 (...) @vararg_fn()
  ret void
}

define void @test_caller_3(i8* %p, i8* %q) {
  call void (i8*, ...) @test_callee_3(i8* nonnull %p, i8* %q)
  ret void
}
; CHECK-LABEL: define void @test_caller_3
; CHECK: call signext i16 (...) @vararg_fn()

define void @test_preserve_cc(i8* %p, ...) {
  musttail call "cc 9" void (i8*, ...) @vararg_fn_cc9(i8* %p, ...)
  ret void
}

define void @test_caller_preserve_cc(i8* %p, i8* %q) {
  call void (i8*, ...) @test_preserve_cc(i8* %p, i8* %q)
  ret void
}
; CHECK-LABEL: define void @test_caller_preserve_cc
; CHECK: call "cc 9" void (i8*, ...) @vararg_fn_cc9(i8* %p, i8* %q)

define internal i32 @varg_accessed(...) {
entry:
  %vargs = alloca i8*, align 8
  %vargs.ptr = bitcast i8** %vargs to i8*
  call void @llvm.va_start(i8* %vargs.ptr)
  %va1 = va_arg i8** %vargs, i32
  call void @llvm.va_end(i8* %vargs.ptr)
  ret i32 %va1
}

define i32 @call_vargs() {
  %res = call i32 (...) @varg_accessed(i32 10)
  ret i32 %res
}
; CHECK-LABEL: @call_vargs
; CHECK: %res = call i32 (...) @varg_accessed(i32 10)

declare void @llvm.va_start(i8*)
declare void @llvm.va_end(i8*)

; CHECK: attributes [[FN_ATTRS]] = { "foo"="bar" }
attributes #0 = { "foo"="bar" }
