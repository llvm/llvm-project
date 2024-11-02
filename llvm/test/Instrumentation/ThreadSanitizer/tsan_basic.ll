; RUN: opt < %s -passes='function(tsan),module(tsan-module)' -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

define i32 @read_4_bytes(ptr %a) sanitize_thread {
entry:
  %tmp1 = load i32, ptr %a, align 4
  ret i32 %tmp1
}

; CHECK: @llvm.used = appending global [1 x ptr] [ptr @tsan.module_ctor]
; CHECK: @llvm.global_ctors = {{.*}}@tsan.module_ctor

; CHECK: define i32 @read_4_bytes(ptr %a)
; CHECK:        call void @__tsan_func_entry(ptr %0)
; CHECK-NEXT:   call void @__tsan_read4(ptr %a)
; CHECK-NEXT:   %tmp1 = load i32, ptr %a, align 4
; CHECK-NEXT:   call void @__tsan_func_exit()
; CHECK: ret i32


declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1)
declare void @llvm.memcpy.inline.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1)
declare void @llvm.memmove.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1)
declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1)
declare void @llvm.memset.inline.p0.i64(ptr nocapture, i8, i64, i1)


; Check that tsan converts mem intrinsics back to function calls.

define void @MemCpyTest(ptr nocapture %x, ptr nocapture %y) sanitize_thread {
entry:
    tail call void @llvm.memcpy.p0.p0.i64(ptr align 4 %x, ptr align 4 %y, i64 16, i1 false)
    ret void
; CHECK: define void @MemCpyTest
; CHECK: call ptr @__tsan_memcpy
; CHECK: ret void
}

define void @MemCpyInlineTest(ptr nocapture %x, ptr nocapture %y) sanitize_thread {
entry:
    tail call void @llvm.memcpy.inline.p0.p0.i64(ptr align 4 %x, ptr align 4 %y, i64 16, i1 false)
    ret void
; CHECK: define void @MemCpyInlineTest
; CHECK: call ptr @__tsan_memcpy
; CHECK: ret void
}

define void @MemMoveTest(ptr nocapture %x, ptr nocapture %y) sanitize_thread {
entry:
    tail call void @llvm.memmove.p0.p0.i64(ptr align 4 %x, ptr align 4 %y, i64 16, i1 false)
    ret void
; CHECK: define void @MemMoveTest
; CHECK: call ptr @__tsan_memmove
; CHECK: ret void
}

define void @MemSetTest(ptr nocapture %x) sanitize_thread {
entry:
    tail call void @llvm.memset.p0.i64(ptr align 4 %x, i8 77, i64 16, i1 false)
    ret void
; CHECK: define void @MemSetTest
; CHECK: call ptr @__tsan_memset
; CHECK: ret void
}

define void @MemSetInlineTest(ptr nocapture %x) sanitize_thread {
entry:
    tail call void @llvm.memset.inline.p0.i64(ptr align 4 %x, i8 77, i64 16, i1 false)
    ret void
; CHECK: define void @MemSetInlineTest
; CHECK: call ptr @__tsan_memset
; CHECK: ret void
}

; CHECK-LABEL: @SwiftError
; CHECK-NOT: __tsan_read
; CHECK-NOT: __tsan_write
; CHECK: ret
define void @SwiftError(ptr swifterror) sanitize_thread {
  %swifterror_ptr_value = load ptr, ptr %0
  store ptr null, ptr %0
  %swifterror_addr = alloca swifterror ptr
  %swifterror_ptr_value_2 = load ptr, ptr %swifterror_addr
  store ptr null, ptr %swifterror_addr
  ret void
}

; CHECK-LABEL: @SwiftErrorCall
; CHECK-NOT: __tsan_read
; CHECK-NOT: __tsan_write
; CHECK: ret
define void @SwiftErrorCall(ptr swifterror) sanitize_thread {
  %swifterror_addr = alloca swifterror ptr
  store ptr null, ptr %0
  call void @SwiftError(ptr %0)
  ret void
}

; CHECK-LABEL: @NakedTest(ptr %a)
; CHECK-NEXT:   call void @foo()
; CHECK-NEXT:   %tmp1 = load i32, ptr %a, align 4
; CHECK-NEXT:   ret i32 %tmp1
define i32 @NakedTest(ptr %a) naked sanitize_thread {
  call void @foo()
  %tmp1 = load i32, ptr %a, align 4
  ret i32 %tmp1
}

declare void @foo() nounwind

; CHECK: define internal void @tsan.module_ctor()
; CHECK: call void @__tsan_init()
