; RUN: opt < %s -tsan -S -enable-new-pm=0 | FileCheck %s
; RUN: opt < %s -passes='function(tsan),module(tsan-module)' -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

define i32 @read_4_bytes(i32* %a) sanitize_thread {
entry:
  %tmp1 = load i32, i32* %a, align 4
  ret i32 %tmp1
}

; CHECK: @llvm.global_ctors = {{.*}}@tsan.module_ctor

; CHECK: define i32 @read_4_bytes(i32* %a)
; CHECK:        call void @__tsan_func_entry(i8* %0)
; CHECK-NEXT:   %1 = bitcast i32* %a to i8*
; CHECK-NEXT:   call void @__tsan_read4(i8* %1)
; CHECK-NEXT:   %tmp1 = load i32, i32* %a, align 4
; CHECK-NEXT:   call void @__tsan_func_exit()
; CHECK: ret i32


declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1)
declare void @llvm.memmove.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1)
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1)


; Check that tsan converts mem intrinsics back to function calls.

define void @MemCpyTest(i8* nocapture %x, i8* nocapture %y) sanitize_thread {
entry:
    tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %x, i8* align 4 %y, i64 16, i1 false)
    ret void
; CHECK: define void @MemCpyTest
; CHECK: call i8* @memcpy
; CHECK: ret void
}

define void @MemMoveTest(i8* nocapture %x, i8* nocapture %y) sanitize_thread {
entry:
    tail call void @llvm.memmove.p0i8.p0i8.i64(i8* align 4 %x, i8* align 4 %y, i64 16, i1 false)
    ret void
; CHECK: define void @MemMoveTest
; CHECK: call i8* @memmove
; CHECK: ret void
}

define void @MemSetTest(i8* nocapture %x) sanitize_thread {
entry:
    tail call void @llvm.memset.p0i8.i64(i8* align 4 %x, i8 77, i64 16, i1 false)
    ret void
; CHECK: define void @MemSetTest
; CHECK: call i8* @memset
; CHECK: ret void
}

; CHECK-LABEL: @SwiftError
; CHECK-NOT: __tsan_read
; CHECK-NOT: __tsan_write
; CHECK: ret
define void @SwiftError(i8** swifterror) sanitize_thread {
  %swifterror_ptr_value = load i8*, i8** %0
  store i8* null, i8** %0
  %swifterror_addr = alloca swifterror i8*
  %swifterror_ptr_value_2 = load i8*, i8** %swifterror_addr
  store i8* null, i8** %swifterror_addr
  ret void
}

; CHECK-LABEL: @SwiftErrorCall
; CHECK-NOT: __tsan_read
; CHECK-NOT: __tsan_write
; CHECK: ret
define void @SwiftErrorCall(i8** swifterror) sanitize_thread {
  %swifterror_addr = alloca swifterror i8*
  store i8* null, i8** %0
  call void @SwiftError(i8** %0)
  ret void
}

; CHECK-LABEL: @NakedTest(i32* %a)
; CHECK-NEXT:   call void @foo()
; CHECK-NEXT:   %tmp1 = load i32, i32* %a, align 4
; CHECK-NEXT:   ret i32 %tmp1
define i32 @NakedTest(i32* %a) naked sanitize_thread {
  call void @foo()
  %tmp1 = load i32, i32* %a, align 4
  ret i32 %tmp1
}

declare void @foo() nounwind

; CHECK: define internal void @tsan.module_ctor()
; CHECK: call void @__tsan_init()
