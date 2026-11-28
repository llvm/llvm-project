; Return value tracing: one __sanitizer_cov_trace_ret call before every return,
; decided by the IR alone. A value that fits in a register is reported as the
; value, a first-class aggregate as its elements, and an indirect (sret) return
; as the buffer the returned object was written to.
;
; opt runs the verifier, so a passing run also proves the emitted IR is
; well-formed.
;
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=1 -sanitizer-coverage-trace-ret -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S = type { i64, i64, i64 }

; CHECK: @__sancov_offsets_ = private unnamed_addr constant [5 x i64] [i64 2, i64 0, i64 8, i64 8, i64 4]
; CHECK: @__sancov_offsets_.1 = private unnamed_addr constant [7 x i64] [i64 3, i64 0, i64 8, i64 8, i64 8, i64 16, i64 8]
; CHECK: @__sancov_offsets_.2 = private unnamed_addr constant [9 x i64] [i64 4, i64 0, i64 8, i64 8, i64 8, i64 16, i64 8, i64 24, i64 8]
; CHECK-NOT: @__sancov_offsets_

define i32 @scalar(i32 noundef %x) {
entry:
  ret i32 %x
}
; CHECK-LABEL: define i32 @scalar(
; CHECK: %[[X:[0-9]+]] = zext i32 %x to i64
; CHECK-NEXT: call void @__sanitizer_cov_trace_ret(i64 4, i64 %[[X]], ptr null)
; CHECK-NEXT: ret i32 %x

; A pointer is reported as the address it holds.
define ptr @pointer(ptr noundef %p) {
entry:
  ret ptr %p
}
; CHECK-LABEL: define ptr @pointer(
; CHECK: %[[P:[0-9]+]] = ptrtoint ptr %p to i64
; CHECK: call void @__sanitizer_cov_trace_ret(i64 8, i64 %[[P]], ptr null)

; A pointer the return attributes say is dereferenceable at the return is
; reported with the bytes it points to, as rustc returns a reference: `val` is
; the pointer, `size` the 12 bytes read.
define noundef nonnull align 4 dereferenceable(12) ptr @reference(ptr noundef %p) {
entry:
  ret ptr %p
}
; CHECK-LABEL: define noundef nonnull align 4 dereferenceable(12) ptr @reference(
; CHECK: %[[P:[0-9]+]] = ptrtoint ptr %p to i64
; CHECK-NEXT: call void @__sanitizer_cov_trace_ret(i64 12, i64 %[[P]], ptr @__sancov_offsets_)
; CHECK-NEXT: ret ptr %p

; Every return is reported, each with its own value.
define i64 @multi(i1 noundef %c, i64 noundef %x) {
entry:
  br i1 %c, label %a, label %b
a:
  ret i64 %x
b:
  ret i64 7
}
; CHECK-LABEL: define i64 @multi(
; CHECK: call void @__sanitizer_cov_trace_ret(i64 8, i64 %x, ptr null)
; CHECK-NEXT: ret i64 %x
; CHECK: call void @__sanitizer_cov_trace_ret(i64 8, i64 7, ptr null)
; CHECK-NEXT: ret i64 7

; A struct returned in registers is reported as its elements.
define { i64, double } @aggregate(i64 noundef %a, double noundef %b) {
entry:
  %0 = insertvalue { i64, double } poison, i64 %a, 0
  %1 = insertvalue { i64, double } %0, double %b, 1
  ret { i64, double } %1
}
; CHECK-LABEL: define { i64, double } @aggregate(
; CHECK: call void @__sanitizer_cov_trace_ret(i64 8,
; CHECK: call void @__sanitizer_cov_trace_ret(i64 8,
; CHECK-NEXT: ret { i64, double }

; An indirect return is reported as the buffer the returned object was written
; to, described by the elements of its IR type.
define void @indirect(ptr sret(%struct.S) %r, i64 noundef %x) {
entry:
  store i64 %x, ptr %r
  ret void
}
; CHECK-LABEL: define void @indirect(
; CHECK: %[[R:[0-9]+]] = ptrtoint ptr %r to i64
; CHECK-NEXT: call void @__sanitizer_cov_trace_ret(i64 24, i64 %[[R]], ptr @__sancov_offsets_.1)
; CHECK-NEXT: ret void

; rustc types the sret buffer of a Rust-ABI function as an opaque byte array:
; it is read in 8-byte pieces, all 32 bytes of it.
define void @opaque_indirect(ptr sret([32 x i8]) %r, i64 noundef %x) {
entry:
  store i64 %x, ptr %r
  ret void
}
; CHECK-LABEL: define void @opaque_indirect(
; CHECK: call void @__sanitizer_cov_trace_ret(i64 32, i64 %{{[0-9]+}}, ptr @__sancov_offsets_.2)

; A void return has nothing to report, and says so with size 0. The call is in
; tail position, but must not become a tail call: the runtime identifies an
; observation by its call's return address.
define void @void() {
entry:
  ret void
}
; CHECK-LABEL: define void @void(
; CHECK: notail call void @__sanitizer_cov_trace_ret(i64 0, i64 0, ptr null)
; CHECK-NEXT: ret void

; A vector is reported as its bit pattern.
define <2 x i32> @vector(<2 x i32> noundef %v) {
entry:
  ret <2 x i32> %v
}
; CHECK-LABEL: define <2 x i32> @vector(
; CHECK: %[[V:[0-9]+]] = bitcast <2 x i32> %v to i64
; CHECK-NEXT: notail call void @__sanitizer_cov_trace_ret(i64 8, i64 %[[V]], ptr null)

; A musttail call has to stay adjacent to the return that forwards it, so that
; return is left alone.
define i64 @forward(i64 noundef %x) {
entry:
  %r = musttail call i64 @forward(i64 %x)
  ret i64 %r
}
; CHECK-LABEL: define i64 @forward(
; CHECK-NOT: call void @__sanitizer_cov_trace_ret
; CHECK: musttail call i64 @forward
; CHECK-NEXT: ret i64 %r

; Unwinding leaves the function without a value.
declare void @may_throw()
declare i32 @__gxx_personality_v0(...)
define i32 @unwinds(i32 noundef %x) personality ptr @__gxx_personality_v0 {
entry:
  invoke void @may_throw() to label %ok unwind label %lp
ok:
  ret i32 %x
lp:
  %l = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %l
}
; CHECK-LABEL: define i32 @unwinds(
; CHECK: call void @__sanitizer_cov_trace_ret(i64 4,
; CHECK-NEXT: ret i32 %x
; CHECK-NOT: call void @__sanitizer_cov_trace_ret
; CHECK: resume
