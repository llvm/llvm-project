; Argument tracing: one __sanitizer_cov_trace_args call per IR argument, in the
; entry block, decided by the IR alone. arg_idx is the position of the argument
; in the IR parameter list; a struct-return pointer and a Swift error slot are
; not observed but keep their positions.
;
; A value that fits in a register is reported as the value; an argument the ABI
; says points at a copy of the parameter in memory is reported as that object,
; with a table of the elements of its IR type, none wider than 8 bytes. A
; pointer the IR says is dereferenceable is reported with a raw snapshot of the
; bytes it points to, as many as the attribute says and
; -sanitizer-coverage-trace-pointee-bytes allows.
;
; An argument the function does not use is passed whatever its callers pass -
; poison, after dead argument elimination - so an object is not read through
; it; the functions below use theirs, and @unused checks the rule.
;
; opt runs the verifier, so a passing run also proves the emitted IR is
; well-formed.
;
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-trace-args -S | FileCheck %s
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-trace-args \
; RUN:   -sanitizer-coverage-trace-pointee-bytes=0 -S | FileCheck %s --check-prefix=NOPTEE

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @use(...)

%struct.T = type { i32, i8, i64 }

; One table per IR type, shared by every object of that type, and one for an
; object that is not a struct.
; CHECK: @__sancov_offsets_ = private unnamed_addr constant [6 x i64] [i64 0, i64 4, i64 4, i64 1, i64 8, i64 8]
; CHECK: @__sancov_offsets_.1 = private unnamed_addr constant [2 x i64] [i64 0, i64 8]
; A pointee snapshot is a byte array: 16, 64 and 12 bytes, in 8-byte pieces.
; CHECK: @__sancov_offsets_.2 = private unnamed_addr constant [4 x i64] [i64 0, i64 8, i64 8, i64 8]
; CHECK: @__sancov_offsets_.3 = private unnamed_addr constant [16 x i64] [i64 0, i64 8, i64 8, i64 8, i64 16, i64 8, i64 24, i64 8, i64 32, i64 8, i64 40, i64 8, i64 48, i64 8, i64 56, i64 8]
; CHECK: @__sancov_offsets_.4 = private unnamed_addr constant [4 x i64] [i64 0, i64 8, i64 8, i64 4]
; An opaque [32 x i8] object, in 8-byte pieces; a struct by the elements of its
; nested structs, and an element wider than 8 bytes in pieces.
; CHECK: @__sancov_offsets_.5 = private unnamed_addr constant [8 x i64] [i64 0, i64 8, i64 8, i64 8, i64 16, i64 8, i64 24, i64 8]
; CHECK: @__sancov_offsets_.6 = private unnamed_addr constant [10 x i64] [i64 0, i64 4, i64 8, i64 1, i64 16, i64 8, i64 24, i64 8, i64 32, i64 4]
; CHECK: @__sancov_offsets_.7 = private unnamed_addr constant [6 x i64] [i64 0, i64 8, i64 8, i64 8, i64 16, i64 1]
; CHECK-NOT: @__sancov_offsets_

; Scalars and pointers are reported as their value, zero-extended to 64 bits.
define void @scalars(i1 noundef %b, i8 noundef %c, i32 noundef %i, i64 noundef %l, ptr noundef %p) {
entry:
  call void (...) @use(i1 %b, i8 %c, i32 %i, i64 %l, ptr %p)
  ret void
}
; CHECK-LABEL: define void @scalars(
; CHECK: %[[B:[0-9]+]] = zext i1 %b to i64
; CHECK: call void @__sanitizer_cov_trace_args(i32 0, i32 1, i64 %[[B]], ptr null, i32 0)
; CHECK: %[[C:[0-9]+]] = zext i8 %c to i64
; CHECK: call void @__sanitizer_cov_trace_args(i32 1, i32 1, i64 %[[C]], ptr null, i32 0)
; CHECK: %[[I:[0-9]+]] = zext i32 %i to i64
; CHECK: call void @__sanitizer_cov_trace_args(i32 2, i32 4, i64 %[[I]], ptr null, i32 0)
; CHECK: call void @__sanitizer_cov_trace_args(i32 3, i32 8, i64 %l, ptr null, i32 0)
; CHECK: %[[P:[0-9]+]] = ptrtoint ptr %p to i64
; CHECK: call void @__sanitizer_cov_trace_args(i32 4, i32 8, i64 %[[P]], ptr null, i32 0)

; A floating-point value or a vector is reported as its bit pattern, a value
; wider than 64 bits by its low 8 bytes, a pointer with the width of its address
; space, and a value with no fixed-size bit pattern with size 0.
define void @wide(float noundef %f, double noundef %d, i128 noundef %w, ptr addrspace(270) noundef %p32, <4 x i32> noundef %v, <2 x float> noundef %v2, <2 x ptr> noundef %vp, <vscale x 2 x i64> noundef %sv) {
entry:
  call void (...) @use(float %f, double %d, i128 %w, ptr addrspace(270) %p32, <4 x i32> %v, <2 x float> %v2, <2 x ptr> %vp, <vscale x 2 x i64> %sv)
  ret void
}
; CHECK-LABEL: define void @wide(
; CHECK: %[[F:[0-9]+]] = bitcast float %f to i32
; CHECK: %[[FZ:[0-9]+]] = zext i32 %[[F]] to i64
; CHECK: call void @__sanitizer_cov_trace_args(i32 0, i32 4, i64 %[[FZ]], ptr null, i32 0)
; CHECK: %[[D:[0-9]+]] = bitcast double %d to i64
; CHECK: call void @__sanitizer_cov_trace_args(i32 1, i32 8, i64 %[[D]], ptr null, i32 0)
; CHECK: %[[W:[0-9]+]] = trunc i128 %w to i64
; CHECK: call void @__sanitizer_cov_trace_args(i32 2, i32 8, i64 %[[W]], ptr null, i32 0)
; CHECK: call void @__sanitizer_cov_trace_args(i32 3, i32 4,
; CHECK: %[[V:[0-9]+]] = bitcast <4 x i32> %v to i128
; CHECK: %[[VT:[0-9]+]] = trunc i128 %[[V]] to i64
; CHECK: call void @__sanitizer_cov_trace_args(i32 4, i32 8, i64 %[[VT]], ptr null, i32 0)
; CHECK: %[[V2:[0-9]+]] = bitcast <2 x float> %v2 to i64
; CHECK: call void @__sanitizer_cov_trace_args(i32 5, i32 8, i64 %[[V2]], ptr null, i32 0)
; CHECK: %[[VP:[0-9]+]] = ptrtoint <2 x ptr> %vp to <2 x i64>
; CHECK: %[[VPB:[0-9]+]] = bitcast <2 x i64> %[[VP]] to i128
; CHECK: %[[VPT:[0-9]+]] = trunc i128 %[[VPB]] to i64
; CHECK: call void @__sanitizer_cov_trace_args(i32 6, i32 8, i64 %[[VPT]], ptr null, i32 0)
; CHECK: call void @__sanitizer_cov_trace_args(i32 7, i32 0, i64 0, ptr null, i32 0)

; A first-class aggregate lives in registers and has no address: each element
; is reported on its own, under the same arg_idx.
define void @aggregate({ i32, i64 } noundef %s, [2 x i8] noundef %a) {
entry:
  call void (...) @use({ i32, i64 } %s, [2 x i8] %a)
  ret void
}
; CHECK-LABEL: define void @aggregate(
; CHECK: call void @__sanitizer_cov_trace_args(i32 0, i32 4,
; CHECK: call void @__sanitizer_cov_trace_args(i32 0, i32 8,
; CHECK: call void @__sanitizer_cov_trace_args(i32 1, i32 1,
; CHECK: call void @__sanitizer_cov_trace_args(i32 1, i32 1,

; byval, inalloca, preallocated and byref arguments are the address of a copy
; of the parameter: they are reported as that object, described by the IR type.
define void @objects(ptr byval(%struct.T) %a, ptr preallocated(i64) %b, ptr byref(%struct.T) %c) {
entry:
  call void (...) @use(ptr %a, ptr %b, ptr %c)
  ret void
}
; CHECK-LABEL: define void @objects(
; CHECK: %[[A:[0-9]+]] = ptrtoint ptr %a to i64
; CHECK: call void @__sanitizer_cov_trace_args(i32 0, i32 16, i64 %[[A]], ptr @__sancov_offsets_, i32 3)
; CHECK: call void @__sanitizer_cov_trace_args(i32 1, i32 8, i64 %{{[0-9]+}}, ptr @__sancov_offsets_.1, i32 1)
; CHECK: call void @__sanitizer_cov_trace_args(i32 2, i32 16, i64 %{{[0-9]+}}, ptr @__sancov_offsets_, i32 3)

; inalloca has to be the last parameter.
define void @inalloca(i32 noundef %x, ptr inalloca(%struct.T) %a) {
entry:
  call void (...) @use(i32 %x, ptr %a)
  ret void
}
; CHECK-LABEL: define void @inalloca(
; CHECK: call void @__sanitizer_cov_trace_args(i32 0, i32 4,
; CHECK: call void @__sanitizer_cov_trace_args(i32 1, i32 16, i64 %{{[0-9]+}}, ptr @__sancov_offsets_, i32 3)

; A struct-return pointer is where the return value goes, which trace-ret
; reports. It is not observed, but %x keeps its IR position, 1.
define void @sret(ptr sret(%struct.T) %r, i32 noundef %x) {
entry:
  call void (...) @use(ptr %r, i32 %x)
  ret void
}
; CHECK-LABEL: define void @sret(
; CHECK-NOT: ptrtoint ptr %r
; CHECK: call void @__sanitizer_cov_trace_args(i32 1, i32 4,
; CHECK-NOT: call void @__sanitizer_cov_trace_args(

; The error slot of a Swift throwing function may only be loaded, stored, or
; passed on as a swifterror argument: it is skipped, and %x keeps index 2.
define swiftcc void @swifterror(ptr noundef swiftself %self, ptr swifterror %err, i64 noundef %x) {
entry:
  call void (...) @use(ptr %self, i64 %x)
  ret void
}
; CHECK-LABEL: define swiftcc void @swifterror(
; CHECK-NOT: ptrtoint ptr %err
; CHECK: call void @__sanitizer_cov_trace_args(i32 0, i32 8,
; CHECK: call void @__sanitizer_cov_trace_args(i32 2, i32 8, i64 %x, ptr null, i32 0)
; CHECK-NOT: call void @__sanitizer_cov_trace_args(

; The named arguments of a variadic function are reported; the variadic ones
; have no IR argument.
define void @varargs(i32 noundef %n, ...) {
entry:
  call void (...) @use(i32 %n)
  ret void
}
; CHECK-LABEL: define void @varargs(
; CHECK: call void @__sanitizer_cov_trace_args(i32 0, i32 4,
; CHECK-NOT: call void @__sanitizer_cov_trace_args(

; The calls go at the first insertion point of the entry block, after its PHI
; nodes and EH pad, where every argument is available. They are never tail
; calls: the runtime identifies an observation by its call's return address.
declare i32 @__gxx_personality_v0(...)
define i32 @landing(i32 noundef %x) personality ptr @__gxx_personality_v0 {
entry:
  ret i32 %x
}
; CHECK-LABEL: define i32 @landing(
; CHECK-NEXT: entry:
; CHECK-NEXT: %[[X:[0-9]+]] = zext i32 %x to i64
; CHECK-NEXT: notail call void @__sanitizer_cov_trace_args(i32 0, i32 4, i64 %[[X]], ptr null, i32 0)
; CHECK-NEXT: ret i32 %x

; An unused argument may be passed anything: an unused value is still
; reported and frozen, but an unused object is not read, as its pointer may
; be poison. A value that may be poison - here
; because none of the arguments is noundef - is frozen, so that the callback
; receives bits.
define void @unused(i32 %x, ptr byval(%struct.T) %o, { i32, i64 } %s) {
entry:
  ret void
}
; CHECK-LABEL: define void @unused(
; CHECK-NEXT: entry:
; CHECK-NEXT: %[[XF:[0-9]+]] = freeze i32 %x
; CHECK-NEXT: %[[X:[0-9]+]] = zext i32 %[[XF]] to i64
; CHECK-NEXT: notail call void @__sanitizer_cov_trace_args(i32 0, i32 4, i64 %[[X]], ptr null, i32 0)
; CHECK-NEXT: notail call void @__sanitizer_cov_trace_args(i32 1, i32 0, i64 0, ptr null, i32 0)
; CHECK:      notail call void @__sanitizer_cov_trace_args(i32 2, i32 4,
; CHECK:      notail call void @__sanitizer_cov_trace_args(i32 2, i32 8,
; CHECK-NEXT: ret void

; A pointer the IR says is dereferenceable(N) is reported with the first N
; bytes it points to, at most -sanitizer-coverage-trace-pointee-bytes (64 by
; default): `val` stays the pointer, `size` is the number of bytes and the
; table covers them. dereferenceable_or_null may be null, which the runtime
; checks. A pointer with no such attribute, one in another address space, and
; one the function does not use are reported by value only.
define void @pointee(ptr noundef nonnull align 8 dereferenceable(16) %p, ptr dereferenceable_or_null(100) %q, ptr noundef %raw, ptr addrspace(1) dereferenceable(16) %as1, ptr dereferenceable(12) %r, ptr dereferenceable(8) %dead) {
entry:
  call void (...) @use(ptr %p, ptr %q, ptr %raw, ptr addrspace(1) %as1, ptr %r)
  ret void
}
; CHECK-LABEL: define void @pointee(
; CHECK: %[[P:[0-9]+]] = ptrtoint ptr %p to i64
; CHECK: call void @__sanitizer_cov_trace_args(i32 0, i32 16, i64 %[[P]], ptr @__sancov_offsets_.2, i32 2)
; CHECK: %[[Q:[0-9]+]] = ptrtoint ptr %{{.*}} to i64
; CHECK: call void @__sanitizer_cov_trace_args(i32 1, i32 64, i64 %[[Q]], ptr @__sancov_offsets_.3, i32 8)
; CHECK: call void @__sanitizer_cov_trace_args(i32 2, i32 8, i64 %{{[0-9]+}}, ptr null, i32 0)
; CHECK: call void @__sanitizer_cov_trace_args(i32 3, i32 8, i64 %{{[0-9]+}}, ptr null, i32 0)
; CHECK: call void @__sanitizer_cov_trace_args(i32 4, i32 12, i64 %{{[0-9]+}}, ptr @__sancov_offsets_.4, i32 2)
; CHECK: call void @__sanitizer_cov_trace_args(i32 5, i32 8, i64 %{{[0-9]+}}, ptr null, i32 0)
; NOPTEE-LABEL: define void @pointee(
; NOPTEE-COUNT-6: call void @__sanitizer_cov_trace_args(i32 {{[0-5]}}, i32 8, i64 %{{[0-9]+}}, ptr null, i32 0)

; An object whose IR type is an opaque byte array, as rustc passes and returns
; one, is read in 8-byte pieces rather than by its first 8 bytes only; so is an
; element wider than 8 bytes, and a nested struct is described by its elements.
define void @wide_objects(ptr byval([32 x i8]) %o, ptr byval({ i32, { i8, i64 }, [3 x i32] }) %n, ptr byval({ i128, i8 }) %w) {
entry:
  call void (...) @use(ptr %o, ptr %n, ptr %w)
  ret void
}
; CHECK-LABEL: define void @wide_objects(
; CHECK: call void @__sanitizer_cov_trace_args(i32 0, i32 32, i64 %{{[0-9]+}}, ptr @__sancov_offsets_.5, i32 4)
; CHECK: call void @__sanitizer_cov_trace_args(i32 1, i32 40, i64 %{{[0-9]+}}, ptr @__sancov_offsets_.6, i32 5)
; CHECK: call void @__sanitizer_cov_trace_args(i32 2, i32 32, i64 %{{[0-9]+}}, ptr @__sancov_offsets_.7, i32 3)
