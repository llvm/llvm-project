; RUN: opt -passes=tsan -tsan-use-escape-analysis < %s -S | FileCheck %s

; This file contains tests for TSan with escape analysis enabled.
; Goal: skip instrumentation for accesses to provably non-escaping locals and
; instrument when the object may escape (return, store to global, via loaded global
; pointers, double-pointer cycles, atomic/volatile, complex/bail-out paths).

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

@g = global i32 0
@GPtr = global ptr null

declare void @external(ptr)
declare noalias ptr @malloc(i64)
declare void @opaque_call()
declare void @llvm.donothing() readnone

; =============================================================================
; LOCAL ALLOCAS THAT DO NOT ESCAPE
; =============================================================================

; Local alloca: not returned, not stored outside -> no __tsan_* calls
define void @local_no_escape() nounwind uwtable sanitize_thread {
  %a = alloca i32, align 4
  store i32 1, ptr %a, align 4
  %v = load i32, ptr %a, align 4
  ret void
}
; CHECK-LABEL: define void @local_no_escape
; CHECK-NOT: call void @__tsan_write4
; CHECK-NOT: call void @__tsan_read4

; Chain through local pointer (double indirection) remains local
define void @double_ptr_local_ok() nounwind uwtable sanitize_thread {
  %x  = alloca i32, align 4
  %p  = alloca ptr, align 8
  %pp = alloca ptr, align 8
  store ptr %x, ptr %p
  store ptr %p, ptr %pp
  store i32 1, ptr %x
  %lv = load i32, ptr %x
  ret void
}
; CHECK-LABEL: define void @double_ptr_local_ok
; CHECK-NOT: call void @__tsan_write4(ptr %x)
; CHECK-NOT: call void @__tsan_read4(ptr %x)

; Loaded destination remains local (two-store memphi pattern)
define void @loaded_dest_memphi_local_ok(i1 %c) nounwind uwtable sanitize_thread {
  %x  = alloca i32, align 4
  %p  = alloca ptr, align 8
  %s1 = alloca ptr, align 8
  %s2 = alloca ptr, align 8
  br i1 %c, label %T, label %F
T:
  store ptr %s1, ptr %p
  br label %M
F:
  store ptr %s2, ptr %p
  br label %M
M:
  %l = load ptr, ptr %p
  store ptr %x, ptr %l
  store i32 9, ptr %x
  %rv = load i32, ptr %x
  ret void
}
; CHECK-LABEL: define void @loaded_dest_memphi_local_ok
; CHECK-NOT: call void @__tsan_write4(ptr %x)
; CHECK-NOT: call void @__tsan_read4(ptr %x)

; Simple local store via intermediate pointer remains local
define void @store_via_local_ptr_ok() nounwind uwtable sanitize_thread {
  %x = alloca i32, align 4
  %slot = alloca ptr, align 8
  store ptr %x, ptr %slot
  %l = load ptr, ptr %slot
  store i32 100, ptr %l
  %r = load i32, ptr %x
  ret void
}
; CHECK-LABEL: define void @store_via_local_ptr_ok
; CHECK-NOT: call void @__tsan_write4(ptr %x)
; CHECK-NOT: call void @__tsan_read4(ptr %x)

; =============================================================================
; ESCAPES VIA GLOBALS, RETURNS, CALLS
; =============================================================================

; Address stored to global -> escape, accesses should be instrumented
define void @store_to_global_escape() nounwind uwtable sanitize_thread {
  %a = alloca i32, align 4
  store ptr %a, ptr @GPtr
  store i32 2, ptr %a, align 4
  %v = load i32, ptr %a, align 4
  ret void
}
; CHECK-LABEL: define void @store_to_global_escape
; CHECK: call void @__tsan_write4(ptr %a)
; CHECK: call void @__tsan_read4(ptr %a)

; Returning alloca pointer -> escape
define ptr @return_alloca_escape() nounwind uwtable sanitize_thread {
  %a = alloca i32, align 4
  store i32 5, ptr %a, align 4
  ret ptr %a
}
; CHECK-LABEL: define ptr @return_alloca_escape
; CHECK: call void @__tsan_write4(ptr %a)

; Chain becomes escaping: the final node is stored to global
define void @double_ptr_escape() nounwind uwtable sanitize_thread {
  %x  = alloca i32, align 4
  %p  = alloca ptr, align 8
  %pp = alloca ptr, align 8
  store ptr %x, ptr %p
  store ptr %p, ptr %pp
  store ptr %pp, ptr @GPtr
  store i32 7, ptr %x
  %r = load i32, ptr %x
  ret void
}
; CHECK-LABEL: define void @double_ptr_escape
; CHECK: call void @__tsan_write4(ptr %x)
; CHECK: call void @__tsan_read4(ptr %x)

; Loaded destination points to global -> escape
define void @loaded_dest_global_escape() nounwind uwtable sanitize_thread {
  %x = alloca i32, align 4
  %p = alloca ptr, align 8
  store ptr @GPtr, ptr %p
  %l = load ptr, ptr %p
  store ptr %x, ptr %l
  store i32 3, ptr %x
  %v = load i32, ptr %x
  ret void
}
; CHECK-LABEL: define void @loaded_dest_global_escape
; CHECK: call void @__tsan_write4(ptr %x)
; CHECK: call void @__tsan_read4(ptr %x)

; Passing address outside via call: argument escape
define void @call_external_escape() nounwind uwtable sanitize_thread {
  %x = alloca i32, align 4
  call void @external(ptr %x)
  store i32 300, ptr %x
  %r = load i32, ptr %x
  ret void
}
; CHECK-LABEL: define void @call_external_escape
; CHECK: call void @__tsan_write4(ptr %x)
; CHECK: call void @__tsan_read4(ptr %x)

; Store to global after intermediate slot -> escape
define void @store_via_local_then_global_escape() nounwind uwtable sanitize_thread {
  %x = alloca i32, align 4
  %slot = alloca ptr, align 8
  store ptr %x, ptr %slot
  %tmp = load ptr, ptr %slot
  store ptr %tmp, ptr @GPtr
  store i32 200, ptr %x
  %r = load i32, ptr %x
  ret void
}
; CHECK-LABEL: define void @store_via_local_then_global_escape
; CHECK: call void @__tsan_write4(ptr %x)
; CHECK: call void @__tsan_read4(ptr %x)

; =============================================================================
; ATOMICS, VOLATILE, AND BAIL-OUTS
; =============================================================================

; Atomic store of local address -> considered escape
define void @atomic_store_escape() nounwind uwtable sanitize_thread {
  %a = alloca i32, align 4
  %p = alloca ptr, align 8
  store atomic ptr %a, ptr %p seq_cst, align 8
  store i32 11, ptr %a
  %v = load i32, ptr %a
  ret void
}
; CHECK-LABEL: define void @atomic_store_escape
; CHECK: call void @__tsan_write4(ptr %a)
; CHECK: call void @__tsan_read4(ptr %a)

; Volatile store of local address -> escape
define void @volatile_store_escape() nounwind uwtable sanitize_thread {
  %a = alloca i32, align 4
  %p = alloca ptr, align 8
  store volatile ptr %a, ptr %p
  store i32 13, ptr %a
  %v = load i32, ptr %a
  ret void
}
; CHECK-LABEL: define void @volatile_store_escape
; CHECK: call void @__tsan_write4(ptr %a)
; CHECK: call void @__tsan_read4(ptr %a)

; Bail-out on complex path: ptrtoint -> conservatively escape
define void @ptrtoint_bailout_escape() nounwind uwtable sanitize_thread {
  %a = alloca i32, align 4
  %c = ptrtoint ptr %a to i64
  store i32 21, ptr %a
  %v = load i32, ptr %a
  ret void
}
; CHECK-LABEL: define void @ptrtoint_bailout_escape
; CHECK: call void @__tsan_write4(ptr %a)
; CHECK: call void @__tsan_read4(ptr %a)

; =============================================================================
; HEAP ALLOCATIONS
; =============================================================================

; Heap allocation is local (not stored, not returned) -> no instrumentation on the heap object
define void @malloc_not_escape1() nounwind uwtable sanitize_thread {
  %m = call noalias ptr @malloc(i64 16)
  store i32 1, ptr %m
  %v = load i32, ptr %m
  ret void
}
; CHECK-LABEL: define void @malloc_not_escape1
; CHECK-NOT: call void @__tsan_write4(ptr %m)
; CHECK-NOT: call void @__tsan_read4(ptr %m)

define void @malloc_not_escape2() nounwind uwtable sanitize_thread {
entry:
  %p = alloca ptr, align 8
  %call = call noalias ptr @malloc(i64 noundef 400) #2
  store ptr %call, ptr %p, align 8
  %0 = load ptr, ptr %p, align 8
  %arrayidx = getelementptr inbounds i32, ptr %0, i64 33
  store i32 42, ptr %arrayidx, align 4
  ret void
}
; CHECK-LABEL: define void @malloc_not_escape2
; CHECK-NOT: call void @__tsan_write4(ptr %arrayidx)

; Heap allocation stored to global -> instrumented
define void @malloc_global_escape() nounwind uwtable sanitize_thread {
  %m = call noalias ptr @malloc(i64 320)
  store ptr %m, ptr @GPtr
  store i32 2, ptr %m
  %v = load i32, ptr %m
  ret void
}
; CHECK-LABEL: define void @malloc_global_escape
; CHECK: call void @__tsan_write8(ptr @GPtr)
; CHECK: call void @__tsan_write4(ptr %m)
; CHECK: call void @__tsan_read4(ptr %m)
