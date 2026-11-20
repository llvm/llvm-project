; RUN: opt -verify-each -passes=attributor -S < %s | FileCheck %s
; RUN: opt -verify-each -passes=attributor-cgscc -S < %s | FileCheck %s
; RUN: opt -verify-each -passes='attributor,default<O2>' -S < %s | FileCheck %s

target datalayout = "e-p:64:64"

@saved = internal global ptr null
@shared = internal global ptr null

declare noalias nonnull ptr @malloc(i64) allocsize(0)
declare i32 @consume(ptr captures(none)) nofree

; The load in read_saved is not equivalent to this store for capture analysis:
; it can access the pointee after save returns.
; CHECK-LABEL: define void @save(
; CHECK-NOT: captures(none)
; CHECK-SAME: %p)
define void @save(ptr %p) noinline {
  store ptr %p, ptr @saved
  ret void
}

define i32 @read_saved() {
  %p = load ptr, ptr @saved
  %v = load i32, ptr %p
  ret i32 %v
}

; A second invocation can read the pointer saved by the first invocation.
; Keeping the uses within the same Function is therefore insufficient.
; CHECK-LABEL: define i32 @save_or_read(
; CHECK-NOT: captures(none)
; CHECK-SAME: %p, i1
; CHECK-SAME: %write)
define i32 @save_or_read(ptr %p, i1 %write) noinline {
  br i1 %write, label %store, label %read
store:
  store ptr %p, ptr @shared
  ret i32 0
read:
  %q = load ptr, ptr @shared
  %v = load i32, ptr %q
  ret i32 %v
}

; The allocation and initialization must survive: the pointer is published by
; save, and the object is read after this function returns.
; CHECK-LABEL: define void @initialize()
; CHECK: %[[INIT:.*]] = {{.*}}call {{.*}}ptr @malloc(
; CHECK: store i32 42, ptr %[[INIT]]
; CHECK: call void @save(
define void @initialize() {
  %p = call ptr @malloc(i64 4) allockind("alloc,uninitialized")
  store i32 42, ptr %p
  call void @save(ptr %p)
  ret void
}

; A pointer forwarded through the global must not be moved into this frame.
; A musttail callee cannot access its caller's stack allocations.
; CHECK-LABEL: define i32 @tail_caller(
; CHECK: %[[TAIL:.*]] = {{.*}}call {{.*}}ptr @malloc(
; CHECK: store i32 42, ptr %[[TAIL]]
; CHECK: musttail call i32 @consume(
define i32 @tail_caller(ptr %unused) {
  %p = call ptr @malloc(i64 4) allockind("alloc,uninitialized")
  store i32 42, ptr %p
  call void @save(ptr %p)
  %q = load ptr, ptr @saved
  %v = musttail call i32 @consume(ptr %q)
  ret i32 %v
}

; Continue looking through a non-escaping local stack slot, including a GEP.
; CHECK-LABEL: define {{.*}}i32 @local_spill(
; CHECK-SAME: captures(none) %p)
define i32 @local_spill(ptr %p) {
  %slots = alloca [2 x ptr]
  %slot = getelementptr [2 x ptr], ptr %slots, i64 0, i64 1
  store ptr %p, ptr %slot
  %q = load ptr, ptr %slot
  %v = load i32, ptr %q
  ret i32 %v
}

; A local slot does not make the stored pointer non-capturing if it is returned.
; CHECK-LABEL: define {{.*}}ptr @local_return(
; CHECK-NOT: captures(none)
; CHECK-SAME: %p)
define ptr @local_return(ptr %p) {
  %slot = alloca ptr
  store ptr %p, ptr %slot
  %q = load ptr, ptr %slot
  ret ptr %q
}

; Looking through a local slot must still discover subsequent publication.
; CHECK-LABEL: define void @local_publish(
; CHECK-NOT: captures(none)
; CHECK-SAME: %p)
define void @local_publish(ptr %p) {
  %slot = alloca ptr
  store ptr %p, ptr %slot
  call void @publish_slot(ptr %slot)
  ret void
}

define internal void @publish_slot(ptr %slot) noinline {
  %p = load ptr, ptr %slot
  store ptr %p, ptr @saved
  ret void
}

; A callback may load the pointer while the slot is live, then use the pointee
; after this function returns. Local storage alone does not justify following
; copies into another function.
; CHECK-LABEL: define void @publish_to_worker(
; CHECK-NOT: captures(none)
; CHECK-SAME: %p)
define void @publish_to_worker(ptr %p) noinline {
  %slot = alloca ptr
  store ptr %p, ptr %slot
  call void @dispatch(ptr @worker, ptr %slot)
  ret void
}

@loaded = external global i32
@resume = external global i32
@result = external global i32

declare !callback !0 void @dispatch(ptr, ptr)

define internal void @worker(ptr %slot) noinline {
  %p = load ptr, ptr %slot
  store atomic i32 1, ptr @loaded release, align 4
  br label %wait
wait:
  %go = load atomic i32, ptr @resume acquire, align 4
  %ready = icmp ne i32 %go, 0
  br i1 %ready, label %read, label %wait
read:
  %v = load i32, ptr %p
  store atomic i32 %v, ptr @result release, align 4
  ret void
}

!0 = !{!1}
!1 = !{i64 0, i64 1, i1 false}
