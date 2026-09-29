; RUN: opt < %s -msan-check-access-address=0 -S -passes=msan 2>&1 | FileCheck %s --check-prefixes=CHECK,NOORIGINS --implicit-check-not="call void @__msan_warning"
; RUN: opt < %s -msan-check-access-address=0 -msan-track-origins=1 -S -passes=msan 2>&1 | FileCheck %s --check-prefixes=CHECK,ORIGINS --implicit-check-not="call void @__msan_warning"
; RUN: opt < %s -msan-check-access-address=0 -msan-track-origins=2 -S -passes=msan 2>&1 | FileCheck %s --check-prefixes=CHECK,ORIGINS --implicit-check-not="call void @__msan_warning"
; RUN: opt < %s -msan-check-access-address=0 -msan-track-origins=1 -S -passes=msan -mtriple=s390x-unknown-linux 2>&1 | FileCheck %s --check-prefix=EXT
; REQUIRES: x86-registered-target, systemz-registered-target

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; atomicrmw xchg: store clean shadow, return clean shadow

define i32 @AtomicRmwXchg(ptr %p, i32 %x) sanitize_memory {
entry:
  %0 = atomicrmw xchg ptr %p, i32 %x seq_cst
  ret i32 %0
}

; CHECK-LABEL: @AtomicRmwXchg
; CHECK: store i32 0,
; CHECK: atomicrmw xchg {{.*}} seq_cst
; CHECK: store i32 0, {{.*}} @__msan_retval_tls
; CHECK: ret i32

; atomicrmw xchg ptr: exactly the same as above

define ptr @AtomicRmwXchgPtr(ptr %p, ptr %x) sanitize_memory {
entry:
  %0 = atomicrmw xchg ptr %p, ptr %x seq_cst
  ret ptr %0
}

; CHECK-LABEL: @AtomicRmwXchgPtr
; CHECK: store i64 0,
; CHECK: atomicrmw xchg {{.*}} seq_cst
; CHECK: store i64 0, {{.*}} @__msan_retval_tls
; CHECK: ret ptr


; atomicrmw max: exactly the same as above

define i32 @AtomicRmwMax(ptr %p, i32 %x) sanitize_memory {
entry:
  %0 = atomicrmw max ptr %p, i32 %x seq_cst
  ret i32 %0
}

; CHECK-LABEL: @AtomicRmwMax
; CHECK: store i32 0,
; CHECK: atomicrmw max {{.*}} seq_cst
; CHECK: store i32 0, {{.*}} @__msan_retval_tls
; CHECK: ret i32


; cmpxchg: the same as above, but also check %a shadow

define i32 @Cmpxchg(ptr %p, i32 %a, i32 %b) sanitize_memory {
entry:
  %pair = cmpxchg ptr %p, i32 %a, i32 %b seq_cst seq_cst
  %0 = extractvalue { i32, i1 } %pair, 0
  ret i32 %0
}

; CHECK-LABEL: @Cmpxchg
; CHECK: store i32 0,
; CHECK: icmp
; CHECK: br
; NOORIGINS: @__msan_warning_noreturn()
; ORIGINS: @__msan_warning_with_origin_noreturn(
; CHECK: cmpxchg {{.*}} seq_cst seq_cst
; CHECK: store i32 0, {{.*}} @__msan_retval_tls
; CHECK: ret i32


; relaxed cmpxchg: bump up to "release monotonic"

define i32 @CmpxchgMonotonic(ptr %p, i32 %a, i32 %b) sanitize_memory {
entry:
  %pair = cmpxchg ptr %p, i32 %a, i32 %b monotonic monotonic
  %0 = extractvalue { i32, i1 } %pair, 0
  ret i32 %0
}

; CHECK-LABEL: @CmpxchgMonotonic
; CHECK: store i32 0,
; CHECK: icmp
; CHECK: br
; NOORIGINS: @__msan_warning_noreturn()
; ORIGINS: @__msan_warning_with_origin_noreturn(
; CHECK: cmpxchg {{.*}} release monotonic
; CHECK: store i32 0, {{.*}} @__msan_retval_tls
; CHECK: ret i32


; atomic load: preserve alignment, load shadow value after app value

define i32 @AtomicLoad(ptr %p) sanitize_memory {
entry:
  %0 = load atomic i32, ptr %p seq_cst, align 16
  ret i32 %0
}

; CHECK-LABEL: @AtomicLoad
; CHECK: load atomic i32, ptr {{.*}} seq_cst, align 16
; CHECK: [[SHADOW:%[01-9a-z_]+]] = load i32, ptr {{.*}}, align 16
; CHECK: store i32 {{.*}}[[SHADOW]], {{.*}} @__msan_retval_tls
; CHECK: ret i32


; atomic load: preserve alignment, load shadow value after app value

define i32 @AtomicLoadAcquire(ptr %p) sanitize_memory {
entry:
  %0 = load atomic i32, ptr %p acquire, align 16
  ret i32 %0
}

; CHECK-LABEL: @AtomicLoadAcquire
; CHECK: load atomic i32, ptr {{.*}} acquire, align 16
; CHECK: [[SHADOW:%[01-9a-z_]+]] = load i32, ptr {{.*}}, align 16
; CHECK: store i32 {{.*}}[[SHADOW]], {{.*}} @__msan_retval_tls
; CHECK: ret i32


; atomic load monotonic: bump up to load acquire

define i32 @AtomicLoadMonotonic(ptr %p) sanitize_memory {
entry:
  %0 = load atomic i32, ptr %p monotonic, align 16
  ret i32 %0
}

; CHECK-LABEL: @AtomicLoadMonotonic
; CHECK: load atomic i32, ptr {{.*}} acquire, align 16
; CHECK: [[SHADOW:%[01-9a-z_]+]] = load i32, ptr {{.*}}, align 16
; CHECK: store i32 {{.*}}[[SHADOW]], {{.*}} @__msan_retval_tls
; CHECK: ret i32


; atomic load unordered: bump up to load acquire

define i32 @AtomicLoadUnordered(ptr %p) sanitize_memory {
entry:
  %0 = load atomic i32, ptr %p unordered, align 16
  ret i32 %0
}

; CHECK-LABEL: @AtomicLoadUnordered
; CHECK: load atomic i32, ptr {{.*}} acquire, align 16
; CHECK: [[SHADOW:%[01-9a-z_]+]] = load i32, ptr {{.*}}, align 16
; CHECK: store i32 {{.*}}[[SHADOW]], {{.*}} @__msan_retval_tls
; CHECK: ret i32


; atomic store: preserve alignment, store clean shadow value before app value

define void @AtomicStore(ptr %p, i32 %x) sanitize_memory {
entry:
  store atomic i32 %x, ptr %p seq_cst, align 16
  ret void
}

; CHECK-LABEL: @AtomicStore
; CHECK-NOT: @__msan_param_tls
; CHECK: store i32 0, ptr {{.*}}, align 16
; CHECK: store atomic i32 %x, ptr %p seq_cst, align 16
; CHECK: ret void


; atomic store: preserve alignment, store clean shadow value before app value

define void @AtomicStoreRelease(ptr %p, i32 %x) sanitize_memory {
entry:
  store atomic i32 %x, ptr %p release, align 16
  ret void
}

; CHECK-LABEL: @AtomicStoreRelease
; CHECK-NOT: @__msan_param_tls
; CHECK: store i32 0, ptr {{.*}}, align 16
; CHECK: store atomic i32 %x, ptr %p release, align 16
; CHECK: ret void


; atomic store monotonic: bumped up to store release

define void @AtomicStoreMonotonic(ptr %p, i32 %x) sanitize_memory {
entry:
  store atomic i32 %x, ptr %p monotonic, align 16
  ret void
}

; CHECK-LABEL: @AtomicStoreMonotonic
; CHECK-NOT: @__msan_param_tls
; CHECK: store i32 0, ptr {{.*}}, align 16
; CHECK: store atomic i32 %x, ptr %p release, align 16
; CHECK: ret void


; atomic store unordered: bumped up to store release

define void @AtomicStoreUnordered(ptr %p, i32 %x) sanitize_memory {
entry:
  store atomic i32 %x, ptr %p unordered, align 16
  ret void
}

; CHECK-LABEL: @AtomicStoreUnordered
; CHECK-NOT: @__msan_param_tls
; CHECK: store i32 0, ptr {{.*}}, align 16
; CHECK: store atomic i32 %x, ptr %p release, align 16
; CHECK: ret void


; ORIGINS: declare i32 @__msan_chain_origin(i32)
; EXT:     declare zeroext i32 @__msan_chain_origin(i32 zeroext)
; ORIGINS: declare void @__msan_set_origin(ptr, i64, i32)
; EXT:     declare void @__msan_set_origin(ptr, i64, i32 zeroext)
; ORIGINS: declare ptr @__msan_memset(ptr, i32, i64)
; EXT:     declare ptr @__msan_memset(ptr, i32 signext, i64)
; ORIGINS: declare void @__msan_warning_with_origin_noreturn(i32)
; EXT:     declare void @__msan_warning_with_origin_noreturn(i32 zeroext)
