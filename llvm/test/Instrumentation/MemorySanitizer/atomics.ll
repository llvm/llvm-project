; RUN: opt < %s -msan-check-access-address=0 -S -passes=msan 2>&1 | FileCheck  \
; RUN: %s
; RUN: opt < %s -msan -msan-check-access-address=0 -S | FileCheck %s
; RUN: opt < %s -msan-check-access-address=0 -msan-track-origins=1 -S          \
; RUN: -passes=msan 2>&1 | FileCheck %s
; RUN: opt < %s -msan -msan-check-access-address=0 -msan-track-origins=1 -S | FileCheck %s
; RUN: opt < %s -msan-check-access-address=0 -msan-track-origins=2 -S          \
; RUN: -passes=msan 2>&1 | FileCheck %s
; RUN: opt < %s -msan -msan-check-access-address=0 -msan-track-origins=2 -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; atomicrmw xchg: store clean shadow, return clean shadow

define i32 @AtomicRmwXchg(i32* %p, i32 %x) sanitize_memory {
entry:
  %0 = atomicrmw xchg i32* %p, i32 %x seq_cst
  ret i32 %0
}

; CHECK-LABEL: @AtomicRmwXchg
; CHECK: store i32 0,
; CHECK: atomicrmw xchg {{.*}} seq_cst
; CHECK: store i32 0, {{.*}} @__msan_retval_tls
; CHECK: ret i32


; atomicrmw max: exactly the same as above

define i32 @AtomicRmwMax(i32* %p, i32 %x) sanitize_memory {
entry:
  %0 = atomicrmw max i32* %p, i32 %x seq_cst
  ret i32 %0
}

; CHECK-LABEL: @AtomicRmwMax
; CHECK: store i32 0,
; CHECK: atomicrmw max {{.*}} seq_cst
; CHECK: store i32 0, {{.*}} @__msan_retval_tls
; CHECK: ret i32


; cmpxchg: the same as above, but also check %a shadow

define i32 @Cmpxchg(i32* %p, i32 %a, i32 %b) sanitize_memory {
entry:
  %pair = cmpxchg i32* %p, i32 %a, i32 %b seq_cst seq_cst
  %0 = extractvalue { i32, i1 } %pair, 0
  ret i32 %0
}

; CHECK-LABEL: @Cmpxchg
; CHECK: store { i32, i1 } zeroinitializer,
; CHECK: icmp
; CHECK: br
; CHECK: @__msan_warning
; CHECK: cmpxchg {{.*}} seq_cst seq_cst
; CHECK: store i32 0, {{.*}} @__msan_retval_tls
; CHECK: ret i32


; relaxed cmpxchg: bump up to "release monotonic"

define i32 @CmpxchgMonotonic(i32* %p, i32 %a, i32 %b) sanitize_memory {
entry:
  %pair = cmpxchg i32* %p, i32 %a, i32 %b monotonic monotonic
  %0 = extractvalue { i32, i1 } %pair, 0
  ret i32 %0
}

; CHECK-LABEL: @CmpxchgMonotonic
; CHECK: store { i32, i1 } zeroinitializer,
; CHECK: icmp
; CHECK: br
; CHECK: @__msan_warning
; CHECK: cmpxchg {{.*}} release monotonic
; CHECK: store i32 0, {{.*}} @__msan_retval_tls
; CHECK: ret i32


; atomic load: preserve alignment, load shadow value after app value

define i32 @AtomicLoad(i32* %p) sanitize_memory {
entry:
  %0 = load atomic i32, i32* %p seq_cst, align 16
  ret i32 %0
}

; CHECK-LABEL: @AtomicLoad
; CHECK: load atomic i32, i32* {{.*}} seq_cst, align 16
; CHECK: [[SHADOW:%[01-9a-z_]+]] = load i32, i32* {{.*}}, align 16
; CHECK: store i32 {{.*}}[[SHADOW]], {{.*}} @__msan_retval_tls
; CHECK: ret i32


; atomic load: preserve alignment, load shadow value after app value

define i32 @AtomicLoadAcquire(i32* %p) sanitize_memory {
entry:
  %0 = load atomic i32, i32* %p acquire, align 16
  ret i32 %0
}

; CHECK-LABEL: @AtomicLoadAcquire
; CHECK: load atomic i32, i32* {{.*}} acquire, align 16
; CHECK: [[SHADOW:%[01-9a-z_]+]] = load i32, i32* {{.*}}, align 16
; CHECK: store i32 {{.*}}[[SHADOW]], {{.*}} @__msan_retval_tls
; CHECK: ret i32


; atomic load monotonic: bump up to load acquire

define i32 @AtomicLoadMonotonic(i32* %p) sanitize_memory {
entry:
  %0 = load atomic i32, i32* %p monotonic, align 16
  ret i32 %0
}

; CHECK-LABEL: @AtomicLoadMonotonic
; CHECK: load atomic i32, i32* {{.*}} acquire, align 16
; CHECK: [[SHADOW:%[01-9a-z_]+]] = load i32, i32* {{.*}}, align 16
; CHECK: store i32 {{.*}}[[SHADOW]], {{.*}} @__msan_retval_tls
; CHECK: ret i32


; atomic load unordered: bump up to load acquire

define i32 @AtomicLoadUnordered(i32* %p) sanitize_memory {
entry:
  %0 = load atomic i32, i32* %p unordered, align 16
  ret i32 %0
}

; CHECK-LABEL: @AtomicLoadUnordered
; CHECK: load atomic i32, i32* {{.*}} acquire, align 16
; CHECK: [[SHADOW:%[01-9a-z_]+]] = load i32, i32* {{.*}}, align 16
; CHECK: store i32 {{.*}}[[SHADOW]], {{.*}} @__msan_retval_tls
; CHECK: ret i32


; atomic store: preserve alignment, store clean shadow value before app value

define void @AtomicStore(i32* %p, i32 %x) sanitize_memory {
entry:
  store atomic i32 %x, i32* %p seq_cst, align 16
  ret void
}

; CHECK-LABEL: @AtomicStore
; CHECK-NOT: @__msan_param_tls
; CHECK: store i32 0, i32* {{.*}}, align 16
; CHECK: store atomic i32 %x, i32* %p seq_cst, align 16
; CHECK: ret void


; atomic store: preserve alignment, store clean shadow value before app value

define void @AtomicStoreRelease(i32* %p, i32 %x) sanitize_memory {
entry:
  store atomic i32 %x, i32* %p release, align 16
  ret void
}

; CHECK-LABEL: @AtomicStoreRelease
; CHECK-NOT: @__msan_param_tls
; CHECK: store i32 0, i32* {{.*}}, align 16
; CHECK: store atomic i32 %x, i32* %p release, align 16
; CHECK: ret void


; atomic store monotonic: bumped up to store release

define void @AtomicStoreMonotonic(i32* %p, i32 %x) sanitize_memory {
entry:
  store atomic i32 %x, i32* %p monotonic, align 16
  ret void
}

; CHECK-LABEL: @AtomicStoreMonotonic
; CHECK-NOT: @__msan_param_tls
; CHECK: store i32 0, i32* {{.*}}, align 16
; CHECK: store atomic i32 %x, i32* %p release, align 16
; CHECK: ret void


; atomic store unordered: bumped up to store release

define void @AtomicStoreUnordered(i32* %p, i32 %x) sanitize_memory {
entry:
  store atomic i32 %x, i32* %p unordered, align 16
  ret void
}

; CHECK-LABEL: @AtomicStoreUnordered
; CHECK-NOT: @__msan_param_tls
; CHECK: store i32 0, i32* {{.*}}, align 16
; CHECK: store atomic i32 %x, i32* %p release, align 16
; CHECK: ret void
