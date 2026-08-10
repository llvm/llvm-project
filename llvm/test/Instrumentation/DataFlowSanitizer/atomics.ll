; RUN: opt < %s -passes=dfsan -S | FileCheck %s
; RUN: opt < %s -passes=dfsan -dfsan-track-origins=1 -S | FileCheck %s --check-prefixes=CHECK,CHECK_ORIGIN
; RUN: opt < %s -passes=dfsan -dfsan-track-origins=1 -dfsan-instrument-with-call-threshold=0 -S | FileCheck %s --check-prefixes=CHECK,CHECK_ORIGIN
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_arg_tls = external thread_local(initialexec) global [[TLS_ARR:\[100 x i64\]]]
; CHECK: @__dfsan_retval_tls = external thread_local(initialexec) global [[TLS_ARR]]
define i32 @AtomicRmwXchg(ptr %p, i32 %x) {
entry:
  ; COMM: atomicrmw xchg: store clean shadow/origin, return clean shadow/origin

  ; CHECK-LABEL:       @AtomicRmwXchg.dfsan
  ; CHECK-NOT:         @__dfsan_arg_origin_tls
  ; CHECK-NOT:         @__dfsan_arg_tls
  ; CHECK:             %[[#INTP:]] = ptrtoint ptr %p to i64
  ; CHECK-NEXT:        %[[#SHADOW_OFFSET:]] = xor i64 %[[#INTP]], [[#%.10d,MASK:]]
  ; CHECK-NEXT:        %[[#SHADOW_PTR:]] = inttoptr i64 %[[#SHADOW_OFFSET]] to ptr
  ; CHECK-NEXT:        store i[[#NUM_BITS:32]] 0, ptr %[[#SHADOW_PTR]], align 1
  ; CHECK-NEXT:        atomicrmw xchg ptr %p, i32 %x seq_cst
  ; CHECK-NEXT:        store i8 0, ptr @__dfsan_retval_tls, align 2
  ; CHECK_ORIGIN-NEXT: store i32 0, ptr @__dfsan_retval_origin_tls, align 4
  ; CHECK-NEXT:        ret i32

  %0 = atomicrmw xchg ptr %p, i32 %x seq_cst
  ret i32 %0
}

define i32 @AtomicRmwMax(ptr %p, i32 %x) {
  ; COMM: atomicrmw max: exactly the same as above

  ; CHECK-LABEL:       @AtomicRmwMax.dfsan
  ; CHECK-NOT:         @__dfsan_arg_origin_tls
  ; CHECK-NOT:         @__dfsan_arg_tls
  ; CHECK:             %[[#INTP:]] = ptrtoint ptr %p to i64
  ; CHECK-NEXT:        %[[#SHADOW_OFFSET:]] = xor i64 %[[#INTP]], [[#%.10d,MASK:]]
  ; CHECK-NEXT:        %[[#SHADOW_PTR:]] = inttoptr i64 %[[#SHADOW_OFFSET]] to ptr
  ; CHECK-NEXT:        store i[[#NUM_BITS]] 0, ptr %[[#SHADOW_PTR]], align 1
  ; CHECK-NEXT:        atomicrmw max ptr %p, i32 %x seq_cst
  ; CHECK-NEXT:        store i8 0, ptr @__dfsan_retval_tls, align 2
  ; CHECK_ORIGIN-NEXT: store i32 0, ptr @__dfsan_retval_origin_tls, align 4
  ; CHECK-NEXT:        ret i32

entry:
  %0 = atomicrmw max ptr %p, i32 %x seq_cst
  ret i32 %0
}


define i32 @Cmpxchg(ptr %p, i32 %a, i32 %b) {
  ; COMM: cmpxchg: store clean shadow/origin, return clean shadow/origin

  ; CHECK-LABEL:       @Cmpxchg.dfsan
  ; CHECK-NOT:         @__dfsan_arg_origin_tls
  ; CHECK-NOT:         @__dfsan_arg_tls
  ; CHECK:             %[[#INTP:]] = ptrtoint ptr %p to i64
  ; CHECK-NEXT:        %[[#SHADOW_OFFSET:]] = xor i64 %[[#INTP]], [[#%.10d,MASK:]]
  ; CHECK-NEXT:        %[[#SHADOW_PTR:]] = inttoptr i64 %[[#SHADOW_OFFSET]] to ptr
  ; CHECK-NEXT:        store i[[#NUM_BITS]] 0, ptr %[[#SHADOW_PTR]], align 1
  ; CHECK-NEXT:        %pair = cmpxchg ptr %p, i32 %a, i32 %b seq_cst seq_cst
  ; CHECK:             store i8 0, ptr @__dfsan_retval_tls, align 2
  ; CHECK_ORIGIN-NEXT: store i32 0, ptr @__dfsan_retval_origin_tls, align 4
  ; CHECK-NEXT:        ret i32

entry:
  %pair = cmpxchg ptr %p, i32 %a, i32 %b seq_cst seq_cst
  %0 = extractvalue { i32, i1 } %pair, 0
  ret i32 %0
}


define i32 @CmpxchgMonotonic(ptr %p, i32 %a, i32 %b) {
  ; COMM: relaxed cmpxchg: bump up to "release monotonic"

  ; CHECK-LABEL:       @CmpxchgMonotonic.dfsan
  ; CHECK-NOT:         @__dfsan_arg_origin_tls
  ; CHECK-NOT:         @__dfsan_arg_tls
  ; CHECK:             %[[#INTP:]] = ptrtoint ptr %p to i64
  ; CHECK-NEXT:        %[[#SHADOW_OFFSET:]] = xor i64 %[[#INTP]], [[#%.10d,MASK:]]
  ; CHECK-NEXT:        %[[#SHADOW_PTR:]] = inttoptr i64 %[[#SHADOW_OFFSET]] to ptr
  ; CHECK-NEXT:        store i[[#NUM_BITS]] 0, ptr %[[#SHADOW_PTR]], align 1
  ; CHECK-NEXT:        %pair = cmpxchg ptr %p, i32 %a, i32 %b release monotonic
  ; CHECK:             store i8 0, ptr @__dfsan_retval_tls, align 2
  ; CHECK_ORIGIN-NEXT: store i32 0, ptr @__dfsan_retval_origin_tls, align 4
  ; CHECK-NEXT:        ret i32

entry:
  %pair = cmpxchg ptr %p, i32 %a, i32 %b monotonic monotonic
  %0 = extractvalue { i32, i1 } %pair, 0
  ret i32 %0
}



define i32 @AtomicLoad(ptr %p) {
  ; COMM: atomic load: load shadow value after app value

  ; CHECK-LABEL:  @AtomicLoad.dfsan
  ; CHECK_ORIGIN: %[[#PO:]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
  ; CHECK:        %[[#PS:]] = load i8, ptr @__dfsan_arg_tls, align 2
  ; CHECK:        %a = load atomic i32, ptr %p seq_cst, align 16
  ; CHECK:        %[[#SHADOW_PTR:]] = inttoptr i64 {{.*}} to ptr
  ; CHECK_ORIGIN: %[[#ORIGIN_PTR:]] = inttoptr i64 {{.*}} to ptr
  ; CHECK_ORIGIN: %[[#AO:]] = load i32, ptr %[[#ORIGIN_PTR]], align 16
  ; CHECK:        load i[[#NUM_BITS]], ptr %[[#SHADOW_PTR]], align 1
  ; CHECK:        %[[#AP_S:]] = or i8 {{.*}}, %[[#PS]]
  ; CHECK_ORIGIN: %[[#PS_NZ:]] = icmp ne i8 %[[#PS]], 0
  ; CHECK_ORIGIN: %[[#AP_O:]] = select i1 %[[#PS_NZ]], i32 %[[#PO]], i32 %[[#AO]]
  ; CHECK:        store i8 %[[#AP_S]], ptr @__dfsan_retval_tls, align 2
  ; CHECK_ORIGIN: store i32 %[[#AP_O]], ptr @__dfsan_retval_origin_tls, align 4
  ; CHECK:        ret i32 %a

entry:
  %a = load atomic i32, ptr %p seq_cst, align 16
  ret i32 %a
}


define i32 @AtomicLoadAcquire(ptr %p) {
  ; COMM: atomic load: load shadow value after app value

  ; CHECK-LABEL:  @AtomicLoadAcquire.dfsan
  ; CHECK_ORIGIN: %[[#PO:]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
  ; CHECK:        %[[#PS:]] = load i8, ptr @__dfsan_arg_tls, align 2
  ; CHECK:        %a = load atomic i32, ptr %p acquire, align 16
  ; CHECK:        %[[#SHADOW_PTR:]] = inttoptr i64 {{.*}} to ptr
  ; CHECK_ORIGIN: %[[#ORIGIN_PTR:]] = inttoptr i64 {{.*}} to ptr
  ; CHECK_ORIGIN: %[[#AO:]] = load i32, ptr %[[#ORIGIN_PTR]], align 16
  ; CHECK:        load i[[#NUM_BITS]], ptr %[[#SHADOW_PTR]], align 1
  ; CHECK:        %[[#AP_S:]] = or i8 {{.*}}, %[[#PS]]
  ; CHECK_ORIGIN: %[[#PS_NZ:]] = icmp ne i8 %[[#PS]], 0
  ; CHECK_ORIGIN: %[[#AP_O:]] = select i1 %[[#PS_NZ]], i32 %[[#PO]], i32 %[[#AO]]
  ; CHECK:        store i8 %[[#AP_S]], ptr @__dfsan_retval_tls, align 2
  ; CHECK_ORIGIN: store i32 %[[#AP_O]], ptr @__dfsan_retval_origin_tls, align 4
  ; CHECK:        ret i32 %a

entry:
  %a = load atomic i32, ptr %p acquire, align 16
  ret i32 %a
}


define i32 @AtomicLoadMonotonic(ptr %p) {
  ; COMM: atomic load monotonic: bump up to load acquire

  ; CHECK-LABEL:  @AtomicLoadMonotonic.dfsan
  ; CHECK_ORIGIN: %[[#PO:]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
  ; CHECK:        %[[#PS:]] = load i8, ptr @__dfsan_arg_tls, align 2
  ; CHECK:        %a = load atomic i32, ptr %p acquire, align 16
  ; CHECK:        %[[#SHADOW_PTR:]] = inttoptr i64 {{.*}} to ptr
  ; CHECK_ORIGIN: %[[#ORIGIN_PTR:]] = inttoptr i64 {{.*}} to ptr
  ; CHECK_ORIGIN: %[[#AO:]] = load i32, ptr %[[#ORIGIN_PTR]], align 16
  ; CHECK:        load i[[#NUM_BITS]], ptr %[[#SHADOW_PTR]], align 1
  ; CHECK:        %[[#AP_S:]] = or i8 {{.*}}, %[[#PS]]
  ; CHECK_ORIGIN: %[[#PS_NZ:]] = icmp ne i8 %[[#PS]], 0
  ; CHECK_ORIGIN: %[[#AP_O:]] = select i1 %[[#PS_NZ]], i32 %[[#PO]], i32 %[[#AO]]
  ; CHECK:        store i8 %[[#AP_S]], ptr @__dfsan_retval_tls, align 2
  ; CHECK_ORIGIN: store i32 %[[#AP_O]], ptr @__dfsan_retval_origin_tls, align 4
  ; CHECK:        ret i32 %a

entry:
  %a = load atomic i32, ptr %p monotonic, align 16
  ret i32 %a
}

define i32 @AtomicLoadUnordered(ptr %p) {
  ; COMM: atomic load unordered: bump up to load acquire

  ; CHECK-LABEL:  @AtomicLoadUnordered.dfsan
  ; CHECK_ORIGIN: %[[#PO:]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
  ; CHECK:        %[[#PS:]] = load i8, ptr @__dfsan_arg_tls, align 2
  ; CHECK:        %a = load atomic i32, ptr %p acquire, align 16
  ; CHECK:        %[[#SHADOW_PTR:]] = inttoptr i64 {{.*}} to ptr
  ; CHECK_ORIGIN: %[[#ORIGIN_PTR:]] = inttoptr i64 {{.*}} to ptr
  ; CHECK_ORIGIN: %[[#AO:]] = load i32, ptr %[[#ORIGIN_PTR]], align 16
  ; CHECK:        load i[[#NUM_BITS]], ptr %[[#SHADOW_PTR]], align 1
  ; CHECK:        %[[#AP_S:]] = or i8 {{.*}}, %[[#PS]]
  ; CHECK_ORIGIN: %[[#PS_NZ:]] = icmp ne i8 %[[#PS]], 0
  ; CHECK_ORIGIN: %[[#AP_O:]] = select i1 %[[#PS_NZ]], i32 %[[#PO]], i32 %[[#AO]]
  ; CHECK:        store i8 %[[#AP_S]], ptr @__dfsan_retval_tls, align 2
  ; CHECK_ORIGIN: store i32 %[[#AP_O]], ptr @__dfsan_retval_origin_tls, align 4
  ; CHECK:        ret i32 %a

entry:
  %a = load atomic i32, ptr %p unordered, align 16
  ret i32 %a
}

define void @AtomicStore(ptr %p, i32 %x) {
  ; COMM: atomic store: store clean shadow value before app value

  ; CHECK-LABEL:       @AtomicStore.dfsan
  ; CHECK-NOT:         @__dfsan_arg_origin_tls
  ; CHECK-NOT:         @__dfsan_arg_tls
  ; CHECK_ORIGIN-NOT:  35184372088832
  ; CHECK:             %[[#INTP:]] = ptrtoint ptr %p to i64
  ; CHECK-NEXT:        %[[#SHADOW_OFFSET:]] = xor i64 %[[#INTP]], [[#%.10d,MASK:]]
  ; CHECK-NEXT:        %[[#SHADOW_PTR:]] = inttoptr i64 %[[#SHADOW_OFFSET]] to ptr
  ; CHECK-NEXT:        store i[[#NUM_BITS]] 0, ptr %[[#SHADOW_PTR]], align 1
  ; CHECK:             store atomic i32 %x, ptr %p seq_cst, align 16
  ; CHECK:             ret void

entry:
  store atomic i32 %x, ptr %p seq_cst, align 16
  ret void
}

define void @AtomicStoreRelease(ptr %p, i32 %x) {
  ; COMM: atomic store: store clean shadow value before app value

  ; CHECK-LABEL:       @AtomicStoreRelease.dfsan
  ; CHECK-NOT:         @__dfsan_arg_origin_tls
  ; CHECK-NOT:         @__dfsan_arg_tls
  ; CHECK_ORIGIN-NOT:  35184372088832
  ; CHECK:             %[[#INTP:]] = ptrtoint ptr %p to i64
  ; CHECK-NEXT:        %[[#SHADOW_OFFSET:]] = xor i64 %[[#INTP]], [[#%.10d,MASK:]]
  ; CHECK-NEXT:        %[[#SHADOW_PTR:]] = inttoptr i64 %[[#SHADOW_OFFSET]] to ptr
  ; CHECK-NEXT:        store i[[#NUM_BITS]] 0, ptr %[[#SHADOW_PTR]], align 1
  ; CHECK:             store atomic i32 %x, ptr %p release, align 16
  ; CHECK:             ret void

entry:
  store atomic i32 %x, ptr %p release, align 16
  ret void
}

define void @AtomicStoreMonotonic(ptr %p, i32 %x) {
  ; COMM: atomic store monotonic: bumped up to store release

  ; CHECK-LABEL:       @AtomicStoreMonotonic.dfsan
  ; CHECK-NOT:         @__dfsan_arg_origin_tls
  ; CHECK-NOT:         @__dfsan_arg_tls
  ; CHECK_ORIGIN-NOT:  35184372088832
  ; CHECK:             %[[#INTP:]] = ptrtoint ptr %p to i64
  ; CHECK-NEXT:        %[[#SHADOW_OFFSET:]] = xor i64 %[[#INTP]], [[#%.10d,MASK:]]
  ; CHECK-NEXT:        %[[#SHADOW_PTR:]] = inttoptr i64 %[[#SHADOW_OFFSET]] to ptr
  ; CHECK-NEXT:        store i[[#NUM_BITS]] 0, ptr %[[#SHADOW_PTR]], align 1
  ; CHECK:             store atomic i32 %x, ptr %p release, align 16
  ; CHECK:             ret void

entry:
  store atomic i32 %x, ptr %p monotonic, align 16
  ret void
}

define void @AtomicStoreUnordered(ptr %p, i32 %x) {
  ; COMM: atomic store unordered: bumped up to store release

  ; CHECK-LABEL: @AtomicStoreUnordered.dfsan
  ; CHECK-NOT:         @__dfsan_arg_origin_tls
  ; CHECK-NOT:         @__dfsan_arg_tls
  ; CHECK_ORIGIN-NOT:  35184372088832
  ; CHECK:             %[[#INTP:]] = ptrtoint ptr %p to i64
  ; CHECK-NEXT:        %[[#SHADOW_OFFSET:]] = xor i64 %[[#INTP]], [[#%.10d,MASK:]]
  ; CHECK-NEXT:        %[[#SHADOW_PTR:]] = inttoptr i64 %[[#SHADOW_OFFSET]] to ptr
  ; CHECK-NEXT:        store i[[#NUM_BITS]] 0, ptr %[[#SHADOW_PTR]], align 1
  ; CHECK:             store atomic i32 %x, ptr %p release, align 16
  ; CHECK:             ret void

entry:
  store atomic i32 %x, ptr %p unordered, align 16
  ret void
}
