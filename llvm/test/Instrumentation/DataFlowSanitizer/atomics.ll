; RUN: opt < %s -dfsan -dfsan-fast-16-labels=true -S | FileCheck %s --check-prefix=CHECK
;
; The patterns about origins cannot be tested until the origin tracking feature is complete.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; atomicrmw xchg: store clean shadow/origin, return clean shadow/origin

define i32 @AtomicRmwXchg(i32* %p, i32 %x) {
entry:
  %0 = atomicrmw xchg i32* %p, i32 %x seq_cst
  ret i32 %0
}

; CHECK-LABEL: @"dfs$AtomicRmwXchg"
; CHECK-NOT: @__dfsan_arg_origin_tls
; CHECK-NOT: @__dfsan_arg_tls
; CHECK: [[INTP:%.*]] = ptrtoint i32* %p to i64
; CHECK-NEXT: [[OFFSET:%.*]] = and i64 [[INTP]], -123145302310913
; CHECK-NEXT: [[SHADOW_ADDR:%.*]]  = mul i64 [[OFFSET]], 2
; CHECK-NEXT: [[SHADOW_PTR:%.*]] = inttoptr i64 [[SHADOW_ADDR]] to i16*
; CHECK-NEXT: [[SHADOW_PTR64:%.*]] = bitcast i16* [[SHADOW_PTR]] to i64*
; CHECK-NEXT: store i64 0, i64* [[SHADOW_PTR64]], align 2
; CHECK-NEXT: atomicrmw xchg i32* %p, i32 %x seq_cst
; CHECK-NEXT: store i16 0, i16* bitcast ([100 x i64]* @__dfsan_retval_tls to i16*), align 2
; CHECK_ORIGIN-NEXT: store i32 0, i32* @__dfsan_retval_origin_tls, align 4
; CHECK-NEXT: ret i32


; atomicrmw max: exactly the same as above

define i32 @AtomicRmwMax(i32* %p, i32 %x) {
entry:
  %0 = atomicrmw max i32* %p, i32 %x seq_cst
  ret i32 %0
}

; CHECK-LABEL: @"dfs$AtomicRmwMax"
; CHECK-NOT: @__dfsan_arg_origin_tls
; CHECK-NOT: @__dfsan_arg_tls
; CHECK: [[INTP:%.*]] = ptrtoint i32* %p to i64
; CHECK-NEXT: [[OFFSET:%.*]] = and i64 [[INTP]], -123145302310913
; CHECK-NEXT: [[SHADOW_ADDR:%.*]]  = mul i64 [[OFFSET]], 2
; CHECK-NEXT: [[SHADOW_PTR:%.*]] = inttoptr i64 [[SHADOW_ADDR]] to i16*
; CHECK-NEXT: [[SHADOW_PTR64:%.*]] = bitcast i16* [[SHADOW_PTR]] to i64*
; CHECK-NEXT: store i64 0, i64* [[SHADOW_PTR64]], align 2
; CHECK-NEXT: atomicrmw max i32* %p, i32 %x seq_cst
; CHECK-NEXT: store i16 0, i16* bitcast ([100 x i64]* @__dfsan_retval_tls to i16*), align 2
; CHECK_ORIGIN-NEXT: store i32 0, i32* @__dfsan_retval_origin_tls, align 4
; CHECK-NEXT: ret i32


; cmpxchg: store clean shadow/origin, return clean shadow/origin

define i32 @Cmpxchg(i32* %p, i32 %a, i32 %b) {
entry:
  %pair = cmpxchg i32* %p, i32 %a, i32 %b seq_cst seq_cst
  %0 = extractvalue { i32, i1 } %pair, 0
  ret i32 %0
}

; CHECK-LABEL: @"dfs$Cmpxchg"
; CHECK-NOT: @__dfsan_arg_origin_tls
; CHECK-NOT: @__dfsan_arg_tls
; CHECK: [[INTP:%.*]] = ptrtoint i32* %p to i64
; CHECK-NEXT: [[OFFSET:%.*]] = and i64 [[INTP]], -123145302310913
; CHECK-NEXT: [[SHADOW_ADDR:%.*]]  = mul i64 [[OFFSET]], 2
; CHECK-NEXT: [[SHADOW_PTR:%.*]] = inttoptr i64 [[SHADOW_ADDR]] to i16*
; CHECK-NEXT: [[SHADOW_PTR64:%.*]] = bitcast i16* [[SHADOW_PTR]] to i64*
; CHECK-NEXT: store i64 0, i64* [[SHADOW_PTR64]], align 2
; CHECK-NEXT: %pair = cmpxchg i32* %p, i32 %a, i32 %b seq_cst seq_cst
; CHECK: store i16 0, i16* bitcast ([100 x i64]* @__dfsan_retval_tls to i16*), align 2
; CHECK_ORIGIN-NEXT: store i32 0, i32* @__dfsan_retval_origin_tls, align 4
; CHECK-NEXT: ret i32


; relaxed cmpxchg: bump up to "release monotonic"

define i32 @CmpxchgMonotonic(i32* %p, i32 %a, i32 %b) {
entry:
  %pair = cmpxchg i32* %p, i32 %a, i32 %b monotonic monotonic
  %0 = extractvalue { i32, i1 } %pair, 0
  ret i32 %0
}

; CHECK-LABEL: @"dfs$CmpxchgMonotonic"
; CHECK-NOT: @__dfsan_arg_origin_tls
; CHECK-NOT: @__dfsan_arg_tls
; CHECK: [[INTP:%.*]] = ptrtoint i32* %p to i64
; CHECK-NEXT: [[OFFSET:%.*]] = and i64 [[INTP]], -123145302310913
; CHECK-NEXT: [[SHADOW_ADDR:%.*]]  = mul i64 [[OFFSET]], 2
; CHECK-NEXT: [[SHADOW_PTR:%.*]] = inttoptr i64 [[SHADOW_ADDR]] to i16*
; CHECK-NEXT: [[SHADOW_PTR64:%.*]] = bitcast i16* [[SHADOW_PTR]] to i64*
; CHECK-NEXT: store i64 0, i64* [[SHADOW_PTR64]], align 2
; CHECK-NEXT: %pair = cmpxchg i32* %p, i32 %a, i32 %b release monotonic
; CHECK: store i16 0, i16* bitcast ([100 x i64]* @__dfsan_retval_tls to i16*), align 2
; CHECK_ORIGIN-NEXT: store i32 0, i32* @__dfsan_retval_origin_tls, align 4
; CHECK-NEXT: ret i32


; atomic load: load shadow value after app value

define i32 @AtomicLoad(i32* %p) {
entry:
  %a = load atomic i32, i32* %p seq_cst, align 16
  ret i32 %a
}

; CHECK-LABEL: @"dfs$AtomicLoad"
; CHECK_ORIGIN: [[PO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
; CHECK: [[PS:%.*]] = load i16, i16* bitcast ([100 x i64]* @__dfsan_arg_tls to i16*), align 2
; CHECK: %a = load atomic i32, i32* %p seq_cst, align 16
; CHECK: [[INTP:%.*]] = ptrtoint {{.*}} %p to i64
; CHECK: [[OFFSET:%.*]] = and i64 [[INTP]], -123145302310913
; CHECK: [[SHADOW_ADDR:%.*]] = mul i64 [[OFFSET]], 2
; CHECK: [[SHADOW_PTR:%.*]] = inttoptr i64 [[SHADOW_ADDR]] to i16*
; CHECK_ORIGIN: [[ORIGIN_ADDR:%.*]] = add i64 [[OFFSET]], 35184372088832
; CHECK_ORIGIN: [[ORIGIN_PTR:%.*]] = inttoptr i64 [[ORIGIN_ADDR]] to i32*
; CHECK_ORIGIN: [[AO:%.*]] = load i32, i32* [[ORIGIN_PTR]], align 16
; CHECK: [[SHADOW_PTR64:%.*]] = bitcast i16* [[SHADOW_PTR]] to i64*
; CHECK: [[SHADOW64:%.*]] = load i64, i64* [[SHADOW_PTR64]], align 2
; CHECK: [[SHADOW64_H32:%.*]] = lshr i64 [[SHADOW64]], 32
; CHECK: [[SHADOW64_HL32:%.*]] = or i64 [[SHADOW64]], [[SHADOW64_H32]]
; CHECK: [[SHADOW64_HL32_H16:%.*]] = lshr i64 [[SHADOW64_HL32]], 16
; CHECK: [[SHADOW64_HL32_HL16:%.*]] = or i64 [[SHADOW64_HL32]], [[SHADOW64_HL32_H16]]
; CHECK: [[AS:%.*]] = trunc i64 [[SHADOW64_HL32_HL16]] to i16
; CHECK: [[AP_S:%.*]] = or i16 [[AS]], [[PS]]
; CHECK_ORIGIN: [[PS_NZ:%.*]] = icmp ne i16 [[PS]], 0
; CHECK_ORIGIN: [[AP_O:%.*]] = select i1 [[PS_NZ]], i32 [[PO]], i32 [[AO]]
; CHECK: store i16 [[AP_S]], i16* bitcast ([100 x i64]* @__dfsan_retval_tls to i16*), align 2
; CHECK_ORIGIN: store i32 [[AP_O]], i32* @__dfsan_retval_origin_tls, align 4
; CHECK: ret i32 %a


; atomic load: load shadow value after app value

define i32 @AtomicLoadAcquire(i32* %p) {
entry:
  %a = load atomic i32, i32* %p acquire, align 16
  ret i32 %a
}

; CHECK-LABEL: @"dfs$AtomicLoadAcquire"
; CHECK_ORIGIN: [[PO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
; CHECK: [[PS:%.*]] = load i16, i16* bitcast ([100 x i64]* @__dfsan_arg_tls to i16*), align 2
; CHECK: %a = load atomic i32, i32* %p acquire, align 16
; CHECK: [[INTP:%.*]] = ptrtoint {{.*}} %p to i64
; CHECK: [[OFFSET:%.*]] = and i64 [[INTP]], -123145302310913
; CHECK: [[SHADOW_ADDR:%.*]] = mul i64 [[OFFSET]], 2
; CHECK: [[SHADOW_PTR:%.*]] = inttoptr i64 [[SHADOW_ADDR]] to i16*
; CHECK_ORIGIN: [[ORIGIN_ADDR:%.*]] = add i64 [[OFFSET]], 35184372088832
; CHECK_ORIGIN: [[ORIGIN_PTR:%.*]] = inttoptr i64 [[ORIGIN_ADDR]] to i32*
; CHECK_ORIGIN: [[AO:%.*]] = load i32, i32* [[ORIGIN_PTR]], align 16
; CHECK: [[SHADOW_PTR64:%.*]] = bitcast i16* [[SHADOW_PTR]] to i64*
; CHECK: [[SHADOW64:%.*]] = load i64, i64* [[SHADOW_PTR64]], align 2
; CHECK: [[SHADOW64_H32:%.*]] = lshr i64 [[SHADOW64]], 32
; CHECK: [[SHADOW64_HL32:%.*]] = or i64 [[SHADOW64]], [[SHADOW64_H32]]
; CHECK: [[SHADOW64_HL32_H16:%.*]] = lshr i64 [[SHADOW64_HL32]], 16
; CHECK: [[SHADOW64_HL32_HL16:%.*]] = or i64 [[SHADOW64_HL32]], [[SHADOW64_HL32_H16]]
; CHECK: [[AS:%.*]] = trunc i64 [[SHADOW64_HL32_HL16]] to i16
; CHECK: [[AP_S:%.*]] = or i16 [[AS]], [[PS]]
; CHECK_ORIGIN: [[PS_NZ:%.*]] = icmp ne i16 [[PS]], 0
; CHECK_ORIGIN: [[AP_O:%.*]] = select i1 [[PS_NZ]], i32 [[PO]], i32 [[AO]]
; CHECK: store i16 [[AP_S]], i16* bitcast ([100 x i64]* @__dfsan_retval_tls to i16*), align 2
; CHECK_ORIGIN: store i32 [[AP_O]], i32* @__dfsan_retval_origin_tls, align 4
; CHECK: ret i32 %a


; atomic load monotonic: bump up to load acquire

define i32 @AtomicLoadMonotonic(i32* %p) {
entry:
  %a = load atomic i32, i32* %p monotonic, align 16
  ret i32 %a
}

; CHECK-LABEL: @"dfs$AtomicLoadMonotonic"
; CHECK_ORIGIN: [[PO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
; CHECK: [[PS:%.*]] = load i16, i16* bitcast ([100 x i64]* @__dfsan_arg_tls to i16*), align 2
; CHECK: %a = load atomic i32, i32* %p acquire, align 16
; CHECK: [[INTP:%.*]] = ptrtoint {{.*}} %p to i64
; CHECK: [[OFFSET:%.*]] = and i64 [[INTP]], -123145302310913
; CHECK: [[SHADOW_ADDR:%.*]] = mul i64 [[OFFSET]], 2
; CHECK: [[SHADOW_PTR:%.*]] = inttoptr i64 [[SHADOW_ADDR]] to i16*
; CHECK_ORIGIN: [[ORIGIN_ADDR:%.*]] = add i64 [[OFFSET]], 35184372088832
; CHECK_ORIGIN: [[ORIGIN_PTR:%.*]] = inttoptr i64 [[ORIGIN_ADDR]] to i32*
; CHECK_ORIGIN: [[AO:%.*]] = load i32, i32* [[ORIGIN_PTR]], align 16
; CHECK: [[SHADOW_PTR64:%.*]] = bitcast i16* [[SHADOW_PTR]] to i64*
; CHECK: [[SHADOW64:%.*]] = load i64, i64* [[SHADOW_PTR64]], align 2
; CHECK: [[SHADOW64_H32:%.*]] = lshr i64 [[SHADOW64]], 32
; CHECK: [[SHADOW64_HL32:%.*]] = or i64 [[SHADOW64]], [[SHADOW64_H32]]
; CHECK: [[SHADOW64_HL32_H16:%.*]] = lshr i64 [[SHADOW64_HL32]], 16
; CHECK: [[SHADOW64_HL32_HL16:%.*]] = or i64 [[SHADOW64_HL32]], [[SHADOW64_HL32_H16]]
; CHECK: [[AS:%.*]] = trunc i64 [[SHADOW64_HL32_HL16]] to i16
; CHECK: [[AP_S:%.*]] = or i16 [[AS]], [[PS]]
; CHECK_ORIGIN: [[PS_NZ:%.*]] = icmp ne i16 [[PS]], 0
; CHECK_ORIGIN: [[AP_O:%.*]] = select i1 [[PS_NZ]], i32 [[PO]], i32 [[AO]]
; CHECK: store i16 [[AP_S]], i16* bitcast ([100 x i64]* @__dfsan_retval_tls to i16*), align 2
; CHECK_ORIGIN: store i32 [[AP_O]], i32* @__dfsan_retval_origin_tls, align 4
; CHECK: ret i32 %a


; atomic load unordered: bump up to load acquire

define i32 @AtomicLoadUnordered(i32* %p) {
entry:
  %a = load atomic i32, i32* %p unordered, align 16
  ret i32 %a
}

; CHECK-LABEL: @"dfs$AtomicLoadUnordered"
; CHECK_ORIGIN: [[PO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
; CHECK: [[PS:%.*]] = load i16, i16* bitcast ([100 x i64]* @__dfsan_arg_tls to i16*), align 2
; CHECK: %a = load atomic i32, i32* %p acquire, align 16
; CHECK: [[INTP:%.*]] = ptrtoint {{.*}} %p to i64
; CHECK: [[OFFSET:%.*]] = and i64 [[INTP]], -123145302310913
; CHECK: [[SHADOW_ADDR:%.*]] = mul i64 [[OFFSET]], 2
; CHECK: [[SHADOW_PTR:%.*]] = inttoptr i64 [[SHADOW_ADDR]] to i16*
; CHECK_ORIGIN: [[ORIGIN_ADDR:%.*]] = add i64 [[OFFSET]], 35184372088832
; CHECK_ORIGIN: [[ORIGIN_PTR:%.*]] = inttoptr i64 [[ORIGIN_ADDR]] to i32*
; CHECK_ORIGIN: [[AO:%.*]] = load i32, i32* [[ORIGIN_PTR]], align 16
; CHECK: [[SHADOW_PTR64:%.*]] = bitcast i16* [[SHADOW_PTR]] to i64*
; CHECK: [[SHADOW64:%.*]] = load i64, i64* [[SHADOW_PTR64]], align 2
; CHECK: [[SHADOW64_H32:%.*]] = lshr i64 [[SHADOW64]], 32
; CHECK: [[SHADOW64_HL32:%.*]] = or i64 [[SHADOW64]], [[SHADOW64_H32]]
; CHECK: [[SHADOW64_HL32_H16:%.*]] = lshr i64 [[SHADOW64_HL32]], 16
; CHECK: [[SHADOW64_HL32_HL16:%.*]] = or i64 [[SHADOW64_HL32]], [[SHADOW64_HL32_H16]]
; CHECK: [[AS:%.*]] = trunc i64 [[SHADOW64_HL32_HL16]] to i16
; CHECK: [[AP_S:%.*]] = or i16 [[AS]], [[PS]]
; CHECK_ORIGIN: [[PS_NZ:%.*]] = icmp ne i16 [[PS]], 0
; CHECK_ORIGIN: [[AP_O:%.*]] = select i1 [[PS_NZ]], i32 [[PO]], i32 [[AO]]
; CHECK: store i16 [[AP_S]], i16* bitcast ([100 x i64]* @__dfsan_retval_tls to i16*), align 2
; CHECK_ORIGIN: store i32 [[AP_O]], i32* @__dfsan_retval_origin_tls, align 4
; CHECK: ret i32 %a


; atomic store: store clean shadow value before app value

define void @AtomicStore(i32* %p, i32 %x) {
entry:
  store atomic i32 %x, i32* %p seq_cst, align 16
  ret void
}

; CHECK-LABEL: @"dfs$AtomicStore"
; CHECK-NOT: @__dfsan_arg_tls
; CHECK-NOT: @__dfsan_arg_origin_tls
; CHECK_ORIGIN-NOT: 35184372088832
; CHECK: [[INTP:%.*]] = ptrtoint i32* %p to i64
; CHECK: [[OFFSET:%.*]] = and i64 [[INTP]], -123145302310913
; CHECK: [[SHADOW_ADDR:%.*]] = mul i64 [[OFFSET]], 2
; CHECK: [[SHADOW_PTR:%.*]] = inttoptr i64 [[SHADOW_ADDR]] to i16*
; CHECK: [[SHADOW_PTR64:%.*]] = bitcast i16* [[SHADOW_PTR]] to i64*
; CHECK: store i64 0, i64* [[SHADOW_PTR64]], align 2
; CHECK: store atomic i32 %x, i32* %p seq_cst, align 16
; CHECK: ret void


; atomic store: store clean shadow value before app value

define void @AtomicStoreRelease(i32* %p, i32 %x) {
entry:
  store atomic i32 %x, i32* %p release, align 16
  ret void
}

; CHECK-LABEL: @"dfs$AtomicStoreRelease"
; CHECK-NOT: @__dfsan_arg_tls
; CHECK-NOT: @__dfsan_arg_origin_tls
; CHECK_ORIGIN-NOT: 35184372088832
; CHECK: [[INTP:%.*]] = ptrtoint i32* %p to i64
; CHECK: [[OFFSET:%.*]] = and i64 [[INTP]], -123145302310913
; CHECK: [[SHADOW_ADDR:%.*]] = mul i64 [[OFFSET]], 2
; CHECK: [[SHADOW_PTR:%.*]] = inttoptr i64 [[SHADOW_ADDR]] to i16*
; CHECK: [[SHADOW_PTR64:%.*]] = bitcast i16* [[SHADOW_PTR]] to i64*
; CHECK: store i64 0, i64* [[SHADOW_PTR64]], align 2
; CHECK: store atomic i32 %x, i32* %p release, align 16
; CHECK: ret void


; atomic store monotonic: bumped up to store release

define void @AtomicStoreMonotonic(i32* %p, i32 %x) {
entry:
  store atomic i32 %x, i32* %p monotonic, align 16
  ret void
}

; CHECK-LABEL: @"dfs$AtomicStoreMonotonic"
; CHECK-NOT: @__dfsan_arg_tls
; CHECK-NOT: @__dfsan_arg_origin_tls
; CHECK_ORIGIN-NOT: 35184372088832
; CHECK: [[INTP:%.*]] = ptrtoint i32* %p to i64
; CHECK: [[OFFSET:%.*]] = and i64 [[INTP]], -123145302310913
; CHECK: [[SHADOW_ADDR:%.*]] = mul i64 [[OFFSET]], 2
; CHECK: [[SHADOW_PTR:%.*]] = inttoptr i64 [[SHADOW_ADDR]] to i16*
; CHECK: [[SHADOW_PTR64:%.*]] = bitcast i16* [[SHADOW_PTR]] to i64*
; CHECK: store i64 0, i64* [[SHADOW_PTR64]], align 2
; CHECK: store atomic i32 %x, i32* %p release, align 16
; CHECK: ret void


; atomic store unordered: bumped up to store release

define void @AtomicStoreUnordered(i32* %p, i32 %x) {
entry:
  store atomic i32 %x, i32* %p unordered, align 16
  ret void
}

; CHECK-LABEL: @"dfs$AtomicStoreUnordered"
; CHECK-NOT: @__dfsan_arg_tls
; CHECK-NOT: @__dfsan_arg_origin_tls
; CHECK_ORIGIN-NOT: 35184372088832
; CHECK: [[INTP:%.*]] = ptrtoint i32* %p to i64
; CHECK: [[OFFSET:%.*]] = and i64 [[INTP]], -123145302310913
; CHECK: [[SHADOW_ADDR:%.*]] = mul i64 [[OFFSET]], 2
; CHECK: [[SHADOW_PTR:%.*]] = inttoptr i64 [[SHADOW_ADDR]] to i16*
; CHECK: [[SHADOW_PTR64:%.*]] = bitcast i16* [[SHADOW_PTR]] to i64*
; CHECK: store i64 0, i64* [[SHADOW_PTR64]], align 2
; CHECK: store atomic i32 %x, i32* %p release, align 16
; CHECK: ret void
