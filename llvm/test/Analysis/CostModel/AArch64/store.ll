; RUN: opt < %s  -cost-model -analyze -mtriple=aarch64-apple-ios | FileCheck %s
; RUN: opt < %s  -cost-model -analyze -mtriple=aarch64-apple-ios -mattr=slow-misaligned-128store | FileCheck %s --check-prefix=SLOW_MISALIGNED_128_STORE

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
; CHECK-LABEL: getMemoryOpCost
; SLOW_MISALIGNED_128_STORE-LABEL: getMemoryOpCost
define void @getMemoryOpCost() {
    ; If FeatureSlowMisaligned128Store is set, we penalize <2 x i64> stores. On
    ; Cyclone, for example, such stores should be expensive because we don't
    ; split them and misaligned 16b stores have bad performance.
    ;
    ; CHECK: cost of 1 {{.*}} store
    ; SLOW_MISALIGNED_128_STORE: cost of 12 {{.*}} store
    store <2 x i64> undef, <2 x i64> * undef

    ; We scalarize the loads/stores because there is no vector register name for
    ; these types (they get extended to v.4h/v.2s).
    ; CHECK: cost of 16 {{.*}} store
    store <2 x i8> undef, <2 x i8> * undef
    ; CHECK: cost of 64 {{.*}} store
    store <4 x i8> undef, <4 x i8> * undef
    ; CHECK: cost of 16 {{.*}} load
    load <2 x i8> , <2 x i8> * undef
    ; CHECK: cost of 64 {{.*}} load
    load <4 x i8> , <4 x i8> * undef

    ret void
}
