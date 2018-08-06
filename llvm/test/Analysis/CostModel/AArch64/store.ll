; RUN: opt < %s -cost-model -analyze -mtriple=aarch64-unknown | FileCheck %s
; RUN: opt < %s -cost-model -analyze -mtriple=aarch64-unknown -mattr=slow-misaligned-128store | FileCheck %s --check-prefix=SLOW_MISALIGNED_128_STORE

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
; CHECK-LABEL: getMemoryOpCost
; SLOW_MISALIGNED_128_STORE-LABEL: getMemoryOpCost
define void @getMemoryOpCost() {
    ; If FeatureSlowMisaligned128Store is set, we penalize 128-bit stores.
    ; The unlegalized 256-bit stores are further penalized when legalized down
    ; to 128-bit stores. 

    ; CHECK: cost of 2 for {{.*}} store <4 x i64>
    ; SLOW_MISALIGNED_128_STORE: cost of 24 for {{.*}} store <4 x i64>
    store <4 x i64> undef, <4 x i64> * undef
    ; CHECK-NEXT: cost of 2 for {{.*}} store <8 x i32>
    ; SLOW_MISALIGNED_128_STORE-NEXT: cost of 24 for {{.*}} store <8 x i32>
    store <8 x i32> undef, <8 x i32> * undef
    ; CHECK-NEXT: cost of 2 for {{.*}} store <16 x i16>
    ; SLOW_MISALIGNED_128_STORE-NEXT: cost of 24 for {{.*}} store <16 x i16>
    store <16 x i16> undef, <16 x i16> * undef
    ; CHECK-NEXT: cost of 2 for {{.*}} store <32 x i8>
    ; SLOW_MISALIGNED_128_STORE-NEXT: cost of 24 for {{.*}} store <32 x i8>
    store <32 x i8> undef, <32 x i8> * undef

    ; CHECK-NEXT: cost of 2 for {{.*}} store <4 x double>
    ; SLOW_MISALIGNED_128_STORE-NEXT: cost of 24 for {{.*}} store <4 x double>
    store <4 x double> undef, <4 x double> * undef
    ; CHECK-NEXT: cost of 2 for {{.*}} store <8 x float>
    ; SLOW_MISALIGNED_128_STORE-NEXT: cost of 24 for {{.*}} store <8 x float>
    store <8 x float> undef, <8 x float> * undef
    ; CHECK-NEXT: cost of 2 for {{.*}} store <16 x half>
    ; SLOW_MISALIGNED_128_STORE-NEXT: cost of 24 for {{.*}} store <16 x half>
    store <16 x half> undef, <16 x half> * undef

    ; CHECK-NEXT: cost of 1 for {{.*}} store <2 x i64>
    ; SLOW_MISALIGNED_128_STORE-NEXT: cost of 12 for {{.*}} store <2 x i64>
    store <2 x i64> undef, <2 x i64> * undef
    ; CHECK-NEXT: cost of 1 for {{.*}} store <4 x i32>
    ; SLOW_MISALIGNED_128_STORE-NEXT: cost of 12 for {{.*}} store <4 x i32>
    store <4 x i32> undef, <4 x i32> * undef
    ; CHECK-NEXT: cost of 1 for {{.*}} store <8 x i16>
    ; SLOW_MISALIGNED_128_STORE-NEXT: cost of 12 for {{.*}} store <8 x i16>
    store <8 x i16> undef, <8 x i16> * undef
    ; CHECK-NEXT: cost of 1 for {{.*}} store <16 x i8>
    ; SLOW_MISALIGNED_128_STORE-NEXT: cost of 12 for {{.*}} store <16 x i8>
    store <16 x i8> undef, <16 x i8> * undef

    ; CHECK-NEXT: cost of 1 for {{.*}} store <2 x double>
    ; SLOW_MISALIGNED_128_STORE-NEXT: cost of 12 for {{.*}} store <2 x double>
    store <2 x double> undef, <2 x double> * undef
    ; CHECK-NEXT: cost of 1 for {{.*}} store <4 x float>
    ; SLOW_MISALIGNED_128_STORE-NEXT: cost of 12 for {{.*}} store <4 x float>
    store <4 x float> undef, <4 x float> * undef
    ; CHECK-NEXT: cost of 1 for {{.*}} store <8 x half>
    ; SLOW_MISALIGNED_128_STORE-NEXT: cost of 12 for {{.*}} store <8 x half>
    store <8 x half> undef, <8 x half> * undef

    ; We scalarize the loads/stores because there is no vector register name for
    ; these types (they get extended to v.4h/v.2s).
    ; CHECK: cost of 16 {{.*}} store
    store <2 x i8> undef, <2 x i8> * undef
    ; CHECK: cost of 1 {{.*}} store
    store <4 x i8> undef, <4 x i8> * undef
    ; CHECK: cost of 16 {{.*}} load
    load <2 x i8> , <2 x i8> * undef
    ; CHECK: cost of 64 {{.*}} load
    load <4 x i8> , <4 x i8> * undef

    ret void
}
