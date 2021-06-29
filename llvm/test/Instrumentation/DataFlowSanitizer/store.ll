; RUN: opt < %s -dfsan -dfsan-combine-pointer-labels-on-store=1 -S | FileCheck %s --check-prefixes=CHECK,COMBINE_PTR_LABEL
; RUN: opt < %s -dfsan -dfsan-combine-pointer-labels-on-store=0 -S | FileCheck %s --check-prefixes=CHECK,NO_COMBINE_PTR_LABEL
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]
; CHECK: @__dfsan_shadow_width_bytes = weak_odr constant i32 [[#SBYTES:]]

define void @store0({} %v, {}* %p) {
  ; CHECK-LABEL: @store0.dfsan
  ; CHECK:       store {} %v, {}* %p
  ; CHECK-NOT:   store
  ; CHECK:       ret void

  store {} %v, {}* %p
  ret void
}

define void @store8(i8 %v, i8* %p) {
  ; CHECK-LABEL:       @store8.dfsan
  ; CHECK:             load i[[#SBITS]], i[[#SBITS]]* {{.*}} @__dfsan_arg_tls
  ; COMBINE_PTR_LABEL: load i[[#SBITS]], i[[#SBITS]]* {{.*}} @__dfsan_arg_tls
  ; COMBINE_PTR_LABEL: or i[[#SBITS]]
  ; CHECK:             ptrtoint i8* {{.*}} i64
  ; CHECK-NEXT:        xor i64
  ; CHECK-NEXT:        inttoptr i64 {{.*}} i[[#SBITS]]*
  ; CHECK-NEXT:        getelementptr i[[#SBITS]], i[[#SBITS]]*
  ; CHECK-NEXT:        store i[[#SBITS]]
  ; CHECK-NEXT:        store i8 %v, i8* %p
  ; CHECK-NEXT:        ret void

  store i8 %v, i8* %p
  ret void
}

define void @store16(i16 %v, i16* %p) {
  ; CHECK-LABEL:       @store16.dfsan
  ; CHECK:             load i[[#SBITS]], i[[#SBITS]]* {{.*}} @__dfsan_arg_tls
  ; COMBINE_PTR_LABEL: load i[[#SBITS]], i[[#SBITS]]* {{.*}} @__dfsan_arg_tls
  ; COMBINE_PTR_LABEL: or i[[#SBITS]]
  ; CHECK:             ptrtoint i16* {{.*}} i64
  ; CHECK-NEXT:        xor i64
  ; CHECK-NEXT:        inttoptr i64 {{.*}} i[[#SBITS]]*
  ; CHECK-NEXT:        getelementptr i[[#SBITS]], i[[#SBITS]]*
  ; CHECK-NEXT:        store i[[#SBITS]]
  ; CHECK-NEXT:        getelementptr i[[#SBITS]], i[[#SBITS]]*
  ; CHECK-NEXT:        store i[[#SBITS]]
  ; CHECK-NEXT:        store i16 %v, i16* %p
  ; CHECK-NEXT:        ret void

  store i16 %v, i16* %p
  ret void
}

define void @store32(i32 %v, i32* %p) {
  ; CHECK-LABEL:       @store32.dfsan
  ; CHECK:             load i[[#SBITS]], i[[#SBITS]]* {{.*}} @__dfsan_arg_tls
  ; COMBINE_PTR_LABEL: load i[[#SBITS]], i[[#SBITS]]* {{.*}} @__dfsan_arg_tls
  ; COMBINE_PTR_LABEL: or i[[#SBITS]]
  ; CHECK:             ptrtoint i32* {{.*}} i64
  ; CHECK-NEXT:        xor i64
  ; CHECK-NEXT:        inttoptr i64 {{.*}} i[[#SBITS]]*
  ; CHECK-NEXT:        getelementptr i[[#SBITS]], i[[#SBITS]]*
  ; CHECK-NEXT:        store i[[#SBITS]]
  ; CHECK-NEXT:        getelementptr i[[#SBITS]], i[[#SBITS]]*
  ; CHECK-NEXT:        store i[[#SBITS]]
  ; CHECK-NEXT:        getelementptr i[[#SBITS]], i[[#SBITS]]*
  ; CHECK-NEXT:        store i[[#SBITS]]
  ; CHECK-NEXT:        getelementptr i[[#SBITS]], i[[#SBITS]]*
  ; CHECK-NEXT:        store i[[#SBITS]]
  ; CHECK-NEXT:        store i32 %v, i32* %p
  ; CHECK-NEXT:        ret void

  store i32 %v, i32* %p
  ret void
}

define void @store64(i64 %v, i64* %p) {
  ; CHECK-LABEL:       @store64.dfsan
  ; CHECK:             load i[[#SBITS]], i[[#SBITS]]* {{.*}} @__dfsan_arg_tls
  ; COMBINE_PTR_LABEL: load i[[#SBITS]], i[[#SBITS]]* {{.*}} @__dfsan_arg_tls
  ; COMBINE_PTR_LABEL: or i[[#SBITS]]
  ; CHECK:             ptrtoint i64* {{.*}} i64
  ; CHECK-NEXT:        xor i64
  ; CHECK-NEXT:        inttoptr i64 {{.*}} i[[#SBITS]]*
  ; CHECK-COUNT-8:     insertelement {{.*}} i[[#SBITS]]
  ; CHECK-NEXT:        bitcast i[[#SBITS]]* {{.*}} <8 x i[[#SBITS]]>*
  ; CHECK-NEXT:        getelementptr <8 x i[[#SBITS]]>
  ; CHECK-NEXT:        store <8 x i[[#SBITS]]>
  ; CHECK-NEXT:        store i64 %v, i64* %p
  ; CHECK-NEXT:        ret void

  store i64 %v, i64* %p
  ret void
}

define void @store_zero(i32* %p) {
  ; CHECK-LABEL:          @store_zero.dfsan
  ; NO_COMBINE_PTR_LABEL: bitcast i[[#SBITS]]* {{.*}} to i[[#mul(4, SBITS)]]*
  ; NO_COMBINE_PTR_LABEL: store i[[#mul(4, SBITS)]] 0, i[[#mul(4, SBITS)]]* {{.*}}
  store i32 0, i32* %p
  ret void
}
