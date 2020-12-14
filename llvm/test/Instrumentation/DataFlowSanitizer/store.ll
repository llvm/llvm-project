; RUN: opt < %s -dfsan -dfsan-combine-pointer-labels-on-store=1 -S | FileCheck %s --check-prefix=COMBINE_PTR_LABEL
; RUN: opt < %s -dfsan -dfsan-combine-pointer-labels-on-store=0 -S | FileCheck %s --check-prefix=NO_COMBINE_PTR_LABEL
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @store0({} %v, {}* %p) {
  ; COMBINE_PTR_LABEL: @"dfs$store0"
  ; COMBINE_PTR_LABEL: store
  ; COMBINE_PTR_LABEL-NOT: store

  ; NO_COMBINE_PTR_LABEL: @"dfs$store0"
  ; NO_COMBINE_PTR_LABEL: store
  ; NO_COMBINE_PTR_LABEL-NOT: store

  store {} %v, {}* %p
  ret void
}

define void @store8(i8 %v, i8* %p) {
  ; NO_COMBINE_PTR_LABEL: @"dfs$store8"
  ; NO_COMBINE_PTR_LABEL: load i16, i16* {{.*}} @__dfsan_arg_tls
  ; NO_COMBINE_PTR_LABEL: ptrtoint i8* {{.*}} i64
  ; NO_COMBINE_PTR_LABEL: and i64
  ; NO_COMBINE_PTR_LABEL: mul i64
  ; NO_COMBINE_PTR_LABEL: inttoptr i64 {{.*}} i16*
  ; NO_COMBINE_PTR_LABEL: getelementptr i16, i16*
  ; NO_COMBINE_PTR_LABEL: store i16
  ; NO_COMBINE_PTR_LABEL: store i8

  ; COMBINE_PTR_LABEL: @"dfs$store8"
  ; COMBINE_PTR_LABEL: load i16, i16*
  ; COMBINE_PTR_LABEL: load i16, i16*
  ; COMBINE_PTR_LABEL: icmp ne i16
  ; COMBINE_PTR_LABEL: call {{.*}} @__dfsan_union
  ; COMBINE_PTR_LABEL: ptrtoint i8* {{.*}} i64
  ; COMBINE_PTR_LABEL: and i64
  ; COMBINE_PTR_LABEL: mul i64
  ; COMBINE_PTR_LABEL: inttoptr i64 {{.*}} i16*
  ; COMBINE_PTR_LABEL: getelementptr i16, i16*
  ; COMBINE_PTR_LABEL: store i16
  ; COMBINE_PTR_LABEL: store i8

  store i8 %v, i8* %p
  ret void
}

define void @store16(i16 %v, i16* %p) {
  ; NO_COMBINE_PTR_LABEL: @"dfs$store16"
  ; NO_COMBINE_PTR_LABEL: load i16, i16* {{.*}} @__dfsan_arg_tls
  ; NO_COMBINE_PTR_LABEL: ptrtoint i16* {{.*}} i64
  ; NO_COMBINE_PTR_LABEL: and i64
  ; NO_COMBINE_PTR_LABEL: mul i64
  ; NO_COMBINE_PTR_LABEL: inttoptr i64 {{.*}} i16*
  ; NO_COMBINE_PTR_LABEL: getelementptr i16, i16*
  ; NO_COMBINE_PTR_LABEL: store i16
  ; NO_COMBINE_PTR_LABEL: getelementptr i16, i16*
  ; NO_COMBINE_PTR_LABEL: store i16
  ; NO_COMBINE_PTR_LABEL: store i16

  ; COMBINE_PTR_LABEL: @"dfs$store16"
  ; COMBINE_PTR_LABEL: load i16, i16* {{.*}} @__dfsan_arg_tls
  ; COMBINE_PTR_LABEL: load i16, i16* {{.*}} @__dfsan_arg_tls
  ; COMBINE_PTR_LABEL: icmp ne i16
  ; COMBINE_PTR_LABEL: call {{.*}} @__dfsan_union
  ; COMBINE_PTR_LABEL: ptrtoint i16* {{.*}} i64
  ; COMBINE_PTR_LABEL: and i64
  ; COMBINE_PTR_LABEL: mul i64
  ; COMBINE_PTR_LABEL: inttoptr i64 {{.*}} i16*
  ; COMBINE_PTR_LABEL: getelementptr i16, i16*
  ; COMBINE_PTR_LABEL: store i16
  ; COMBINE_PTR_LABEL: getelementptr i16, i16*
  ; COMBINE_PTR_LABEL: store i16
  ; COMBINE_PTR_LABEL: store i16

  store i16 %v, i16* %p
  ret void
}

define void @store32(i32 %v, i32* %p) {
  ; NO_COMBINE_PTR_LABEL: @"dfs$store32"
  ; NO_COMBINE_PTR_LABEL: load i16, i16* {{.*}} @__dfsan_arg_tls
  ; NO_COMBINE_PTR_LABEL: ptrtoint i32* {{.*}} i64
  ; NO_COMBINE_PTR_LABEL: and i64
  ; NO_COMBINE_PTR_LABEL: mul i64
  ; NO_COMBINE_PTR_LABEL: inttoptr i64 {{.*}} i16*
  ; NO_COMBINE_PTR_LABEL: getelementptr i16, i16*
  ; NO_COMBINE_PTR_LABEL: store i16
  ; NO_COMBINE_PTR_LABEL: getelementptr i16, i16*
  ; NO_COMBINE_PTR_LABEL: store i16
  ; NO_COMBINE_PTR_LABEL: getelementptr i16, i16*
  ; NO_COMBINE_PTR_LABEL: store i16
  ; NO_COMBINE_PTR_LABEL: getelementptr i16, i16*
  ; NO_COMBINE_PTR_LABEL: store i16
  ; NO_COMBINE_PTR_LABEL: store i32

  ; COMBINE_PTR_LABEL: @"dfs$store32"
  ; COMBINE_PTR_LABEL: load i16, i16* {{.*}} @__dfsan_arg_tls
  ; COMBINE_PTR_LABEL: load i16, i16* {{.*}} @__dfsan_arg_tls
  ; COMBINE_PTR_LABEL: icmp ne i16
  ; COMBINE_PTR_LABEL: call {{.*}} @__dfsan_union
  ; COMBINE_PTR_LABEL: ptrtoint i32* {{.*}} i64
  ; COMBINE_PTR_LABEL: and i64
  ; COMBINE_PTR_LABEL: mul i64
  ; COMBINE_PTR_LABEL: inttoptr i64 {{.*}} i16*
  ; COMBINE_PTR_LABEL: getelementptr i16, i16*
  ; COMBINE_PTR_LABEL: store i16
  ; COMBINE_PTR_LABEL: getelementptr i16, i16*
  ; COMBINE_PTR_LABEL: store i16
  ; COMBINE_PTR_LABEL: getelementptr i16, i16*
  ; COMBINE_PTR_LABEL: store i16
  ; COMBINE_PTR_LABEL: getelementptr i16, i16*
  ; COMBINE_PTR_LABEL: store i16
  ; COMBINE_PTR_LABEL: store i32

  store i32 %v, i32* %p
  ret void
}

define void @store64(i64 %v, i64* %p) {
  ; NO_COMBINE_PTR_LABEL: @"dfs$store64"
  ; NO_COMBINE_PTR_LABEL: load i16, i16* {{.*}} @__dfsan_arg_tls
  ; NO_COMBINE_PTR_LABEL: ptrtoint i64* {{.*}} i64
  ; NO_COMBINE_PTR_LABEL: and i64
  ; NO_COMBINE_PTR_LABEL: mul i64
  ; NO_COMBINE_PTR_LABEL: inttoptr i64 {{.*}} i16*
  ; NO_COMBINE_PTR_LABEL: insertelement {{.*}} i16
  ; NO_COMBINE_PTR_LABEL: insertelement {{.*}} i16
  ; NO_COMBINE_PTR_LABEL: insertelement {{.*}} i16
  ; NO_COMBINE_PTR_LABEL: insertelement {{.*}} i16
  ; NO_COMBINE_PTR_LABEL: insertelement {{.*}} i16
  ; NO_COMBINE_PTR_LABEL: insertelement {{.*}} i16
  ; NO_COMBINE_PTR_LABEL: insertelement {{.*}} i16
  ; NO_COMBINE_PTR_LABEL: insertelement {{.*}} i16
  ; NO_COMBINE_PTR_LABEL: bitcast i16* {{.*}} <8 x i16>*
  ; NO_COMBINE_PTR_LABEL: store i64

  ; COMBINE_PTR_LABEL: @"dfs$store64"
  ; COMBINE_PTR_LABEL: load i16, i16* {{.*}} @__dfsan_arg_tls
  ; COMBINE_PTR_LABEL: load i16, i16* {{.*}} @__dfsan_arg_tls
  ; COMBINE_PTR_LABEL: icmp ne i16
  ; COMBINE_PTR_LABEL: call {{.*}} @__dfsan_union
  ; COMBINE_PTR_LABEL: ptrtoint i64* {{.*}} i64
  ; COMBINE_PTR_LABEL: and i64
  ; COMBINE_PTR_LABEL: mul i64
  ; COMBINE_PTR_LABEL: inttoptr i64 {{.*}} i16*
  ; COMBINE_PTR_LABEL: insertelement {{.*}} i16
  ; COMBINE_PTR_LABEL: insertelement {{.*}} i16
  ; COMBINE_PTR_LABEL: insertelement {{.*}} i16
  ; COMBINE_PTR_LABEL: insertelement {{.*}} i16
  ; COMBINE_PTR_LABEL: insertelement {{.*}} i16
  ; COMBINE_PTR_LABEL: insertelement {{.*}} i16
  ; COMBINE_PTR_LABEL: insertelement {{.*}} i16
  ; COMBINE_PTR_LABEL: insertelement {{.*}} i16
  ; COMBINE_PTR_LABEL: bitcast i16* {{.*}} <8 x i16>*
  ; COMBINE_PTR_LABEL: store <8 x i16>
  ; COMBINE_PTR_LABEL: store i64

  store i64 %v, i64* %p
  ret void
}

define void @store_zero(i32* %p) {
  ;  NO_COMBINE_PTR_LABEL: store i64 0, i64* {{.*}}, align 2
  store i32 0, i32* %p
  ret void
}