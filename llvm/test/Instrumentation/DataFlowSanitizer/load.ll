; RUN: opt < %s -dfsan -dfsan-combine-pointer-labels-on-load=1 -S | FileCheck %s --check-prefix=COMBINE_PTR_LABEL
; RUN: opt < %s -dfsan -dfsan-combine-pointer-labels-on-load=0 -S | FileCheck %s --check-prefix=NO_COMBINE_PTR_LABEL
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define {} @load0({}* %p) {
  ; COMBINE_PTR_LABEL: @"dfs$load0"
  ; COMBINE_PTR_LABEL: load
  ; COMBINE_PTR_LABEL-NOT: load

  ; NO_COMBINE_PTR_LABEL: @"dfs$load0"
  ; NO_COMBINE_PTR_LABEL: load
  ; NO_COMBINE_PTR_LABEL-NOT: load
  %a = load {}, {}* %p
  ret {} %a
}

define i8 @load8(i8* %p) {
  ; COMBINE_PTR_LABEL: @"dfs$load8"
  ; COMBINE_PTR_LABEL: load i16, i16*
  ; COMBINE_PTR_LABEL: ptrtoint i8* {{.*}} to i64
  ; COMBINE_PTR_LABEL: and i64
  ; COMBINE_PTR_LABEL: mul i64
  ; COMBINE_PTR_LABEL: inttoptr i64
  ; COMBINE_PTR_LABEL: load i16, i16*
  ; COMBINE_PTR_LABEL: icmp ne i16
  ; COMBINE_PTR_LABEL: call zeroext i16 @__dfsan_union
  ; COMBINE_PTR_LABEL: load i8, i8*
  ; COMBINE_PTR_LABEL: store i16 {{.*}} @__dfsan_retval_tls
  ; COMBINE_PTR_LABEL: ret i8

  ; NO_COMBINE_PTR_LABEL: @"dfs$load8"
  ; NO_COMBINE_PTR_LABEL: ptrtoint i8*
  ; NO_COMBINE_PTR_LABEL: and i64
  ; NO_COMBINE_PTR_LABEL: mul i64
  ; NO_COMBINE_PTR_LABEL: inttoptr i64 {{.*}} to i16*
  ; NO_COMBINE_PTR_LABEL: load i16, i16*
  ; NO_COMBINE_PTR_LABEL: load i8, i8*
  ; NO_COMBINE_PTR_LABEL: store i16 {{.*}} @__dfsan_retval_tls
  ; NO_COMBINE_PTR_LABEL: ret i8

  %a = load i8, i8* %p
  ret i8 %a
}

define i16 @load16(i16* %p) {
  ; COMBINE_PTR_LABEL: @"dfs$load16"
  ; COMBINE_PTR_LABEL: ptrtoint i16*
  ; COMBINE_PTR_LABEL: and i64
  ; COMBINE_PTR_LABEL: mul i64
  ; COMBINE_PTR_LABEL: inttoptr i64 {{.*}} i16*
  ; COMBINE_PTR_LABEL: getelementptr i16
  ; COMBINE_PTR_LABEL: load i16, i16*
  ; COMBINE_PTR_LABEL: load i16, i16*
  ; COMBINE_PTR_LABEL: icmp ne
  ; COMBINE_PTR_LABEL: call {{.*}} @__dfsan_union
  ; COMBINE_PTR_LABEL: icmp ne i16
  ; COMBINE_PTR_LABEL: call {{.*}} @__dfsan_union
  ; COMBINE_PTR_LABEL: load i16, i16*
  ; COMBINE_PTR_LABEL: store {{.*}} @__dfsan_retval_tls
  ; COMBINE_PTR_LABEL: ret i16

  ; NO_COMBINE_PTR_LABEL: @"dfs$load16"
  ; NO_COMBINE_PTR_LABEL: ptrtoint i16*
  ; NO_COMBINE_PTR_LABEL: and i64
  ; NO_COMBINE_PTR_LABEL: mul i64
  ; NO_COMBINE_PTR_LABEL: inttoptr i64 {{.*}} i16*
  ; NO_COMBINE_PTR_LABEL: getelementptr i16, i16*
  ; NO_COMBINE_PTR_LABEL: load i16, i16*
  ; NO_COMBINE_PTR_LABEL: load i16, i16*
  ; NO_COMBINE_PTR_LABEL: icmp ne i16
  ; NO_COMBINE_PTR_LABEL: call {{.*}} @__dfsan_union
  ; NO_COMBINE_PTR_LABEL: load i16, i16*
  ; NO_COMBINE_PTR_LABEL: store i16 {{.*}} @__dfsan_retval_tls
  ; NO_COMBINE_PTR_LABEL: ret i16

  %a = load i16, i16* %p
  ret i16 %a
}

define i32 @load32(i32* %p) {
  ; COMBINE_PTR_LABEL: @"dfs$load32"
  ; COMBINE_PTR_LABEL: ptrtoint i32*
  ; COMBINE_PTR_LABEL: and i64
  ; COMBINE_PTR_LABEL: mul i64
  ; COMBINE_PTR_LABEL: inttoptr i64 {{.*}} i16*
  ; COMBINE_PTR_LABEL: bitcast i16* {{.*}} i64*
  ; COMBINE_PTR_LABEL: load i64, i64*
  ; COMBINE_PTR_LABEL: trunc i64 {{.*}} i16
  ; COMBINE_PTR_LABEL: shl i64
  ; COMBINE_PTR_LABEL: lshr i64
  ; COMBINE_PTR_LABEL: or i64
  ; COMBINE_PTR_LABEL: icmp eq i64
  ; COMBINE_PTR_LABEL: icmp ne i16
  ; COMBINE_PTR_LABEL: call {{.*}} @__dfsan_union
  ; COMBINE_PTR_LABEL: load i32, i32*
  ; COMBINE_PTR_LABEL: store i16 {{.*}} @__dfsan_retval_tls
  ; COMBINE_PTR_LABEL: ret i32
  ; COMBINE_PTR_LABEL: call {{.*}} @__dfsan_union_load

  ; NO_COMBINE_PTR_LABEL: @"dfs$load32"
  ; NO_COMBINE_PTR_LABEL: ptrtoint i32*
  ; NO_COMBINE_PTR_LABEL: and i64
  ; NO_COMBINE_PTR_LABEL: mul i64
  ; NO_COMBINE_PTR_LABEL: inttoptr i64 {{.*}} i16*
  ; NO_COMBINE_PTR_LABEL: bitcast i16* {{.*}} i64*
  ; NO_COMBINE_PTR_LABEL: load i64, i64*
  ; NO_COMBINE_PTR_LABEL: trunc i64 {{.*}} i16
  ; NO_COMBINE_PTR_LABEL: shl i64
  ; NO_COMBINE_PTR_LABEL: lshr i64
  ; NO_COMBINE_PTR_LABEL: or i64
  ; NO_COMBINE_PTR_LABEL: icmp eq i64
  ; NO_COMBINE_PTR_LABEL: load i32, i32*
  ; NO_COMBINE_PTR_LABEL: store i16 {{.*}} @__dfsan_retval_tls
  ; NO_COMBINE_PTR_LABEL: ret i32
  ; NO_COMBINE_PTR_LABEL: call {{.*}} @__dfsan_union_load
  

  %a = load i32, i32* %p
  ret i32 %a
}

define i64 @load64(i64* %p) {
  ; COMBINE_PTR_LABEL: @"dfs$load64"
  ; COMBINE_PTR_LABEL: ptrtoint i64*
  ; COMBINE_PTR_LABEL: and i64
  ; COMBINE_PTR_LABEL: mul i64
  ; COMBINE_PTR_LABEL: inttoptr i64 {{.*}} i16*
  ; COMBINE_PTR_LABEL: bitcast i16* {{.*}} i64*
  ; COMBINE_PTR_LABEL: load i64, i64*
  ; COMBINE_PTR_LABEL: trunc i64 {{.*}} i16
  ; COMBINE_PTR_LABEL: shl i64
  ; COMBINE_PTR_LABEL: lshr i64
  ; COMBINE_PTR_LABEL: or i64
  ; COMBINE_PTR_LABEL: icmp eq i64
  ; COMBINE_PTR_LABEL: icmp ne i16
  ; COMBINE_PTR_LABEL: call {{.*}} @__dfsan_union
  ; COMBINE_PTR_LABEL: load i64, i64*
  ; COMBINE_PTR_LABEL: store i16 {{.*}} @__dfsan_retval_tls
  ; COMBINE_PTR_LABEL: ret i64
  ; COMBINE_PTR_LABEL: call {{.*}} @__dfsan_union_load
  ; COMBINE_PTR_LABEL: getelementptr i64, i64* {{.*}} i64
  ; COMBINE_PTR_LABEL: load i64, i64*
  ; COMBINE_PTR_LABEL: icmp eq i64

  ; NO_COMBINE_PTR_LABEL: @"dfs$load64"
  ; NO_COMBINE_PTR_LABEL: ptrtoint i64*
  ; NO_COMBINE_PTR_LABEL: and i64
  ; NO_COMBINE_PTR_LABEL: mul i64
  ; NO_COMBINE_PTR_LABEL: inttoptr i64 {{.*}} i16*
  ; NO_COMBINE_PTR_LABEL: bitcast i16* {{.*}} i64*
  ; NO_COMBINE_PTR_LABEL: load i64, i64*
  ; NO_COMBINE_PTR_LABEL: trunc i64 {{.*}} i16
  ; NO_COMBINE_PTR_LABEL: shl i64
  ; NO_COMBINE_PTR_LABEL: lshr i64
  ; NO_COMBINE_PTR_LABEL: or i64
  ; NO_COMBINE_PTR_LABEL: icmp eq i64
  ; NO_COMBINE_PTR_LABEL: load i64, i64*
  ; NO_COMBINE_PTR_LABEL: store i16 {{.*}} @__dfsan_retval_tls
  ; NO_COMBINE_PTR_LABEL: ret i64
  ; NO_COMBINE_PTR_LABEL: call {{.*}} @__dfsan_union_load
  ; NO_COMBINE_PTR_LABEL: getelementptr i64, i64* {{.*}} i64
  ; NO_COMBINE_PTR_LABEL: load i64, i64*
  ; NO_COMBINE_PTR_LABEL: icmp eq i64

  %a = load i64, i64* %p
  ret i64 %a
}

@X = constant i1 1
define i1 @load_global() {
  ; NO_COMBINE_PTR_LABEL: @"dfs$load_global"
  ; NO_COMBINE_PTR_LABEL: store i16 0, i16* bitcast ([100 x i64]* @__dfsan_retval_tls to i16*), align 2

  %a = load i1, i1* @X
  ret i1 %a
}
