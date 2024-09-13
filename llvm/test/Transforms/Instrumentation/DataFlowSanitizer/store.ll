; RUN: opt < %s -passes=dfsan -dfsan-combine-pointer-labels-on-store=1 -S | FileCheck %s --check-prefixes=CHECK,COMBINE_PTR_LABEL
; RUN: opt < %s -passes=dfsan -dfsan-combine-pointer-labels-on-store=0 -S | FileCheck %s --check-prefixes=CHECK,NO_COMBINE_PTR_LABEL
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @store0({} %v, ptr %p) {
  ; CHECK-LABEL: @store0.dfsan
  ; CHECK:       store {} %v, ptr %p
  ; CHECK-NOT:   store
  ; CHECK:       ret void

  store {} %v, ptr %p
  ret void
}

define void @store8(i8 %v, ptr %p) {
  ; CHECK-LABEL:       @store8.dfsan
  ; NO_COMBINE_PTR_LABEL: load i8, ptr @__dfsan_arg_tls
  ; COMBINE_PTR_LABEL:    load i8, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align 2

  ; COMBINE_PTR_LABEL: load i8, ptr @__dfsan_arg_tls
  ; COMBINE_PTR_LABEL: or i8
  ; CHECK:             ptrtoint ptr {{.*}} i64
  ; CHECK-NEXT:        xor i64
  ; CHECK-NEXT:        inttoptr i64 {{.*}} ptr
  ; CHECK-NEXT:        getelementptr i8, ptr
  ; CHECK-NEXT:        store i8
  ; CHECK-NEXT:        store i8 %v, ptr %p
  ; CHECK-NEXT:        ret void

  store i8 %v, ptr %p
  ret void
}

define void @store16(i16 %v, ptr %p) {
  ; CHECK-LABEL:       @store16.dfsan
  ; NO_COMBINE_PTR_LABEL: load i8, ptr @__dfsan_arg_tls
  ; COMBINE_PTR_LABEL:    load i8, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align 2
  ; COMBINE_PTR_LABEL: load i8, ptr @__dfsan_arg_tls
  ; COMBINE_PTR_LABEL: or i8
  ; CHECK:             ptrtoint ptr {{.*}} i64
  ; CHECK-NEXT:        xor i64
  ; CHECK-NEXT:        inttoptr i64 {{.*}} ptr
  ; CHECK-NEXT:        getelementptr i8, ptr
  ; CHECK-NEXT:        store i8
  ; CHECK-NEXT:        getelementptr i8, ptr
  ; CHECK-NEXT:        store i8
  ; CHECK-NEXT:        store i16 %v, ptr %p
  ; CHECK-NEXT:        ret void

  store i16 %v, ptr %p
  ret void
}

define void @store32(i32 %v, ptr %p) {
  ; CHECK-LABEL:       @store32.dfsan
  ; NO_COMBINE_PTR_LABEL: load i8, ptr @__dfsan_arg_tls
  ; COMBINE_PTR_LABEL:    load i8, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align 2
  ; COMBINE_PTR_LABEL: load i8, ptr @__dfsan_arg_tls
  ; COMBINE_PTR_LABEL: or i8
  ; CHECK:             ptrtoint ptr {{.*}} i64
  ; CHECK-NEXT:        xor i64
  ; CHECK-NEXT:        inttoptr i64 {{.*}} ptr
  ; CHECK-NEXT:        getelementptr i8, ptr
  ; CHECK-NEXT:        store i8
  ; CHECK-NEXT:        getelementptr i8, ptr
  ; CHECK-NEXT:        store i8
  ; CHECK-NEXT:        getelementptr i8, ptr
  ; CHECK-NEXT:        store i8
  ; CHECK-NEXT:        getelementptr i8, ptr
  ; CHECK-NEXT:        store i8
  ; CHECK-NEXT:        store i32 %v, ptr %p
  ; CHECK-NEXT:        ret void

  store i32 %v, ptr %p
  ret void
}

define void @store64(i64 %v, ptr %p) {
  ; CHECK-LABEL:       @store64.dfsan
  ; NO_COMBINE_PTR_LABEL: load i8, ptr @__dfsan_arg_tls
  ; COMBINE_PTR_LABEL:    load i8, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align 2
  ; COMBINE_PTR_LABEL: load i8, ptr @__dfsan_arg_tls
  ; COMBINE_PTR_LABEL: or i8
  ; CHECK:             ptrtoint ptr {{.*}} i64
  ; CHECK-NEXT:        xor i64
  ; CHECK-NEXT:        inttoptr i64 {{.*}} ptr
  ; CHECK-COUNT-8:     insertelement {{.*}} i8
  ; CHECK-NEXT:        getelementptr <8 x i8>
  ; CHECK-NEXT:        store <8 x i8>
  ; CHECK-NEXT:        store i64 %v, ptr %p
  ; CHECK-NEXT:        ret void

  store i64 %v, ptr %p
  ret void
}

define void @store_zero(ptr %p) {
  ; CHECK-LABEL:          @store_zero.dfsan
  ; NO_COMBINE_PTR_LABEL: store i32 0, ptr {{.*}}
  store i32 0, ptr %p
  ret void
}
