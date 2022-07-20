; RUN: opt -passes=instrorderfile -S < %s | FileCheck %s

target triple = "x86_64-apple-macosx10.10.0"

; CHECK: @_llvm_order_file_buffer ={{.*}}global [131072 x i64] zeroinitializer
; CHECK: @_llvm_order_file_buffer_idx = linkonce_odr global i32 0
; CHECK: @bitmap_0 ={{.*}}global [1 x i8] zeroinitializer

define i32 @_Z1fv() {
  ret i32 0
}
; CHECK-LABEL: define i32 @_Z1fv
; CHECK: order_file_entry
; CHECK: %[[T1:.+]] = load i8, i8* getelementptr inbounds ([1 x i8], [1 x i8]* @bitmap_0, i32 0, i32 0
; CHECK: store i8 1, i8* getelementptr inbounds ([1 x i8], [1 x i8]* @bitmap_0, i32 0, i32 0)
; CHECK: %[[T2:.+]] = icmp eq i8 %[[T1]], 0
; CHECK: br i1 %[[T2]], label %order_file_set, label

; CHECK: order_file_set
; CHECK: %[[T3:.+]] = atomicrmw add i32* @_llvm_order_file_buffer_idx, i32 1 seq_cst
; CHECK: %[[T5:.+]] = and i32 %[[T3]], 131071
; CHECK: %[[T4:.+]] = getelementptr [131072 x i64], [131072 x i64]* @_llvm_order_file_buffer, i32 0, i32 %[[T5]]
; CHECK: store i64 {{.*}}, i64* %[[T4]]
