; Test basic memory profiler instrumentation with histograms.
;
; RUN: opt < %s -passes='function(memprof),memprof-module' -memprof-histogram -S | FileCheck --check-prefixes=CHECK,CHECK-S3 %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @llvm.used = appending global [1 x ptr] [ptr @memprof.module_ctor]
; CHECK: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @memprof.module_ctor, ptr null }]

define i32 @test_load(ptr %a) {
entry:
  %tmp1 = load i32, ptr %a, align 4
  ret i32 %tmp1
}
; CHECK-LABEL: @test_load
; CHECK:         %[[SHADOW_OFFSET:[^ ]*]] = load i64, ptr @__memprof_shadow_memory_dynamic_address
; CHECK-NEXT:    %[[LOAD_ADDR:[^ ]*]] = ptrtoint ptr %a to i64
; CHECK-NEXT:    %[[MASKED_ADDR:[^ ]*]] = and i64 %[[LOAD_ADDR]], -8
; CHECK-S3-NEXT: %[[SHIFTED_ADDR:[^ ]*]] = lshr i64 %[[MASKED_ADDR]], 3
; CHECK-NEXT:    add i64 %[[SHIFTED_ADDR]], %[[SHADOW_OFFSET]]
; CHECK-NEXT:    %[[LOAD_SHADOW_PTR:[^ ]*]] = inttoptr
; CHECK-NEXT:    %[[LOAD_SHADOW:[^ ]*]] = load i8, ptr %[[LOAD_SHADOW_PTR]]
; CHECK-NEXT:    %[[ICMP_MAX_COUNT:[^ ]*]] = icmp ult i8 %[[LOAD_SHADOW]], -1
; CHECK-NEXT:    br i1 %[[ICMP_MAX_COUNT]], label %[[INC_LABEL:[^ ]*]], label %[[ELSE_LABEL:[^ ]*]]
; CHECK:         [[INC_LABEL]]:
; CHECK-NEXT:    %[[NEW_SHADOW:[^ ]*]] = add i8 %[[LOAD_SHADOW]], 1
; CHECK-NEXT:    store i8 %[[NEW_SHADOW]], ptr %[[LOAD_SHADOW_PTR]]
; CHECK-NEXT:    br label %[[ELSE_LABEL]]
; The actual load.
; CHECK:         [[ELSE_LABEL]]:
; CHECK-NEXT:    %tmp1 = load i32, ptr %a
; CHECK-NEXT:    ret i32 %tmp1

define void @test_store(ptr %a) {
entry:
  store i32 42, ptr %a, align 4
  ret void
}
; CHECK-LABEL: @test_store
; CHECK:         %[[SHADOW_OFFSET:[^ ]*]] = load i64, ptr @__memprof_shadow_memory_dynamic_address
; CHECK-NEXT:    %[[LOAD_ADDR:[^ ]*]] = ptrtoint ptr %a to i64
; CHECK-NEXT:    %[[MASKED_ADDR:[^ ]*]] = and i64 %[[LOAD_ADDR]], -8
; CHECK-S3-NEXT: %[[SHIFTED_ADDR:[^ ]*]] = lshr i64 %[[MASKED_ADDR]], 3
; CHECK-NEXT:    add i64 %[[SHIFTED_ADDR]], %[[SHADOW_OFFSET]]
; CHECK-NEXT:    %[[STORE_SHADOW_PTR:[^ ]*]] = inttoptr
; CHECK-NEXT:    %[[STORE_SHADOW:[^ ]*]] = load i8, ptr %[[STORE_SHADOW_PTR]]
; CHECK-NEXT:    %[[ICMP_MAX_COUNT:[^ ]*]] = icmp ult i8 %[[STORE_SHADOW]], -1
; CHECK-NEXT:    br i1 %[[ICMP_MAX_COUNT]], label %[[INC_LABEL:[^ ]*]], label %[[ELSE_LABEL:[^ ]*]]
; CHECK:         [[INC_LABEL]]:
; CHECK-NEXT:    %[[NEW_SHADOW:[^ ]*]] = add i8 %[[STORE_SHADOW]], 1
; CHECK-NEXT:    store i8 %[[NEW_SHADOW]], ptr %[[STORE_SHADOW_PTR]]
; CHECK-NEXT:    br label %[[ELSE_LABEL]]
; The actual store.
; CHECK:         [[ELSE_LABEL]]:
; CHECK-NEXT:    store i32 42, ptr %a, align 4
; CHECK-NEXT:    ret void