; Test basic memory profiler instrumentation.
;
; RUN: opt < %s -passes='function(memprof),memprof-module' -S | FileCheck --check-prefixes=CHECK,CHECK-S3 %s
; RUN: opt < %s -passes='function(memprof),memprof-module' -memprof-mapping-scale=5 -S | FileCheck --check-prefixes=CHECK,CHECK-S5 %s

; We need the requires since both memprof and memprof-module require reading module level metadata which is done once by the memprof-globals-md analysis
; RUN: opt < %s -passes='function(memprof),module(memprof-module)' -S | FileCheck --check-prefixes=CHECK,CHECK-S3 %s
; RUN: opt < %s -passes='function(memprof),module(memprof-module)' -memprof-mapping-scale=5 -S | FileCheck --check-prefixes=CHECK,CHECK-S5 %s

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
; CHECK-NEXT:    %[[MASKED_ADDR:[^ ]*]] = and i64 %[[LOAD_ADDR]], -64
; CHECK-S3-NEXT: %[[SHIFTED_ADDR:[^ ]*]] = lshr i64 %[[MASKED_ADDR]], 3
; CHECK-S5-NEXT: %[[SHIFTED_ADDR:[^ ]*]] = lshr i64 %[[MASKED_ADDR]], 5
; CHECK-NEXT:    add i64 %[[SHIFTED_ADDR]], %[[SHADOW_OFFSET]]
; CHECK-NEXT:    %[[LOAD_SHADOW_PTR:[^ ]*]] = inttoptr
; CHECK-NEXT:    %[[LOAD_SHADOW:[^ ]*]] = load i64, ptr %[[LOAD_SHADOW_PTR]]
; CHECK-NEXT:    %[[NEW_SHADOW:[^ ]*]] = add i64 %[[LOAD_SHADOW]], 1
; CHECK-NEXT:    store i64 %[[NEW_SHADOW]], ptr %[[LOAD_SHADOW_PTR]]
; The actual load.
; CHECK-NEXT:    %tmp1 = load i32, ptr %a
; CHECK-NEXT:    ret i32 %tmp1

define void @test_store(ptr %a) {
entry:
  store i32 42, ptr %a, align 4
  ret void
}
; CHECK-LABEL: @test_store
; CHECK:         %[[SHADOW_OFFSET:[^ ]*]] = load i64, ptr @__memprof_shadow_memory_dynamic_address
; CHECK-NEXT:    %[[STORE_ADDR:[^ ]*]] = ptrtoint ptr %a to i64
; CHECK-NEXT:    %[[MASKED_ADDR:[^ ]*]] = and i64 %[[STORE_ADDR]], -64
; CHECK-S3-NEXT: %[[SHIFTED_ADDR:[^ ]*]] = lshr i64 %[[MASKED_ADDR]], 3
; CHECK-S5-NEXT: %[[SHIFTED_ADDR:[^ ]*]] = lshr i64 %[[MASKED_ADDR]], 5
; CHECK-NEXT:    add i64 %[[SHIFTED_ADDR]], %[[SHADOW_OFFSET]]
; CHECK-NEXT:    %[[STORE_SHADOW_PTR:[^ ]*]] = inttoptr
; CHECK-NEXT:    %[[STORE_SHADOW:[^ ]*]] = load i64, ptr %[[STORE_SHADOW_PTR]]
; CHECK-NEXT:    %[[NEW_SHADOW:[^ ]*]] = add i64 %[[STORE_SHADOW]], 1
; CHECK-NEXT:    store i64 %[[NEW_SHADOW]], ptr %[[STORE_SHADOW_PTR]]
; The actual store.
; CHECK-NEXT:    store i32 42, ptr %a
; CHECK-NEXT:    ret void

define void @FP80Test(ptr nocapture %a) nounwind uwtable {
entry:
    store x86_fp80 0xK3FFF8000000000000000, ptr %a, align 16
    ret void
}
; CHECK-LABEL: @FP80Test
; Exactly one shadow update for store access.
; CHECK-NOT:  store i64
; CHECK:      %[[NEW_ST_SHADOW:[^ ]*]] = add i64 %{{.*}}, 1
; CHECK-NEXT: store i64 %[[NEW_ST_SHADOW]]
; CHECK-NOT:  store i64
; The actual store.
; CHECK:      store x86_fp80 0xK3FFF8000000000000000, ptr %a
; CHECK:      ret void

define void @i40test(ptr %a, ptr %b) nounwind uwtable {
entry:
  %t = load i40, ptr %a
  store i40 %t, ptr %b, align 8
  ret void
}
; CHECK-LABEL: @i40test
; Exactly one shadow update for load access.
; CHECK-NOT:  store i64
; CHECK:      %[[NEW_LD_SHADOW:[^ ]*]] = add i64 %{{.*}}, 1
; CHECK-NEXT: store i64 %[[NEW_LD_SHADOW]]
; CHECK-NOT:  store i64
; The actual load.
; CHECK:      %t = load i40, ptr %a
; Exactly one shadow update for store access.
; CHECK-NOT:  store i64
; CHECK:      %[[NEW_ST_SHADOW:[^ ]*]] = add i64 %{{.*}}, 1
; CHECK-NEXT: store i64 %[[NEW_ST_SHADOW]]
; CHECK-NOT:  store i64
; The actual store.
; CHECK:      store i40 %t, ptr %b
; CHECK:      ret void

define void @i64test_align1(ptr %b) nounwind uwtable {
  entry:
  store i64 0, ptr %b, align 1
  ret void
}
; CHECK-LABEL: @i64test
; Exactly one shadow update for store access.
; CHECK-NOT:  store i64
; CHECK: %[[NEW_ST_SHADOW:[^ ]*]] = add i64 %{{.*}}, 1
; CHECK-NEXT: store i64 %[[NEW_ST_SHADOW]]
; CHECK-NOT:  store i64
; The actual store.
; CHECK:      store i64 0, ptr %b
; CHECK:      ret void

define void @i80test(ptr %a, ptr %b) nounwind uwtable {
  entry:
  %t = load i80, ptr %a
  store i80 %t, ptr %b, align 8
  ret void
}
; CHECK-LABEL: i80test
; Exactly one shadow update for load access.
; CHECK-NOT:  store i64
; CHECK:      %[[NEW_LD_SHADOW:[^ ]*]] = add i64 %{{.*}}, 1
; CHECK-NEXT: store i64 %[[NEW_LD_SHADOW]]
; CHECK-NOT:  store i64
; The actual load.
; CHECK:      %t = load i80, ptr %a
; Exactly one shadow update for store access.
; CHECK-NOT:  store i64
; CHECK:      %[[NEW_ST_SHADOW:[^ ]*]] = add i64 %{{.*}}, 1
; CHECK-NEXT: store i64 %[[NEW_ST_SHADOW]]
; CHECK-NOT:  store i64
; The actual store.
; CHECK:      store i80 %t, ptr %b
; CHECK:      ret void

; memprof should not instrument functions with available_externally linkage.
define available_externally i32 @f_available_externally(ptr %a)  {
entry:
  %tmp1 = load i32, ptr %a
  ret i32 %tmp1
}
; CHECK-LABEL: @f_available_externally
; CHECK-NOT: __memprof_shadow_memory_dynamic_address
; CHECK: ret i32

declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1) nounwind
declare void @llvm.memmove.p0.p0.i64(ptr nocapture, ptr nocapture readonly, i64, i1) nounwind
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture readonly, i64, i1) nounwind

define void @memintr_test(ptr %a, ptr %b) nounwind uwtable {
  entry:
  tail call void @llvm.memset.p0.i64(ptr %a, i8 0, i64 100, i1 false)
  tail call void @llvm.memmove.p0.p0.i64(ptr %a, ptr %b, i64 100, i1 false)
  tail call void @llvm.memcpy.p0.p0.i64(ptr %a, ptr %b, i64 100, i1 false)
  ret void
}

; CHECK-LABEL: memintr_test
; CHECK: __memprof_memset
; CHECK: __memprof_memmove
; CHECK: __memprof_memcpy
; CHECK: ret void

declare void @llvm.memset.element.unordered.atomic.p0.i64(ptr nocapture writeonly, i8, i64, i32) nounwind
declare void @llvm.memmove.element.unordered.atomic.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i32) nounwind
declare void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i32) nounwind

define void @memintr_element_atomic_test(ptr %a, ptr %b) nounwind uwtable {
  ; This is a canary test to make sure that these don't get lowered into calls that don't
  ; have the element-atomic property. Eventually, memprof will have to be enhanced to lower
  ; these properly.
  ; CHECK-LABEL: memintr_element_atomic_test
  ; CHECK: tail call void @llvm.memset.element.unordered.atomic.p0.i64(ptr align 1 %a, i8 0, i64 100, i32 1)
  ; CHECK: tail call void @llvm.memmove.element.unordered.atomic.p0.p0.i64(ptr align 1 %a, ptr align 1 %b, i64 100, i32 1)
  ; CHECK: tail call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %a, ptr align 1 %b, i64 100, i32 1)
  ; CHECK: ret void
  tail call void @llvm.memset.element.unordered.atomic.p0.i64(ptr align 1 %a, i8 0, i64 100, i32 1)
  tail call void @llvm.memmove.element.unordered.atomic.p0.p0.i64(ptr align 1 %a, ptr align 1 %b, i64 100, i32 1)
  tail call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %a, ptr align 1 %b, i64 100, i32 1)
  ret void
}


; CHECK: define internal void @memprof.module_ctor()
; CHECK: call void @__memprof_init()
