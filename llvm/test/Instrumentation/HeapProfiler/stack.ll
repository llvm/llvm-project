; Test optional annotation of stack addresses.
;
; RUN: opt < %s -passes='function(memprof),module(memprof-module)' -S | FileCheck --check-prefixes=CHECK %s
; RUN: opt < %s -passes='function(memprof),module(memprof-module)' -memprof-instrument-stack -S | FileCheck --check-prefixes=CHECK,STACK %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

define i32 @test_stack_load() {
entry:
  %x = alloca i32, align 4
  %tmp1 = load i32, i32* %x, align 4
  ret i32 %tmp1
}
; CHECK-LABEL: @test_stack_load
; CHECK:         %[[SHADOW_OFFSET:[^ ]*]] = load i64, i64* @__memprof_shadow_memory_dynamic_address
; CHECK-NEXT:	 %x = alloca i32
; STACK-NEXT:    %[[LOAD_ADDR:[^ ]*]] = ptrtoint i32* %x to i64
; STACK-NEXT:    %[[MASKED_ADDR:[^ ]*]] = and i64 %[[LOAD_ADDR]], -64
; STACK-NEXT:    %[[SHIFTED_ADDR:[^ ]*]] = lshr i64 %[[MASKED_ADDR]], 3
; STACK-NEXT:    add i64 %[[SHIFTED_ADDR]], %[[SHADOW_OFFSET]]
; STACK-NEXT:    %[[LOAD_SHADOW_PTR:[^ ]*]] = inttoptr
; STACK-NEXT:    %[[LOAD_SHADOW:[^ ]*]] = load i64, i64* %[[LOAD_SHADOW_PTR]]
; STACK-NEXT:    %[[NEW_SHADOW:[^ ]*]] = add i64 %[[LOAD_SHADOW]], 1
; STACK-NEXT:    store i64 %[[NEW_SHADOW]], i64* %[[LOAD_SHADOW_PTR]]
; The actual load.
; CHECK-NEXT:    %tmp1 = load i32, i32* %x
; CHECK-NEXT:    ret i32 %tmp1

define void @test_stack_store() {
entry:
  %x = alloca i32, align 4
  store i32 1, i32* %x, align 4
  ret void
}
; CHECK-LABEL: @test_stack_store
; CHECK:         %[[SHADOW_OFFSET:[^ ]*]] = load i64, i64* @__memprof_shadow_memory_dynamic_address
; CHECK-NEXT:	 %x = alloca i32
; STACK-NEXT:    %[[STORE_ADDR:[^ ]*]] = ptrtoint i32* %x to i64
; STACK-NEXT:    %[[MASKED_ADDR:[^ ]*]] = and i64 %[[STORE_ADDR]], -64
; STACK-NEXT:    %[[SHIFTED_ADDR:[^ ]*]] = lshr i64 %[[MASKED_ADDR]], 3
; STACK-NEXT:    add i64 %[[SHIFTED_ADDR]], %[[SHADOW_OFFSET]]
; STACK-NEXT:    %[[STORE_SHADOW_PTR:[^ ]*]] = inttoptr
; STACK-NEXT:    %[[STORE_SHADOW:[^ ]*]] = load i64, i64* %[[STORE_SHADOW_PTR]]
; STACK-NEXT:    %[[NEW_SHADOW:[^ ]*]] = add i64 %[[STORE_SHADOW]], 1
; STACK-NEXT:    store i64 %[[NEW_SHADOW]], i64* %[[STORE_SHADOW_PTR]]
; The actual store.
; CHECK-NEXT:    store i32 1, i32* %x
; CHECK-NEXT:    ret void
