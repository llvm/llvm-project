; RUN: opt < %s -msan-check-access-address=0 -S -passes=msan 2>&1 | FileCheck %s
; RUN: opt < %s -msan-check-access-address=0 -msan-track-origins=2 -S -passes=msan 2>&1 | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @nofreeze(i32* %ptr) sanitize_memory {
    ; CHECK-LABEL: @nofreeze
    %val = load i32, i32* %ptr
    ; CHECK: [[SHADOW_PTR:%.*]] = inttoptr
    ; CHECK: [[SHADOW:%.*]] = load i32, i32* [[SHADOW_PTR]]
    ; CHECK: store i32 [[SHADOW]], {{.*}} @__msan_retval_tls
    ret i32 %val
}

define i32 @freeze_inst(i32* %ptr) sanitize_memory {
    ; CHECK-LABEL: @freeze_inst
    %val = load i32, i32* %ptr
    %freeze_val = freeze i32 %val
    ; CHECK-NOT: __msan_warning
    ; CHECK: store i32 0, {{.*}} @__msan_retval_tls
    ret i32 %freeze_val
}
