; RUN: opt < %s -dfsan -dfsan-abilist=%S/Inputs/abilist.txt -S | FileCheck %s -DSHADOW_XOR_MASK=87960930222080
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

define i32 @function_to_force_zero(i32 %0, i32* %1) {
  ; CHECK-LABEL: define i32 @function_to_force_zero.dfsan
  ; CHECK: %[[#SHADOW_XOR:]] = xor i64 {{.*}}, [[SHADOW_XOR_MASK]]
  ; CHECK: %[[#SHADOW_PTR:]] = inttoptr i64 %[[#SHADOW_XOR]] to i8*
  ; CHECK: %[[#SHADOW_BITCAST:]] = bitcast i8* %[[#SHADOW_PTR]] to i32*
  ; CHECK: store i32 0, i32* %[[#SHADOW_BITCAST]]
  ; CHECK: store i32 %{{.*}}
  store i32 %0, i32* %1, align 4
  ; CHECK: store i8 0, {{.*}}@__dfsan_retval_tls
  ; CHECK: ret i32
  ret i32 %0
}
