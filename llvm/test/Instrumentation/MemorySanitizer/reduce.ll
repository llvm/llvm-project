
; RUN: opt < %s -msan-check-access-address=0 -msan-track-origins=1 -S -passes='module(msan)' 2>&1 | \
; RUN:   FileCheck -allow-deprecated-dag-overlap --check-prefix=CHECK %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i32 @llvm.vector.reduce.add(<3 x i32>)
declare i32 @llvm.vector.reduce.and(<3 x i32>)
declare i32 @llvm.vector.reduce.or(<3 x i32>)

; CHECK-LABEL: @reduce_add
define i32 @reduce_add() sanitize_memory {
; CHECK: [[P:%.*]] = inttoptr i64 0 to ptr
  %p = inttoptr i64 0 to ptr
; CHECK: [[O:%.*]] = load <3 x i32>, ptr [[P]]
  %o = load <3 x i32>, ptr %p
; CHECK: [[O_SHADOW:%.*]] = load <3 x i32>, ptr
; CHECK: [[O_ORIGIN:%.*]] = load i32, ptr
; CHECK: [[R_SHADOW:%.*]] = call i32 @llvm.vector.reduce.or.v3i32(<3 x i32> [[O_SHADOW]])
; CHECK: [[R:%.*]] = call i32 @llvm.vector.reduce.add.v3i32(<3 x i32> [[O]])
  %r = call i32 @llvm.vector.reduce.add(<3 x i32> %o)
; CHECK: store i32 [[R_SHADOW]], {{.*}} @__msan_retval_tls
; CHECK: store i32 [[O_ORIGIN]], {{.*}} @__msan_retval_origin_tls
; CHECK: ret i32 [[R]]
  ret i32 %r
}

; CHECK-LABEL: @reduce_and
define i32 @reduce_and() sanitize_memory {
; CHECK: [[P:%.*]] = inttoptr i64 0 to ptr
  %p = inttoptr i64 0 to ptr
; CHECK: [[O:%.*]] = load <3 x i32>, ptr [[P]]
  %o = load <3 x i32>, ptr %p
; CHECK: [[O_SHADOW:%.*]] = load <3 x i32>, ptr
; CHECK: [[O_ORIGIN:%.*]] = load i32, ptr
; CHECK: [[O_SHADOW_1:%.*]] = or <3 x i32> [[O]], [[O_SHADOW]]
; CHECK: [[O_SHADOW_2:%.*]] = call i32 @llvm.vector.reduce.and.v3i32(<3 x i32> [[O_SHADOW_1]]
; CHECK: [[O_SHADOW_3:%.*]] = call i32 @llvm.vector.reduce.or.v3i32(<3 x i32> [[O_SHADOW]])
; CHECK: [[R_SHADOW:%.*]] = and i32 [[O_SHADOW_2]], [[O_SHADOW_3]]
; CHECK: [[R:%.*]] = call i32 @llvm.vector.reduce.and.v3i32(<3 x i32> [[O]])
  %r = call i32 @llvm.vector.reduce.and(<3 x i32> %o)
; CHECK: store i32 [[R_SHADOW]], {{.*}} @__msan_retval_tls
; CHECK: store i32 [[O_ORIGIN]], {{.*}} @__msan_retval_origin_tls
; CHECK: ret i32 [[R]]
  ret i32 %r
}

; CHECK-LABEL: @reduce_or
define i32 @reduce_or() sanitize_memory {
; CHECK: [[P:%.*]] = inttoptr i64 0 to ptr
  %p = inttoptr i64 0 to ptr
; CHECK: [[O:%.*]] = load <3 x i32>, ptr [[P]]
  %o = load <3 x i32>, ptr %p
; CHECK: [[O_SHADOW:%.*]] = load <3 x i32>, ptr
; CHECK: [[O_ORIGIN:%.*]] = load i32, ptr
; CHECK: [[NOT_O:%.*]] = xor <3 x i32> [[O]], <i32 -1, i32 -1, i32 -1>
; CHECK: [[O_SHADOW_1:%.*]] = or <3 x i32> [[NOT_O]], [[O_SHADOW]]
; CHECK: [[O_SHADOW_2:%.*]] = call i32 @llvm.vector.reduce.and.v3i32(<3 x i32> [[O_SHADOW_1]]
; CHECK: [[O_SHADOW_3:%.*]] = call i32 @llvm.vector.reduce.or.v3i32(<3 x i32> [[O_SHADOW]])
; CHECK: [[R_SHADOW:%.*]] = and i32 [[O_SHADOW_2]], [[O_SHADOW_3]]
; CHECK: [[R:%.*]] = call i32 @llvm.vector.reduce.or.v3i32(<3 x i32> [[O]])
  %r = call i32 @llvm.vector.reduce.or(<3 x i32> %o)
; CHECK: store i32 [[R_SHADOW]], {{.*}} @__msan_retval_tls
; CHECK: store i32 [[O_ORIGIN]], {{.*}} @__msan_retval_origin_tls
; CHECK: ret i32 [[R]]
  ret i32 %r
}
