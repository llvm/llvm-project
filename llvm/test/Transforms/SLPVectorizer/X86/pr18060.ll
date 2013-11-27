; RUN: opt < %s -slp-vectorizer -S -mtriple=i386-pc-linux

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32-S128"
target triple = "i386-pc-linux"

; Function Attrs: nounwind
define i32 @_Z16adjustFixupValueyj(i64 %Value, i32 %Kind) {
entry:
  %extract.t = trunc i64 %Value to i32
  %extract = lshr i64 %Value, 12
  %extract.t6 = trunc i64 %extract to i32
  switch i32 %Kind, label %sw.default [
    i32 0, label %return
    i32 1, label %return
    i32 129, label %sw.bb1
    i32 130, label %sw.bb2
  ]

sw.default:                                       ; preds = %entry
  call void @_Z25llvm_unreachable_internalv()
  unreachable

sw.bb1:                                           ; preds = %entry
  %shr = lshr i64 %Value, 16
  %extract.t5 = trunc i64 %shr to i32
  %extract7 = lshr i64 %Value, 28
  %extract.t8 = trunc i64 %extract7 to i32
  br label %sw.bb2

sw.bb2:                                           ; preds = %sw.bb1, %entry
  %Value.addr.0.off0 = phi i32 [ %extract.t, %entry ], [ %extract.t5, %sw.bb1 ]
  %Value.addr.0.off12 = phi i32 [ %extract.t6, %entry ], [ %extract.t8, %sw.bb1 ]
  %conv6 = and i32 %Value.addr.0.off0, 4095
  %conv4 = shl i32 %Value.addr.0.off12, 16
  %shl = and i32 %conv4, 983040
  %or = or i32 %shl, %conv6
  %or11 = or i32 %or, 8388608
  br label %return

return:                                           ; preds = %sw.bb2, %entry, %entry
  %retval.0 = phi i32 [ %or11, %sw.bb2 ], [ %extract.t, %entry ], [ %extract.t, %entry ]
  ret i32 %retval.0
}

; Function Attrs: noreturn
declare void @_Z25llvm_unreachable_internalv()

