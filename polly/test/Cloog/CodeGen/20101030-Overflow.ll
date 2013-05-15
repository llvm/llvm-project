; RUN: opt %loadPolly %defaultOpts -polly-codegen %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

define void @compdecomp() nounwind {
entry:
  %max = alloca i64
  %i = load i64* undef
  br label %bb37

bb37:                                             ; preds = %bb36, %bb28
  %tmp = icmp ugt i64 %i, 0
  br i1 %tmp, label %bb38, label %bb39

bb38:                                             ; preds = %bb37
  store i64 %i, i64* %max
  br label %bb39

bb39:                                             ; preds = %bb38, %bb37
  unreachable

}
