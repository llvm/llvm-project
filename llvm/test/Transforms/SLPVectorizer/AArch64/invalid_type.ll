; RUN: opt < %s -slp-vectorizer -S -pass-remarks-missed=slp-vectorizer 2>&1 | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

; This test check that slp vectorizer is not trying to vectorize instructions already vectorized.
; CHECK: remark: <unknown>:0:0: Cannot SLP vectorize list: type <16 x i8> is unsupported by vectorizer

define void @vector() {
  %load0 = tail call <16 x i8> @vector.load(<16 x i8> *undef, i32 1)
  %load1 = tail call <16 x i8> @vector.load(<16 x i8> *undef, i32 2)
  %add = add <16 x i8> %load1, %load0
  tail call void @vector.store(<16 x i8> %add, <16 x i8>* undef, i32 1)
  ret void
}

declare <16 x i8> @vector.load(<16 x i8>*, i32)
declare void @vector.store(<16 x i8>, <16 x i8>*, i32)
