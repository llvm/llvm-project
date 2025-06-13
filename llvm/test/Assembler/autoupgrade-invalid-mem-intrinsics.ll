; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; Check that remangling code doesn't fail on an intrinsic with wrong signature
; TODO: This should probably produce an error.

; CHECK: declare void @llvm.memset.i64
declare void @llvm.memset.i64(ptr nocapture, i8, i64) nounwind

; CHECK: declare void @llvm.memcpy.i64
declare void @llvm.memcpy.i64(ptr nocapture, i8, i64) nounwind

; CHECK: declare void @llvm.memmove.i64
declare void @llvm.memmove.i64(ptr nocapture, i8, i64) nounwind
