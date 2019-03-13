; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; Check that remangling code doesn't fail on an intrinsic with wrong signature

; CHECK: Attribute after last parameter!
; CHECK-NEXT: void (i8*, i8, i64)* @llvm.memset.i64
declare void @llvm.memset.i64(i8* nocapture, i8, i64) nounwind

; CHECK: Attribute after last parameter!
; CHECK-NEXT: void (i8*, i8, i64)* @llvm.memcpy.i64
declare void @llvm.memcpy.i64(i8* nocapture, i8, i64) nounwind

; CHECK: Attribute after last parameter!
; CHECK-NEXT: void (i8*, i8, i64)* @llvm.memmove.i64
declare void @llvm.memmove.i64(i8* nocapture, i8, i64) nounwind
