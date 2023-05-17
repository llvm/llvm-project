; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; Check that remangling code doesn't fail on an intrinsic with wrong signature

; CHECK: Attribute after last parameter!
; CHECK-NEXT: ptr @llvm.memset.i64
declare void @llvm.memset.i64(ptr nocapture, i8, i64) nounwind

; CHECK: Attribute after last parameter!
; CHECK-NEXT: ptr @llvm.memcpy.i64
declare void @llvm.memcpy.i64(ptr nocapture, i8, i64) nounwind

; CHECK: Attribute after last parameter!
; CHECK-NEXT: ptr @llvm.memmove.i64
declare void @llvm.memmove.i64(ptr nocapture, i8, i64) nounwind
