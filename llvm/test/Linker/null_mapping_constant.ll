; RUN: llvm-link %s -S -o - | FileCheck %s
; Check that the constant is not linked and the metadata is correctly referencing a nullptr
; CHECK: !0 = !{!"foo", null, i64 16}



target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"
@foo = external unnamed_addr constant { [4 x i8*], [32 x i8] }, align 32
!llvm.bitsets = !{!0}
!0 = !{!"foo", [4 x i8*]* getelementptr inbounds ({ [4 x i8*], [32 x i8] }, { [4 x i8*], [32 x i8] }* @foo, i32 0, i32 0), i64 16}
