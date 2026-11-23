; Test that a lowered type test for a type is simplified to true
; if the target is a constant member of that type.

; RUN: opt -S %s -passes=simplify-type-tests | FileCheck %s

; Test that the simplification does not occur if the type is wrong.
 
; RUN: sed -e 's/"_ZTSFvvE"/"wrongtype"/g' %s | opt -S -passes=simplify-type-tests | FileCheck --check-prefix=WRONGTYPE %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__typeid__ZTSFvvE_global_addr = external hidden global [0 x i8], code_model "small"

define void @_Z2fpv.cfi() !type !0 {
  ret void
}

define i64 @main() {
  %1 = icmp eq ptr @_Z2fpv, @__typeid__ZTSFvvE_global_addr
  ; CHECK: br i1 true
  ; WRONGTYPE: br i1 %
  br i1 %1, label %3, label %2

2:
  tail call void @llvm.ubsantrap(i8 2)
  unreachable

3:
  ; CHECK: br i1 true
  ; WRONGTYPE: br i1 %
  %c = icmp eq i64 ptrtoint (ptr @_Z2fpv to i64), ptrtoint (ptr @__typeid__ZTSFvvE_global_addr to i64)
  br i1 %c, label %4, label %2

4:
  tail call void @_Z2fpv()
  ; CHECK: ret i64 0
  ; WRONGTYPE: ret i64 sub
  ret i64 sub (i64 ptrtoint (ptr @__typeid__ZTSFvvE_global_addr to i64), i64 ptrtoint (ptr @_Z2fpv to i64))
}

declare void @llvm.ubsantrap(i8 immarg)

declare hidden void @_Z2fpv()

!0 = !{i64 0, !"_ZTSFvvE"}
