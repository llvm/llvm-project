; RUN: split-file %s %t

;--- test1.ll
; Verify that auto-upgrade eliminate the unused invalid declaration of lifetime
; start and end intrinsics. 
; RUN: llvm-as < %t/test1.ll | llvm-dis | FileCheck %t/test1.ll
; CHECK-NOT: @llvm.lifetime.start.i64(i64)
; CHECK-NOT: @llvm.lifetime.end.i64(i64)

declare void @llvm.lifetime.start.i64(i64)
declare void @llvm.lifetime.end.i64(i64)

;--- test2.ll
; Verify that if there is an actual call to the invalid lifetime intrinsics,
; the calls and the declaration will be deleted.
; RUN: llvm-as < %t/test2.ll | llvm-dis | FileCheck %t/test2.ll

; CHECK-NOT: @llvm.lifetime.start.i64(i64)
; CHECK-NOT: @llvm.lifetime.end.i64(i64)
; CHECK-NOT: call void @llvm.lifetime.start.i64(i64 0)
; CHECK-NOT: call void @llvm.lifetime.end.i64(i64 0)

declare void @llvm.lifetime.start.i64(i64)
declare void @llvm.lifetime.end.i64(i64)

define void @foo() {
  call void @llvm.lifetime.start.i64(i64 0)
  call void @llvm.lifetime.end.i64(i64 0)
  ret void
}
