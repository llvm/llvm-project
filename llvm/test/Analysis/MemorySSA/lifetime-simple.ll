; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>,verify<memoryssa>' -disable-output < %s 2>&1 | FileCheck %s
; This test checks that lifetime markers are considered clobbers of %P,
; and due to lack of noalias information, of %Q as well.

define i8 @test(ptr %P, ptr %Q) {
entry:
; CHECK:  1 = MemoryDef(liveOnEntry)
; CHECK-NEXT:   call void @llvm.lifetime.start.p0(i64 32, ptr %P)
  call void @llvm.lifetime.start.p0(i64 32, ptr %P)
; CHECK:  MemoryUse(1)
; CHECK-NEXT:   %0 = load i8, ptr %P
  %0 = load i8, ptr %P
; CHECK:  2 = MemoryDef(1)
; CHECK-NEXT:   store i8 1, ptr %P
  store i8 1, ptr %P
; CHECK:  3 = MemoryDef(2)
; CHECK-NEXT:   call void @llvm.lifetime.end.p0(i64 32, ptr %P)
  call void @llvm.lifetime.end.p0(i64 32, ptr %P)
; CHECK:  MemoryUse(3)
; CHECK-NEXT:   %1 = load i8, ptr %P
  %1 = load i8, ptr %P
; CHECK:  MemoryUse(3)
; CHECK-NEXT:   %2 = load i8, ptr %Q
  %2 = load i8, ptr %Q
  ret i8 %1
}
declare void @llvm.lifetime.start.p0(i64 %S, ptr nocapture %P) readonly
declare void @llvm.lifetime.end.p0(i64 %S, ptr nocapture %P)
