; RUN: opt -passes=aa-eval -print-all-alias-modref-info -disable-output < %s 2>&1 | FileCheck %s

target datalayout = "p:32:32"

; Make sure that using a LocationSize larget than the index space does not
; assert.

; CHECK: Just Mod:  Ptr: i32* %gep	<->  call void @llvm.memset.p0.i64(ptr %p, i8 0, i64 4294967296, i1 false)
define void @test(ptr %p, i32 %idx) {
  %gep = getelementptr i8, ptr %p, i32 %idx
  load i32, ptr %gep
  call void @llvm.memset.i64(ptr %p, i8 0, i64 u0x100000000, i1 false)
  ret void
}
