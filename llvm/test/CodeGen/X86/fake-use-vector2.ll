; RUN: llc -stop-after=finalize-isel -mtriple=x86_64-unknown-linux -filetype=asm -o - %s | FileCheck %s
;
; Make sure we can split vectors that are used as operands of FAKE_USE.

; Generated from:
;
; typedef long __attribute__((ext_vector_type(8))) long8;
; void test0() { long8 id208 {0, 1, 2, 3, 4, 5, 6, 7}; }

; ModuleID = 't5.cpp'
source_filename = "t5.cpp"


; CHECK:     %0:vr256 = VMOV
; CHECK:     %1:vr256 = VMOV
; CHECK-DAG: FAKE_USE killed %1
; CHECK-DAG: FAKE_USE killed %0
; CHECK:     RET
define void @_Z5test0v() local_unnamed_addr #0 {
entry:
  tail call void (...) @llvm.fake.use(<8 x i64> <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7>) #1
  ret void
}

declare void @llvm.fake.use(...)

attributes #0 = { "target-cpu"="btver2" optdebug }
