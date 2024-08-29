; assert in DAGlegalizer with fake use of half precision float.
; Changes to half float promotion.
; RUN: llc -stop-after=finalize-isel -o - %s | FileCheck %s
;
; CHECK:      bb.0.entry:
; CHECK-NEXT: %0:fr16 = FsFLD0SH
; CHECK-NEXT: FAKE_USE killed %0
;
target triple = "x86_64-unknown-unknown"

define void @_Z6doTestv() local_unnamed_addr optdebug {
entry:
  tail call void (...) @llvm.fake.use(half 0xH0000)
  ret void
}
