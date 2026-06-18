; RUN: opt -S -passes=mergefunc < %s | FileCheck %s

; Verify that !callgraph metadata is correctly preserved and copied to the new
; thunk/backing functions when two identical functions are merged, and that
; the metadata nodes are correctly uniqued.

declare void @dummy()

; CHECK: define void @func1(ptr {{.*}}) {{.*}} !callgraph [[MD:![0-9]+]] {
; CHECK: define void @func2(ptr {{.*}}) {{.*}} !callgraph [[MD]] {

define void @func1(ptr %a) unnamed_addr !callgraph !0 {
  call void @dummy()
  ret void
}

define void @func2(ptr %a) unnamed_addr !callgraph !1 {
  call void @dummy()
  ret void
}

!0 = !{!"_ZTSFvPvE.generalized"}
!1 = !{!"_ZTSFvPvE.generalized"}

; CHECK: [[MD]] = !{!"_ZTSFvPvE.generalized"}
