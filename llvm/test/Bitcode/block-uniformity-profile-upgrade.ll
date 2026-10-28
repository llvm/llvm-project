; RUN: llvm-dis < %s.bc | FileCheck %s
; RUN: opt -passes=verify -disable-output < %s.bc

; The bitcode was generated before block.uniformity.profile used presence-only
; metadata. Preserve true hints as empty nodes and remove false hints.

define void @branch_metadata(i1 %cond) {
; CHECK-LABEL: define void @branch_metadata(
entry:
  br i1 %cond, label %uniform, label %divergent, !block.uniformity.profile !0
; CHECK: br i1 %cond, label %uniform, label %divergent, !block.uniformity.profile [[UNIFORM:![0-9]+]]

uniform:
  ret void

divergent:
  br label %uniform, !block.uniformity.profile !1
; CHECK: br label %uniform{{$}}
}

; CHECK: [[UNIFORM]] = !{}
!0 = !{i1 true}
!1 = !{i1 false}
