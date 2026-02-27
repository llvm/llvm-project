; RUN: llc -mtriple=aarch64 -mattr=+sme -stop-after=aarch64-isel < %s | FileCheck %s

target triple = "aarch64"

declare void @foo() "aarch64_pstate_sm_enabled"

define dso_local void @bar() local_unnamed_addr {
; CHECK-LABEL: name: bar
; CHECK: hasStreamingModeChanges: true
entry:
  tail call void @foo() "aarch64_pstate_sm_enabled"
  ret void
}
