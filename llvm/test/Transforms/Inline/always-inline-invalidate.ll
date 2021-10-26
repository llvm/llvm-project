; RUN: opt -passes='require<no-op-module>,always-inline' < %s 2>&1 -debug-pass-manager | FileCheck %s

; CHECK: Invalidating {{.*}} NoOpModuleAnalysis
; CHECK-NOT: @f

define internal void @f() alwaysinline {
  unreachable
}
