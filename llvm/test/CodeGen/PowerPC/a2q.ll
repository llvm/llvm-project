; RUN: llc -verify-machineinstrs < %s -march=ppc64 -mcpu=a2q | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -march=ppc64 -mcpu=a2 -mattr=+qpx | FileCheck %s

define void @foo() {
entry:
  ret void
}

; CHECK: @foo

