; RUN: llc -mtriple=i386-apple-darwin11 -O2 < %s | FileCheck %s

%struct.I = type { i32 (...)** }
define x86_stdcallcc void @bar(%struct.I* nocapture %this) ssp align 2 {
; CHECK: bar:
; CHECK-NOT: jmp
; CHECK: ret $4
entry:
  tail call void @foo()
  ret void
}

declare void @foo()
