; RUN: llc -mtriple=nvptx64-nvidia-cuda -mattr=+ptx70 -addrsig < %s | FileCheck %s

; CHECK-NOT: .addrsig
; CHECK-NOT: .addrsig_sym
; CHECK: .visible .func foo

@p = addrspace(1) global ptr @foo

define void @foo() {
entry:
  ret void
}
