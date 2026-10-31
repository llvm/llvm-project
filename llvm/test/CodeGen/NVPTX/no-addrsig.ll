; RUN: llc -mtriple=nvptx64-nvidia-cuda -addrsig < %s | FileCheck %s --implicit-check-not=.addrsig

; CHECK: .visible .func foo

@p = addrspace(1) global ptr @foo

define void @foo() {
  ret void
}
