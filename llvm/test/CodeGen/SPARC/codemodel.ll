; RUN: not --crash llc -verify-machineinstrs -o - -mtriple=sparc64-unknown-linux -code-model=tiny < %s 2>&1 | FileCheck %s --check-prefix=TINY
; RUN: not --crash llc -verify-machineinstrs -o - -mtriple=sparc64-unknown-linux -code-model=kernel < %s 2>&1 | FileCheck %s --check-prefix=KERNEL

; TINY:    Target does not support the tiny CodeModel
; KERNEL:    Target does not support the kernel CodeModel

define void @foo() {
  ret void
}
