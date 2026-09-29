; RUN: not llc -verify-machineinstrs -o - -mtriple=s390x-linux-gnu -code-model=tiny < %s 2>&1 | FileCheck %s --check-prefix=TINY
; RUN: not llc -verify-machineinstrs -o - -mtriple=s390x-linux-gnu -code-model=kernel < %s 2>&1 | FileCheck %s --check-prefix=KERNEL

; TINY:    Target does not support the tiny CodeModel
; KERNEL:    Target does not support the kernel CodeModel

define void @foo() {
  ret void
}
