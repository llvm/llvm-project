;; This tests that llc accepts all valid LoongArch CPUs.
;; Note the 'generic' names have been tested in cpu-name-generic.ll.

; RUN: llc < %s --mtriple=loongarch64 -mattr=+d --mcpu=loongarch64 2>&1 | FileCheck %s
; RUN: llc < %s --mtriple=loongarch64 -mattr=+d --mcpu=la464 2>&1 | FileCheck %s
; RUN: llc < %s --mtriple=loongarch64 -mattr=+d 2>&1 | FileCheck %s

; CHECK-NOT: {{.*}} is not a recognized processor for this target

define void @f() {
  ret void
}

define void @tune_cpu_loongarch64() "tune-cpu"="loongarch64" {
  ret void
}

define void @tune_cpu_la464() "tune-cpu"="la464" {
  ret void
}
