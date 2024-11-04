; RUN: llc -mtriple=arm64-apple-ios %s -o - -O0 -global-isel=1 | FileCheck %s
define void @foo(i512 %in) {
; CHECK-LABEL: foo:
; CHECK: cbz
  switch i512 %in, label %default [
    i512 3923188584616675477397368389504791510063972152790021570560, label %l1
    i512 3923188584616675477397368389504791510063972152790021570561, label %l2
    i512 3923188584616675477397368389504791510063972152790021570562, label %l3
  ]

default:
  ret void

l1:
  ret void

l2:
  ret void

l3:
  ret void
}
