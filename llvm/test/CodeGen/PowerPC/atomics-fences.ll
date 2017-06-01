; RUN: llc < %s -mtriple=powerpc-apple-darwin -march=ppc32 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=powerpc-apple-darwin -march=ppc64 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=powerpc-apple-darwin -mcpu=440 | FileCheck %s --check-prefix=PPC440

; Fences
define void @fence_acquire() {
; CHECK-LABEL: fence_acquire
; CHECK: lwsync
; PPC440-NOT: lwsync
; PPC440: msync
  fence acquire
  ret void
}
define void @fence_release() {
; CHECK-LABEL: fence_release
; CHECK: lwsync
; PPC440-NOT: lwsync
; PPC440: msync
  fence release
  ret void
}
define void @fence_seq_cst() {
; CHECK-LABEL: fence_seq_cst
; CHECK: sync
; PPC440: msync
  fence seq_cst
  ret void
}
