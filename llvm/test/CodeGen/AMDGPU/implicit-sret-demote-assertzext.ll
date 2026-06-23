; RUN: llc -mtriple=amdgcn-- -mcpu=hawaii -debug-only=isel-dump -filter-print-funcs=ret_large_implicit_sret -o /dev/null %s 2>&1 | FileCheck %s
; REQUIRES: asserts

; The demoted return pointer should carry an explicit AssertZext at the
; return-store use. Do not recover this through a broad CopyFromReg known-bits
; query, since that cannot prove which CopyFromReg is being inspected.

define [64 x i32] @ret_large_implicit_sret(i32 %x) #0 {
; CHECK-LABEL: Initial selection DAG: %bb.0 'ret_large_implicit_sret:'
; CHECK: CopyToReg {{.*}} Register:i32 %[[SRET:[0-9]+]],
; CHECK: CopyFromReg {{.*}} Register:i32 %[[SRET]]
; CHECK-NEXT: {{ *}}t{{[0-9]+}}: i32 = AssertZext {{.*}} ValueType:ch:i17
  %ins = insertvalue [64 x i32] undef, i32 %x, 0
  %ins2 = insertvalue [64 x i32] %ins, i32 %x, 33
  %ins3 = insertvalue [64 x i32] %ins2, i32 %x, 63
  ret [64 x i32] %ins3
}

attributes #0 = { nounwind }
