; RUN: not llc -mtriple=arm64e-apple-ios -global-isel=false %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=arm64e-apple-ios -global-isel %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=arm64e-apple-ios -fast-isel %s -o /dev/null 2>&1 | FileCheck %s

; nonlazybind emits an unauthenticated indirect branch (blr) that bypasses
; pointer authentication, which is incompatible with arm64e. Verify each ISel
; rejects it.

; CHECK: error: {{.*}}: nonlazybind attribute is not compatible with arm64e

declare void @external() nonlazybind

define void @caller() nounwind {
  call void @external()
  ret void
}
