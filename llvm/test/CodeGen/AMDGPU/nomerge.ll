; RUN: llc -mtriple=amdgpu10.30-amd-amdhsa -global-isel=0 < %s | FileCheck %s
; RUN: llc -mtriple=amdgpu10.30-amd-amdhsa -global-isel=1 < %s | FileCheck %s

declare void @report() noreturn
declare void @bar()

; CHECK-LABEL: {{^}}noreturn_merge:
; CHECK:       s_swappc_b64
; CHECK-NOT:   s_swappc_b64
define void @noreturn_merge(i1 inreg %c) {
  br i1 %c, label %a, label %b
a:
  call void @report()
  unreachable
b:
  call void @report()
  unreachable
}

; CHECK-LABEL: {{^}}noreturn_nomerge:
; CHECK:       s_swappc_b64
; CHECK:       s_swappc_b64
; CHECK-NOT:   s_swappc_b64
define void @noreturn_nomerge(i1 inreg %c) {
  br i1 %c, label %a, label %b
a:
  call void @report() nomerge
  unreachable
b:
  call void @report() nomerge
  unreachable
}

; CHECK-LABEL: {{^}}tail_merge:
; CHECK:       s_setpc_b64
; CHECK-NOT:   s_setpc_b64
define void @tail_merge(i1 inreg %c) {
  br i1 %c, label %a, label %b
a:
  tail call void @bar()
  ret void
b:
  tail call void @bar()
  ret void
}

; CHECK-LABEL: {{^}}tail_nomerge:
; CHECK:       s_setpc_b64
; CHECK:       s_setpc_b64
; CHECK-NOT:   s_setpc_b64
define void @tail_nomerge(i1 inreg %c) {
  br i1 %c, label %a, label %b
a:
  tail call void @bar() nomerge
  ret void
b:
  tail call void @bar() nomerge
  ret void
}
