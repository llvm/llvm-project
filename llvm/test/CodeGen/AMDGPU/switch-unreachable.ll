; RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck %s

; This testcase was discovered in si-annotate-cf.ll, where none of the
; RUN lines was actually exercising it. See that files git log for its
; history.

; FIXME: should emit s_endpgm
; CHECK-LABEL: {{^}}switch_unreachable:
; CHECK-NOT: s_endpgm
; CHECK: .Lfunc_end
define amdgpu_kernel void @switch_unreachable(ptr addrspace(1) %g, ptr addrspace(3) %l, i32 %x) nounwind {
centry:
  switch i32 %x, label %sw.default [
    i32 0, label %sw.bb
    i32 60, label %sw.bb
  ]

sw.bb:
  unreachable

sw.default:
  unreachable

sw.epilog:
  ret void
}
