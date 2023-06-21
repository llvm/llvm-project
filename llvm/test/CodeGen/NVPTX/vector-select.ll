; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas && !ptxas-12.0 %{ llc < %s -march=nvptx -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; This test makes sure that vector selects are scalarized by the type legalizer.
; If not, type legalization will fail.

; CHECK-LABEL: .visible .func foo(
define void @foo(ptr addrspace(1) %def_a, ptr addrspace(1) %def_b, ptr addrspace(1) %def_c) {
entry:
; CHECK:  ld.global.v2.u32
; CHECK:  ld.global.v2.u32
; CHECK:  ld.global.v2.u32
  %tmp4 = load <2 x i32>, ptr addrspace(1) %def_a
  %tmp6 = load <2 x i32>, ptr addrspace(1) %def_c
  %tmp8 = load <2 x i32>, ptr addrspace(1) %def_b
; CHECK:  setp.gt.s32
; CHECK:  setp.gt.s32
  %0 = icmp sge <2 x i32> %tmp4, zeroinitializer
; CHECK:  selp.b32
; CHECK:  selp.b32
  %cond = select <2 x i1> %0, <2 x i32> %tmp6, <2 x i32> %tmp8
; CHECK:  st.global.v2.u32
  store <2 x i32> %cond, ptr addrspace(1) %def_c
  ret void
}
