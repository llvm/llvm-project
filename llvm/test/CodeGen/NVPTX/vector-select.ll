; RUN: llc < %s -march=nvptx -mcpu=sm_20 %if ptxas %{ | %ptxas-verify %}
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 %if ptxas %{ | %ptxas-verify %}

; This test makes sure that vector selects are scalarized by the type legalizer.
; If not, type legalization will fail.

define void @foo(<2 x i32> addrspace(1)* %def_a, <2 x i32> addrspace(1)* %def_b, <2 x i32> addrspace(1)* %def_c) {
entry:
  %tmp4 = load <2 x i32>, <2 x i32> addrspace(1)* %def_a
  %tmp6 = load <2 x i32>, <2 x i32> addrspace(1)* %def_c
  %tmp8 = load <2 x i32>, <2 x i32> addrspace(1)* %def_b
  %0 = icmp sge <2 x i32> %tmp4, zeroinitializer
  %cond = select <2 x i1> %0, <2 x i32> %tmp6, <2 x i32> %tmp8
  store <2 x i32> %cond, <2 x i32> addrspace(1)* %def_c
  ret void
}
