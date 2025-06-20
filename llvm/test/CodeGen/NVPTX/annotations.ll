; RUN: llc < %s -mtriple=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas && !ptxas-12.0 %{ llc < %s -mtriple=nvptx -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

@texture = internal addrspace(1) global i64 0, align 8
; CHECK: .global .texref texture
@surface = internal addrspace(1) global i64 0, align 8
; CHECK: .global .surfref surface

; CHECK: .entry kernel_func_maxntid
define ptx_kernel void @kernel_func_maxntid(ptr %a) "nvvm.maxntid"="10,20,30" {
; CHECK: .maxntid 10, 20, 30
; CHECK: ret
  ret void
}

; CHECK: .entry kernel_func_reqntid
define ptx_kernel void @kernel_func_reqntid(ptr %a) "nvvm.reqntid"="11,22,33" {
; CHECK: .reqntid 11, 22, 33
; CHECK: ret
  ret void
}

; CHECK: .entry kernel_func_minctasm
define ptx_kernel void @kernel_func_minctasm(ptr %a) "nvvm.minctasm"="42" {
; CHECK: .minnctapersm 42
; CHECK: ret
  ret void
}

; CHECK-LABEL: .entry kernel_func_maxnreg
define ptx_kernel void @kernel_func_maxnreg() "nvvm.maxnreg"="1234" {
; CHECK: .maxnreg 1234
; CHECK: ret
  ret void
}

!nvvm.annotations = !{!9, !10}

!9 = !{ptr addrspace(1) @texture, !"texture", i32 1}
!10 = !{ptr addrspace(1) @surface, !"surface", i32 1}
