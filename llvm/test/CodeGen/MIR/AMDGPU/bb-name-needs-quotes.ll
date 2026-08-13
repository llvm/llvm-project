; Block names that are not plain identifiers must be quoted, otherwise llc
; cannot parse back the MIR it just wrote.

; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -stop-before=greedy -o %t.mir %s
; RUN: FileCheck %s < %t.mir
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -x mir -start-before=greedy -filetype=null %t.mir

; CHECK: bb.0.entry:
; CHECK: bb.1.", bb71":
; CHECK: bb.2."file f.f90, line 12, bb99":

define amdgpu_kernel void @k(ptr addrspace(1) %o, i32 %n) {
entry:
  %c = icmp sgt i32 %n, 0
  br i1 %c, label %", bb71", label %"file f.f90, line 12, bb99"

", bb71":
  store i32 1, ptr addrspace(1) %o, align 4
  br label %"file f.f90, line 12, bb99"

"file f.f90, line 12, bb99":
  store i32 2, ptr addrspace(1) %o, align 4
  ret void
}
