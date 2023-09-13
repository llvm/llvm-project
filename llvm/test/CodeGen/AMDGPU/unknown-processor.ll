; RUN: llc -mtriple=amdgcn -mtriple=amdgcn-- -mcpu=unknown -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefix=ERROR -check-prefix=GCN %s
; RUN: llc -march=r600 -mtriple=r600-- -mcpu=unknown -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefix=ERROR -check-prefix=R600 %s
target datalayout = "A5"

; Should not crash when the processor is not recognized and the
; wavefront size feature not set.

; Should also not have fragments of r600 and gcn isa mixed.

; ERROR: 'unknown' is not a recognized processor for this target (ignoring processor)

; GCN-NOT: MOV
; GCN: buffer_store_dword
; GCN: ScratchSize: 8{{$}}

; R600: MOV
define amdgpu_kernel void @foo() {
  %alloca = alloca i32, align 4, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %alloca
  ret void
}
