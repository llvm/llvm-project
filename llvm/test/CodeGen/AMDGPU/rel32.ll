; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck %s

@g = protected local_unnamed_addr addrspace(4) externally_initialized global i32 0, align 4

; CHECK-LABEL: rel32_neg_offset:
; CHECK: s_getpc_b64 s{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]{{]}}
; CHECK-NEXT: s_add_u32 s[[LO]], s[[LO]], g@rel32@lo-4
; CHECK-NEXT: s_addc_u32 s[[HI]], s[[HI]], g@rel32@hi+4
define i32 addrspace(4)* @rel32_neg_offset() {
  %r = getelementptr i32, i32 addrspace(4)* @g, i64 -2
  ret i32 addrspace(4)* %r
}
