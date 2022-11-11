; RUN: llc -global-isel=0 -mtriple=amdgcn-amd-amdhsa --amdhsa-code-object-version=2 < %s | FileCheck -check-prefix=ALL -check-prefix=HSA %s
; RUN: llc -global-isel=1 -mtriple=amdgcn-amd-amdhsa --amdhsa-code-object-version=2 < %s | FileCheck -check-prefix=ALL -check-prefix=HSA %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=ALL -check-prefix=EG %s

; This test makes sure we do not double count global values when they are
; used in different basic blocks.

; GCN: .long 47180
; GCN-NEXT: .long 32900

; EG: .long 166120
; EG-NEXT: .long 1
; ALL: {{^}}test:

; HSA: granulated_lds_size = 0
; HSA: workgroup_group_segment_byte_size = 4

; GCN: ; LDSByteSize: 4 bytes/workgroup (compile time only)
@lds = internal unnamed_addr addrspace(3) global i32 undef, align 4

define amdgpu_kernel void @test(i32 addrspace(1)* %out, i32 %cond) {
entry:
  %0 = icmp eq i32 %cond, 0
  br i1 %0, label %if, label %else

if:
  store i32 1, i32 addrspace(3)* @lds
  br label %endif

else:
  store i32 2, i32 addrspace(3)* @lds
  br label %endif

endif:
  ret void
}
