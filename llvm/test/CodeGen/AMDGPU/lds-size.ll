; RUN: llc -global-isel=0 -mtriple=amdgcn-amd-amdhsa < %s | FileCheck -check-prefix=ALL -check-prefix=HSA %s
; RUN: llc -global-isel=1 -mtriple=amdgcn-amd-amdhsa < %s | FileCheck -check-prefix=ALL -check-prefix=HSA %s
; RUN: llc -mtriple=r600 -mcpu=redwood < %s | FileCheck -check-prefix=ALL -check-prefix=EG %s

; This test makes sure we do not double count global values when they are
; used in different basic blocks.

; GCN: .long 47180
; GCN-NEXT: .long 32900

; EG: .long 166120
; EG-NEXT: .long 1
; ALL: {{^}}test:

; HSA-NOT: COMPUTE_PGM_RSRC2.LDS_SIZE
; HSA: .amdhsa_group_segment_fixed_size 4

; GCN: ; LDSByteSize: 4 bytes/workgroup (compile time only)
@lds = internal unnamed_addr addrspace(3) global i32 undef, align 4

define amdgpu_kernel void @test(ptr addrspace(1) %out, i32 %cond) {
entry:
  %0 = icmp eq i32 %cond, 0
  br i1 %0, label %if, label %else

if:
  store i32 1, ptr addrspace(3) @lds
  br label %endif

else:
  store i32 2, ptr addrspace(3) @lds
  br label %endif

endif:
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 400}
