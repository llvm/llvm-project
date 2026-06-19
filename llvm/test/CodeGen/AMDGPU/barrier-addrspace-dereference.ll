; Check we cannot dereference a barrier GV.

; RUN: not --crash llc -O0 -global-isel=0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 < %s                       2>&1  | FileCheck -check-prefixes=DAGISEL %s
; RUN: not         llc -O0 -global-isel=1 -new-reg-bank-select -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 < %s  2>&1  | FileCheck -check-prefixes=GISEL %s

; TODO: It'd be nicer to have a Verifier diagnostic for this.

; DAGISEL: LLVM ERROR: {{.*}} store<(store (s32) into @bar, addrspace 15)>
; GISEL:   LLVM ERROR: {{.*}} G_LOAD %6:sgpr(p15) :: (load (s32) from @bar, addrspace 15) (in function: func1)
@bar = internal addrspace(15) global target("amdgcn.named.barrier", 0) poison

define amdgpu_kernel void @func1() {
  %val = load i32, ptr addrspace(15) @bar
  store i32 %val, ptr addrspace(15) @bar
  ret void
}
