; RUN: not llc -global-isel=1 -new-reg-bank-select -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -o - < %s 2>&1 | FileCheck -check-prefix=GISEL-ERR %s

; GISEL-ERR: LLVM ERROR: cannot select: %{{[0-9]+}}:vgpr_32(s32) = G_LOAD %{{[0-9]+}}:vgpr(p8) :: (load (s32) from %ir.rsrc, addrspace 8)

define float @load_from_rsrc(ptr addrspace(8) %rsrc) {
body:
  %res = load float, ptr addrspace(8) %rsrc
  ret float %res
}
