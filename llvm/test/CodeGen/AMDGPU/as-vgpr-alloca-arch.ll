; "VGPR as memory" (addrspace(13)) is only enabled on gfx942/gfx950 (CDNA3+)
; and gfx12xx/gfx13xx. On a supported target the object is kept in addrspace(13)
; (and lowered to VGPRs); on any other target it falls back to addrspace(5)
; scratch.

; RUN: opt -S -mtriple=amdgcn -mcpu=gfx942  -passes=amdgpu-vgpr-allocate %s 2>/dev/null | FileCheck %s --check-prefix=SUPP
; RUN: opt -S -mtriple=amdgcn -mcpu=gfx950  -passes=amdgpu-vgpr-allocate %s 2>/dev/null | FileCheck %s --check-prefix=SUPP
; RUN: opt -S -mtriple=amdgcn -mcpu=gfx1200 -passes=amdgpu-vgpr-allocate %s 2>/dev/null | FileCheck %s --check-prefix=SUPP
; RUN: opt -S -mtriple=amdgcn -mcpu=gfx1310 -passes=amdgpu-vgpr-allocate %s 2>/dev/null | FileCheck %s --check-prefix=SUPP
; RUN: opt -S -mtriple=amdgcn -mcpu=gfx90a  -passes=amdgpu-vgpr-allocate %s 2>/dev/null | FileCheck %s --check-prefix=UNSUPP
; RUN: opt -S -mtriple=amdgcn -mcpu=gfx1030 -passes=amdgpu-vgpr-allocate %s 2>/dev/null | FileCheck %s --check-prefix=UNSUPP
; RUN: opt -S -mtriple=amdgcn -mcpu=gfx1100 -passes=amdgpu-vgpr-allocate %s 2>/dev/null | FileCheck %s --check-prefix=UNSUPP

define void @vgpr_obj() {
; SUPP:   alloca [4 x i32], align 4, addrspace(13), !amdgpu.allocated.vgprs
; UNSUPP: alloca [4 x i32], align 4, addrspace(5){{$}}
  %a = alloca [4 x i32], align 4, addrspace(13)
  store i32 0, ptr addrspace(13) %a
  ret void
}
