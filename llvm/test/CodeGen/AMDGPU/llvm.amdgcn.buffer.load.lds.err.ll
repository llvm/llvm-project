; RUN: split-file %s %t
;
; RUN: not --crash llc -mtriple=amdgcn -mcpu=gfx1100 %t/struct.ll 2>&1 | FileCheck --ignore-case %s
; RUN: not llc -global-isel -mtriple=amdgcn -mcpu=gfx1100 %t/struct.ll 2>&1 | FileCheck --ignore-case %s
; RUN: not --crash llc -mtriple=amdgcn -mcpu=gfx1100 %t/struct.ptr.ll 2>&1 | FileCheck --ignore-case --check-prefix=LEGALIZER-FAIL %s
; RUN: not llc -global-isel -mtriple=amdgcn -mcpu=gfx1100 %t/struct.ptr.ll 2>&1 | FileCheck --ignore-case %s
; RUN: not --crash llc -mtriple=amdgcn -mcpu=gfx1100 %t/raw.ll 2>&1 | FileCheck --ignore-case %s
; RUN: not llc -global-isel -mtriple=amdgcn -mcpu=gfx1100 %t/raw.ll 2>&1 | FileCheck --ignore-case %s
; RUN: not --crash llc -mtriple=amdgcn -mcpu=gfx1100 %t/raw.ptr.ll 2>&1 | FileCheck --ignore-case --check-prefix=LEGALIZER-FAIL %s
; RUN: not llc -global-isel -mtriple=amdgcn -mcpu=gfx1100 %t/raw.ptr.ll 2>&1 | FileCheck --ignore-case %s
;
; CHECK: LLVM ERROR: Cannot select
; LEGALIZER-FAIL: Do not know how to expand this operator's operand!

;--- struct.ll
define amdgpu_ps void @buffer_load_lds(<4 x i32> inreg %rsrc, ptr addrspace(3) inreg %lds) {
  call void @llvm.amdgcn.struct.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) %lds, i32 4, i32 0, i32 0, i32 0, i32 0, i32 0)
  ret void
}

;--- struct.ptr.ll
define amdgpu_ps void @buffer_load_lds(ptr addrspace(8) inreg %rsrc, ptr addrspace(3) inreg %lds) {
  call void @llvm.amdgcn.struct.ptr.buffer.load.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) %lds, i32 4, i32 0, i32 0, i32 0, i32 0, i32 0)
  ret void
}

;--- raw.ll
define amdgpu_ps void @buffer_load_lds(<4 x i32> inreg %rsrc, ptr addrspace(3) inreg %lds) {
  call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) %lds, i32 4, i32 0, i32 0, i32 0, i32 0)
  ret void
}

;--- raw.ptr.ll
define amdgpu_ps void @buffer_load_lds(ptr addrspace(8) inreg %rsrc, ptr addrspace(3) inreg %lds) {
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) %lds, i32 4, i32 0, i32 0, i32 0, i32 0)
  ret void
}
