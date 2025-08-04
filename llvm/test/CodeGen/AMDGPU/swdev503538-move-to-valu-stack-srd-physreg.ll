; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -verify-machineinstrs=0 -O0 2> %t.err < %s | FileCheck %s
; RUN: FileCheck -check-prefix=ERR %s < %t.err

; FIXME: This error will be fixed by supporting arbitrary divergent
; dynamic allocas by performing a wave umax of the size.

; ERR: error: <unknown>:0:0: in function move_to_valu_assert_srd_is_physreg_swdev503538 i32 (ptr addrspace(1)): illegal VGPR to SGPR copy

; CHECK: ; illegal copy v0 to s32

define i32 @move_to_valu_assert_srd_is_physreg_swdev503538(ptr addrspace(1) %ptr) {
entry:
  %idx = load i32, ptr addrspace(1) %ptr, align 4
  %zero = extractelement <4 x i32> zeroinitializer, i32 %idx
  %alloca = alloca [2048 x i8], i32 %zero, align 8, addrspace(5)
  %ld = load i32, ptr addrspace(5) %alloca, align 8
  call void @llvm.memset.p5.i32(ptr addrspace(5) %alloca, i8 0, i32 2048, i1 false)
  ret i32 %ld
}

declare void @llvm.memset.p5.i32(ptr addrspace(5) nocapture writeonly, i8, i32, i1 immarg) #0

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: write) }
