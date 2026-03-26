; RUN: llc -mcpu=gfx1200 < %s | FileCheck %s
target triple = "amdgcn--amdpal"

@global = addrspace(1) global i32 poison, align 4

; CHECK-LABEL: amdpal.pipelines:

; Shouldn't report the part of %vgpr_args that's not used
; CHECK-LABEL: entry_point_symbol: cs_calling_chain
; CHECK: .vgpr_count:     0xa
define amdgpu_cs void @cs_calling_chain(i32 %vgpr, i32 inreg %sgpr) {
  %vgpr_args = insertvalue {i32, i32, i32, i32} poison, i32 %vgpr, 1
  call void (ptr, i32, i32, {i32, i32, i32, i32}, i32, ...) @llvm.amdgcn.cs.chain.p0.i32.i32.s(
    ptr @chain_func, i32 0, i32 inreg %sgpr, {i32, i32, i32, i32} %vgpr_args, i32 0)
  unreachable
}

; Neither uses not writes a VGPR
; CHECK-LABEL: chain_func:
; CHECK: .vgpr_count:     0x1
define amdgpu_cs_chain void @chain_func([32 x i32] %args) {
entry:
  call void (ptr, i32, {}, [32 x i32], i32, ...) @llvm.amdgcn.cs.chain.p0.i32.s.a(
        ptr @chain_func, i32 0, {} inreg {}, [32 x i32] %args, i32 0)
  unreachable
}
