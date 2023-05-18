; RUN: llc -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx900 -filetype=obj -o %t.o < %s && llvm-readobj -r %t.o | FileCheck --check-prefix=ELF %s

; RUN: llc -global-isel -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -global-isel -mtriple=amdgcn--amdpal -mcpu=gfx900 -filetype=obj -o %t.o < %s && llvm-readobj -r %t.o | FileCheck --check-prefix=ELF %s

; GCN-LABEL: {{^}}ps_main:
; GCN: v_mov_b32_{{.*}} v[[relocreg:[0-9]+]], doff_0_0_b@abs32@lo
; GCN-NEXT: exp {{.*}} v[[relocreg]], {{.*}}
; GCN-NEXT: s_endpgm
; GCN-NEXT: .Lfunc_end

; ELF: Relocations [
; ELF-NEXT: Section (3) .rel.text {
; ELF-NEXT: 0x{{[0-9]+}} R_AMDGPU_ABS32_LO doff_0_0_b{{$}}

define amdgpu_ps void @ps_main(i32 %arg, i32 inreg %arg1, i32 inreg %arg2) local_unnamed_addr #0 {
  %rc = call i32 @llvm.amdgcn.reloc.constant(metadata !1)
  %rcf = bitcast i32 %rc to float
  call void @llvm.amdgcn.exp.f32(i32 immarg 40, i32 immarg 15, float %rcf, float undef, float undef, float undef, i1 immarg false, i1 immarg false) #0
  ret void
}

; Function Attrs: inaccessiblememonly nounwind
declare void @llvm.amdgcn.exp.f32(i32 immarg, i32 immarg, float, float, float, float, i1 immarg, i1 immarg) #1

; Function Attrs: nounwind readnone speculatable
declare i32 @llvm.amdgcn.reloc.constant(metadata) #2

attributes #0 = { nounwind }
attributes #1 = { inaccessiblememonly nounwind }
attributes #2 = { nounwind readnone speculatable }

!1 = !{!"doff_0_0_b"}
