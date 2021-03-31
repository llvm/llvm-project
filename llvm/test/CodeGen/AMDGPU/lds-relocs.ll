; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx900 -verify-machineinstrs -show-mc-encoding < %s | FileCheck -check-prefixes=GCN %s
; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx900 -filetype=obj < %s | llvm-readobj -r -t - | FileCheck -check-prefixes=ELF %s

@lds.external = external unnamed_addr addrspace(3) global [0 x i32]
@lds.defined = unnamed_addr addrspace(3) global [8 x i32] undef, align 8

; ELF:      Relocations [
; ELF-NEXT:   Section (3) .rel.text {
; ELF-NEXT:     0x{{[0-9a-f]*}} R_AMDGPU_ABS32 lds.external
; ELF-NEXT:     0x{{[0-9a-f]*}} R_AMDGPU_ABS32 lds.defined
; ELF-NEXT:   }
; ELF-NEXT: ]

; ELF:      Symbol {
; ELF:        Name: lds.external
; ELF-NEXT:   Value: 0x4
; ELF-NEXT:   Size: 0
; ELF-NEXT:   Binding: Global (0x1)
; ELF-NEXT:   Type: Object (0x1)
; ELF-NEXT:   Other: 0
; ELF-NEXT:   Section: Processor Specific (0xFF00)
; ELF-NEXT: }

; ELF:      Symbol {
; ELF:        Name: lds.defined
; ELF-NEXT:   Value: 0x8
; ELF-NEXT:   Size: 32
; ELF-NEXT:   Binding: Global (0x1)
; ELF-NEXT:   Type: Object (0x1)
; ELF-NEXT:   Other: 0
; ELF-NEXT:   Section: Processor Specific (0xFF00)
; ELF-NEXT: }

; GCN-LABEL: {{^}}test_basic:
; GCN: v_mov_b32_e32 v1, lds.external@abs32@lo ; encoding: [0xff,0x02,0x02,0x7e,A,A,A,A]
; GCN-NEXT:              ; fixup A - offset: 4, value: lds.external@abs32@lo, kind: FK_Data_4{{$}}
;
; GCN: s_add_i32 s0, s0, lds.defined@abs32@lo ; encoding: [0x00,0xff,0x00,0x81,A,A,A,A]
; GCN-NEXT:          ; fixup A - offset: 4, value: lds.defined@abs32@lo, kind: FK_Data_4{{$}}
;
; GCN: .globl lds.external
; GCN: .amdgpu_lds lds.external, 0, 4
; GCN: .globl lds.defined
; GCN: .amdgpu_lds lds.defined, 32, 8
define amdgpu_gs float @test_basic(i32 inreg %wave, i32 %arg1) #0 {
main_body:
  %gep0 = getelementptr [0 x i32], [0 x i32] addrspace(3)* @lds.external, i32 0, i32 %arg1
  %tmp = load i32, i32 addrspace(3)* %gep0

  %gep1 = getelementptr [8 x i32], [8 x i32] addrspace(3)* @lds.defined, i32 0, i32 %wave
  store i32 123, i32 addrspace(3)* %gep1

  %r = bitcast i32 %tmp to float
  ret float %r
}

; Function Attrs: convergent nounwind readnone
declare i64 @llvm.amdgcn.icmp.i64.i32(i32, i32, i32) #4

attributes #0 = { "no-signed-zeros-fp-math"="true" }
attributes #4 = { convergent nounwind readnone }
