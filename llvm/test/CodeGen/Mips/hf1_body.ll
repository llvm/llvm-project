; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel -mattr=mips16 \
; RUN:     -relocation-model=pic -no-integrated-as < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,GAS

; The integrated assembler expands assembly macros before printing.
; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel -mattr=mips16 \
; RUN:     -relocation-model=pic < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,IAS

@x = external global float

; Function Attrs: nounwind
define void @v_sf(float %p) #0 {
entry:
  %p.addr = alloca float, align 4
  store float %p, ptr %p.addr, align 4
  %0 = load float, ptr %p.addr, align 4
  store float %0, ptr @x, align 4
  ret void
}
; ALL-LABEL: .ent __fn_stub_v_sf
; ALL:       .cpload $25
; ALL:       .set reorder
; ALL:       .reloc 0, R_MIPS_NONE, v_sf
; GAS:       la $25, $__fn_local_v_sf
; IAS:       lw $25, %got($__fn_local_v_sf)($gp)
; IAS:       addiu $25, $25, %lo($__fn_local_v_sf)
; ALL:       mfc1 $4, $f12
; ALL:       jr $25
; ALL:       .end __fn_stub_v_sf
