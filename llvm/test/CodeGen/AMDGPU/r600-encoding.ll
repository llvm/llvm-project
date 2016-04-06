; RUN: llc < %s -march=r600 -show-mc-encoding -mcpu=redwood | FileCheck --check-prefix=EG %s
; RUN: llc < %s -march=r600 -show-mc-encoding -mcpu=rs880 | FileCheck --check-prefix=R600 %s

; The earliest R600 GPUs have a slightly different encoding than the rest of
; the VLIW4/5 GPUs.

; EG: {{^}}test:
; EG: MUL_IEEE {{[ *TXYZWPVxyzw.,0-9]+}} ; encoding: [{{0x[0-9a-f]+,0x[0-9a-f]+,0x[0-9a-f]+,0x[0-9a-f]+,0x10,0x01,0x[0-9a-f]+,0x[0-9a-f]+}}]

; R600: {{^}}test:
; R600: MUL_IEEE {{[ *TXYZWPVxyzw.,0-9]+}} ; encoding: [{{0x[0-9a-f]+,0x[0-9a-f]+,0x[0-9a-f]+,0x[0-9a-f]+,0x10,0x02,0x[0-9a-f]+,0x[0-9a-f]+}}]

define amdgpu_ps void @test(<4 x float> inreg %reg0) {
entry:
  %r0 = extractelement <4 x float> %reg0, i32 0
  %r1 = extractelement <4 x float> %reg0, i32 1
  %r2 = fmul float %r0, %r1
  %vec = insertelement <4 x float> undef, float %r2, i32 0
  call void @llvm.R600.store.swizzle(<4 x float> %vec, i32 0, i32 0)
  ret void
}

declare void @llvm.R600.store.swizzle(<4 x float>, i32, i32)
