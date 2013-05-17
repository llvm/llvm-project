; RUN: llc < %s -march=r600 -show-mc-encoding -mcpu=redwood | FileCheck --check-prefix=EG-CHECK %s
; RUN: llc < %s -march=r600 -show-mc-encoding -mcpu=rs880 | FileCheck --check-prefix=R600-CHECK %s

; The earliest R600 GPUs have a slightly different encoding than the rest of
; the VLIW4/5 GPUs.

; EG-CHECK: @test
; EG-CHECK: MUL_IEEE {{[ *TXYZW.,0-9]+}} ; encoding: [{{0x[0-9a-f]+,0x[0-9a-f]+,0x[0-9a-f]+,0x[0-9a-f]+,0x10,0x01,0x[0-9a-f]+,0x[0-9a-f]+}}]

; R600-CHECK: @test
; R600-CHECK: MUL_IEEE {{[ *TXYZW.,0-9]+}} ; encoding: [{{0x[0-9a-f]+,0x[0-9a-f]+,0x[0-9a-f]+,0x[0-9a-f]+,0x10,0x02,0x[0-9a-f]+,0x[0-9a-f]+}}]

define void @test() {
entry:
  %0 = call float @llvm.R600.load.input(i32 0)
  %1 = call float @llvm.R600.load.input(i32 1)
  %2 = fmul float %0, %1
  call void @llvm.AMDGPU.store.output(float %2, i32 0)
  ret void
}

declare float @llvm.R600.load.input(i32) readnone

declare void @llvm.AMDGPU.store.output(float, i32)
