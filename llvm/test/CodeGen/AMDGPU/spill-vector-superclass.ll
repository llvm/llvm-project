; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -stop-after=greedy,1 -verify-machineinstrs -o - %s | FileCheck -check-prefix=GCN %s
; Convert AV spills into VGPR spills by introducing appropriate copies in between.

define amdgpu_kernel void @test_spill_av_class(<4 x i32> %arg) #0 {
  ; GCN-LABEL: name: test_spill_av_class
  ; GCN:   INLINEASM &"; def $0", 1 /* sideeffect attdialect */, 2752522 /* regdef:VGPR_32 */, def undef %22.sub0
  ; GCN-NEXT:   undef [[AV_REG:%[0-9]+]].sub0:av_64 = COPY %{{[0-9]+}}.sub0
  ; GCN-NEXT:   SI_SPILL_AV64_SAVE [[AV_REG]], %stack.0, $sgpr32, 0, implicit $exec
  ; GCN:   [[SI_SPILL_AV64_RESTORE:%[0-9]+]]:av_64 = SI_SPILL_AV64_RESTORE %stack.0, $sgpr32, 0, implicit $exec
  ; GCN-NEXT:   undef %23.sub0:vreg_64 = COPY [[SI_SPILL_AV64_RESTORE]].sub0
  %v0 = call i32 asm sideeffect "; def $0", "=v"()
  %tmp = insertelement <2 x i32> undef, i32 %v0, i32 0
  %mai = tail call <4 x i32> @llvm.amdgcn.mfma.i32.4x4x4i8(i32 1, i32 2, <4 x i32> %arg, i32 0, i32 0, i32 0)
  store volatile <4 x i32> %mai, <4 x i32> addrspace(1)* undef
  call void asm sideeffect "; use $0", "v"(<2 x i32> %tmp);
  ret void
}

declare <4 x i32> @llvm.amdgcn.mfma.i32.4x4x4i8(i32, i32, <4 x i32>, i32, i32, i32)

attributes #0 = { nounwind "amdgpu-num-vgpr"="5" }
