; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -stop-after=finalize-isel < %s | FileCheck %s

; This test verifies that instruction selection will propagate alias metadata
; to split loads and stores.

; CHECK:      %{{[0-9]+}}:vreg_128 = DS_READ_B128_gfx9 {{.*}} :: (load (s128) from %{{.*}}, align 32, !alias.scope ![[IN:[0-9]+]], !noalias ![[OUT:[0-9]+]], addrspace 3)
; CHECK-NEXT: %{{[0-9]+}}:vreg_128 = DS_READ_B128_gfx9 {{.*}} :: (load (s128) from %{{.*}}, !alias.scope ![[IN]], !noalias ![[OUT]], addrspace 3)
; CHECK-NEXT: %{{[0-9]+}}:vreg_128 = DS_READ_B128_gfx9 {{.*}} :: (load (s128) from %{{.*}}, align 32, !alias.scope ![[IN]], !noalias ![[OUT]], addrspace 3)
; CHECK-NEXT: %{{[0-9]+}}:vreg_128 = DS_READ_B128_gfx9 {{.*}} :: (load (s128) from %{{.*}}, !alias.scope ![[IN]], !noalias ![[OUT]], addrspace 3)
; CHECK:      DS_WRITE_B128_gfx9 {{.*}} :: (store (s128) into %{{.*}}, !alias.scope ![[OUT]], !noalias ![[IN]], addrspace 3)
; CHECK-NEXT: %{{[0-9]+}}:vreg_128 = REG_SEQUENCE
; CHECK-NEXT: DS_WRITE_B128_gfx9 {{.*}} :: (store (s128) into %{{.*}}, !alias.scope ![[OUT]], !noalias ![[IN]], addrspace 3)
; CHECK-NEXT: %{{[0-9]+}}:vreg_128 = REG_SEQUENCE
; CHECK-NEXT: DS_WRITE_B128_gfx9 {{.*}} :: (store (s128) into %{{.*}}, !alias.scope ![[OUT]], !noalias ![[IN]], addrspace 3)
; CHECK-NEXT: %{{[0-9]+}}:vreg_128 = REG_SEQUENCE
; CHECK-NEXT: DS_WRITE_B128_gfx9 {{.*}} :: (store (s128) into %{{.*}}, !alias.scope ![[OUT]], !noalias ![[IN]], addrspace 3)

define amdgpu_kernel void @test(ptr addrspace(3) noalias %in, ptr addrspace(3) noalias %out) {
  %idx = call i32 @llvm.amdgcn.workitem.id.x()
  %in.addr = getelementptr <16 x float>, ptr addrspace(3) %in, i32 %idx
  %val.0 = load <16 x float>, ptr addrspace(3) %in.addr, align 32, !alias.scope !4, !noalias !5
  %val.1 = call <16 x float> @llvm.amdgcn.wmma.f32.16x16x16.f32.v16f32.v16f32(<16 x float> %val.0, <16 x float> %val.0, <16 x float> %val.0, i1 false)
  %out.addr = getelementptr <16 x float>, ptr addrspace(3) %out, i32 %idx
  store <16 x float> %val.1, ptr addrspace(3) %out.addr, align 32, !alias.scope !5, !noalias !4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare <16 x float> @llvm.amdgcn.wmma.f32.16x16x16.f32.v16f32.v16f32(<16 x float>, <16 x float>, <16 x float>, i1 immarg)

!0 = !{!"inout.domain"}
!1 = !{!"in.scope", !0}
!2 = !{!"out.scope", !0}
!4 = !{!1}
!5 = !{!2}
