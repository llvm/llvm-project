; RUN: llc -march=amdgcn -mcpu=gfx900 -global-isel=0 -verify-machineinstrs -stop-after=finalize-isel -o - %s | FileCheck -check-prefixes=GCN,SDAG %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -global-isel=1 -verify-machineinstrs -stop-after=finalize-isel -o - %s | FileCheck -check-prefixes=GCN,GISEL %s

@0 = external dso_local addrspace(4) constant [4 x <2 x float>]
@1 = external dso_local addrspace(4) constant i32

; Test that DAG->DAG ISel is able to pick up the S_LOAD_DWORDX4_SGPR instruction that fetches the offset
; from a register.
; GCN-LABEL: name: test_load_zext
; GCN: %[[OFFSET:[0-9]+]]:sreg_32 = S_MOV_B32 target-flags(amdgpu-abs32-lo) @DescriptorBuffer
; SDAG: %{{[0-9]+}}:sgpr_128 = S_LOAD_DWORDX4_SGPR_IMM killed %{{[0-9]+}}, killed %[[OFFSET]], 0, 0 :: (invariant load (s128) from %ir.12, addrspace 4)
; GISEL: %{{[0-9]+}}:sgpr_128 = S_LOAD_DWORDX4_SGPR_IMM %{{[0-9]+}}, %[[OFFSET]], 0, 0 :: (invariant load (<4 x s32>) from {{.*}}, addrspace 4)
define amdgpu_cs void @test_load_zext(i32 inreg %0, i32 inreg %1, i32 inreg %resNode0, i32 inreg %resNode1, <3 x i32> inreg %2, i32 inreg %3, <3 x i32> %4) local_unnamed_addr #2 {
.entry:
  %5 = call i64 @llvm.amdgcn.s.getpc() #3
  %6 = bitcast i64 %5 to <2 x i32>
  %7 = insertelement <2 x i32> %6, i32 %resNode0, i32 0
  %8 = bitcast <2 x i32> %7 to i64
  %9 = inttoptr i64 %8 to ptr addrspace(4)
  %10 = call i32 @llvm.amdgcn.reloc.constant(metadata !4)
  %11 = zext i32 %10 to i64
  %12 = getelementptr [4294967295 x i8], ptr addrspace(4) %9, i64 0, i64 %11
  %13 = load <4 x i32>, ptr addrspace(4) %12, align 16, !invariant.load !5
  %14 = call <4 x i32> @llvm.amdgcn.s.buffer.load.v4i32(<4 x i32> %13, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.store.v4i32(<4 x i32> %14, <4 x i32> %13, i32 0, i32 0, i32 0)
  ret void
}

; Make sure we match constant bases with register offests, in which case
; the base may be the RHS operand of the load in SDAG.
; GCN-LABEL: name: test_complex_reg_offset
; GCN-DAG: %[[BASE:.*]]:sreg_64 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-rel32-lo) @0 + 4,
; GCN-DAG: %[[OFFSET:.*]]:sreg_32 = S_LSHL_B32
; SDAG: S_LOAD_DWORD_SGPR_IMM killed %[[BASE]], killed %[[OFFSET]], 0, 0
; GISEL: S_LOAD_DWORD_SGPR_IMM %[[BASE]], %[[OFFSET]], 0, 0
define amdgpu_ps void @test_complex_reg_offset(ptr addrspace(1) %out) {
  %i = load i32, ptr addrspace(4) @1
  %i1 = and i32 %i, 3
  %i2 = zext i32 %i1 to i64
  %i3 = getelementptr [4 x <2 x float>], ptr addrspace(4) @0, i64 0, i64 %i2, i64 0
  %i4 = load float, ptr addrspace(4) %i3, align 4
  store float %i4, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: name: test_sgpr_plus_imm_offset
; SDAG-DAG: %[[BASE0:.*]]:sgpr_32 = PRED_COPY $sgpr0
; SDAG-DAG: %[[BASE1:.*]]:sgpr_32 = PRED_COPY $sgpr1
; SDAG-DAG: %[[OFFSET:.*]]:sgpr_32 = PRED_COPY $sgpr2
; SDAG-DAG: %[[BASE:.*]]:sgpr_64 = REG_SEQUENCE %[[BASE0]], %subreg.sub0, %[[BASE1]], %subreg.sub1
; SDAG: S_LOAD_DWORD_SGPR_IMM killed %[[BASE]], %[[OFFSET]], 16,
; GISEL-DAG: %[[BASE0:.*]]:sreg_32 = PRED_COPY $sgpr0
; GISEL-DAG: %[[BASE1:.*]]:sreg_32 = PRED_COPY $sgpr1
; GISEL-DAG: %[[OFFSET:.*]]:sreg_32 = PRED_COPY $sgpr2
; GISEL-DAG: %[[BASE:.*]]:sreg_64 = REG_SEQUENCE %[[BASE0]], %subreg.sub0, %[[BASE1]], %subreg.sub1
; GISEL: S_LOAD_DWORD_SGPR_IMM %[[BASE]], %[[OFFSET]], 16,
define amdgpu_ps void @test_sgpr_plus_imm_offset(ptr addrspace(4) inreg %base, i32 inreg %offset,
                                                 ptr addrspace(1) inreg %out) {
  %v1 = getelementptr i8, ptr addrspace(4) %base, i64 16
  %v2 = zext i32 %offset to i64
  %v3 = getelementptr i8, ptr addrspace(4) %v1, i64 %v2
  %v5 = load i32, ptr addrspace(4) %v3, align 4
  store i32 %v5, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: name: test_sgpr_plus_imm_offset_x2
; SDAG-DAG: %[[BASE0:.*]]:sgpr_32 = PRED_COPY $sgpr0
; SDAG-DAG: %[[BASE1:.*]]:sgpr_32 = PRED_COPY $sgpr1
; SDAG-DAG: %[[OFFSET:.*]]:sgpr_32 = PRED_COPY $sgpr2
; SDAG-DAG: %[[BASE:.*]]:sgpr_64 = REG_SEQUENCE %[[BASE0]], %subreg.sub0, %[[BASE1]], %subreg.sub1
; SDAG: S_LOAD_DWORDX2_SGPR_IMM killed %[[BASE]], %[[OFFSET]], 16,
; GISEL-DAG: %[[BASE0:.*]]:sreg_32 = PRED_COPY $sgpr0
; GISEL-DAG: %[[BASE1:.*]]:sreg_32 = PRED_COPY $sgpr1
; GISEL-DAG: %[[OFFSET:.*]]:sreg_32 = PRED_COPY $sgpr2
; GISEL-DAG: %[[BASE:.*]]:sreg_64 = REG_SEQUENCE %[[BASE0]], %subreg.sub0, %[[BASE1]], %subreg.sub1
; GISEL: S_LOAD_DWORDX2_SGPR_IMM %[[BASE]], %[[OFFSET]], 16,
define amdgpu_ps void @test_sgpr_plus_imm_offset_x2(ptr addrspace(4) inreg %base, i32 inreg %offset,
                                                    ptr addrspace(1) inreg %out) {
  %v1 = getelementptr i8, ptr addrspace(4) %base, i64 16
  %v2 = zext i32 %offset to i64
  %v3 = getelementptr i8, ptr addrspace(4) %v1, i64 %v2
  %v5 = load <2 x i32>, ptr addrspace(4) %v3, align 4
  store <2 x i32> %v5, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: name: test_buffer_load_sgpr_plus_imm_offset
; SDAG-DAG: %[[BASE0:.*]]:sgpr_32 = PRED_COPY $sgpr0
; SDAG-DAG: %[[BASE1:.*]]:sgpr_32 = PRED_COPY $sgpr1
; SDAG-DAG: %[[BASE2:.*]]:sgpr_32 = PRED_COPY $sgpr2
; SDAG-DAG: %[[BASE3:.*]]:sgpr_32 = PRED_COPY $sgpr3
; SDAG-DAG: %[[OFFSET:.*]]:sgpr_32 = PRED_COPY $sgpr4
; SDAG-DAG: %[[BASE:.*]]:sgpr_128 = REG_SEQUENCE %[[BASE0]], %subreg.sub0, %[[BASE1]], %subreg.sub1, %[[BASE2]], %subreg.sub2, %[[BASE3]], %subreg.sub3
; SDAG: S_BUFFER_LOAD_DWORD_SGPR_IMM killed %[[BASE]], %[[OFFSET]], 77,
; GISEL-DAG: %[[BASE0:.*]]:sreg_32 = PRED_COPY $sgpr0
; GISEL-DAG: %[[BASE1:.*]]:sreg_32 = PRED_COPY $sgpr1
; GISEL-DAG: %[[BASE2:.*]]:sreg_32 = PRED_COPY $sgpr2
; GISEL-DAG: %[[BASE3:.*]]:sreg_32 = PRED_COPY $sgpr3
; GISEL-DAG: %[[OFFSET:.*]]:sreg_32 = PRED_COPY $sgpr4
; GISEL-DAG: %[[BASE:.*]]:sgpr_128 = REG_SEQUENCE %[[BASE0]], %subreg.sub0, %[[BASE1]], %subreg.sub1, %[[BASE2]], %subreg.sub2, %[[BASE3]], %subreg.sub3
; GISEL: S_BUFFER_LOAD_DWORD_SGPR_IMM %[[BASE]], %[[OFFSET]], 77,
define amdgpu_cs void @test_buffer_load_sgpr_plus_imm_offset(<4 x i32> inreg %base, i32 inreg %i, ptr addrspace(1) inreg %out) {
  %off = add nuw nsw i32 %i, 77
  %v = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %base, i32 %off, i32 0)
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: name: test_buffer_load_sgpr_or_imm_offset
; SDAG-DAG: %[[BASE0:.*]]:sgpr_32 = PRED_COPY $sgpr0
; SDAG-DAG: %[[BASE1:.*]]:sgpr_32 = PRED_COPY $sgpr1
; SDAG-DAG: %[[BASE2:.*]]:sgpr_32 = PRED_COPY $sgpr2
; SDAG-DAG: %[[BASE3:.*]]:sgpr_32 = PRED_COPY $sgpr3
; SDAG-DAG: %[[INDEX:.*]]:sgpr_32 = PRED_COPY $sgpr4
; SDAG-DAG: %[[SHIFT:.*]]:sreg_32 = S_LSHL_B32 %[[INDEX]],
; SDAG-DAG: %[[BASE:.*]]:sgpr_128 = REG_SEQUENCE %[[BASE0]], %subreg.sub0, %[[BASE1]], %subreg.sub1, %[[BASE2]], %subreg.sub2, %[[BASE3]], %subreg.sub3
; SDAG: S_BUFFER_LOAD_DWORD_SGPR_IMM killed %[[BASE]], killed %[[SHIFT]], 5,
; GISEL-DAG: %[[BASE0:.*]]:sreg_32 = PRED_COPY $sgpr0
; GISEL-DAG: %[[BASE1:.*]]:sreg_32 = PRED_COPY $sgpr1
; GISEL-DAG: %[[BASE2:.*]]:sreg_32 = PRED_COPY $sgpr2
; GISEL-DAG: %[[BASE3:.*]]:sreg_32 = PRED_COPY $sgpr3
; GISEL-DAG: %[[INDEX:.*]]:sreg_32 = PRED_COPY $sgpr4
; GISEL-DAG: %[[SHIFT:.*]]:sreg_32 = S_LSHL_B32 %[[INDEX]],
; GISEL-DAG: %[[BASE:.*]]:sgpr_128 = REG_SEQUENCE %[[BASE0]], %subreg.sub0, %[[BASE1]], %subreg.sub1, %[[BASE2]], %subreg.sub2, %[[BASE3]], %subreg.sub3
; GISEL: S_BUFFER_LOAD_DWORD_SGPR_IMM %[[BASE]], %[[SHIFT]], 5,
define amdgpu_cs void @test_buffer_load_sgpr_or_imm_offset(<4 x i32> inreg %base, i32 inreg %i, ptr addrspace(1) inreg %out) {
  %shift = shl i32 %i, 7
  %off = or i32 %shift, 5
  %v = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %base, i32 %off, i32 0)
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}

declare void @llvm.amdgcn.raw.buffer.store.v4i32(<4 x i32>, <4 x i32>, i32, i32, i32 immarg) #1

declare i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32>, i32, i32 immarg) nounwind readnone willreturn

; Function Attrs: nounwind readnone speculatable
declare i32 @llvm.amdgcn.reloc.constant(metadata) #3

; Function Attrs: nounwind readnone speculatable
declare i64 @llvm.amdgcn.s.getpc() #3

; Function Attrs: nounwind readnone
declare <4 x i32> @llvm.amdgcn.s.buffer.load.v4i32(<4 x i32>, i32, i32 immarg) #1

attributes #0 = { argmemonly nounwind willreturn }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind "amdgpu-unroll-threshold"="700" }
attributes #3 = { nounwind readnone speculatable }
attributes #4 = { nounwind writeonly }

!llpc.compute.mode = !{!0}
!llpc.options = !{!1}
!llpc.options.CS = !{!2}
!llpc.user.data.nodes = !{!3, !4, !5, !6}
!amdgpu.pal.metadata.msgpack = !{!7}

!0 = !{i32 2, i32 3, i32 1}
!1 = !{i32 245227952, i32 996822128, i32 2024708198, i32 497230408}
!2 = !{i32 1381820427, i32 1742110173, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 64}
!3 = !{!"DescriptorTableVaPtr", i32 0, i32 1, i32 1}
!4 = !{!"DescriptorBuffer", i32 4, i32 8, i32 0, i32 0}
!5 = !{!"DescriptorTableVaPtr", i32 1, i32 1, i32 1}
!6 = !{!"DescriptorBuffer", i32 4, i32 8, i32 1, i32 0}
!7 = !{!"\82\B0amdpal.pipelines\91\88\A4.api\A6Vulkan\B0.hardware_stages\81\A3.cs\82\AB.sgpr_limith\AB.vgpr_limit\CD\01\00\B7.internal_pipeline_hash\92\CF;jLp\0E\9D\E1\B0\CF\1D\A3\22Hx\AE\98f\AA.registers\88\CD.\07\02\CD.\08\03\CD.\09\01\CD.\12\CE\00,\00\00\CD.\13\CD\0F\88\CD.@\CE\10\00\00\00\CD.B\00\CD.C\01\A8.shaders\81\A8.compute\82\B0.api_shader_hash\92\CFg\D6}\DDR\\\E8\0B\00\B1.hardware_mapping\91\A3.cs\B0.spill_threshold\CE\FF\FF\FF\FF\A5.type\A2Cs\B0.user_data_limit\02\AEamdpal.version\92\02\03"}
!8 = !{i32 5}
!9 = !{!"doff_0_0_b"}
!10 = !{}
!11 = !{!"doff_1_0_b"}
