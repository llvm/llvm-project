; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -global-isel=0 -verify-machineinstrs -stop-after=finalize-isel -o - %s | FileCheck -check-prefixes=GCN,SDAG %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -global-isel=1 -verify-machineinstrs -stop-after=finalize-isel -o - %s | FileCheck -check-prefixes=GCN,GISEL %s

@0 = external dso_local addrspace(4) constant [4 x <2 x float>]
@1 = external dso_local addrspace(4) constant i32

; GCN-LABEL: name: test_buffer_load_sgpr_plus_imm_offset
; SDAG-DAG: %[[BASE0:.*]]:sgpr_32 = COPY $sgpr0
; SDAG-DAG: %[[BASE1:.*]]:sgpr_32 = COPY $sgpr1
; SDAG-DAG: %[[BASE2:.*]]:sgpr_32 = COPY $sgpr2
; SDAG-DAG: %[[BASE3:.*]]:sgpr_32 = COPY $sgpr3
; SDAG-DAG: %[[OFFSET:.*]]:sgpr_32 = COPY $sgpr4
; SDAG-DAG: %[[BASE:.*]]:sgpr_128 = REG_SEQUENCE %[[BASE0]], %subreg.sub0, %[[BASE1]], %subreg.sub1, %[[BASE2]], %subreg.sub2, %[[BASE3]], %subreg.sub3
; SDAG: S_BUFFER_LOAD_DWORD_SGPR_IMM killed %[[BASE]], %[[OFFSET]], 77,
; GISEL-DAG: %[[BASE0:.*]]:sreg_32 = COPY $sgpr0
; GISEL-DAG: %[[BASE1:.*]]:sreg_32 = COPY $sgpr1
; GISEL-DAG: %[[BASE2:.*]]:sreg_32 = COPY $sgpr2
; GISEL-DAG: %[[BASE3:.*]]:sreg_32 = COPY $sgpr3
; GISEL-DAG: %[[OFFSET:.*]]:sreg_32 = COPY $sgpr4
; GISEL-DAG: %[[BASE:.*]]:sgpr_128 = REG_SEQUENCE %[[BASE0]], %subreg.sub0, %[[BASE1]], %subreg.sub1, %[[BASE2]], %subreg.sub2, %[[BASE3]], %subreg.sub3
; GISEL: S_BUFFER_LOAD_DWORD_SGPR_IMM %[[BASE]], %[[OFFSET]], 77,
define void @test_buffer_load_sgpr_plus_imm_offset(<4 x i32> %base, i32 %i, i32 inreg %j, ptr addrspace(1) inreg %out) {
  %off = add i32 %i, %j
  %v = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %base, i32 %off, i32 0)
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}

declare void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32>, ptr addrspace(8) nocapture, i32, i32, i32 immarg) #1

declare i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32>, i32, i32 immarg) nounwind readnone willreturn

; Function Attrs: nounwind readnone speculatable
declare i32 @llvm.amdgcn.reloc.constant(metadata) #3

; Function Attrs: nounwind readnone speculatable
declare i64 @llvm.amdgcn.s.getpc() #3

; Function Attrs: nounwind readnone
declare <4 x i32> @llvm.amdgcn.s.buffer.load.v4i32(<4 x i32>, i32, i32 immarg) #1

attributes #0 = { argmemonly nounwind willreturn }
attributes #1 = { nounwind memory(argmem: write) }
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
