; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -enable-var-scope %s

; We want to make sure that RSRC2 is left untouched
; GCN:       '0x2e13 (COMPUTE_PGM_RSRC2)': 0x78a
define amdgpu_cs half @cs_amdpal(half %arg0, half inreg %arg1) {
  %add = fadd half %arg0, 1.0
  ret half %add
}

!amdgpu.pal.metadata.msgpack = !{!0}

!0 = !{!"\82\B0amdpal.pipelines\91\89\A4.api\A6Vulkan\B0.hardware_stages\81\A3.cs\83\AB.sgpr_limith\AB.vgpr_limit\CD\01\00\AF.wavefront_size@\B7.internal_pipeline_hash\92\CF\E83\B3\C2\D1)\7FG\CF[\8A\DF\EE[\7FD,\AA.registers\8A\CD.\07\01\CD.\08\01\CD.\09\01\CD.\12\CE@,\00\00\CD.\13\CD\07\8A\CD.(\00\CD.*\CE\16\0B\22Y\CD.@\CE\10\00\00\00\CD.B\CE\10\00\00\06\CD.D\00\A8.shaders\81\A8.compute\82\B0.api_shader_hash\92\CF\D3s\A6\8D\C5x\84\D4\00\B1.hardware_mapping\91\A3.cs\B0.spill_threshold\CE\FF\FF\FF\FF\A5.type\A2Cs\B0.user_data_limit\01\AF.xgl_cache_info\82\B3.128_bit_cache_hash\92\CF\E5\A0\EB\F9}\C6\C1\13\CF\1A_\E7\F7\F2.mR\AD.llpc_version\A454.5\AEamdpal.version\92\02\03"}
