; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx802 <%s | FileCheck %s

; Test that the mnemonic names for PAL metadata user data registers work in a
; full tessellation-and-geometry pipeline, compiled on gfx8 so it uses all six
; hardware shader types.

; CHECK-DAG: '0x2c0c (SPI_SHADER_USER_DATA_PS_0)': 0x10000000
; CHECK-DAG: '0x2c4c (SPI_SHADER_USER_DATA_VS_0)': 0x10000000
; CHECK-DAG: '0x2c8c (SPI_SHADER_USER_DATA_GS_0)': 0x10000000
; CHECK-DAG: '0x2ccc (SPI_SHADER_USER_DATA_ES_0)': 0x10000000
; CHECK-DAG: '0x2d0c (SPI_SHADER_USER_DATA_HS_0)': 0x10000000
; CHECK-DAG: '0x2d4c (SPI_SHADER_USER_DATA_LS_0)': 0x10000000

!amdgpu.pal.metadata.msgpack = !{!0}

!0 = !{!"\82\B0amdpal.pipelines\91\88\A4.api\A6Vulkan\B0.hardware_stages\86\A3.es\82\AB.sgpr_limith\AB.vgpr_limit\CD\01\00\A3.gs\82\AB.sgpr_limith\AB.vgpr_limit\CD\01\00\A3.hs\82\AB.sgpr_limith\AB.vgpr_limit\CD\01\00\A3.ls\83\A9.lds_size\CD\03\00\AB.sgpr_limith\AB.vgpr_limit\CD\01\00\A3.ps\82\AB.sgpr_limith\AB.vgpr_limit\CD\01\00\A3.vs\82\AB.sgpr_limith\AB.vgpr_limit\CD\01\00\B7.internal_pipeline_hash\92\CF\0BE\F2u\8CK\FA\BC\CF\E5\A2\84o\83\86\1C\F8\AA.registers\DE\00I\CD,\0A\CE\00,\00\00\CD,\0B\04\CD,\0C\CE\10\00\00\00\CD,J\CE\00,\00\00\CD,K\08\CD,L\CE\10\00\00\00\CD,\8A\CE\00,\00\00\CD,\8B\06\CD,\8C\CE\10\00\00\00\CD,\8E\01\CD,\CA\CE\03,\00\00\CD,\CB\06\CD,\CC\CE\10\00\00\00\CD,\CE\01\CD-\0A\CE\00,\00\00\CD-\0B\06\CD-\0C\CE\10\00\00\00\CD-\0E\01\CD-J\CE\01,\00\00\CD-K\CD\01\0A\CD-L\CE\10\00\00\00\CD-N\01\CD-O\CE\10\00\00\03\CD-P\CE\10\00\00\04\CD\A0\8F\01\CD\A1\91\00\CD\A1\B1\00\CD\A1\B3\00\CD\A1\B4\00\CD\A1\B5\00\CD\A1\B6\00\CD\A1\B8\CE\01\00\00\00\CD\A1\C3\04\CD\A1\C4\00\CD\A1\C5\01\CD\A2\03\10\CD\A2\04\CE\01\08\00\00\CD\A2\06\CD\04?\CD\A2\07\00\CD\A2\86\CEB\80\00\00\CD\A2\87\CE?\80\00\00\CD\A2\90\CE\00\18\003\CD\A2\91\00\CD\A2\93\CE\06\02\01\8C\CD\A2\95\CD\01\00\CD\A2\96\CC\80\CD\A2\97\02\CD\A2\98\04\CD\A2\99\04\CD\A2\9A\04\CD\A2\9B\00\CD\A2\A1\01\CD\A2\AA\CE\00\0C\00\00\CD\A2\AB\04\CD\A2\AC\04\CD\A2\AD\00\CD\A2\B5\00\CD\A2\B9\00\CD\A2\BD\00\CD\A2\C1\00\CD\A2\CE\01\CD\A2\D5\CC\AD\CD\A2\D6\CDA\10\CD\A2\D7\04\CD\A2\D8\00\CD\A2\D9\00\CD\A2\DA\00\CD\A2\DB \CD\A2\E4\00\CD\A2\E5\00\CD\A2\E6\00\CD\A2\F9-\CD\A3\16\0E\A8.shaders\85\A7.domain\82\B0.api_shader_hash\92\CF\B6\08\8De\FF\A1`\85\00\B1.hardware_mapping\91\A3.es\A9.geometry\82\B0.api_shader_hash\92\CF\B0Hn\E7\C1\9Etq\00\B1.hardware_mapping\92\A3.gs\A3.vs\A5.hull\82\B0.api_shader_hash\92\CF}\F2\C7\CF\AB\DB,B\00\B1.hardware_mapping\91\A3.hs\A6.pixel\82\B0.api_shader_hash\92\CF\A9\C3\05H\E3(Ay\00\B1.hardware_mapping\91\A3.ps\A7.vertex\82\B0.api_shader_hash\92\CF\A8\A3\C7x\DB\C5\88\84\00\B1.hardware_mapping\91\A3.ls\B0.spill_threshold\CE\FF\FF\FF\FF\A5.type\A6GsTess\B0.user_data_limit\02\AEamdpal.version\92\02\03"}
