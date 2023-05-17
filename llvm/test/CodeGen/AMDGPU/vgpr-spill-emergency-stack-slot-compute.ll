; XFAIL: *
; RUN: llc -march=amdgcn -mtriple=amdgcn-- -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GCNMESA -check-prefix=SIMESA %s
; RUN: llc -march=amdgcn -mtriple=amdgcn-- -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GCNMESA -check-prefix=VIMESA %s
; RUN: llc -march=amdgcn -mtriple=amdgcn-- -mcpu=gfx900 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GCNMESA -check-prefix=GFX9MESA %s
; RUN: llc -march=amdgcn  -mcpu=hawaii -mtriple=amdgcn-unknown-amdhsa -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=CIHSA -check-prefix=HSA %s
; RUN: llc -march=amdgcn  -mcpu=fiji -mtriple=amdgcn-unknown-amdhsa -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VIHSA -check-prefix=HSA %s

; This ends up using all 256 registers and requires register
; scavenging which will fail to find an unsued register.

; Check the ScratchSize to avoid regressions from spilling
; intermediate register class copies.

; FIXME: The same register is initialized to 0 for every spill.

; GCN-LABEL: {{^}}spill_vgpr_compute:

; HSA: enable_sgpr_private_segment_buffer = 1
; HSA: enable_sgpr_flat_scratch_init = 0
; HSA: workitem_private_segment_byte_size = 1536

; GCN-NOT: flat_scr
; MESA-NOT: s_mov_b32 s3
; HSA-NOT: s_mov_b32 s7

; GCNMESA-DAG: s_mov_b32 s16, SCRATCH_RSRC_DWORD0
; GCNMESA-DAG: s_mov_b32 s17, SCRATCH_RSRC_DWORD1
; GCNMESA-DAG: s_mov_b32 s18, -1
; SIMESA-DAG: s_mov_b32 s19, 0xe8f000
; VIMESA-DAG: s_mov_b32 s19, 0xe80000
; GFX9MESA-DAG: s_mov_b32 s19, 0xe00000


; GCNMESAMESA: buffer_store_dword {{v[0-9]+}}, off, s[16:19], s3 offset:{{[0-9]+}} ; 4-byte Folded Spill

; GCNMESA: buffer_store_dword {{v[0-9]}}, off, s[16:19], s3 offset:{{[0-9]+}}
; GCNMESA: buffer_store_dword {{v[0-9]}}, off, s[16:19], s3 offset:{{[0-9]+}}
; GCNMESA: buffer_store_dword {{v[0-9]}}, off, s[16:19], s3 offset:{{[0-9]+}}
; GCNMESA: buffer_store_dword {{v[0-9]}}, off, s[16:19], s3 offset:{{[0-9]+}}

; GCNMESA: buffer_load_dword {{v[0-9]+}}, off, s[16:19], s3 offset:{{[0-9]+}}
; GCNMESA: buffer_load_dword {{v[0-9]+}}, off, s[16:19], s3 offset:{{[0-9]+}}
; GCNMESA: buffer_load_dword {{v[0-9]+}}, off, s[16:19], s3 offset:{{[0-9]+}}
; GCNMESA: buffer_load_dword {{v[0-9]+}}, off, s[16:19], s3 offset:{{[0-9]+}}



; HSA: buffer_store_dword {{v[0-9]+}}, off, s[0:3], s7 offset:{{[0-9]+}} ; 4-byte Folded Spill

; HSA: buffer_store_dword {{v[0-9]}}, off, s[0:3], s7 offset:{{[0-9]+}}
; HSA: buffer_store_dword {{v[0-9]}}, off, s[0:3], s7 offset:{{[0-9]+}}
; HSA: buffer_store_dword {{v[0-9]}}, off, s[0:3], s7 offset:{{[0-9]+}}
; HSA: buffer_store_dword {{v[0-9]}}, off, s[0:3], s7 offset:{{[0-9]+}}

; HSA: buffer_load_dword {{v[0-9]+}}, off, s[0:3], s7 offset:{{[0-9]+}}
; HSA: buffer_load_dword {{v[0-9]+}}, off, s[0:3], s7 offset:{{[0-9]+}}
; HSA: buffer_load_dword {{v[0-9]+}}, off, s[0:3], s7 offset:{{[0-9]+}}
; HSA: buffer_load_dword {{v[0-9]+}}, off, s[0:3], s7 offset:{{[0-9]+}}


; GCN: NumVgprs: 256
; GCN: ScratchSize: 1536

; s[0:3] input user SGPRs. s4,s5,s6 = workgroup IDs. s8 scratch offset.
define amdgpu_kernel void @spill_vgpr_compute(<4 x float> %arg6, ptr addrspace(1) %arg, i32 %arg1, i32 %arg2, float %arg3, float %arg4, float %arg5) #0 {
bb:
  %tmp = add i32 %arg1, %arg2
  %tmp7 = extractelement <4 x float> %arg6, i32 0
  %tmp8 = extractelement <4 x float> %arg6, i32 1
  %tmp9 = extractelement <4 x float> %arg6, i32 2
  %tmp10 = extractelement <4 x float> %arg6, i32 3
  %tmp11 = bitcast float %arg5 to i32
  br label %bb12

bb12:                                             ; preds = %bb145, %bb
  %tmp13 = phi float [ 0.000000e+00, %bb ], [ %tmp338, %bb145 ]
  %tmp14 = phi float [ 0.000000e+00, %bb ], [ %tmp337, %bb145 ]
  %tmp15 = phi float [ 0.000000e+00, %bb ], [ %tmp336, %bb145 ]
  %tmp16 = phi float [ 0.000000e+00, %bb ], [ %tmp339, %bb145 ]
  %tmp17 = phi float [ 0.000000e+00, %bb ], [ %tmp335, %bb145 ]
  %tmp18 = phi float [ 0.000000e+00, %bb ], [ %tmp334, %bb145 ]
  %tmp19 = phi float [ 0.000000e+00, %bb ], [ %tmp333, %bb145 ]
  %tmp20 = phi float [ 0.000000e+00, %bb ], [ %tmp340, %bb145 ]
  %tmp21 = phi float [ 0.000000e+00, %bb ], [ %tmp332, %bb145 ]
  %tmp22 = phi float [ 0.000000e+00, %bb ], [ %tmp331, %bb145 ]
  %tmp23 = phi float [ 0.000000e+00, %bb ], [ %tmp330, %bb145 ]
  %tmp24 = phi float [ 0.000000e+00, %bb ], [ %tmp341, %bb145 ]
  %tmp25 = phi float [ 0.000000e+00, %bb ], [ %tmp329, %bb145 ]
  %tmp26 = phi float [ 0.000000e+00, %bb ], [ %tmp328, %bb145 ]
  %tmp27 = phi float [ 0.000000e+00, %bb ], [ %tmp327, %bb145 ]
  %tmp28 = phi float [ 0.000000e+00, %bb ], [ %tmp342, %bb145 ]
  %tmp29 = phi float [ 0.000000e+00, %bb ], [ %tmp326, %bb145 ]
  %tmp30 = phi float [ 0.000000e+00, %bb ], [ %tmp325, %bb145 ]
  %tmp31 = phi float [ 0.000000e+00, %bb ], [ %tmp324, %bb145 ]
  %tmp32 = phi float [ 0.000000e+00, %bb ], [ %tmp343, %bb145 ]
  %tmp33 = phi float [ 0.000000e+00, %bb ], [ %tmp323, %bb145 ]
  %tmp34 = phi float [ 0.000000e+00, %bb ], [ %tmp322, %bb145 ]
  %tmp35 = phi float [ 0.000000e+00, %bb ], [ %tmp321, %bb145 ]
  %tmp36 = phi float [ 0.000000e+00, %bb ], [ %tmp344, %bb145 ]
  %tmp37 = phi float [ 0.000000e+00, %bb ], [ %tmp320, %bb145 ]
  %tmp38 = phi float [ 0.000000e+00, %bb ], [ %tmp319, %bb145 ]
  %tmp39 = phi float [ 0.000000e+00, %bb ], [ %tmp318, %bb145 ]
  %tmp40 = phi float [ 0.000000e+00, %bb ], [ %tmp345, %bb145 ]
  %tmp41 = phi float [ 0.000000e+00, %bb ], [ %tmp317, %bb145 ]
  %tmp42 = phi float [ 0.000000e+00, %bb ], [ %tmp316, %bb145 ]
  %tmp43 = phi float [ 0.000000e+00, %bb ], [ %tmp315, %bb145 ]
  %tmp44 = phi float [ 0.000000e+00, %bb ], [ %tmp346, %bb145 ]
  %tmp45 = phi float [ 0.000000e+00, %bb ], [ %tmp314, %bb145 ]
  %tmp46 = phi float [ 0.000000e+00, %bb ], [ %tmp313, %bb145 ]
  %tmp47 = phi float [ 0.000000e+00, %bb ], [ %tmp312, %bb145 ]
  %tmp48 = phi float [ 0.000000e+00, %bb ], [ %tmp347, %bb145 ]
  %tmp49 = phi float [ 0.000000e+00, %bb ], [ %tmp311, %bb145 ]
  %tmp50 = phi float [ 0.000000e+00, %bb ], [ %tmp310, %bb145 ]
  %tmp51 = phi float [ 0.000000e+00, %bb ], [ %tmp309, %bb145 ]
  %tmp52 = phi float [ 0.000000e+00, %bb ], [ %tmp348, %bb145 ]
  %tmp53 = phi float [ 0.000000e+00, %bb ], [ %tmp308, %bb145 ]
  %tmp54 = phi float [ 0.000000e+00, %bb ], [ %tmp307, %bb145 ]
  %tmp55 = phi float [ 0.000000e+00, %bb ], [ %tmp306, %bb145 ]
  %tmp56 = phi float [ 0.000000e+00, %bb ], [ %tmp349, %bb145 ]
  %tmp57 = phi float [ 0.000000e+00, %bb ], [ %tmp305, %bb145 ]
  %tmp58 = phi float [ 0.000000e+00, %bb ], [ %tmp304, %bb145 ]
  %tmp59 = phi float [ 0.000000e+00, %bb ], [ %tmp303, %bb145 ]
  %tmp60 = phi float [ 0.000000e+00, %bb ], [ %tmp350, %bb145 ]
  %tmp61 = phi float [ 0.000000e+00, %bb ], [ %tmp302, %bb145 ]
  %tmp62 = phi float [ 0.000000e+00, %bb ], [ %tmp301, %bb145 ]
  %tmp63 = phi float [ 0.000000e+00, %bb ], [ %tmp300, %bb145 ]
  %tmp64 = phi float [ 0.000000e+00, %bb ], [ %tmp351, %bb145 ]
  %tmp65 = phi float [ 0.000000e+00, %bb ], [ %tmp299, %bb145 ]
  %tmp66 = phi float [ 0.000000e+00, %bb ], [ %tmp298, %bb145 ]
  %tmp67 = phi float [ 0.000000e+00, %bb ], [ %tmp297, %bb145 ]
  %tmp68 = phi float [ 0.000000e+00, %bb ], [ %tmp352, %bb145 ]
  %tmp69 = phi float [ 0.000000e+00, %bb ], [ %tmp296, %bb145 ]
  %tmp70 = phi float [ 0.000000e+00, %bb ], [ %tmp295, %bb145 ]
  %tmp71 = phi float [ 0.000000e+00, %bb ], [ %tmp294, %bb145 ]
  %tmp72 = phi float [ 0.000000e+00, %bb ], [ %tmp353, %bb145 ]
  %tmp73 = phi float [ 0.000000e+00, %bb ], [ %tmp293, %bb145 ]
  %tmp74 = phi float [ 0.000000e+00, %bb ], [ %tmp292, %bb145 ]
  %tmp75 = phi float [ 0.000000e+00, %bb ], [ %tmp291, %bb145 ]
  %tmp76 = phi float [ 0.000000e+00, %bb ], [ %tmp354, %bb145 ]
  %tmp77 = phi float [ 0.000000e+00, %bb ], [ %tmp290, %bb145 ]
  %tmp78 = phi float [ 0.000000e+00, %bb ], [ %tmp289, %bb145 ]
  %tmp79 = phi float [ 0.000000e+00, %bb ], [ %tmp288, %bb145 ]
  %tmp80 = phi float [ 0.000000e+00, %bb ], [ %tmp355, %bb145 ]
  %tmp81 = phi float [ 0.000000e+00, %bb ], [ %tmp287, %bb145 ]
  %tmp82 = phi float [ 0.000000e+00, %bb ], [ %tmp286, %bb145 ]
  %tmp83 = phi float [ 0.000000e+00, %bb ], [ %tmp285, %bb145 ]
  %tmp84 = phi float [ 0.000000e+00, %bb ], [ %tmp356, %bb145 ]
  %tmp85 = phi float [ 0.000000e+00, %bb ], [ %tmp284, %bb145 ]
  %tmp86 = phi float [ 0.000000e+00, %bb ], [ %tmp283, %bb145 ]
  %tmp87 = phi float [ 0.000000e+00, %bb ], [ %tmp282, %bb145 ]
  %tmp88 = phi float [ 0.000000e+00, %bb ], [ %tmp357, %bb145 ]
  %tmp89 = phi float [ 0.000000e+00, %bb ], [ %tmp281, %bb145 ]
  %tmp90 = phi float [ 0.000000e+00, %bb ], [ %tmp280, %bb145 ]
  %tmp91 = phi float [ 0.000000e+00, %bb ], [ %tmp279, %bb145 ]
  %tmp92 = phi float [ 0.000000e+00, %bb ], [ %tmp358, %bb145 ]
  %tmp93 = phi float [ 0.000000e+00, %bb ], [ %tmp359, %bb145 ]
  %tmp94 = phi float [ 0.000000e+00, %bb ], [ %tmp360, %bb145 ]
  %tmp95 = phi float [ 0.000000e+00, %bb ], [ %tmp409, %bb145 ]
  %tmp96 = phi float [ 0.000000e+00, %bb ], [ %tmp361, %bb145 ]
  %tmp97 = phi float [ 0.000000e+00, %bb ], [ %tmp362, %bb145 ]
  %tmp98 = phi float [ 0.000000e+00, %bb ], [ %tmp363, %bb145 ]
  %tmp99 = phi float [ 0.000000e+00, %bb ], [ %tmp364, %bb145 ]
  %tmp100 = phi float [ 0.000000e+00, %bb ], [ %tmp365, %bb145 ]
  %tmp101 = phi float [ 0.000000e+00, %bb ], [ %tmp366, %bb145 ]
  %tmp102 = phi float [ 0.000000e+00, %bb ], [ %tmp367, %bb145 ]
  %tmp103 = phi float [ 0.000000e+00, %bb ], [ %tmp368, %bb145 ]
  %tmp104 = phi float [ 0.000000e+00, %bb ], [ %tmp369, %bb145 ]
  %tmp105 = phi float [ 0.000000e+00, %bb ], [ %tmp370, %bb145 ]
  %tmp106 = phi float [ 0.000000e+00, %bb ], [ %tmp371, %bb145 ]
  %tmp107 = phi float [ 0.000000e+00, %bb ], [ %tmp372, %bb145 ]
  %tmp108 = phi float [ 0.000000e+00, %bb ], [ %tmp373, %bb145 ]
  %tmp109 = phi float [ 0.000000e+00, %bb ], [ %tmp374, %bb145 ]
  %tmp110 = phi float [ 0.000000e+00, %bb ], [ %tmp375, %bb145 ]
  %tmp111 = phi float [ 0.000000e+00, %bb ], [ %tmp376, %bb145 ]
  %tmp112 = phi float [ 0.000000e+00, %bb ], [ %tmp377, %bb145 ]
  %tmp113 = phi float [ 0.000000e+00, %bb ], [ %tmp378, %bb145 ]
  %tmp114 = phi float [ 0.000000e+00, %bb ], [ %tmp379, %bb145 ]
  %tmp115 = phi float [ 0.000000e+00, %bb ], [ %tmp380, %bb145 ]
  %tmp116 = phi float [ 0.000000e+00, %bb ], [ %tmp381, %bb145 ]
  %tmp117 = phi float [ 0.000000e+00, %bb ], [ %tmp382, %bb145 ]
  %tmp118 = phi float [ 0.000000e+00, %bb ], [ %tmp383, %bb145 ]
  %tmp119 = phi float [ 0.000000e+00, %bb ], [ %tmp384, %bb145 ]
  %tmp120 = phi float [ 0.000000e+00, %bb ], [ %tmp385, %bb145 ]
  %tmp121 = phi float [ 0.000000e+00, %bb ], [ %tmp386, %bb145 ]
  %tmp122 = phi float [ 0.000000e+00, %bb ], [ %tmp387, %bb145 ]
  %tmp123 = phi float [ 0.000000e+00, %bb ], [ %tmp388, %bb145 ]
  %tmp124 = phi float [ 0.000000e+00, %bb ], [ %tmp389, %bb145 ]
  %tmp125 = phi float [ 0.000000e+00, %bb ], [ %tmp390, %bb145 ]
  %tmp126 = phi float [ 0.000000e+00, %bb ], [ %tmp391, %bb145 ]
  %tmp127 = phi float [ 0.000000e+00, %bb ], [ %tmp392, %bb145 ]
  %tmp128 = phi float [ 0.000000e+00, %bb ], [ %tmp393, %bb145 ]
  %tmp129 = phi float [ 0.000000e+00, %bb ], [ %tmp394, %bb145 ]
  %tmp130 = phi float [ 0.000000e+00, %bb ], [ %tmp395, %bb145 ]
  %tmp131 = phi float [ 0.000000e+00, %bb ], [ %tmp396, %bb145 ]
  %tmp132 = phi float [ 0.000000e+00, %bb ], [ %tmp397, %bb145 ]
  %tmp133 = phi float [ 0.000000e+00, %bb ], [ %tmp398, %bb145 ]
  %tmp134 = phi float [ 0.000000e+00, %bb ], [ %tmp399, %bb145 ]
  %tmp135 = phi float [ 0.000000e+00, %bb ], [ %tmp400, %bb145 ]
  %tmp136 = phi float [ 0.000000e+00, %bb ], [ %tmp401, %bb145 ]
  %tmp137 = phi float [ 0.000000e+00, %bb ], [ %tmp402, %bb145 ]
  %tmp138 = phi float [ 0.000000e+00, %bb ], [ %tmp403, %bb145 ]
  %tmp139 = phi float [ 0.000000e+00, %bb ], [ %tmp404, %bb145 ]
  %tmp140 = phi float [ 0.000000e+00, %bb ], [ %tmp405, %bb145 ]
  %tmp141 = phi float [ 0.000000e+00, %bb ], [ %tmp406, %bb145 ]
  %tmp142 = bitcast float %tmp95 to i32
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tmp143 = icmp sgt i32 %tmp142, %tid
  br i1 %tmp143, label %bb144, label %bb145

bb144:                                            ; preds = %bb12
  store volatile float %arg3, ptr addrspace(1) %arg
  store volatile float %tmp91, ptr addrspace(1) %arg
  store volatile float %tmp90, ptr addrspace(1) %arg
  store volatile float %tmp89, ptr addrspace(1) %arg
  store volatile float %tmp87, ptr addrspace(1) %arg
  store volatile float %tmp86, ptr addrspace(1) %arg
  store volatile float %tmp85, ptr addrspace(1) %arg
  store volatile float %tmp83, ptr addrspace(1) %arg
  store volatile float %tmp82, ptr addrspace(1) %arg
  store volatile float %tmp81, ptr addrspace(1) %arg
  store volatile float %tmp79, ptr addrspace(1) %arg
  store volatile float %tmp78, ptr addrspace(1) %arg
  store volatile float %tmp77, ptr addrspace(1) %arg
  store volatile float %tmp75, ptr addrspace(1) %arg
  store volatile float %tmp74, ptr addrspace(1) %arg
  store volatile float %tmp73, ptr addrspace(1) %arg
  store volatile float %tmp71, ptr addrspace(1) %arg
  store volatile float %tmp70, ptr addrspace(1) %arg
  store volatile float %tmp69, ptr addrspace(1) %arg
  store volatile float %tmp67, ptr addrspace(1) %arg
  store volatile float %tmp66, ptr addrspace(1) %arg
  store volatile float %tmp65, ptr addrspace(1) %arg
  store volatile float %tmp63, ptr addrspace(1) %arg
  store volatile float %tmp62, ptr addrspace(1) %arg
  store volatile float %tmp61, ptr addrspace(1) %arg
  store volatile float %tmp59, ptr addrspace(1) %arg
  store volatile float %tmp58, ptr addrspace(1) %arg
  store volatile float %tmp57, ptr addrspace(1) %arg
  store volatile float %tmp55, ptr addrspace(1) %arg
  store volatile float %tmp54, ptr addrspace(1) %arg
  store volatile float %tmp53, ptr addrspace(1) %arg
  store volatile float %tmp51, ptr addrspace(1) %arg
  store volatile float %tmp50, ptr addrspace(1) %arg
  store volatile float %tmp49, ptr addrspace(1) %arg
  store volatile float %tmp47, ptr addrspace(1) %arg
  store volatile float %tmp46, ptr addrspace(1) %arg
  store volatile float %tmp45, ptr addrspace(1) %arg
  store volatile float %tmp43, ptr addrspace(1) %arg
  store volatile float %tmp42, ptr addrspace(1) %arg
  store volatile float %tmp41, ptr addrspace(1) %arg
  store volatile float %tmp39, ptr addrspace(1) %arg
  store volatile float %tmp38, ptr addrspace(1) %arg
  store volatile float %tmp37, ptr addrspace(1) %arg
  store volatile float %tmp35, ptr addrspace(1) %arg
  store volatile float %tmp34, ptr addrspace(1) %arg
  store volatile float %tmp33, ptr addrspace(1) %arg
  store volatile float %tmp31, ptr addrspace(1) %arg
  store volatile float %tmp30, ptr addrspace(1) %arg
  store volatile float %tmp29, ptr addrspace(1) %arg
  store volatile float %tmp27, ptr addrspace(1) %arg
  store volatile float %tmp26, ptr addrspace(1) %arg
  store volatile float %tmp25, ptr addrspace(1) %arg
  store volatile float %tmp23, ptr addrspace(1) %arg
  store volatile float %tmp22, ptr addrspace(1) %arg
  store volatile float %tmp21, ptr addrspace(1) %arg
  store volatile float %tmp19, ptr addrspace(1) %arg
  store volatile float %tmp18, ptr addrspace(1) %arg
  store volatile float %tmp17, ptr addrspace(1) %arg
  store volatile float %tmp15, ptr addrspace(1) %arg
  store volatile float %tmp14, ptr addrspace(1) %arg
  store volatile float %tmp13, ptr addrspace(1) %arg
  store volatile float %tmp16, ptr addrspace(1) %arg
  store volatile float %tmp20, ptr addrspace(1) %arg
  store volatile float %tmp24, ptr addrspace(1) %arg
  store volatile float %tmp28, ptr addrspace(1) %arg
  store volatile float %tmp32, ptr addrspace(1) %arg
  store volatile float %tmp36, ptr addrspace(1) %arg
  store volatile float %tmp40, ptr addrspace(1) %arg
  store volatile float %tmp44, ptr addrspace(1) %arg
  store volatile float %tmp48, ptr addrspace(1) %arg
  store volatile float %tmp52, ptr addrspace(1) %arg
  store volatile float %tmp56, ptr addrspace(1) %arg
  store volatile float %tmp60, ptr addrspace(1) %arg
  store volatile float %tmp64, ptr addrspace(1) %arg
  store volatile float %tmp68, ptr addrspace(1) %arg
  store volatile float %tmp72, ptr addrspace(1) %arg
  store volatile float %tmp76, ptr addrspace(1) %arg
  store volatile float %tmp80, ptr addrspace(1) %arg
  store volatile float %tmp84, ptr addrspace(1) %arg
  store volatile float %tmp88, ptr addrspace(1) %arg
  store volatile float %tmp92, ptr addrspace(1) %arg
  store volatile float %tmp93, ptr addrspace(1) %arg
  store volatile float %tmp94, ptr addrspace(1) %arg
  store volatile float %tmp96, ptr addrspace(1) %arg
  store volatile float %tmp97, ptr addrspace(1) %arg
  store volatile float %tmp98, ptr addrspace(1) %arg
  store volatile float %tmp99, ptr addrspace(1) %arg
  store volatile float %tmp100, ptr addrspace(1) %arg
  store volatile float %tmp101, ptr addrspace(1) %arg
  store volatile float %tmp102, ptr addrspace(1) %arg
  store volatile float %tmp103, ptr addrspace(1) %arg
  store volatile float %tmp104, ptr addrspace(1) %arg
  store volatile float %tmp105, ptr addrspace(1) %arg
  store volatile float %tmp106, ptr addrspace(1) %arg
  store volatile float %tmp107, ptr addrspace(1) %arg
  store volatile float %tmp108, ptr addrspace(1) %arg
  store volatile float %tmp109, ptr addrspace(1) %arg
  store volatile float %tmp110, ptr addrspace(1) %arg
  store volatile float %tmp111, ptr addrspace(1) %arg
  store volatile float %tmp112, ptr addrspace(1) %arg
  store volatile float %tmp113, ptr addrspace(1) %arg
  store volatile float %tmp114, ptr addrspace(1) %arg
  store volatile float %tmp115, ptr addrspace(1) %arg
  store volatile float %tmp116, ptr addrspace(1) %arg
  store volatile float %tmp117, ptr addrspace(1) %arg
  store volatile float %tmp118, ptr addrspace(1) %arg
  store volatile float %tmp119, ptr addrspace(1) %arg
  store volatile float %tmp120, ptr addrspace(1) %arg
  store volatile float %tmp121, ptr addrspace(1) %arg
  store volatile float %tmp122, ptr addrspace(1) %arg
  store volatile float %tmp123, ptr addrspace(1) %arg
  store volatile float %tmp124, ptr addrspace(1) %arg
  store volatile float %tmp125, ptr addrspace(1) %arg
  store volatile float %tmp126, ptr addrspace(1) %arg
  store volatile float %tmp127, ptr addrspace(1) %arg
  store volatile float %tmp128, ptr addrspace(1) %arg
  store volatile float %tmp129, ptr addrspace(1) %arg
  store volatile float %tmp130, ptr addrspace(1) %arg
  store volatile float %tmp131, ptr addrspace(1) %arg
  store volatile float %tmp132, ptr addrspace(1) %arg
  store volatile float %tmp133, ptr addrspace(1) %arg
  store volatile float %tmp134, ptr addrspace(1) %arg
  store volatile float %tmp135, ptr addrspace(1) %arg
  store volatile float %tmp136, ptr addrspace(1) %arg
  store volatile float %tmp137, ptr addrspace(1) %arg
  store volatile float %tmp138, ptr addrspace(1) %arg
  store volatile float %tmp139, ptr addrspace(1) %arg
  store volatile float %arg4, ptr addrspace(1) %arg
  store volatile float %tmp7, ptr addrspace(1) %arg
  store volatile float %tmp8, ptr addrspace(1) %arg
  store volatile float %tmp9, ptr addrspace(1) %arg
  store volatile float %tmp10, ptr addrspace(1) %arg
  ret void

bb145:                                            ; preds = %bb12
  %tmp146 = bitcast float %tmp95 to i32
  %tmp147 = bitcast float %tmp95 to i32
  %tmp148 = add i32 %tmp11, %tmp147
  %tmp149 = bitcast i32 %tmp148 to float
  %tmp150 = insertelement <128 x float> undef, float %tmp91, i32 0
  %tmp151 = insertelement <128 x float> %tmp150, float %tmp90, i32 1
  %tmp152 = insertelement <128 x float> %tmp151, float %tmp89, i32 2
  %tmp153 = insertelement <128 x float> %tmp152, float %tmp87, i32 3
  %tmp154 = insertelement <128 x float> %tmp153, float %tmp86, i32 4
  %tmp155 = insertelement <128 x float> %tmp154, float %tmp85, i32 5
  %tmp156 = insertelement <128 x float> %tmp155, float %tmp83, i32 6
  %tmp157 = insertelement <128 x float> %tmp156, float %tmp82, i32 7
  %tmp158 = insertelement <128 x float> %tmp157, float %tmp81, i32 8
  %tmp159 = insertelement <128 x float> %tmp158, float %tmp79, i32 9
  %tmp160 = insertelement <128 x float> %tmp159, float %tmp78, i32 10
  %tmp161 = insertelement <128 x float> %tmp160, float %tmp77, i32 11
  %tmp162 = insertelement <128 x float> %tmp161, float %tmp75, i32 12
  %tmp163 = insertelement <128 x float> %tmp162, float %tmp74, i32 13
  %tmp164 = insertelement <128 x float> %tmp163, float %tmp73, i32 14
  %tmp165 = insertelement <128 x float> %tmp164, float %tmp71, i32 15
  %tmp166 = insertelement <128 x float> %tmp165, float %tmp70, i32 16
  %tmp167 = insertelement <128 x float> %tmp166, float %tmp69, i32 17
  %tmp168 = insertelement <128 x float> %tmp167, float %tmp67, i32 18
  %tmp169 = insertelement <128 x float> %tmp168, float %tmp66, i32 19
  %tmp170 = insertelement <128 x float> %tmp169, float %tmp65, i32 20
  %tmp171 = insertelement <128 x float> %tmp170, float %tmp63, i32 21
  %tmp172 = insertelement <128 x float> %tmp171, float %tmp62, i32 22
  %tmp173 = insertelement <128 x float> %tmp172, float %tmp61, i32 23
  %tmp174 = insertelement <128 x float> %tmp173, float %tmp59, i32 24
  %tmp175 = insertelement <128 x float> %tmp174, float %tmp58, i32 25
  %tmp176 = insertelement <128 x float> %tmp175, float %tmp57, i32 26
  %tmp177 = insertelement <128 x float> %tmp176, float %tmp55, i32 27
  %tmp178 = insertelement <128 x float> %tmp177, float %tmp54, i32 28
  %tmp179 = insertelement <128 x float> %tmp178, float %tmp53, i32 29
  %tmp180 = insertelement <128 x float> %tmp179, float %tmp51, i32 30
  %tmp181 = insertelement <128 x float> %tmp180, float %tmp50, i32 31
  %tmp182 = insertelement <128 x float> %tmp181, float %tmp49, i32 32
  %tmp183 = insertelement <128 x float> %tmp182, float %tmp47, i32 33
  %tmp184 = insertelement <128 x float> %tmp183, float %tmp46, i32 34
  %tmp185 = insertelement <128 x float> %tmp184, float %tmp45, i32 35
  %tmp186 = insertelement <128 x float> %tmp185, float %tmp43, i32 36
  %tmp187 = insertelement <128 x float> %tmp186, float %tmp42, i32 37
  %tmp188 = insertelement <128 x float> %tmp187, float %tmp41, i32 38
  %tmp189 = insertelement <128 x float> %tmp188, float %tmp39, i32 39
  %tmp190 = insertelement <128 x float> %tmp189, float %tmp38, i32 40
  %tmp191 = insertelement <128 x float> %tmp190, float %tmp37, i32 41
  %tmp192 = insertelement <128 x float> %tmp191, float %tmp35, i32 42
  %tmp193 = insertelement <128 x float> %tmp192, float %tmp34, i32 43
  %tmp194 = insertelement <128 x float> %tmp193, float %tmp33, i32 44
  %tmp195 = insertelement <128 x float> %tmp194, float %tmp31, i32 45
  %tmp196 = insertelement <128 x float> %tmp195, float %tmp30, i32 46
  %tmp197 = insertelement <128 x float> %tmp196, float %tmp29, i32 47
  %tmp198 = insertelement <128 x float> %tmp197, float %tmp27, i32 48
  %tmp199 = insertelement <128 x float> %tmp198, float %tmp26, i32 49
  %tmp200 = insertelement <128 x float> %tmp199, float %tmp25, i32 50
  %tmp201 = insertelement <128 x float> %tmp200, float %tmp23, i32 51
  %tmp202 = insertelement <128 x float> %tmp201, float %tmp22, i32 52
  %tmp203 = insertelement <128 x float> %tmp202, float %tmp21, i32 53
  %tmp204 = insertelement <128 x float> %tmp203, float %tmp19, i32 54
  %tmp205 = insertelement <128 x float> %tmp204, float %tmp18, i32 55
  %tmp206 = insertelement <128 x float> %tmp205, float %tmp17, i32 56
  %tmp207 = insertelement <128 x float> %tmp206, float %tmp15, i32 57
  %tmp208 = insertelement <128 x float> %tmp207, float %tmp14, i32 58
  %tmp209 = insertelement <128 x float> %tmp208, float %tmp13, i32 59
  %tmp210 = insertelement <128 x float> %tmp209, float %tmp16, i32 60
  %tmp211 = insertelement <128 x float> %tmp210, float %tmp20, i32 61
  %tmp212 = insertelement <128 x float> %tmp211, float %tmp24, i32 62
  %tmp213 = insertelement <128 x float> %tmp212, float %tmp28, i32 63
  %tmp214 = insertelement <128 x float> %tmp213, float %tmp32, i32 64
  %tmp215 = insertelement <128 x float> %tmp214, float %tmp36, i32 65
  %tmp216 = insertelement <128 x float> %tmp215, float %tmp40, i32 66
  %tmp217 = insertelement <128 x float> %tmp216, float %tmp44, i32 67
  %tmp218 = insertelement <128 x float> %tmp217, float %tmp48, i32 68
  %tmp219 = insertelement <128 x float> %tmp218, float %tmp52, i32 69
  %tmp220 = insertelement <128 x float> %tmp219, float %tmp56, i32 70
  %tmp221 = insertelement <128 x float> %tmp220, float %tmp60, i32 71
  %tmp222 = insertelement <128 x float> %tmp221, float %tmp64, i32 72
  %tmp223 = insertelement <128 x float> %tmp222, float %tmp68, i32 73
  %tmp224 = insertelement <128 x float> %tmp223, float %tmp72, i32 74
  %tmp225 = insertelement <128 x float> %tmp224, float %tmp76, i32 75
  %tmp226 = insertelement <128 x float> %tmp225, float %tmp80, i32 76
  %tmp227 = insertelement <128 x float> %tmp226, float %tmp84, i32 77
  %tmp228 = insertelement <128 x float> %tmp227, float %tmp88, i32 78
  %tmp229 = insertelement <128 x float> %tmp228, float %tmp92, i32 79
  %tmp230 = insertelement <128 x float> %tmp229, float %tmp93, i32 80
  %tmp231 = insertelement <128 x float> %tmp230, float %tmp94, i32 81
  %tmp232 = insertelement <128 x float> %tmp231, float %tmp96, i32 82
  %tmp233 = insertelement <128 x float> %tmp232, float %tmp97, i32 83
  %tmp234 = insertelement <128 x float> %tmp233, float %tmp98, i32 84
  %tmp235 = insertelement <128 x float> %tmp234, float %tmp99, i32 85
  %tmp236 = insertelement <128 x float> %tmp235, float %tmp100, i32 86
  %tmp237 = insertelement <128 x float> %tmp236, float %tmp101, i32 87
  %tmp238 = insertelement <128 x float> %tmp237, float %tmp102, i32 88
  %tmp239 = insertelement <128 x float> %tmp238, float %tmp103, i32 89
  %tmp240 = insertelement <128 x float> %tmp239, float %tmp104, i32 90
  %tmp241 = insertelement <128 x float> %tmp240, float %tmp105, i32 91
  %tmp242 = insertelement <128 x float> %tmp241, float %tmp106, i32 92
  %tmp243 = insertelement <128 x float> %tmp242, float %tmp107, i32 93
  %tmp244 = insertelement <128 x float> %tmp243, float %tmp108, i32 94
  %tmp245 = insertelement <128 x float> %tmp244, float %tmp109, i32 95
  %tmp246 = insertelement <128 x float> %tmp245, float %tmp110, i32 96
  %tmp247 = insertelement <128 x float> %tmp246, float %tmp111, i32 97
  %tmp248 = insertelement <128 x float> %tmp247, float %tmp112, i32 98
  %tmp249 = insertelement <128 x float> %tmp248, float %tmp113, i32 99
  %tmp250 = insertelement <128 x float> %tmp249, float %tmp114, i32 100
  %tmp251 = insertelement <128 x float> %tmp250, float %tmp115, i32 101
  %tmp252 = insertelement <128 x float> %tmp251, float %tmp116, i32 102
  %tmp253 = insertelement <128 x float> %tmp252, float %tmp117, i32 103
  %tmp254 = insertelement <128 x float> %tmp253, float %tmp118, i32 104
  %tmp255 = insertelement <128 x float> %tmp254, float %tmp119, i32 105
  %tmp256 = insertelement <128 x float> %tmp255, float %tmp120, i32 106
  %tmp257 = insertelement <128 x float> %tmp256, float %tmp121, i32 107
  %tmp258 = insertelement <128 x float> %tmp257, float %tmp122, i32 108
  %tmp259 = insertelement <128 x float> %tmp258, float %tmp123, i32 109
  %tmp260 = insertelement <128 x float> %tmp259, float %tmp124, i32 110
  %tmp261 = insertelement <128 x float> %tmp260, float %tmp125, i32 111
  %tmp262 = insertelement <128 x float> %tmp261, float %tmp126, i32 112
  %tmp263 = insertelement <128 x float> %tmp262, float %tmp127, i32 113
  %tmp264 = insertelement <128 x float> %tmp263, float %tmp128, i32 114
  %tmp265 = insertelement <128 x float> %tmp264, float %tmp129, i32 115
  %tmp266 = insertelement <128 x float> %tmp265, float %tmp130, i32 116
  %tmp267 = insertelement <128 x float> %tmp266, float %tmp131, i32 117
  %tmp268 = insertelement <128 x float> %tmp267, float %tmp132, i32 118
  %tmp269 = insertelement <128 x float> %tmp268, float %tmp133, i32 119
  %tmp270 = insertelement <128 x float> %tmp269, float %tmp134, i32 120
  %tmp271 = insertelement <128 x float> %tmp270, float %tmp135, i32 121
  %tmp272 = insertelement <128 x float> %tmp271, float %tmp136, i32 122
  %tmp273 = insertelement <128 x float> %tmp272, float %tmp137, i32 123
  %tmp274 = insertelement <128 x float> %tmp273, float %tmp138, i32 124
  %tmp275 = insertelement <128 x float> %tmp274, float %tmp139, i32 125
  %tmp276 = insertelement <128 x float> %tmp275, float %tmp140, i32 126
  %tmp277 = insertelement <128 x float> %tmp276, float %tmp141, i32 127
  %tmp278 = insertelement <128 x float> %tmp277, float %tmp149, i32 %tmp146
  %tmp279 = extractelement <128 x float> %tmp278, i32 0
  %tmp280 = extractelement <128 x float> %tmp278, i32 1
  %tmp281 = extractelement <128 x float> %tmp278, i32 2
  %tmp282 = extractelement <128 x float> %tmp278, i32 3
  %tmp283 = extractelement <128 x float> %tmp278, i32 4
  %tmp284 = extractelement <128 x float> %tmp278, i32 5
  %tmp285 = extractelement <128 x float> %tmp278, i32 6
  %tmp286 = extractelement <128 x float> %tmp278, i32 7
  %tmp287 = extractelement <128 x float> %tmp278, i32 8
  %tmp288 = extractelement <128 x float> %tmp278, i32 9
  %tmp289 = extractelement <128 x float> %tmp278, i32 10
  %tmp290 = extractelement <128 x float> %tmp278, i32 11
  %tmp291 = extractelement <128 x float> %tmp278, i32 12
  %tmp292 = extractelement <128 x float> %tmp278, i32 13
  %tmp293 = extractelement <128 x float> %tmp278, i32 14
  %tmp294 = extractelement <128 x float> %tmp278, i32 15
  %tmp295 = extractelement <128 x float> %tmp278, i32 16
  %tmp296 = extractelement <128 x float> %tmp278, i32 17
  %tmp297 = extractelement <128 x float> %tmp278, i32 18
  %tmp298 = extractelement <128 x float> %tmp278, i32 19
  %tmp299 = extractelement <128 x float> %tmp278, i32 20
  %tmp300 = extractelement <128 x float> %tmp278, i32 21
  %tmp301 = extractelement <128 x float> %tmp278, i32 22
  %tmp302 = extractelement <128 x float> %tmp278, i32 23
  %tmp303 = extractelement <128 x float> %tmp278, i32 24
  %tmp304 = extractelement <128 x float> %tmp278, i32 25
  %tmp305 = extractelement <128 x float> %tmp278, i32 26
  %tmp306 = extractelement <128 x float> %tmp278, i32 27
  %tmp307 = extractelement <128 x float> %tmp278, i32 28
  %tmp308 = extractelement <128 x float> %tmp278, i32 29
  %tmp309 = extractelement <128 x float> %tmp278, i32 30
  %tmp310 = extractelement <128 x float> %tmp278, i32 31
  %tmp311 = extractelement <128 x float> %tmp278, i32 32
  %tmp312 = extractelement <128 x float> %tmp278, i32 33
  %tmp313 = extractelement <128 x float> %tmp278, i32 34
  %tmp314 = extractelement <128 x float> %tmp278, i32 35
  %tmp315 = extractelement <128 x float> %tmp278, i32 36
  %tmp316 = extractelement <128 x float> %tmp278, i32 37
  %tmp317 = extractelement <128 x float> %tmp278, i32 38
  %tmp318 = extractelement <128 x float> %tmp278, i32 39
  %tmp319 = extractelement <128 x float> %tmp278, i32 40
  %tmp320 = extractelement <128 x float> %tmp278, i32 41
  %tmp321 = extractelement <128 x float> %tmp278, i32 42
  %tmp322 = extractelement <128 x float> %tmp278, i32 43
  %tmp323 = extractelement <128 x float> %tmp278, i32 44
  %tmp324 = extractelement <128 x float> %tmp278, i32 45
  %tmp325 = extractelement <128 x float> %tmp278, i32 46
  %tmp326 = extractelement <128 x float> %tmp278, i32 47
  %tmp327 = extractelement <128 x float> %tmp278, i32 48
  %tmp328 = extractelement <128 x float> %tmp278, i32 49
  %tmp329 = extractelement <128 x float> %tmp278, i32 50
  %tmp330 = extractelement <128 x float> %tmp278, i32 51
  %tmp331 = extractelement <128 x float> %tmp278, i32 52
  %tmp332 = extractelement <128 x float> %tmp278, i32 53
  %tmp333 = extractelement <128 x float> %tmp278, i32 54
  %tmp334 = extractelement <128 x float> %tmp278, i32 55
  %tmp335 = extractelement <128 x float> %tmp278, i32 56
  %tmp336 = extractelement <128 x float> %tmp278, i32 57
  %tmp337 = extractelement <128 x float> %tmp278, i32 58
  %tmp338 = extractelement <128 x float> %tmp278, i32 59
  %tmp339 = extractelement <128 x float> %tmp278, i32 60
  %tmp340 = extractelement <128 x float> %tmp278, i32 61
  %tmp341 = extractelement <128 x float> %tmp278, i32 62
  %tmp342 = extractelement <128 x float> %tmp278, i32 63
  %tmp343 = extractelement <128 x float> %tmp278, i32 64
  %tmp344 = extractelement <128 x float> %tmp278, i32 65
  %tmp345 = extractelement <128 x float> %tmp278, i32 66
  %tmp346 = extractelement <128 x float> %tmp278, i32 67
  %tmp347 = extractelement <128 x float> %tmp278, i32 68
  %tmp348 = extractelement <128 x float> %tmp278, i32 69
  %tmp349 = extractelement <128 x float> %tmp278, i32 70
  %tmp350 = extractelement <128 x float> %tmp278, i32 71
  %tmp351 = extractelement <128 x float> %tmp278, i32 72
  %tmp352 = extractelement <128 x float> %tmp278, i32 73
  %tmp353 = extractelement <128 x float> %tmp278, i32 74
  %tmp354 = extractelement <128 x float> %tmp278, i32 75
  %tmp355 = extractelement <128 x float> %tmp278, i32 76
  %tmp356 = extractelement <128 x float> %tmp278, i32 77
  %tmp357 = extractelement <128 x float> %tmp278, i32 78
  %tmp358 = extractelement <128 x float> %tmp278, i32 79
  %tmp359 = extractelement <128 x float> %tmp278, i32 80
  %tmp360 = extractelement <128 x float> %tmp278, i32 81
  %tmp361 = extractelement <128 x float> %tmp278, i32 82
  %tmp362 = extractelement <128 x float> %tmp278, i32 83
  %tmp363 = extractelement <128 x float> %tmp278, i32 84
  %tmp364 = extractelement <128 x float> %tmp278, i32 85
  %tmp365 = extractelement <128 x float> %tmp278, i32 86
  %tmp366 = extractelement <128 x float> %tmp278, i32 87
  %tmp367 = extractelement <128 x float> %tmp278, i32 88
  %tmp368 = extractelement <128 x float> %tmp278, i32 89
  %tmp369 = extractelement <128 x float> %tmp278, i32 90
  %tmp370 = extractelement <128 x float> %tmp278, i32 91
  %tmp371 = extractelement <128 x float> %tmp278, i32 92
  %tmp372 = extractelement <128 x float> %tmp278, i32 93
  %tmp373 = extractelement <128 x float> %tmp278, i32 94
  %tmp374 = extractelement <128 x float> %tmp278, i32 95
  %tmp375 = extractelement <128 x float> %tmp278, i32 96
  %tmp376 = extractelement <128 x float> %tmp278, i32 97
  %tmp377 = extractelement <128 x float> %tmp278, i32 98
  %tmp378 = extractelement <128 x float> %tmp278, i32 99
  %tmp379 = extractelement <128 x float> %tmp278, i32 100
  %tmp380 = extractelement <128 x float> %tmp278, i32 101
  %tmp381 = extractelement <128 x float> %tmp278, i32 102
  %tmp382 = extractelement <128 x float> %tmp278, i32 103
  %tmp383 = extractelement <128 x float> %tmp278, i32 104
  %tmp384 = extractelement <128 x float> %tmp278, i32 105
  %tmp385 = extractelement <128 x float> %tmp278, i32 106
  %tmp386 = extractelement <128 x float> %tmp278, i32 107
  %tmp387 = extractelement <128 x float> %tmp278, i32 108
  %tmp388 = extractelement <128 x float> %tmp278, i32 109
  %tmp389 = extractelement <128 x float> %tmp278, i32 110
  %tmp390 = extractelement <128 x float> %tmp278, i32 111
  %tmp391 = extractelement <128 x float> %tmp278, i32 112
  %tmp392 = extractelement <128 x float> %tmp278, i32 113
  %tmp393 = extractelement <128 x float> %tmp278, i32 114
  %tmp394 = extractelement <128 x float> %tmp278, i32 115
  %tmp395 = extractelement <128 x float> %tmp278, i32 116
  %tmp396 = extractelement <128 x float> %tmp278, i32 117
  %tmp397 = extractelement <128 x float> %tmp278, i32 118
  %tmp398 = extractelement <128 x float> %tmp278, i32 119
  %tmp399 = extractelement <128 x float> %tmp278, i32 120
  %tmp400 = extractelement <128 x float> %tmp278, i32 121
  %tmp401 = extractelement <128 x float> %tmp278, i32 122
  %tmp402 = extractelement <128 x float> %tmp278, i32 123
  %tmp403 = extractelement <128 x float> %tmp278, i32 124
  %tmp404 = extractelement <128 x float> %tmp278, i32 125
  %tmp405 = extractelement <128 x float> %tmp278, i32 126
  %tmp406 = extractelement <128 x float> %tmp278, i32 127
  %tmp407 = bitcast float %tmp95 to i32
  %tmp408 = add i32 %tmp407, 1
  %tmp409 = bitcast i32 %tmp408 to float
  br label %bb12
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
