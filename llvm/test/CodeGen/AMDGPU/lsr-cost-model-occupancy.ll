; RUN: llc -O3 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 < %s | FileCheck %s

; Reduced from the AITER Triton paged-attention _fwd_kernel regressed by the
; GFX9+ LSR cost model (#184138). The old EffInsns-first comparator makes LSR
; add IV registers (260 VGPRs, Occupancy 1); the NumRegs-first fix keeps
; register pressure at 244 VGPRs, restoring Occupancy 2.

; CHECK-LABEL: {{^}}lsr_cost_occupancy:
; CHECK: ; Occupancy: 2

target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @lsr_cost_occupancy(i32 %arg2, ptr addrspace(3) %0, ptr addrspace(3) %1, ptr addrspace(3) %2, ptr addrspace(3) %3, ptr addrspace(3) %4, ptr addrspace(3) %5, ptr addrspace(3) %6, <8 x float> %i469, <4 x float> %i757, <2 x float> %i835, i1 %i1118) #0 {
.lr.ph:
  %i90 = load <8 x half>, ptr addrspace(3) null, align 16
  %i93 = load <8 x half>, ptr addrspace(3) %0, align 16
  %i96 = load <8 x half>, ptr addrspace(3) %1, align 16
  %i99 = load <8 x half>, ptr addrspace(3) %2, align 16
  %i102 = load <8 x half>, ptr addrspace(3) %3, align 16
  %i105 = load <8 x half>, ptr addrspace(3) %4, align 16
  %i108 = load <8 x half>, ptr addrspace(3) %5, align 16
  %i111 = load <8 x half>, ptr addrspace(3) %6, align 16
  %i = tail call i32 @llvm.amdgcn.workitem.id.x()
  %i121 = insertelement <4 x i32> zeroinitializer, i32 %i, i64 0
  %i1224 = shufflevector <4 x i32> %i121, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer
  %i128 = or <4 x i32> %i1224, splat (i32 1)
  %i123 = or <4 x i32> %i1224, <i32 8, i32 9, i32 10, i32 11>
  %i136 = extractelement <4 x i32> %i123, i64 1
  %i413 = and i32 %i, 1
  %i140 = extractelement <4 x i32> %i123, i64 3
  %i124 = insertelement <8 x i32> zeroinitializer, i32 %i, i64 0
  %i125 = shufflevector <8 x i32> %i124, <8 x i32> zeroinitializer, <8 x i32> zeroinitializer
  %i126 = or <8 x i32> %i125, <i32 17, i32 18, i32 19, i32 24, i32 25, i32 26, i32 27, i32 32>
  %i144 = extractelement <8 x i32> %i126, i64 1
  %i146 = extractelement <8 x i32> %i126, i64 2
  %i148 = extractelement <8 x i32> %i126, i64 3
  %i150 = extractelement <8 x i32> %i126, i64 4
  %i152 = extractelement <8 x i32> %i126, i64 5
  %i154 = extractelement <8 x i32> %i126, i64 6
  %i127 = or <8 x i32> %i125, <i32 34, i32 35, i32 40, i32 41, i32 42, i32 43, i32 48, i32 49>
  %i160 = extractelement <8 x i32> %i127, i64 1
  %i162 = extractelement <8 x i32> %i127, i64 2
  %i164 = extractelement <8 x i32> %i127, i64 3
  %i166 = extractelement <8 x i32> %i127, i64 4
  %i168 = extractelement <8 x i32> %i127, i64 5
  %i117 = lshr i32 %i, 3
  br label %bb

bb:                                               ; preds = %bb, %.lr.ph
  %i472 = phi float [ 0.000000e+00, %.lr.ph ], [ %i929, %bb ]
  %i473 = phi float [ 0.000000e+00, %.lr.ph ], [ 1.000000e+00, %bb ]
  %i474 = phi i64 [ 0, %.lr.ph ], [ %i111731, %bb ]
  %i475 = phi <2 x float> [ zeroinitializer, %.lr.ph ], [ %i1119, %bb ]
  %i490 = phi <2 x float> [ zeroinitializer, %.lr.ph ], [ %i1134, %bb ]
  %i25 = or disjoint i32 %i, 1
  %i114 = zext i32 %i25 to i64
  %i504 = add i64 %i114, %i474
  %i20 = or disjoint i32 %arg2, %i
  %i113 = zext i32 %i20 to i64
  %i492 = or disjoint i64 %i474, %i113
  %i497 = icmp slt i64 %i492, 0
  %i503 = add i64 %i113, %i474
  %i507 = trunc i64 %i503 to i32
  %i498 = icmp slt i64 %i474, 1
  %i508 = trunc i64 %i504 to i32
  %i517 = shl i32 %i508, 3
  %i518 = select i1 %i498, i32 %i517, i32 -2147483648
  %i519 = tail call i64 @llvm.amdgcn.raw.ptr.buffer.load.i64(ptr addrspace(8) null, i32 %i518, i32 0, i32 0)
  %i134 = extractelement <4 x i32> %i123, i64 0
  %i135 = zext i32 %i134 to i64
  %i530 = or disjoint i64 %i474, %i135
  %i719 = insertelement <8 x i64> zeroinitializer, i64 %i530, i64 4
  %i137 = zext i32 %i136 to i64
  %i531 = or disjoint i64 %i474, %i137
  %i720 = insertelement <8 x i64> %i719, i64 %i531, i64 5
  %i139 = zext i32 %i413 to i64
  %i532 = or disjoint i64 %i474, %i139
  %i721 = insertelement <8 x i64> %i720, i64 %i532, i64 6
  %i141 = zext i32 %i140 to i64
  %i533 = or disjoint i64 %i474, %i141
  %i722 = insertelement <8 x i64> %i721, i64 %i533, i64 7
  %i723 = icmp slt <8 x i64> %i722, zeroinitializer
  %i142 = extractelement <8 x i32> %i126, i64 0
  %i143 = zext i32 %i142 to i64
  %i537 = or disjoint i64 %i474, %i143
  %i727 = insertelement <8 x i64> zeroinitializer, i64 %i537, i64 0
  %i145 = zext i32 %i144 to i64
  %i538 = or disjoint i64 %i474, %i145
  %i728 = insertelement <8 x i64> %i727, i64 %i538, i64 1
  %i147 = zext i32 %i146 to i64
  %i539 = or disjoint i64 %i474, %i147
  %i729 = insertelement <8 x i64> %i728, i64 %i539, i64 2
  %i149 = zext i32 %i148 to i64
  %i540 = or disjoint i64 %i474, %i149
  %i730 = insertelement <8 x i64> %i729, i64 %i540, i64 3
  %i151 = zext i32 %i150 to i64
  %i541 = or disjoint i64 %i474, %i151
  %i731 = insertelement <8 x i64> %i730, i64 %i541, i64 4
  %i153 = zext i32 %i152 to i64
  %i542 = or disjoint i64 %i474, %i153
  %i732 = insertelement <8 x i64> %i731, i64 %i542, i64 5
  %i155 = zext i32 %i154 to i64
  %i543 = or disjoint i64 %i474, %i155
  %i733 = insertelement <8 x i64> %i732, i64 %i543, i64 6
  %i156 = extractelement <8 x i32> %i126, i64 7
  %i157 = zext i32 %i156 to i64
  %i544 = or disjoint i64 %i474, %i157
  %i158 = extractelement <8 x i32> %i127, i64 0
  %i159 = zext i32 %i158 to i64
  %i545 = or disjoint i64 %i474, %i159
  %i739 = insertelement <8 x i64> zeroinitializer, i64 %i545, i64 0
  %i161 = zext i32 %i160 to i64
  %i546 = or disjoint i64 %i474, %i161
  %i740 = insertelement <8 x i64> %i739, i64 %i546, i64 1
  %i163 = zext i32 %i162 to i64
  %i547 = or disjoint i64 %i474, %i163
  %i741 = insertelement <8 x i64> %i740, i64 %i547, i64 2
  %i165 = zext i32 %i164 to i64
  %i548 = or disjoint i64 %i474, %i165
  %i742 = insertelement <8 x i64> %i741, i64 %i548, i64 3
  %i167 = zext i32 %i166 to i64
  %i549 = or disjoint i64 %i474, %i167
  %i743 = insertelement <8 x i64> %i742, i64 %i549, i64 4
  %i169 = zext i32 %i168 to i64
  %i550 = or disjoint i64 %i474, %i169
  %i744 = insertelement <8 x i64> %i743, i64 %i550, i64 5
  %i171 = zext i32 %i117 to i64
  %i551 = or disjoint i64 %i474, %i171
  %i745 = insertelement <8 x i64> %i744, i64 %i551, i64 6
  %i173 = zext i32 %i to i64
  %i552 = or disjoint i64 %i474, %i173
  %i534 = insertelement <4 x i64> zeroinitializer, i64 %i474, i64 0
  %i53515 = shufflevector <4 x i64> %i534, <4 x i64> zeroinitializer, <4 x i32> zeroinitializer
  %i174 = zext <4 x i32> %i121 to <4 x i64>
  %i553 = or disjoint <4 x i64> %i53515, %i174
  %i758 = fmul <4 x float> zeroinitializer, %i757
  %i751 = icmp slt <4 x i64> %i553, zeroinitializer
  %i753 = select <4 x i1> %i751, <4 x float> zeroinitializer, <4 x float> splat (float +qnan)
  %i807 = extractelement <4 x float> %i753, i64 2
  %i809 = extractelement <4 x float> %i753, i64 1
  %i810 = extractelement <4 x float> %i758, i64 0
  %i811 = fmul float %i809, %i810
  %i818 = fmul float %i807, %i811
  %i82220 = bitcast float %i818 to i32
  %i823 = tail call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %i82220, i32 0, i1 false, i1 false)
  %i824 = extractvalue { i32, i32 } %i823, 0
  %i826 = bitcast i32 %i824 to float
  %i82821 = tail call float @llvm.maxnum.f32(float %i826, float 0.000000e+00)
  %i725 = select <8 x i1> %i723, <8 x float> zeroinitializer, <8 x float> splat (float -inf)
  %i72617 = fmul <8 x float> %i469, %i725
  %i829 = shufflevector <8 x float> %i72617, <8 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %i830 = insertelement <2 x float> zeroinitializer, float %i82821, i64 0
  %i83122 = shufflevector <2 x float> %i830, <2 x float> zeroinitializer, <2 x i32> zeroinitializer
  %i832 = fsub <2 x float> %i829, %i83122
  %i833 = fmul <2 x float> %i832, zeroinitializer
  %i834 = tail call <2 x float> @llvm.exp2.v2f32(<2 x float> %i833)
  %i836 = fsub <2 x float> %i835, %i83122
  %i837 = fmul <2 x float> %i836, zeroinitializer
  %i838 = tail call <2 x float> @llvm.exp2.v2f32(<2 x float> %i837)
  %i839 = shufflevector <8 x float> %i72617, <8 x float> zeroinitializer, <2 x i32> <i32 4, i32 5>
  %i840 = fsub <2 x float> %i839, splat (float 1.000000e+00)
  %i841 = fmul <2 x float> %i840, zeroinitializer
  %i842 = tail call <2 x float> @llvm.exp2.v2f32(<2 x float> %i841)
  %i843 = shufflevector <8 x float> %i72617, <8 x float> zeroinitializer, <2 x i32> <i32 6, i32 7>
  %i844 = fsub <2 x float> %i843, %i83122
  %i845 = fmul <2 x float> %i844, zeroinitializer
  %i846 = tail call <2 x float> @llvm.exp2.v2f32(<2 x float> %i845)
  %i734 = insertelement <8 x i64> %i733, i64 %i544, i64 7
  %i735 = icmp slt <8 x i64> %i734, zeroinitializer
  %i737 = select <8 x i1> %i735, <8 x float> zeroinitializer, <8 x float> splat (float 1.000000e+00)
  %i848 = shufflevector <8 x float> zeroinitializer, <8 x float> %i737, <2 x i32> <i32 0, i32 8>
  %i849 = fsub <2 x float> %i848, %i83122
  %i850 = fmul <2 x float> %i849, zeroinitializer
  %i851 = tail call <2 x float> @llvm.exp2.v2f32(<2 x float> %i850)
  %i738 = fmul <8 x float> %i469, %i737
  %i852 = shufflevector <8 x float> %i738, <8 x float> zeroinitializer, <2 x i32> <i32 1, i32 2>
  %i853 = fsub <2 x float> %i852, splat (float 1.000000e+00)
  %i854 = fmul <2 x float> %i853, zeroinitializer
  %i855 = tail call <2 x float> @llvm.exp2.v2f32(<2 x float> %i854)
  %i856 = shufflevector <8 x float> %i737, <8 x float> zeroinitializer, <2 x i32> <i32 3, i32 4>
  %i857 = fsub <2 x float> %i856, %i83122
  %i858 = fmul <2 x float> %i857, zeroinitializer
  %i859 = tail call <2 x float> @llvm.exp2.v2f32(<2 x float> %i858)
  %i860 = shufflevector <8 x float> %i738, <8 x float> zeroinitializer, <2 x i32> <i32 5, i32 6>
  %i861 = fsub <2 x float> %i860, splat (float 1.000000e+00)
  %i862 = fmul <2 x float> %i861, zeroinitializer
  %i863 = tail call <2 x float> @llvm.exp2.v2f32(<2 x float> %i862)
  %i847 = shufflevector <4 x float> %i758, <4 x float> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %i864 = shufflevector <8 x float> %i738, <8 x float> %i847, <2 x i32> <i32 7, i32 9>
  %i865 = fsub <2 x float> %i864, %i83122
  %i866 = fmul <2 x float> %i865, zeroinitializer
  %i867 = tail call <2 x float> @llvm.exp2.v2f32(<2 x float> %i866)
  %i746 = insertelement <8 x i64> %i745, i64 %i552, i64 7
  %i747 = icmp slt <8 x i64> %i746, zeroinitializer
  %i749 = select <8 x i1> %i747, <8 x float> zeroinitializer, <8 x float> splat (float 1.000000e+00)
  %i868 = shufflevector <8 x float> %i749, <8 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %i869 = fsub <2 x float> %i868, %i83122
  %i870 = fmul <2 x float> %i869, zeroinitializer
  %i871 = tail call <2 x float> @llvm.exp2.v2f32(<2 x float> %i870)
  %i872 = shufflevector <8 x float> %i749, <8 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  %i873 = fsub <2 x float> %i872, %i83122
  %i874 = fmul <2 x float> %i873, zeroinitializer
  %i875 = tail call <2 x float> @llvm.exp2.v2f32(<2 x float> %i874)
  %i876 = shufflevector <8 x float> %i749, <8 x float> zeroinitializer, <2 x i32> <i32 4, i32 5>
  %i877 = fsub <2 x float> %i876, %i83122
  %i878 = fmul <2 x float> %i877, zeroinitializer
  %i879 = tail call <2 x float> @llvm.exp2.v2f32(<2 x float> %i878)
  %i880 = shufflevector <8 x float> %i749, <8 x float> zeroinitializer, <2 x i32> <i32 6, i32 7>
  %i881 = fsub <2 x float> %i880, %i83122
  %i882 = fmul <2 x float> %i881, zeroinitializer
  %i883 = tail call <2 x float> @llvm.exp2.v2f32(<2 x float> %i882)
  %i884 = shufflevector <4 x float> splat (float 1.000000e+00), <4 x float> %i753, <2 x i32> <i32 2, i32 4>
  %i885 = fsub <2 x float> %i884, %i83122
  %i886 = fmul <2 x float> %i885, zeroinitializer
  %i887 = tail call <2 x float> @llvm.exp2.v2f32(<2 x float> %i886)
  %i900 = fadd <2 x float> %i867, %i871
  %i901 = fadd <2 x float> %i875, %i879
  %i896 = fadd <2 x float> %i834, %i838
  %i897 = fadd <2 x float> %i842, %i846
  %i904 = fadd <2 x float> %i896, %i897
  %i898 = fadd <2 x float> %i851, %i855
  %i899 = fadd <2 x float> %i859, %i863
  %i905 = fadd <2 x float> %i898, %i899
  %i908 = fadd <2 x float> %i904, %i905
  %i906 = fadd <2 x float> %i900, %i901
  %i902 = fadd <2 x float> %i883, %i887
  %i909 = fadd <2 x float> %i906, %i902
  %i910 = fadd <2 x float> %i908, %i909
  %foldExtExtBinop215 = fadd <2 x float> %i910, %i902
  %bc = bitcast <2 x float> %foldExtExtBinop215 to <2 x i32>
  %i91124 = extractelement <2 x i32> %bc, i64 0
  %i912 = tail call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %i91124, i32 0, i1 false, i1 false)
  %i919 = insertelement <2 x float> zeroinitializer, float %i473, i64 0
  %i920 = insertelement <2 x float> %i919, float %i82821, i64 1
  %i913 = extractvalue { i32, i32 } %i912, 0
  %i915 = bitcast i32 %i913 to float
  %i924 = fmul <2 x float> %i920, zeroinitializer
  %i926 = insertelement <2 x float> zeroinitializer, float %i472, i64 0
  %i927 = insertelement <2 x float> %i926, float %i915, i64 1
  %i925 = tail call <2 x float> @llvm.exp2.v2f32(<2 x float> %i924)
  %i928 = fmul <2 x float> %i927, %i925
  %shift217 = shufflevector <2 x float> %i928, <2 x float> zeroinitializer, <2 x i32> <i32 1, i32 poison>
  %foldExtExtBinop218 = fadd <2 x float> %i928, %shift217
  %i929 = extractelement <2 x float> %foldExtExtBinop218, i64 0
  tail call void @llvm.amdgcn.s.waitcnt(i32 0)
  %i954 = load <1 x float>, ptr addrspace(3) null, align 4
  %i514 = shl i32 %i507, 3
  %i515 = select i1 %i497, i32 %i514, i32 0
  %i969 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 %i515, i32 0, i32 0)
  %i958 = trunc i64 %i519 to i32
  %i973 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 %i958, i32 0, i32 0)
  %i97026 = bitcast <4 x i32> %i969 to <8 x half>
  %i1055 = shufflevector <8 x half> %i97026, <8 x half> zeroinitializer, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %i430 = lshr i32 %i, 1
  %i4329 = getelementptr i8, ptr addrspace(3) null, i32 %i430
  store <4 x half> %i1055, ptr addrspace(3) %i4329, align 8
  store <4 x half> splat (half +qnan), ptr addrspace(3) %1, align 8
  store <4 x half> zeroinitializer, ptr addrspace(3) %0, align 8
  %i436 = xor i32 %i430, 1
  %i43712 = getelementptr i8, ptr addrspace(3) null, i32 %i436
  store <4 x half> splat (half 1.000000e+00), ptr addrspace(3) %i43712, align 8
  %i974 = bitcast <4 x i32> %i973 to <8 x half>
  %i1060 = shufflevector <8 x half> %i974, <8 x half> zeroinitializer, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  store <4 x half> %i1060, ptr addrspace(3) null, align 8
  %i1094 = shufflevector <2 x float> zeroinitializer, <2 x float> %i490, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %i1096 = shufflevector <4 x float> zeroinitializer, <4 x float> %i1094, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %i1097 = shufflevector <8 x float> zeroinitializer, <8 x float> %i1096, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %i1098 = shufflevector <1 x float> %i954, <1 x float> zeroinitializer, <16 x i32> zeroinitializer
  %i1099 = fmul <16 x float> %i1097, %i1098
  %i1100 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <16 x float> %i1099, i32 0, i32 0, i32 0)
  %i1101 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <16 x float> %i1100, i32 0, i32 0, i32 0)
  %i1104 = shufflevector <2 x float> %i475, <2 x float> zeroinitializer, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %i1108 = shufflevector <4 x float> %i1104, <4 x float> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %i1110 = shufflevector <8 x float> %i1108, <8 x float> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %i1113 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <16 x float> %i1110, i32 0, i32 0, i32 0)
  %i1114 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <16 x float> %i1113, i32 0, i32 0, i32 0)
  %i1115 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <16 x float> %i1114, i32 0, i32 0, i32 0)
  %i111731 = add i64 %i474, 1
  %i1119 = shufflevector <16 x float> %i1115, <16 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %i1134 = shufflevector <16 x float> %i1101, <16 x float> zeroinitializer, <2 x i32> <i32 14, i32 15>
  br i1 %i1118, label %bb, label %._crit_edge

._crit_edge:                                      ; preds = %bb
  %i1388 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> zeroinitializer, <8 x half> %i90, <16 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %i1389 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> zeroinitializer, <8 x half> %i93, <16 x float> %i1388, i32 0, i32 0, i32 0)
  %i1390 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> zeroinitializer, <8 x half> %i96, <16 x float> %i1389, i32 0, i32 0, i32 0)
  %i1391 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> zeroinitializer, <8 x half> %i99, <16 x float> %i1390, i32 0, i32 0, i32 0)
  %i1324 = shufflevector <4 x i32> %i123, <4 x i32> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %i1325 = shufflevector <8 x i32> zeroinitializer, <8 x i32> %i1324, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  %i1392 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> zeroinitializer, <8 x half> %i102, <16 x float> %i1391, i32 0, i32 0, i32 0)
  %i1393 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> zeroinitializer, <8 x half> %i105, <16 x float> %i1392, i32 0, i32 0, i32 0)
  %i1394 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> zeroinitializer, <8 x half> %i108, <16 x float> %i1393, i32 0, i32 0, i32 0)
  %i1398 = icmp slt <8 x i32> zeroinitializer, %i1325
  %i1399 = select <8 x i1> %i1398, <8 x float> splat (float 1.000000e+00), <8 x float> zeroinitializer
  %i1402 = icmp slt <8 x i32> splat (i32 1), %i126
  %i1403 = select <8 x i1> %i1402, <8 x float> splat (float 1.000000e+00), <8 x float> zeroinitializer
  %i1441 = extractelement <8 x float> %i1403, i64 6
  %i1442 = extractelement <8 x float> %i1403, i64 7
  %i1406 = icmp slt <8 x i32> zeroinitializer, %i125
  %i1407 = select <8 x i1> %i1406, <8 x float> splat (float 1.000000e+00), <8 x float> zeroinitializer
  %i1456 = extractelement <8 x float> %i1407, i64 6
  %i1457 = extractelement <8 x float> %i1407, i64 7
  %i1410 = icmp slt <4 x i32> splat (i32 1), %i128
  %i1395 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> zeroinitializer, <8 x half> %i111, <16 x float> %i1394, i32 0, i32 0, i32 0)
  %i1408 = shufflevector <16 x float> %i1395, <16 x float> zeroinitializer, <4 x i32> <i32 11, i32 12, i32 13, i32 14>
  %i1411 = select <4 x i1> %i1410, <4 x float> zeroinitializer, <4 x float> %i1408
  %i1461 = extractelement <4 x float> %i1411, i64 0
  %i1462 = extractelement <4 x float> %i1411, i64 1
  %i1431 = extractelement <8 x float> %i1403, i64 0
  %i1432 = extractelement <8 x float> %i1403, i64 1
  %i1433 = fmul float %i1431, %i1432
  %i1434 = extractelement <8 x float> %i1403, i64 2
  %i1435 = fmul float %i1433, %i1434
  %i1436 = extractelement <8 x float> %i1403, i64 3
  %i1437 = extractelement <8 x float> %i1403, i64 4
  %i1438 = fmul float %i1436, %i1437
  %i1439 = extractelement <8 x float> %i1403, i64 5
  %i1440 = fmul float %i1438, %i1439
  %i1446 = extractelement <8 x float> %i1407, i64 0
  %i1447 = extractelement <8 x float> %i1407, i64 1
  %i1448 = fmul float %i1446, %i1447
  %i1449 = extractelement <8 x float> %i1407, i64 2
  %i1450 = fmul float %i1448, %i1449
  %i1451 = extractelement <8 x float> %i1407, i64 3
  %i1452 = extractelement <8 x float> %i1407, i64 4
  %i1453 = fmul float %i1451, %i1452
  %i1454 = extractelement <8 x float> %i1407, i64 5
  %i1455 = fmul float %i1453, %i1454
  %i1422 = extractelement <8 x float> %i1399, i64 4
  %i1424 = extractelement <8 x float> %i1399, i64 5
  %i1425 = fmul float %i1422, %i1424
  %i1426 = extractelement <8 x float> %i1399, i64 6
  %i1427 = extractelement <8 x float> %i1399, i64 7
  %i1428 = fmul float %i1426, %i1427
  %i1470 = fmul float %i1425, %i1428
  %i1471 = fmul float %i1435, %i1440
  %i1443 = fmul float %i1441, %i1442
  %i1472 = fmul float %i1471, %i1443
  %i1476 = fmul float %i1470, %i1472
  %i1473 = fmul float %i1450, %i1455
  %i1458 = fmul float %i1456, %i1457
  %i1474 = fmul float %i1473, %i1458
  %i1477 = fmul float %i1476, %i1474
  %i1463 = fmul float %i1461, %i1462
  %i1464 = extractelement <4 x float> %i1411, i64 2
  %i1465 = fmul float %i1463, %i1464
  %i1478 = fmul float %i1477, %i1465
  %i1610 = insertelement <1 x float> zeroinitializer, float %i1478, i64 0
  store <1 x float> %i1610, ptr addrspace(3) null, align 4
  ret void
}

declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

declare <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #2

declare void @llvm.amdgcn.s.waitcnt(i32 immarg) #3

declare { i32, i32 } @llvm.amdgcn.permlane32.swap(i32, i32, i1 immarg, i1 immarg) #4

declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half>, <8 x half>, <16 x float>, i32 immarg, i32 immarg, i32 immarg) #5

declare float @llvm.maxnum.f32(float, float) #6

declare i64 @llvm.amdgcn.raw.ptr.buffer.load.i64(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #2

declare <2 x float> @llvm.exp2.v2f32(<2 x float>) #6

attributes #0 = { nofree norecurse nounwind "amdgpu-agpr-alloc"="0" "amdgpu-cluster-dims"="1,1,1" "amdgpu-flat-work-group-size"="1,256" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" "amdgpu-waves-per-eu"="0, 0" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #3 = { nocallback nofree nounwind willreturn }
attributes #4 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #5 = { convergent nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #6 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
