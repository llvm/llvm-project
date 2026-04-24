; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -pass-remarks-analysis=kernel-resource-usage < %s 2>&1 | FileCheck %s
;
; Show reduction in VGPR spilling from avoiding to use subregister sources in the coalescable copy
; optimization for REG_SQUENCE instructions.

; CHECK: ScratchSize [bytes/lane]: 24{{$}}
; CHECK: VGPRs Spill: 5{{$}}


define amdgpu_kernel void @snork(ptr %arg4, ptr addrspace(3) %arg21, ptr addrspace(3) %arg26, ptr addrspace(3) %arg30, ptr addrspace(1) %arg33, i32 %arg34, i32 %arg35, <8 x i32> %arg37, i32 %arg38, i32 %arg39, i32 %arg40, i32 %arg41, <2 x i128> %arg42, i32 %arg43, i32 %arg44, <2 x i128> %arg46, i32 %arg47, i32 %arg48, i32 %arg49, i32 %arg50, i1 %arg52) #0 {
bb:
  %call = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
  %lshr = lshr i32 %call, 1
  %shl = shl nuw nsw i32 %call, 10
  %and = and i32 %shl, 1040384
  %shl61 = shl nuw nsw i32 %call, 1
  %zext62 = zext nneg i32 %shl61 to i64
  %getelementptr74 = getelementptr inbounds nuw i8, ptr %arg4, i64 %zext62
  %ptrtoint75 = ptrtoint ptr %getelementptr74 to i64
  %trunc76 = trunc i64 %ptrtoint75 to i32
  %add77 = add i32 %trunc76, 1
  %getelementptr82 = getelementptr inbounds nuw i8, ptr addrspace(3) null, i32 %lshr
  %addrspacecast = addrspacecast ptr addrspace(3) %getelementptr82 to ptr
  %ptrtoint83 = ptrtoint ptr %addrspacecast to i64
  %trunc84 = trunc i64 %ptrtoint83 to i32
  %inttoptr = inttoptr i32 %trunc84 to ptr addrspace(3)
  %getelementptr85 = getelementptr inbounds nuw i8, ptr %addrspacecast, i64 8192
  %ptrtoint86 = ptrtoint ptr %getelementptr85 to i64
  %trunc87 = trunc i64 %ptrtoint86 to i32
  %inttoptr88 = inttoptr i32 %trunc87 to ptr addrspace(3)
  %getelementptr99 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg21, i32 %lshr
  %addrspacecast100 = addrspacecast ptr addrspace(3) %getelementptr99 to ptr
  %ptrtoint101 = ptrtoint ptr %addrspacecast100 to i64
  %trunc102 = trunc i64 %ptrtoint101 to i32
  %inttoptr103 = inttoptr i32 %trunc102 to ptr addrspace(3)
  %getelementptr104 = getelementptr inbounds nuw i8, ptr %addrspacecast100, i64 8192
  %ptrtoint105 = ptrtoint ptr %getelementptr104 to i64
  %trunc106 = trunc i64 %ptrtoint105 to i32
  %inttoptr107 = inttoptr i32 %trunc106 to ptr addrspace(3)
  %getelementptr113 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg26, i32 %lshr
  %addrspacecast114 = addrspacecast ptr addrspace(3) %getelementptr113 to ptr
  %ptrtoint115 = ptrtoint ptr %addrspacecast114 to i64
  %trunc116 = trunc i64 %ptrtoint115 to i32
  %inttoptr117 = inttoptr i32 %trunc116 to ptr addrspace(3)
  %getelementptr118 = getelementptr inbounds nuw i8, ptr %addrspacecast114, i64 8192
  %ptrtoint119 = ptrtoint ptr %getelementptr118 to i64
  %trunc120 = trunc i64 %ptrtoint119 to i32
  %inttoptr121 = inttoptr i32 %trunc120 to ptr addrspace(3)
  %getelementptr127 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg30, i32 %lshr
  %addrspacecast128 = addrspacecast ptr addrspace(3) %getelementptr127 to ptr
  %ptrtoint129 = ptrtoint ptr %addrspacecast128 to i64
  %trunc130 = trunc i64 %ptrtoint129 to i32
  %inttoptr131 = inttoptr i32 %trunc130 to ptr addrspace(3)
  %getelementptr132 = getelementptr inbounds nuw i8, ptr %addrspacecast128, i64 8192
  %ptrtoint133 = ptrtoint ptr %getelementptr132 to i64
  %trunc134 = trunc i64 %ptrtoint133 to i32
  %inttoptr135 = inttoptr i32 %trunc134 to ptr addrspace(3)
  br label %bb166

bb136:                                            ; preds = %bb166
  %and57 = and i32 %call, 48
  %and141 = shl nuw nsw i32 %call, 1
  %shl142 = and i32 %and141, 2
  %or143 = or disjoint i32 %shl142, %and57
  %call148 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 1, i32 0) #6
  %call151 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 1, i32 0) #6
  %call153 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 1, i32 0) #6
  %add154 = add i32 %or143, %trunc76
  %call155 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 %add154, i32 0) #6
  %call157 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 1, i32 0) #6
  %call161 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 1, i32 0) #6
  store i16 0, ptr addrspace(1) %arg33, align 2
  %call165 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 1, i32 0) #6
  ret void

bb166:                                            ; preds = %bb166, %bb
  %phi167 = phi <4 x float> [ zeroinitializer, %bb ], [ %call312, %bb166 ]
  %phi168 = phi <4 x float> [ zeroinitializer, %bb ], [ %call313, %bb166 ]
  %phi169 = phi <4 x float> [ zeroinitializer, %bb ], [ %call314, %bb166 ]
  %phi170 = phi <4 x float> [ zeroinitializer, %bb ], [ %call315, %bb166 ]
  %phi171 = phi <4 x float> [ zeroinitializer, %bb ], [ %call316, %bb166 ]
  %phi172 = phi <4 x float> [ zeroinitializer, %bb ], [ %call317, %bb166 ]
  %phi173 = phi <4 x float> [ zeroinitializer, %bb ], [ %call318, %bb166 ]
  %phi174 = phi <4 x float> [ zeroinitializer, %bb ], [ %call304, %bb166 ]
  %phi175 = phi <4 x float> [ zeroinitializer, %bb ], [ %call305, %bb166 ]
  %phi176 = phi <4 x float> [ zeroinitializer, %bb ], [ %call307, %bb166 ]
  %phi177 = phi <4 x float> [ zeroinitializer, %bb ], [ %call308, %bb166 ]
  %phi178 = phi <4 x float> [ zeroinitializer, %bb ], [ %call309, %bb166 ]
  %phi179 = phi <4 x float> [ zeroinitializer, %bb ], [ %call310, %bb166 ]
  %phi180 = phi <4 x float> [ zeroinitializer, %bb ], [ %call311, %bb166 ]
  %phi181 = phi <4 x float> [ zeroinitializer, %bb ], [ %call289, %bb166 ]
  %phi182 = phi <4 x float> [ zeroinitializer, %bb ], [ %call290, %bb166 ]
  %phi183 = phi <4 x float> [ zeroinitializer, %bb ], [ %call291, %bb166 ]
  %phi184 = phi <4 x float> [ zeroinitializer, %bb ], [ %call292, %bb166 ]
  %phi185 = phi <4 x float> [ zeroinitializer, %bb ], [ %call293, %bb166 ]
  %phi186 = phi <4 x float> [ zeroinitializer, %bb ], [ %call294, %bb166 ]
  %phi187 = phi <4 x float> [ zeroinitializer, %bb ], [ %call295, %bb166 ]
  %phi188 = phi <4 x float> [ zeroinitializer, %bb ], [ %call277, %bb166 ]
  %phi189 = phi <4 x float> [ zeroinitializer, %bb ], [ %call278, %bb166 ]
  %phi190 = phi <4 x float> [ zeroinitializer, %bb ], [ %call280, %bb166 ]
  %phi191 = phi <4 x float> [ zeroinitializer, %bb ], [ %call281, %bb166 ]
  %phi192 = phi <4 x float> [ zeroinitializer, %bb ], [ %call283, %bb166 ]
  %phi193 = phi <4 x float> [ zeroinitializer, %bb ], [ %call284, %bb166 ]
  %call194 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 %arg34, i32 0) #6
  %call195 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 0, i32 0) #6
  %call196 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 1, i32 0) #6
  %call197 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 0, i32 0) #6
  %call198 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 %arg35, i32 0) #6
  %call199 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 0, i32 0) #6
  %call200 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 0, i32 0) #6
  %call201 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 0, i32 0) #6
  %insertelement202 = insertelement <2 x i128> <i128 poison, i128 0>, i128 %call201, i64 0
  %call203 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 0, i32 0) #6
  %insertelement204 = insertelement <2 x i128> <i128 poison, i128 0>, i128 %call203, i64 0
  %call205 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %phi188, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call206 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> %arg37, <4 x float> %phi189, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %bitcast207 = bitcast <2 x i128> %insertelement202 to <8 x i32>
  %call208 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %bitcast207, <8 x i32> zeroinitializer, <4 x float> %phi190, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call209 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %phi191, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %bitcast210 = bitcast <2 x i128> %insertelement204 to <8 x i32>
  %call211 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %bitcast210, <8 x i32> zeroinitializer, <4 x float> %phi192, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call212 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %phi193, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call213 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 %arg38, i32 0) #6
  %insertelement214 = insertelement <2 x i128> poison, i128 %call213, i64 0
  %call215 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 0, i32 0) #6
  %call216 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 %arg39, i32 0) #6
  %insertelement217 = insertelement <2 x i128> %insertelement214, i128 %call216, i64 1
  %call218 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 0, i32 0) #6
  %insertelement219 = insertelement <2 x i128> <i128 poison, i128 0>, i128 %call218, i64 0
  %bitcast220 = bitcast <2 x i128> %insertelement217 to <8 x i32>
  %bitcast221 = bitcast <2 x i128> %insertelement219 to <8 x i32>
  %call222 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> %bitcast221, <4 x float> %phi181, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call223 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %phi182, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call224 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %phi183, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call225 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %bitcast207, <8 x i32> zeroinitializer, <4 x float> %phi184, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call226 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %phi185, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call227 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %phi186, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call228 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %bitcast210, <8 x i32> zeroinitializer, <4 x float> %phi187, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call229 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 %add77, i32 0) #6
  %call230 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 0, i32 0) #6
  %call231 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 0, i32 0) #6
  %insertelement232 = insertelement <2 x i128> <i128 0, i128 poison>, i128 %call231, i64 1
  %call233 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 0, i32 0) #6
  %insertelement234 = insertelement <2 x i128> <i128 0, i128 poison>, i128 %call233, i64 1
  %call235 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 %arg40, i32 0) #6
  %insertelement236 = insertelement <2 x i128> <i128 0, i128 poison>, i128 %call235, i64 1
  %call237 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 0, i32 0) #6
  %insertelement238 = insertelement <2 x i128> <i128 0, i128 poison>, i128 %call237, i64 1
  tail call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> noundef zeroinitializer, ptr addrspace(3) noundef %inttoptr, i32 noundef 1, i32 noundef %and, i32 noundef 0, i32 noundef 0, i32 noundef 0) #6
  tail call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> noundef zeroinitializer, ptr addrspace(3) noundef %inttoptr88, i32 noundef 1, i32 noundef 524288, i32 noundef 0, i32 noundef 0, i32 noundef 0) #6
  tail call void @llvm.amdgcn.s.barrier()
  tail call void asm sideeffect "s_waitcnt lgkmcnt(0)", ""() #6
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  %bitcast239 = bitcast <2 x i128> %insertelement232 to <8 x i32>
  %call240 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> splat (i32 1), <8 x i32> zeroinitializer, <4 x float> %phi174, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call241 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %phi175, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %bitcast242 = bitcast <2 x i128> %insertelement234 to <8 x i32>
  %call243 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %bitcast242, <8 x i32> zeroinitializer, <4 x float> %phi176, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call244 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> %arg37, <4 x float> %phi177, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %bitcast245 = bitcast <2 x i128> %insertelement236 to <8 x i32>
  %call246 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %bitcast245, <8 x i32> zeroinitializer, <4 x float> %phi178, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %bitcast247 = bitcast <2 x i128> %insertelement238 to <8 x i32>
  %call248 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %bitcast247, <8 x i32> zeroinitializer, <4 x float> %phi179, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call249 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %phi180, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call250 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %bitcast239, <8 x i32> %bitcast221, <4 x float> %phi167, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call251 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %bitcast242, <8 x i32> %bitcast220, <4 x float> %phi168, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call252 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %phi169, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call253 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %phi170, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call254 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %phi171, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call255 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %phi172, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call256 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %bitcast247, <8 x i32> zeroinitializer, <4 x float> %phi173, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  %call257 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 %arg41, i32 0) #6
  %call259 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 0, i32 0) #6
  %call261 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 %arg43, i32 0) #6
  %call263 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 %arg44, i32 0) #6
  %call264 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 1, i32 0) #6
  %insertelement265 = insertelement <2 x i128> <i128 poison, i128 0>, i128 %call264, i64 0
  %call266 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 0, i32 0) #6
  %call268 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 0, i32 0) #6
  %insertelement269 = insertelement <2 x i128> <i128 poison, i128 0>, i128 %call268, i64 0
  %call270 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 0, i32 0) #6
  %insertelement271 = insertelement <2 x i128> <i128 poison, i128 0>, i128 %call270, i64 0
  tail call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> noundef zeroinitializer, ptr addrspace(3) noundef %inttoptr103, i32 noundef 1, i32 noundef 0, i32 noundef 0, i32 noundef 0, i32 noundef 0) #6
  tail call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> noundef zeroinitializer, ptr addrspace(3) noundef %inttoptr107, i32 noundef 1, i32 noundef 0, i32 noundef 0, i32 noundef 0, i32 noundef 0) #6
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  %bitcast273 = bitcast <2 x i128> %insertelement265 to <8 x i32>
  %bitcast274 = bitcast <2 x i128> %arg46 to <8 x i32>
  %bitcast275 = bitcast <2 x i128> %arg42 to <8 x i32>
  %bitcast276 = bitcast <2 x i128> %arg42 to <8 x i32>
  %call277 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> splat (i32 1), <8 x i32> %bitcast274, <4 x float> %call205, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call278 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> %bitcast275, <4 x float> %call206, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %bitcast279 = bitcast <2 x i128> %insertelement269 to <8 x i32>
  %call280 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %call208, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call281 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %call209, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %bitcast282 = bitcast <2 x i128> %insertelement271 to <8 x i32>
  %call283 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %bitcast282, <8 x i32> zeroinitializer, <4 x float> %call211, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call284 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %call212, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  %call285 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 %arg47, i32 0) #6
  %call286 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 %arg48, i32 0) #6
  %insertelement287 = insertelement <2 x i128> <i128 poison, i128 0>, i128 %call286, i64 0
  tail call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> noundef zeroinitializer, ptr addrspace(3) noundef %inttoptr117, i32 noundef 1, i32 noundef 0, i32 noundef 0, i32 noundef 0, i32 noundef 0) #6
  tail call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> noundef zeroinitializer, ptr addrspace(3) noundef %inttoptr121, i32 noundef 1, i32 noundef 524288, i32 noundef 0, i32 noundef 0, i32 noundef 0) #6
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  %bitcast288 = bitcast <2 x i128> %insertelement287 to <8 x i32>
  %call289 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %bitcast273, <8 x i32> zeroinitializer, <4 x float> %call222, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call290 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %bitcast276, <8 x i32> %bitcast288, <4 x float> %call223, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call291 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %call224, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call292 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %call225, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call293 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %bitcast279, <8 x i32> zeroinitializer, <4 x float> %call226, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call294 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %call227, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call295 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %bitcast282, <8 x i32> zeroinitializer, <4 x float> %call228, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.s.barrier()
  %call296 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 %arg49, i32 0) #6
  %call297 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 0, i32 0) #6
  %call298 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 %arg50, i32 0) #6
  %insertelement299 = insertelement <2 x i128> <i128 poison, i128 0>, i128 %call298, i64 0
  %call300 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 0, i32 0) #6
  %insertelement301 = insertelement <2 x i128> <i128 poison, i128 0>, i128 %call300, i64 0
  %call302 = tail call i128 asm sideeffect "ds_read_b128 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 0, i32 0) #6
  tail call void @llvm.amdgcn.s.barrier()
  tail call void asm sideeffect "s_waitcnt lgkmcnt(0)", ""() #6
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  %bitcast303 = bitcast <2 x i128> %insertelement299 to <8 x i32>
  %call304 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %bitcast303, <8 x i32> zeroinitializer, <4 x float> %call240, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call305 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %call241, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %bitcast306 = bitcast <2 x i128> %insertelement301 to <8 x i32>
  %call307 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> splat (i32 1), <8 x i32> zeroinitializer, <4 x float> %call243, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call308 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %call244, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call309 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> splat (i32 1), <4 x float> %call246, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call310 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> %bitcast274, <4 x float> %call248, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call311 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %call249, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.s.barrier()
  tail call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> noundef zeroinitializer, ptr addrspace(3) noundef %inttoptr131, i32 noundef 1, i32 noundef 0, i32 noundef 0, i32 noundef 0, i32 noundef 0) #6
  tail call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> noundef zeroinitializer, ptr addrspace(3) noundef %inttoptr135, i32 noundef 1, i32 noundef 0, i32 noundef 0, i32 noundef 0, i32 noundef 0) #6
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  %call312 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %bitcast303, <8 x i32> zeroinitializer, <4 x float> %call250, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call313 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %call251, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call314 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %bitcast306, <8 x i32> zeroinitializer, <4 x float> %call252, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call315 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> %bitcast288, <4 x float> %call253, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call316 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %call254, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call317 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %call255, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %call318 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <4 x float> %call256, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  br i1 %arg52, label %bb136, label %bb166
}

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #1

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.amdgcn.s.setprio(i16 immarg) #2

; Function Attrs: convergent nocallback nofree nounwind willreturn

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32>, ptr addrspace(3) captures(none), i32 immarg, i32, i32, i32 immarg, i32 immarg) #3

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32>, <8 x i32>, <4 x float>, i32 immarg, i32 immarg, i32 immarg, i32, i32 immarg, i32) #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #5

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)


attributes #0 = { "amdgpu-agpr-alloc"="0" "amdgpu-flat-work-group-size"="1,512" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-waves-per-eu"="2" }
attributes #2 = { nocallback nofree nosync nounwind willreturn }
attributes #3 = { nocallback nofree nounwind willreturn }
attributes #4 = { convergent nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #5 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #6 = { convergent nounwind }
