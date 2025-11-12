; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1201 < %s | FileCheck %s
; CHECK: Occupancy: 16

%struct.zot = type { ptr }
%struct.bar = type { i32, i32, i32, i32, i8, i32, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%struct.snork = type { i32, i32, float, float, i32, i32, i32 }
%struct.barney = type { ptr }
%struct.zot.0 = type { %struct.ham }
%struct.ham = type { float, float, float, float }
%struct.zot.1 = type { %struct.wobble }
%struct.wobble = type { i32, i32, i32, i32 }

@global = external local_unnamed_addr addrspace(4) constant %struct.zot
@global.1 = external local_unnamed_addr addrspace(4) constant ptr
@global.2 = external local_unnamed_addr addrspace(4) constant ptr
@global.3 = external local_unnamed_addr addrspace(4) constant ptr
@global.4 = external local_unnamed_addr addrspace(4) constant ptr

declare void @llvm.trap() #0

declare void @llvm.lifetime.end.p5(ptr addrspace(5) captures(none)) #1

define amdgpu_kernel void @eggs(ptr addrspace(4) noundef readonly byref(%struct.bar) align 8 captures(none) %arg) local_unnamed_addr #2 {
bb:
  %alloca = alloca [128 x float], align 16, addrspace(5)
  %load = load i32, ptr addrspace(4) %arg, align 8, !amdgpu.noclobber !0
  %getelementptr = getelementptr inbounds nuw i8, ptr addrspace(4) %arg, i64 8
  %load1 = load i32, ptr addrspace(4) %getelementptr, align 8, !amdgpu.noclobber !0
  %getelementptr2 = getelementptr inbounds nuw i8, ptr addrspace(4) %arg, i64 12
  %load3 = load i32, ptr addrspace(4) %getelementptr2, align 4, !amdgpu.noclobber !0
  %getelementptr4 = getelementptr inbounds nuw i8, ptr addrspace(4) %arg, i64 24
  %load5 = load i32, ptr addrspace(4) %getelementptr4, align 8, !amdgpu.noclobber !0
  %getelementptr6 = getelementptr inbounds nuw i8, ptr addrspace(4) %arg, i64 36
  %load7 = load i32, ptr addrspace(4) %getelementptr6, align 4, !amdgpu.noclobber !0
  %load8 = load i32, ptr addrspace(4) null, align 4294967296
  %getelementptr9 = getelementptr inbounds nuw i8, ptr addrspace(4) %arg, i64 56
  %load10 = load ptr, ptr addrspace(4) %getelementptr9, align 8, !amdgpu.noclobber !0
  %addrspacecast = addrspacecast ptr %load10 to ptr addrspace(1)
  %getelementptr11 = getelementptr inbounds nuw i8, ptr addrspace(4) %arg, i64 72
  %load12 = load ptr, ptr addrspace(4) %getelementptr11, align 8, !amdgpu.noclobber !0
  %addrspacecast13 = addrspacecast ptr %load12 to ptr addrspace(1)
  %getelementptr14 = getelementptr inbounds nuw i8, ptr addrspace(4) %arg, i64 80
  %load15 = load ptr, ptr addrspace(4) %getelementptr14, align 8, !amdgpu.noclobber !0
  %addrspacecast16 = addrspacecast ptr %load15 to ptr addrspace(1)
  %getelementptr17 = getelementptr inbounds nuw i8, ptr addrspace(4) %arg, i64 88
  %load18 = load ptr, ptr addrspace(4) %getelementptr17, align 8, !amdgpu.noclobber !0
  %addrspacecast19 = addrspacecast ptr %load18 to ptr addrspace(1)
  %getelementptr20 = getelementptr inbounds nuw i8, ptr addrspace(4) %arg, i64 96
  %load21 = load ptr, ptr addrspace(4) %getelementptr20, align 8, !amdgpu.noclobber !0
  %addrspacecast22 = addrspacecast ptr %load21 to ptr addrspace(1)
  %getelementptr23 = getelementptr inbounds nuw i8, ptr addrspace(4) %arg, i64 112
  %load24 = load ptr, ptr addrspace(4) %getelementptr23, align 8, !amdgpu.noclobber !0
  %addrspacecast25 = addrspacecast ptr %load24 to ptr addrspace(1)
  %getelementptr26 = getelementptr inbounds nuw i8, ptr addrspace(4) %arg, i64 120
  %load27 = load ptr, ptr addrspace(4) %getelementptr26, align 8, !amdgpu.noclobber !0
  %addrspacecast28 = addrspacecast ptr %load27 to ptr addrspace(1)
  %getelementptr29 = getelementptr inbounds nuw i8, ptr addrspace(4) %arg, i64 136
  %load30 = load i32, ptr addrspace(4) %getelementptr29, align 8, !amdgpu.noclobber !0
  %getelementptr31 = getelementptr inbounds nuw i8, ptr addrspace(4) %arg, i64 144
  %load32 = load ptr, ptr addrspace(4) %getelementptr31, align 8, !amdgpu.noclobber !0
  %addrspacecast33 = addrspacecast ptr %load32 to ptr addrspace(1)
  %getelementptr34 = getelementptr inbounds nuw i8, ptr addrspace(4) %arg, i64 152
  %load35 = load ptr, ptr addrspace(4) %getelementptr34, align 8, !amdgpu.noclobber !0
  %addrspacecast36 = addrspacecast ptr %load35 to ptr addrspace(1)
  %getelementptr37 = getelementptr inbounds nuw i8, ptr addrspace(4) %arg, i64 176
  %load38 = load ptr, ptr addrspace(4) %getelementptr37, align 8, !amdgpu.noclobber !0
  %addrspacecast39 = addrspacecast ptr %load38 to ptr addrspace(1)
  %getelementptr40 = getelementptr inbounds nuw i8, ptr addrspace(4) %arg, i64 184
  %load41 = load ptr, ptr addrspace(4) %getelementptr40, align 8, !amdgpu.noclobber !0
  %addrspacecast42 = addrspacecast ptr %load41 to ptr addrspace(1)
  %getelementptr43 = getelementptr inbounds nuw i8, ptr addrspace(4) %arg, i64 192
  %load44 = load ptr, ptr addrspace(4) %getelementptr43, align 8, !amdgpu.noclobber !0
  %addrspacecast45 = addrspacecast ptr %load44 to ptr addrspace(1)
  %getelementptr46 = getelementptr inbounds nuw i8, ptr addrspace(4) %arg, i64 200
  %load47 = load ptr, ptr addrspace(4) %getelementptr46, align 8, !amdgpu.noclobber !0
  %addrspacecast48 = addrspacecast ptr %load47 to ptr addrspace(1)
  %getelementptr49 = getelementptr inbounds nuw i8, ptr addrspace(4) %arg, i64 208
  %load50 = load ptr, ptr addrspace(4) %getelementptr49, align 8, !amdgpu.noclobber !0
  %addrspacecast51 = addrspacecast ptr %load50 to ptr addrspace(1)
  %getelementptr52 = getelementptr inbounds nuw i8, ptr addrspace(4) %arg, i64 216
  %load53 = load ptr, ptr addrspace(4) %getelementptr52, align 8, !amdgpu.noclobber !0
  %addrspacecast54 = addrspacecast ptr %load53 to ptr addrspace(1)
  %getelementptr55 = getelementptr inbounds nuw i8, ptr addrspace(4) %arg, i64 224
  %load56 = load ptr, ptr addrspace(4) %getelementptr55, align 8, !amdgpu.noclobber !0
  %addrspacecast57 = addrspacecast ptr %load56 to ptr addrspace(1)
  %getelementptr58 = getelementptr inbounds nuw i8, ptr addrspace(4) %arg, i64 232
  %load59 = load ptr, ptr addrspace(4) %getelementptr58, align 8, !amdgpu.noclobber !0
  %addrspacecast60 = addrspacecast ptr %load59 to ptr addrspace(1)
  %getelementptr61 = getelementptr inbounds nuw i8, ptr addrspace(4) %arg, i64 280
  %load62 = load ptr, ptr addrspace(4) %getelementptr61, align 8, !amdgpu.noclobber !0
  %addrspacecast63 = addrspacecast ptr %load62 to ptr addrspace(1)
  %getelementptr64 = getelementptr inbounds nuw i8, ptr addrspace(4) %arg, i64 296
  %load65 = load ptr, ptr addrspace(4) %getelementptr64, align 8, !amdgpu.noclobber !0
  %addrspacecast66 = addrspacecast ptr %load65 to ptr addrspace(1)
  %call = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
  %and = and i32 %call, 31
  %icmp = icmp eq i32 %and, 0
  %lshr = lshr i32 %call, 5
  %getelementptr67 = getelementptr inbounds nuw %struct.snork, ptr addrspace(3) null, i32 %call
  %addrspacecast68 = addrspacecast ptr addrspace(3) %getelementptr67 to ptr
  %getelementptr69 = getelementptr inbounds nuw i8, ptr addrspace(3) null, i32 %lshr
  %addrspacecast70 = addrspacecast ptr addrspace(3) %getelementptr69 to ptr
  %getelementptr71 = getelementptr inbounds nuw i32, ptr addrspace(3) null, i32 %lshr
  %addrspacecast72 = addrspacecast ptr addrspace(3) %getelementptr71 to ptr
  %load73 = load ptr, ptr addrspace(4) @global, align 8
  %addrspacecast74 = addrspacecast ptr %load73 to ptr addrspace(4)
  %load75 = load ptr, ptr addrspace(4) @global.2, align 8
  %addrspacecast76 = addrspacecast ptr %load75 to ptr addrspace(1)
  %icmp77 = icmp ne i32 %load5, -1
  %add = add i32 %load8, -1
  %uitofp = uitofp i32 %add to float
  %getelementptr78 = getelementptr inbounds nuw i8, ptr addrspace(3) %getelementptr67, i32 20
  %addrspacecast79 = addrspacecast ptr addrspace(3) %getelementptr78 to ptr
  %getelementptr80 = getelementptr inbounds nuw i8, ptr addrspace(3) %getelementptr67, i32 4
  %addrspacecast81 = addrspacecast ptr addrspace(3) %getelementptr80 to ptr
  %getelementptr82 = getelementptr inbounds nuw i8, ptr addrspace(3) %getelementptr67, i32 16
  %addrspacecast83 = addrspacecast ptr addrspace(3) %getelementptr82 to ptr
  %getelementptr84 = getelementptr inbounds nuw i8, ptr addrspace(3) %getelementptr67, i32 12
  %addrspacecast85 = addrspacecast ptr addrspace(3) %getelementptr84 to ptr
  %getelementptr86 = getelementptr inbounds nuw i8, ptr addrspace(3) %getelementptr67, i32 8
  %addrspacecast87 = addrspacecast ptr addrspace(3) %getelementptr86 to ptr
  %getelementptr88 = getelementptr inbounds nuw i8, ptr addrspace(3) %getelementptr67, i32 24
  %addrspacecast89 = addrspacecast ptr addrspace(3) %getelementptr88 to ptr
  %load90 = load ptr, ptr addrspace(4) null, align 4294967296
  %addrspacecast91 = addrspacecast ptr %load90 to ptr addrspace(4)
  %load92 = load ptr, ptr addrspace(4) @global.4, align 8
  %addrspacecast93 = addrspacecast ptr %load92 to ptr addrspace(1)
  %load94 = load ptr, ptr addrspace(4) @global.3, align 8
  %addrspacecast95 = addrspacecast ptr %load94 to ptr addrspace(1)
  %load96 = load ptr, ptr addrspace(4) @global.1, align 8
  %addrspacecast97 = addrspacecast ptr %load96 to ptr addrspace(1)
  %icmp98 = icmp eq ptr addrspace(1) %addrspacecast63, addrspacecast (ptr null to ptr addrspace(1))
  %sext = sext i32 %load to i64
  %icmp99 = icmp ne i32 %add, 0
  %zext = zext i1 %icmp99 to i32
  %add100 = add i32 %load7, %zext
  %getelementptr101 = getelementptr inbounds nuw i8, ptr addrspace(1) %addrspacecast33, i64 4294967295
  %getelementptr102 = getelementptr inbounds nuw i8, ptr addrspace(1) %addrspacecast63, i64 8
  br label %bb103

bb103:                                            ; preds = %bb364, %bb
  %phi = phi i32 [ -1, %bb ], [ %phi143, %bb364 ]
  %phi104 = phi nsz float [ 0.0, %bb ], [ %phi144, %bb364 ]
  %phi105 = phi i32 [ -1, %bb ], [ %phi365, %bb364 ]
  %call106 = tail call i32 @llvm.amdgcn.ballot.i32(i1 true)
  %icmp107 = icmp slt i32 %phi105, 0
  %call108 = tail call i32 asm sideeffect "", "=v,0"(i32 range(i32 0, 2) 0) #7
  %icmp109 = icmp ne i32 %call108, 0
  %call110 = tail call i32 @llvm.amdgcn.ballot.i32(i1 %icmp109)
  %icmp111 = icmp eq i32 %call110, 0
  br i1 %icmp111, label %bb113, label %bb112

bb112:                                            ; preds = %bb103
  tail call void @llvm.trap()
  unreachable

bb113:                                            ; preds = %bb103
  %call114 = tail call i32 @llvm.amdgcn.ballot.i32(i1 %icmp107)
  %and115 = and i32 %call114, %call106
  %call116 = tail call noundef range(i32 0, 33) i32 @llvm.ctpop.i32(i32 %and115)
  %icmp117 = icmp samesign ugt i32 %call116, 3
  br i1 %icmp117, label %bb118, label %bb141

bb118:                                            ; preds = %bb113
  %icmp119 = icmp eq i32 %and115, -1
  br i1 %icmp119, label %bb120, label %bb122

bb120:                                            ; preds = %bb118
  %load121 = load volatile i8, ptr %addrspacecast70, align 1, !noalias.addrspace !1
  %trunc = trunc nuw i8 %load121 to i1
  br i1 %trunc, label %bb398, label %bb122

bb122:                                            ; preds = %bb120, %bb118
  br i1 %icmp, label %bb123, label %bb127

bb123:                                            ; preds = %bb122
  %atomicrmw = atomicrmw add ptr addrspace(1) %addrspacecast, i32 %call116 syncscope("agent") monotonic, align 4
  %load124 = load volatile i32, ptr %addrspacecast72, align 4, !noalias.addrspace !1
  %icmp125 = icmp ult i32 %load124, %load1
  br i1 %icmp125, label %bb127, label %bb126

bb126:                                            ; preds = %bb123
  store volatile i8 1, ptr %addrspacecast70, align 1, !noalias.addrspace !1
  br label %bb127

bb127:                                            ; preds = %bb126, %bb123, %bb122
  br i1 %icmp107, label %bb128, label %bb141

bb128:                                            ; preds = %bb127
  %load129 = load i32, ptr addrspace(1) %addrspacecast16, align 4
  %urem = urem i32 %load129, %load3
  %load130 = load i32, ptr addrspace(1) %addrspacecast39, align 4
  %urem131 = urem i32 %load130, %load3
  %zext132 = zext i32 %urem131 to i64
  %getelementptr133 = getelementptr inbounds nuw i32, ptr addrspace(1) %addrspacecast42, i64 %zext132
  %load134 = load i32, ptr addrspace(1) %getelementptr133, align 4
  %load135 = load <4 x i32>, ptr addrspace(4) %addrspacecast74, align 16
  %call136 = tail call float @llvm.amdgcn.struct.buffer.load.format.f32(<4 x i32> %load135, i32 %load134, i32 0, i32 0, i32 0)
  %bitcast = bitcast float %call136 to i32
  %and137 = and i32 %bitcast, 65536
  %icmp138 = icmp eq i32 %and137, 0
  %select = select i1 %icmp138, i32 -1, i32 %load134
  %load139 = load float, ptr addrspace(1) %addrspacecast76, align 4
  store i32 -1, ptr addrspace(1) null, align 4294967296
  store float 0x3FF0000100000000, ptr addrspace(1) %addrspacecast45, align 4
  store float 0.000000e+00, ptr addrspace(5) %alloca, align 16
  %zext140 = zext i32 %urem to i64
  br label %bb141

bb141:                                            ; preds = %bb128, %bb127, %bb113
  %phi142 = phi i32 [ %load30, %bb128 ], [ 0, %bb127 ], [ 0, %bb113 ]
  %phi143 = phi i32 [ %select, %bb128 ], [ %phi, %bb127 ], [ %phi, %bb113 ]
  %phi144 = phi nsz float [ %load139, %bb128 ], [ %phi104, %bb127 ], [ %phi104, %bb113 ]
  %phi145 = phi i32 [ 1, %bb128 ], [ %phi105, %bb127 ], [ %phi105, %bb113 ]
  %phi146 = phi i64 [ %sext, %bb128 ], [ 0, %bb127 ], [ 0, %bb113 ]
  %phi147 = phi i64 [ %zext140, %bb128 ], [ 0, %bb127 ], [ 0, %bb113 ]
  %phi148 = phi i32 [ %load129, %bb128 ], [ 0, %bb127 ], [ 0, %bb113 ]
  %icmp149 = icmp sgt i32 %phi145, 0
  %icmp150 = icmp ult i32 %phi142, 1073741824
  %select151 = select i1 %icmp149, i1 %icmp150, i1 false
  br i1 %select151, label %bb152, label %bb166

bb152:                                            ; preds = %bb141
  %and153 = and i32 %phi142, 134217727
  %call154 = tail call <3 x float> @llvm.amdgcn.struct.buffer.load.format.v3f32(<4 x i32> zeroinitializer, i32 %and153, i32 0, i32 0, i32 0)
  %bitcast155 = bitcast <3 x float> %call154 to <3 x i32>
  %extractelement = extractelement <3 x i32> %bitcast155, i64 2
  %lshr156 = lshr i32 %extractelement, 8
  %zext157 = zext nneg i32 %lshr156 to i64
  %getelementptr158 = getelementptr inbounds nuw i8, ptr addrspace(1) %addrspacecast33, i64 %zext157
  store i8 2, ptr addrspace(1) %getelementptr158, align 1
  br label %bb159

bb159:                                            ; preds = %bb159, %bb152
  %call160 = tail call <3 x float> @llvm.amdgcn.struct.buffer.load.format.v3f32(<4 x i32> zeroinitializer, i32 0, i32 0, i32 0, i32 0)
  %bitcast161 = bitcast <3 x float> %call160 to <3 x i32>
  %extractelement162 = extractelement <3 x i32> %bitcast161, i64 2
  %lshr163 = lshr i32 %extractelement162, 8
  %zext164 = zext nneg i32 %lshr163 to i64
  %getelementptr165 = getelementptr inbounds nuw i8, ptr addrspace(1) %addrspacecast33, i64 %zext164
  store i8 2, ptr addrspace(1) %getelementptr165, align 1
  br label %bb159, !llvm.loop !2

bb166:                                            ; preds = %bb141
  %load167 = load float, ptr addrspace(1) %addrspacecast25, align 16
  %load168 = load float, ptr addrspace(1) inttoptr (i64 4 to ptr addrspace(1)), align 4
  %load169 = load float, ptr addrspace(1) inttoptr (i64 8 to ptr addrspace(1)), align 8
  %and170 = and i32 %phi142, 536870911
  %load171 = load <4 x i32>, ptr addrspace(4) null, align 4294967296
  %call172 = tail call float @llvm.amdgcn.struct.buffer.load.format.f32(<4 x i32> %load171, i32 %and170, i32 0, i32 0, i32 0)
  %bitcast173 = bitcast float %call172 to i32
  %zext174 = zext i32 %bitcast173 to i64
  %getelementptr175 = getelementptr inbounds nuw i32, ptr addrspace(1) %addrspacecast36, i64 %zext174
  %load176 = load i32, ptr addrspace(1) %getelementptr175, align 4
  %icmp177 = icmp ne i32 %and170, %load5
  %and178 = and i1 %icmp77, %icmp177
  %icmp179 = icmp eq i32 %phi143, %and170
  br i1 %icmp179, label %bb180, label %bb181

bb180:                                            ; preds = %bb166
  br label %bb181

bb181:                                            ; preds = %bb180, %bb166
  %phi182 = phi i32 [ 0, %bb180 ], [ %load176, %bb166 ]
  %phi183 = phi i1 [ true, %bb180 ], [ %and178, %bb166 ]
  %phi184 = phi i32 [ -1, %bb180 ], [ %and170, %bb166 ]
  %phi185 = phi i32 [ 0, %bb180 ], [ %phi145, %bb166 ]
  br i1 %phi183, label %bb204, label %bb186

bb186:                                            ; preds = %bb181
  %fmul = fmul reassoc nnan ninf nsz arcp contract float %phi144, %uitofp
  %sub = sub nuw i32 %phi184, %load7
  %mul = mul i32 %sub, %load8
  %add187 = add i32 %add100, %mul
  %mul188 = mul i32 %add187, 3
  %call189 = tail call <2 x float> @llvm.amdgcn.struct.buffer.load.format.v2f32(<4 x i32> zeroinitializer, i32 %mul188, i32 0, i32 0, i32 0)
  %extractelement190 = extractelement <2 x float> %call189, i64 1
  %call191 = tail call <3 x float> @llvm.amdgcn.struct.buffer.load.format.v3f32(<4 x i32> zeroinitializer, i32 0, i32 0, i32 0, i32 0)
  %extractelement192 = extractelement <3 x float> %call191, i64 2
  %extractelement193 = extractelement <3 x float> %call191, i64 0
  %extractelement194 = extractelement <3 x float> %call191, i64 1
  %fmul195 = fmul reassoc nnan ninf nsz arcp contract float %extractelement192, %fmul
  %fmul196 = fmul reassoc nnan ninf nsz arcp contract float %fmul, %fmul
  %fmul197 = fmul reassoc nnan ninf nsz arcp contract float %fmul196, %extractelement190
  %fmul198 = fmul reassoc nnan ninf nsz arcp contract float %fmul197, %extractelement193
  %fmul199 = fmul reassoc nnan ninf nsz arcp contract float %fmul198, %fmul195
  %fneg = fneg reassoc nnan ninf nsz arcp contract float %fmul
  %fmul200 = fmul reassoc nnan ninf nsz arcp contract float %extractelement194, %fneg
  %load201 = load float, ptr addrspace(1) %addrspacecast28, align 16
  %fmul202 = fmul reassoc nnan ninf nsz arcp contract float %fmul200, %load201
  %fmul203 = fmul reassoc nnan ninf nsz arcp contract float %fmul202, %fmul195
  %fdiv = fdiv reassoc nnan ninf nsz arcp contract float %fmul203, %fmul199
  br label %bb204

bb204:                                            ; preds = %bb186, %bb181
  %phi205 = phi float [ %load169, %bb181 ], [ 0.000000e+00, %bb186 ]
  %phi206 = phi float [ %load168, %bb181 ], [ 0.000000e+00, %bb186 ]
  %phi207 = phi float [ 0.000000e+00, %bb181 ], [ %fdiv, %bb186 ]
  %phi208 = phi float [ %load167, %bb181 ], [ 0.000000e+00, %bb186 ]
  %lshr209 = lshr i32 %phi182, 27
  %and210 = and i32 %lshr209, 7
  %and211 = and i32 %phi182, 134217727
  %zext212 = zext nneg i32 %and210 to i64
  %getelementptr213 = getelementptr inbounds nuw %struct.barney, ptr addrspace(4) null, i64 %zext212
  %load214 = load i64, ptr addrspace(4) %getelementptr213, align 8
  %inttoptr = inttoptr i64 %load214 to ptr
  %addrspacecast215 = addrspacecast ptr %inttoptr to ptr addrspace(1)
  %shl = shl nuw nsw i32 %and211, 4
  %zext216 = zext nneg i32 %shl to i64
  %getelementptr217 = getelementptr inbounds nuw i8, ptr addrspace(1) %addrspacecast215, i64 %zext216
  %getelementptr218 = getelementptr inbounds nuw i8, ptr addrspace(1) %getelementptr217, i64 4
  %load219 = load i16, ptr addrspace(1) %getelementptr218, align 4
  %load220 = load i32, ptr addrspace(1) null, align 4294967296
  %lshr221 = lshr i32 %load220, 12
  %getelementptr222 = getelementptr inbounds nuw i8, ptr addrspace(1) %getelementptr217, i64 12
  %load223 = load i32, ptr addrspace(1) %getelementptr222, align 4
  %lshr224 = lshr i32 %load223, 25
  %and225 = and i32 %lshr224, 31
  %add226 = add nsw i32 %and225, -1
  %uitofp227 = uitofp i32 %add226 to float
  %call228 = tail call reassoc nnan ninf nsz arcp contract noundef float @llvm.minnum.f32(float nofpclass(nan inf) %phi144, float 0x3FEFFFEB00000000)
  %fmul229 = fmul reassoc nnan ninf nsz arcp contract float %call228, %uitofp227
  %fptoui = fptoui float %fmul229 to i32
  %and230 = and i32 %load220, 65535
  %and231 = and i16 %load219, 32767
  %zext232 = zext nneg i16 %and231 to i32
  %add233 = add nuw nsw i32 %and230, %zext232
  %and234 = and i32 %load223, 1073741824
  %icmp235 = icmp eq i32 %and234, 0
  br i1 %icmp235, label %bb277, label %bb236

bb236:                                            ; preds = %bb204
  %icmp237 = icmp eq i16 %and231, 0
  br i1 %icmp237, label %bb351, label %bb238

bb238:                                            ; preds = %bb236
  %getelementptr239 = getelementptr inbounds nuw i8, ptr addrspace(1) %getelementptr217, i64 16
  %sub240 = sub nsw i32 %and211, %lshr221
  %add241 = add i32 %sub240, %fptoui
  %addrspacecast242 = addrspacecast ptr %inttoptr to ptr addrspace(4)
  %load243 = load <4 x i32>, ptr addrspace(4) %addrspacecast242, align 16
  %fmul244 = fmul reassoc nnan ninf nsz arcp contract float %phi205, %phi205
  %call245 = tail call reassoc nnan ninf nsz arcp contract noundef float @llvm.fma.f32(float nofpclass(nan inf) %phi206, float nofpclass(nan inf) %phi206, float nofpclass(nan inf) %fmul244)
  %call246 = tail call reassoc nnan ninf nsz arcp contract noundef float @llvm.fma.f32(float nofpclass(nan inf) %phi207, float nofpclass(nan inf) %phi207, float nofpclass(nan inf) %call245)
  %call247 = tail call reassoc nnan ninf nsz arcp contract noundef float @llvm.sqrt.f32(float nofpclass(nan inf) %call246)
  %getelementptr248 = getelementptr inbounds %struct.zot.0, ptr addrspace(1) %addrspacecast51, i64 %phi146
  %call249 = tail call float @llvm.amdgcn.rsq.f32(float 0.000000e+00)
  %load250 = load i32, ptr addrspace(1) %getelementptr239, align 4
  %mul251 = mul i32 %load250, %and225
  %add252 = add i32 %add241, %mul251
  %call253 = tail call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %load243, i32 %add252, i32 0, i32 0, i32 0)
  %extractelement254 = extractelement <4 x float> %call253, i64 3
  %fmul255 = fmul reassoc nnan ninf nsz arcp contract float %extractelement254, %extractelement254
  %call256 = tail call reassoc nnan ninf nsz arcp contract noundef float @llvm.sqrt.f32(float nofpclass(nan inf) %fmul255)
  store volatile i32 %phi184, ptr %addrspacecast79, align 4, !noalias.addrspace !1
  store volatile i32 %phi182, ptr %addrspacecast68, align 4, !noalias.addrspace !1
  store volatile i32 %and230, ptr %addrspacecast81, align 4, !noalias.addrspace !1
  %fmul257 = fmul reassoc nnan ninf nsz arcp contract float %phi207, %call256
  %fdiv258 = fdiv reassoc nnan ninf nsz arcp contract float %fmul257, %call247
  %fmul259 = fmul reassoc nnan ninf nsz arcp contract float %fdiv258, %fdiv258
  %fcmp = fcmp reassoc nnan ninf nsz arcp contract ogt float %fmul259, 0.000000e+00
  %select260 = select reassoc nnan ninf nsz arcp contract i1 %fcmp, float %call249, float 0.000000e+00
  %fmul261 = fmul reassoc nnan ninf nsz arcp contract float %select260, %fdiv258
  store float %fmul261, ptr addrspace(1) %getelementptr248, align 16
  %fdiv262 = fdiv reassoc nnan ninf nsz arcp contract float 1.000000e+00, %call247
  br label %bb263

bb263:                                            ; preds = %bb263, %bb238
  %load264 = load i32, ptr addrspace(1) null, align 4294967296
  %mul265 = mul i32 %load264, %and225
  %add266 = add i32 %add241, %mul265
  %call267 = tail call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %load243, i32 %add266, i32 0, i32 0, i32 0)
  %extractelement268 = extractelement <4 x float> %call267, i64 3
  %fmul269 = fmul reassoc nnan ninf nsz arcp contract float %extractelement268, %extractelement268
  %call270 = tail call reassoc nnan ninf nsz arcp contract noundef float @llvm.sqrt.f32(float nofpclass(nan inf) %fmul269)
  store volatile i32 %phi184, ptr %addrspacecast79, align 4, !noalias.addrspace !1
  store volatile i32 %phi182, ptr %addrspacecast68, align 4, !noalias.addrspace !1
  store volatile i32 0, ptr %addrspacecast81, align 4, !noalias.addrspace !1
  %fmul271 = fmul reassoc nnan ninf nsz arcp contract float %phi207, %call270
  %fmul272 = fmul reassoc nnan ninf nsz arcp contract float %fmul271, %fdiv262
  %fmul273 = fmul reassoc nnan ninf nsz arcp contract float %fmul272, %fmul272
  %fcmp274 = fcmp reassoc nnan ninf nsz arcp contract ogt float %fmul273, 0.000000e+00
  %select275 = select reassoc nnan ninf nsz arcp contract i1 %fcmp274, float %call249, float 0.000000e+00
  %fmul276 = fmul reassoc nnan ninf nsz arcp contract float %select275, %fmul272
  store float %fmul276, ptr addrspace(1) %getelementptr248, align 16
  br label %bb263, !llvm.loop !4

bb277:                                            ; preds = %bb204
  %icmp278 = icmp slt i32 %load223, 0
  br i1 %icmp278, label %bb279, label %bb348

bb279:                                            ; preds = %bb277
  %fmul280 = fmul reassoc nnan ninf nsz arcp contract float %phi207, %phi207
  %call281 = tail call reassoc nnan ninf nsz arcp contract noundef float @llvm.sqrt.f32(float nofpclass(nan inf) %fmul280)
  %fdiv282 = fdiv reassoc nnan ninf nsz arcp contract float 1.000000e+00, %call281
  %icmp283 = icmp eq i16 %and231, 0
  br i1 %icmp283, label %bb351, label %bb284

bb284:                                            ; preds = %bb279
  %sub285 = sub nsw i32 %and211, %lshr221
  %addrspacecast286 = addrspacecast ptr %inttoptr to ptr addrspace(4)
  %fmul287 = fmul reassoc nnan ninf nsz arcp contract float %fdiv282, %phi207
  %fmul288 = fmul reassoc nnan ninf nsz arcp contract float %fdiv282, %phi206
  %fmul289 = fmul reassoc nnan ninf nsz arcp contract float %fdiv282, %phi205
  %getelementptr290 = getelementptr inbounds %struct.zot.0, ptr addrspace(1) %addrspacecast51, i64 %phi146
  %add291 = add i32 %sub285, %fptoui
  %load292 = load <4 x i32>, ptr addrspace(4) %addrspacecast286, align 16
  %add293 = add i32 %add291, 1
  %add294 = add i32 %add291, %and225
  %add295 = add i32 %add294, 1
  br label %bb296

bb296:                                            ; preds = %bb341, %bb284
  %phi297 = phi i32 [ %and230, %bb284 ], [ %add346, %bb341 ]
  %call298 = tail call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %load292, i32 %add291, i32 0, i32 0, i32 0)
  %extractelement299 = extractelement <4 x float> %call298, i64 0
  %extractelement300 = extractelement <4 x float> %call298, i64 3
  %call301 = tail call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %load292, i32 %add293, i32 0, i32 0, i32 0)
  %extractelement302 = extractelement <4 x float> %call301, i64 3
  %fsub = fsub reassoc nnan ninf nsz arcp contract float %extractelement302, %extractelement300
  %call303 = tail call reassoc nnan ninf nsz arcp contract noundef float @llvm.fma.f32(float nofpclass(nan inf) %fsub, float nofpclass(nan inf) %fmul229, float nofpclass(nan inf) %extractelement300)
  %call304 = tail call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %load292, i32 %add294, i32 0, i32 0, i32 0)
  %extractelement305 = extractelement <4 x float> %call304, i64 3
  %call306 = tail call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %load292, i32 %add295, i32 0, i32 0, i32 0)
  %extractelement307 = extractelement <4 x float> %call306, i64 3
  %fsub308 = fsub reassoc nnan ninf nsz arcp contract float %extractelement307, %extractelement305
  %call309 = tail call reassoc nnan ninf nsz arcp contract noundef float @llvm.fma.f32(float nofpclass(nan inf) %fsub308, float nofpclass(nan inf) %fmul229, float nofpclass(nan inf) %extractelement305)
  %fneg310 = fneg reassoc nnan ninf nsz arcp contract float %extractelement299
  %fmul311 = fmul reassoc nnan ninf nsz arcp contract float %extractelement299, %extractelement299
  %call312 = tail call reassoc nnan ninf nsz arcp contract noundef float @llvm.sqrt.f32(float nofpclass(nan inf) %fmul311)
  %fdiv313 = fdiv reassoc nnan ninf nsz arcp contract float 1.000000e+00, %call312
  %fmul314 = fmul reassoc nnan ninf nsz arcp contract float %fdiv313, %fneg310
  %fmul315 = fmul reassoc nnan ninf nsz arcp contract float %fmul314, %fmul287
  %fmul316 = fmul reassoc nnan ninf nsz arcp contract float %fmul315, %fmul315
  %fsub317 = fsub reassoc nnan ninf nsz arcp contract float 1.000000e+00, %fmul316
  %fcmp318 = fcmp reassoc nnan ninf nsz arcp contract oeq float %fsub317, 0.000000e+00
  br i1 %fcmp318, label %bb341, label %bb319

bb319:                                            ; preds = %bb296
  %extractelement320 = extractelement <4 x float> %call301, i64 2
  %extractelement321 = extractelement <4 x float> %call298, i64 2
  %fsub322 = fsub reassoc nnan ninf nsz arcp contract float %extractelement320, %extractelement321
  %call323 = tail call reassoc nnan ninf nsz arcp contract noundef float @llvm.fma.f32(float nofpclass(nan inf) %fsub322, float nofpclass(nan inf) %fmul229, float nofpclass(nan inf) %extractelement321)
  %extractelement324 = extractelement <4 x float> %call301, i64 1
  %extractelement325 = extractelement <4 x float> %call298, i64 1
  %fsub326 = fsub reassoc nnan ninf nsz arcp contract float %extractelement324, %extractelement325
  %call327 = tail call reassoc nnan ninf nsz arcp contract noundef float @llvm.fma.f32(float nofpclass(nan inf) %fsub326, float nofpclass(nan inf) %fmul229, float nofpclass(nan inf) %extractelement325)
  %fsub328 = fsub reassoc nnan ninf nsz arcp contract float %extractelement299, %phi208
  %fmul329 = fmul reassoc nnan ninf nsz arcp contract float %fmul314, %fsub328
  %fmul330 = fmul reassoc nnan ninf nsz arcp contract float %fmul289, %call323
  %call331 = tail call reassoc nnan ninf nsz arcp contract noundef float @llvm.fma.f32(float nofpclass(nan inf) %call327, float nofpclass(nan inf) %fmul288, float nofpclass(nan inf) %fmul330)
  %call332 = tail call reassoc nnan ninf nsz arcp contract noundef float @llvm.fma.f32(float nofpclass(nan inf) %fsub328, float nofpclass(nan inf) %fmul287, float nofpclass(nan inf) %call331)
  %fmul333 = fmul reassoc nnan ninf nsz arcp contract float %fmul329, %fmul315
  %fsub334 = fsub reassoc nnan ninf nsz arcp contract float %call332, %fmul333
  %fmul335 = fmul reassoc nnan ninf nsz arcp contract float %fsub334, %fdiv282
  %fdiv336 = fdiv reassoc nnan ninf nsz arcp contract float %fmul335, %fsub317
  %call337 = tail call reassoc nnan ninf nsz arcp contract noundef float @llvm.minnum.f32(float nofpclass(nan inf) %call312, float nofpclass(nan inf) 0.000000e+00)
  %call338 = tail call reassoc nnan ninf nsz arcp contract noundef float @llvm.maxnum.f32(float nofpclass(nan inf) %call337, float 0.000000e+00)
  %fmul339 = fmul reassoc nnan ninf nsz arcp contract float %call338, %fdiv313
  %call340 = tail call reassoc nnan ninf nsz arcp contract float @llvm.fabs.f32(float nofpclass(nan inf) %fdiv336)
  br label %bb341

bb341:                                            ; preds = %bb319, %bb296
  %phi342 = phi nsz float [ %fmul339, %bb319 ], [ 0.000000e+00, %bb296 ]
  %phi343 = phi float [ %call340, %bb319 ], [ 0.000000e+00, %bb296 ]
  %fsub344 = fsub reassoc nnan ninf nsz arcp contract float %call309, %call303
  %fmul345 = fmul reassoc nnan ninf nsz arcp contract float %phi342, %fsub344
  %fadd = fadd reassoc nnan ninf nsz arcp contract float %fmul345, %call303
  store volatile i32 %load220, ptr %addrspacecast83, align 4, !noalias.addrspace !1
  store float %fadd, ptr addrspace(1) %getelementptr290, align 16
  store float 1.000000e+00, ptr addrspace(1) null, align 4294967296
  %add346 = add nuw nsw i32 %phi297, 1
  %icmp347 = icmp samesign ult i32 %add346, %add233
  br i1 %icmp347, label %bb296, label %bb351

bb348:                                            ; preds = %bb277
  %icmp349 = icmp eq i16 %and231, 0
  br i1 %icmp349, label %bb351, label %bb350

bb350:                                            ; preds = %bb398, %bb348
  ret void

bb351:                                            ; preds = %bb348, %bb341, %bb279, %bb236
  %phi352 = phi float [ 0.000000e+00, %bb279 ], [ 0.000000e+00, %bb348 ], [ 0.000000e+00, %bb236 ], [ %phi343, %bb341 ]
  br label %bb353

bb353:                                            ; preds = %bb353, %bb351
  %phi354 = phi i32 [ %phi185, %bb351 ], [ %add355, %bb353 ]
  %add355 = add nsw i32 %phi354, -1
  %getelementptr356 = getelementptr inbounds float, ptr addrspace(5) %alloca, i32 %add355
  %load357 = load float, ptr addrspace(5) %getelementptr356, align 4
  %icmp358 = icmp sgt i32 %phi354, 1
  %fcmp359 = fcmp reassoc nnan ninf nsz arcp contract ogt float %load357, %phi352
  %select360 = select i1 %icmp358, i1 %fcmp359, i1 false
  br i1 %select360, label %bb353, label %bb361

bb361:                                            ; preds = %bb353
  %icmp362 = icmp eq i32 %add355, 0
  store i8 1, ptr addrspace(1) %getelementptr101, align 1
  %atomicrmw363 = atomicrmw add ptr addrspace(1) %addrspacecast13, i32 1 syncscope("agent") monotonic, align 4
  store i32 %phi148, ptr addrspace(1) %addrspacecast19, align 4
  store volatile i32 536870911, ptr %addrspacecast89, align 4, !noalias.addrspace !1
  br i1 %icmp362, label %bb366, label %bb364

bb364:                                            ; preds = %bb397, %bb389, %bb386, %bb385, %bb361
  %phi365 = phi i32 [ %add355, %bb361 ], [ -1, %bb385 ], [ -1, %bb386 ], [ -1, %bb389 ], [ -1, %bb397 ]
  br label %bb103

bb366:                                            ; preds = %bb361
  %load367 = load volatile i32, ptr %addrspacecast79, align 4, !noalias.addrspace !1
  %load368 = load volatile i32, ptr %addrspacecast68, align 4, !noalias.addrspace !1
  %load369 = load volatile i32, ptr %addrspacecast81, align 4, !noalias.addrspace !1
  %load370 = load volatile float, ptr %addrspacecast87, align 4, !noalias.addrspace !1
  %load371 = load volatile float, ptr %addrspacecast85, align 4, !noalias.addrspace !1
  %call372 = tail call float @llvm.amdgcn.struct.buffer.load.format.f32(<4 x i32> %load171, i32 %load367, i32 0, i32 0, i32 0)
  %bitcast373 = bitcast float %call372 to i32
  %load374 = load <4 x i32>, ptr addrspace(4) %addrspacecast91, align 16
  %call375 = tail call float @llvm.amdgcn.struct.buffer.load.format.f32(<4 x i32> %load374, i32 %bitcast373, i32 0, i32 0, i32 0)
  %getelementptr376 = getelementptr inbounds nuw %struct.zot.1, ptr addrspace(1) %addrspacecast93, i64 %phi147
  %load377 = load i32, ptr addrspace(1) %addrspacecast22, align 4
  %and378 = and i32 %load377, -285212672
  %or = or disjoint i32 %and378, 268435456
  store i32 0, ptr addrspace(1) %addrspacecast95, align 4
  store i32 %or, ptr addrspace(1) %addrspacecast97, align 4
  %getelementptr379 = getelementptr inbounds nuw i8, ptr addrspace(1) %getelementptr376, i64 8
  store float %call375, ptr addrspace(1) %getelementptr379, align 8
  %load380 = load i32, ptr addrspace(1) null, align 4294967296
  %load381 = load i16, ptr addrspace(1) inttoptr (i64 6 to ptr addrspace(1)), align 2
  %zext382 = zext i16 %load381 to i32
  %and383 = and i32 %load380, 1073741824
  %icmp384 = icmp eq i32 %and383, 0
  br i1 %icmp384, label %bb387, label %bb385

bb385:                                            ; preds = %bb366
  store i32 0, ptr addrspace(1) %addrspacecast60, align 4
  br i1 %icmp98, label %bb364, label %bb386

bb386:                                            ; preds = %bb385
  store float 0x47EFFFFFE0000000, ptr addrspace(1) %getelementptr102, align 8
  br label %bb364

bb387:                                            ; preds = %bb366
  %icmp388 = icmp slt i32 %load380, 0
  br i1 %icmp388, label %bb389, label %bb397

bb389:                                            ; preds = %bb387
  %sub390 = sub i32 %load369, %zext382
  %zext391 = zext i32 %sub390 to i64
  %getelementptr392 = getelementptr inbounds nuw %struct.zot.1, ptr addrspace(1) null, i64 %zext391
  %getelementptr393 = getelementptr inbounds nuw i8, ptr addrspace(1) %getelementptr392, i64 4
  %load394 = load i32, ptr addrspace(1) %getelementptr393, align 4
  store i32 0, ptr addrspace(1) %addrspacecast66, align 4
  %and395 = and i32 %load394, 1073741823
  %or396 = or disjoint i32 %and395, -2147483648
  store i32 %or396, ptr addrspace(1) %addrspacecast48, align 4
  br label %bb364

bb397:                                            ; preds = %bb387
  store float 0.000000e+00, ptr addrspace(1) inttoptr (i64 12 to ptr addrspace(1)), align 4
  store float 0.000000e+00, ptr addrspace(1) %addrspacecast54, align 16
  store float 0.000000e+00, ptr addrspace(1) %addrspacecast57, align 4
  br label %bb364

bb398:                                            ; preds = %bb120
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %alloca) #8
  br label %bb350
}

declare float @llvm.minnum.f32(float, float) #3

declare float @llvm.maxnum.f32(float, float) #3

declare float @llvm.fma.f32(float, float, float) #3

declare float @llvm.fabs.f32(float) #3

declare float @llvm.sqrt.f32(float) #3

declare i32 @llvm.ctpop.i32(i32) #3

declare float @llvm.amdgcn.rsq.f32(float) #4

declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #4

declare <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32>, i32, i32, i32, i32 immarg) #5

declare i32 @llvm.amdgcn.ballot.i32(i1) #6

declare float @llvm.amdgcn.struct.buffer.load.format.f32(<4 x i32>, i32, i32, i32, i32 immarg) #5

declare <2 x float> @llvm.amdgcn.struct.buffer.load.format.v2f32(<4 x i32>, i32, i32, i32, i32 immarg) #5

declare <3 x float> @llvm.amdgcn.struct.buffer.load.format.v3f32(<4 x i32>, i32, i32, i32, i32 immarg) #5

attributes #0 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { convergent norecurse nounwind "amdgpu-flat-work-group-size"="1,1024" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1201" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-buffer-pk-add-bf16-inst,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-fmin-fmax-global-f32,+atomic-global-pk-add-bf16-inst,+ci-insts,+dl-insts,+dot10-insts,+dot11-insts,+dot12-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+fp8-conversion-insts,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx12-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32" "uniform-work-group-size"="true" }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { nocallback nofree nosync nounwind willreturn memory(read) }
attributes #6 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #7 = { convergent nounwind }
attributes #8 = { nounwind }

!0 = !{}
!1 = !{i32 1, i32 3, i32 4, i32 10}
!2 = distinct !{!2, !3}
!3 = !{!"llvm.loop.peeled.count", i32 1}
!4 = distinct !{!4, !3}
