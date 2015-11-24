; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

; CHECK: ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK:   [tmp180, tmp177, tmp183, tmp162, tmp157, tmp150, tmp146, tmp140, tmp] -> { Stmt_bb203[i0, i1, i2] -> MemRef_tmp173[o0, 1 + i1, 1 + i2] : exists (e0 = floor((-i0 + o0)/3): 3e0 = -i0 + o0 and o0 <= 2 and o0 >= 0) };
; CHECK: ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK:   [tmp180, tmp177, tmp183, tmp162, tmp157, tmp150, tmp146, tmp140, tmp] -> { Stmt_bb203[i0, i1, i2] -> MemRef_tmp173[o0, 1 + i1, 1 + i2] : exists (e0 = floor((-2 - i0 + o0)/3): 3e0 = -2 - i0 + o0 and o0 <= 2 and o0 >= 0) };


define void @pluto(i32* noalias %arg, [0 x i32]* noalias %arg2, [0 x i32]* noalias %arg3, [0 x i32]* noalias %arg4, [0 x i32]* noalias %arg5, [0 x i32]* noalias %arg6, [0 x i32]* noalias %arg7, [0 x i32]* noalias %arg8, [0 x i32]* noalias %arg9, double* noalias %arg10, double* noalias %arg11, [0 x double]* noalias %arg12, [0 x double]* noalias %arg13, [0 x i32]* noalias %arg14, i32* noalias %arg15, [0 x i32]* noalias %arg16, i32* noalias %arg17, i32* noalias %arg18, i32* noalias %arg19, i32* noalias %arg20, i32* noalias %arg21, i32* noalias %arg22, i32* noalias %arg23, i32* noalias %arg24, i32* noalias %arg25, i32* noalias %arg26, i32* noalias %arg27, [0 x double]* noalias %arg28, [0 x double]* noalias %arg29, [0 x double]* noalias %arg30, [0 x double]* noalias %arg31, [0 x double]* noalias %arg32, [0 x double]* noalias %arg33, [0 x double]* noalias %arg34, [0 x double]* noalias %arg35, [0 x double]* noalias %arg36, [0 x double]* noalias %arg37, [0 x double]* noalias %arg38, [0 x double]* noalias %arg39, [0 x double]* noalias %arg40, [0 x double]* noalias %arg41, [0 x double]* noalias %arg42, [0 x double]* noalias %arg43, [0 x double]* noalias %arg44, [0 x double]* noalias %arg45, [0 x double]* noalias %arg46, [0 x double]* noalias %arg47, [0 x double]* noalias %arg48, [0 x double]* noalias %arg49, [0 x double]* noalias %arg50, [0 x double]* noalias %arg51, [0 x double]* noalias %arg52, [0 x double]* noalias %arg53, [0 x double]* noalias %arg54, [0 x double]* noalias %arg55, [0 x double]* noalias %arg56, [0 x double]* noalias %arg57, [0 x double]* noalias %arg58, [0 x double]* noalias %arg59, [0 x double]* noalias %arg60, [0 x double]* noalias %arg61, [0 x double]* noalias %arg62, [0 x double]* noalias %arg63, [0 x double]* noalias %arg64, [0 x double]* noalias %arg65, [0 x double]* noalias %arg66, [0 x double]* noalias %arg67, [0 x double]* noalias %arg68, [0 x double]* noalias %arg69, i32* noalias %arg70, i32* noalias %arg71, i32* noalias %arg72, i32* noalias %arg73, i32* noalias %arg74, i32* noalias %arg75, i32* noalias %arg76, i32* noalias %arg77, i32* noalias %arg78, i32* noalias %arg79, i32* noalias %arg80, i32* noalias %arg81, i32* noalias %arg82, i32* noalias %arg83, i32* noalias %arg84, i32* noalias %arg85, i32* noalias %arg86, i32* noalias %arg87, i32* noalias %arg88, i32* noalias %arg89, i32* noalias %arg90, i32* noalias %arg91, i32* noalias %arg92, i32* noalias %arg93, i32* noalias %arg94, i32* noalias %arg95, i32* noalias %arg96, i32* noalias %arg97, [0 x double]* noalias %arg98, [0 x double]* noalias %arg99, [0 x double]* noalias %arg100, [0 x double]* noalias %arg101, double* noalias %arg102, double* noalias %arg103, double* noalias %arg104, i32* noalias %arg105, double* noalias %arg106, double* noalias %arg107, [0 x double]* noalias %arg108, [0 x double]* noalias %arg109, [0 x double]* noalias %arg110, [0 x double]* noalias %arg111, [0 x double]* noalias %arg112, [0 x double]* noalias %arg113, [0 x double]* noalias %arg114, [0 x double]* noalias %arg115, [0 x double]* noalias %arg116, [0 x double]* noalias %arg117, [0 x double]* noalias %arg118, [0 x double]* noalias %arg119, [0 x double]* noalias %arg120, [0 x double]* noalias %arg121, [0 x double]* noalias %arg122, [0 x double]* noalias %arg123, [0 x double]* noalias %arg124, [0 x double]* noalias %arg125, [0 x double]* noalias %arg126, [0 x double]* noalias %arg127, [0 x double]* noalias %arg128, [0 x double]* noalias %arg129, [0 x double]* noalias %arg130, [0 x double]* noalias %arg131, i32* noalias %arg132, [0 x double]* noalias %arg133, [0 x double]* noalias %arg134, [0 x double]* noalias %arg135) {
bb:
  br label %bb136

bb136:                                            ; preds = %bb
  %tmp = load i32, i32* %arg19, align 4
  %tmp137 = sext i32 %tmp to i64
  %tmp138 = icmp slt i64 %tmp137, 0
  %tmp139 = select i1 %tmp138, i64 0, i64 %tmp137
  %tmp140 = load i32, i32* %arg20, align 4
  %tmp141 = sext i32 %tmp140 to i64
  %tmp142 = mul nsw i64 %tmp139, %tmp141
  %tmp143 = icmp slt i64 %tmp142, 0
  %tmp144 = select i1 %tmp143, i64 0, i64 %tmp142
  %tmp145 = xor i64 %tmp139, -1
  %tmp146 = load i32, i32* %arg19, align 4
  %tmp147 = sext i32 %tmp146 to i64
  %tmp148 = icmp slt i64 %tmp147, 0
  %tmp149 = select i1 %tmp148, i64 0, i64 %tmp147
  %tmp150 = load i32, i32* %arg20, align 4
  %tmp151 = sext i32 %tmp150 to i64
  %tmp152 = mul nsw i64 %tmp149, %tmp151
  %tmp153 = icmp slt i64 %tmp152, 0
  %tmp154 = select i1 %tmp153, i64 0, i64 %tmp152
  %tmp155 = xor i64 %tmp149, -1
  %tmp156 = getelementptr inbounds [0 x i32], [0 x i32]* %arg3, i64 0, i64 0
  %tmp157 = load i32, i32* %tmp156, align 4
  %tmp158 = sext i32 %tmp157 to i64
  %tmp159 = icmp slt i64 %tmp158, 0
  %tmp160 = select i1 %tmp159, i64 0, i64 %tmp158
  %tmp161 = getelementptr [0 x i32], [0 x i32]* %arg3, i64 0, i64 1
  %tmp162 = load i32, i32* %tmp161, align 4
  %tmp163 = sext i32 %tmp162 to i64
  %tmp164 = mul nsw i64 %tmp160, %tmp163
  %tmp165 = icmp slt i64 %tmp164, 0
  %tmp166 = select i1 %tmp165, i64 0, i64 %tmp164
  %tmp167 = mul i64 %tmp166, 3
  %tmp168 = icmp slt i64 %tmp167, 0
  %tmp169 = select i1 %tmp168, i64 0, i64 %tmp167
  %tmp170 = shl i64 %tmp169, 3
  %tmp171 = icmp ne i64 %tmp170, 0
  %tmp172 = select i1 %tmp171, i64 %tmp170, i64 1
  %tmp173 = tail call noalias i8* @wobble(i64 %tmp172) #1
  %tmp174 = xor i64 %tmp160, -1
  %tmp175 = sub i64 %tmp174, %tmp166
  %tmp176 = getelementptr inbounds [0 x i32], [0 x i32]* %arg3, i64 0, i64 0
  %tmp177 = load i32, i32* %tmp176, align 4
  %tmp178 = sext i32 %tmp177 to i64
  %tmp179 = getelementptr [0 x i32], [0 x i32]* %arg3, i64 0, i64 1
  %tmp180 = load i32, i32* %tmp179, align 4
  %tmp181 = sext i32 %tmp180 to i64
  %tmp182 = getelementptr [0 x i32], [0 x i32]* %arg3, i64 0, i64 2
  %tmp183 = load i32, i32* %tmp182, align 4
  %tmp184 = sext i32 %tmp183 to i64
  %tmp185 = add nsw i64 %tmp184, -1
  %tmp186 = icmp sgt i64 %tmp185, 1
  br i1 %tmp186, label %bb187, label %bb249

bb187:                                            ; preds = %bb136
  br label %bb188

bb188:                                            ; preds = %bb187, %bb245
  %tmp189 = phi i64 [ %tmp247, %bb245 ], [ 2, %bb187 ]
  %tmp190 = add i64 %tmp189, -2
  %tmp191 = srem i64 %tmp190, 3
  %tmp192 = add nsw i64 %tmp191, 1
  %tmp193 = srem i64 %tmp189, 3
  %tmp194 = add nsw i64 %tmp193, 1
  %tmp195 = add nsw i64 %tmp181, -1
  %tmp196 = icmp sgt i64 %tmp195, 1
  br i1 %tmp196, label %bb197, label %bb245

bb197:                                            ; preds = %bb188
  br label %bb198

bb198:                                            ; preds = %bb197, %bb241
  %tmp199 = phi i64 [ %tmp243, %bb241 ], [ 2, %bb197 ]
  %tmp200 = add nsw i64 %tmp178, -1
  %tmp201 = icmp sgt i64 %tmp200, 1
  br i1 %tmp201, label %bb202, label %bb241

bb202:                                            ; preds = %bb198
  br label %bb203

bb203:                                            ; preds = %bb202, %bb203
  %tmp204 = phi i64 [ %tmp239, %bb203 ], [ 2, %bb202 ]
  %tmp205 = mul i64 %tmp199, %tmp160
  %tmp206 = mul i64 %tmp192, %tmp166
  %tmp207 = add i64 %tmp206, %tmp175
  %tmp208 = add i64 %tmp207, %tmp205
  %tmp209 = add i64 %tmp208, %tmp204
  %tmp210 = bitcast i8* %tmp173 to double*
  %tmp211 = getelementptr double, double* %tmp210, i64 %tmp209
  %tmp212 = load double, double* %tmp211, align 8
  %tmp213 = mul i64 %tmp199, %tmp160
  %tmp214 = mul i64 %tmp194, %tmp166
  %tmp215 = add i64 %tmp214, %tmp175
  %tmp216 = add i64 %tmp215, %tmp213
  %tmp217 = add i64 %tmp216, %tmp204
  %tmp218 = bitcast i8* %tmp173 to double*
  %tmp219 = getelementptr double, double* %tmp218, i64 %tmp217
  %tmp220 = load double, double* %tmp219, align 8
  %tmp221 = fadd double %tmp212, %tmp220
  %tmp222 = mul i64 %tmp199, %tmp139
  %tmp223 = mul i64 %tmp189, %tmp144
  %tmp224 = sub i64 %tmp145, %tmp144
  %tmp225 = add i64 %tmp224, %tmp223
  %tmp226 = add i64 %tmp225, %tmp222
  %tmp227 = add i64 %tmp226, %tmp204
  %tmp228 = mul i64 %tmp199, %tmp149
  %tmp229 = mul i64 %tmp189, %tmp154
  %tmp230 = sub i64 %tmp155, %tmp154
  %tmp231 = add i64 %tmp230, %tmp229
  %tmp232 = add i64 %tmp231, %tmp228
  %tmp233 = add i64 %tmp232, %tmp204
  %tmp234 = getelementptr [0 x double], [0 x double]* %arg56, i64 0, i64 %tmp233
  %tmp235 = load double, double* %tmp234, align 8
  %tmp236 = fadd double %tmp235, %tmp221
  %tmp237 = getelementptr [0 x double], [0 x double]* %arg55, i64 0, i64 %tmp227
  store double %tmp236, double* %tmp237, align 8
  %tmp238 = icmp eq i64 %tmp204, %tmp200
  %tmp239 = add i64 %tmp204, 1
  br i1 %tmp238, label %bb240, label %bb203

bb240:                                            ; preds = %bb203
  br label %bb241

bb241:                                            ; preds = %bb240, %bb198
  %tmp242 = icmp eq i64 %tmp199, %tmp195
  %tmp243 = add i64 %tmp199, 1
  br i1 %tmp242, label %bb244, label %bb198

bb244:                                            ; preds = %bb241
  br label %bb245

bb245:                                            ; preds = %bb244, %bb188
  %tmp246 = icmp eq i64 %tmp189, %tmp185
  %tmp247 = add i64 %tmp189, 1
  br i1 %tmp246, label %bb248, label %bb188

bb248:                                            ; preds = %bb245
  br label %bb249

bb249:                                            ; preds = %bb248, %bb136
  %tmp250 = icmp eq i8* %tmp173, null
  br i1 %tmp250, label %bb252, label %bb251

bb251:                                            ; preds = %bb249
  tail call void @snork(i8* %tmp173) #1
  br label %bb252

bb252:                                            ; preds = %bb251, %bb249
  ret void
}

; Function Attrs: nounwind
declare noalias i8* @wobble(i64) #1

; Function Attrs: nounwind
declare void @snork(i8*) #1
