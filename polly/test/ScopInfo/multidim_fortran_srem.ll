; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-print-scops -disable-output < %s | FileCheck %s
target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

; CHECK:      Statements {
; CHECK-NEXT:     Stmt_bb188
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [tmp180, tmp177, tmp183, tmp162, tmp157, tmp150, tmp146, tmp140, tmp] -> { Stmt_bb188[i0] : 0 <= i0 <= -3 + tmp183 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [tmp180, tmp177, tmp183, tmp162, tmp157, tmp150, tmp146, tmp140, tmp] -> { Stmt_bb188[i0] -> [i0, 0, 0, 0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [tmp180, tmp177, tmp183, tmp162, tmp157, tmp150, tmp146, tmp140, tmp] -> { Stmt_bb188[i0] -> MemRef_tmp192[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [tmp180, tmp177, tmp183, tmp162, tmp157, tmp150, tmp146, tmp140, tmp] -> { Stmt_bb188[i0] -> MemRef_tmp194[] };
; CHECK-NEXT:     Stmt_bb203
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [tmp180, tmp177, tmp183, tmp162, tmp157, tmp150, tmp146, tmp140, tmp] -> { Stmt_bb203[i0, i1, i2] : 0 <= i0 <= -3 + tmp183 and 0 <= i1 <= -3 + tmp180 and 0 <= i2 <= -3 + tmp177 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [tmp180, tmp177, tmp183, tmp162, tmp157, tmp150, tmp146, tmp140, tmp] -> { Stmt_bb203[i0, i1, i2] -> [i0, 1, i1, i2] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [tmp180, tmp177, tmp183, tmp162, tmp157, tmp150, tmp146, tmp140, tmp] -> { Stmt_bb203[i0, i1, i2] -> MemRef_tmp192[] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [tmp180, tmp177, tmp183, tmp162, tmp157, tmp150, tmp146, tmp140, tmp] -> { Stmt_bb203[i0, i1, i2] -> MemRef_tmp173[o0, 1 + i1, 1 + i2] : (-i0 + o0) mod 3 = 0 and 0 <= o0 <= 2 }
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [tmp180, tmp177, tmp183, tmp162, tmp157, tmp150, tmp146, tmp140, tmp] -> { Stmt_bb203[i0, i1, i2] -> MemRef_tmp194[] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [tmp180, tmp177, tmp183, tmp162, tmp157, tmp150, tmp146, tmp140, tmp] -> { Stmt_bb203[i0, i1, i2] -> MemRef_tmp173[o0, 1 + i1, 1 + i2] : (1 - i0 + o0) mod 3 = 0 and 0 <= o0 <= 2 }
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [tmp180, tmp177, tmp183, tmp162, tmp157, tmp150, tmp146, tmp140, tmp] -> { Stmt_bb203[i0, i1, i2] -> MemRef_arg56[1 + i0, 1 + i1, 1 + i2] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [tmp180, tmp177, tmp183, tmp162, tmp157, tmp150, tmp146, tmp140, tmp] -> { Stmt_bb203[i0, i1, i2] -> MemRef_arg55[1 + i0, 1 + i1, 1 + i2] };
; CHECK-NEXT: }

define void @pluto(ptr noalias %arg, ptr noalias %arg2, ptr noalias %arg3, ptr noalias %arg4, ptr noalias %arg5, ptr noalias %arg6, ptr noalias %arg7, ptr noalias %arg8, ptr noalias %arg9, ptr noalias %arg10, ptr noalias %arg11, ptr noalias %arg12, ptr noalias %arg13, ptr noalias %arg14, ptr noalias %arg15, ptr noalias %arg16, ptr noalias %arg17, ptr noalias %arg18, ptr noalias %arg19, ptr noalias %arg20, ptr noalias %arg21, ptr noalias %arg22, ptr noalias %arg23, ptr noalias %arg24, ptr noalias %arg25, ptr noalias %arg26, ptr noalias %arg27, ptr noalias %arg28, ptr noalias %arg29, ptr noalias %arg30, ptr noalias %arg31, ptr noalias %arg32, ptr noalias %arg33, ptr noalias %arg34, ptr noalias %arg35, ptr noalias %arg36, ptr noalias %arg37, ptr noalias %arg38, ptr noalias %arg39, ptr noalias %arg40, ptr noalias %arg41, ptr noalias %arg42, ptr noalias %arg43, ptr noalias %arg44, ptr noalias %arg45, ptr noalias %arg46, ptr noalias %arg47, ptr noalias %arg48, ptr noalias %arg49, ptr noalias %arg50, ptr noalias %arg51, ptr noalias %arg52, ptr noalias %arg53, ptr noalias %arg54, ptr noalias %arg55, ptr noalias %arg56, ptr noalias %arg57, ptr noalias %arg58, ptr noalias %arg59, ptr noalias %arg60, ptr noalias %arg61, ptr noalias %arg62, ptr noalias %arg63, ptr noalias %arg64, ptr noalias %arg65, ptr noalias %arg66, ptr noalias %arg67, ptr noalias %arg68, ptr noalias %arg69, ptr noalias %arg70, ptr noalias %arg71, ptr noalias %arg72, ptr noalias %arg73, ptr noalias %arg74, ptr noalias %arg75, ptr noalias %arg76, ptr noalias %arg77, ptr noalias %arg78, ptr noalias %arg79, ptr noalias %arg80, ptr noalias %arg81, ptr noalias %arg82, ptr noalias %arg83, ptr noalias %arg84, ptr noalias %arg85, ptr noalias %arg86, ptr noalias %arg87, ptr noalias %arg88, ptr noalias %arg89, ptr noalias %arg90, ptr noalias %arg91, ptr noalias %arg92, ptr noalias %arg93, ptr noalias %arg94, ptr noalias %arg95, ptr noalias %arg96, ptr noalias %arg97, ptr noalias %arg98, ptr noalias %arg99, ptr noalias %arg100, ptr noalias %arg101, ptr noalias %arg102, ptr noalias %arg103, ptr noalias %arg104, ptr noalias %arg105, ptr noalias %arg106, ptr noalias %arg107, ptr noalias %arg108, ptr noalias %arg109, ptr noalias %arg110, ptr noalias %arg111, ptr noalias %arg112, ptr noalias %arg113, ptr noalias %arg114, ptr noalias %arg115, ptr noalias %arg116, ptr noalias %arg117, ptr noalias %arg118, ptr noalias %arg119, ptr noalias %arg120, ptr noalias %arg121, ptr noalias %arg122, ptr noalias %arg123, ptr noalias %arg124, ptr noalias %arg125, ptr noalias %arg126, ptr noalias %arg127, ptr noalias %arg128, ptr noalias %arg129, ptr noalias %arg130, ptr noalias %arg131, ptr noalias %arg132, ptr noalias %arg133, ptr noalias %arg134, ptr noalias %arg135) {
bb:
  br label %bb136

bb136:                                            ; preds = %bb
  %tmp = load i32, ptr %arg19, align 4
  %tmp137 = sext i32 %tmp to i64
  %tmp138 = icmp slt i64 %tmp137, 0
  %tmp139 = select i1 %tmp138, i64 0, i64 %tmp137
  %tmp140 = load i32, ptr %arg20, align 4
  %tmp141 = sext i32 %tmp140 to i64
  %tmp142 = mul nsw i64 %tmp139, %tmp141
  %tmp143 = icmp slt i64 %tmp142, 0
  %tmp144 = select i1 %tmp143, i64 0, i64 %tmp142
  %tmp145 = xor i64 %tmp139, -1
  %tmp146 = load i32, ptr %arg19, align 4
  %tmp147 = sext i32 %tmp146 to i64
  %tmp148 = icmp slt i64 %tmp147, 0
  %tmp149 = select i1 %tmp148, i64 0, i64 %tmp147
  %tmp150 = load i32, ptr %arg20, align 4
  %tmp151 = sext i32 %tmp150 to i64
  %tmp152 = mul nsw i64 %tmp149, %tmp151
  %tmp153 = icmp slt i64 %tmp152, 0
  %tmp154 = select i1 %tmp153, i64 0, i64 %tmp152
  %tmp155 = xor i64 %tmp149, -1
  %tmp157 = load i32, ptr %arg3, align 4
  %tmp158 = sext i32 %tmp157 to i64
  %tmp159 = icmp slt i64 %tmp158, 0
  %tmp160 = select i1 %tmp159, i64 0, i64 %tmp158
  %tmp161 = getelementptr [0 x i32], ptr %arg3, i64 0, i64 1
  %tmp162 = load i32, ptr %tmp161, align 4
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
  %tmp173 = tail call noalias ptr @wobble(i64 %tmp172) #1
  %tmp174 = xor i64 %tmp160, -1
  %tmp175 = sub i64 %tmp174, %tmp166
  %tmp177 = load i32, ptr %arg3, align 4
  %tmp178 = sext i32 %tmp177 to i64
  %tmp179 = getelementptr [0 x i32], ptr %arg3, i64 0, i64 1
  %tmp180 = load i32, ptr %tmp179, align 4
  %tmp181 = sext i32 %tmp180 to i64
  %tmp182 = getelementptr [0 x i32], ptr %arg3, i64 0, i64 2
  %tmp183 = load i32, ptr %tmp182, align 4
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
  %tmp211 = getelementptr double, ptr %tmp173, i64 %tmp209
  %tmp212 = load double, ptr %tmp211, align 8
  %tmp213 = mul i64 %tmp199, %tmp160
  %tmp214 = mul i64 %tmp194, %tmp166
  %tmp215 = add i64 %tmp214, %tmp175
  %tmp216 = add i64 %tmp215, %tmp213
  %tmp217 = add i64 %tmp216, %tmp204
  %tmp219 = getelementptr double, ptr %tmp173, i64 %tmp217
  %tmp220 = load double, ptr %tmp219, align 8
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
  %tmp234 = getelementptr [0 x double], ptr %arg56, i64 0, i64 %tmp233
  %tmp235 = load double, ptr %tmp234, align 8
  %tmp236 = fadd double %tmp235, %tmp221
  %tmp237 = getelementptr [0 x double], ptr %arg55, i64 0, i64 %tmp227
  store double %tmp236, ptr %tmp237, align 8
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
  %tmp250 = icmp eq ptr %tmp173, null
  br i1 %tmp250, label %bb252, label %bb251

bb251:                                            ; preds = %bb249
  tail call void @snork(ptr %tmp173) #1
  br label %bb252

bb252:                                            ; preds = %bb251, %bb249
  ret void
}

; Function Attrs: nounwind
declare noalias ptr @wobble(i64) #1

; Function Attrs: nounwind
declare void @snork(ptr) #1
