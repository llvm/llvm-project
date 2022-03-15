; RUN: opt %loadPolly -polly-print-scops -disable-output < %s

; Verify that the compilation of this test case does not take infinite time.
; At some point Polly tried to model this test case and got stuck in
; computing a lexicographic minima. Today it should gracefully bail out.
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android"

%0 = type { i8*, i64, i64, i64, i64, i64, i64 }

define void @_Z1fR1SS0_Ph(%0* nocapture readonly dereferenceable(56) %arg, %0* nocapture readonly dereferenceable(56) %arg1, i8* nocapture readonly %arg2) {
bb:
  %tmp = getelementptr inbounds %0, %0* %arg1, i64 0, i32 1
  %tmp3 = getelementptr inbounds %0, %0* %arg, i64 0, i32 0
  %tmp4 = load i8*, i8** %tmp3, align 8
  %tmp5 = getelementptr inbounds %0, %0* %arg, i64 0, i32 4
  %tmp6 = load i64, i64* %tmp5, align 8
  %tmp7 = getelementptr inbounds %0, %0* %arg, i64 0, i32 1
  %tmp8 = load i64, i64* %tmp7, align 8
  %tmp9 = mul i64 %tmp8, %tmp6
  %tmp10 = getelementptr inbounds i8, i8* %tmp4, i64 %tmp9
  %tmp11 = getelementptr inbounds %0, %0* %arg, i64 0, i32 3
  %tmp12 = load i64, i64* %tmp11, align 8
  %tmp13 = getelementptr inbounds i8, i8* %tmp10, i64 %tmp12
  %tmp14 = getelementptr inbounds %0, %0* %arg, i64 0, i32 6
  %tmp15 = load i64, i64* %tmp14, align 8
  %tmp16 = add i64 %tmp15, 1
  %tmp17 = icmp eq i64 %tmp16, %tmp6
  br i1 %tmp17, label %bb51, label %bb18

bb18:                                             ; preds = %bb
  %tmp19 = getelementptr inbounds %0, %0* %arg, i64 0, i32 2
  %tmp20 = load i64, i64* %tmp19, align 8
  %tmp21 = mul i64 %tmp20, %tmp8
  %tmp22 = getelementptr inbounds i8, i8* %tmp13, i64 %tmp21
  %tmp23 = getelementptr inbounds i8, i8* %tmp22, i64 %tmp9
  %tmp24 = getelementptr inbounds i8, i8* %tmp23, i64 %tmp12
  %tmp25 = bitcast %0* %arg1 to i16**
  %tmp26 = load i16*, i16** %tmp25, align 8
  %tmp27 = load i64, i64* %tmp, align 8
  %tmp28 = getelementptr inbounds %0, %0* %arg1, i64 0, i32 4
  %tmp29 = load i64, i64* %tmp28, align 8
  %tmp30 = mul i64 %tmp27, %tmp29
  %tmp31 = getelementptr inbounds i16, i16* %tmp26, i64 %tmp30
  %tmp32 = getelementptr inbounds %0, %0* %arg1, i64 0, i32 3
  %tmp33 = load i64, i64* %tmp32, align 8
  %tmp34 = getelementptr inbounds i16, i16* %tmp31, i64 %tmp33
  %tmp35 = getelementptr inbounds %0, %0* %arg, i64 0, i32 5
  %tmp36 = load i64, i64* %tmp35, align 8
  br label %bb37

bb37:                                             ; preds = %bb57, %bb18
  %tmp38 = phi i64 [ %tmp6, %bb18 ], [ %tmp58, %bb57 ]
  %tmp39 = phi i64 [ %tmp15, %bb18 ], [ %tmp59, %bb57 ]
  %tmp40 = phi i64 [ %tmp27, %bb18 ], [ %tmp60, %bb57 ]
  %tmp41 = phi i64 [ %tmp8, %bb18 ], [ %tmp61, %bb57 ]
  %tmp42 = phi i64 [ %tmp12, %bb18 ], [ %tmp62, %bb57 ]
  %tmp43 = phi i64 [ %tmp36, %bb18 ], [ %tmp63, %bb57 ]
  %tmp44 = phi i16* [ %tmp34, %bb18 ], [ %tmp69, %bb57 ]
  %tmp45 = phi i8* [ %tmp13, %bb18 ], [ %tmp64, %bb57 ]
  %tmp46 = phi i8* [ %tmp24, %bb18 ], [ %tmp68, %bb57 ]
  %tmp47 = phi i64 [ 0, %bb18 ], [ %tmp70, %bb57 ]
  %tmp48 = add i64 %tmp43, 1
  %tmp49 = sub i64 %tmp48, %tmp42
  %tmp50 = icmp eq i64 %tmp49, 0
  br i1 %tmp50, label %bb57, label %bb74

bb51:                                             ; preds = %bb57, %bb
  ret void

bb52:                                             ; preds = %bb176
  %tmp53 = load i64, i64* %tmp7, align 8
  %tmp54 = load i64, i64* %tmp, align 8
  %tmp55 = load i64, i64* %tmp14, align 8
  %tmp56 = load i64, i64* %tmp5, align 8
  br label %bb57

bb57:                                             ; preds = %bb52, %bb37
  %tmp58 = phi i64 [ %tmp56, %bb52 ], [ %tmp38, %bb37 ]
  %tmp59 = phi i64 [ %tmp55, %bb52 ], [ %tmp39, %bb37 ]
  %tmp60 = phi i64 [ %tmp54, %bb52 ], [ %tmp40, %bb37 ]
  %tmp61 = phi i64 [ %tmp53, %bb52 ], [ %tmp41, %bb37 ]
  %tmp62 = phi i64 [ %tmp179, %bb52 ], [ %tmp42, %bb37 ]
  %tmp63 = phi i64 [ %tmp178, %bb52 ], [ %tmp43, %bb37 ]
  %tmp64 = getelementptr inbounds i8, i8* %tmp45, i64 %tmp61
  %tmp65 = and i64 %tmp47, 1
  %tmp66 = icmp eq i64 %tmp65, 0
  %tmp67 = getelementptr inbounds i8, i8* %tmp46, i64 %tmp61
  %tmp68 = select i1 %tmp66, i8* %tmp46, i8* %tmp67
  %tmp69 = getelementptr inbounds i16, i16* %tmp44, i64 %tmp60
  %tmp70 = add i64 %tmp47, 1
  %tmp71 = add i64 %tmp59, 1
  %tmp72 = sub i64 %tmp71, %tmp58
  %tmp73 = icmp ult i64 %tmp70, %tmp72
  br i1 %tmp73, label %bb37, label %bb51

bb74:                                             ; preds = %bb176, %bb37
  %tmp75 = phi i64 [ %tmp181, %bb176 ], [ %tmp49, %bb37 ]
  %tmp76 = phi i64 [ %tmp177, %bb176 ], [ 0, %bb37 ]
  %tmp77 = getelementptr inbounds i8, i8* %tmp45, i64 %tmp76
  %tmp78 = load i8, i8* %tmp77, align 1
  %tmp79 = zext i8 %tmp78 to i32
  %tmp80 = or i64 %tmp76, 1
  %tmp81 = getelementptr inbounds i8, i8* %tmp45, i64 %tmp80
  %tmp82 = load i8, i8* %tmp81, align 1
  %tmp83 = zext i8 %tmp82 to i32
  %tmp84 = getelementptr inbounds i8, i8* %tmp46, i64 %tmp76
  %tmp85 = load i8, i8* %tmp84, align 1
  %tmp86 = zext i8 %tmp85 to i32
  %tmp87 = getelementptr inbounds i8, i8* %tmp46, i64 %tmp80
  %tmp88 = load i8, i8* %tmp87, align 1
  %tmp89 = zext i8 %tmp88 to i32
  %tmp90 = mul nuw nsw i32 %tmp86, 517
  %tmp91 = add nsw i32 %tmp90, -66176
  %tmp92 = sub nsw i32 128, %tmp86
  %tmp93 = mul nsw i32 %tmp92, 100
  %tmp94 = sub nsw i32 128, %tmp89
  %tmp95 = mul nsw i32 %tmp94, 208
  %tmp96 = mul nuw nsw i32 %tmp89, 409
  %tmp97 = add nsw i32 %tmp96, -52352
  %tmp98 = mul nuw nsw i32 %tmp79, 298
  %tmp99 = add nsw i32 %tmp98, -4768
  %tmp100 = add nsw i32 %tmp91, %tmp99
  %tmp101 = sdiv i32 %tmp100, 256
  %tmp102 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %tmp99, i32 %tmp95)
  %tmp103 = extractvalue { i32, i1 } %tmp102, 1
  br i1 %tmp103, label %bb104, label %bb105

bb104:                                            ; preds = %bb120, %bb109, %bb105, %bb74
  tail call void @llvm.trap()
  unreachable

bb105:                                            ; preds = %bb74
  %tmp106 = extractvalue { i32, i1 } %tmp102, 0
  %tmp107 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %tmp106, i32 %tmp93)
  %tmp108 = extractvalue { i32, i1 } %tmp107, 1
  br i1 %tmp108, label %bb104, label %bb109

bb109:                                            ; preds = %bb105
  %tmp110 = extractvalue { i32, i1 } %tmp107, 0
  %tmp111 = sdiv i32 %tmp110, 256
  %tmp112 = add nsw i32 %tmp97, %tmp99
  %tmp113 = sdiv i32 %tmp112, 256
  %tmp114 = mul nuw nsw i32 %tmp83, 298
  %tmp115 = add nsw i32 %tmp114, -4768
  %tmp116 = add nsw i32 %tmp91, %tmp115
  %tmp117 = sdiv i32 %tmp116, 256
  %tmp118 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %tmp115, i32 %tmp95)
  %tmp119 = extractvalue { i32, i1 } %tmp118, 1
  br i1 %tmp119, label %bb104, label %bb120

bb120:                                            ; preds = %bb109
  %tmp121 = extractvalue { i32, i1 } %tmp118, 0
  %tmp122 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %tmp121, i32 %tmp93)
  %tmp123 = extractvalue { i32, i1 } %tmp122, 1
  br i1 %tmp123, label %bb104, label %bb124

bb124:                                            ; preds = %bb120
  %tmp125 = sext i32 %tmp101 to i64
  %tmp126 = getelementptr inbounds i8, i8* %arg2, i64 %tmp125
  %tmp127 = load i8, i8* %tmp126, align 1
  %tmp128 = zext i8 %tmp127 to i32
  %tmp129 = lshr i32 %tmp128, 3
  %tmp130 = shl nuw nsw i32 %tmp129, 11
  %tmp131 = sext i32 %tmp111 to i64
  %tmp132 = getelementptr inbounds i8, i8* %arg2, i64 %tmp131
  %tmp133 = load i8, i8* %tmp132, align 1
  %tmp134 = zext i8 %tmp133 to i32
  %tmp135 = lshr i32 %tmp134, 2
  %tmp136 = shl nuw nsw i32 %tmp135, 5
  %tmp137 = or i32 %tmp136, %tmp130
  %tmp138 = sext i32 %tmp113 to i64
  %tmp139 = getelementptr inbounds i8, i8* %arg2, i64 %tmp138
  %tmp140 = load i8, i8* %tmp139, align 1
  %tmp141 = zext i8 %tmp140 to i32
  %tmp142 = lshr i32 %tmp141, 3
  %tmp143 = or i32 %tmp137, %tmp142
  %tmp144 = icmp ult i64 %tmp80, %tmp75
  br i1 %tmp144, label %bb145, label %bb173

bb145:                                            ; preds = %bb124
  %tmp146 = add nsw i32 %tmp97, %tmp115
  %tmp147 = sdiv i32 %tmp146, 256
  %tmp148 = sext i32 %tmp147 to i64
  %tmp149 = getelementptr inbounds i8, i8* %arg2, i64 %tmp148
  %tmp150 = load i8, i8* %tmp149, align 1
  %tmp151 = extractvalue { i32, i1 } %tmp122, 0
  %tmp152 = sdiv i32 %tmp151, 256
  %tmp153 = sext i32 %tmp152 to i64
  %tmp154 = getelementptr inbounds i8, i8* %arg2, i64 %tmp153
  %tmp155 = load i8, i8* %tmp154, align 1
  %tmp156 = sext i32 %tmp117 to i64
  %tmp157 = getelementptr inbounds i8, i8* %arg2, i64 %tmp156
  %tmp158 = load i8, i8* %tmp157, align 1
  %tmp159 = zext i8 %tmp158 to i32
  %tmp160 = lshr i32 %tmp159, 3
  %tmp161 = shl nuw nsw i32 %tmp160, 11
  %tmp162 = zext i8 %tmp155 to i32
  %tmp163 = lshr i32 %tmp162, 2
  %tmp164 = shl nuw nsw i32 %tmp163, 5
  %tmp165 = zext i8 %tmp150 to i32
  %tmp166 = lshr i32 %tmp165, 3
  %tmp167 = or i32 %tmp164, %tmp166
  %tmp168 = or i32 %tmp167, %tmp161
  %tmp169 = shl nuw i32 %tmp168, 16
  %tmp170 = or i32 %tmp169, %tmp143
  %tmp171 = getelementptr inbounds i16, i16* %tmp44, i64 %tmp76
  %tmp172 = bitcast i16* %tmp171 to i32*
  store i32 %tmp170, i32* %tmp172, align 4
  br label %bb176

bb173:                                            ; preds = %bb124
  %tmp174 = trunc i32 %tmp143 to i16
  %tmp175 = getelementptr inbounds i16, i16* %tmp44, i64 %tmp76
  store i16 %tmp174, i16* %tmp175, align 2
  br label %bb176

bb176:                                            ; preds = %bb173, %bb145
  %tmp177 = add i64 %tmp76, 2
  %tmp178 = load i64, i64* %tmp35, align 8
  %tmp179 = load i64, i64* %tmp11, align 8
  %tmp180 = add i64 %tmp178, 1
  %tmp181 = sub i64 %tmp180, %tmp179
  %tmp182 = icmp ult i64 %tmp177, %tmp181
  br i1 %tmp182, label %bb74, label %bb52
}

; Function Attrs: noreturn nounwind
declare void @llvm.trap() #0

; Function Attrs: nounwind readnone speculatable
declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32) #1

attributes #0 = { noreturn nounwind }
attributes #1 = { nounwind readnone speculatable }
