; RUN: opt %loadNPMPolly '-passes=print<polly-detect>,print<polly-function-scops>' -disable-output \
; RUN:                < %s 2>&1 | FileCheck %s

; CHECK-NOT:   Assumed Context:
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.hoge = type { i32, ptr, ptr, ptr, i32, ptr, ptr, i32, i32, ptr, i32, ptr, [6 x i32], i32, ptr, i32 }
%struct.widget = type { i32, i32, ptr, ptr, ptr, i32, ptr, i32, i32, [3 x i32], i32 }
%struct.quux = type { ptr, ptr, ptr, i32, i32, i32, ptr }
%struct.hoge.0 = type { i32, ptr }
%struct.barney = type { [3 x i32], [3 x i32] }
%struct.ham = type { ptr, i32, i32, i32, i32 }
%struct.wombat = type { ptr, i32, i32 }
%struct.foo = type { i32, i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%struct.wibble = type { ptr, i32 }
%struct.foo.1 = type { [3 x i32], [3 x i32], i32, i32, [4 x i32], [4 x i32] }

; Function Attrs: nounwind uwtable
define void @hoge() #0 {
bb:
  %tmp52 = alloca ptr, align 8
  %tmp53 = alloca ptr, align 8
  %tmp54 = alloca ptr, align 8
  %tmp55 = alloca ptr, align 8
  br label %bb56

bb56:                                             ; preds = %bb
  switch i32 undef, label %bb59 [
    i32 0, label %bb57
    i32 1, label %bb58
  ]

bb57:                                             ; preds = %bb56
  unreachable

bb58:                                             ; preds = %bb56
  unreachable

bb59:                                             ; preds = %bb56
  %tmp = load ptr, ptr %tmp54, align 8, !tbaa !1
  %tmp60 = getelementptr inbounds %struct.barney, ptr %tmp, i32 0, i32 1
  %tmp62 = load i32, ptr %tmp60, align 4, !tbaa !5
  %tmp63 = load ptr, ptr %tmp54, align 8, !tbaa !1
  %tmp66 = sub nsw i32 %tmp62, 0
  %tmp67 = add nsw i32 %tmp66, 1
  %tmp68 = icmp slt i32 0, %tmp67
  br i1 %tmp68, label %bb69, label %bb70

bb69:                                             ; preds = %bb59
  br label %bb70

bb70:                                             ; preds = %bb69, %bb59
  %tmp71 = load ptr, ptr %tmp54, align 8, !tbaa !1
  %tmp72 = getelementptr inbounds %struct.barney, ptr %tmp71, i32 0, i32 1
  %tmp73 = getelementptr inbounds [3 x i32], ptr %tmp72, i64 0, i64 1
  %tmp74 = load i32, ptr %tmp73, align 4, !tbaa !5
  %tmp75 = load ptr, ptr %tmp54, align 8, !tbaa !1
  %tmp77 = getelementptr inbounds [3 x i32], ptr %tmp75, i64 0, i64 1
  %tmp78 = sub nsw i32 %tmp74, 0
  %tmp79 = add nsw i32 %tmp78, 1
  %tmp80 = icmp slt i32 0, %tmp79
  br i1 %tmp80, label %bb81, label %bb82

bb81:                                             ; preds = %bb70
  br label %bb82

bb82:                                             ; preds = %bb81, %bb70
  %tmp83 = load ptr, ptr %tmp54, align 8, !tbaa !1
  %tmp84 = getelementptr inbounds %struct.barney, ptr %tmp83, i32 0, i32 1
  %tmp86 = load i32, ptr %tmp84, align 4, !tbaa !5
  %tmp87 = load ptr, ptr %tmp54, align 8, !tbaa !1
  %tmp90 = sub nsw i32 %tmp86, 0
  %tmp91 = add nsw i32 %tmp90, 1
  %tmp92 = icmp slt i32 0, %tmp91
  br i1 %tmp92, label %bb93, label %bb94

bb93:                                             ; preds = %bb82
  br label %bb94

bb94:                                             ; preds = %bb93, %bb82
  %tmp95 = load ptr, ptr %tmp54, align 8, !tbaa !1
  %tmp96 = getelementptr inbounds %struct.barney, ptr %tmp95, i32 0, i32 1
  %tmp98 = load i32, ptr %tmp96, align 4, !tbaa !5
  %tmp99 = load ptr, ptr %tmp54, align 8, !tbaa !1
  %tmp102 = sub nsw i32 %tmp98, 0
  %tmp103 = add nsw i32 %tmp102, 1
  %tmp104 = icmp slt i32 0, %tmp103
  br i1 %tmp104, label %bb105, label %bb106

bb105:                                            ; preds = %bb94
  br label %bb106

bb106:                                            ; preds = %bb105, %bb94
  %tmp107 = load ptr, ptr %tmp53, align 8, !tbaa !1
  %tmp108 = getelementptr inbounds %struct.barney, ptr %tmp107, i32 0, i32 1
  %tmp109 = getelementptr inbounds [3 x i32], ptr %tmp108, i64 0, i64 1
  %tmp110 = load i32, ptr %tmp109, align 4, !tbaa !5
  %tmp111 = load ptr, ptr %tmp53, align 8, !tbaa !1
  %tmp113 = getelementptr inbounds [3 x i32], ptr %tmp111, i64 0, i64 1
  %tmp114 = sub nsw i32 %tmp110, 0
  %tmp115 = add nsw i32 %tmp114, 1
  %tmp116 = icmp slt i32 0, %tmp115
  br i1 %tmp116, label %bb117, label %bb118

bb117:                                            ; preds = %bb106
  br label %bb118

bb118:                                            ; preds = %bb117, %bb106
  %tmp119 = load ptr, ptr %tmp53, align 8, !tbaa !1
  %tmp120 = getelementptr inbounds %struct.barney, ptr %tmp119, i32 0, i32 1
  %tmp122 = load i32, ptr %tmp120, align 4, !tbaa !5
  %tmp123 = load ptr, ptr %tmp53, align 8, !tbaa !1
  %tmp126 = sub nsw i32 %tmp122, 0
  %tmp127 = add nsw i32 %tmp126, 1
  %tmp128 = icmp slt i32 0, %tmp127
  br i1 %tmp128, label %bb129, label %bb130

bb129:                                            ; preds = %bb118
  br label %bb130

bb130:                                            ; preds = %bb129, %bb118
  %tmp131 = load ptr, ptr %tmp54, align 8, !tbaa !1
  %tmp132 = getelementptr inbounds %struct.barney, ptr %tmp131, i32 0, i32 1
  %tmp134 = load i32, ptr %tmp132, align 4, !tbaa !5
  %tmp135 = load ptr, ptr %tmp54, align 8, !tbaa !1
  %tmp138 = sub nsw i32 %tmp134, 0
  %tmp139 = add nsw i32 %tmp138, 1
  %tmp140 = icmp slt i32 0, %tmp139
  br i1 %tmp140, label %bb141, label %bb142

bb141:                                            ; preds = %bb130
  br label %bb142

bb142:                                            ; preds = %bb141, %bb130
  %tmp143 = load ptr, ptr %tmp55, align 8, !tbaa !1
  %tmp144 = getelementptr inbounds %struct.barney, ptr %tmp143, i32 0, i32 1
  %tmp146 = load i32, ptr %tmp144, align 4, !tbaa !5
  %tmp147 = load ptr, ptr %tmp55, align 8, !tbaa !1
  %tmp150 = sub nsw i32 %tmp146, 0
  %tmp151 = add nsw i32 %tmp150, 1
  %tmp152 = icmp slt i32 0, %tmp151
  br i1 %tmp152, label %bb153, label %bb154

bb153:                                            ; preds = %bb142
  br label %bb154

bb154:                                            ; preds = %bb153, %bb142
  %tmp155 = load ptr, ptr %tmp53, align 8, !tbaa !1
  %tmp156 = getelementptr inbounds %struct.barney, ptr %tmp155, i32 0, i32 1
  %tmp158 = load i32, ptr %tmp156, align 4, !tbaa !5
  %tmp159 = load ptr, ptr %tmp53, align 8, !tbaa !1
  %tmp162 = load i32, ptr %tmp159, align 4, !tbaa !5
  %tmp163 = sub nsw i32 %tmp158, %tmp162
  %tmp164 = add nsw i32 %tmp163, 1
  %tmp165 = icmp slt i32 0, %tmp164
  br i1 %tmp165, label %bb166, label %bb167

bb166:                                            ; preds = %bb154
  br label %bb167

bb167:                                            ; preds = %bb166, %bb154
  %tmp168 = load ptr, ptr %tmp53, align 8, !tbaa !1
  %tmp169 = getelementptr inbounds %struct.barney, ptr %tmp168, i32 0, i32 1
  %tmp171 = load i32, ptr %tmp169, align 4, !tbaa !5
  %tmp172 = load ptr, ptr %tmp53, align 8, !tbaa !1
  %tmp175 = load i32, ptr %tmp172, align 4, !tbaa !5
  %tmp176 = sub nsw i32 %tmp171, %tmp175
  %tmp177 = add nsw i32 %tmp176, 1
  %tmp178 = icmp slt i32 0, %tmp177
  br i1 %tmp178, label %bb179, label %bb180

bb179:                                            ; preds = %bb167
  br label %bb180

bb180:                                            ; preds = %bb179, %bb167
  %tmp181 = load ptr, ptr %tmp53, align 8, !tbaa !1
  %tmp182 = getelementptr inbounds %struct.barney, ptr %tmp181, i32 0, i32 1
  %tmp183 = getelementptr inbounds [3 x i32], ptr %tmp182, i64 0, i64 1
  %tmp184 = load i32, ptr %tmp183, align 4, !tbaa !5
  %tmp185 = load ptr, ptr %tmp53, align 8, !tbaa !1
  %tmp187 = getelementptr inbounds [3 x i32], ptr %tmp185, i64 0, i64 1
  %tmp188 = load i32, ptr %tmp187, align 4, !tbaa !5
  %tmp189 = sub nsw i32 %tmp184, %tmp188
  %tmp190 = add nsw i32 %tmp189, 1
  %tmp191 = icmp slt i32 0, %tmp190
  br i1 %tmp191, label %bb192, label %bb193

bb192:                                            ; preds = %bb180
  br label %bb193

bb193:                                            ; preds = %bb192, %bb180
  %tmp194 = load ptr, ptr %tmp54, align 8, !tbaa !1
  %tmp195 = getelementptr inbounds %struct.barney, ptr %tmp194, i32 0, i32 1
  %tmp197 = load i32, ptr %tmp195, align 4, !tbaa !5
  %tmp198 = load ptr, ptr %tmp54, align 8, !tbaa !1
  %tmp201 = load i32, ptr %tmp198, align 4, !tbaa !5
  %tmp202 = sub nsw i32 %tmp197, %tmp201
  %tmp203 = add nsw i32 %tmp202, 1
  %tmp204 = icmp slt i32 0, %tmp203
  br i1 %tmp204, label %bb205, label %bb206

bb205:                                            ; preds = %bb193
  br label %bb206

bb206:                                            ; preds = %bb205, %bb193
  %tmp207 = load ptr, ptr %tmp54, align 8, !tbaa !1
  %tmp208 = getelementptr inbounds %struct.barney, ptr %tmp207, i32 0, i32 1
  %tmp210 = load i32, ptr %tmp208, align 4, !tbaa !5
  %tmp211 = load ptr, ptr %tmp54, align 8, !tbaa !1
  %tmp214 = load i32, ptr %tmp211, align 4, !tbaa !5
  %tmp215 = sub nsw i32 %tmp210, %tmp214
  %tmp216 = add nsw i32 %tmp215, 1
  %tmp217 = icmp slt i32 0, %tmp216
  br i1 %tmp217, label %bb218, label %bb219

bb218:                                            ; preds = %bb206
  br label %bb219

bb219:                                            ; preds = %bb218, %bb206
  %tmp220 = load ptr, ptr %tmp54, align 8, !tbaa !1
  %tmp221 = getelementptr inbounds %struct.barney, ptr %tmp220, i32 0, i32 1
  %tmp222 = getelementptr inbounds [3 x i32], ptr %tmp221, i64 0, i64 1
  %tmp223 = load i32, ptr %tmp222, align 4, !tbaa !5
  %tmp224 = load ptr, ptr %tmp54, align 8, !tbaa !1
  %tmp226 = getelementptr inbounds [3 x i32], ptr %tmp224, i64 0, i64 1
  %tmp227 = load i32, ptr %tmp226, align 4, !tbaa !5
  %tmp228 = sub nsw i32 %tmp223, %tmp227
  %tmp229 = add nsw i32 %tmp228, 1
  %tmp230 = icmp slt i32 0, %tmp229
  br i1 %tmp230, label %bb231, label %bb232

bb231:                                            ; preds = %bb219
  br label %bb232

bb232:                                            ; preds = %bb231, %bb219
  %tmp233 = load ptr, ptr %tmp55, align 8, !tbaa !1
  %tmp234 = getelementptr inbounds %struct.barney, ptr %tmp233, i32 0, i32 1
  %tmp236 = load i32, ptr %tmp234, align 4, !tbaa !5
  %tmp237 = load ptr, ptr %tmp55, align 8, !tbaa !1
  %tmp240 = load i32, ptr %tmp237, align 4, !tbaa !5
  %tmp241 = sub nsw i32 %tmp236, %tmp240
  %tmp242 = add nsw i32 %tmp241, 1
  %tmp243 = icmp slt i32 0, %tmp242
  br i1 %tmp243, label %bb244, label %bb245

bb244:                                            ; preds = %bb232
  br label %bb245

bb245:                                            ; preds = %bb244, %bb232
  unreachable
}

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}

!0 = !{!"clang version 3.8.0 (trunk 252261) (llvm/trunk 252271)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"any pointer", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !3, i64 0}
