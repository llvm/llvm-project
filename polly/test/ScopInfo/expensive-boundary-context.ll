; RUN: opt %loadPolly -polly-scops -analyze \
; RUN:                < %s | FileCheck %s

; CHECK-NOT:   Assumed Context:
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.hoge = type { i32, %struct.widget*, %struct.ham*, %struct.ham*, i32, %struct.wombat*, double*, i32, i32, i32**, i32, i32*, [6 x i32], i32, %struct.foo*, i32 }
%struct.widget = type { i32, i32, %struct.wombat*, i32*, %struct.quux*, i32, %struct.barney*, i32, i32, [3 x i32], i32 }
%struct.quux = type { %struct.wombat*, i32*, i32*, i32, i32, i32, [3 x [3 x [3 x %struct.hoge.0*]]]* }
%struct.hoge.0 = type { i32, %struct.hoge.0* }
%struct.barney = type { [3 x i32], [3 x i32] }
%struct.ham = type { [3 x i32]*, i32, i32, i32, i32 }
%struct.wombat = type { %struct.barney*, i32, i32 }
%struct.foo = type { i32, i32, i32, i32, i32*, i32*, %struct.wibble**, %struct.wibble**, i32*, i32*, %struct.wibble*, %struct.wibble* }
%struct.wibble = type { %struct.foo.1**, i32 }
%struct.foo.1 = type { [3 x i32], [3 x i32], i32, i32, [4 x i32], [4 x i32] }

; Function Attrs: nounwind uwtable
define void @hoge() #0 {
bb:
  %tmp52 = alloca %struct.hoge*, align 8
  %tmp53 = alloca %struct.barney*, align 8
  %tmp54 = alloca %struct.barney*, align 8
  %tmp55 = alloca %struct.barney*, align 8
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
  %tmp = load %struct.barney*, %struct.barney** %tmp54, align 8, !tbaa !1
  %tmp60 = getelementptr inbounds %struct.barney, %struct.barney* %tmp, i32 0, i32 1
  %tmp61 = getelementptr inbounds [3 x i32], [3 x i32]* %tmp60, i64 0, i64 0
  %tmp62 = load i32, i32* %tmp61, align 4, !tbaa !5
  %tmp63 = load %struct.barney*, %struct.barney** %tmp54, align 8, !tbaa !1
  %tmp64 = getelementptr inbounds %struct.barney, %struct.barney* %tmp63, i32 0, i32 0
  %tmp65 = getelementptr inbounds [3 x i32], [3 x i32]* %tmp64, i64 0, i64 0
  %tmp66 = sub nsw i32 %tmp62, 0
  %tmp67 = add nsw i32 %tmp66, 1
  %tmp68 = icmp slt i32 0, %tmp67
  br i1 %tmp68, label %bb69, label %bb70

bb69:                                             ; preds = %bb59
  br label %bb70

bb70:                                             ; preds = %bb69, %bb59
  %tmp71 = load %struct.barney*, %struct.barney** %tmp54, align 8, !tbaa !1
  %tmp72 = getelementptr inbounds %struct.barney, %struct.barney* %tmp71, i32 0, i32 1
  %tmp73 = getelementptr inbounds [3 x i32], [3 x i32]* %tmp72, i64 0, i64 1
  %tmp74 = load i32, i32* %tmp73, align 4, !tbaa !5
  %tmp75 = load %struct.barney*, %struct.barney** %tmp54, align 8, !tbaa !1
  %tmp76 = getelementptr inbounds %struct.barney, %struct.barney* %tmp75, i32 0, i32 0
  %tmp77 = getelementptr inbounds [3 x i32], [3 x i32]* %tmp76, i64 0, i64 1
  %tmp78 = sub nsw i32 %tmp74, 0
  %tmp79 = add nsw i32 %tmp78, 1
  %tmp80 = icmp slt i32 0, %tmp79
  br i1 %tmp80, label %bb81, label %bb82

bb81:                                             ; preds = %bb70
  br label %bb82

bb82:                                             ; preds = %bb81, %bb70
  %tmp83 = load %struct.barney*, %struct.barney** %tmp54, align 8, !tbaa !1
  %tmp84 = getelementptr inbounds %struct.barney, %struct.barney* %tmp83, i32 0, i32 1
  %tmp85 = getelementptr inbounds [3 x i32], [3 x i32]* %tmp84, i64 0, i64 0
  %tmp86 = load i32, i32* %tmp85, align 4, !tbaa !5
  %tmp87 = load %struct.barney*, %struct.barney** %tmp54, align 8, !tbaa !1
  %tmp88 = getelementptr inbounds %struct.barney, %struct.barney* %tmp87, i32 0, i32 0
  %tmp89 = getelementptr inbounds [3 x i32], [3 x i32]* %tmp88, i64 0, i64 0
  %tmp90 = sub nsw i32 %tmp86, 0
  %tmp91 = add nsw i32 %tmp90, 1
  %tmp92 = icmp slt i32 0, %tmp91
  br i1 %tmp92, label %bb93, label %bb94

bb93:                                             ; preds = %bb82
  br label %bb94

bb94:                                             ; preds = %bb93, %bb82
  %tmp95 = load %struct.barney*, %struct.barney** %tmp54, align 8, !tbaa !1
  %tmp96 = getelementptr inbounds %struct.barney, %struct.barney* %tmp95, i32 0, i32 1
  %tmp97 = getelementptr inbounds [3 x i32], [3 x i32]* %tmp96, i64 0, i64 0
  %tmp98 = load i32, i32* %tmp97, align 4, !tbaa !5
  %tmp99 = load %struct.barney*, %struct.barney** %tmp54, align 8, !tbaa !1
  %tmp100 = getelementptr inbounds %struct.barney, %struct.barney* %tmp99, i32 0, i32 0
  %tmp101 = getelementptr inbounds [3 x i32], [3 x i32]* %tmp100, i64 0, i64 0
  %tmp102 = sub nsw i32 %tmp98, 0
  %tmp103 = add nsw i32 %tmp102, 1
  %tmp104 = icmp slt i32 0, %tmp103
  br i1 %tmp104, label %bb105, label %bb106

bb105:                                            ; preds = %bb94
  br label %bb106

bb106:                                            ; preds = %bb105, %bb94
  %tmp107 = load %struct.barney*, %struct.barney** %tmp53, align 8, !tbaa !1
  %tmp108 = getelementptr inbounds %struct.barney, %struct.barney* %tmp107, i32 0, i32 1
  %tmp109 = getelementptr inbounds [3 x i32], [3 x i32]* %tmp108, i64 0, i64 1
  %tmp110 = load i32, i32* %tmp109, align 4, !tbaa !5
  %tmp111 = load %struct.barney*, %struct.barney** %tmp53, align 8, !tbaa !1
  %tmp112 = getelementptr inbounds %struct.barney, %struct.barney* %tmp111, i32 0, i32 0
  %tmp113 = getelementptr inbounds [3 x i32], [3 x i32]* %tmp112, i64 0, i64 1
  %tmp114 = sub nsw i32 %tmp110, 0
  %tmp115 = add nsw i32 %tmp114, 1
  %tmp116 = icmp slt i32 0, %tmp115
  br i1 %tmp116, label %bb117, label %bb118

bb117:                                            ; preds = %bb106
  br label %bb118

bb118:                                            ; preds = %bb117, %bb106
  %tmp119 = load %struct.barney*, %struct.barney** %tmp53, align 8, !tbaa !1
  %tmp120 = getelementptr inbounds %struct.barney, %struct.barney* %tmp119, i32 0, i32 1
  %tmp121 = getelementptr inbounds [3 x i32], [3 x i32]* %tmp120, i64 0, i64 0
  %tmp122 = load i32, i32* %tmp121, align 4, !tbaa !5
  %tmp123 = load %struct.barney*, %struct.barney** %tmp53, align 8, !tbaa !1
  %tmp124 = getelementptr inbounds %struct.barney, %struct.barney* %tmp123, i32 0, i32 0
  %tmp125 = getelementptr inbounds [3 x i32], [3 x i32]* %tmp124, i64 0, i64 0
  %tmp126 = sub nsw i32 %tmp122, 0
  %tmp127 = add nsw i32 %tmp126, 1
  %tmp128 = icmp slt i32 0, %tmp127
  br i1 %tmp128, label %bb129, label %bb130

bb129:                                            ; preds = %bb118
  br label %bb130

bb130:                                            ; preds = %bb129, %bb118
  %tmp131 = load %struct.barney*, %struct.barney** %tmp54, align 8, !tbaa !1
  %tmp132 = getelementptr inbounds %struct.barney, %struct.barney* %tmp131, i32 0, i32 1
  %tmp133 = getelementptr inbounds [3 x i32], [3 x i32]* %tmp132, i64 0, i64 0
  %tmp134 = load i32, i32* %tmp133, align 4, !tbaa !5
  %tmp135 = load %struct.barney*, %struct.barney** %tmp54, align 8, !tbaa !1
  %tmp136 = getelementptr inbounds %struct.barney, %struct.barney* %tmp135, i32 0, i32 0
  %tmp137 = getelementptr inbounds [3 x i32], [3 x i32]* %tmp136, i64 0, i64 0
  %tmp138 = sub nsw i32 %tmp134, 0
  %tmp139 = add nsw i32 %tmp138, 1
  %tmp140 = icmp slt i32 0, %tmp139
  br i1 %tmp140, label %bb141, label %bb142

bb141:                                            ; preds = %bb130
  br label %bb142

bb142:                                            ; preds = %bb141, %bb130
  %tmp143 = load %struct.barney*, %struct.barney** %tmp55, align 8, !tbaa !1
  %tmp144 = getelementptr inbounds %struct.barney, %struct.barney* %tmp143, i32 0, i32 1
  %tmp145 = getelementptr inbounds [3 x i32], [3 x i32]* %tmp144, i64 0, i64 0
  %tmp146 = load i32, i32* %tmp145, align 4, !tbaa !5
  %tmp147 = load %struct.barney*, %struct.barney** %tmp55, align 8, !tbaa !1
  %tmp148 = getelementptr inbounds %struct.barney, %struct.barney* %tmp147, i32 0, i32 0
  %tmp149 = getelementptr inbounds [3 x i32], [3 x i32]* %tmp148, i64 0, i64 0
  %tmp150 = sub nsw i32 %tmp146, 0
  %tmp151 = add nsw i32 %tmp150, 1
  %tmp152 = icmp slt i32 0, %tmp151
  br i1 %tmp152, label %bb153, label %bb154

bb153:                                            ; preds = %bb142
  br label %bb154

bb154:                                            ; preds = %bb153, %bb142
  %tmp155 = load %struct.barney*, %struct.barney** %tmp53, align 8, !tbaa !1
  %tmp156 = getelementptr inbounds %struct.barney, %struct.barney* %tmp155, i32 0, i32 1
  %tmp157 = getelementptr inbounds [3 x i32], [3 x i32]* %tmp156, i64 0, i64 0
  %tmp158 = load i32, i32* %tmp157, align 4, !tbaa !5
  %tmp159 = load %struct.barney*, %struct.barney** %tmp53, align 8, !tbaa !1
  %tmp160 = getelementptr inbounds %struct.barney, %struct.barney* %tmp159, i32 0, i32 0
  %tmp161 = getelementptr inbounds [3 x i32], [3 x i32]* %tmp160, i64 0, i64 0
  %tmp162 = load i32, i32* %tmp161, align 4, !tbaa !5
  %tmp163 = sub nsw i32 %tmp158, %tmp162
  %tmp164 = add nsw i32 %tmp163, 1
  %tmp165 = icmp slt i32 0, %tmp164
  br i1 %tmp165, label %bb166, label %bb167

bb166:                                            ; preds = %bb154
  br label %bb167

bb167:                                            ; preds = %bb166, %bb154
  %tmp168 = load %struct.barney*, %struct.barney** %tmp53, align 8, !tbaa !1
  %tmp169 = getelementptr inbounds %struct.barney, %struct.barney* %tmp168, i32 0, i32 1
  %tmp170 = getelementptr inbounds [3 x i32], [3 x i32]* %tmp169, i64 0, i64 0
  %tmp171 = load i32, i32* %tmp170, align 4, !tbaa !5
  %tmp172 = load %struct.barney*, %struct.barney** %tmp53, align 8, !tbaa !1
  %tmp173 = getelementptr inbounds %struct.barney, %struct.barney* %tmp172, i32 0, i32 0
  %tmp174 = getelementptr inbounds [3 x i32], [3 x i32]* %tmp173, i64 0, i64 0
  %tmp175 = load i32, i32* %tmp174, align 4, !tbaa !5
  %tmp176 = sub nsw i32 %tmp171, %tmp175
  %tmp177 = add nsw i32 %tmp176, 1
  %tmp178 = icmp slt i32 0, %tmp177
  br i1 %tmp178, label %bb179, label %bb180

bb179:                                            ; preds = %bb167
  br label %bb180

bb180:                                            ; preds = %bb179, %bb167
  %tmp181 = load %struct.barney*, %struct.barney** %tmp53, align 8, !tbaa !1
  %tmp182 = getelementptr inbounds %struct.barney, %struct.barney* %tmp181, i32 0, i32 1
  %tmp183 = getelementptr inbounds [3 x i32], [3 x i32]* %tmp182, i64 0, i64 1
  %tmp184 = load i32, i32* %tmp183, align 4, !tbaa !5
  %tmp185 = load %struct.barney*, %struct.barney** %tmp53, align 8, !tbaa !1
  %tmp186 = getelementptr inbounds %struct.barney, %struct.barney* %tmp185, i32 0, i32 0
  %tmp187 = getelementptr inbounds [3 x i32], [3 x i32]* %tmp186, i64 0, i64 1
  %tmp188 = load i32, i32* %tmp187, align 4, !tbaa !5
  %tmp189 = sub nsw i32 %tmp184, %tmp188
  %tmp190 = add nsw i32 %tmp189, 1
  %tmp191 = icmp slt i32 0, %tmp190
  br i1 %tmp191, label %bb192, label %bb193

bb192:                                            ; preds = %bb180
  br label %bb193

bb193:                                            ; preds = %bb192, %bb180
  %tmp194 = load %struct.barney*, %struct.barney** %tmp54, align 8, !tbaa !1
  %tmp195 = getelementptr inbounds %struct.barney, %struct.barney* %tmp194, i32 0, i32 1
  %tmp196 = getelementptr inbounds [3 x i32], [3 x i32]* %tmp195, i64 0, i64 0
  %tmp197 = load i32, i32* %tmp196, align 4, !tbaa !5
  %tmp198 = load %struct.barney*, %struct.barney** %tmp54, align 8, !tbaa !1
  %tmp199 = getelementptr inbounds %struct.barney, %struct.barney* %tmp198, i32 0, i32 0
  %tmp200 = getelementptr inbounds [3 x i32], [3 x i32]* %tmp199, i64 0, i64 0
  %tmp201 = load i32, i32* %tmp200, align 4, !tbaa !5
  %tmp202 = sub nsw i32 %tmp197, %tmp201
  %tmp203 = add nsw i32 %tmp202, 1
  %tmp204 = icmp slt i32 0, %tmp203
  br i1 %tmp204, label %bb205, label %bb206

bb205:                                            ; preds = %bb193
  br label %bb206

bb206:                                            ; preds = %bb205, %bb193
  %tmp207 = load %struct.barney*, %struct.barney** %tmp54, align 8, !tbaa !1
  %tmp208 = getelementptr inbounds %struct.barney, %struct.barney* %tmp207, i32 0, i32 1
  %tmp209 = getelementptr inbounds [3 x i32], [3 x i32]* %tmp208, i64 0, i64 0
  %tmp210 = load i32, i32* %tmp209, align 4, !tbaa !5
  %tmp211 = load %struct.barney*, %struct.barney** %tmp54, align 8, !tbaa !1
  %tmp212 = getelementptr inbounds %struct.barney, %struct.barney* %tmp211, i32 0, i32 0
  %tmp213 = getelementptr inbounds [3 x i32], [3 x i32]* %tmp212, i64 0, i64 0
  %tmp214 = load i32, i32* %tmp213, align 4, !tbaa !5
  %tmp215 = sub nsw i32 %tmp210, %tmp214
  %tmp216 = add nsw i32 %tmp215, 1
  %tmp217 = icmp slt i32 0, %tmp216
  br i1 %tmp217, label %bb218, label %bb219

bb218:                                            ; preds = %bb206
  br label %bb219

bb219:                                            ; preds = %bb218, %bb206
  %tmp220 = load %struct.barney*, %struct.barney** %tmp54, align 8, !tbaa !1
  %tmp221 = getelementptr inbounds %struct.barney, %struct.barney* %tmp220, i32 0, i32 1
  %tmp222 = getelementptr inbounds [3 x i32], [3 x i32]* %tmp221, i64 0, i64 1
  %tmp223 = load i32, i32* %tmp222, align 4, !tbaa !5
  %tmp224 = load %struct.barney*, %struct.barney** %tmp54, align 8, !tbaa !1
  %tmp225 = getelementptr inbounds %struct.barney, %struct.barney* %tmp224, i32 0, i32 0
  %tmp226 = getelementptr inbounds [3 x i32], [3 x i32]* %tmp225, i64 0, i64 1
  %tmp227 = load i32, i32* %tmp226, align 4, !tbaa !5
  %tmp228 = sub nsw i32 %tmp223, %tmp227
  %tmp229 = add nsw i32 %tmp228, 1
  %tmp230 = icmp slt i32 0, %tmp229
  br i1 %tmp230, label %bb231, label %bb232

bb231:                                            ; preds = %bb219
  br label %bb232

bb232:                                            ; preds = %bb231, %bb219
  %tmp233 = load %struct.barney*, %struct.barney** %tmp55, align 8, !tbaa !1
  %tmp234 = getelementptr inbounds %struct.barney, %struct.barney* %tmp233, i32 0, i32 1
  %tmp235 = getelementptr inbounds [3 x i32], [3 x i32]* %tmp234, i64 0, i64 0
  %tmp236 = load i32, i32* %tmp235, align 4, !tbaa !5
  %tmp237 = load %struct.barney*, %struct.barney** %tmp55, align 8, !tbaa !1
  %tmp238 = getelementptr inbounds %struct.barney, %struct.barney* %tmp237, i32 0, i32 0
  %tmp239 = getelementptr inbounds [3 x i32], [3 x i32]* %tmp238, i64 0, i64 0
  %tmp240 = load i32, i32* %tmp239, align 4, !tbaa !5
  %tmp241 = sub nsw i32 %tmp236, %tmp240
  %tmp242 = add nsw i32 %tmp241, 1
  %tmp243 = icmp slt i32 0, %tmp242
  br i1 %tmp243, label %bb244, label %bb245

bb244:                                            ; preds = %bb232
  br label %bb245

bb245:                                            ; preds = %bb244, %bb232
  unreachable
}

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}

!0 = !{!"clang version 3.8.0 (trunk 252261) (llvm/trunk 252271)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"any pointer", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !3, i64 0}
