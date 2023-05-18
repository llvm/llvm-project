; RUN: opt --vec-extabi=true -passes='default<O3>' -mcpu=pwr10 \
; RUN:   -pgo-kind=pgo-instr-gen-pipeline -mtriple=powerpc-ibm-aix -S < %s | \
; RUN: FileCheck %s
; RUN: opt -passes='default<O3>' -mcpu=pwr10 -pgo-kind=pgo-instr-gen-pipeline \
; RUN:   -mtriple=powerpc64le-unknown-linux-gnu -S < %s | FileCheck %s

; When running this test case under opt + PGO, the SLPVectorizer previously had
; an opportunity to produce wide vector types (such as <256 x i1>) within the
; IR as it deemed these wide vector types to be cheap enough to produce.
; Having this test ensures that the optimizer no longer generates wide vectors
; within the IR.

%0 = type <{ double }>
%1 = type <{ ptr, i8, i8, i8, i8, i32, i32, i32, [1 x i32], [1 x i32], [1 x i32], [24 x i8] }>
declare ptr @__malloc()
; CHECK-NOT: <256 x i1>
; CHECK-NOT: <512 x i1>
define dso_local void @test(ptr %arg, ptr %arg1, ptr %arg2, ptr %arg3, ptr %arg4) {
  %i = alloca ptr, align 4
  store ptr %arg, ptr %i, align 4
  %i7 = alloca ptr, align 4
  store ptr %arg1, ptr %i7, align 4
  %i9 = alloca ptr, align 4
  store ptr %arg2, ptr %i9, align 4
  %i10 = alloca ptr, align 4
  store ptr %arg3, ptr %i10, align 4
  %i11 = alloca ptr, align 4
  store ptr %arg4, ptr %i11, align 4
  %i14 = alloca %1, align 4
  %i15 = alloca i32, align 4
  %i16 = alloca i32, align 4
  %i17 = alloca i32, align 4
  %i18 = alloca i32, align 4
  %i20 = alloca i32, align 4
  %i21 = alloca i32, align 4
  %i22 = alloca i32, align 4
  %i23 = alloca i32, align 4
  %i25 = alloca double, align 8
  %i26 = load ptr, ptr %i9, align 4
  %i27 = load i32, ptr %i26, align 4
  %i28 = select i1 false, i32 0, i32 %i27
  store i32 %i28, ptr %i15, align 4
  %i29 = load ptr, ptr %i7, align 4
  %i30 = load i32, ptr %i29, align 4
  %i31 = select i1 false, i32 0, i32 %i30
  store i32 %i31, ptr %i16, align 4
  %i32 = load i32, ptr %i15, align 4
  %i33 = mul i32 8, %i32
  store i32 %i33, ptr %i17, align 4
  %i34 = load i32, ptr %i17, align 4
  %i35 = load i32, ptr %i16, align 4
  %i36 = mul i32 %i34, %i35
  store i32 %i36, ptr %i18, align 4
  %i37 = load ptr, ptr %i9, align 4
  %i38 = load i32, ptr %i37, align 4
  %i39 = select i1 false, i32 0, i32 %i38
  store i32 %i39, ptr %i22, align 4
  %i40 = load ptr, ptr %i10, align 4
  %i41 = load i32, ptr %i40, align 4
  %i42 = select i1 false, i32 0, i32 %i41
  store i32 %i42, ptr %i23, align 4
  %i43 = getelementptr inbounds %1, ptr %i14, i32 0, i32 10
  %i45 = getelementptr i8, ptr %i43, i32 -12
  %i46 = getelementptr inbounds i8, ptr %i45, i32 12
  %i48 = load i32, ptr %i23, align 4
  %i49 = select i1 false, i32 0, i32 %i48
  %i50 = load i32, ptr %i22, align 4
  %i51 = select i1 false, i32 0, i32 %i50
  %i52 = mul i32 %i51, 8
  %i53 = mul i32 %i49, %i52
  store i32 %i53, ptr %i46, align 4
  %i54 = getelementptr inbounds %1, ptr %i14, i32 0, i32 10
  %i56 = getelementptr i8, ptr %i54, i32 -12
  %i57 = getelementptr inbounds i8, ptr %i56, i32 36
  store i32 8, ptr %i57, align 4
  %i61 = call ptr @__malloc()
  store ptr %i61, ptr %i14, align 4
  br label %bb63
bb63:                                             ; preds = %bb66, %bb
  %i64 = load ptr, ptr %i11, align 4
  %i65 = load i32, ptr %i64, align 4
  br label %bb66
bb66:                                             ; preds = %bb165, %bb63
  %i67 = load i32, ptr %i21, align 4
  %i68 = icmp sle i32 %i67, %i65
  br i1 %i68, label %bb69, label %bb63
bb69:                                             ; preds = %bb66
  store i32 1, ptr %i20, align 4
  br label %bb70
bb70:                                             ; preds = %bb163, %bb69
  %i71 = load i32, ptr %i20, align 4
  %i72 = icmp sle i32 %i71, 11
  br i1 %i72, label %bb73, label %bb165
bb73:                                             ; preds = %bb70
  %i74 = load i32, ptr %i21, align 4
  %i76 = mul i32 %i74, 8
  %i77 = getelementptr inbounds i8, ptr null, i32 %i76
  %i79 = load double, ptr %i77, align 8
  %i80 = fcmp fast olt double %i79, 0.000000e+00
  %i81 = zext i1 %i80 to i32
  %i82 = trunc i32 %i81 to i1
  br i1 %i82, label %bb83, label %bb102
bb83:                                             ; preds = %bb73
  %i85 = load ptr, ptr %i14, align 4
  %i88 = load i32, ptr %i20, align 4
  %i89 = getelementptr inbounds %1, ptr %i14, i32 0, i32 10
  %i91 = load i32, ptr %i89, align 4
  %i92 = mul i32 %i88, %i91
  %i93 = getelementptr inbounds i8, ptr %i85, i32 %i92
  %i95 = load i32, ptr %i21, align 4
  %i96 = getelementptr inbounds %1, ptr %i14, i32 0, i32 10
  %i97 = getelementptr inbounds [1 x i32], ptr %i96, i32 0, i32 6
  %i98 = load i32, ptr %i97, align 4
  %i99 = mul i32 %i95, %i98
  %i100 = getelementptr inbounds i8, ptr %i93, i32 %i99
  store double 0.000000e+00, ptr %i100, align 8
  br label %bb163
bb102:                                            ; preds = %bb73
  %i103 = getelementptr i8, ptr null, i32 -8
  %i104 = getelementptr inbounds i8, ptr %i103, i32 undef
  %i106 = load double, ptr %i104, align 8
  %i107 = load ptr, ptr %i, align 4
  %i109 = getelementptr i8, ptr %i107, i32 -8
  %i110 = getelementptr inbounds i8, ptr %i109, i32 undef
  %i112 = load double, ptr %i110, align 8
  %i113 = fmul fast double %i106, %i112
  %i114 = fcmp fast ogt double 0.000000e+00, %i113
  %i115 = zext i1 %i114 to i32
  %i116 = trunc i32 %i115 to i1
  br i1 %i116, label %bb117, label %bb136
bb117:                                            ; preds = %bb102
  %i119 = load ptr, ptr %i14, align 4
  %i122 = load i32, ptr %i20, align 4
  %i123 = getelementptr inbounds %1, ptr %i14, i32 0, i32 10
  %i125 = load i32, ptr %i123, align 4
  %i126 = mul i32 %i122, %i125
  %i127 = getelementptr inbounds i8, ptr %i119, i32 %i126
  %i129 = load i32, ptr %i21, align 4
  %i130 = getelementptr inbounds %1, ptr %i14, i32 0, i32 10
  %i131 = getelementptr inbounds [1 x i32], ptr %i130, i32 0, i32 6
  %i132 = load i32, ptr %i131, align 4
  %i133 = mul i32 %i129, %i132
  %i134 = getelementptr inbounds i8, ptr %i127, i32 %i133
  store double 0.000000e+00, ptr %i134, align 8
  br label %bb163
bb136:                                            ; preds = %bb102
  %i137 = load double, ptr null, align 8
  %i138 = load double, ptr null, align 8
  %i139 = fmul fast double %i137, %i138
  %i140 = fsub fast double 0.000000e+00, %i139
  store double %i140, ptr %i25, align 8
  %i141 = load i32, ptr %i21, align 4
  %i143 = getelementptr inbounds [1 x i32], ptr null, i32 0, i32 6
  %i144 = load i32, ptr %i143, align 4
  %i145 = mul i32 %i141, %i144
  %i146 = getelementptr inbounds i8, ptr null, i32 %i145
  %i148 = load i32, ptr %i20, align 4
  %i149 = load i32, ptr %i18, align 4
  %i151 = mul i32 %i148, %i149
  %i152 = getelementptr i8, ptr null, i32 %i151
  %i156 = load double, ptr %i152, align 8
  %i157 = load double, ptr %i25, align 8
  %i158 = fmul fast double %i156, %i157
  %i159 = fadd fast double 0.000000e+00, %i158
  %i160 = load double, ptr %i25, align 8
  %i161 = fadd fast double 0.000000e+00, %i160
  %i162 = fdiv fast double %i159, %i161
  store double %i162, ptr %i146, align 8
  br label %bb163
bb163:                                            ; preds = %bb136, %bb117, %bb83
  %i164 = add nsw i32 %i71, 1
  store i32 %i164, ptr %i20, align 4
  br label %bb70
bb165:                                            ; preds = %bb70
  %i166 = add nsw i32 %i67, 1
  store i32 %i166, ptr %i21, align 4
  br label %bb66
}
