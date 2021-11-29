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
%1 = type <{ [1 x %0]*, i8, i8, i8, i8, i32, i32, i32, [1 x i32], [1 x i32], [1 x i32], [24 x i8] }>
declare i8* @__malloc()
; CHECK-NOT: <256 x i1>
; CHECK-NOT: <512 x i1>
define dso_local void @test([0 x %0]* %arg, i32* %arg1, i32* %arg2, i32* %arg3, i32* %arg4) {
  %i = alloca [0 x %0]*, align 4
  store [0 x %0]* %arg, [0 x %0]** %i, align 4
  %i7 = alloca i32*, align 4
  store i32* %arg1, i32** %i7, align 4
  %i9 = alloca i32*, align 4
  store i32* %arg2, i32** %i9, align 4
  %i10 = alloca i32*, align 4
  store i32* %arg3, i32** %i10, align 4
  %i11 = alloca i32*, align 4
  store i32* %arg4, i32** %i11, align 4
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
  %i26 = load i32*, i32** %i9, align 4
  %i27 = load i32, i32* %i26, align 4
  %i28 = select i1 false, i32 0, i32 %i27
  store i32 %i28, i32* %i15, align 4
  %i29 = load i32*, i32** %i7, align 4
  %i30 = load i32, i32* %i29, align 4
  %i31 = select i1 false, i32 0, i32 %i30
  store i32 %i31, i32* %i16, align 4
  %i32 = load i32, i32* %i15, align 4
  %i33 = mul i32 8, %i32
  store i32 %i33, i32* %i17, align 4
  %i34 = load i32, i32* %i17, align 4
  %i35 = load i32, i32* %i16, align 4
  %i36 = mul i32 %i34, %i35
  store i32 %i36, i32* %i18, align 4
  %i37 = load i32*, i32** %i9, align 4
  %i38 = load i32, i32* %i37, align 4
  %i39 = select i1 false, i32 0, i32 %i38
  store i32 %i39, i32* %i22, align 4
  %i40 = load i32*, i32** %i10, align 4
  %i41 = load i32, i32* %i40, align 4
  %i42 = select i1 false, i32 0, i32 %i41
  store i32 %i42, i32* %i23, align 4
  %i43 = getelementptr inbounds %1, %1* %i14, i32 0, i32 10
  %i44 = bitcast [1 x i32]* %i43 to i8*
  %i45 = getelementptr i8, i8* %i44, i32 -12
  %i46 = getelementptr inbounds i8, i8* %i45, i32 12
  %i47 = bitcast i8* %i46 to i32*
  %i48 = load i32, i32* %i23, align 4
  %i49 = select i1 false, i32 0, i32 %i48
  %i50 = load i32, i32* %i22, align 4
  %i51 = select i1 false, i32 0, i32 %i50
  %i52 = mul i32 %i51, 8
  %i53 = mul i32 %i49, %i52
  store i32 %i53, i32* %i47, align 4
  %i54 = getelementptr inbounds %1, %1* %i14, i32 0, i32 10
  %i55 = bitcast [1 x i32]* %i54 to i8*
  %i56 = getelementptr i8, i8* %i55, i32 -12
  %i57 = getelementptr inbounds i8, i8* %i56, i32 36
  %i58 = bitcast i8* %i57 to i32*
  store i32 8, i32* %i58, align 4
  %i60 = getelementptr inbounds %1, %1* %i14, i32 0, i32 0
  %i61 = call i8* @__malloc()
  %i62 = bitcast [1 x %0]** %i60 to i8**
  store i8* %i61, i8** %i62, align 4
  br label %bb63
bb63:                                             ; preds = %bb66, %bb
  %i64 = load i32*, i32** %i11, align 4
  %i65 = load i32, i32* %i64, align 4
  br label %bb66
bb66:                                             ; preds = %bb165, %bb63
  %i67 = load i32, i32* %i21, align 4
  %i68 = icmp sle i32 %i67, %i65
  br i1 %i68, label %bb69, label %bb63
bb69:                                             ; preds = %bb66
  store i32 1, i32* %i20, align 4
  br label %bb70
bb70:                                             ; preds = %bb163, %bb69
  %i71 = load i32, i32* %i20, align 4
  %i72 = icmp sle i32 %i71, 11
  br i1 %i72, label %bb73, label %bb165
bb73:                                             ; preds = %bb70
  %i74 = load i32, i32* %i21, align 4
  %i76 = mul i32 %i74, 8
  %i77 = getelementptr inbounds i8, i8* null, i32 %i76
  %i78 = bitcast i8* %i77 to double*
  %i79 = load double, double* %i78, align 8
  %i80 = fcmp fast olt double %i79, 0.000000e+00
  %i81 = zext i1 %i80 to i32
  %i82 = trunc i32 %i81 to i1
  br i1 %i82, label %bb83, label %bb102
bb83:                                             ; preds = %bb73
  %i84 = getelementptr inbounds %1, %1* %i14, i32 0, i32 0
  %i85 = load [1 x %0]*, [1 x %0]** %i84, align 4
  %i86 = bitcast [1 x %0]* %i85 to i8*
  %i87 = getelementptr i8, i8* %i86, i32 0
  %i88 = load i32, i32* %i20, align 4
  %i89 = getelementptr inbounds %1, %1* %i14, i32 0, i32 10
  %i90 = getelementptr inbounds [1 x i32], [1 x i32]* %i89, i32 0, i32 0
  %i91 = load i32, i32* %i90, align 4
  %i92 = mul i32 %i88, %i91
  %i93 = getelementptr inbounds i8, i8* %i87, i32 %i92
  %i94 = getelementptr inbounds i8, i8* %i93, i32 0
  %i95 = load i32, i32* %i21, align 4
  %i96 = getelementptr inbounds %1, %1* %i14, i32 0, i32 10
  %i97 = getelementptr inbounds [1 x i32], [1 x i32]* %i96, i32 0, i32 6
  %i98 = load i32, i32* %i97, align 4
  %i99 = mul i32 %i95, %i98
  %i100 = getelementptr inbounds i8, i8* %i94, i32 %i99
  %i101 = bitcast i8* %i100 to double*
  store double 0.000000e+00, double* %i101, align 8
  br label %bb163
bb102:                                            ; preds = %bb73
  %i103 = getelementptr i8, i8* null, i32 -8
  %i104 = getelementptr inbounds i8, i8* %i103, i32 undef
  %i105 = bitcast i8* %i104 to double*
  %i106 = load double, double* %i105, align 8
  %i107 = load [0 x %0]*, [0 x %0]** %i, align 4
  %i108 = bitcast [0 x %0]* %i107 to i8*
  %i109 = getelementptr i8, i8* %i108, i32 -8
  %i110 = getelementptr inbounds i8, i8* %i109, i32 undef
  %i111 = bitcast i8* %i110 to double*
  %i112 = load double, double* %i111, align 8
  %i113 = fmul fast double %i106, %i112
  %i114 = fcmp fast ogt double 0.000000e+00, %i113
  %i115 = zext i1 %i114 to i32
  %i116 = trunc i32 %i115 to i1
  br i1 %i116, label %bb117, label %bb136
bb117:                                            ; preds = %bb102
  %i118 = getelementptr inbounds %1, %1* %i14, i32 0, i32 0
  %i119 = load [1 x %0]*, [1 x %0]** %i118, align 4
  %i120 = bitcast [1 x %0]* %i119 to i8*
  %i121 = getelementptr i8, i8* %i120, i32 0
  %i122 = load i32, i32* %i20, align 4
  %i123 = getelementptr inbounds %1, %1* %i14, i32 0, i32 10
  %i124 = getelementptr inbounds [1 x i32], [1 x i32]* %i123, i32 0, i32 0
  %i125 = load i32, i32* %i124, align 4
  %i126 = mul i32 %i122, %i125
  %i127 = getelementptr inbounds i8, i8* %i121, i32 %i126
  %i128 = getelementptr inbounds i8, i8* %i127, i32 0
  %i129 = load i32, i32* %i21, align 4
  %i130 = getelementptr inbounds %1, %1* %i14, i32 0, i32 10
  %i131 = getelementptr inbounds [1 x i32], [1 x i32]* %i130, i32 0, i32 6
  %i132 = load i32, i32* %i131, align 4
  %i133 = mul i32 %i129, %i132
  %i134 = getelementptr inbounds i8, i8* %i128, i32 %i133
  %i135 = bitcast i8* %i134 to double*
  store double 0.000000e+00, double* %i135, align 8
  br label %bb163
bb136:                                            ; preds = %bb102
  %i137 = load double, double* null, align 8
  %i138 = load double, double* null, align 8
  %i139 = fmul fast double %i137, %i138
  %i140 = fsub fast double 0.000000e+00, %i139
  store double %i140, double* %i25, align 8
  %i141 = load i32, i32* %i21, align 4
  %i143 = getelementptr inbounds [1 x i32], [1 x i32]* null, i32 0, i32 6
  %i144 = load i32, i32* %i143, align 4
  %i145 = mul i32 %i141, %i144
  %i146 = getelementptr inbounds i8, i8* null, i32 %i145
  %i147 = bitcast i8* %i146 to double*
  %i148 = load i32, i32* %i20, align 4
  %i149 = load i32, i32* %i18, align 4
  %i151 = mul i32 %i148, %i149
  %i152 = getelementptr i8, i8* null, i32 %i151
  %i153 = getelementptr i8, i8* %i152, i32 0
  %i154 = getelementptr inbounds i8, i8* %i153, i32 0
  %i155 = bitcast i8* %i154 to double*
  %i156 = load double, double* %i155, align 8
  %i157 = load double, double* %i25, align 8
  %i158 = fmul fast double %i156, %i157
  %i159 = fadd fast double 0.000000e+00, %i158
  %i160 = load double, double* %i25, align 8
  %i161 = fadd fast double 0.000000e+00, %i160
  %i162 = fdiv fast double %i159, %i161
  store double %i162, double* %i147, align 8
  br label %bb163
bb163:                                            ; preds = %bb136, %bb117, %bb83
  %i164 = add nsw i32 %i71, 1
  store i32 %i164, i32* %i20, align 4
  br label %bb70
bb165:                                            ; preds = %bb70
  %i166 = add nsw i32 %i67, 1
  store i32 %i166, i32* %i21, align 4
  br label %bb66
}
