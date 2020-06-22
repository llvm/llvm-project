; RUN: opt -lower-matrix-intrinsics -fuse-matrix-tile-size=2 -matrix-allow-contract -force-fuse-matrix -instcombine -verify-dom-info %s -S | FileCheck %s

; REQUIRES: aarch64-registered-target

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "aarch64-apple-ios"

define void @multiply(<16 x double> * %A, <16 x double> * %B, <16 x double>* %C) {
; CHECK-LABEL: @multiply(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[ST_B:%.*]] = ptrtoint <16 x double>* [[C:%.*]] to i64
; CHECK-NEXT:    [[ST_E:%.*]] = add nuw nsw i64 [[ST_B]], 128
; CHECK-NEXT:    [[LD_B:%.*]] = ptrtoint <16 x double>* [[A:%.*]] to i64
; CHECK-NEXT:    [[TMP0:%.*]] = icmp ugt i64 [[ST_E]], [[LD_B]]
; CHECK-NEXT:    br i1 [[TMP0]], label [[ALIAS_CONT:%.*]], label [[NO_ALIAS:%.*]]
; CHECK:       alias_cont:
; CHECK-NEXT:    [[LD_E:%.*]] = add nuw nsw i64 [[LD_B]], 128
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ugt i64 [[LD_E]], [[ST_B]]
; CHECK-NEXT:    br i1 [[TMP1]], label [[COPY:%.*]], label [[NO_ALIAS]]
; CHECK:       copy:
; CHECK-NEXT:    [[TMP2:%.*]] = alloca <16 x double>, align 128
; CHECK-NEXT:    [[TMP3:%.*]] = bitcast <16 x double>* [[TMP2]] to i8*
; CHECK-NEXT:    [[TMP4:%.*]] = bitcast <16 x double>* [[A]] to i8*
; CHECK-NEXT:    call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 128 dereferenceable(128) [[TMP3]], i8* nonnull align 8 dereferenceable(128) [[TMP4]], i64 128, i1 false)
; CHECK-NEXT:    br label [[NO_ALIAS]]
; CHECK:       no_alias:
; CHECK-NEXT:    [[TMP5:%.*]] = phi <16 x double>* [ [[A]], [[ENTRY:%.*]] ], [ [[A]], [[ALIAS_CONT]] ], [ [[TMP2]], [[COPY]] ]
; CHECK-NEXT:    [[ST_B1:%.*]] = ptrtoint <16 x double>* [[C]] to i64
; CHECK-NEXT:    [[ST_E2:%.*]] = add nuw nsw i64 [[ST_B1]], 128
; CHECK-NEXT:    [[LD_B6:%.*]] = ptrtoint <16 x double>* [[B:%.*]] to i64
; CHECK-NEXT:    [[TMP6:%.*]] = icmp ugt i64 [[ST_E2]], [[LD_B6]]
; CHECK-NEXT:    br i1 [[TMP6]], label [[ALIAS_CONT3:%.*]], label [[NO_ALIAS5:%.*]]
; CHECK:       alias_cont1:
; CHECK-NEXT:    [[LD_E7:%.*]] = add nuw nsw i64 [[LD_B6]], 128
; CHECK-NEXT:    [[TMP7:%.*]] = icmp ugt i64 [[LD_E7]], [[ST_B1]]
; CHECK-NEXT:    br i1 [[TMP7]], label [[COPY4:%.*]], label [[NO_ALIAS5]]
; CHECK:       copy2:
; CHECK-NEXT:    [[TMP8:%.*]] = alloca <16 x double>, align 128
; CHECK-NEXT:    [[TMP9:%.*]] = bitcast <16 x double>* [[TMP8]] to i8*
; CHECK-NEXT:    [[TMP10:%.*]] = bitcast <16 x double>* [[B]] to i8*
; CHECK-NEXT:    call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 128 dereferenceable(128) [[TMP9]], i8* nonnull align 8 dereferenceable(128) [[TMP10]], i64 128, i1 false)
; CHECK-NEXT:    br label [[NO_ALIAS5]]

; CHECK:       no_alias3:
; CHECK-NEXT:    [[TMP11:%.*]] = phi <16 x double>* [ [[B]], [[NO_ALIAS]] ], [ [[B]], [[ALIAS_CONT3]] ], [ [[TMP8]], [[COPY4]] ]

;; np.dot(a[0:2, 0:2], b[0:2, 0:2])

; CHECK-NEXT:    [[COL_CAST8:%.*]] = bitcast <16 x double>* [[TMP5]] to <2 x double>*
; CHECK-NEXT:    [[COL_LOAD:%.*]] = load <2 x double>, <2 x double>* [[COL_CAST8]], align 8
; CHECK-NEXT:    [[COL_GEP:%.*]] = getelementptr <16 x double>, <16 x double>* [[TMP5]], i64 0, i64 4
; CHECK-NEXT:    [[COL_CAST9:%.*]] = bitcast double* [[COL_GEP]] to <2 x double>*
; CHECK-NEXT:    [[COL_LOAD10:%.*]] = load <2 x double>, <2 x double>* [[COL_CAST9]], align 8
; CHECK-NEXT:    [[COL_CAST12:%.*]] = bitcast <16 x double>* [[TMP11]] to <2 x double>*
; CHECK-NEXT:    [[COL_LOAD13:%.*]] = load <2 x double>, <2 x double>* [[COL_CAST12]], align 8
; CHECK-NEXT:    [[COL_GEP14:%.*]] = getelementptr <16 x double>, <16 x double>* [[TMP11]], i64 0, i64 4
; CHECK-NEXT:    [[COL_CAST15:%.*]] = bitcast double* [[COL_GEP14]] to <2 x double>*
; CHECK-NEXT:    [[COL_LOAD16:%.*]] = load <2 x double>, <2 x double>* [[COL_CAST15]], align 8
; CHECK-NEXT:    [[SPLAT_SPLAT:%.*]] = shufflevector <2 x double> [[COL_LOAD13]], <2 x double> undef, <2 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP12:%.*]] = fmul <2 x double> [[COL_LOAD]], [[SPLAT_SPLAT]]
; CHECK-NEXT:    [[SPLAT_SPLAT19:%.*]] = shufflevector <2 x double> [[COL_LOAD13]], <2 x double> undef, <2 x i32> <i32 1, i32 1>
; CHECK-NEXT:    [[TMP13:%.*]] = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> [[COL_LOAD10]], <2 x double> [[SPLAT_SPLAT19]], <2 x double> [[TMP12]])
; CHECK-NEXT:    [[SPLAT_SPLAT22:%.*]] = shufflevector <2 x double> [[COL_LOAD16]], <2 x double> undef, <2 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP14:%.*]] = fmul <2 x double> [[COL_LOAD]], [[SPLAT_SPLAT22]]
; CHECK-NEXT:    [[SPLAT_SPLAT25:%.*]] = shufflevector <2 x double> [[COL_LOAD16]], <2 x double> undef, <2 x i32> <i32 1, i32 1>
; CHECK-NEXT:    [[TMP15:%.*]] = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> [[COL_LOAD10]], <2 x double> [[SPLAT_SPLAT25]], <2 x double> [[TMP14]])

;; + np.dot(a[0:2, 2:4], b[2:4, 0:2])

; CHECK-NEXT:    [[TMP16:%.*]] = getelementptr <16 x double>, <16 x double>* [[TMP5]], i64 0, i64 8
; CHECK-NEXT:    [[COL_CAST27:%.*]] = bitcast double* [[TMP16]] to <2 x double>*
; CHECK-NEXT:    [[COL_LOAD28:%.*]] = load <2 x double>, <2 x double>* [[COL_CAST27]], align 8
; CHECK-NEXT:    [[COL_GEP29:%.*]] = getelementptr <16 x double>, <16 x double>* [[TMP5]], i64 0, i64 12
; CHECK-NEXT:    [[COL_CAST30:%.*]] = bitcast double* [[COL_GEP29]] to <2 x double>*
; CHECK-NEXT:    [[COL_LOAD31:%.*]] = load <2 x double>, <2 x double>* [[COL_CAST30]], align 8
; CHECK-NEXT:    [[TMP17:%.*]] = getelementptr <16 x double>, <16 x double>* [[TMP11]], i64 0, i64 2
; CHECK-NEXT:    [[COL_CAST33:%.*]] = bitcast double* [[TMP17]] to <2 x double>*
; CHECK-NEXT:    [[COL_LOAD34:%.*]] = load <2 x double>, <2 x double>* [[COL_CAST33]], align 8
; CHECK-NEXT:    [[COL_GEP35:%.*]] = getelementptr <16 x double>, <16 x double>* [[TMP11]], i64 0, i64 6
; CHECK-NEXT:    [[COL_CAST36:%.*]] = bitcast double* [[COL_GEP35]] to <2 x double>*
; CHECK-NEXT:    [[COL_LOAD37:%.*]] = load <2 x double>, <2 x double>* [[COL_CAST36]], align 8
; CHECK-NEXT:    [[SPLAT_SPLAT41:%.*]] = shufflevector <2 x double> [[COL_LOAD34]], <2 x double> undef, <2 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP18:%.*]] = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> [[COL_LOAD28]], <2 x double> [[SPLAT_SPLAT41]], <2 x double> [[TMP13]])
; CHECK-NEXT:    [[SPLAT_SPLAT44:%.*]] = shufflevector <2 x double> [[COL_LOAD34]], <2 x double> undef, <2 x i32> <i32 1, i32 1>
; CHECK-NEXT:    [[TMP19:%.*]] = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> [[COL_LOAD31]], <2 x double> [[SPLAT_SPLAT44]], <2 x double> [[TMP18]])
; CHECK-NEXT:    [[SPLAT_SPLAT48:%.*]] = shufflevector <2 x double> [[COL_LOAD37]], <2 x double> undef, <2 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP20:%.*]] = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> [[COL_LOAD28]], <2 x double> [[SPLAT_SPLAT48]], <2 x double> [[TMP15]])
; CHECK-NEXT:    [[SPLAT_SPLAT51:%.*]] = shufflevector <2 x double> [[COL_LOAD37]], <2 x double> undef, <2 x i32> <i32 1, i32 1>
; CHECK-NEXT:    [[TMP21:%.*]] = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> [[COL_LOAD31]], <2 x double> [[SPLAT_SPLAT51]], <2 x double> [[TMP20]])

;; -> c[0:2, 0:2]

; CHECK-NEXT:    [[COL_CAST53:%.*]] = bitcast <16 x double>* [[C]] to <2 x double>*
; CHECK-NEXT:    store <2 x double> [[TMP19]], <2 x double>* [[COL_CAST53]], align 8
; CHECK-NEXT:    [[COL_GEP54:%.*]] = getelementptr <16 x double>, <16 x double>* [[C]], i64 0, i64 4
; CHECK-NEXT:    [[COL_CAST55:%.*]] = bitcast double* [[COL_GEP54]] to <2 x double>*
; CHECK-NEXT:    store <2 x double> [[TMP21]], <2 x double>* [[COL_CAST55]], align 8

;; np.dot(a[2:4, 0:2], b[0:2, 0:2])

; CHECK-NEXT:    [[TMP22:%.*]] = getelementptr <16 x double>, <16 x double>* [[TMP5]], i64 0, i64 2
; CHECK-NEXT:    [[COL_CAST57:%.*]] = bitcast double* [[TMP22]] to <2 x double>*
; CHECK-NEXT:    [[COL_LOAD58:%.*]] = load <2 x double>, <2 x double>* [[COL_CAST57]], align 8
; CHECK-NEXT:    [[COL_GEP59:%.*]] = getelementptr <16 x double>, <16 x double>* [[TMP5]], i64 0, i64 6
; CHECK-NEXT:    [[COL_CAST60:%.*]] = bitcast double* [[COL_GEP59]] to <2 x double>*
; CHECK-NEXT:    [[COL_LOAD61:%.*]] = load <2 x double>, <2 x double>* [[COL_CAST60]], align 8
; CHECK-NEXT:    [[COL_CAST63:%.*]] = bitcast <16 x double>* [[TMP11]] to <2 x double>*
; CHECK-NEXT:    [[COL_LOAD64:%.*]] = load <2 x double>, <2 x double>* [[COL_CAST63]], align 8
; CHECK-NEXT:    [[COL_GEP65:%.*]] = getelementptr <16 x double>, <16 x double>* [[TMP11]], i64 0, i64 4
; CHECK-NEXT:    [[COL_CAST66:%.*]] = bitcast double* [[COL_GEP65]] to <2 x double>*
; CHECK-NEXT:    [[COL_LOAD67:%.*]] = load <2 x double>, <2 x double>* [[COL_CAST66]], align 8
; CHECK-NEXT:    [[SPLAT_SPLAT70:%.*]] = shufflevector <2 x double> [[COL_LOAD64]], <2 x double> undef, <2 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP23:%.*]] = fmul <2 x double> [[COL_LOAD58]], [[SPLAT_SPLAT70]]
; CHECK-NEXT:    [[SPLAT_SPLAT73:%.*]] = shufflevector <2 x double> [[COL_LOAD64]], <2 x double> undef, <2 x i32> <i32 1, i32 1>
; CHECK-NEXT:    [[TMP24:%.*]] = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> [[COL_LOAD61]], <2 x double> [[SPLAT_SPLAT73]], <2 x double> [[TMP23]])
; CHECK-NEXT:    [[SPLAT_SPLAT76:%.*]] = shufflevector <2 x double> [[COL_LOAD67]], <2 x double> undef, <2 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP25:%.*]] = fmul <2 x double> [[COL_LOAD58]], [[SPLAT_SPLAT76]]
; CHECK-NEXT:    [[SPLAT_SPLAT79:%.*]] = shufflevector <2 x double> [[COL_LOAD67]], <2 x double> undef, <2 x i32> <i32 1, i32 1>
; CHECK-NEXT:    [[TMP26:%.*]] = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> [[COL_LOAD61]], <2 x double> [[SPLAT_SPLAT79]], <2 x double> [[TMP25]])

;; + np.dot(a[2:4, 2:4], b[2:4, 0:2])

; CHECK-NEXT:    [[TMP27:%.*]] = getelementptr <16 x double>, <16 x double>* [[TMP5]], i64 0, i64 10
; CHECK-NEXT:    [[COL_CAST81:%.*]] = bitcast double* [[TMP27]] to <2 x double>*
; CHECK-NEXT:    [[COL_LOAD82:%.*]] = load <2 x double>, <2 x double>* [[COL_CAST81]], align 8
; CHECK-NEXT:    [[COL_GEP83:%.*]] = getelementptr <16 x double>, <16 x double>* [[TMP5]], i64 0, i64 14
; CHECK-NEXT:    [[COL_CAST84:%.*]] = bitcast double* [[COL_GEP83]] to <2 x double>*
; CHECK-NEXT:    [[COL_LOAD85:%.*]] = load <2 x double>, <2 x double>* [[COL_CAST84]], align 8
; CHECK-NEXT:    [[TMP28:%.*]] = getelementptr <16 x double>, <16 x double>* [[TMP11]], i64 0, i64 2
; CHECK-NEXT:    [[COL_CAST87:%.*]] = bitcast double* [[TMP28]] to <2 x double>*
; CHECK-NEXT:    [[COL_LOAD88:%.*]] = load <2 x double>, <2 x double>* [[COL_CAST87]], align 8
; CHECK-NEXT:    [[COL_GEP89:%.*]] = getelementptr <16 x double>, <16 x double>* [[TMP11]], i64 0, i64 6
; CHECK-NEXT:    [[COL_CAST90:%.*]] = bitcast double* [[COL_GEP89]] to <2 x double>*
; CHECK-NEXT:    [[COL_LOAD91:%.*]] = load <2 x double>, <2 x double>* [[COL_CAST90]], align 8
; CHECK-NEXT:    [[SPLAT_SPLAT95:%.*]] = shufflevector <2 x double> [[COL_LOAD88]], <2 x double> undef, <2 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP29:%.*]] = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> [[COL_LOAD82]], <2 x double> [[SPLAT_SPLAT95]], <2 x double> [[TMP24]])
; CHECK-NEXT:    [[SPLAT_SPLAT98:%.*]] = shufflevector <2 x double> [[COL_LOAD88]], <2 x double> undef, <2 x i32> <i32 1, i32 1>
; CHECK-NEXT:    [[TMP30:%.*]] = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> [[COL_LOAD85]], <2 x double> [[SPLAT_SPLAT98]], <2 x double> [[TMP29]])
; CHECK-NEXT:    [[SPLAT_SPLAT102:%.*]] = shufflevector <2 x double> [[COL_LOAD91]], <2 x double> undef, <2 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP31:%.*]] = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> [[COL_LOAD82]], <2 x double> [[SPLAT_SPLAT102]], <2 x double> [[TMP26]])
; CHECK-NEXT:    [[SPLAT_SPLAT105:%.*]] = shufflevector <2 x double> [[COL_LOAD91]], <2 x double> undef, <2 x i32> <i32 1, i32 1>
; CHECK-NEXT:    [[TMP32:%.*]] = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> [[COL_LOAD85]], <2 x double> [[SPLAT_SPLAT105]], <2 x double> [[TMP31]])

;; -> c[2:4, 0:2]

; CHECK-NEXT:    [[TMP33:%.*]] = getelementptr <16 x double>, <16 x double>* [[C]], i64 0, i64 2
; CHECK-NEXT:    [[COL_CAST107:%.*]] = bitcast double* [[TMP33]] to <2 x double>*
; CHECK-NEXT:    store <2 x double> [[TMP30]], <2 x double>* [[COL_CAST107]], align 8
; CHECK-NEXT:    [[COL_GEP108:%.*]] = getelementptr <16 x double>, <16 x double>* [[C]], i64 0, i64 6
; CHECK-NEXT:    [[COL_CAST109:%.*]] = bitcast double* [[COL_GEP108]] to <2 x double>*
; CHECK-NEXT:    store <2 x double> [[TMP32]], <2 x double>* [[COL_CAST109]], align 8

;; np.dot(a[0:2, 0:2], b[0:2, 2:4])

; CHECK-NEXT:    [[COL_CAST111:%.*]] = bitcast <16 x double>* [[TMP5]] to <2 x double>*
; CHECK-NEXT:    [[COL_LOAD112:%.*]] = load <2 x double>, <2 x double>* [[COL_CAST111]], align 8
; CHECK-NEXT:    [[COL_GEP113:%.*]] = getelementptr <16 x double>, <16 x double>* [[TMP5]], i64 0, i64 4
; CHECK-NEXT:    [[COL_CAST114:%.*]] = bitcast double* [[COL_GEP113]] to <2 x double>*
; CHECK-NEXT:    [[COL_LOAD115:%.*]] = load <2 x double>, <2 x double>* [[COL_CAST114]], align 8
; CHECK-NEXT:    [[TMP34:%.*]] = getelementptr <16 x double>, <16 x double>* [[TMP11]], i64 0, i64 8
; CHECK-NEXT:    [[COL_CAST117:%.*]] = bitcast double* [[TMP34]] to <2 x double>*
; CHECK-NEXT:    [[COL_LOAD118:%.*]] = load <2 x double>, <2 x double>* [[COL_CAST117]], align 8
; CHECK-NEXT:    [[COL_GEP119:%.*]] = getelementptr <16 x double>, <16 x double>* [[TMP11]], i64 0, i64 12
; CHECK-NEXT:    [[COL_CAST120:%.*]] = bitcast double* [[COL_GEP119]] to <2 x double>*
; CHECK-NEXT:    [[COL_LOAD121:%.*]] = load <2 x double>, <2 x double>* [[COL_CAST120]], align 8
; CHECK-NEXT:    [[SPLAT_SPLAT124:%.*]] = shufflevector <2 x double> [[COL_LOAD118]], <2 x double> undef, <2 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP35:%.*]] = fmul <2 x double> [[COL_LOAD112]], [[SPLAT_SPLAT124]]
; CHECK-NEXT:    [[SPLAT_SPLAT127:%.*]] = shufflevector <2 x double> [[COL_LOAD118]], <2 x double> undef, <2 x i32> <i32 1, i32 1>
; CHECK-NEXT:    [[TMP36:%.*]] = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> [[COL_LOAD115]], <2 x double> [[SPLAT_SPLAT127]], <2 x double> [[TMP35]])
; CHECK-NEXT:    [[SPLAT_SPLAT130:%.*]] = shufflevector <2 x double> [[COL_LOAD121]], <2 x double> undef, <2 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP37:%.*]] = fmul <2 x double> [[COL_LOAD112]], [[SPLAT_SPLAT130]]
; CHECK-NEXT:    [[SPLAT_SPLAT133:%.*]] = shufflevector <2 x double> [[COL_LOAD121]], <2 x double> undef, <2 x i32> <i32 1, i32 1>
; CHECK-NEXT:    [[TMP38:%.*]] = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> [[COL_LOAD115]], <2 x double> [[SPLAT_SPLAT133]], <2 x double> [[TMP37]])

;; + np.dot(a[0:2, 2:4], b[2:4, 2:4])

; CHECK-NEXT:    [[TMP39:%.*]] = getelementptr <16 x double>, <16 x double>* [[TMP5]], i64 0, i64 8
; CHECK-NEXT:    [[COL_CAST135:%.*]] = bitcast double* [[TMP39]] to <2 x double>*
; CHECK-NEXT:    [[COL_LOAD136:%.*]] = load <2 x double>, <2 x double>* [[COL_CAST135]], align 8
; CHECK-NEXT:    [[COL_GEP137:%.*]] = getelementptr <16 x double>, <16 x double>* [[TMP5]], i64 0, i64 12
; CHECK-NEXT:    [[COL_CAST138:%.*]] = bitcast double* [[COL_GEP137]] to <2 x double>*
; CHECK-NEXT:    [[COL_LOAD139:%.*]] = load <2 x double>, <2 x double>* [[COL_CAST138]], align 8
; CHECK-NEXT:    [[TMP40:%.*]] = getelementptr <16 x double>, <16 x double>* [[TMP11]], i64 0, i64 10
; CHECK-NEXT:    [[COL_CAST141:%.*]] = bitcast double* [[TMP40]] to <2 x double>*
; CHECK-NEXT:    [[COL_LOAD142:%.*]] = load <2 x double>, <2 x double>* [[COL_CAST141]], align 8
; CHECK-NEXT:    [[COL_GEP143:%.*]] = getelementptr <16 x double>, <16 x double>* [[TMP11]], i64 0, i64 14
; CHECK-NEXT:    [[COL_CAST144:%.*]] = bitcast double* [[COL_GEP143]] to <2 x double>*
; CHECK-NEXT:    [[COL_LOAD145:%.*]] = load <2 x double>, <2 x double>* [[COL_CAST144]], align 8
; CHECK-NEXT:    [[SPLAT_SPLAT149:%.*]] = shufflevector <2 x double> [[COL_LOAD142]], <2 x double> undef, <2 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP41:%.*]] = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> [[COL_LOAD136]], <2 x double> [[SPLAT_SPLAT149]], <2 x double> [[TMP36]])
; CHECK-NEXT:    [[SPLAT_SPLAT152:%.*]] = shufflevector <2 x double> [[COL_LOAD142]], <2 x double> undef, <2 x i32> <i32 1, i32 1>
; CHECK-NEXT:    [[TMP42:%.*]] = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> [[COL_LOAD139]], <2 x double> [[SPLAT_SPLAT152]], <2 x double> [[TMP41]])
; CHECK-NEXT:    [[SPLAT_SPLAT156:%.*]] = shufflevector <2 x double> [[COL_LOAD145]], <2 x double> undef, <2 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP43:%.*]] = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> [[COL_LOAD136]], <2 x double> [[SPLAT_SPLAT156]], <2 x double> [[TMP38]])
; CHECK-NEXT:    [[SPLAT_SPLAT159:%.*]] = shufflevector <2 x double> [[COL_LOAD145]], <2 x double> undef, <2 x i32> <i32 1, i32 1>
; CHECK-NEXT:    [[TMP44:%.*]] = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> [[COL_LOAD139]], <2 x double> [[SPLAT_SPLAT159]], <2 x double> [[TMP43]])

;; -> c[0:2, 2:4]

; CHECK-NEXT:    [[TMP45:%.*]] = getelementptr <16 x double>, <16 x double>* [[C]], i64 0, i64 8
; CHECK-NEXT:    [[COL_CAST161:%.*]] = bitcast double* [[TMP45]] to <2 x double>*
; CHECK-NEXT:    store <2 x double> [[TMP42]], <2 x double>* [[COL_CAST161]], align 8
; CHECK-NEXT:    [[COL_GEP162:%.*]] = getelementptr <16 x double>, <16 x double>* [[C]], i64 0, i64 12
; CHECK-NEXT:    [[COL_CAST163:%.*]] = bitcast double* [[COL_GEP162]] to <2 x double>*
; CHECK-NEXT:    store <2 x double> [[TMP44]], <2 x double>* [[COL_CAST163]], align 8

;;  np.dot(a[2:4, 0:2], b[2:4, 0:2])

; CHECK-NEXT:    [[TMP46:%.*]] = getelementptr <16 x double>, <16 x double>* [[TMP5]], i64 0, i64 2
; CHECK-NEXT:    [[COL_CAST165:%.*]] = bitcast double* [[TMP46]] to <2 x double>*
; CHECK-NEXT:    [[COL_LOAD166:%.*]] = load <2 x double>, <2 x double>* [[COL_CAST165]], align 8
; CHECK-NEXT:    [[COL_GEP167:%.*]] = getelementptr <16 x double>, <16 x double>* [[TMP5]], i64 0, i64 6
; CHECK-NEXT:    [[COL_CAST168:%.*]] = bitcast double* [[COL_GEP167]] to <2 x double>*
; CHECK-NEXT:    [[COL_LOAD169:%.*]] = load <2 x double>, <2 x double>* [[COL_CAST168]], align 8
; CHECK-NEXT:    [[TMP47:%.*]] = getelementptr <16 x double>, <16 x double>* [[TMP11]], i64 0, i64 8
; CHECK-NEXT:    [[COL_CAST171:%.*]] = bitcast double* [[TMP47]] to <2 x double>*
; CHECK-NEXT:    [[COL_LOAD172:%.*]] = load <2 x double>, <2 x double>* [[COL_CAST171]], align 8
; CHECK-NEXT:    [[COL_GEP173:%.*]] = getelementptr <16 x double>, <16 x double>* [[TMP11]], i64 0, i64 12
; CHECK-NEXT:    [[COL_CAST174:%.*]] = bitcast double* [[COL_GEP173]] to <2 x double>*
; CHECK-NEXT:    [[COL_LOAD175:%.*]] = load <2 x double>, <2 x double>* [[COL_CAST174]], align 8
; CHECK-NEXT:    [[SPLAT_SPLAT178:%.*]] = shufflevector <2 x double> [[COL_LOAD172]], <2 x double> undef, <2 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP48:%.*]] = fmul <2 x double> [[COL_LOAD166]], [[SPLAT_SPLAT178]]
; CHECK-NEXT:    [[SPLAT_SPLAT181:%.*]] = shufflevector <2 x double> [[COL_LOAD172]], <2 x double> undef, <2 x i32> <i32 1, i32 1>
; CHECK-NEXT:    [[TMP49:%.*]] = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> [[COL_LOAD169]], <2 x double> [[SPLAT_SPLAT181]], <2 x double> [[TMP48]])
; CHECK-NEXT:    [[SPLAT_SPLAT184:%.*]] = shufflevector <2 x double> [[COL_LOAD175]], <2 x double> undef, <2 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP50:%.*]] = fmul <2 x double> [[COL_LOAD166]], [[SPLAT_SPLAT184]]
; CHECK-NEXT:    [[SPLAT_SPLAT187:%.*]] = shufflevector <2 x double> [[COL_LOAD175]], <2 x double> undef, <2 x i32> <i32 1, i32 1>
; CHECK-NEXT:    [[TMP51:%.*]] = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> [[COL_LOAD169]], <2 x double> [[SPLAT_SPLAT187]], <2 x double> [[TMP50]])

;; + np.dot(a[2:4, 2:4], b[2:4, 2:4])

; CHECK-NEXT:    [[TMP52:%.*]] = getelementptr <16 x double>, <16 x double>* [[TMP5]], i64 0, i64 10
; CHECK-NEXT:    [[COL_CAST189:%.*]] = bitcast double* [[TMP52]] to <2 x double>*
; CHECK-NEXT:    [[COL_LOAD190:%.*]] = load <2 x double>, <2 x double>* [[COL_CAST189]], align 8
; CHECK-NEXT:    [[COL_GEP191:%.*]] = getelementptr <16 x double>, <16 x double>* [[TMP5]], i64 0, i64 14
; CHECK-NEXT:    [[COL_CAST192:%.*]] = bitcast double* [[COL_GEP191]] to <2 x double>*
; CHECK-NEXT:    [[COL_LOAD193:%.*]] = load <2 x double>, <2 x double>* [[COL_CAST192]], align 8
; CHECK-NEXT:    [[TMP53:%.*]] = getelementptr <16 x double>, <16 x double>* [[TMP11]], i64 0, i64 10
; CHECK-NEXT:    [[COL_CAST195:%.*]] = bitcast double* [[TMP53]] to <2 x double>*
; CHECK-NEXT:    [[COL_LOAD196:%.*]] = load <2 x double>, <2 x double>* [[COL_CAST195]], align 8
; CHECK-NEXT:    [[COL_GEP197:%.*]] = getelementptr <16 x double>, <16 x double>* [[TMP11]], i64 0, i64 14
; CHECK-NEXT:    [[COL_CAST198:%.*]] = bitcast double* [[COL_GEP197]] to <2 x double>*
; CHECK-NEXT:    [[COL_LOAD199:%.*]] = load <2 x double>, <2 x double>* [[COL_CAST198]], align 8
; CHECK-NEXT:    [[SPLAT_SPLAT203:%.*]] = shufflevector <2 x double> [[COL_LOAD196]], <2 x double> undef, <2 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP54:%.*]] = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> [[COL_LOAD190]], <2 x double> [[SPLAT_SPLAT203]], <2 x double> [[TMP49]])
; CHECK-NEXT:    [[SPLAT_SPLAT206:%.*]] = shufflevector <2 x double> [[COL_LOAD196]], <2 x double> undef, <2 x i32> <i32 1, i32 1>
; CHECK-NEXT:    [[TMP55:%.*]] = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> [[COL_LOAD193]], <2 x double> [[SPLAT_SPLAT206]], <2 x double> [[TMP54]])
; CHECK-NEXT:    [[SPLAT_SPLAT210:%.*]] = shufflevector <2 x double> [[COL_LOAD199]], <2 x double> undef, <2 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP56:%.*]] = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> [[COL_LOAD190]], <2 x double> [[SPLAT_SPLAT210]], <2 x double> [[TMP51]])
; CHECK-NEXT:    [[SPLAT_SPLAT213:%.*]] = shufflevector <2 x double> [[COL_LOAD199]], <2 x double> undef, <2 x i32> <i32 1, i32 1>
; CHECK-NEXT:    [[TMP57:%.*]] = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> [[COL_LOAD193]], <2 x double> [[SPLAT_SPLAT213]], <2 x double> [[TMP56]])

;; ->  c[2:4, 2:4]

; CHECK-NEXT:    [[TMP58:%.*]] = getelementptr <16 x double>, <16 x double>* [[C]], i64 0, i64 10
; CHECK-NEXT:    [[COL_CAST215:%.*]] = bitcast double* [[TMP58]] to <2 x double>*
; CHECK-NEXT:    store <2 x double> [[TMP55]], <2 x double>* [[COL_CAST215]], align 8
; CHECK-NEXT:    [[COL_GEP216:%.*]] = getelementptr <16 x double>, <16 x double>* [[C]], i64 0, i64 14
; CHECK-NEXT:    [[COL_CAST217:%.*]] = bitcast double* [[COL_GEP216]] to <2 x double>*
; CHECK-NEXT:    store <2 x double> [[TMP57]], <2 x double>* [[COL_CAST217]], align 8
; CHECK-NEXT:    ret void
;
entry:
  %a = load <16 x double>, <16 x double>* %A, align 8
  %b = load <16 x double>, <16 x double>* %B, align 8

  %c = call <16 x double> @llvm.matrix.multiply(<16 x double> %a, <16 x double> %b, i32 4, i32 4, i32 4)

  store <16 x double> %c, <16 x double>* %C, align 8
  ret void
}

declare <16 x double> @llvm.matrix.multiply(<16 x double>, <16 x double>, i32, i32, i32)
