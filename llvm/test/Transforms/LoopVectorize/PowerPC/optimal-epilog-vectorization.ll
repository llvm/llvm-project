; RUN: opt < %s -passes='loop-vectorize' -enable-epilogue-vectorization -epilogue-vectorization-force-VF=2 -S | FileCheck %s --check-prefix VF-TWO-CHECK
; RUN: opt < %s -passes='loop-vectorize' -enable-epilogue-vectorization -epilogue-vectorization-force-VF=4 -S | FileCheck %s --check-prefix VF-FOUR-CHECK

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

; Function Attrs: nounwind
define dso_local void @f1(float* noalias %aa, float* noalias %bb, float* noalias %cc, i32 signext %N) #0 {
; VF-TWO-CHECK-LABEL: @f1(
; VF-TWO-CHECK-NEXT:  entry:
; VF-TWO-CHECK-NEXT:    [[CMP1:%.*]] = icmp sgt i32 [[N:%.*]], 0
; VF-TWO-CHECK-NEXT:    br i1 [[CMP1]], label [[ITER_CHECK:%.*]], label [[FOR_END:%.*]]
; VF-TWO-CHECK:       iter.check:
; VF-TWO-CHECK-NEXT:    [[WIDE_TRIP_COUNT:%.*]] = zext i32 [[N]] to i64
; VF-TWO-CHECK-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 [[WIDE_TRIP_COUNT]], 2
; VF-TWO-CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label [[VEC_EPILOG_SCALAR_PH:%.*]], label [[VECTOR_MAIN_LOOP_ITER_CHECK:%.*]]
; VF-TWO-CHECK:       vector.main.loop.iter.check:
; VF-TWO-CHECK-NEXT:    [[MIN_ITERS_CHECK1:%.*]] = icmp ult i64 [[WIDE_TRIP_COUNT]], 48
; VF-TWO-CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK1]], label [[VEC_EPILOG_PH:%.*]], label [[VECTOR_PH:%.*]]
; VF-TWO-CHECK:       vector.ph:
; VF-TWO-CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[WIDE_TRIP_COUNT]], 48
; VF-TWO-CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[WIDE_TRIP_COUNT]], [[N_MOD_VF]]
; VF-TWO-CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; VF-TWO-CHECK:       vector.body:
; VF-TWO-CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; VF-TWO-CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; VF-TWO-CHECK-NEXT:    [[TMP1:%.*]] = add i64 [[INDEX]], 4
; VF-TWO-CHECK-NEXT:    [[TMP2:%.*]] = add i64 [[INDEX]], 8
; VF-TWO-CHECK-NEXT:    [[TMP3:%.*]] = add i64 [[INDEX]], 12
; VF-TWO-CHECK-NEXT:    [[TMP4:%.*]] = add i64 [[INDEX]], 16
; VF-TWO-CHECK-NEXT:    [[TMP5:%.*]] = add i64 [[INDEX]], 20
; VF-TWO-CHECK-NEXT:    [[TMP6:%.*]] = add i64 [[INDEX]], 24
; VF-TWO-CHECK-NEXT:    [[TMP7:%.*]] = add i64 [[INDEX]], 28
; VF-TWO-CHECK-NEXT:    [[TMP8:%.*]] = add i64 [[INDEX]], 32
; VF-TWO-CHECK-NEXT:    [[TMP9:%.*]] = add i64 [[INDEX]], 36
; VF-TWO-CHECK-NEXT:    [[TMP10:%.*]] = add i64 [[INDEX]], 40
; VF-TWO-CHECK-NEXT:    [[TMP11:%.*]] = add i64 [[INDEX]], 44
; VF-TWO-CHECK-NEXT:    [[TMP12:%.*]] = getelementptr inbounds float, float* [[BB:%.*]], i64 [[TMP0]]
; VF-TWO-CHECK-NEXT:    [[TMP13:%.*]] = getelementptr inbounds float, float* [[BB]], i64 [[TMP1]]
; VF-TWO-CHECK-NEXT:    [[TMP14:%.*]] = getelementptr inbounds float, float* [[BB]], i64 [[TMP2]]
; VF-TWO-CHECK-NEXT:    [[TMP15:%.*]] = getelementptr inbounds float, float* [[BB]], i64 [[TMP3]]
; VF-TWO-CHECK-NEXT:    [[TMP16:%.*]] = getelementptr inbounds float, float* [[BB]], i64 [[TMP4]]
; VF-TWO-CHECK-NEXT:    [[TMP17:%.*]] = getelementptr inbounds float, float* [[BB]], i64 [[TMP5]]
; VF-TWO-CHECK-NEXT:    [[TMP18:%.*]] = getelementptr inbounds float, float* [[BB]], i64 [[TMP6]]
; VF-TWO-CHECK-NEXT:    [[TMP19:%.*]] = getelementptr inbounds float, float* [[BB]], i64 [[TMP7]]
; VF-TWO-CHECK-NEXT:    [[TMP20:%.*]] = getelementptr inbounds float, float* [[BB]], i64 [[TMP8]]
; VF-TWO-CHECK-NEXT:    [[TMP21:%.*]] = getelementptr inbounds float, float* [[BB]], i64 [[TMP9]]
; VF-TWO-CHECK-NEXT:    [[TMP22:%.*]] = getelementptr inbounds float, float* [[BB]], i64 [[TMP10]]
; VF-TWO-CHECK-NEXT:    [[TMP23:%.*]] = getelementptr inbounds float, float* [[BB]], i64 [[TMP11]]
; VF-TWO-CHECK-NEXT:    [[TMP24:%.*]] = getelementptr inbounds float, float* [[TMP12]], i32 0
; VF-TWO-CHECK-NEXT:    [[TMP25:%.*]] = bitcast float* [[TMP24]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <4 x float>, <4 x float>* [[TMP25]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP26:%.*]] = getelementptr inbounds float, float* [[TMP12]], i32 4
; VF-TWO-CHECK-NEXT:    [[TMP27:%.*]] = bitcast float* [[TMP26]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD2:%.*]] = load <4 x float>, <4 x float>* [[TMP27]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP28:%.*]] = getelementptr inbounds float, float* [[TMP12]], i32 8
; VF-TWO-CHECK-NEXT:    [[TMP29:%.*]] = bitcast float* [[TMP28]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD3:%.*]] = load <4 x float>, <4 x float>* [[TMP29]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP30:%.*]] = getelementptr inbounds float, float* [[TMP12]], i32 12
; VF-TWO-CHECK-NEXT:    [[TMP31:%.*]] = bitcast float* [[TMP30]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD4:%.*]] = load <4 x float>, <4 x float>* [[TMP31]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP32:%.*]] = getelementptr inbounds float, float* [[TMP12]], i32 16
; VF-TWO-CHECK-NEXT:    [[TMP33:%.*]] = bitcast float* [[TMP32]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD5:%.*]] = load <4 x float>, <4 x float>* [[TMP33]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP34:%.*]] = getelementptr inbounds float, float* [[TMP12]], i32 20
; VF-TWO-CHECK-NEXT:    [[TMP35:%.*]] = bitcast float* [[TMP34]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD6:%.*]] = load <4 x float>, <4 x float>* [[TMP35]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP36:%.*]] = getelementptr inbounds float, float* [[TMP12]], i32 24
; VF-TWO-CHECK-NEXT:    [[TMP37:%.*]] = bitcast float* [[TMP36]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD7:%.*]] = load <4 x float>, <4 x float>* [[TMP37]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP38:%.*]] = getelementptr inbounds float, float* [[TMP12]], i32 28
; VF-TWO-CHECK-NEXT:    [[TMP39:%.*]] = bitcast float* [[TMP38]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD8:%.*]] = load <4 x float>, <4 x float>* [[TMP39]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP40:%.*]] = getelementptr inbounds float, float* [[TMP12]], i32 32
; VF-TWO-CHECK-NEXT:    [[TMP41:%.*]] = bitcast float* [[TMP40]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD9:%.*]] = load <4 x float>, <4 x float>* [[TMP41]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP42:%.*]] = getelementptr inbounds float, float* [[TMP12]], i32 36
; VF-TWO-CHECK-NEXT:    [[TMP43:%.*]] = bitcast float* [[TMP42]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD10:%.*]] = load <4 x float>, <4 x float>* [[TMP43]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP44:%.*]] = getelementptr inbounds float, float* [[TMP12]], i32 40
; VF-TWO-CHECK-NEXT:    [[TMP45:%.*]] = bitcast float* [[TMP44]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD11:%.*]] = load <4 x float>, <4 x float>* [[TMP45]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP46:%.*]] = getelementptr inbounds float, float* [[TMP12]], i32 44
; VF-TWO-CHECK-NEXT:    [[TMP47:%.*]] = bitcast float* [[TMP46]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD12:%.*]] = load <4 x float>, <4 x float>* [[TMP47]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP48:%.*]] = getelementptr inbounds float, float* [[CC:%.*]], i64 [[TMP0]]
; VF-TWO-CHECK-NEXT:    [[TMP49:%.*]] = getelementptr inbounds float, float* [[CC]], i64 [[TMP1]]
; VF-TWO-CHECK-NEXT:    [[TMP50:%.*]] = getelementptr inbounds float, float* [[CC]], i64 [[TMP2]]
; VF-TWO-CHECK-NEXT:    [[TMP51:%.*]] = getelementptr inbounds float, float* [[CC]], i64 [[TMP3]]
; VF-TWO-CHECK-NEXT:    [[TMP52:%.*]] = getelementptr inbounds float, float* [[CC]], i64 [[TMP4]]
; VF-TWO-CHECK-NEXT:    [[TMP53:%.*]] = getelementptr inbounds float, float* [[CC]], i64 [[TMP5]]
; VF-TWO-CHECK-NEXT:    [[TMP54:%.*]] = getelementptr inbounds float, float* [[CC]], i64 [[TMP6]]
; VF-TWO-CHECK-NEXT:    [[TMP55:%.*]] = getelementptr inbounds float, float* [[CC]], i64 [[TMP7]]
; VF-TWO-CHECK-NEXT:    [[TMP56:%.*]] = getelementptr inbounds float, float* [[CC]], i64 [[TMP8]]
; VF-TWO-CHECK-NEXT:    [[TMP57:%.*]] = getelementptr inbounds float, float* [[CC]], i64 [[TMP9]]
; VF-TWO-CHECK-NEXT:    [[TMP58:%.*]] = getelementptr inbounds float, float* [[CC]], i64 [[TMP10]]
; VF-TWO-CHECK-NEXT:    [[TMP59:%.*]] = getelementptr inbounds float, float* [[CC]], i64 [[TMP11]]
; VF-TWO-CHECK-NEXT:    [[TMP60:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 0
; VF-TWO-CHECK-NEXT:    [[TMP61:%.*]] = bitcast float* [[TMP60]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD13:%.*]] = load <4 x float>, <4 x float>* [[TMP61]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP62:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 4
; VF-TWO-CHECK-NEXT:    [[TMP63:%.*]] = bitcast float* [[TMP62]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD14:%.*]] = load <4 x float>, <4 x float>* [[TMP63]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP64:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 8
; VF-TWO-CHECK-NEXT:    [[TMP65:%.*]] = bitcast float* [[TMP64]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD15:%.*]] = load <4 x float>, <4 x float>* [[TMP65]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP66:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 12
; VF-TWO-CHECK-NEXT:    [[TMP67:%.*]] = bitcast float* [[TMP66]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD16:%.*]] = load <4 x float>, <4 x float>* [[TMP67]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP68:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 16
; VF-TWO-CHECK-NEXT:    [[TMP69:%.*]] = bitcast float* [[TMP68]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD17:%.*]] = load <4 x float>, <4 x float>* [[TMP69]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP70:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 20
; VF-TWO-CHECK-NEXT:    [[TMP71:%.*]] = bitcast float* [[TMP70]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD18:%.*]] = load <4 x float>, <4 x float>* [[TMP71]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP72:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 24
; VF-TWO-CHECK-NEXT:    [[TMP73:%.*]] = bitcast float* [[TMP72]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD19:%.*]] = load <4 x float>, <4 x float>* [[TMP73]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP74:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 28
; VF-TWO-CHECK-NEXT:    [[TMP75:%.*]] = bitcast float* [[TMP74]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD20:%.*]] = load <4 x float>, <4 x float>* [[TMP75]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP76:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 32
; VF-TWO-CHECK-NEXT:    [[TMP77:%.*]] = bitcast float* [[TMP76]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD21:%.*]] = load <4 x float>, <4 x float>* [[TMP77]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP78:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 36
; VF-TWO-CHECK-NEXT:    [[TMP79:%.*]] = bitcast float* [[TMP78]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD22:%.*]] = load <4 x float>, <4 x float>* [[TMP79]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP80:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 40
; VF-TWO-CHECK-NEXT:    [[TMP81:%.*]] = bitcast float* [[TMP80]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD23:%.*]] = load <4 x float>, <4 x float>* [[TMP81]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP82:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 44
; VF-TWO-CHECK-NEXT:    [[TMP83:%.*]] = bitcast float* [[TMP82]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD24:%.*]] = load <4 x float>, <4 x float>* [[TMP83]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP84:%.*]] = fadd fast <4 x float> [[WIDE_LOAD]], [[WIDE_LOAD13]]
; VF-TWO-CHECK-NEXT:    [[TMP85:%.*]] = fadd fast <4 x float> [[WIDE_LOAD2]], [[WIDE_LOAD14]]
; VF-TWO-CHECK-NEXT:    [[TMP86:%.*]] = fadd fast <4 x float> [[WIDE_LOAD3]], [[WIDE_LOAD15]]
; VF-TWO-CHECK-NEXT:    [[TMP87:%.*]] = fadd fast <4 x float> [[WIDE_LOAD4]], [[WIDE_LOAD16]]
; VF-TWO-CHECK-NEXT:    [[TMP88:%.*]] = fadd fast <4 x float> [[WIDE_LOAD5]], [[WIDE_LOAD17]]
; VF-TWO-CHECK-NEXT:    [[TMP89:%.*]] = fadd fast <4 x float> [[WIDE_LOAD6]], [[WIDE_LOAD18]]
; VF-TWO-CHECK-NEXT:    [[TMP90:%.*]] = fadd fast <4 x float> [[WIDE_LOAD7]], [[WIDE_LOAD19]]
; VF-TWO-CHECK-NEXT:    [[TMP91:%.*]] = fadd fast <4 x float> [[WIDE_LOAD8]], [[WIDE_LOAD20]]
; VF-TWO-CHECK-NEXT:    [[TMP92:%.*]] = fadd fast <4 x float> [[WIDE_LOAD9]], [[WIDE_LOAD21]]
; VF-TWO-CHECK-NEXT:    [[TMP93:%.*]] = fadd fast <4 x float> [[WIDE_LOAD10]], [[WIDE_LOAD22]]
; VF-TWO-CHECK-NEXT:    [[TMP94:%.*]] = fadd fast <4 x float> [[WIDE_LOAD11]], [[WIDE_LOAD23]]
; VF-TWO-CHECK-NEXT:    [[TMP95:%.*]] = fadd fast <4 x float> [[WIDE_LOAD12]], [[WIDE_LOAD24]]
; VF-TWO-CHECK-NEXT:    [[TMP96:%.*]] = getelementptr inbounds float, float* [[AA:%.*]], i64 [[TMP0]]
; VF-TWO-CHECK-NEXT:    [[TMP97:%.*]] = getelementptr inbounds float, float* [[AA]], i64 [[TMP1]]
; VF-TWO-CHECK-NEXT:    [[TMP98:%.*]] = getelementptr inbounds float, float* [[AA]], i64 [[TMP2]]
; VF-TWO-CHECK-NEXT:    [[TMP99:%.*]] = getelementptr inbounds float, float* [[AA]], i64 [[TMP3]]
; VF-TWO-CHECK-NEXT:    [[TMP100:%.*]] = getelementptr inbounds float, float* [[AA]], i64 [[TMP4]]
; VF-TWO-CHECK-NEXT:    [[TMP101:%.*]] = getelementptr inbounds float, float* [[AA]], i64 [[TMP5]]
; VF-TWO-CHECK-NEXT:    [[TMP102:%.*]] = getelementptr inbounds float, float* [[AA]], i64 [[TMP6]]
; VF-TWO-CHECK-NEXT:    [[TMP103:%.*]] = getelementptr inbounds float, float* [[AA]], i64 [[TMP7]]
; VF-TWO-CHECK-NEXT:    [[TMP104:%.*]] = getelementptr inbounds float, float* [[AA]], i64 [[TMP8]]
; VF-TWO-CHECK-NEXT:    [[TMP105:%.*]] = getelementptr inbounds float, float* [[AA]], i64 [[TMP9]]
; VF-TWO-CHECK-NEXT:    [[TMP106:%.*]] = getelementptr inbounds float, float* [[AA]], i64 [[TMP10]]
; VF-TWO-CHECK-NEXT:    [[TMP107:%.*]] = getelementptr inbounds float, float* [[AA]], i64 [[TMP11]]
; VF-TWO-CHECK-NEXT:    [[TMP108:%.*]] = getelementptr inbounds float, float* [[TMP96]], i32 0
; VF-TWO-CHECK-NEXT:    [[TMP109:%.*]] = bitcast float* [[TMP108]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    store <4 x float> [[TMP84]], <4 x float>* [[TMP109]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP110:%.*]] = getelementptr inbounds float, float* [[TMP96]], i32 4
; VF-TWO-CHECK-NEXT:    [[TMP111:%.*]] = bitcast float* [[TMP110]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    store <4 x float> [[TMP85]], <4 x float>* [[TMP111]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP112:%.*]] = getelementptr inbounds float, float* [[TMP96]], i32 8
; VF-TWO-CHECK-NEXT:    [[TMP113:%.*]] = bitcast float* [[TMP112]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    store <4 x float> [[TMP86]], <4 x float>* [[TMP113]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP114:%.*]] = getelementptr inbounds float, float* [[TMP96]], i32 12
; VF-TWO-CHECK-NEXT:    [[TMP115:%.*]] = bitcast float* [[TMP114]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    store <4 x float> [[TMP87]], <4 x float>* [[TMP115]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP116:%.*]] = getelementptr inbounds float, float* [[TMP96]], i32 16
; VF-TWO-CHECK-NEXT:    [[TMP117:%.*]] = bitcast float* [[TMP116]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    store <4 x float> [[TMP88]], <4 x float>* [[TMP117]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP118:%.*]] = getelementptr inbounds float, float* [[TMP96]], i32 20
; VF-TWO-CHECK-NEXT:    [[TMP119:%.*]] = bitcast float* [[TMP118]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    store <4 x float> [[TMP89]], <4 x float>* [[TMP119]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP120:%.*]] = getelementptr inbounds float, float* [[TMP96]], i32 24
; VF-TWO-CHECK-NEXT:    [[TMP121:%.*]] = bitcast float* [[TMP120]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    store <4 x float> [[TMP90]], <4 x float>* [[TMP121]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP122:%.*]] = getelementptr inbounds float, float* [[TMP96]], i32 28
; VF-TWO-CHECK-NEXT:    [[TMP123:%.*]] = bitcast float* [[TMP122]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    store <4 x float> [[TMP91]], <4 x float>* [[TMP123]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP124:%.*]] = getelementptr inbounds float, float* [[TMP96]], i32 32
; VF-TWO-CHECK-NEXT:    [[TMP125:%.*]] = bitcast float* [[TMP124]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    store <4 x float> [[TMP92]], <4 x float>* [[TMP125]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP126:%.*]] = getelementptr inbounds float, float* [[TMP96]], i32 36
; VF-TWO-CHECK-NEXT:    [[TMP127:%.*]] = bitcast float* [[TMP126]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    store <4 x float> [[TMP93]], <4 x float>* [[TMP127]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP128:%.*]] = getelementptr inbounds float, float* [[TMP96]], i32 40
; VF-TWO-CHECK-NEXT:    [[TMP129:%.*]] = bitcast float* [[TMP128]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    store <4 x float> [[TMP94]], <4 x float>* [[TMP129]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP130:%.*]] = getelementptr inbounds float, float* [[TMP96]], i32 44
; VF-TWO-CHECK-NEXT:    [[TMP131:%.*]] = bitcast float* [[TMP130]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    store <4 x float> [[TMP95]], <4 x float>* [[TMP131]], align 4
; VF-TWO-CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 48
; VF-TWO-CHECK-NEXT:    [[TMP132:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; VF-TWO-CHECK-NEXT:    br i1 [[TMP132]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOPID_MV:![0-9]+]]
; VF-TWO-CHECK:       middle.block:
; VF-TWO-CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 [[WIDE_TRIP_COUNT]], [[N_VEC]]
; VF-TWO-CHECK-NEXT:    br i1 [[CMP_N]], label [[FOR_END_LOOPEXIT:%.*]], label [[VEC_EPILOG_ITER_CHECK:%.*]]
; VF-TWO-CHECK:       vec.epilog.iter.check:
; VF-TWO-CHECK-NEXT:    [[N_VEC_REMAINING:%.*]] = sub i64 [[WIDE_TRIP_COUNT]], [[N_VEC]]
; VF-TWO-CHECK-NEXT:    [[MIN_EPILOG_ITERS_CHECK:%.*]] = icmp ult i64 [[N_VEC_REMAINING]], 2
; VF-TWO-CHECK-NEXT:    br i1 [[MIN_EPILOG_ITERS_CHECK]], label [[VEC_EPILOG_SCALAR_PH]], label [[VEC_EPILOG_PH]]
; VF-TWO-CHECK:       vec.epilog.ph:
; VF-TWO-CHECK-NEXT:    [[VEC_EPILOG_RESUME_VAL:%.*]] = phi i64 [ [[N_VEC]], [[VEC_EPILOG_ITER_CHECK]] ], [ 0, [[VECTOR_MAIN_LOOP_ITER_CHECK]] ]
; VF-TWO-CHECK-NEXT:    [[N_MOD_VF25:%.*]] = urem i64 [[WIDE_TRIP_COUNT]], 2
; VF-TWO-CHECK-NEXT:    [[N_VEC26:%.*]] = sub i64 [[WIDE_TRIP_COUNT]], [[N_MOD_VF25]]
; VF-TWO-CHECK-NEXT:    br label [[VEC_EPILOG_VECTOR_BODY:%.*]]
; VF-TWO-CHECK:       vec.epilog.vector.body:
; VF-TWO-CHECK-NEXT:    [[OFFSET_IDX:%.*]] = phi i64 [ [[VEC_EPILOG_RESUME_VAL]], [[VEC_EPILOG_PH]] ], [ [[INDEX_NEXT31:%.*]], [[VEC_EPILOG_VECTOR_BODY]] ]
; VF-TWO-CHECK-NEXT:    [[TMP133:%.*]] = add i64 [[OFFSET_IDX]], 0
; VF-TWO-CHECK-NEXT:    [[TMP134:%.*]] = getelementptr inbounds float, float* [[BB]], i64 [[TMP133]]
; VF-TWO-CHECK-NEXT:    [[TMP135:%.*]] = getelementptr inbounds float, float* [[TMP134]], i32 0
; VF-TWO-CHECK-NEXT:    [[TMP136:%.*]] = bitcast float* [[TMP135]] to <2 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD29:%.*]] = load <2 x float>, <2 x float>* [[TMP136]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP137:%.*]] = getelementptr inbounds float, float* [[CC]], i64 [[TMP133]]
; VF-TWO-CHECK-NEXT:    [[TMP138:%.*]] = getelementptr inbounds float, float* [[TMP137]], i32 0
; VF-TWO-CHECK-NEXT:    [[TMP139:%.*]] = bitcast float* [[TMP138]] to <2 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD30:%.*]] = load <2 x float>, <2 x float>* [[TMP139]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP140:%.*]] = fadd fast <2 x float> [[WIDE_LOAD29]], [[WIDE_LOAD30]]
; VF-TWO-CHECK-NEXT:    [[TMP141:%.*]] = getelementptr inbounds float, float* [[AA]], i64 [[TMP133]]
; VF-TWO-CHECK-NEXT:    [[TMP142:%.*]] = getelementptr inbounds float, float* [[TMP141]], i32 0
; VF-TWO-CHECK-NEXT:    [[TMP143:%.*]] = bitcast float* [[TMP142]] to <2 x float>*
; VF-TWO-CHECK-NEXT:    store <2 x float> [[TMP140]], <2 x float>* [[TMP143]], align 4
; VF-TWO-CHECK-NEXT:    [[INDEX_NEXT31]] = add nuw i64 [[OFFSET_IDX]], 2
; VF-TWO-CHECK-NEXT:    [[TMP144:%.*]] = icmp eq i64 [[INDEX_NEXT31]], [[N_VEC26]]
; VF-TWO-CHECK-NEXT:    br i1 [[TMP144]], label [[VEC_EPILOG_MIDDLE_BLOCK:%.*]], label [[VEC_EPILOG_VECTOR_BODY]], !llvm.loop [[LOOPID_EV:![0-9]+]]
; VF-TWO-CHECK:       vec.epilog.middle.block:
; VF-TWO-CHECK-NEXT:    [[CMP_N27:%.*]] = icmp eq i64 [[WIDE_TRIP_COUNT]], [[N_VEC26]]
; VF-TWO-CHECK-NEXT:    br i1 [[CMP_N27]], label [[FOR_END_LOOPEXIT_LOOPEXIT:%.*]], label [[VEC_EPILOG_SCALAR_PH]]
; VF-TWO-CHECK:       vec.epilog.scalar.ph:
; VF-TWO-CHECK-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ [[N_VEC26]], [[VEC_EPILOG_MIDDLE_BLOCK]] ], [ [[N_VEC]], [[VEC_EPILOG_ITER_CHECK]] ], [ 0, [[ITER_CHECK]] ]
; VF-TWO-CHECK-NEXT:    br label [[FOR_BODY:%.*]]
; VF-TWO-CHECK:       for.body:
; VF-TWO-CHECK-NEXT:    [[INDVARS_IV:%.*]] = phi i64 [ [[BC_RESUME_VAL]], [[VEC_EPILOG_SCALAR_PH]] ], [ [[INDVARS_IV_NEXT:%.*]], [[FOR_BODY]] ]
; VF-TWO-CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds float, float* [[BB]], i64 [[INDVARS_IV]]
; VF-TWO-CHECK-NEXT:    [[TMP145:%.*]] = load float, float* [[ARRAYIDX]], align 4
; VF-TWO-CHECK-NEXT:    [[ARRAYIDX2:%.*]] = getelementptr inbounds float, float* [[CC]], i64 [[INDVARS_IV]]
; VF-TWO-CHECK-NEXT:    [[TMP146:%.*]] = load float, float* [[ARRAYIDX2]], align 4
; VF-TWO-CHECK-NEXT:    [[ADD:%.*]] = fadd fast float [[TMP145]], [[TMP146]]
; VF-TWO-CHECK-NEXT:    [[ARRAYIDX4:%.*]] = getelementptr inbounds float, float* [[AA]], i64 [[INDVARS_IV]]
; VF-TWO-CHECK-NEXT:    store float [[ADD]], float* [[ARRAYIDX4]], align 4
; VF-TWO-CHECK-NEXT:    [[INDVARS_IV_NEXT]] = add nuw nsw i64 [[INDVARS_IV]], 1
; VF-TWO-CHECK-NEXT:    [[EXITCOND:%.*]] = icmp ne i64 [[INDVARS_IV_NEXT]], [[WIDE_TRIP_COUNT]]
; VF-TWO-CHECK-NEXT:    br i1 [[EXITCOND]], label [[FOR_BODY]], label [[FOR_END_LOOPEXIT_LOOPEXIT]], !llvm.loop [[LOOP4:![0-9]+]]
; VF-TWO-CHECK:       for.end.loopexit.loopexit:
; VF-TWO-CHECK-NEXT:    br label [[FOR_END_LOOPEXIT]]
; VF-TWO-CHECK:       for.end.loopexit:
; VF-TWO-CHECK-NEXT:    br label [[FOR_END]]
; VF-TWO-CHECK:       for.end:
; VF-TWO-CHECK-NEXT:    ret void
;
; VF-FOUR-CHECK-LABEL: @f1(
; VF-FOUR-CHECK-NEXT:  entry:
; VF-FOUR-CHECK-NEXT:    [[CMP1:%.*]] = icmp sgt i32 [[N:%.*]], 0
; VF-FOUR-CHECK-NEXT:    br i1 [[CMP1]], label [[ITER_CHECK:%.*]], label [[FOR_END:%.*]]
; VF-FOUR-CHECK:       iter.check:
; VF-FOUR-CHECK-NEXT:    [[WIDE_TRIP_COUNT:%.*]] = zext i32 [[N]] to i64
; VF-FOUR-CHECK-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 [[WIDE_TRIP_COUNT]], 4
; VF-FOUR-CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label [[VEC_EPILOG_SCALAR_PH:%.*]], label [[VECTOR_MAIN_LOOP_ITER_CHECK:%.*]]
; VF-FOUR-CHECK:       vector.main.loop.iter.check:
; VF-FOUR-CHECK-NEXT:    [[MIN_ITERS_CHECK1:%.*]] = icmp ult i64 [[WIDE_TRIP_COUNT]], 48
; VF-FOUR-CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK1]], label [[VEC_EPILOG_PH:%.*]], label [[VECTOR_PH:%.*]]
; VF-FOUR-CHECK:       vector.ph:
; VF-FOUR-CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[WIDE_TRIP_COUNT]], 48
; VF-FOUR-CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[WIDE_TRIP_COUNT]], [[N_MOD_VF]]
; VF-FOUR-CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; VF-FOUR-CHECK:       vector.body:
; VF-FOUR-CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; VF-FOUR-CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; VF-FOUR-CHECK-NEXT:    [[TMP1:%.*]] = add i64 [[INDEX]], 4
; VF-FOUR-CHECK-NEXT:    [[TMP2:%.*]] = add i64 [[INDEX]], 8
; VF-FOUR-CHECK-NEXT:    [[TMP3:%.*]] = add i64 [[INDEX]], 12
; VF-FOUR-CHECK-NEXT:    [[TMP4:%.*]] = add i64 [[INDEX]], 16
; VF-FOUR-CHECK-NEXT:    [[TMP5:%.*]] = add i64 [[INDEX]], 20
; VF-FOUR-CHECK-NEXT:    [[TMP6:%.*]] = add i64 [[INDEX]], 24
; VF-FOUR-CHECK-NEXT:    [[TMP7:%.*]] = add i64 [[INDEX]], 28
; VF-FOUR-CHECK-NEXT:    [[TMP8:%.*]] = add i64 [[INDEX]], 32
; VF-FOUR-CHECK-NEXT:    [[TMP9:%.*]] = add i64 [[INDEX]], 36
; VF-FOUR-CHECK-NEXT:    [[TMP10:%.*]] = add i64 [[INDEX]], 40
; VF-FOUR-CHECK-NEXT:    [[TMP11:%.*]] = add i64 [[INDEX]], 44
; VF-FOUR-CHECK-NEXT:    [[TMP12:%.*]] = getelementptr inbounds float, float* [[BB:%.*]], i64 [[TMP0]]
; VF-FOUR-CHECK-NEXT:    [[TMP13:%.*]] = getelementptr inbounds float, float* [[BB]], i64 [[TMP1]]
; VF-FOUR-CHECK-NEXT:    [[TMP14:%.*]] = getelementptr inbounds float, float* [[BB]], i64 [[TMP2]]
; VF-FOUR-CHECK-NEXT:    [[TMP15:%.*]] = getelementptr inbounds float, float* [[BB]], i64 [[TMP3]]
; VF-FOUR-CHECK-NEXT:    [[TMP16:%.*]] = getelementptr inbounds float, float* [[BB]], i64 [[TMP4]]
; VF-FOUR-CHECK-NEXT:    [[TMP17:%.*]] = getelementptr inbounds float, float* [[BB]], i64 [[TMP5]]
; VF-FOUR-CHECK-NEXT:    [[TMP18:%.*]] = getelementptr inbounds float, float* [[BB]], i64 [[TMP6]]
; VF-FOUR-CHECK-NEXT:    [[TMP19:%.*]] = getelementptr inbounds float, float* [[BB]], i64 [[TMP7]]
; VF-FOUR-CHECK-NEXT:    [[TMP20:%.*]] = getelementptr inbounds float, float* [[BB]], i64 [[TMP8]]
; VF-FOUR-CHECK-NEXT:    [[TMP21:%.*]] = getelementptr inbounds float, float* [[BB]], i64 [[TMP9]]
; VF-FOUR-CHECK-NEXT:    [[TMP22:%.*]] = getelementptr inbounds float, float* [[BB]], i64 [[TMP10]]
; VF-FOUR-CHECK-NEXT:    [[TMP23:%.*]] = getelementptr inbounds float, float* [[BB]], i64 [[TMP11]]
; VF-FOUR-CHECK-NEXT:    [[TMP24:%.*]] = getelementptr inbounds float, float* [[TMP12]], i32 0
; VF-FOUR-CHECK-NEXT:    [[TMP25:%.*]] = bitcast float* [[TMP24]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <4 x float>, <4 x float>* [[TMP25]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP26:%.*]] = getelementptr inbounds float, float* [[TMP12]], i32 4
; VF-FOUR-CHECK-NEXT:    [[TMP27:%.*]] = bitcast float* [[TMP26]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD2:%.*]] = load <4 x float>, <4 x float>* [[TMP27]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP28:%.*]] = getelementptr inbounds float, float* [[TMP12]], i32 8
; VF-FOUR-CHECK-NEXT:    [[TMP29:%.*]] = bitcast float* [[TMP28]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD3:%.*]] = load <4 x float>, <4 x float>* [[TMP29]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP30:%.*]] = getelementptr inbounds float, float* [[TMP12]], i32 12
; VF-FOUR-CHECK-NEXT:    [[TMP31:%.*]] = bitcast float* [[TMP30]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD4:%.*]] = load <4 x float>, <4 x float>* [[TMP31]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP32:%.*]] = getelementptr inbounds float, float* [[TMP12]], i32 16
; VF-FOUR-CHECK-NEXT:    [[TMP33:%.*]] = bitcast float* [[TMP32]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD5:%.*]] = load <4 x float>, <4 x float>* [[TMP33]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP34:%.*]] = getelementptr inbounds float, float* [[TMP12]], i32 20
; VF-FOUR-CHECK-NEXT:    [[TMP35:%.*]] = bitcast float* [[TMP34]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD6:%.*]] = load <4 x float>, <4 x float>* [[TMP35]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP36:%.*]] = getelementptr inbounds float, float* [[TMP12]], i32 24
; VF-FOUR-CHECK-NEXT:    [[TMP37:%.*]] = bitcast float* [[TMP36]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD7:%.*]] = load <4 x float>, <4 x float>* [[TMP37]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP38:%.*]] = getelementptr inbounds float, float* [[TMP12]], i32 28
; VF-FOUR-CHECK-NEXT:    [[TMP39:%.*]] = bitcast float* [[TMP38]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD8:%.*]] = load <4 x float>, <4 x float>* [[TMP39]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP40:%.*]] = getelementptr inbounds float, float* [[TMP12]], i32 32
; VF-FOUR-CHECK-NEXT:    [[TMP41:%.*]] = bitcast float* [[TMP40]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD9:%.*]] = load <4 x float>, <4 x float>* [[TMP41]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP42:%.*]] = getelementptr inbounds float, float* [[TMP12]], i32 36
; VF-FOUR-CHECK-NEXT:    [[TMP43:%.*]] = bitcast float* [[TMP42]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD10:%.*]] = load <4 x float>, <4 x float>* [[TMP43]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP44:%.*]] = getelementptr inbounds float, float* [[TMP12]], i32 40
; VF-FOUR-CHECK-NEXT:    [[TMP45:%.*]] = bitcast float* [[TMP44]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD11:%.*]] = load <4 x float>, <4 x float>* [[TMP45]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP46:%.*]] = getelementptr inbounds float, float* [[TMP12]], i32 44
; VF-FOUR-CHECK-NEXT:    [[TMP47:%.*]] = bitcast float* [[TMP46]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD12:%.*]] = load <4 x float>, <4 x float>* [[TMP47]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP48:%.*]] = getelementptr inbounds float, float* [[CC:%.*]], i64 [[TMP0]]
; VF-FOUR-CHECK-NEXT:    [[TMP49:%.*]] = getelementptr inbounds float, float* [[CC]], i64 [[TMP1]]
; VF-FOUR-CHECK-NEXT:    [[TMP50:%.*]] = getelementptr inbounds float, float* [[CC]], i64 [[TMP2]]
; VF-FOUR-CHECK-NEXT:    [[TMP51:%.*]] = getelementptr inbounds float, float* [[CC]], i64 [[TMP3]]
; VF-FOUR-CHECK-NEXT:    [[TMP52:%.*]] = getelementptr inbounds float, float* [[CC]], i64 [[TMP4]]
; VF-FOUR-CHECK-NEXT:    [[TMP53:%.*]] = getelementptr inbounds float, float* [[CC]], i64 [[TMP5]]
; VF-FOUR-CHECK-NEXT:    [[TMP54:%.*]] = getelementptr inbounds float, float* [[CC]], i64 [[TMP6]]
; VF-FOUR-CHECK-NEXT:    [[TMP55:%.*]] = getelementptr inbounds float, float* [[CC]], i64 [[TMP7]]
; VF-FOUR-CHECK-NEXT:    [[TMP56:%.*]] = getelementptr inbounds float, float* [[CC]], i64 [[TMP8]]
; VF-FOUR-CHECK-NEXT:    [[TMP57:%.*]] = getelementptr inbounds float, float* [[CC]], i64 [[TMP9]]
; VF-FOUR-CHECK-NEXT:    [[TMP58:%.*]] = getelementptr inbounds float, float* [[CC]], i64 [[TMP10]]
; VF-FOUR-CHECK-NEXT:    [[TMP59:%.*]] = getelementptr inbounds float, float* [[CC]], i64 [[TMP11]]
; VF-FOUR-CHECK-NEXT:    [[TMP60:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 0
; VF-FOUR-CHECK-NEXT:    [[TMP61:%.*]] = bitcast float* [[TMP60]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD13:%.*]] = load <4 x float>, <4 x float>* [[TMP61]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP62:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 4
; VF-FOUR-CHECK-NEXT:    [[TMP63:%.*]] = bitcast float* [[TMP62]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD14:%.*]] = load <4 x float>, <4 x float>* [[TMP63]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP64:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 8
; VF-FOUR-CHECK-NEXT:    [[TMP65:%.*]] = bitcast float* [[TMP64]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD15:%.*]] = load <4 x float>, <4 x float>* [[TMP65]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP66:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 12
; VF-FOUR-CHECK-NEXT:    [[TMP67:%.*]] = bitcast float* [[TMP66]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD16:%.*]] = load <4 x float>, <4 x float>* [[TMP67]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP68:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 16
; VF-FOUR-CHECK-NEXT:    [[TMP69:%.*]] = bitcast float* [[TMP68]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD17:%.*]] = load <4 x float>, <4 x float>* [[TMP69]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP70:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 20
; VF-FOUR-CHECK-NEXT:    [[TMP71:%.*]] = bitcast float* [[TMP70]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD18:%.*]] = load <4 x float>, <4 x float>* [[TMP71]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP72:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 24
; VF-FOUR-CHECK-NEXT:    [[TMP73:%.*]] = bitcast float* [[TMP72]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD19:%.*]] = load <4 x float>, <4 x float>* [[TMP73]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP74:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 28
; VF-FOUR-CHECK-NEXT:    [[TMP75:%.*]] = bitcast float* [[TMP74]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD20:%.*]] = load <4 x float>, <4 x float>* [[TMP75]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP76:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 32
; VF-FOUR-CHECK-NEXT:    [[TMP77:%.*]] = bitcast float* [[TMP76]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD21:%.*]] = load <4 x float>, <4 x float>* [[TMP77]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP78:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 36
; VF-FOUR-CHECK-NEXT:    [[TMP79:%.*]] = bitcast float* [[TMP78]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD22:%.*]] = load <4 x float>, <4 x float>* [[TMP79]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP80:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 40
; VF-FOUR-CHECK-NEXT:    [[TMP81:%.*]] = bitcast float* [[TMP80]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD23:%.*]] = load <4 x float>, <4 x float>* [[TMP81]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP82:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 44
; VF-FOUR-CHECK-NEXT:    [[TMP83:%.*]] = bitcast float* [[TMP82]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD24:%.*]] = load <4 x float>, <4 x float>* [[TMP83]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP84:%.*]] = fadd fast <4 x float> [[WIDE_LOAD]], [[WIDE_LOAD13]]
; VF-FOUR-CHECK-NEXT:    [[TMP85:%.*]] = fadd fast <4 x float> [[WIDE_LOAD2]], [[WIDE_LOAD14]]
; VF-FOUR-CHECK-NEXT:    [[TMP86:%.*]] = fadd fast <4 x float> [[WIDE_LOAD3]], [[WIDE_LOAD15]]
; VF-FOUR-CHECK-NEXT:    [[TMP87:%.*]] = fadd fast <4 x float> [[WIDE_LOAD4]], [[WIDE_LOAD16]]
; VF-FOUR-CHECK-NEXT:    [[TMP88:%.*]] = fadd fast <4 x float> [[WIDE_LOAD5]], [[WIDE_LOAD17]]
; VF-FOUR-CHECK-NEXT:    [[TMP89:%.*]] = fadd fast <4 x float> [[WIDE_LOAD6]], [[WIDE_LOAD18]]
; VF-FOUR-CHECK-NEXT:    [[TMP90:%.*]] = fadd fast <4 x float> [[WIDE_LOAD7]], [[WIDE_LOAD19]]
; VF-FOUR-CHECK-NEXT:    [[TMP91:%.*]] = fadd fast <4 x float> [[WIDE_LOAD8]], [[WIDE_LOAD20]]
; VF-FOUR-CHECK-NEXT:    [[TMP92:%.*]] = fadd fast <4 x float> [[WIDE_LOAD9]], [[WIDE_LOAD21]]
; VF-FOUR-CHECK-NEXT:    [[TMP93:%.*]] = fadd fast <4 x float> [[WIDE_LOAD10]], [[WIDE_LOAD22]]
; VF-FOUR-CHECK-NEXT:    [[TMP94:%.*]] = fadd fast <4 x float> [[WIDE_LOAD11]], [[WIDE_LOAD23]]
; VF-FOUR-CHECK-NEXT:    [[TMP95:%.*]] = fadd fast <4 x float> [[WIDE_LOAD12]], [[WIDE_LOAD24]]
; VF-FOUR-CHECK-NEXT:    [[TMP96:%.*]] = getelementptr inbounds float, float* [[AA:%.*]], i64 [[TMP0]]
; VF-FOUR-CHECK-NEXT:    [[TMP97:%.*]] = getelementptr inbounds float, float* [[AA]], i64 [[TMP1]]
; VF-FOUR-CHECK-NEXT:    [[TMP98:%.*]] = getelementptr inbounds float, float* [[AA]], i64 [[TMP2]]
; VF-FOUR-CHECK-NEXT:    [[TMP99:%.*]] = getelementptr inbounds float, float* [[AA]], i64 [[TMP3]]
; VF-FOUR-CHECK-NEXT:    [[TMP100:%.*]] = getelementptr inbounds float, float* [[AA]], i64 [[TMP4]]
; VF-FOUR-CHECK-NEXT:    [[TMP101:%.*]] = getelementptr inbounds float, float* [[AA]], i64 [[TMP5]]
; VF-FOUR-CHECK-NEXT:    [[TMP102:%.*]] = getelementptr inbounds float, float* [[AA]], i64 [[TMP6]]
; VF-FOUR-CHECK-NEXT:    [[TMP103:%.*]] = getelementptr inbounds float, float* [[AA]], i64 [[TMP7]]
; VF-FOUR-CHECK-NEXT:    [[TMP104:%.*]] = getelementptr inbounds float, float* [[AA]], i64 [[TMP8]]
; VF-FOUR-CHECK-NEXT:    [[TMP105:%.*]] = getelementptr inbounds float, float* [[AA]], i64 [[TMP9]]
; VF-FOUR-CHECK-NEXT:    [[TMP106:%.*]] = getelementptr inbounds float, float* [[AA]], i64 [[TMP10]]
; VF-FOUR-CHECK-NEXT:    [[TMP107:%.*]] = getelementptr inbounds float, float* [[AA]], i64 [[TMP11]]
; VF-FOUR-CHECK-NEXT:    [[TMP108:%.*]] = getelementptr inbounds float, float* [[TMP96]], i32 0
; VF-FOUR-CHECK-NEXT:    [[TMP109:%.*]] = bitcast float* [[TMP108]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    store <4 x float> [[TMP84]], <4 x float>* [[TMP109]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP110:%.*]] = getelementptr inbounds float, float* [[TMP96]], i32 4
; VF-FOUR-CHECK-NEXT:    [[TMP111:%.*]] = bitcast float* [[TMP110]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    store <4 x float> [[TMP85]], <4 x float>* [[TMP111]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP112:%.*]] = getelementptr inbounds float, float* [[TMP96]], i32 8
; VF-FOUR-CHECK-NEXT:    [[TMP113:%.*]] = bitcast float* [[TMP112]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    store <4 x float> [[TMP86]], <4 x float>* [[TMP113]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP114:%.*]] = getelementptr inbounds float, float* [[TMP96]], i32 12
; VF-FOUR-CHECK-NEXT:    [[TMP115:%.*]] = bitcast float* [[TMP114]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    store <4 x float> [[TMP87]], <4 x float>* [[TMP115]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP116:%.*]] = getelementptr inbounds float, float* [[TMP96]], i32 16
; VF-FOUR-CHECK-NEXT:    [[TMP117:%.*]] = bitcast float* [[TMP116]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    store <4 x float> [[TMP88]], <4 x float>* [[TMP117]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP118:%.*]] = getelementptr inbounds float, float* [[TMP96]], i32 20
; VF-FOUR-CHECK-NEXT:    [[TMP119:%.*]] = bitcast float* [[TMP118]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    store <4 x float> [[TMP89]], <4 x float>* [[TMP119]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP120:%.*]] = getelementptr inbounds float, float* [[TMP96]], i32 24
; VF-FOUR-CHECK-NEXT:    [[TMP121:%.*]] = bitcast float* [[TMP120]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    store <4 x float> [[TMP90]], <4 x float>* [[TMP121]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP122:%.*]] = getelementptr inbounds float, float* [[TMP96]], i32 28
; VF-FOUR-CHECK-NEXT:    [[TMP123:%.*]] = bitcast float* [[TMP122]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    store <4 x float> [[TMP91]], <4 x float>* [[TMP123]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP124:%.*]] = getelementptr inbounds float, float* [[TMP96]], i32 32
; VF-FOUR-CHECK-NEXT:    [[TMP125:%.*]] = bitcast float* [[TMP124]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    store <4 x float> [[TMP92]], <4 x float>* [[TMP125]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP126:%.*]] = getelementptr inbounds float, float* [[TMP96]], i32 36
; VF-FOUR-CHECK-NEXT:    [[TMP127:%.*]] = bitcast float* [[TMP126]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    store <4 x float> [[TMP93]], <4 x float>* [[TMP127]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP128:%.*]] = getelementptr inbounds float, float* [[TMP96]], i32 40
; VF-FOUR-CHECK-NEXT:    [[TMP129:%.*]] = bitcast float* [[TMP128]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    store <4 x float> [[TMP94]], <4 x float>* [[TMP129]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP130:%.*]] = getelementptr inbounds float, float* [[TMP96]], i32 44
; VF-FOUR-CHECK-NEXT:    [[TMP131:%.*]] = bitcast float* [[TMP130]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    store <4 x float> [[TMP95]], <4 x float>* [[TMP131]], align 4
; VF-FOUR-CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 48
; VF-FOUR-CHECK-NEXT:    [[TMP132:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; VF-FOUR-CHECK-NEXT:    br i1 [[TMP132]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP0:![0-9]+]]
; VF-FOUR-CHECK:       middle.block:
; VF-FOUR-CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 [[WIDE_TRIP_COUNT]], [[N_VEC]]
; VF-FOUR-CHECK-NEXT:    br i1 [[CMP_N]], label [[FOR_END_LOOPEXIT:%.*]], label [[VEC_EPILOG_ITER_CHECK:%.*]]
; VF-FOUR-CHECK:       vec.epilog.iter.check:
; VF-FOUR-CHECK-NEXT:    [[N_VEC_REMAINING:%.*]] = sub i64 [[WIDE_TRIP_COUNT]], [[N_VEC]]
; VF-FOUR-CHECK-NEXT:    [[MIN_EPILOG_ITERS_CHECK:%.*]] = icmp ult i64 [[N_VEC_REMAINING]], 4
; VF-FOUR-CHECK-NEXT:    br i1 [[MIN_EPILOG_ITERS_CHECK]], label [[VEC_EPILOG_SCALAR_PH]], label [[VEC_EPILOG_PH]]
; VF-FOUR-CHECK:       vec.epilog.ph:
; VF-FOUR-CHECK-NEXT:    [[VEC_EPILOG_RESUME_VAL:%.*]] = phi i64 [ [[N_VEC]], [[VEC_EPILOG_ITER_CHECK]] ], [ 0, [[VECTOR_MAIN_LOOP_ITER_CHECK]] ]
; VF-FOUR-CHECK-NEXT:    [[N_MOD_VF25:%.*]] = urem i64 [[WIDE_TRIP_COUNT]], 4
; VF-FOUR-CHECK-NEXT:    [[N_VEC26:%.*]] = sub i64 [[WIDE_TRIP_COUNT]], [[N_MOD_VF25]]
; VF-FOUR-CHECK-NEXT:    br label [[VEC_EPILOG_VECTOR_BODY:%.*]]
; VF-FOUR-CHECK:       vec.epilog.vector.body:
; VF-FOUR-CHECK-NEXT:    [[OFFSET_IDX:%.*]] = phi i64 [ [[VEC_EPILOG_RESUME_VAL]], [[VEC_EPILOG_PH]] ], [ [[INDEX_NEXT31:%.*]], [[VEC_EPILOG_VECTOR_BODY]] ]
; VF-FOUR-CHECK-NEXT:    [[TMP133:%.*]] = add i64 [[OFFSET_IDX]], 0
; VF-FOUR-CHECK-NEXT:    [[TMP134:%.*]] = getelementptr inbounds float, float* [[BB]], i64 [[TMP133]]
; VF-FOUR-CHECK-NEXT:    [[TMP135:%.*]] = getelementptr inbounds float, float* [[TMP134]], i32 0
; VF-FOUR-CHECK-NEXT:    [[TMP136:%.*]] = bitcast float* [[TMP135]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD29:%.*]] = load <4 x float>, <4 x float>* [[TMP136]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP137:%.*]] = getelementptr inbounds float, float* [[CC]], i64 [[TMP133]]
; VF-FOUR-CHECK-NEXT:    [[TMP138:%.*]] = getelementptr inbounds float, float* [[TMP137]], i32 0
; VF-FOUR-CHECK-NEXT:    [[TMP139:%.*]] = bitcast float* [[TMP138]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD30:%.*]] = load <4 x float>, <4 x float>* [[TMP139]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP140:%.*]] = fadd fast <4 x float> [[WIDE_LOAD29]], [[WIDE_LOAD30]]
; VF-FOUR-CHECK-NEXT:    [[TMP141:%.*]] = getelementptr inbounds float, float* [[AA]], i64 [[TMP133]]
; VF-FOUR-CHECK-NEXT:    [[TMP142:%.*]] = getelementptr inbounds float, float* [[TMP141]], i32 0
; VF-FOUR-CHECK-NEXT:    [[TMP143:%.*]] = bitcast float* [[TMP142]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    store <4 x float> [[TMP140]], <4 x float>* [[TMP143]], align 4
; VF-FOUR-CHECK-NEXT:    [[INDEX_NEXT31]] = add nuw i64 [[OFFSET_IDX]], 4
; VF-FOUR-CHECK-NEXT:    [[TMP144:%.*]] = icmp eq i64 [[INDEX_NEXT31]], [[N_VEC26]]
; VF-FOUR-CHECK-NEXT:    br i1 [[TMP144]], label [[VEC_EPILOG_MIDDLE_BLOCK:%.*]], label [[VEC_EPILOG_VECTOR_BODY]], !llvm.loop [[LOOP2:![0-9]+]]
; VF-FOUR-CHECK:       vec.epilog.middle.block:
; VF-FOUR-CHECK-NEXT:    [[CMP_N27:%.*]] = icmp eq i64 [[WIDE_TRIP_COUNT]], [[N_VEC26]]
; VF-FOUR-CHECK-NEXT:    br i1 [[CMP_N27]], label [[FOR_END_LOOPEXIT_LOOPEXIT:%.*]], label [[VEC_EPILOG_SCALAR_PH]]
; VF-FOUR-CHECK:       vec.epilog.scalar.ph:
; VF-FOUR-CHECK-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ [[N_VEC26]], [[VEC_EPILOG_MIDDLE_BLOCK]] ], [ [[N_VEC]], [[VEC_EPILOG_ITER_CHECK]] ], [ 0, [[ITER_CHECK]] ]
; VF-FOUR-CHECK-NEXT:    br label [[FOR_BODY:%.*]]
; VF-FOUR-CHECK:       for.body:
; VF-FOUR-CHECK-NEXT:    [[INDVARS_IV:%.*]] = phi i64 [ [[BC_RESUME_VAL]], [[VEC_EPILOG_SCALAR_PH]] ], [ [[INDVARS_IV_NEXT:%.*]], [[FOR_BODY]] ]
; VF-FOUR-CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds float, float* [[BB]], i64 [[INDVARS_IV]]
; VF-FOUR-CHECK-NEXT:    [[TMP145:%.*]] = load float, float* [[ARRAYIDX]], align 4
; VF-FOUR-CHECK-NEXT:    [[ARRAYIDX2:%.*]] = getelementptr inbounds float, float* [[CC]], i64 [[INDVARS_IV]]
; VF-FOUR-CHECK-NEXT:    [[TMP146:%.*]] = load float, float* [[ARRAYIDX2]], align 4
; VF-FOUR-CHECK-NEXT:    [[ADD:%.*]] = fadd fast float [[TMP145]], [[TMP146]]
; VF-FOUR-CHECK-NEXT:    [[ARRAYIDX4:%.*]] = getelementptr inbounds float, float* [[AA]], i64 [[INDVARS_IV]]
; VF-FOUR-CHECK-NEXT:    store float [[ADD]], float* [[ARRAYIDX4]], align 4
; VF-FOUR-CHECK-NEXT:    [[INDVARS_IV_NEXT]] = add nuw nsw i64 [[INDVARS_IV]], 1
; VF-FOUR-CHECK-NEXT:    [[EXITCOND:%.*]] = icmp ne i64 [[INDVARS_IV_NEXT]], [[WIDE_TRIP_COUNT]]
; VF-FOUR-CHECK-NEXT:    br i1 [[EXITCOND]], label [[FOR_BODY]], label [[FOR_END_LOOPEXIT_LOOPEXIT]], !llvm.loop [[LOOP4:![0-9]+]]
; VF-FOUR-CHECK:       for.end.loopexit.loopexit:
; VF-FOUR-CHECK-NEXT:    br label [[FOR_END_LOOPEXIT]]
; VF-FOUR-CHECK:       for.end.loopexit:
; VF-FOUR-CHECK-NEXT:    br label [[FOR_END]]
; VF-FOUR-CHECK:       for.end:
; VF-FOUR-CHECK-NEXT:    ret void
;


entry:
  %cmp1 = icmp sgt i32 %N, 0
  br i1 %cmp1, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %N to i64
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %bb, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, float* %cc, i64 %indvars.iv
  %1 = load float, float* %arrayidx2, align 4
  %add = fadd fast float %0, %1
  %arrayidx4 = getelementptr inbounds float, float* %aa, i64 %indvars.iv
  store float %add, float* %arrayidx4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}

define dso_local signext i32 @f2(float* noalias %A, float* noalias %B, i32 signext %n) #0 {
; VF-TWO-CHECK-LABEL: @f2(
; VF-TWO-CHECK-NEXT:  entry:
; VF-TWO-CHECK-NEXT:    [[CMP1:%.*]] = icmp sgt i32 [[N:%.*]], 1
; VF-TWO-CHECK-NEXT:    br i1 [[CMP1]], label [[ITER_CHECK:%.*]], label [[FOR_END:%.*]]
; VF-TWO-CHECK:       iter.check:
; VF-TWO-CHECK-NEXT:    [[TMP0:%.*]] = add i32 [[N]], -1
; VF-TWO-CHECK-NEXT:    [[WIDE_TRIP_COUNT:%.*]] = zext i32 [[TMP0]] to i64
; VF-TWO-CHECK-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 [[WIDE_TRIP_COUNT]], 2
; VF-TWO-CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label [[VEC_EPILOG_SCALAR_PH:%.*]], label [[VECTOR_SCEVCHECK:%.*]]
; VF-TWO-CHECK:       vector.scevcheck:
; VF-TWO-CHECK-NEXT:    [[TMP1:%.*]] = add nsw i64 [[WIDE_TRIP_COUNT]], -1
; VF-TWO-CHECK-NEXT:    [[TMP2:%.*]] = trunc i64 [[TMP1]] to i32
; VF-TWO-CHECK-NEXT:    [[MUL:%.*]] = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 1, i32 [[TMP2]])
; VF-TWO-CHECK-NEXT:    [[MUL_RESULT:%.*]] = extractvalue { i32, i1 } [[MUL]], 0
; VF-TWO-CHECK-NEXT:    [[MUL_OVERFLOW:%.*]] = extractvalue { i32, i1 } [[MUL]], 1
; VF-TWO-CHECK-NEXT:    [[TMP3:%.*]] = sub i32 [[TMP0]], [[MUL_RESULT]]
; VF-TWO-CHECK-NEXT:    [[TMP4:%.*]] = icmp sgt i32 [[TMP3]], [[TMP0]]
; VF-TWO-CHECK-NEXT:    [[TMP5:%.*]] = or i1 [[TMP4]], [[MUL_OVERFLOW]]
; VF-TWO-CHECK-NEXT:    [[TMP6:%.*]] = icmp ugt i64 [[TMP1]], 4294967295
; VF-TWO-CHECK-NEXT:    [[TMP7:%.*]] = or i1 [[TMP5]], [[TMP6]]
; VF-TWO-CHECK-NEXT:    br i1 [[TMP7]], label [[VEC_EPILOG_SCALAR_PH]], label [[VECTOR_MAIN_LOOP_ITER_CHECK:%.*]]
; VF-TWO-CHECK:       vector.main.loop.iter.check:
; VF-TWO-CHECK-NEXT:    [[MIN_ITERS_CHECK1:%.*]] = icmp ult i64 [[WIDE_TRIP_COUNT]], 32
; VF-TWO-CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK1]], label [[VEC_EPILOG_PH:%.*]], label [[VECTOR_PH:%.*]]
; VF-TWO-CHECK:       vector.ph:
; VF-TWO-CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[WIDE_TRIP_COUNT]], 32
; VF-TWO-CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[WIDE_TRIP_COUNT]], [[N_MOD_VF]]
; VF-TWO-CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; VF-TWO-CHECK:       vector.body:
; VF-TWO-CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; VF-TWO-CHECK-NEXT:    [[OFFSET_IDX:%.*]] = trunc i64 [[INDEX]] to i32
; VF-TWO-CHECK-NEXT:    [[TMP8:%.*]] = add i32 [[OFFSET_IDX]], 0
; VF-TWO-CHECK-NEXT:    [[TMP9:%.*]] = add i32 [[OFFSET_IDX]], 4
; VF-TWO-CHECK-NEXT:    [[TMP10:%.*]] = add i32 [[OFFSET_IDX]], 8
; VF-TWO-CHECK-NEXT:    [[TMP11:%.*]] = add i32 [[OFFSET_IDX]], 12
; VF-TWO-CHECK-NEXT:    [[TMP12:%.*]] = add i32 [[OFFSET_IDX]], 16
; VF-TWO-CHECK-NEXT:    [[TMP13:%.*]] = add i32 [[OFFSET_IDX]], 20
; VF-TWO-CHECK-NEXT:    [[TMP14:%.*]] = add i32 [[OFFSET_IDX]], 24
; VF-TWO-CHECK-NEXT:    [[TMP15:%.*]] = add i32 [[OFFSET_IDX]], 28
; VF-TWO-CHECK-NEXT:    [[TMP16:%.*]] = add i64 [[INDEX]], 0
; VF-TWO-CHECK-NEXT:    [[TMP17:%.*]] = add i64 [[INDEX]], 4
; VF-TWO-CHECK-NEXT:    [[TMP18:%.*]] = add i64 [[INDEX]], 8
; VF-TWO-CHECK-NEXT:    [[TMP19:%.*]] = add i64 [[INDEX]], 12
; VF-TWO-CHECK-NEXT:    [[TMP20:%.*]] = add i64 [[INDEX]], 16
; VF-TWO-CHECK-NEXT:    [[TMP21:%.*]] = add i64 [[INDEX]], 20
; VF-TWO-CHECK-NEXT:    [[TMP22:%.*]] = add i64 [[INDEX]], 24
; VF-TWO-CHECK-NEXT:    [[TMP23:%.*]] = add i64 [[INDEX]], 28
; VF-TWO-CHECK-NEXT:    [[TMP24:%.*]] = xor i32 [[TMP8]], -1
; VF-TWO-CHECK-NEXT:    [[TMP25:%.*]] = xor i32 [[TMP9]], -1
; VF-TWO-CHECK-NEXT:    [[TMP26:%.*]] = xor i32 [[TMP10]], -1
; VF-TWO-CHECK-NEXT:    [[TMP27:%.*]] = xor i32 [[TMP11]], -1
; VF-TWO-CHECK-NEXT:    [[TMP28:%.*]] = xor i32 [[TMP12]], -1
; VF-TWO-CHECK-NEXT:    [[TMP29:%.*]] = xor i32 [[TMP13]], -1
; VF-TWO-CHECK-NEXT:    [[TMP30:%.*]] = xor i32 [[TMP14]], -1
; VF-TWO-CHECK-NEXT:    [[TMP31:%.*]] = xor i32 [[TMP15]], -1
; VF-TWO-CHECK-NEXT:    [[TMP32:%.*]] = add i32 [[TMP24]], [[N]]
; VF-TWO-CHECK-NEXT:    [[TMP33:%.*]] = add i32 [[TMP25]], [[N]]
; VF-TWO-CHECK-NEXT:    [[TMP34:%.*]] = add i32 [[TMP26]], [[N]]
; VF-TWO-CHECK-NEXT:    [[TMP35:%.*]] = add i32 [[TMP27]], [[N]]
; VF-TWO-CHECK-NEXT:    [[TMP36:%.*]] = add i32 [[TMP28]], [[N]]
; VF-TWO-CHECK-NEXT:    [[TMP37:%.*]] = add i32 [[TMP29]], [[N]]
; VF-TWO-CHECK-NEXT:    [[TMP38:%.*]] = add i32 [[TMP30]], [[N]]
; VF-TWO-CHECK-NEXT:    [[TMP39:%.*]] = add i32 [[TMP31]], [[N]]
; VF-TWO-CHECK-NEXT:    [[TMP40:%.*]] = sext i32 [[TMP32]] to i64
; VF-TWO-CHECK-NEXT:    [[TMP41:%.*]] = sext i32 [[TMP33]] to i64
; VF-TWO-CHECK-NEXT:    [[TMP42:%.*]] = sext i32 [[TMP34]] to i64
; VF-TWO-CHECK-NEXT:    [[TMP43:%.*]] = sext i32 [[TMP35]] to i64
; VF-TWO-CHECK-NEXT:    [[TMP44:%.*]] = sext i32 [[TMP36]] to i64
; VF-TWO-CHECK-NEXT:    [[TMP45:%.*]] = sext i32 [[TMP37]] to i64
; VF-TWO-CHECK-NEXT:    [[TMP46:%.*]] = sext i32 [[TMP38]] to i64
; VF-TWO-CHECK-NEXT:    [[TMP47:%.*]] = sext i32 [[TMP39]] to i64
; VF-TWO-CHECK-NEXT:    [[TMP48:%.*]] = getelementptr inbounds float, float* [[B:%.*]], i64 [[TMP40]]
; VF-TWO-CHECK-NEXT:    [[TMP49:%.*]] = getelementptr inbounds float, float* [[B]], i64 [[TMP41]]
; VF-TWO-CHECK-NEXT:    [[TMP50:%.*]] = getelementptr inbounds float, float* [[B]], i64 [[TMP42]]
; VF-TWO-CHECK-NEXT:    [[TMP51:%.*]] = getelementptr inbounds float, float* [[B]], i64 [[TMP43]]
; VF-TWO-CHECK-NEXT:    [[TMP52:%.*]] = getelementptr inbounds float, float* [[B]], i64 [[TMP44]]
; VF-TWO-CHECK-NEXT:    [[TMP53:%.*]] = getelementptr inbounds float, float* [[B]], i64 [[TMP45]]
; VF-TWO-CHECK-NEXT:    [[TMP54:%.*]] = getelementptr inbounds float, float* [[B]], i64 [[TMP46]]
; VF-TWO-CHECK-NEXT:    [[TMP55:%.*]] = getelementptr inbounds float, float* [[B]], i64 [[TMP47]]
; VF-TWO-CHECK-NEXT:    [[TMP56:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 0
; VF-TWO-CHECK-NEXT:    [[TMP57:%.*]] = getelementptr inbounds float, float* [[TMP56]], i32 -3
; VF-TWO-CHECK-NEXT:    [[TMP58:%.*]] = bitcast float* [[TMP57]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <4 x float>, <4 x float>* [[TMP58]], align 4
; VF-TWO-CHECK-NEXT:    [[REVERSE:%.*]] = shufflevector <4 x float> [[WIDE_LOAD]], <4 x float> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; VF-TWO-CHECK-NEXT:    [[TMP59:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 -4
; VF-TWO-CHECK-NEXT:    [[TMP60:%.*]] = getelementptr inbounds float, float* [[TMP59]], i32 -3
; VF-TWO-CHECK-NEXT:    [[TMP61:%.*]] = bitcast float* [[TMP60]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD2:%.*]] = load <4 x float>, <4 x float>* [[TMP61]], align 4
; VF-TWO-CHECK-NEXT:    [[REVERSE3:%.*]] = shufflevector <4 x float> [[WIDE_LOAD2]], <4 x float> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; VF-TWO-CHECK-NEXT:    [[TMP62:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 -8
; VF-TWO-CHECK-NEXT:    [[TMP63:%.*]] = getelementptr inbounds float, float* [[TMP62]], i32 -3
; VF-TWO-CHECK-NEXT:    [[TMP64:%.*]] = bitcast float* [[TMP63]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD4:%.*]] = load <4 x float>, <4 x float>* [[TMP64]], align 4
; VF-TWO-CHECK-NEXT:    [[REVERSE5:%.*]] = shufflevector <4 x float> [[WIDE_LOAD4]], <4 x float> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; VF-TWO-CHECK-NEXT:    [[TMP65:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 -12
; VF-TWO-CHECK-NEXT:    [[TMP66:%.*]] = getelementptr inbounds float, float* [[TMP65]], i32 -3
; VF-TWO-CHECK-NEXT:    [[TMP67:%.*]] = bitcast float* [[TMP66]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD6:%.*]] = load <4 x float>, <4 x float>* [[TMP67]], align 4
; VF-TWO-CHECK-NEXT:    [[REVERSE7:%.*]] = shufflevector <4 x float> [[WIDE_LOAD6]], <4 x float> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; VF-TWO-CHECK-NEXT:    [[TMP68:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 -16
; VF-TWO-CHECK-NEXT:    [[TMP69:%.*]] = getelementptr inbounds float, float* [[TMP68]], i32 -3
; VF-TWO-CHECK-NEXT:    [[TMP70:%.*]] = bitcast float* [[TMP69]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD8:%.*]] = load <4 x float>, <4 x float>* [[TMP70]], align 4
; VF-TWO-CHECK-NEXT:    [[REVERSE9:%.*]] = shufflevector <4 x float> [[WIDE_LOAD8]], <4 x float> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; VF-TWO-CHECK-NEXT:    [[TMP71:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 -20
; VF-TWO-CHECK-NEXT:    [[TMP72:%.*]] = getelementptr inbounds float, float* [[TMP71]], i32 -3
; VF-TWO-CHECK-NEXT:    [[TMP73:%.*]] = bitcast float* [[TMP72]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD10:%.*]] = load <4 x float>, <4 x float>* [[TMP73]], align 4
; VF-TWO-CHECK-NEXT:    [[REVERSE11:%.*]] = shufflevector <4 x float> [[WIDE_LOAD10]], <4 x float> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; VF-TWO-CHECK-NEXT:    [[TMP74:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 -24
; VF-TWO-CHECK-NEXT:    [[TMP75:%.*]] = getelementptr inbounds float, float* [[TMP74]], i32 -3
; VF-TWO-CHECK-NEXT:    [[TMP76:%.*]] = bitcast float* [[TMP75]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD12:%.*]] = load <4 x float>, <4 x float>* [[TMP76]], align 4
; VF-TWO-CHECK-NEXT:    [[REVERSE13:%.*]] = shufflevector <4 x float> [[WIDE_LOAD12]], <4 x float> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; VF-TWO-CHECK-NEXT:    [[TMP77:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 -28
; VF-TWO-CHECK-NEXT:    [[TMP78:%.*]] = getelementptr inbounds float, float* [[TMP77]], i32 -3
; VF-TWO-CHECK-NEXT:    [[TMP79:%.*]] = bitcast float* [[TMP78]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD14:%.*]] = load <4 x float>, <4 x float>* [[TMP79]], align 4
; VF-TWO-CHECK-NEXT:    [[REVERSE15:%.*]] = shufflevector <4 x float> [[WIDE_LOAD14]], <4 x float> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; VF-TWO-CHECK-NEXT:    [[TMP80:%.*]] = fadd fast <4 x float> [[REVERSE]], <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
; VF-TWO-CHECK-NEXT:    [[TMP81:%.*]] = fadd fast <4 x float> [[REVERSE3]], <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
; VF-TWO-CHECK-NEXT:    [[TMP82:%.*]] = fadd fast <4 x float> [[REVERSE5]], <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
; VF-TWO-CHECK-NEXT:    [[TMP83:%.*]] = fadd fast <4 x float> [[REVERSE7]], <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
; VF-TWO-CHECK-NEXT:    [[TMP84:%.*]] = fadd fast <4 x float> [[REVERSE9]], <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
; VF-TWO-CHECK-NEXT:    [[TMP85:%.*]] = fadd fast <4 x float> [[REVERSE11]], <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
; VF-TWO-CHECK-NEXT:    [[TMP86:%.*]] = fadd fast <4 x float> [[REVERSE13]], <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
; VF-TWO-CHECK-NEXT:    [[TMP87:%.*]] = fadd fast <4 x float> [[REVERSE15]], <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
; VF-TWO-CHECK-NEXT:    [[TMP88:%.*]] = getelementptr inbounds float, float* [[A:%.*]], i64 [[TMP16]]
; VF-TWO-CHECK-NEXT:    [[TMP89:%.*]] = getelementptr inbounds float, float* [[A]], i64 [[TMP17]]
; VF-TWO-CHECK-NEXT:    [[TMP90:%.*]] = getelementptr inbounds float, float* [[A]], i64 [[TMP18]]
; VF-TWO-CHECK-NEXT:    [[TMP91:%.*]] = getelementptr inbounds float, float* [[A]], i64 [[TMP19]]
; VF-TWO-CHECK-NEXT:    [[TMP92:%.*]] = getelementptr inbounds float, float* [[A]], i64 [[TMP20]]
; VF-TWO-CHECK-NEXT:    [[TMP93:%.*]] = getelementptr inbounds float, float* [[A]], i64 [[TMP21]]
; VF-TWO-CHECK-NEXT:    [[TMP94:%.*]] = getelementptr inbounds float, float* [[A]], i64 [[TMP22]]
; VF-TWO-CHECK-NEXT:    [[TMP95:%.*]] = getelementptr inbounds float, float* [[A]], i64 [[TMP23]]
; VF-TWO-CHECK-NEXT:    [[TMP96:%.*]] = getelementptr inbounds float, float* [[TMP88]], i32 0
; VF-TWO-CHECK-NEXT:    [[TMP97:%.*]] = bitcast float* [[TMP96]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    store <4 x float> [[TMP80]], <4 x float>* [[TMP97]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP98:%.*]] = getelementptr inbounds float, float* [[TMP88]], i32 4
; VF-TWO-CHECK-NEXT:    [[TMP99:%.*]] = bitcast float* [[TMP98]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    store <4 x float> [[TMP81]], <4 x float>* [[TMP99]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP100:%.*]] = getelementptr inbounds float, float* [[TMP88]], i32 8
; VF-TWO-CHECK-NEXT:    [[TMP101:%.*]] = bitcast float* [[TMP100]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    store <4 x float> [[TMP82]], <4 x float>* [[TMP101]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP102:%.*]] = getelementptr inbounds float, float* [[TMP88]], i32 12
; VF-TWO-CHECK-NEXT:    [[TMP103:%.*]] = bitcast float* [[TMP102]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    store <4 x float> [[TMP83]], <4 x float>* [[TMP103]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP104:%.*]] = getelementptr inbounds float, float* [[TMP88]], i32 16
; VF-TWO-CHECK-NEXT:    [[TMP105:%.*]] = bitcast float* [[TMP104]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    store <4 x float> [[TMP84]], <4 x float>* [[TMP105]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP106:%.*]] = getelementptr inbounds float, float* [[TMP88]], i32 20
; VF-TWO-CHECK-NEXT:    [[TMP107:%.*]] = bitcast float* [[TMP106]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    store <4 x float> [[TMP85]], <4 x float>* [[TMP107]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP108:%.*]] = getelementptr inbounds float, float* [[TMP88]], i32 24
; VF-TWO-CHECK-NEXT:    [[TMP109:%.*]] = bitcast float* [[TMP108]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    store <4 x float> [[TMP86]], <4 x float>* [[TMP109]], align 4
; VF-TWO-CHECK-NEXT:    [[TMP110:%.*]] = getelementptr inbounds float, float* [[TMP88]], i32 28
; VF-TWO-CHECK-NEXT:    [[TMP111:%.*]] = bitcast float* [[TMP110]] to <4 x float>*
; VF-TWO-CHECK-NEXT:    store <4 x float> [[TMP87]], <4 x float>* [[TMP111]], align 4
; VF-TWO-CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 32
; VF-TWO-CHECK-NEXT:    [[TMP112:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; VF-TWO-CHECK-NEXT:    br i1 [[TMP112]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP5:![0-9]+]]
; VF-TWO-CHECK:       middle.block:
; VF-TWO-CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 [[WIDE_TRIP_COUNT]], [[N_VEC]]
; VF-TWO-CHECK-NEXT:    br i1 [[CMP_N]], label [[FOR_END_LOOPEXIT:%.*]], label [[VEC_EPILOG_ITER_CHECK:%.*]]
; VF-TWO-CHECK:       vec.epilog.iter.check:
; VF-TWO-CHECK-NEXT:    [[IND_END19:%.*]] = trunc i64 [[N_VEC]] to i32
; VF-TWO-CHECK-NEXT:    [[N_VEC_REMAINING:%.*]] = sub i64 [[WIDE_TRIP_COUNT]], [[N_VEC]]
; VF-TWO-CHECK-NEXT:    [[MIN_EPILOG_ITERS_CHECK:%.*]] = icmp ult i64 [[N_VEC_REMAINING]], 2
; VF-TWO-CHECK-NEXT:    br i1 [[MIN_EPILOG_ITERS_CHECK]], label [[VEC_EPILOG_SCALAR_PH]], label [[VEC_EPILOG_PH]]
; VF-TWO-CHECK:       vec.epilog.ph:
; VF-TWO-CHECK-NEXT:    [[VEC_EPILOG_RESUME_VAL:%.*]] = phi i64 [ [[N_VEC]], [[VEC_EPILOG_ITER_CHECK]] ], [ 0, [[VECTOR_MAIN_LOOP_ITER_CHECK]] ]
; VF-TWO-CHECK-NEXT:    [[N_MOD_VF16:%.*]] = urem i64 [[WIDE_TRIP_COUNT]], 2
; VF-TWO-CHECK-NEXT:    [[N_VEC17:%.*]] = sub i64 [[WIDE_TRIP_COUNT]], [[N_MOD_VF16]]
; VF-TWO-CHECK-NEXT:    [[IND_END:%.*]] = trunc i64 [[N_VEC17]] to i32
; VF-TWO-CHECK-NEXT:    br label [[VEC_EPILOG_VECTOR_BODY:%.*]]
; VF-TWO-CHECK:       vec.epilog.vector.body:
; VF-TWO-CHECK-NEXT:    [[OFFSET_IDX23:%.*]] = phi i64 [ [[VEC_EPILOG_RESUME_VAL]], [[VEC_EPILOG_PH]] ], [ [[INDEX_NEXT26:%.*]], [[VEC_EPILOG_VECTOR_BODY]] ]
; VF-TWO-CHECK-NEXT:    [[OFFSET_IDX22:%.*]] = trunc i64 [[OFFSET_IDX23]] to i32
; VF-TWO-CHECK-NEXT:    [[TMP113:%.*]] = add i32 [[OFFSET_IDX22]], 0
; VF-TWO-CHECK-NEXT:    [[TMP114:%.*]] = add i64 [[OFFSET_IDX23]], 0
; VF-TWO-CHECK-NEXT:    [[TMP115:%.*]] = xor i32 [[TMP113]], -1
; VF-TWO-CHECK-NEXT:    [[TMP116:%.*]] = add i32 [[TMP115]], [[N]]
; VF-TWO-CHECK-NEXT:    [[TMP117:%.*]] = sext i32 [[TMP116]] to i64
; VF-TWO-CHECK-NEXT:    [[TMP118:%.*]] = getelementptr inbounds float, float* [[B]], i64 [[TMP117]]
; VF-TWO-CHECK-NEXT:    [[TMP119:%.*]] = getelementptr inbounds float, float* [[TMP118]], i32 0
; VF-TWO-CHECK-NEXT:    [[TMP120:%.*]] = getelementptr inbounds float, float* [[TMP119]], i32 -1
; VF-TWO-CHECK-NEXT:    [[TMP121:%.*]] = bitcast float* [[TMP120]] to <2 x float>*
; VF-TWO-CHECK-NEXT:    [[WIDE_LOAD24:%.*]] = load <2 x float>, <2 x float>* [[TMP121]], align 4
; VF-TWO-CHECK-NEXT:    [[REVERSE25:%.*]] = shufflevector <2 x float> [[WIDE_LOAD24]], <2 x float> poison, <2 x i32> <i32 1, i32 0>
; VF-TWO-CHECK-NEXT:    [[TMP122:%.*]] = fadd fast <2 x float> [[REVERSE25]], <float 1.000000e+00, float 1.000000e+00>
; VF-TWO-CHECK-NEXT:    [[TMP123:%.*]] = getelementptr inbounds float, float* [[A]], i64 [[TMP114]]
; VF-TWO-CHECK-NEXT:    [[TMP124:%.*]] = getelementptr inbounds float, float* [[TMP123]], i32 0
; VF-TWO-CHECK-NEXT:    [[TMP125:%.*]] = bitcast float* [[TMP124]] to <2 x float>*
; VF-TWO-CHECK-NEXT:    store <2 x float> [[TMP122]], <2 x float>* [[TMP125]], align 4
; VF-TWO-CHECK-NEXT:    [[INDEX_NEXT26]] = add nuw i64 [[OFFSET_IDX23]], 2
; VF-TWO-CHECK-NEXT:    [[TMP126:%.*]] = icmp eq i64 [[INDEX_NEXT26]], [[N_VEC17]]
; VF-TWO-CHECK-NEXT:    br i1 [[TMP126]], label [[VEC_EPILOG_MIDDLE_BLOCK:%.*]], label [[VEC_EPILOG_VECTOR_BODY]], !llvm.loop [[LOOP6:![0-9]+]]
; VF-TWO-CHECK:       vec.epilog.middle.block:
; VF-TWO-CHECK-NEXT:    [[CMP_N20:%.*]] = icmp eq i64 [[WIDE_TRIP_COUNT]], [[N_VEC17]]
; VF-TWO-CHECK-NEXT:    br i1 [[CMP_N20]], label [[FOR_END_LOOPEXIT_LOOPEXIT:%.*]], label [[VEC_EPILOG_SCALAR_PH]]
; VF-TWO-CHECK:       vec.epilog.scalar.ph:
; VF-TWO-CHECK-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ [[N_VEC17]], [[VEC_EPILOG_MIDDLE_BLOCK]] ], [ [[N_VEC]], [[VEC_EPILOG_ITER_CHECK]] ], [ 0, [[VECTOR_SCEVCHECK]] ], [ 0, [[ITER_CHECK]] ]
; VF-TWO-CHECK-NEXT:    [[BC_RESUME_VAL18:%.*]] = phi i32 [ [[IND_END]], [[VEC_EPILOG_MIDDLE_BLOCK]] ], [ [[IND_END19]], [[VEC_EPILOG_ITER_CHECK]] ], [ 0, [[VECTOR_SCEVCHECK]] ], [ 0, [[ITER_CHECK]] ]
; VF-TWO-CHECK-NEXT:    br label [[FOR_BODY:%.*]]
; VF-TWO-CHECK:       for.body:
; VF-TWO-CHECK-NEXT:    [[INDVARS_IV:%.*]] = phi i64 [ [[BC_RESUME_VAL]], [[VEC_EPILOG_SCALAR_PH]] ], [ [[INDVARS_IV_NEXT:%.*]], [[FOR_BODY]] ]
; VF-TWO-CHECK-NEXT:    [[I_014:%.*]] = phi i32 [ [[BC_RESUME_VAL18]], [[VEC_EPILOG_SCALAR_PH]] ], [ [[INC:%.*]], [[FOR_BODY]] ]
; VF-TWO-CHECK-NEXT:    [[TMP127:%.*]] = xor i32 [[I_014]], -1
; VF-TWO-CHECK-NEXT:    [[SUB2:%.*]] = add i32 [[TMP127]], [[N]]
; VF-TWO-CHECK-NEXT:    [[IDXPROM:%.*]] = sext i32 [[SUB2]] to i64
; VF-TWO-CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds float, float* [[B]], i64 [[IDXPROM]]
; VF-TWO-CHECK-NEXT:    [[TMP128:%.*]] = load float, float* [[ARRAYIDX]], align 4
; VF-TWO-CHECK-NEXT:    [[CONV3:%.*]] = fadd fast float [[TMP128]], 1.000000e+00
; VF-TWO-CHECK-NEXT:    [[ARRAYIDX5:%.*]] = getelementptr inbounds float, float* [[A]], i64 [[INDVARS_IV]]
; VF-TWO-CHECK-NEXT:    store float [[CONV3]], float* [[ARRAYIDX5]], align 4
; VF-TWO-CHECK-NEXT:    [[INDVARS_IV_NEXT]] = add nuw nsw i64 [[INDVARS_IV]], 1
; VF-TWO-CHECK-NEXT:    [[INC]] = add nuw nsw i32 [[I_014]], 1
; VF-TWO-CHECK-NEXT:    [[EXITCOND:%.*]] = icmp ne i64 [[INDVARS_IV_NEXT]], [[WIDE_TRIP_COUNT]]
; VF-TWO-CHECK-NEXT:    br i1 [[EXITCOND]], label [[FOR_BODY]], label [[FOR_END_LOOPEXIT_LOOPEXIT]], !llvm.loop [[LOOPID_MS_CM:![0-9]+]]
; VF-TWO-CHECK:       for.end.loopexit.loopexit:
; VF-TWO-CHECK-NEXT:    br label [[FOR_END_LOOPEXIT]]
; VF-TWO-CHECK:       for.end.loopexit:
; VF-TWO-CHECK-NEXT:    br label [[FOR_END]]
; VF-TWO-CHECK:       for.end:
; VF-TWO-CHECK-NEXT:    ret i32 0
;
; VF-FOUR-CHECK-LABEL: @f2(
; VF-FOUR-CHECK-NEXT:  entry:
; VF-FOUR-CHECK-NEXT:    [[CMP1:%.*]] = icmp sgt i32 [[N:%.*]], 1
; VF-FOUR-CHECK-NEXT:    br i1 [[CMP1]], label [[ITER_CHECK:%.*]], label [[FOR_END:%.*]]
; VF-FOUR-CHECK:       iter.check:
; VF-FOUR-CHECK-NEXT:    [[TMP0:%.*]] = add i32 [[N]], -1
; VF-FOUR-CHECK-NEXT:    [[WIDE_TRIP_COUNT:%.*]] = zext i32 [[TMP0]] to i64
; VF-FOUR-CHECK-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 [[WIDE_TRIP_COUNT]], 4
; VF-FOUR-CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label [[VEC_EPILOG_SCALAR_PH:%.*]], label [[VECTOR_SCEVCHECK:%.*]]
; VF-FOUR-CHECK:       vector.scevcheck:
; VF-FOUR-CHECK-NEXT:    [[TMP1:%.*]] = add nsw i64 [[WIDE_TRIP_COUNT]], -1
; VF-FOUR-CHECK-NEXT:    [[TMP2:%.*]] = trunc i64 [[TMP1]] to i32
; VF-FOUR-CHECK-NEXT:    [[MUL:%.*]] = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 1, i32 [[TMP2]])
; VF-FOUR-CHECK-NEXT:    [[MUL_RESULT:%.*]] = extractvalue { i32, i1 } [[MUL]], 0
; VF-FOUR-CHECK-NEXT:    [[MUL_OVERFLOW:%.*]] = extractvalue { i32, i1 } [[MUL]], 1
; VF-FOUR-CHECK-NEXT:    [[TMP3:%.*]] = sub i32 [[TMP0]], [[MUL_RESULT]]
; VF-FOUR-CHECK-NEXT:    [[TMP4:%.*]] = icmp sgt i32 [[TMP3]], [[TMP0]]
; VF-FOUR-CHECK-NEXT:    [[TMP5:%.*]] = or i1 [[TMP4]], [[MUL_OVERFLOW]]
; VF-FOUR-CHECK-NEXT:    [[TMP6:%.*]] = icmp ugt i64 [[TMP1]], 4294967295
; VF-FOUR-CHECK-NEXT:    [[TMP7:%.*]] = or i1 [[TMP5]], [[TMP6]]
; VF-FOUR-CHECK-NEXT:    br i1 [[TMP7]], label [[VEC_EPILOG_SCALAR_PH]], label [[VECTOR_MAIN_LOOP_ITER_CHECK:%.*]]
; VF-FOUR-CHECK:       vector.main.loop.iter.check:
; VF-FOUR-CHECK-NEXT:    [[MIN_ITERS_CHECK1:%.*]] = icmp ult i64 [[WIDE_TRIP_COUNT]], 32
; VF-FOUR-CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK1]], label [[VEC_EPILOG_PH:%.*]], label [[VECTOR_PH:%.*]]
; VF-FOUR-CHECK:       vector.ph:
; VF-FOUR-CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[WIDE_TRIP_COUNT]], 32
; VF-FOUR-CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[WIDE_TRIP_COUNT]], [[N_MOD_VF]]
; VF-FOUR-CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; VF-FOUR-CHECK:       vector.body:
; VF-FOUR-CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; VF-FOUR-CHECK-NEXT:    [[OFFSET_IDX:%.*]] = trunc i64 [[INDEX]] to i32
; VF-FOUR-CHECK-NEXT:    [[TMP8:%.*]] = add i32 [[OFFSET_IDX]], 0
; VF-FOUR-CHECK-NEXT:    [[TMP9:%.*]] = add i32 [[OFFSET_IDX]], 4
; VF-FOUR-CHECK-NEXT:    [[TMP10:%.*]] = add i32 [[OFFSET_IDX]], 8
; VF-FOUR-CHECK-NEXT:    [[TMP11:%.*]] = add i32 [[OFFSET_IDX]], 12
; VF-FOUR-CHECK-NEXT:    [[TMP12:%.*]] = add i32 [[OFFSET_IDX]], 16
; VF-FOUR-CHECK-NEXT:    [[TMP13:%.*]] = add i32 [[OFFSET_IDX]], 20
; VF-FOUR-CHECK-NEXT:    [[TMP14:%.*]] = add i32 [[OFFSET_IDX]], 24
; VF-FOUR-CHECK-NEXT:    [[TMP15:%.*]] = add i32 [[OFFSET_IDX]], 28
; VF-FOUR-CHECK-NEXT:    [[TMP16:%.*]] = add i64 [[INDEX]], 0
; VF-FOUR-CHECK-NEXT:    [[TMP17:%.*]] = add i64 [[INDEX]], 4
; VF-FOUR-CHECK-NEXT:    [[TMP18:%.*]] = add i64 [[INDEX]], 8
; VF-FOUR-CHECK-NEXT:    [[TMP19:%.*]] = add i64 [[INDEX]], 12
; VF-FOUR-CHECK-NEXT:    [[TMP20:%.*]] = add i64 [[INDEX]], 16
; VF-FOUR-CHECK-NEXT:    [[TMP21:%.*]] = add i64 [[INDEX]], 20
; VF-FOUR-CHECK-NEXT:    [[TMP22:%.*]] = add i64 [[INDEX]], 24
; VF-FOUR-CHECK-NEXT:    [[TMP23:%.*]] = add i64 [[INDEX]], 28
; VF-FOUR-CHECK-NEXT:    [[TMP24:%.*]] = xor i32 [[TMP8]], -1
; VF-FOUR-CHECK-NEXT:    [[TMP25:%.*]] = xor i32 [[TMP9]], -1
; VF-FOUR-CHECK-NEXT:    [[TMP26:%.*]] = xor i32 [[TMP10]], -1
; VF-FOUR-CHECK-NEXT:    [[TMP27:%.*]] = xor i32 [[TMP11]], -1
; VF-FOUR-CHECK-NEXT:    [[TMP28:%.*]] = xor i32 [[TMP12]], -1
; VF-FOUR-CHECK-NEXT:    [[TMP29:%.*]] = xor i32 [[TMP13]], -1
; VF-FOUR-CHECK-NEXT:    [[TMP30:%.*]] = xor i32 [[TMP14]], -1
; VF-FOUR-CHECK-NEXT:    [[TMP31:%.*]] = xor i32 [[TMP15]], -1
; VF-FOUR-CHECK-NEXT:    [[TMP32:%.*]] = add i32 [[TMP24]], [[N]]
; VF-FOUR-CHECK-NEXT:    [[TMP33:%.*]] = add i32 [[TMP25]], [[N]]
; VF-FOUR-CHECK-NEXT:    [[TMP34:%.*]] = add i32 [[TMP26]], [[N]]
; VF-FOUR-CHECK-NEXT:    [[TMP35:%.*]] = add i32 [[TMP27]], [[N]]
; VF-FOUR-CHECK-NEXT:    [[TMP36:%.*]] = add i32 [[TMP28]], [[N]]
; VF-FOUR-CHECK-NEXT:    [[TMP37:%.*]] = add i32 [[TMP29]], [[N]]
; VF-FOUR-CHECK-NEXT:    [[TMP38:%.*]] = add i32 [[TMP30]], [[N]]
; VF-FOUR-CHECK-NEXT:    [[TMP39:%.*]] = add i32 [[TMP31]], [[N]]
; VF-FOUR-CHECK-NEXT:    [[TMP40:%.*]] = sext i32 [[TMP32]] to i64
; VF-FOUR-CHECK-NEXT:    [[TMP41:%.*]] = sext i32 [[TMP33]] to i64
; VF-FOUR-CHECK-NEXT:    [[TMP42:%.*]] = sext i32 [[TMP34]] to i64
; VF-FOUR-CHECK-NEXT:    [[TMP43:%.*]] = sext i32 [[TMP35]] to i64
; VF-FOUR-CHECK-NEXT:    [[TMP44:%.*]] = sext i32 [[TMP36]] to i64
; VF-FOUR-CHECK-NEXT:    [[TMP45:%.*]] = sext i32 [[TMP37]] to i64
; VF-FOUR-CHECK-NEXT:    [[TMP46:%.*]] = sext i32 [[TMP38]] to i64
; VF-FOUR-CHECK-NEXT:    [[TMP47:%.*]] = sext i32 [[TMP39]] to i64
; VF-FOUR-CHECK-NEXT:    [[TMP48:%.*]] = getelementptr inbounds float, float* [[B:%.*]], i64 [[TMP40]]
; VF-FOUR-CHECK-NEXT:    [[TMP49:%.*]] = getelementptr inbounds float, float* [[B]], i64 [[TMP41]]
; VF-FOUR-CHECK-NEXT:    [[TMP50:%.*]] = getelementptr inbounds float, float* [[B]], i64 [[TMP42]]
; VF-FOUR-CHECK-NEXT:    [[TMP51:%.*]] = getelementptr inbounds float, float* [[B]], i64 [[TMP43]]
; VF-FOUR-CHECK-NEXT:    [[TMP52:%.*]] = getelementptr inbounds float, float* [[B]], i64 [[TMP44]]
; VF-FOUR-CHECK-NEXT:    [[TMP53:%.*]] = getelementptr inbounds float, float* [[B]], i64 [[TMP45]]
; VF-FOUR-CHECK-NEXT:    [[TMP54:%.*]] = getelementptr inbounds float, float* [[B]], i64 [[TMP46]]
; VF-FOUR-CHECK-NEXT:    [[TMP55:%.*]] = getelementptr inbounds float, float* [[B]], i64 [[TMP47]]
; VF-FOUR-CHECK-NEXT:    [[TMP56:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 0
; VF-FOUR-CHECK-NEXT:    [[TMP57:%.*]] = getelementptr inbounds float, float* [[TMP56]], i32 -3
; VF-FOUR-CHECK-NEXT:    [[TMP58:%.*]] = bitcast float* [[TMP57]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <4 x float>, <4 x float>* [[TMP58]], align 4
; VF-FOUR-CHECK-NEXT:    [[REVERSE:%.*]] = shufflevector <4 x float> [[WIDE_LOAD]], <4 x float> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; VF-FOUR-CHECK-NEXT:    [[TMP59:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 -4
; VF-FOUR-CHECK-NEXT:    [[TMP60:%.*]] = getelementptr inbounds float, float* [[TMP59]], i32 -3
; VF-FOUR-CHECK-NEXT:    [[TMP61:%.*]] = bitcast float* [[TMP60]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD2:%.*]] = load <4 x float>, <4 x float>* [[TMP61]], align 4
; VF-FOUR-CHECK-NEXT:    [[REVERSE3:%.*]] = shufflevector <4 x float> [[WIDE_LOAD2]], <4 x float> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; VF-FOUR-CHECK-NEXT:    [[TMP62:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 -8
; VF-FOUR-CHECK-NEXT:    [[TMP63:%.*]] = getelementptr inbounds float, float* [[TMP62]], i32 -3
; VF-FOUR-CHECK-NEXT:    [[TMP64:%.*]] = bitcast float* [[TMP63]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD4:%.*]] = load <4 x float>, <4 x float>* [[TMP64]], align 4
; VF-FOUR-CHECK-NEXT:    [[REVERSE5:%.*]] = shufflevector <4 x float> [[WIDE_LOAD4]], <4 x float> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; VF-FOUR-CHECK-NEXT:    [[TMP65:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 -12
; VF-FOUR-CHECK-NEXT:    [[TMP66:%.*]] = getelementptr inbounds float, float* [[TMP65]], i32 -3
; VF-FOUR-CHECK-NEXT:    [[TMP67:%.*]] = bitcast float* [[TMP66]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD6:%.*]] = load <4 x float>, <4 x float>* [[TMP67]], align 4
; VF-FOUR-CHECK-NEXT:    [[REVERSE7:%.*]] = shufflevector <4 x float> [[WIDE_LOAD6]], <4 x float> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; VF-FOUR-CHECK-NEXT:    [[TMP68:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 -16
; VF-FOUR-CHECK-NEXT:    [[TMP69:%.*]] = getelementptr inbounds float, float* [[TMP68]], i32 -3
; VF-FOUR-CHECK-NEXT:    [[TMP70:%.*]] = bitcast float* [[TMP69]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD8:%.*]] = load <4 x float>, <4 x float>* [[TMP70]], align 4
; VF-FOUR-CHECK-NEXT:    [[REVERSE9:%.*]] = shufflevector <4 x float> [[WIDE_LOAD8]], <4 x float> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; VF-FOUR-CHECK-NEXT:    [[TMP71:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 -20
; VF-FOUR-CHECK-NEXT:    [[TMP72:%.*]] = getelementptr inbounds float, float* [[TMP71]], i32 -3
; VF-FOUR-CHECK-NEXT:    [[TMP73:%.*]] = bitcast float* [[TMP72]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD10:%.*]] = load <4 x float>, <4 x float>* [[TMP73]], align 4
; VF-FOUR-CHECK-NEXT:    [[REVERSE11:%.*]] = shufflevector <4 x float> [[WIDE_LOAD10]], <4 x float> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; VF-FOUR-CHECK-NEXT:    [[TMP74:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 -24
; VF-FOUR-CHECK-NEXT:    [[TMP75:%.*]] = getelementptr inbounds float, float* [[TMP74]], i32 -3
; VF-FOUR-CHECK-NEXT:    [[TMP76:%.*]] = bitcast float* [[TMP75]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD12:%.*]] = load <4 x float>, <4 x float>* [[TMP76]], align 4
; VF-FOUR-CHECK-NEXT:    [[REVERSE13:%.*]] = shufflevector <4 x float> [[WIDE_LOAD12]], <4 x float> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; VF-FOUR-CHECK-NEXT:    [[TMP77:%.*]] = getelementptr inbounds float, float* [[TMP48]], i32 -28
; VF-FOUR-CHECK-NEXT:    [[TMP78:%.*]] = getelementptr inbounds float, float* [[TMP77]], i32 -3
; VF-FOUR-CHECK-NEXT:    [[TMP79:%.*]] = bitcast float* [[TMP78]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD14:%.*]] = load <4 x float>, <4 x float>* [[TMP79]], align 4
; VF-FOUR-CHECK-NEXT:    [[REVERSE15:%.*]] = shufflevector <4 x float> [[WIDE_LOAD14]], <4 x float> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; VF-FOUR-CHECK-NEXT:    [[TMP80:%.*]] = fadd fast <4 x float> [[REVERSE]], <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
; VF-FOUR-CHECK-NEXT:    [[TMP81:%.*]] = fadd fast <4 x float> [[REVERSE3]], <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
; VF-FOUR-CHECK-NEXT:    [[TMP82:%.*]] = fadd fast <4 x float> [[REVERSE5]], <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
; VF-FOUR-CHECK-NEXT:    [[TMP83:%.*]] = fadd fast <4 x float> [[REVERSE7]], <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
; VF-FOUR-CHECK-NEXT:    [[TMP84:%.*]] = fadd fast <4 x float> [[REVERSE9]], <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
; VF-FOUR-CHECK-NEXT:    [[TMP85:%.*]] = fadd fast <4 x float> [[REVERSE11]], <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
; VF-FOUR-CHECK-NEXT:    [[TMP86:%.*]] = fadd fast <4 x float> [[REVERSE13]], <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
; VF-FOUR-CHECK-NEXT:    [[TMP87:%.*]] = fadd fast <4 x float> [[REVERSE15]], <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
; VF-FOUR-CHECK-NEXT:    [[TMP88:%.*]] = getelementptr inbounds float, float* [[A:%.*]], i64 [[TMP16]]
; VF-FOUR-CHECK-NEXT:    [[TMP89:%.*]] = getelementptr inbounds float, float* [[A]], i64 [[TMP17]]
; VF-FOUR-CHECK-NEXT:    [[TMP90:%.*]] = getelementptr inbounds float, float* [[A]], i64 [[TMP18]]
; VF-FOUR-CHECK-NEXT:    [[TMP91:%.*]] = getelementptr inbounds float, float* [[A]], i64 [[TMP19]]
; VF-FOUR-CHECK-NEXT:    [[TMP92:%.*]] = getelementptr inbounds float, float* [[A]], i64 [[TMP20]]
; VF-FOUR-CHECK-NEXT:    [[TMP93:%.*]] = getelementptr inbounds float, float* [[A]], i64 [[TMP21]]
; VF-FOUR-CHECK-NEXT:    [[TMP94:%.*]] = getelementptr inbounds float, float* [[A]], i64 [[TMP22]]
; VF-FOUR-CHECK-NEXT:    [[TMP95:%.*]] = getelementptr inbounds float, float* [[A]], i64 [[TMP23]]
; VF-FOUR-CHECK-NEXT:    [[TMP96:%.*]] = getelementptr inbounds float, float* [[TMP88]], i32 0
; VF-FOUR-CHECK-NEXT:    [[TMP97:%.*]] = bitcast float* [[TMP96]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    store <4 x float> [[TMP80]], <4 x float>* [[TMP97]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP98:%.*]] = getelementptr inbounds float, float* [[TMP88]], i32 4
; VF-FOUR-CHECK-NEXT:    [[TMP99:%.*]] = bitcast float* [[TMP98]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    store <4 x float> [[TMP81]], <4 x float>* [[TMP99]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP100:%.*]] = getelementptr inbounds float, float* [[TMP88]], i32 8
; VF-FOUR-CHECK-NEXT:    [[TMP101:%.*]] = bitcast float* [[TMP100]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    store <4 x float> [[TMP82]], <4 x float>* [[TMP101]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP102:%.*]] = getelementptr inbounds float, float* [[TMP88]], i32 12
; VF-FOUR-CHECK-NEXT:    [[TMP103:%.*]] = bitcast float* [[TMP102]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    store <4 x float> [[TMP83]], <4 x float>* [[TMP103]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP104:%.*]] = getelementptr inbounds float, float* [[TMP88]], i32 16
; VF-FOUR-CHECK-NEXT:    [[TMP105:%.*]] = bitcast float* [[TMP104]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    store <4 x float> [[TMP84]], <4 x float>* [[TMP105]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP106:%.*]] = getelementptr inbounds float, float* [[TMP88]], i32 20
; VF-FOUR-CHECK-NEXT:    [[TMP107:%.*]] = bitcast float* [[TMP106]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    store <4 x float> [[TMP85]], <4 x float>* [[TMP107]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP108:%.*]] = getelementptr inbounds float, float* [[TMP88]], i32 24
; VF-FOUR-CHECK-NEXT:    [[TMP109:%.*]] = bitcast float* [[TMP108]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    store <4 x float> [[TMP86]], <4 x float>* [[TMP109]], align 4
; VF-FOUR-CHECK-NEXT:    [[TMP110:%.*]] = getelementptr inbounds float, float* [[TMP88]], i32 28
; VF-FOUR-CHECK-NEXT:    [[TMP111:%.*]] = bitcast float* [[TMP110]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    store <4 x float> [[TMP87]], <4 x float>* [[TMP111]], align 4
; VF-FOUR-CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 32
; VF-FOUR-CHECK-NEXT:    [[TMP112:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; VF-FOUR-CHECK-NEXT:    br i1 [[TMP112]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOPID_MV_CM:![0-9]+]]
; VF-FOUR-CHECK:       middle.block:
; VF-FOUR-CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 [[WIDE_TRIP_COUNT]], [[N_VEC]]
; VF-FOUR-CHECK-NEXT:    br i1 [[CMP_N]], label [[FOR_END_LOOPEXIT:%.*]], label [[VEC_EPILOG_ITER_CHECK:%.*]]
; VF-FOUR-CHECK:       vec.epilog.iter.check:
; VF-FOUR-CHECK-NEXT:    [[IND_END19:%.*]] = trunc i64 [[N_VEC]] to i32
; VF-FOUR-CHECK-NEXT:    [[N_VEC_REMAINING:%.*]] = sub i64 [[WIDE_TRIP_COUNT]], [[N_VEC]]
; VF-FOUR-CHECK-NEXT:    [[MIN_EPILOG_ITERS_CHECK:%.*]] = icmp ult i64 [[N_VEC_REMAINING]], 4
; VF-FOUR-CHECK-NEXT:    br i1 [[MIN_EPILOG_ITERS_CHECK]], label [[VEC_EPILOG_SCALAR_PH]], label [[VEC_EPILOG_PH]]
; VF-FOUR-CHECK:       vec.epilog.ph:
; VF-FOUR-CHECK-NEXT:    [[VEC_EPILOG_RESUME_VAL:%.*]] = phi i64 [ [[N_VEC]], [[VEC_EPILOG_ITER_CHECK]] ], [ 0, [[VECTOR_MAIN_LOOP_ITER_CHECK]] ]
; VF-FOUR-CHECK-NEXT:    [[N_MOD_VF16:%.*]] = urem i64 [[WIDE_TRIP_COUNT]], 4
; VF-FOUR-CHECK-NEXT:    [[N_VEC17:%.*]] = sub i64 [[WIDE_TRIP_COUNT]], [[N_MOD_VF16]]
; VF-FOUR-CHECK-NEXT:    [[IND_END:%.*]] = trunc i64 [[N_VEC17]] to i32
; VF-FOUR-CHECK-NEXT:    br label [[VEC_EPILOG_VECTOR_BODY:%.*]]
; VF-FOUR-CHECK:       vec.epilog.vector.body:
; VF-FOUR-CHECK-NEXT:    [[OFFSET_IDX23:%.*]] = phi i64 [ [[VEC_EPILOG_RESUME_VAL]], [[VEC_EPILOG_PH]] ], [ [[INDEX_NEXT26:%.*]], [[VEC_EPILOG_VECTOR_BODY]] ]
; VF-FOUR-CHECK-NEXT:    [[OFFSET_IDX22:%.*]] = trunc i64 [[OFFSET_IDX23]] to i32
; VF-FOUR-CHECK-NEXT:    [[TMP113:%.*]] = add i32 [[OFFSET_IDX22]], 0
; VF-FOUR-CHECK-NEXT:    [[TMP114:%.*]] = add i64 [[OFFSET_IDX23]], 0
; VF-FOUR-CHECK-NEXT:    [[TMP115:%.*]] = xor i32 [[TMP113]], -1
; VF-FOUR-CHECK-NEXT:    [[TMP116:%.*]] = add i32 [[TMP115]], [[N]]
; VF-FOUR-CHECK-NEXT:    [[TMP117:%.*]] = sext i32 [[TMP116]] to i64
; VF-FOUR-CHECK-NEXT:    [[TMP118:%.*]] = getelementptr inbounds float, float* [[B]], i64 [[TMP117]]
; VF-FOUR-CHECK-NEXT:    [[TMP119:%.*]] = getelementptr inbounds float, float* [[TMP118]], i32 0
; VF-FOUR-CHECK-NEXT:    [[TMP120:%.*]] = getelementptr inbounds float, float* [[TMP119]], i32 -3
; VF-FOUR-CHECK-NEXT:    [[TMP121:%.*]] = bitcast float* [[TMP120]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    [[WIDE_LOAD24:%.*]] = load <4 x float>, <4 x float>* [[TMP121]], align 4
; VF-FOUR-CHECK-NEXT:    [[REVERSE25:%.*]] = shufflevector <4 x float> [[WIDE_LOAD24]], <4 x float> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; VF-FOUR-CHECK-NEXT:    [[TMP122:%.*]] = fadd fast <4 x float> [[REVERSE25]], <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
; VF-FOUR-CHECK-NEXT:    [[TMP123:%.*]] = getelementptr inbounds float, float* [[A]], i64 [[TMP114]]
; VF-FOUR-CHECK-NEXT:    [[TMP124:%.*]] = getelementptr inbounds float, float* [[TMP123]], i32 0
; VF-FOUR-CHECK-NEXT:    [[TMP125:%.*]] = bitcast float* [[TMP124]] to <4 x float>*
; VF-FOUR-CHECK-NEXT:    store <4 x float> [[TMP122]], <4 x float>* [[TMP125]], align 4
; VF-FOUR-CHECK-NEXT:    [[INDEX_NEXT26]] = add nuw i64 [[OFFSET_IDX23]], 4
; VF-FOUR-CHECK-NEXT:    [[TMP126:%.*]] = icmp eq i64 [[INDEX_NEXT26]], [[N_VEC17]]
; VF-FOUR-CHECK-NEXT:    br i1 [[TMP126]], label [[VEC_EPILOG_MIDDLE_BLOCK:%.*]], label [[VEC_EPILOG_VECTOR_BODY]], !llvm.loop [[LOOPID_EV_CM:![0-9]+]]
; VF-FOUR-CHECK:       vec.epilog.middle.block:
; VF-FOUR-CHECK-NEXT:    [[CMP_N20:%.*]] = icmp eq i64 [[WIDE_TRIP_COUNT]], [[N_VEC17]]
; VF-FOUR-CHECK-NEXT:    br i1 [[CMP_N20]], label [[FOR_END_LOOPEXIT_LOOPEXIT:%.*]], label [[VEC_EPILOG_SCALAR_PH]]
; VF-FOUR-CHECK:       vec.epilog.scalar.ph:
; VF-FOUR-CHECK-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ [[N_VEC17]], [[VEC_EPILOG_MIDDLE_BLOCK]] ], [ [[N_VEC]], [[VEC_EPILOG_ITER_CHECK]] ], [ 0, [[VECTOR_SCEVCHECK]] ], [ 0, [[ITER_CHECK]] ]
; VF-FOUR-CHECK-NEXT:    [[BC_RESUME_VAL18:%.*]] = phi i32 [ [[IND_END]], [[VEC_EPILOG_MIDDLE_BLOCK]] ], [ [[IND_END19]], [[VEC_EPILOG_ITER_CHECK]] ], [ 0, [[VECTOR_SCEVCHECK]] ], [ 0, [[ITER_CHECK]] ]
; VF-FOUR-CHECK-NEXT:    br label [[FOR_BODY:%.*]]
; VF-FOUR-CHECK:       for.body:
; VF-FOUR-CHECK-NEXT:    [[INDVARS_IV:%.*]] = phi i64 [ [[BC_RESUME_VAL]], [[VEC_EPILOG_SCALAR_PH]] ], [ [[INDVARS_IV_NEXT:%.*]], [[FOR_BODY]] ]
; VF-FOUR-CHECK-NEXT:    [[I_014:%.*]] = phi i32 [ [[BC_RESUME_VAL18]], [[VEC_EPILOG_SCALAR_PH]] ], [ [[INC:%.*]], [[FOR_BODY]] ]
; VF-FOUR-CHECK-NEXT:    [[TMP127:%.*]] = xor i32 [[I_014]], -1
; VF-FOUR-CHECK-NEXT:    [[SUB2:%.*]] = add i32 [[TMP127]], [[N]]
; VF-FOUR-CHECK-NEXT:    [[IDXPROM:%.*]] = sext i32 [[SUB2]] to i64
; VF-FOUR-CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds float, float* [[B]], i64 [[IDXPROM]]
; VF-FOUR-CHECK-NEXT:    [[TMP128:%.*]] = load float, float* [[ARRAYIDX]], align 4
; VF-FOUR-CHECK-NEXT:    [[CONV3:%.*]] = fadd fast float [[TMP128]], 1.000000e+00
; VF-FOUR-CHECK-NEXT:    [[ARRAYIDX5:%.*]] = getelementptr inbounds float, float* [[A]], i64 [[INDVARS_IV]]
; VF-FOUR-CHECK-NEXT:    store float [[CONV3]], float* [[ARRAYIDX5]], align 4
; VF-FOUR-CHECK-NEXT:    [[INDVARS_IV_NEXT]] = add nuw nsw i64 [[INDVARS_IV]], 1
; VF-FOUR-CHECK-NEXT:    [[INC]] = add nuw nsw i32 [[I_014]], 1
; VF-FOUR-CHECK-NEXT:    [[EXITCOND:%.*]] = icmp ne i64 [[INDVARS_IV_NEXT]], [[WIDE_TRIP_COUNT]]
; VF-FOUR-CHECK-NEXT:    br i1 [[EXITCOND]], label [[FOR_BODY]], label [[FOR_END_LOOPEXIT_LOOPEXIT]], !llvm.loop [[LOOP7:![0-9]+]]
; VF-FOUR-CHECK:       for.end.loopexit.loopexit:
; VF-FOUR-CHECK-NEXT:    br label [[FOR_END_LOOPEXIT]]
; VF-FOUR-CHECK:       for.end.loopexit:
; VF-FOUR-CHECK-NEXT:    br label [[FOR_END]]
; VF-FOUR-CHECK:       for.end:
; VF-FOUR-CHECK-NEXT:    ret i32 0
;
entry:
  %cmp1 = icmp sgt i32 %n, 1
  br i1 %cmp1, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  %0 = add i32 %n, -1
  %wide.trip.count = zext i32 %0 to i64
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %i.014 = phi i32 [ 0, %for.body.preheader ], [ %inc, %for.body ]
  %1 = xor i32 %i.014, -1
  %sub2 = add i32 %1, %n
  %idxprom = sext i32 %sub2 to i64
  %arrayidx = getelementptr inbounds float, float* %B, i64 %idxprom
  %2 = load float, float* %arrayidx, align 4
  %conv3 = fadd fast float %2, 1.000000e+00
  %arrayidx5 = getelementptr inbounds float, float* %A, i64 %indvars.iv
  store float %conv3, float* %arrayidx5, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %inc = add nuw nsw i32 %i.014, 1
  %exitcond = icmp ne i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret i32 0
}

; VF-TWO-CHECK-DAG: [[LOOPID_MV]] = distinct !{[[LOOPID_MV]], [[LOOPID_DISABLE_VECT:!.*]]}
; VF-TWO-CHECK-DAG: [[LOOPID_EV]] = distinct !{[[LOOPID_EV]], [[LOOPID_DISABLE_VECT]], [[LOOPID_DISABLE_UNROLL:!.*]]}
; VF-TWO-CHECK-DAG: [[LOOPID_DISABLE_VECT]] = [[DISABLE_VECT_STR:!{!"llvm.loop.isvectorized".*}.*]]
; VF-TWO-CHECK-DAG: [[LOOPID_DISABLE_UNROLL]] = [[DISABLE_UNROLL_STR:!{!"llvm.loop.unroll.runtime.disable"}.*]]
;
; VF-FOUR-CHECK-DAG: [[LOOPID_MV_CM]] = distinct !{[[LOOPID_MV_CM]], [[LOOPID_DISABLE_VECT_CM:!.*]]}
; VF-FOUR-CHECK-DAG: [[LOOPID_EV_CM]] = distinct !{[[LOOPID_EV_CM]], [[LOOPID_DISABLE_VECT_CM]], [[LOOPID_DISABLE_UNROLL_CM:!.*]]}
; VF-FOUR-CHECK-DAG: [[LOOPID_DISABLE_VECT_CM]] = [[DISABLE_VECT_STR_CM:!{!"llvm.loop.isvectorized".*}.*]]
; VF-FOUR-CHECK-DAG: [[LOOPID_DISABLE_UNROLL_CM]] = [[DISABLE_UNROLL_STR_CM:!{!"llvm.loop.unroll.runtime.disable"}.*]]
;
attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="ppc64le" "target-features"="+altivec,+bpermd,+crypto,+direct-move,+extdiv,+htm,+power8-vector,+vsx,-power9-vector,-spe" "unsafe-fp-math"="true" "use-soft-float"="false" }
