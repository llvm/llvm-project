; Check that Tapir lowering to the Cilk runtime correctly updates PHI
; nodes in detach-continuation blocks.
;
; RUN: opt < %s -tapir2target -tapir-target=cilk -S | FileCheck %s
; RUN: opt < %s -passes=tapir2target -tapir-target=cilk -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.InternalNode = type { [4 x %struct.InternalNode*] }

@.str.1 = private unnamed_addr constant [11 x i8] c"cholesky.c\00", align 1
@.str.5 = private unnamed_addr constant [23 x i8] c"a != NULL && b != NULL\00", align 1
@__PRETTY_FUNCTION__.mul_and_subT = private unnamed_addr constant [54 x i8] c"Matrix mul_and_subT(int, int, Matrix, Matrix, Matrix)\00", align 1
@.str.6 = private unnamed_addr constant [46 x i8] c"r->child[_00] == NULL || r->child[_00] == r00\00", align 1
@.str.7 = private unnamed_addr constant [46 x i8] c"r->child[_01] == NULL || r->child[_01] == r01\00", align 1
@.str.8 = private unnamed_addr constant [46 x i8] c"r->child[_10] == NULL || r->child[_10] == r10\00", align 1
@.str.9 = private unnamed_addr constant [46 x i8] c"r->child[_11] == NULL || r->child[_11] == r11\00", align 1
@str.46 = private unnamed_addr constant [15 x i8] c"out of memory!\00"

; Function Attrs: nounwind uwtable
define dso_local %struct.InternalNode* @mul_and_subT(i32 %depth, i32 %lower, %struct.InternalNode* readonly %a, %struct.InternalNode* readonly %b, %struct.InternalNode* %r) local_unnamed_addr #3 {
entry:
  %r00.sroa.0 = alloca i64, align 8
  %r01.sroa.0 = alloca i64, align 8
  %r10.sroa.0 = alloca i64, align 8
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp = icmp ne %struct.InternalNode* %a, null
  %cmp1 = icmp ne %struct.InternalNode* %b, null
  %or.cond = and i1 %cmp, %cmp1
  br i1 %or.cond, label %if.end, label %if.else

if.else:                                          ; preds = %entry
  tail call void @__assert_fail(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.5, i64 0, i64 0), i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.1, i64 0, i64 0), i32 559, i8* getelementptr inbounds ([54 x i8], [54 x i8]* @__PRETTY_FUNCTION__.mul_and_subT, i64 0, i64 0)) #10
  unreachable

if.end:                                           ; preds = %entry
  %cmp2 = icmp eq i32 %depth, 2
  br i1 %cmp2, label %if.then3, label %if.end23

if.then3:                                         ; preds = %if.end
  %cmp4 = icmp eq %struct.InternalNode* %r, null
  br i1 %cmp4, label %if.then5, label %if.end7

if.then5:                                         ; preds = %if.then3
  %call.i = tail call noalias i8* @malloc(i64 128) #9
  %cmp.i = icmp eq i8* %call.i, null
  br i1 %cmp.i, label %if.then.i, label %new_block_leaf.exit

if.then.i:                                        ; preds = %if.then5
  %puts.i = tail call i32 @puts(i8* getelementptr inbounds ([15 x i8], [15 x i8]* @str.46, i64 0, i64 0)) #9
  tail call void @exit(i32 1) #10
  unreachable

new_block_leaf.exit:                              ; preds = %if.then5
  %0 = bitcast i8* %call.i to %struct.InternalNode*
  tail call void @llvm.memset.p0i8.i64(i8* nonnull align 8 %call.i, i8 0, i64 128, i1 false) #9
  br label %if.end7

if.end7:                                          ; preds = %if.then3, %new_block_leaf.exit
  %r.addr.0 = phi %struct.InternalNode* [ %0, %new_block_leaf.exit ], [ %r, %if.then3 ]
  %tobool = icmp eq i32 %lower, 0
  %arraydecay10 = bitcast %struct.InternalNode* %r.addr.0 to [4 x double]*
  %arraydecay12 = bitcast %struct.InternalNode* %a to [4 x double]*
  %arrayidx12.i336 = bitcast %struct.InternalNode* %b to double*
  %arrayidx12.1.i337 = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %b, i64 0, i32 0, i64 1
  %1 = bitcast %struct.InternalNode** %arrayidx12.1.i337 to double*
  %arrayidx12.2.i338 = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %b, i64 0, i32 0, i64 2
  %2 = bitcast %struct.InternalNode** %arrayidx12.2.i338 to double*
  %arrayidx12.3.i339 = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %b, i64 0, i32 0, i64 3
  %3 = bitcast %struct.InternalNode** %arrayidx12.3.i339 to double*
  %arrayidx12.138.i = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %b, i64 1
  %4 = bitcast %struct.InternalNode* %arrayidx12.138.i to double*
  %arrayidx12.1.1.i = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %b, i64 1, i32 0, i64 1
  %5 = bitcast %struct.InternalNode** %arrayidx12.1.1.i to double*
  %arrayidx12.2.1.i = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %b, i64 1, i32 0, i64 2
  %6 = bitcast %struct.InternalNode** %arrayidx12.2.1.i to double*
  %arrayidx12.3.1.i = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %b, i64 1, i32 0, i64 3
  %7 = bitcast %struct.InternalNode** %arrayidx12.3.1.i to double*
  %arrayidx12.241.i = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %b, i64 2
  %8 = bitcast %struct.InternalNode* %arrayidx12.241.i to double*
  %arrayidx12.1.2.i = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %b, i64 2, i32 0, i64 1
  %9 = bitcast %struct.InternalNode** %arrayidx12.1.2.i to double*
  %arrayidx12.2.2.i = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %b, i64 2, i32 0, i64 2
  %10 = bitcast %struct.InternalNode** %arrayidx12.2.2.i to double*
  %arrayidx12.3.2.i = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %b, i64 2, i32 0, i64 3
  %11 = bitcast %struct.InternalNode** %arrayidx12.3.2.i to double*
  %arrayidx12.344.i = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %b, i64 3
  %12 = bitcast %struct.InternalNode* %arrayidx12.344.i to double*
  %arrayidx12.1.3.i = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %b, i64 3, i32 0, i64 1
  %13 = bitcast %struct.InternalNode** %arrayidx12.1.3.i to double*
  %arrayidx12.2.3.i = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %b, i64 3, i32 0, i64 2
  %14 = bitcast %struct.InternalNode** %arrayidx12.2.3.i to double*
  %arrayidx12.3.3.i = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %b, i64 3, i32 0, i64 3
  %15 = bitcast %struct.InternalNode** %arrayidx12.3.3.i to double*
  br i1 %tobool, label %for.body6.3.i, label %for.cond1.preheader.i

for.cond1.preheader.i:                            ; preds = %if.end7, %for.inc20.i
  %indvars.iv43.i = phi i64 [ %indvars.iv.next44.i, %for.inc20.i ], [ 0, %if.end7 ]
  %indvars.iv41.i = phi i64 [ %indvars.iv.next42.i, %for.inc20.i ], [ 1, %if.end7 ]
  %arrayidx8.i = getelementptr inbounds [4 x double], [4 x double]* %arraydecay12, i64 %indvars.iv43.i, i64 0
  %arrayidx8.1.i = getelementptr inbounds [4 x double], [4 x double]* %arraydecay12, i64 %indvars.iv43.i, i64 1
  %arrayidx8.2.i = getelementptr inbounds [4 x double], [4 x double]* %arraydecay12, i64 %indvars.iv43.i, i64 2
  %arrayidx8.3.i = getelementptr inbounds [4 x double], [4 x double]* %arraydecay12, i64 %indvars.iv43.i, i64 3
  %arrayidx16.i = getelementptr inbounds [4 x double], [4 x double]* %arraydecay10, i64 %indvars.iv43.i, i64 0
  %16 = load double, double* %arrayidx8.i, align 8, !tbaa !11
  %17 = load double, double* %arrayidx12.i336, align 8, !tbaa !11
  %mul.i = fmul double %16, %17
  %18 = load double, double* %arrayidx16.i, align 8, !tbaa !11
  %sub.i = fsub double %18, %mul.i
  store double %sub.i, double* %arrayidx16.i, align 8, !tbaa !11
  %19 = load double, double* %arrayidx8.1.i, align 8, !tbaa !11
  %20 = load double, double* %1, align 8, !tbaa !11
  %mul.1.i = fmul double %19, %20
  %sub.1.i = fsub double %sub.i, %mul.1.i
  store double %sub.1.i, double* %arrayidx16.i, align 8, !tbaa !11
  %21 = load double, double* %arrayidx8.2.i, align 8, !tbaa !11
  %22 = load double, double* %2, align 8, !tbaa !11
  %mul.2.i = fmul double %21, %22
  %sub.2.i = fsub double %sub.1.i, %mul.2.i
  store double %sub.2.i, double* %arrayidx16.i, align 8, !tbaa !11
  %23 = load double, double* %arrayidx8.3.i, align 8, !tbaa !11
  %24 = load double, double* %3, align 8, !tbaa !11
  %mul.3.i = fmul double %23, %24
  %sub.3.i = fsub double %sub.2.i, %mul.3.i
  store double %sub.3.i, double* %arrayidx16.i, align 8, !tbaa !11
  %exitcond.i = icmp eq i64 %indvars.iv41.i, 1
  br i1 %exitcond.i, label %for.inc20.i, label %for.body6.i.1

for.inc20.i:                                      ; preds = %for.body6.i.3, %for.body6.i.2, %for.body6.i.1, %for.cond1.preheader.i
  %indvars.iv.next44.i = add nuw nsw i64 %indvars.iv43.i, 1
  %indvars.iv.next42.i = add nuw nsw i64 %indvars.iv41.i, 1
  %exitcond45.i = icmp eq i64 %indvars.iv.next44.i, 4
  br i1 %exitcond45.i, label %cleanup, label %for.cond1.preheader.i

for.body6.3.i:                                    ; preds = %if.end7, %for.body6.3.i
  %indvars.iv.i340 = phi i64 [ %indvars.iv.next.i354, %for.body6.3.i ], [ 0, %if.end7 ]
  %arrayidx8.i341 = getelementptr inbounds [4 x double], [4 x double]* %arraydecay12, i64 %indvars.iv.i340, i64 0
  %arrayidx16.i342 = getelementptr inbounds [4 x double], [4 x double]* %arraydecay10, i64 %indvars.iv.i340, i64 0
  %25 = load double, double* %arrayidx8.i341, align 8, !tbaa !11
  %26 = load double, double* %arrayidx12.i336, align 8, !tbaa !11
  %mul.i343 = fmul double %25, %26
  %27 = load double, double* %arrayidx16.i342, align 8, !tbaa !11
  %sub.i344 = fsub double %27, %mul.i343
  store double %sub.i344, double* %arrayidx16.i342, align 8, !tbaa !11
  %arrayidx8.1.i345 = getelementptr inbounds [4 x double], [4 x double]* %arraydecay12, i64 %indvars.iv.i340, i64 1
  %28 = load double, double* %arrayidx8.1.i345, align 8, !tbaa !11
  %29 = load double, double* %1, align 8, !tbaa !11
  %mul.1.i346 = fmul double %28, %29
  %sub.1.i347 = fsub double %sub.i344, %mul.1.i346
  store double %sub.1.i347, double* %arrayidx16.i342, align 8, !tbaa !11
  %arrayidx8.2.i348 = getelementptr inbounds [4 x double], [4 x double]* %arraydecay12, i64 %indvars.iv.i340, i64 2
  %30 = load double, double* %arrayidx8.2.i348, align 8, !tbaa !11
  %31 = load double, double* %2, align 8, !tbaa !11
  %mul.2.i349 = fmul double %30, %31
  %sub.2.i350 = fsub double %sub.1.i347, %mul.2.i349
  store double %sub.2.i350, double* %arrayidx16.i342, align 8, !tbaa !11
  %arrayidx8.3.i351 = getelementptr inbounds [4 x double], [4 x double]* %arraydecay12, i64 %indvars.iv.i340, i64 3
  %32 = load double, double* %arrayidx8.3.i351, align 8, !tbaa !11
  %33 = load double, double* %3, align 8, !tbaa !11
  %mul.3.i352 = fmul double %32, %33
  %sub.3.i353 = fsub double %sub.2.i350, %mul.3.i352
  store double %sub.3.i353, double* %arrayidx16.i342, align 8, !tbaa !11
  %arrayidx16.1.i = getelementptr inbounds [4 x double], [4 x double]* %arraydecay10, i64 %indvars.iv.i340, i64 1
  %34 = load double, double* %arrayidx8.i341, align 8, !tbaa !11
  %35 = load double, double* %4, align 8, !tbaa !11
  %mul.139.i = fmul double %34, %35
  %36 = load double, double* %arrayidx16.1.i, align 8, !tbaa !11
  %sub.140.i = fsub double %36, %mul.139.i
  store double %sub.140.i, double* %arrayidx16.1.i, align 8, !tbaa !11
  %37 = load double, double* %arrayidx8.1.i345, align 8, !tbaa !11
  %38 = load double, double* %5, align 8, !tbaa !11
  %mul.1.1.i = fmul double %37, %38
  %sub.1.1.i = fsub double %sub.140.i, %mul.1.1.i
  store double %sub.1.1.i, double* %arrayidx16.1.i, align 8, !tbaa !11
  %39 = load double, double* %arrayidx8.2.i348, align 8, !tbaa !11
  %40 = load double, double* %6, align 8, !tbaa !11
  %mul.2.1.i = fmul double %39, %40
  %sub.2.1.i = fsub double %sub.1.1.i, %mul.2.1.i
  store double %sub.2.1.i, double* %arrayidx16.1.i, align 8, !tbaa !11
  %41 = load double, double* %arrayidx8.3.i351, align 8, !tbaa !11
  %42 = load double, double* %7, align 8, !tbaa !11
  %mul.3.1.i = fmul double %41, %42
  %sub.3.1.i = fsub double %sub.2.1.i, %mul.3.1.i
  store double %sub.3.1.i, double* %arrayidx16.1.i, align 8, !tbaa !11
  %arrayidx16.2.i = getelementptr inbounds [4 x double], [4 x double]* %arraydecay10, i64 %indvars.iv.i340, i64 2
  %43 = load double, double* %arrayidx8.i341, align 8, !tbaa !11
  %44 = load double, double* %8, align 8, !tbaa !11
  %mul.242.i = fmul double %43, %44
  %45 = load double, double* %arrayidx16.2.i, align 8, !tbaa !11
  %sub.243.i = fsub double %45, %mul.242.i
  store double %sub.243.i, double* %arrayidx16.2.i, align 8, !tbaa !11
  %46 = load double, double* %arrayidx8.1.i345, align 8, !tbaa !11
  %47 = load double, double* %9, align 8, !tbaa !11
  %mul.1.2.i = fmul double %46, %47
  %sub.1.2.i = fsub double %sub.243.i, %mul.1.2.i
  store double %sub.1.2.i, double* %arrayidx16.2.i, align 8, !tbaa !11
  %48 = load double, double* %arrayidx8.2.i348, align 8, !tbaa !11
  %49 = load double, double* %10, align 8, !tbaa !11
  %mul.2.2.i = fmul double %48, %49
  %sub.2.2.i = fsub double %sub.1.2.i, %mul.2.2.i
  store double %sub.2.2.i, double* %arrayidx16.2.i, align 8, !tbaa !11
  %50 = load double, double* %arrayidx8.3.i351, align 8, !tbaa !11
  %51 = load double, double* %11, align 8, !tbaa !11
  %mul.3.2.i = fmul double %50, %51
  %sub.3.2.i = fsub double %sub.2.2.i, %mul.3.2.i
  store double %sub.3.2.i, double* %arrayidx16.2.i, align 8, !tbaa !11
  %arrayidx16.3.i = getelementptr inbounds [4 x double], [4 x double]* %arraydecay10, i64 %indvars.iv.i340, i64 3
  %52 = load double, double* %arrayidx8.i341, align 8, !tbaa !11
  %53 = load double, double* %12, align 8, !tbaa !11
  %mul.345.i = fmul double %52, %53
  %54 = load double, double* %arrayidx16.3.i, align 8, !tbaa !11
  %sub.346.i = fsub double %54, %mul.345.i
  store double %sub.346.i, double* %arrayidx16.3.i, align 8, !tbaa !11
  %55 = load double, double* %arrayidx8.1.i345, align 8, !tbaa !11
  %56 = load double, double* %13, align 8, !tbaa !11
  %mul.1.3.i = fmul double %55, %56
  %sub.1.3.i = fsub double %sub.346.i, %mul.1.3.i
  store double %sub.1.3.i, double* %arrayidx16.3.i, align 8, !tbaa !11
  %57 = load double, double* %arrayidx8.2.i348, align 8, !tbaa !11
  %58 = load double, double* %14, align 8, !tbaa !11
  %mul.2.3.i = fmul double %57, %58
  %sub.2.3.i = fsub double %sub.1.3.i, %mul.2.3.i
  store double %sub.2.3.i, double* %arrayidx16.3.i, align 8, !tbaa !11
  %59 = load double, double* %arrayidx8.3.i351, align 8, !tbaa !11
  %60 = load double, double* %15, align 8, !tbaa !11
  %mul.3.3.i = fmul double %59, %60
  %sub.3.3.i = fsub double %sub.2.3.i, %mul.3.3.i
  store double %sub.3.3.i, double* %arrayidx16.3.i, align 8, !tbaa !11
  %indvars.iv.next.i354 = add nuw nsw i64 %indvars.iv.i340, 1
  %exitcond.i355 = icmp eq i64 %indvars.iv.next.i354, 4
  br i1 %exitcond.i355, label %cleanup, label %for.body6.3.i

if.end23:                                         ; preds = %if.end
  %r00.sroa.0.0.r00.0..sroa_cast381 = bitcast i64* %r00.sroa.0 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %r00.sroa.0.0.r00.0..sroa_cast381)
  %r01.sroa.0.0.r01.0..sroa_cast373 = bitcast i64* %r01.sroa.0 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %r01.sroa.0.0.r01.0..sroa_cast373)
  %r10.sroa.0.0.r10.0..sroa_cast365 = bitcast i64* %r10.sroa.0 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %r10.sroa.0.0.r10.0..sroa_cast365)
  %dec = add nsw i32 %depth, -1
  %cmp24 = icmp eq %struct.InternalNode* %r, null
  br i1 %cmp24, label %if.else32, label %if.then25

if.then25:                                        ; preds = %if.end23
  %61 = bitcast %struct.InternalNode* %r to i64*
  %62 = load i64, i64* %61, align 8, !tbaa !9
  store i64 %62, i64* %r00.sroa.0, align 8, !tbaa !9
  %arrayidx27 = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %r, i64 0, i32 0, i64 1
  %63 = bitcast %struct.InternalNode** %arrayidx27 to i64*
  %64 = load i64, i64* %63, align 8, !tbaa !9
  store i64 %64, i64* %r01.sroa.0, align 8, !tbaa !9
  %arrayidx29 = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %r, i64 0, i32 0, i64 2
  %65 = bitcast %struct.InternalNode** %arrayidx29 to i64*
  %66 = load i64, i64* %65, align 8, !tbaa !9
  store i64 %66, i64* %r10.sroa.0, align 8, !tbaa !9
  %arrayidx31 = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %r, i64 0, i32 0, i64 3
  %67 = load %struct.InternalNode*, %struct.InternalNode** %arrayidx31, align 8, !tbaa !9
  br label %if.end33

if.else32:                                        ; preds = %if.end23
  store i64 0, i64* %r00.sroa.0, align 8, !tbaa !9
  store i64 0, i64* %r01.sroa.0, align 8, !tbaa !9
  store i64 0, i64* %r10.sroa.0, align 8, !tbaa !9
  br label %if.end33

if.end33:                                         ; preds = %if.else32, %if.then25
  %r11.0 = phi %struct.InternalNode* [ %67, %if.then25 ], [ null, %if.else32 ]
  %arrayidx35 = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %a, i64 0, i32 0, i64 0
  %68 = load %struct.InternalNode*, %struct.InternalNode** %arrayidx35, align 8, !tbaa !9
  %tobool36 = icmp eq %struct.InternalNode* %68, null
  br i1 %tobool36, label %if.end47.thread, label %land.lhs.true37

if.end47.thread:                                  ; preds = %if.end33
  %tobool48396 = icmp ne i32 %lower, 0
  br label %if.end65

land.lhs.true37:                                  ; preds = %if.end33
  %arrayidx39 = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %b, i64 0, i32 0, i64 0
  %69 = load %struct.InternalNode*, %struct.InternalNode** %arrayidx39, align 8, !tbaa !9
  %tobool40 = icmp eq %struct.InternalNode* %69, null
  br i1 %tobool40, label %if.end47, label %if.then41

if.then41:                                        ; preds = %land.lhs.true37
  %70 = bitcast i64* %r00.sroa.0 to %struct.InternalNode**
  %r00.sroa.0.0.load377385 = load %struct.InternalNode*, %struct.InternalNode** %70, align 8
  detach within %syncreg, label %det.achd, label %if.end47

det.achd:                                         ; preds = %if.then41
  %call46 = tail call %struct.InternalNode* @mul_and_subT(i32 %dec, i32 %lower, %struct.InternalNode* nonnull %68, %struct.InternalNode* nonnull %69, %struct.InternalNode* %r00.sroa.0.0.load377385)
  %71 = ptrtoint %struct.InternalNode* %call46 to i64
  store i64 %71, i64* %r00.sroa.0, align 8, !tbaa !9
  reattach within %syncreg, label %if.end47

if.end47:                                         ; preds = %land.lhs.true37, %if.then41, %det.achd
  %tobool48 = icmp ne i32 %lower, 0
  %brmerge = or i1 %tobool48, %tobool36
  br i1 %brmerge, label %if.end65, label %land.lhs.true53

land.lhs.true53:                                  ; preds = %if.end47
  %arrayidx55 = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %b, i64 0, i32 0, i64 2
  %72 = load %struct.InternalNode*, %struct.InternalNode** %arrayidx55, align 8, !tbaa !9
  %tobool56 = icmp eq %struct.InternalNode* %72, null
  br i1 %tobool56, label %if.end65, label %if.then57

if.then57:                                        ; preds = %land.lhs.true53
  %73 = bitcast i64* %r01.sroa.0 to %struct.InternalNode**
  %r01.sroa.0.0.load369386 = load %struct.InternalNode*, %struct.InternalNode** %73, align 8
  detach within %syncreg, label %det.achd62, label %if.end65

; CHECK: if.then57:
; CHECK: %[[SETJMPRET:.+]] = call i32 @llvm.eh.sjlj.setjmp
; CHECK-NEXT: %[[SETJMPCMP:.+]] = icmp eq i32 %[[SETJMPRET]], 0
; CHECK-NEXT: br i1 %[[SETJMPCMP]], label %if.then57.split, label %if.end65

; CHECK: if.then57.split:
; CHECK: br label %if.end65

det.achd62:                                       ; preds = %if.then57
  %call63 = tail call %struct.InternalNode* @mul_and_subT(i32 %dec, i32 0, %struct.InternalNode* nonnull %68, %struct.InternalNode* nonnull %72, %struct.InternalNode* %r01.sroa.0.0.load369386)
  %74 = ptrtoint %struct.InternalNode* %call63 to i64
  store i64 %74, i64* %r01.sroa.0, align 8, !tbaa !9
  reattach within %syncreg, label %if.end65

if.end65:                                         ; preds = %if.end47.thread, %if.end47, %land.lhs.true53, %if.then57, %det.achd62
  %tobool48398 = phi i1 [ %tobool48396, %if.end47.thread ], [ %tobool48, %if.end47 ], [ %tobool48, %land.lhs.true53 ], [ %tobool48, %if.then57 ], [ %tobool48, %det.achd62 ]
  %arrayidx67 = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %a, i64 0, i32 0, i64 2
  %75 = load %struct.InternalNode*, %struct.InternalNode** %arrayidx67, align 8, !tbaa !9
  %tobool68 = icmp eq %struct.InternalNode* %75, null
  br i1 %tobool68, label %if.end97, label %land.lhs.true69

; CHECK: if.end65:
; CHECK: %tobool48398 = phi i1
; CHECK-DAG: [ %tobool48, %if.then57.split ]
; CHECK-DAG: [ %tobool48, %if.then57 ]
; CHECK: %arrayidx67 = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %a, i64 0, i32 0, i64 2

land.lhs.true69:                                  ; preds = %if.end65
  %arrayidx71 = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %b, i64 0, i32 0, i64 0
  %76 = load %struct.InternalNode*, %struct.InternalNode** %arrayidx71, align 8, !tbaa !9
  %tobool72 = icmp eq %struct.InternalNode* %76, null
  br i1 %tobool72, label %land.lhs.true85, label %if.then73

if.then73:                                        ; preds = %land.lhs.true69
  %77 = bitcast i64* %r10.sroa.0 to %struct.InternalNode**
  %r10.sroa.0.0.load361387 = load %struct.InternalNode*, %struct.InternalNode** %77, align 8
  detach within %syncreg, label %det.achd78, label %land.lhs.true85

det.achd78:                                       ; preds = %if.then73
  %call79 = tail call %struct.InternalNode* @mul_and_subT(i32 %dec, i32 0, %struct.InternalNode* nonnull %75, %struct.InternalNode* nonnull %76, %struct.InternalNode* %r10.sroa.0.0.load361387)
  %78 = ptrtoint %struct.InternalNode* %call79 to i64
  store i64 %78, i64* %r10.sroa.0, align 8, !tbaa !9
  reattach within %syncreg, label %land.lhs.true85

land.lhs.true85:                                  ; preds = %land.lhs.true69, %if.then73, %det.achd78
  %arrayidx87 = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %b, i64 0, i32 0, i64 2
  %79 = load %struct.InternalNode*, %struct.InternalNode** %arrayidx87, align 8, !tbaa !9
  %tobool88 = icmp eq %struct.InternalNode* %79, null
  br i1 %tobool88, label %if.end97, label %if.then89

if.then89:                                        ; preds = %land.lhs.true85
  %call95 = tail call %struct.InternalNode* @mul_and_subT(i32 %dec, i32 %lower, %struct.InternalNode* nonnull %75, %struct.InternalNode* nonnull %79, %struct.InternalNode* %r11.0)
  br label %if.end97

if.end97:                                         ; preds = %if.end65, %land.lhs.true85, %if.then89
  %r11.1 = phi %struct.InternalNode* [ %call95, %if.then89 ], [ %r11.0, %land.lhs.true85 ], [ %r11.0, %if.end65 ]
  sync within %syncreg, label %sync.continue

sync.continue:                                    ; preds = %if.end97
  %arrayidx99 = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %a, i64 0, i32 0, i64 1
  %80 = load %struct.InternalNode*, %struct.InternalNode** %arrayidx99, align 8, !tbaa !9
  %tobool100 = icmp eq %struct.InternalNode* %80, null
  br i1 %tobool100, label %if.end131, label %land.lhs.true101

land.lhs.true101:                                 ; preds = %sync.continue
  %arrayidx103 = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %b, i64 0, i32 0, i64 1
  %81 = load %struct.InternalNode*, %struct.InternalNode** %arrayidx103, align 8, !tbaa !9
  %tobool104 = icmp eq %struct.InternalNode* %81, null
  br i1 %tobool104, label %if.end113, label %if.then105

if.then105:                                       ; preds = %land.lhs.true101
  %82 = bitcast i64* %r00.sroa.0 to %struct.InternalNode**
  %r00.sroa.0.0.load378388 = load %struct.InternalNode*, %struct.InternalNode** %82, align 8
  detach within %syncreg, label %det.achd110, label %if.end113

det.achd110:                                      ; preds = %if.then105
  %call111 = tail call %struct.InternalNode* @mul_and_subT(i32 %dec, i32 %lower, %struct.InternalNode* nonnull %80, %struct.InternalNode* nonnull %81, %struct.InternalNode* %r00.sroa.0.0.load378388)
  %83 = ptrtoint %struct.InternalNode* %call111 to i64
  store i64 %83, i64* %r00.sroa.0, align 8, !tbaa !9
  reattach within %syncreg, label %if.end113

if.end113:                                        ; preds = %land.lhs.true101, %if.then105, %det.achd110
  %brmerge384 = or i1 %tobool48398, %tobool100
  br i1 %brmerge384, label %if.end131, label %land.lhs.true119

land.lhs.true119:                                 ; preds = %if.end113
  %arrayidx121 = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %b, i64 0, i32 0, i64 3
  %84 = load %struct.InternalNode*, %struct.InternalNode** %arrayidx121, align 8, !tbaa !9
  %tobool122 = icmp eq %struct.InternalNode* %84, null
  br i1 %tobool122, label %if.end131, label %if.then123

if.then123:                                       ; preds = %land.lhs.true119
  %85 = bitcast i64* %r01.sroa.0 to %struct.InternalNode**
  %r01.sroa.0.0.load370389 = load %struct.InternalNode*, %struct.InternalNode** %85, align 8
  detach within %syncreg, label %det.achd128, label %if.end131

det.achd128:                                      ; preds = %if.then123
  %call129 = tail call %struct.InternalNode* @mul_and_subT(i32 %dec, i32 0, %struct.InternalNode* nonnull %80, %struct.InternalNode* nonnull %84, %struct.InternalNode* %r01.sroa.0.0.load370389)
  %86 = ptrtoint %struct.InternalNode* %call129 to i64
  store i64 %86, i64* %r01.sroa.0, align 8, !tbaa !9
  reattach within %syncreg, label %if.end131

if.end131:                                        ; preds = %sync.continue, %if.end113, %land.lhs.true119, %if.then123, %det.achd128
  %arrayidx133 = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %a, i64 0, i32 0, i64 3
  %87 = load %struct.InternalNode*, %struct.InternalNode** %arrayidx133, align 8, !tbaa !9
  %tobool134 = icmp eq %struct.InternalNode* %87, null
  br i1 %tobool134, label %if.end163, label %land.lhs.true135

land.lhs.true135:                                 ; preds = %if.end131
  %arrayidx137 = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %b, i64 0, i32 0, i64 1
  %88 = load %struct.InternalNode*, %struct.InternalNode** %arrayidx137, align 8, !tbaa !9
  %tobool138 = icmp eq %struct.InternalNode* %88, null
  br i1 %tobool138, label %land.lhs.true151, label %if.then139

if.then139:                                       ; preds = %land.lhs.true135
  %89 = bitcast i64* %r10.sroa.0 to %struct.InternalNode**
  %r10.sroa.0.0.load362390 = load %struct.InternalNode*, %struct.InternalNode** %89, align 8
  detach within %syncreg, label %det.achd144, label %land.lhs.true151

det.achd144:                                      ; preds = %if.then139
  %call145 = tail call %struct.InternalNode* @mul_and_subT(i32 %dec, i32 0, %struct.InternalNode* nonnull %87, %struct.InternalNode* nonnull %88, %struct.InternalNode* %r10.sroa.0.0.load362390)
  %90 = ptrtoint %struct.InternalNode* %call145 to i64
  store i64 %90, i64* %r10.sroa.0, align 8, !tbaa !9
  reattach within %syncreg, label %land.lhs.true151

land.lhs.true151:                                 ; preds = %land.lhs.true135, %if.then139, %det.achd144
  %arrayidx153 = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %b, i64 0, i32 0, i64 3
  %91 = load %struct.InternalNode*, %struct.InternalNode** %arrayidx153, align 8, !tbaa !9
  %tobool154 = icmp eq %struct.InternalNode* %91, null
  br i1 %tobool154, label %if.end163, label %if.then155

if.then155:                                       ; preds = %land.lhs.true151
  %call161 = tail call %struct.InternalNode* @mul_and_subT(i32 %dec, i32 %lower, %struct.InternalNode* nonnull %87, %struct.InternalNode* nonnull %91, %struct.InternalNode* %r11.1)
  br label %if.end163

if.end163:                                        ; preds = %if.end131, %land.lhs.true151, %if.then155
  %r11.2 = phi %struct.InternalNode* [ %call161, %if.then155 ], [ %r11.1, %land.lhs.true151 ], [ %r11.1, %if.end131 ]
  sync within %syncreg, label %sync.continue164

sync.continue164:                                 ; preds = %if.end163
  br i1 %cmp24, label %if.then166, label %if.else176

if.then166:                                       ; preds = %sync.continue164
  %92 = bitcast i64* %r00.sroa.0 to %struct.InternalNode**
  %r00.sroa.0.0.load379391 = load %struct.InternalNode*, %struct.InternalNode** %92, align 8
  %tobool167 = icmp ne %struct.InternalNode* %r00.sroa.0.0.load379391, null
  %93 = bitcast i64* %r01.sroa.0 to %struct.InternalNode**
  %r01.sroa.0.0.load371392 = load %struct.InternalNode*, %struct.InternalNode** %93, align 8
  %tobool168 = icmp ne %struct.InternalNode* %r01.sroa.0.0.load371392, null
  %or.cond227 = or i1 %tobool167, %tobool168
  %94 = bitcast i64* %r10.sroa.0 to %struct.InternalNode**
  %r10.sroa.0.0.load363393 = load %struct.InternalNode*, %struct.InternalNode** %94, align 8
  %tobool170 = icmp ne %struct.InternalNode* %r10.sroa.0.0.load363393, null
  %or.cond228 = or i1 %or.cond227, %tobool170
  %tobool172 = icmp ne %struct.InternalNode* %r11.2, null
  %or.cond229 = or i1 %tobool172, %or.cond228
  br i1 %or.cond229, label %if.then173, label %if.end225

if.then173:                                       ; preds = %if.then166
  %call.i356 = tail call noalias i8* @malloc(i64 32) #9
  %cmp.i357 = icmp eq i8* %call.i356, null
  br i1 %cmp.i357, label %if.then.i359, label %new_internal.exit

if.then.i359:                                     ; preds = %if.then173
  %puts.i358 = tail call i32 @puts(i8* getelementptr inbounds ([15 x i8], [15 x i8]* @str.46, i64 0, i64 0)) #9
  tail call void @exit(i32 1) #10
  unreachable

new_internal.exit:                                ; preds = %if.then173
  %95 = bitcast i8* %call.i356 to %struct.InternalNode*
  %arrayidx.i = bitcast i8* %call.i356 to %struct.InternalNode**
  store %struct.InternalNode* %r00.sroa.0.0.load379391, %struct.InternalNode** %arrayidx.i, align 8, !tbaa !9
  %arrayidx3.i = getelementptr inbounds i8, i8* %call.i356, i64 8
  %96 = bitcast i8* %arrayidx3.i to %struct.InternalNode**
  store %struct.InternalNode* %r01.sroa.0.0.load371392, %struct.InternalNode** %96, align 8, !tbaa !9
  %arrayidx5.i = getelementptr inbounds i8, i8* %call.i356, i64 16
  %97 = bitcast i8* %arrayidx5.i to %struct.InternalNode**
  store %struct.InternalNode* %r10.sroa.0.0.load363393, %struct.InternalNode** %97, align 8, !tbaa !9
  %arrayidx7.i = getelementptr inbounds i8, i8* %call.i356, i64 24
  %98 = bitcast i8* %arrayidx7.i to %struct.InternalNode**
  br label %if.end225.sink.split

if.else176:                                       ; preds = %sync.continue164
  %arrayidx178 = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %r, i64 0, i32 0, i64 0
  %99 = load %struct.InternalNode*, %struct.InternalNode** %arrayidx178, align 8, !tbaa !9
  %cmp179 = icmp eq %struct.InternalNode* %99, null
  %r00.sroa.0.0.load380 = load i64, i64* %r00.sroa.0, align 8
  %100 = inttoptr i64 %r00.sroa.0.0.load380 to %struct.InternalNode*
  %cmp183 = icmp eq %struct.InternalNode* %99, %100
  %or.cond332 = or i1 %cmp179, %cmp183
  br i1 %or.cond332, label %if.end186, label %if.else185

if.else185:                                       ; preds = %if.else176
  tail call void @__assert_fail(i8* getelementptr inbounds ([46 x i8], [46 x i8]* @.str.6, i64 0, i64 0), i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.1, i64 0, i64 0), i32 646, i8* getelementptr inbounds ([54 x i8], [54 x i8]* @__PRETTY_FUNCTION__.mul_and_subT, i64 0, i64 0)) #10
  unreachable

if.end186:                                        ; preds = %if.else176
  %arrayidx188 = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %r, i64 0, i32 0, i64 1
  %101 = load %struct.InternalNode*, %struct.InternalNode** %arrayidx188, align 8, !tbaa !9
  %cmp189 = icmp eq %struct.InternalNode* %101, null
  %r01.sroa.0.0.load372 = load i64, i64* %r01.sroa.0, align 8
  %102 = inttoptr i64 %r01.sroa.0.0.load372 to %struct.InternalNode*
  %cmp193 = icmp eq %struct.InternalNode* %101, %102
  %or.cond333 = or i1 %cmp189, %cmp193
  br i1 %or.cond333, label %if.end196, label %if.else195

if.else195:                                       ; preds = %if.end186
  tail call void @__assert_fail(i8* getelementptr inbounds ([46 x i8], [46 x i8]* @.str.7, i64 0, i64 0), i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.1, i64 0, i64 0), i32 647, i8* getelementptr inbounds ([54 x i8], [54 x i8]* @__PRETTY_FUNCTION__.mul_and_subT, i64 0, i64 0)) #10
  unreachable

if.end196:                                        ; preds = %if.end186
  %arrayidx198 = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %r, i64 0, i32 0, i64 2
  %103 = load %struct.InternalNode*, %struct.InternalNode** %arrayidx198, align 8, !tbaa !9
  %cmp199 = icmp eq %struct.InternalNode* %103, null
  %r10.sroa.0.0.load364 = load i64, i64* %r10.sroa.0, align 8
  %104 = inttoptr i64 %r10.sroa.0.0.load364 to %struct.InternalNode*
  %cmp203 = icmp eq %struct.InternalNode* %103, %104
  %or.cond334 = or i1 %cmp199, %cmp203
  br i1 %or.cond334, label %if.end206, label %if.else205

if.else205:                                       ; preds = %if.end196
  tail call void @__assert_fail(i8* getelementptr inbounds ([46 x i8], [46 x i8]* @.str.8, i64 0, i64 0), i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.1, i64 0, i64 0), i32 648, i8* getelementptr inbounds ([54 x i8], [54 x i8]* @__PRETTY_FUNCTION__.mul_and_subT, i64 0, i64 0)) #10
  unreachable

if.end206:                                        ; preds = %if.end196
  %arrayidx208 = getelementptr inbounds %struct.InternalNode, %struct.InternalNode* %r, i64 0, i32 0, i64 3
  %105 = load %struct.InternalNode*, %struct.InternalNode** %arrayidx208, align 8, !tbaa !9
  %cmp209 = icmp eq %struct.InternalNode* %105, null
  %cmp213 = icmp eq %struct.InternalNode* %105, %r11.2
  %or.cond335 = or i1 %cmp209, %cmp213
  br i1 %or.cond335, label %if.end216, label %if.else215

if.else215:                                       ; preds = %if.end206
  tail call void @__assert_fail(i8* getelementptr inbounds ([46 x i8], [46 x i8]* @.str.9, i64 0, i64 0), i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.1, i64 0, i64 0), i32 649, i8* getelementptr inbounds ([54 x i8], [54 x i8]* @__PRETTY_FUNCTION__.mul_and_subT, i64 0, i64 0)) #10
  unreachable

if.end216:                                        ; preds = %if.end206
  %106 = bitcast %struct.InternalNode* %r to i64*
  store i64 %r00.sroa.0.0.load380, i64* %106, align 8, !tbaa !9
  %107 = bitcast %struct.InternalNode** %arrayidx188 to i64*
  store i64 %r01.sroa.0.0.load372, i64* %107, align 8, !tbaa !9
  %108 = bitcast %struct.InternalNode** %arrayidx198 to i64*
  store i64 %r10.sroa.0.0.load364, i64* %108, align 8, !tbaa !9
  br label %if.end225.sink.split

if.end225.sink.split:                             ; preds = %if.end216, %new_internal.exit
  %.sink = phi %struct.InternalNode** [ %98, %new_internal.exit ], [ %arrayidx208, %if.end216 ]
  %r.addr.1.ph = phi %struct.InternalNode* [ %95, %new_internal.exit ], [ %r, %if.end216 ]
  store %struct.InternalNode* %r11.2, %struct.InternalNode** %.sink, align 8, !tbaa !9
  br label %if.end225

if.end225:                                        ; preds = %if.end225.sink.split, %if.then166
  %r.addr.1 = phi %struct.InternalNode* [ null, %if.then166 ], [ %r.addr.1.ph, %if.end225.sink.split ]
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %r10.sroa.0.0.r10.0..sroa_cast365)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %r01.sroa.0.0.r01.0..sroa_cast373)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %r00.sroa.0.0.r00.0..sroa_cast381)
  br label %cleanup

cleanup:                                          ; preds = %for.inc20.i, %for.body6.3.i, %if.end225
  %retval.0 = phi %struct.InternalNode* [ %r.addr.1, %if.end225 ], [ %r.addr.0, %for.body6.3.i ], [ %r.addr.0, %for.inc20.i ]
  ret %struct.InternalNode* %retval.0

for.body6.i.1:                                    ; preds = %for.cond1.preheader.i
  %arrayidx16.i.1 = getelementptr inbounds [4 x double], [4 x double]* %arraydecay10, i64 %indvars.iv43.i, i64 1
  %109 = load double, double* %arrayidx8.i, align 8, !tbaa !11
  %110 = load double, double* %4, align 8, !tbaa !11
  %mul.i.1 = fmul double %109, %110
  %111 = load double, double* %arrayidx16.i.1, align 8, !tbaa !11
  %sub.i.1 = fsub double %111, %mul.i.1
  store double %sub.i.1, double* %arrayidx16.i.1, align 8, !tbaa !11
  %112 = load double, double* %arrayidx8.1.i, align 8, !tbaa !11
  %113 = load double, double* %5, align 8, !tbaa !11
  %mul.1.i.1 = fmul double %112, %113
  %sub.1.i.1 = fsub double %sub.i.1, %mul.1.i.1
  store double %sub.1.i.1, double* %arrayidx16.i.1, align 8, !tbaa !11
  %114 = load double, double* %arrayidx8.2.i, align 8, !tbaa !11
  %115 = load double, double* %6, align 8, !tbaa !11
  %mul.2.i.1 = fmul double %114, %115
  %sub.2.i.1 = fsub double %sub.1.i.1, %mul.2.i.1
  store double %sub.2.i.1, double* %arrayidx16.i.1, align 8, !tbaa !11
  %116 = load double, double* %arrayidx8.3.i, align 8, !tbaa !11
  %117 = load double, double* %7, align 8, !tbaa !11
  %mul.3.i.1 = fmul double %116, %117
  %sub.3.i.1 = fsub double %sub.2.i.1, %mul.3.i.1
  store double %sub.3.i.1, double* %arrayidx16.i.1, align 8, !tbaa !11
  %exitcond.i.1 = icmp eq i64 %indvars.iv41.i, 2
  br i1 %exitcond.i.1, label %for.inc20.i, label %for.body6.i.2

for.body6.i.2:                                    ; preds = %for.body6.i.1
  %arrayidx16.i.2 = getelementptr inbounds [4 x double], [4 x double]* %arraydecay10, i64 %indvars.iv43.i, i64 2
  %118 = load double, double* %arrayidx8.i, align 8, !tbaa !11
  %119 = load double, double* %8, align 8, !tbaa !11
  %mul.i.2 = fmul double %118, %119
  %120 = load double, double* %arrayidx16.i.2, align 8, !tbaa !11
  %sub.i.2 = fsub double %120, %mul.i.2
  store double %sub.i.2, double* %arrayidx16.i.2, align 8, !tbaa !11
  %121 = load double, double* %arrayidx8.1.i, align 8, !tbaa !11
  %122 = load double, double* %9, align 8, !tbaa !11
  %mul.1.i.2 = fmul double %121, %122
  %sub.1.i.2 = fsub double %sub.i.2, %mul.1.i.2
  store double %sub.1.i.2, double* %arrayidx16.i.2, align 8, !tbaa !11
  %123 = load double, double* %arrayidx8.2.i, align 8, !tbaa !11
  %124 = load double, double* %10, align 8, !tbaa !11
  %mul.2.i.2 = fmul double %123, %124
  %sub.2.i.2 = fsub double %sub.1.i.2, %mul.2.i.2
  store double %sub.2.i.2, double* %arrayidx16.i.2, align 8, !tbaa !11
  %125 = load double, double* %arrayidx8.3.i, align 8, !tbaa !11
  %126 = load double, double* %11, align 8, !tbaa !11
  %mul.3.i.2 = fmul double %125, %126
  %sub.3.i.2 = fsub double %sub.2.i.2, %mul.3.i.2
  store double %sub.3.i.2, double* %arrayidx16.i.2, align 8, !tbaa !11
  %exitcond.i.2 = icmp eq i64 %indvars.iv41.i, 3
  br i1 %exitcond.i.2, label %for.inc20.i, label %for.body6.i.3

for.body6.i.3:                                    ; preds = %for.body6.i.2
  %arrayidx16.i.3 = getelementptr inbounds [4 x double], [4 x double]* %arraydecay10, i64 %indvars.iv43.i, i64 3
  %127 = load double, double* %arrayidx8.i, align 8, !tbaa !11
  %128 = load double, double* %12, align 8, !tbaa !11
  %mul.i.3 = fmul double %127, %128
  %129 = load double, double* %arrayidx16.i.3, align 8, !tbaa !11
  %sub.i.3 = fsub double %129, %mul.i.3
  store double %sub.i.3, double* %arrayidx16.i.3, align 8, !tbaa !11
  %130 = load double, double* %arrayidx8.1.i, align 8, !tbaa !11
  %131 = load double, double* %13, align 8, !tbaa !11
  %mul.1.i.3 = fmul double %130, %131
  %sub.1.i.3 = fsub double %sub.i.3, %mul.1.i.3
  store double %sub.1.i.3, double* %arrayidx16.i.3, align 8, !tbaa !11
  %132 = load double, double* %arrayidx8.2.i, align 8, !tbaa !11
  %133 = load double, double* %14, align 8, !tbaa !11
  %mul.2.i.3 = fmul double %132, %133
  %sub.2.i.3 = fsub double %sub.1.i.3, %mul.2.i.3
  store double %sub.2.i.3, double* %arrayidx16.i.3, align 8, !tbaa !11
  %134 = load double, double* %arrayidx8.3.i, align 8, !tbaa !11
  %135 = load double, double* %15, align 8, !tbaa !11
  %mul.3.i.3 = fmul double %134, %135
  %sub.3.i.3 = fsub double %sub.2.i.3, %mul.3.i.3
  store double %sub.3.i.3, double* %arrayidx16.i.3, align 8, !tbaa !11
  br label %for.inc20.i
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #2

; Function Attrs: noreturn nounwind
declare dso_local void @__assert_fail(i8*, i8*, i32, i8*) local_unnamed_addr #5

; Function Attrs: noreturn nounwind
declare dso_local void @exit(i32) local_unnamed_addr #5

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #2

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #2

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #4

; Function Attrs: nounwind
declare i32 @puts(i8* nocapture readonly) local_unnamed_addr #9

attributes #2 = { argmemonly nounwind }
attributes #3 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #9 = { nounwind }
attributes #10 = { noreturn nounwind }

!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"any pointer", !5, i64 0}
!11 = !{!12, !12, i64 0}
!12 = !{!"double", !5, i64 0}
