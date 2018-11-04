; RUN: opt %s -csan -S | FileCheck %s
; RUN: opt %s -aa-pipeline=default -passes='cilksan' -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
%struct.timeval = type { i64, i64 }
%struct.timezone = type { i32, i32 }

@stderr = external local_unnamed_addr global %struct._IO_FILE*, align 8
@.str = private unnamed_addr constant [32 x i8] c"Usage: %s [<cilk-options>] <n>\0A\00", align 1
@.str.1 = private unnamed_addr constant [33 x i8] c"Use default board size, n = 13.\0A\00", align 1
@.str.2 = private unnamed_addr constant [25 x i8] c"Running %s with n = %d.\0A\00", align 1
@.str.3 = private unnamed_addr constant [4 x i8] c"%f\0A\00", align 1
@.str.4 = private unnamed_addr constant [20 x i8] c"No solution found.\0A\00", align 1
@.str.5 = private unnamed_addr constant [32 x i8] c"Total number of solutions : %d\0A\00", align 1

; Function Attrs: argmemonly nounwind readonly uwtable
define i64 @todval(%struct.timeval* nocapture readonly %tp) local_unnamed_addr #0 !dbg !14 {
entry:
  call void @llvm.dbg.value(metadata %struct.timeval* %tp, metadata !29, metadata !DIExpression()), !dbg !30
  %tv_sec = getelementptr inbounds %struct.timeval, %struct.timeval* %tp, i64 0, i32 0, !dbg !31
  %0 = load i64, i64* %tv_sec, align 8, !dbg !31, !tbaa !32
  %mul1 = mul i64 %0, 1000000, !dbg !37
  %tv_usec = getelementptr inbounds %struct.timeval, %struct.timeval* %tp, i64 0, i32 1, !dbg !38
  %1 = load i64, i64* %tv_usec, align 8, !dbg !38, !tbaa !39
  %add = add nsw i64 %mul1, %1, !dbg !40
  ret i64 %add, !dbg !41
}

; Function Attrs: argmemonly nounwind readonly uwtable
define i32 @ok(i32 %n, i8* nocapture readonly %a) local_unnamed_addr #0 !dbg !42 {
entry:
  call void @llvm.dbg.value(metadata i32 %n, metadata !46, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i8* %a, metadata !47, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i32 0, metadata !48, metadata !DIExpression()), !dbg !54
  %cmp48 = icmp sgt i32 %n, 0, !dbg !55
  br i1 %cmp48, label %for.body.lr.ph, label %cleanup, !dbg !58

for.body.lr.ph:                                   ; preds = %entry
  %0 = sext i32 %n to i64, !dbg !58
  %1 = sext i32 %n to i64, !dbg !58
  %2 = sext i32 %n to i64, !dbg !58
  br label %for.body, !dbg !58

for.body:                                         ; preds = %for.body.lr.ph, %for.inc21
  %indvars.iv56 = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next57, %for.inc21 ]
  %indvars.iv = phi i64 [ 1, %for.body.lr.ph ], [ %indvars.iv.next, %for.inc21 ]
  call void @llvm.dbg.value(metadata i64 %indvars.iv56, metadata !48, metadata !DIExpression()), !dbg !54
  %arrayidx = getelementptr inbounds i8, i8* %a, i64 %indvars.iv56, !dbg !59
  %3 = load i8, i8* %arrayidx, align 1, !dbg !59, !tbaa !61
  call void @llvm.dbg.value(metadata i8 %3, metadata !50, metadata !DIExpression()), !dbg !62
  %indvars.iv.next57 = add nuw nsw i64 %indvars.iv56, 1, !dbg !63
  %cmp246 = icmp slt i64 %indvars.iv.next57, %1, !dbg !65
  br i1 %cmp246, label %for.body3.lr.ph, label %for.inc21, !dbg !67

for.body3.lr.ph:                                  ; preds = %for.body
  %4 = sext i8 %3 to i64, !dbg !67
  %5 = sext i8 %3 to i64, !dbg !67
  br label %for.body3, !dbg !67

for.cond1:                                        ; preds = %lor.lhs.false
  %cmp2 = icmp slt i64 %indvars.iv.next52, %0, !dbg !65
  br i1 %cmp2, label %for.body3, label %for.inc21, !dbg !67, !llvm.loop !68

for.body3:                                        ; preds = %for.body3.lr.ph, %for.cond1
  %indvars.iv51 = phi i64 [ %indvars.iv, %for.body3.lr.ph ], [ %indvars.iv.next52, %for.cond1 ]
  call void @llvm.dbg.value(metadata i64 %indvars.iv51, metadata !49, metadata !DIExpression()), !dbg !70
  %arrayidx5 = getelementptr inbounds i8, i8* %a, i64 %indvars.iv51, !dbg !71
  %6 = load i8, i8* %arrayidx5, align 1, !dbg !71, !tbaa !61
  call void @llvm.dbg.value(metadata i8 %6, metadata !51, metadata !DIExpression()), !dbg !73
  %conv = sext i8 %6 to i32, !dbg !74
  %cmp7 = icmp eq i8 %6, %3, !dbg !76
  br i1 %cmp7, label %cleanup, label %lor.lhs.false, !dbg !77

lor.lhs.false:                                    ; preds = %for.body3
  %7 = sub nuw nsw i64 %indvars.iv51, %indvars.iv56, !dbg !78
  %8 = sub nsw i64 %4, %7, !dbg !79
  %9 = trunc i64 %8 to i32, !dbg !80
  %cmp12 = icmp eq i32 %9, %conv, !dbg !80
  %10 = add nsw i64 %7, %5, !dbg !81
  %11 = trunc i64 %10 to i32, !dbg !82
  %cmp19 = icmp eq i32 %11, %conv, !dbg !82
  %or.cond = or i1 %cmp12, %cmp19, !dbg !83
  %indvars.iv.next52 = add nuw nsw i64 %indvars.iv51, 1, !dbg !84
  br i1 %or.cond, label %cleanup, label %for.cond1, !dbg !83

for.inc21:                                        ; preds = %for.cond1, %for.body
  %cmp = icmp slt i64 %indvars.iv.next57, %2, !dbg !55
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !58
  br i1 %cmp, label %for.body, label %cleanup, !dbg !58, !llvm.loop !85

cleanup:                                          ; preds = %for.inc21, %for.body3, %lor.lhs.false, %entry
  %retval.0 = phi i32 [ 1, %entry ], [ 0, %lor.lhs.false ], [ 0, %for.body3 ], [ 1, %for.inc21 ]
  ret i32 %retval.0, !dbg !87
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind uwtable
define i32 @nqueens(i32 %n, i32 %j, i8* nocapture readonly %a) local_unnamed_addr #2 !dbg !88 {
; CHECK-LABEL: @nqueens
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  call void @llvm.dbg.value(metadata i32 %n, metadata !92, metadata !DIExpression()), !dbg !102
  call void @llvm.dbg.value(metadata i32 %j, metadata !93, metadata !DIExpression()), !dbg !103
  call void @llvm.dbg.value(metadata i8* %a, metadata !94, metadata !DIExpression()), !dbg !104
  call void @llvm.dbg.value(metadata i32 0, metadata !97, metadata !DIExpression()), !dbg !105
  %cmp = icmp eq i32 %n, %j, !dbg !106
  br i1 %cmp, label %cleanup, label %if.end, !dbg !108

if.end:                                           ; preds = %entry
  %conv = sext i32 %n to i64, !dbg !109
  %mul = shl nsw i64 %conv, 2, !dbg !109
  %0 = alloca i32, i64 %conv, align 16, !dbg !109
  %tmpcast = bitcast i32* %0 to i8*, !dbg !109
  call void @llvm.dbg.value(metadata i32* %0, metadata !96, metadata !DIExpression()), !dbg !110
  call void @llvm.memset.p0i8.i64(i8* nonnull %tmpcast, i8 0, i64 %mul, i32 16, i1 false), !dbg !111
  call void @llvm.dbg.value(metadata i32 0, metadata !98, metadata !DIExpression()), !dbg !112
  %cmp362 = icmp sgt i32 %n, 0, !dbg !113
  br i1 %cmp362, label %for.body.lr.ph, label %for.cond.cleanup, !dbg !115

for.body.lr.ph:                                   ; preds = %if.end
  %add = add nsw i32 %j, 1
  %conv5 = sext i32 %add to i64
  %conv7 = sext i32 %j to i64
  %wide.trip.count66 = zext i32 %n to i64
  br label %for.body, !dbg !115

for.cond.cleanup:                                 ; preds = %for.inc, %if.end
  sync within %syncreg, label %sync.continue, !dbg !116

for.body:                                         ; preds = %for.inc, %for.body.lr.ph
  %indvars.iv64 = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next65, %for.inc ]
  call void @llvm.dbg.value(metadata i64 %indvars.iv64, metadata !98, metadata !DIExpression()), !dbg !112
  %1 = alloca i8, i64 %conv5, align 16, !dbg !117
  call void @llvm.dbg.value(metadata i8* %1, metadata !95, metadata !DIExpression()), !dbg !119
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull %1, i8* %a, i64 %conv7, i32 1, i1 false), !dbg !120
  %conv9 = trunc i64 %indvars.iv64 to i8, !dbg !121
  %arrayidx = getelementptr inbounds i8, i8* %1, i64 %conv7, !dbg !122
  store i8 %conv9, i8* %arrayidx, align 1, !dbg !123, !tbaa !61
  %call = call i32 @ok(i32 %add, i8* nonnull %1), !dbg !124
  %tobool = icmp eq i32 %call, 0, !dbg !124
  br i1 %tobool, label %for.inc, label %if.then11, !dbg !126
; CHECK: for.body:
; CHECK-NOT: @__csi_after_alloca
; CHECK: @__csan_large_load
; CHECK-NEXT: call void @llvm.memcpy
; CHECK: i8*{{.*}} %a,
; CHECK-NOT: @__csan_store
; CHECK: store i8 %conv9, i8* %arrayidx
; CHECK: @__cilksan_disable_checking
; CHECK-NEXT: call i32 @ok
; CHECK-NEXT: @__cilksan_enable_checking

if.then11:                                        ; preds = %for.body
  detach within %syncreg, label %det.achd, label %for.inc, !dbg !127

det.achd:                                         ; preds = %if.then11
  %arrayidx13 = getelementptr inbounds i32, i32* %0, i64 %indvars.iv64, !dbg !129
  %call15 = call i32 @nqueens(i32 %n, i32 %add, i8* nonnull %1), !dbg !127
  store i32 %call15, i32* %arrayidx13, align 4, !dbg !130, !tbaa !131
  reattach within %syncreg, label %for.inc, !dbg !130
; CHECK-LABEL: det.achd:
; CHECK: @__cilksan_disable_checking
; CHECK-NEXT: call i32 @nqueens
; CHECK-NEXT: @__cilksan_enable_checking

for.inc:                                          ; preds = %for.body, %det.achd, %if.then11
  %indvars.iv.next65 = add nuw nsw i64 %indvars.iv64, 1, !dbg !133
  %exitcond67 = icmp eq i64 %indvars.iv.next65, %wide.trip.count66, !dbg !113
  br i1 %exitcond67, label %for.cond.cleanup, label %for.body, !dbg !115, !llvm.loop !134

sync.continue:                                    ; preds = %for.cond.cleanup
  call void @llvm.dbg.value(metadata i32 0, metadata !100, metadata !DIExpression()), !dbg !136
  call void @llvm.dbg.value(metadata i32 0, metadata !97, metadata !DIExpression()), !dbg !105
  %cmp1959 = icmp sgt i32 %n, 0, !dbg !137
  br i1 %cmp1959, label %for.body22.lr.ph, label %cleanup, !dbg !139

for.body22.lr.ph:                                 ; preds = %sync.continue
  %wide.trip.count = zext i32 %n to i64
  br label %for.body22, !dbg !139

for.body22:                                       ; preds = %for.body22, %for.body22.lr.ph
  %indvars.iv = phi i64 [ 0, %for.body22.lr.ph ], [ %indvars.iv.next, %for.body22 ]
  %solNum.060 = phi i32 [ 0, %for.body22.lr.ph ], [ %add25, %for.body22 ]
  call void @llvm.dbg.value(metadata i32 %solNum.060, metadata !97, metadata !DIExpression()), !dbg !105
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !100, metadata !DIExpression()), !dbg !136
  %arrayidx24 = getelementptr inbounds i32, i32* %0, i64 %indvars.iv, !dbg !140
  %2 = load i32, i32* %arrayidx24, align 4, !dbg !140, !tbaa !131
  %add25 = add nsw i32 %2, %solNum.060, !dbg !142
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !143
  call void @llvm.dbg.value(metadata i32 %add25, metadata !97, metadata !DIExpression()), !dbg !105
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count, !dbg !137
  br i1 %exitcond, label %cleanup, label %for.body22, !dbg !139, !llvm.loop !144

cleanup:                                          ; preds = %for.body22, %sync.continue, %entry
  %retval.0 = phi i32 [ 1, %entry ], [ 0, %sync.continue ], [ %add25, %for.body22 ]
  ret i32 %retval.0, !dbg !146
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i32, i1) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32, i1) #1

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #1

; Function Attrs: nounwind uwtable
define i32 @main(i32 %argc, i8** nocapture readonly %argv) local_unnamed_addr #3 !dbg !147 {
entry:
  %t1 = alloca %struct.timeval, align 8
  %t2 = alloca %struct.timeval, align 8
  call void @llvm.dbg.value(metadata i32 %argc, metadata !151, metadata !DIExpression()), !dbg !159
  call void @llvm.dbg.value(metadata i8** %argv, metadata !152, metadata !DIExpression()), !dbg !160
  call void @llvm.dbg.value(metadata i32 13, metadata !153, metadata !DIExpression()), !dbg !161
  %cmp = icmp slt i32 %argc, 2, !dbg !162
  br i1 %cmp, label %if.then, label %if.else, !dbg !164

if.then:                                          ; preds = %entry
  %0 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !dbg !165, !tbaa !167
  %1 = load i8*, i8** %argv, align 8, !dbg !169, !tbaa !167
  %call = tail call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %0, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @.str, i64 0, i64 0), i8* %1) #8, !dbg !170
  %2 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !dbg !171, !tbaa !167
  %3 = tail call i64 @fwrite(i8* getelementptr inbounds ([33 x i8], [33 x i8]* @.str.1, i64 0, i64 0), i64 32, i64 1, %struct._IO_FILE* %2) #8, !dbg !172
  br label %if.end, !dbg !173

if.else:                                          ; preds = %entry
  %arrayidx2 = getelementptr inbounds i8*, i8** %argv, i64 1, !dbg !174
  %4 = load i8*, i8** %arrayidx2, align 8, !dbg !174, !tbaa !167
  %call3 = tail call i32 @atoi(i8* %4) #9, !dbg !176
  call void @llvm.dbg.value(metadata i32 %call3, metadata !153, metadata !DIExpression()), !dbg !161
  %5 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !dbg !177, !tbaa !167
  %6 = load i8*, i8** %argv, align 8, !dbg !178, !tbaa !167
  %call5 = tail call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %5, i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.2, i64 0, i64 0), i8* %6, i32 %call3) #8, !dbg !179
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %n.0 = phi i32 [ 13, %if.then ], [ %call3, %if.else ]
  call void @llvm.dbg.value(metadata i32 %n.0, metadata !153, metadata !DIExpression()), !dbg !161
  %conv = sext i32 %n.0 to i64, !dbg !180
  %7 = alloca i8, i64 %conv, align 16, !dbg !180
  call void @llvm.dbg.value(metadata i8* %7, metadata !154, metadata !DIExpression()), !dbg !181
  call void @llvm.dbg.value(metadata i32 0, metadata !155, metadata !DIExpression()), !dbg !182
  %8 = bitcast %struct.timeval* %t1 to i8*, !dbg !183
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %8) #7, !dbg !183
  %9 = bitcast %struct.timeval* %t2 to i8*, !dbg !183
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %9) #7, !dbg !183
  call void @llvm.dbg.value(metadata %struct.timeval* %t1, metadata !156, metadata !DIExpression()), !dbg !184
  %call6 = call i32 @gettimeofday(%struct.timeval* nonnull %t1, %struct.timezone* null) #7, !dbg !185
  %call7 = call i32 @nqueens(i32 %n.0, i32 0, i8* nonnull %7), !dbg !186
  call void @llvm.dbg.value(metadata i32 %call7, metadata !155, metadata !DIExpression()), !dbg !182
  call void @llvm.dbg.value(metadata %struct.timeval* %t2, metadata !157, metadata !DIExpression()), !dbg !187
  %call8 = call i32 @gettimeofday(%struct.timeval* nonnull %t2, %struct.timezone* null) #7, !dbg !188
  call void @llvm.dbg.value(metadata %struct.timeval* %t2, metadata !157, metadata !DIExpression()), !dbg !187
  %call9 = call i64 @todval(%struct.timeval* nonnull %t2), !dbg !189
  call void @llvm.dbg.value(metadata %struct.timeval* %t1, metadata !156, metadata !DIExpression()), !dbg !184
  %call10 = call i64 @todval(%struct.timeval* nonnull %t1), !dbg !190
  %sub = sub i64 %call9, %call10, !dbg !191
  %div = udiv i64 %sub, 1000, !dbg !192
  call void @llvm.dbg.value(metadata i64 %div, metadata !158, metadata !DIExpression()), !dbg !193
  %conv11 = uitofp i64 %div to double, !dbg !194
  %div12 = fdiv double %conv11, 1.000000e+03, !dbg !195
  %call13 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.3, i64 0, i64 0), double %div12), !dbg !196
  %cmp14 = icmp eq i32 %call7, 0, !dbg !197
  %10 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !167
  br i1 %cmp14, label %if.then16, label %if.else18, !dbg !199
; CHECK-LABEL: @main
; CHECK: @__cilksan_disable_checking
; CHECK-NEXT: call i32 @nqueens
; CHECK-NEXT: @__cilksan_enable_checking

if.then16:                                        ; preds = %if.end
  %11 = tail call i64 @fwrite(i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.str.4, i64 0, i64 0), i64 19, i64 1, %struct._IO_FILE* %10) #8, !dbg !200
  br label %if.end20, !dbg !202

if.else18:                                        ; preds = %if.end
  %call19 = tail call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %10, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @.str.5, i64 0, i64 0), i32 %call7) #8, !dbg !203
  br label %if.end20

if.end20:                                         ; preds = %if.else18, %if.then16
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %9) #7, !dbg !205
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %8) #7, !dbg !205
  ret i32 0, !dbg !205
}

; Function Attrs: nounwind
declare i32 @fprintf(%struct._IO_FILE* nocapture, i8* nocapture readonly, ...) local_unnamed_addr #4

; Function Attrs: inlinehint nounwind readonly uwtable
define available_externally i32 @atoi(i8* nonnull %__nptr) local_unnamed_addr #5 !dbg !206 {
entry:
  call void @llvm.dbg.value(metadata i8* %__nptr, metadata !213, metadata !DIExpression()), !dbg !214
  %call = tail call i64 @strtol(i8* nocapture nonnull %__nptr, i8** null, i32 10) #7, !dbg !215
  %conv = trunc i64 %call to i32, !dbg !216
  ret i32 %conv, !dbg !217
}

; Function Attrs: nounwind
declare i32 @gettimeofday(%struct.timeval* nocapture, %struct.timezone* nocapture) local_unnamed_addr #4

; Function Attrs: nounwind
declare i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #4

; Function Attrs: nounwind
declare i64 @strtol(i8* readonly, i8** nocapture, i32) local_unnamed_addr #4

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #6

; Function Attrs: nounwind
declare i64 @fwrite(i8* nocapture, i64, i64, %struct._IO_FILE* nocapture) local_unnamed_addr #7

attributes #0 = { argmemonly nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { argmemonly nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { inlinehint nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { nounwind readnone speculatable }
attributes #7 = { nounwind }
attributes #8 = { cold }
attributes #9 = { nounwind readonly }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11, !12}
!llvm.ident = !{!13}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0 (git@github.com:wsmoses/Tapir-Clang.git 051bd478f93bf64db3934d14f97a36137bffef5e) (git@github.mit.edu:SuperTech/Tapir-CSI-llvm.git 9de43afffece94ca0534b391544bbfd246fc7b91)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3)
!1 = !DIFile(filename: "nqueens.c", directory: "/data/compilers/tests/cilksan")
!2 = !{}
!3 = !{!4, !6, !5, !8, !9}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!7 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{!"clang version 6.0.0 (git@github.com:wsmoses/Tapir-Clang.git 051bd478f93bf64db3934d14f97a36137bffef5e) (git@github.mit.edu:SuperTech/Tapir-CSI-llvm.git 9de43afffece94ca0534b391544bbfd246fc7b91)"}
!14 = distinct !DISubprogram(name: "todval", scope: !1, file: !1, line: 12, type: !15, isLocal: false, isDefinition: true, scopeLine: 12, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !28)
!15 = !DISubroutineType(types: !16)
!16 = !{!17, !18}
!17 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !19, size: 64)
!19 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "timeval", file: !20, line: 8, size: 128, elements: !21)
!20 = !DIFile(filename: "/usr/include/bits/types/struct_timeval.h", directory: "/data/compilers/tests/cilksan")
!21 = !{!22, !26}
!22 = !DIDerivedType(tag: DW_TAG_member, name: "tv_sec", scope: !19, file: !20, line: 10, baseType: !23, size: 64)
!23 = !DIDerivedType(tag: DW_TAG_typedef, name: "__time_t", file: !24, line: 148, baseType: !25)
!24 = !DIFile(filename: "/usr/include/bits/types.h", directory: "/data/compilers/tests/cilksan")
!25 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!26 = !DIDerivedType(tag: DW_TAG_member, name: "tv_usec", scope: !19, file: !20, line: 11, baseType: !27, size: 64, offset: 64)
!27 = !DIDerivedType(tag: DW_TAG_typedef, name: "__suseconds_t", file: !24, line: 150, baseType: !25)
!28 = !{!29}
!29 = !DILocalVariable(name: "tp", arg: 1, scope: !14, file: !1, line: 12, type: !18)
!30 = !DILocation(line: 12, column: 44, scope: !14)
!31 = !DILocation(line: 13, column: 16, scope: !14)
!32 = !{!33, !34, i64 0}
!33 = !{!"timeval", !34, i64 0, !34, i64 8}
!34 = !{!"long", !35, i64 0}
!35 = !{!"omnipotent char", !36, i64 0}
!36 = !{!"Simple C/C++ TBAA"}
!37 = !DILocation(line: 13, column: 30, scope: !14)
!38 = !DILocation(line: 13, column: 43, scope: !14)
!39 = !{!33, !34, i64 8}
!40 = !DILocation(line: 13, column: 37, scope: !14)
!41 = !DILocation(line: 14, column: 1, scope: !14)
!42 = distinct !DISubprogram(name: "ok", scope: !1, file: !1, line: 37, type: !43, isLocal: false, isDefinition: true, scopeLine: 37, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !45)
!43 = !DISubroutineType(types: !44)
!44 = !{!5, !5, !6}
!45 = !{!46, !47, !48, !49, !50, !51}
!46 = !DILocalVariable(name: "n", arg: 1, scope: !42, file: !1, line: 37, type: !5)
!47 = !DILocalVariable(name: "a", arg: 2, scope: !42, file: !1, line: 37, type: !6)
!48 = !DILocalVariable(name: "i", scope: !42, file: !1, line: 39, type: !5)
!49 = !DILocalVariable(name: "j", scope: !42, file: !1, line: 39, type: !5)
!50 = !DILocalVariable(name: "p", scope: !42, file: !1, line: 40, type: !7)
!51 = !DILocalVariable(name: "q", scope: !42, file: !1, line: 40, type: !7)
!52 = !DILocation(line: 37, column: 13, scope: !42)
!53 = !DILocation(line: 37, column: 22, scope: !42)
!54 = !DILocation(line: 39, column: 7, scope: !42)
!55 = !DILocation(line: 42, column: 17, scope: !56)
!56 = distinct !DILexicalBlock(scope: !57, file: !1, line: 42, column: 3)
!57 = distinct !DILexicalBlock(scope: !42, file: !1, line: 42, column: 3)
!58 = !DILocation(line: 42, column: 3, scope: !57)
!59 = !DILocation(line: 43, column: 9, scope: !60)
!60 = distinct !DILexicalBlock(scope: !56, file: !1, line: 42, column: 27)
!61 = !{!35, !35, i64 0}
!62 = !DILocation(line: 40, column: 8, scope: !42)
!63 = !DILocation(line: 44, column: 16, scope: !64)
!64 = distinct !DILexicalBlock(scope: !60, file: !1, line: 44, column: 5)
!65 = !DILocation(line: 44, column: 23, scope: !66)
!66 = distinct !DILexicalBlock(scope: !64, file: !1, line: 44, column: 5)
!67 = !DILocation(line: 44, column: 5, scope: !64)
!68 = distinct !{!68, !67, !69}
!69 = !DILocation(line: 48, column: 5, scope: !64)
!70 = !DILocation(line: 39, column: 10, scope: !42)
!71 = !DILocation(line: 45, column: 11, scope: !72)
!72 = distinct !DILexicalBlock(scope: !66, file: !1, line: 44, column: 33)
!73 = !DILocation(line: 40, column: 11, scope: !42)
!74 = !DILocation(line: 46, column: 11, scope: !75)
!75 = distinct !DILexicalBlock(scope: !72, file: !1, line: 46, column: 11)
!76 = !DILocation(line: 46, column: 13, scope: !75)
!77 = !DILocation(line: 46, column: 18, scope: !75)
!78 = !DILocation(line: 46, column: 33, scope: !75)
!79 = !DILocation(line: 46, column: 28, scope: !75)
!80 = !DILocation(line: 46, column: 23, scope: !75)
!81 = !DILocation(line: 46, column: 48, scope: !75)
!82 = !DILocation(line: 46, column: 43, scope: !75)
!83 = !DILocation(line: 46, column: 38, scope: !75)
!84 = !DILocation(line: 44, column: 29, scope: !66)
!85 = distinct !{!85, !58, !86}
!86 = !DILocation(line: 49, column: 3, scope: !57)
!87 = !DILocation(line: 52, column: 1, scope: !42)
!88 = distinct !DISubprogram(name: "nqueens", scope: !1, file: !1, line: 54, type: !89, isLocal: false, isDefinition: true, scopeLine: 54, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !91)
!89 = !DISubroutineType(types: !90)
!90 = !{!5, !5, !5, !6}
!91 = !{!92, !93, !94, !95, !96, !97, !98, !100}
!92 = !DILocalVariable(name: "n", arg: 1, scope: !88, file: !1, line: 54, type: !5)
!93 = !DILocalVariable(name: "j", arg: 2, scope: !88, file: !1, line: 54, type: !5)
!94 = !DILocalVariable(name: "a", arg: 3, scope: !88, file: !1, line: 54, type: !6)
!95 = !DILocalVariable(name: "b", scope: !88, file: !1, line: 56, type: !6)
!96 = !DILocalVariable(name: "count", scope: !88, file: !1, line: 57, type: !4)
!97 = !DILocalVariable(name: "solNum", scope: !88, file: !1, line: 58, type: !5)
!98 = !DILocalVariable(name: "i", scope: !99, file: !1, line: 67, type: !5)
!99 = distinct !DILexicalBlock(scope: !88, file: !1, line: 67, column: 3)
!100 = !DILocalVariable(name: "i", scope: !101, file: !1, line: 93, type: !5)
!101 = distinct !DILexicalBlock(scope: !88, file: !1, line: 93, column: 3)
!102 = !DILocation(line: 54, column: 18, scope: !88)
!103 = !DILocation(line: 54, column: 25, scope: !88)
!104 = !DILocation(line: 54, column: 34, scope: !88)
!105 = !DILocation(line: 58, column: 7, scope: !88)
!106 = !DILocation(line: 60, column: 9, scope: !107)
!107 = distinct !DILexicalBlock(scope: !88, file: !1, line: 60, column: 7)
!108 = !DILocation(line: 60, column: 7, scope: !88)
!109 = !DILocation(line: 64, column: 19, scope: !88)
!110 = !DILocation(line: 57, column: 8, scope: !88)
!111 = !DILocation(line: 65, column: 10, scope: !88)
!112 = !DILocation(line: 67, column: 12, scope: !99)
!113 = !DILocation(line: 67, column: 21, scope: !114)
!114 = distinct !DILexicalBlock(scope: !99, file: !1, line: 67, column: 3)
!115 = !DILocation(line: 67, column: 3, scope: !99)
!116 = !DILocation(line: 90, column: 3, scope: !88)
!117 = !DILocation(line: 74, column: 18, scope: !118)
!118 = distinct !DILexicalBlock(scope: !114, file: !1, line: 67, column: 31)
!119 = !DILocation(line: 56, column: 9, scope: !88)
!120 = !DILocation(line: 75, column: 5, scope: !118)
!121 = !DILocation(line: 76, column: 12, scope: !118)
!122 = !DILocation(line: 76, column: 5, scope: !118)
!123 = !DILocation(line: 76, column: 10, scope: !118)
!124 = !DILocation(line: 77, column: 9, scope: !125)
!125 = distinct !DILexicalBlock(scope: !118, file: !1, line: 77, column: 9)
!126 = !DILocation(line: 77, column: 9, scope: !118)
!127 = !DILocation(line: 78, column: 29, scope: !128)
!128 = distinct !DILexicalBlock(scope: !125, file: !1, line: 77, column: 23)
!129 = !DILocation(line: 78, column: 7, scope: !128)
!130 = !DILocation(line: 78, column: 16, scope: !128)
!131 = !{!132, !132, i64 0}
!132 = !{!"int", !35, i64 0}
!133 = !DILocation(line: 67, column: 27, scope: !114)
!134 = distinct !{!134, !115, !135}
!135 = !DILocation(line: 80, column: 3, scope: !99)
!136 = !DILocation(line: 93, column: 12, scope: !101)
!137 = !DILocation(line: 93, column: 21, scope: !138)
!138 = distinct !DILexicalBlock(scope: !101, file: !1, line: 93, column: 3)
!139 = !DILocation(line: 93, column: 3, scope: !101)
!140 = !DILocation(line: 94, column: 15, scope: !141)
!141 = distinct !DILexicalBlock(scope: !138, file: !1, line: 93, column: 31)
!142 = !DILocation(line: 94, column: 12, scope: !141)
!143 = !DILocation(line: 93, column: 27, scope: !138)
!144 = distinct !{!144, !139, !145}
!145 = !DILocation(line: 95, column: 3, scope: !101)
!146 = !DILocation(line: 98, column: 1, scope: !88)
!147 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 101, type: !148, isLocal: false, isDefinition: true, scopeLine: 101, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !150)
!148 = !DISubroutineType(types: !149)
!149 = !{!5, !5, !8}
!150 = !{!151, !152, !153, !154, !155, !156, !157, !158}
!151 = !DILocalVariable(name: "argc", arg: 1, scope: !147, file: !1, line: 101, type: !5)
!152 = !DILocalVariable(name: "argv", arg: 2, scope: !147, file: !1, line: 101, type: !8)
!153 = !DILocalVariable(name: "n", scope: !147, file: !1, line: 103, type: !5)
!154 = !DILocalVariable(name: "a", scope: !147, file: !1, line: 104, type: !6)
!155 = !DILocalVariable(name: "res", scope: !147, file: !1, line: 105, type: !5)
!156 = !DILocalVariable(name: "t1", scope: !147, file: !1, line: 119, type: !19)
!157 = !DILocalVariable(name: "t2", scope: !147, file: !1, line: 119, type: !19)
!158 = !DILocalVariable(name: "runtime_ms", scope: !147, file: !1, line: 125, type: !17)
!159 = !DILocation(line: 101, column: 14, scope: !147)
!160 = !DILocation(line: 101, column: 26, scope: !147)
!161 = !DILocation(line: 103, column: 7, scope: !147)
!162 = !DILocation(line: 107, column: 12, scope: !163)
!163 = distinct !DILexicalBlock(scope: !147, file: !1, line: 107, column: 7)
!164 = !DILocation(line: 107, column: 7, scope: !147)
!165 = !DILocation(line: 108, column: 14, scope: !166)
!166 = distinct !DILexicalBlock(scope: !163, file: !1, line: 107, column: 17)
!167 = !{!168, !168, i64 0}
!168 = !{!"any pointer", !35, i64 0}
!169 = !DILocation(line: 108, column: 58, scope: !166)
!170 = !DILocation(line: 108, column: 5, scope: !166)
!171 = !DILocation(line: 109, column: 14, scope: !166)
!172 = !DILocation(line: 109, column: 5, scope: !166)
!173 = !DILocation(line: 111, column: 3, scope: !166)
!174 = !DILocation(line: 112, column: 15, scope: !175)
!175 = distinct !DILexicalBlock(scope: !163, file: !1, line: 111, column: 10)
!176 = !DILocation(line: 112, column: 9, scope: !175)
!177 = !DILocation(line: 113, column: 14, scope: !175)
!178 = !DILocation(line: 113, column: 51, scope: !175)
!179 = !DILocation(line: 113, column: 5, scope: !175)
!180 = !DILocation(line: 116, column: 16, scope: !147)
!181 = !DILocation(line: 104, column: 9, scope: !147)
!182 = !DILocation(line: 105, column: 7, scope: !147)
!183 = !DILocation(line: 119, column: 3, scope: !147)
!184 = !DILocation(line: 119, column: 18, scope: !147)
!185 = !DILocation(line: 120, column: 3, scope: !147)
!186 = !DILocation(line: 122, column: 9, scope: !147)
!187 = !DILocation(line: 119, column: 22, scope: !147)
!188 = !DILocation(line: 124, column: 3, scope: !147)
!189 = !DILocation(line: 125, column: 36, scope: !147)
!190 = !DILocation(line: 125, column: 48, scope: !147)
!191 = !DILocation(line: 125, column: 47, scope: !147)
!192 = !DILocation(line: 125, column: 60, scope: !147)
!193 = !DILocation(line: 125, column: 22, scope: !147)
!194 = !DILocation(line: 126, column: 18, scope: !147)
!195 = !DILocation(line: 126, column: 28, scope: !147)
!196 = !DILocation(line: 126, column: 3, scope: !147)
!197 = !DILocation(line: 128, column: 11, scope: !198)
!198 = distinct !DILexicalBlock(scope: !147, file: !1, line: 128, column: 7)
!199 = !DILocation(line: 128, column: 7, scope: !147)
!200 = !DILocation(line: 129, column: 5, scope: !201)
!201 = distinct !DILexicalBlock(scope: !198, file: !1, line: 128, column: 17)
!202 = !DILocation(line: 130, column: 3, scope: !201)
!203 = !DILocation(line: 131, column: 5, scope: !204)
!204 = distinct !DILexicalBlock(scope: !198, file: !1, line: 130, column: 10)
!205 = !DILocation(line: 135, column: 1, scope: !147)
!206 = distinct !DISubprogram(name: "atoi", scope: !207, file: !207, line: 361, type: !208, isLocal: false, isDefinition: true, scopeLine: 362, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !212)
!207 = !DIFile(filename: "/usr/include/stdlib.h", directory: "/data/compilers/tests/cilksan")
!208 = !DISubroutineType(types: !209)
!209 = !{!5, !210}
!210 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !211, size: 64)
!211 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !7)
!212 = !{!213}
!213 = !DILocalVariable(name: "__nptr", arg: 1, scope: !206, file: !207, line: 361, type: !210)
!214 = !DILocation(line: 361, column: 1, scope: !206)
!215 = !DILocation(line: 363, column: 16, scope: !206)
!216 = !DILocation(line: 363, column: 10, scope: !206)
!217 = !DILocation(line: 364, column: 1, scope: !206)
