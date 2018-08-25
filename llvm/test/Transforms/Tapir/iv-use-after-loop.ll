; RUN: opt < %s -loop-spawning-ti -S | FileCheck %s
; RUN: opt < %s -passes=loop-spawning -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$_Z8randPermIiEvPT_i = comdat any

; Function Attrs: uwtable
define linkonce_odr void @_Z8randPermIiEvPT_i(i32* %A, i32 %n) local_unnamed_addr #4 comdat personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %syncreg45 = tail call token @llvm.syncregion.start()
  %syncreg87 = tail call token @llvm.syncregion.start()
  %conv = sext i32 %n to i64
  %mul = shl nsw i64 %conv, 2
  %call = tail call noalias i8* @malloc(i64 %mul) #2
  %0 = bitcast i8* %call to i32*
  %call3 = tail call noalias i8* @malloc(i64 %mul) #2
  %1 = bitcast i8* %call3 to i32*
  %call6 = tail call noalias i8* @malloc(i64 %mul) #2
  %2 = bitcast i8* %call6 to i32*
  %cmp = icmp slt i32 %n, 100000
  br i1 %cmp, label %if.then, label %pfor.detach.lr.ph

if.then:                                          ; preds = %entry
  %cmp7289 = icmp sgt i32 %n, 1
  br i1 %cmp7289, label %for.body.preheader, label %cleanup

for.body.preheader:                               ; preds = %if.then
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ %conv, %for.body.preheader ]
  %i.0291.in = phi i32 [ %i.0291, %for.body ], [ %n, %for.body.preheader ]
  %indvars.iv.next = add nsw i64 %indvars.iv, -1
  %i.0291 = add nsw i32 %i.0291.in, -1
  %add.i = add i32 %i.0291.in, 2127912213
  %3 = trunc i64 %indvars.iv.next to i32
  %shl.i = shl i32 %3, 12
  %add1.i = add i32 %add.i, %shl.i
  %xor.i = xor i32 %add1.i, -949894596
  %shr.i = lshr i32 %add1.i, 19
  %xor2.i = xor i32 %xor.i, %shr.i
  %add3.i = add i32 %xor2.i, 374761393
  %shl4.i = shl i32 %xor2.i, 5
  %add5.i = add i32 %add3.i, %shl4.i
  %add6.i = add i32 %add5.i, -744332180
  %shl7.i = shl i32 %add5.i, 9
  %xor8.i = xor i32 %add6.i, %shl7.i
  %add9.i = add i32 %xor8.i, -42973499
  %shl10.i = shl i32 %xor8.i, 3
  %add11.i = add i32 %add9.i, %shl10.i
  %xor12.i = xor i32 %add11.i, -1252372727
  %shr13.i = lshr i32 %add11.i, 16
  %xor14.i = xor i32 %xor12.i, %shr13.i
  %rem = urem i32 %xor14.i, %i.0291.in
  %idxprom = zext i32 %rem to i64
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %idxprom
  %arrayidx10 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv.next
  %4 = load i32, i32* %arrayidx, align 4, !tbaa !42
  %5 = load i32, i32* %arrayidx10, align 4, !tbaa !42
  store i32 %5, i32* %arrayidx, align 4, !tbaa !42
  store i32 %4, i32* %arrayidx10, align 4, !tbaa !42
  %cmp7 = icmp sgt i64 %indvars.iv, 2
  br i1 %cmp7, label %for.body, label %cleanup

pfor.detach.lr.ph:                                ; preds = %entry
  %wide.trip.count310 = zext i32 %n to i64
  br label %pfor.detach

pfor.cond.cleanup:                                ; preds = %pfor.inc
  sync within %syncreg, label %sync.continue

pfor.detach:                                      ; preds = %pfor.inc, %pfor.detach.lr.ph
  %indvars.iv306 = phi i64 [ 0, %pfor.detach.lr.ph ], [ %indvars.iv.next307, %pfor.inc ]
  detach within %syncreg, label %pfor.body, label %pfor.inc unwind label %lpad27

pfor.body:                                        ; preds = %pfor.detach
  %6 = trunc i64 %indvars.iv306 to i32
  %shl.i270 = shl i32 %6, 12
  %7 = trunc i64 %indvars.iv306 to i32
  %8 = add i32 %7, 2127912214
  %add1.i271 = add i32 %8, %shl.i270
  %xor.i272 = xor i32 %add1.i271, -949894596
  %shr.i273 = lshr i32 %add1.i271, 19
  %xor2.i274 = xor i32 %xor.i272, %shr.i273
  %add3.i275 = add i32 %xor2.i274, 374761393
  %shl4.i276 = shl i32 %xor2.i274, 5
  %add5.i277 = add i32 %add3.i275, %shl4.i276
  %add6.i278 = add i32 %add5.i277, -744332180
  %shl7.i279 = shl i32 %add5.i277, 9
  %xor8.i280 = xor i32 %add6.i278, %shl7.i279
  %add9.i281 = add i32 %xor8.i280, -42973499
  %shl10.i282 = shl i32 %xor8.i280, 3
  %add11.i283 = add i32 %add9.i281, %shl10.i282
  %xor12.i284 = xor i32 %add11.i283, -1252372727
  %shr13.i285 = lshr i32 %add11.i283, 16
  %xor14.i286 = xor i32 %xor12.i284, %shr13.i285
  %9 = trunc i64 %indvars.iv306 to i32
  %10 = add i32 %9, 1
  %rem20 = urem i32 %xor14.i286, %10
  %arrayidx22 = getelementptr inbounds i32, i32* %1, i64 %indvars.iv306
  store i32 %rem20, i32* %arrayidx22, align 4, !tbaa !42
  %arrayidx24 = getelementptr inbounds i32, i32* %0, i64 %indvars.iv306
  store i32 %6, i32* %arrayidx24, align 4, !tbaa !42
  %arrayidx26 = getelementptr inbounds i32, i32* %2, i64 %indvars.iv306
  store i32 %6, i32* %arrayidx26, align 4, !tbaa !42
  reattach within %syncreg, label %pfor.inc

pfor.inc:                                         ; preds = %pfor.detach, %pfor.body
  %indvars.iv.next307 = add nuw nsw i64 %indvars.iv306, 1
  %exitcond311 = icmp eq i64 %indvars.iv.next307, %wide.trip.count310
  br i1 %exitcond311, label %pfor.cond.cleanup, label %pfor.detach, !llvm.loop !136

lpad27:                                           ; preds = %pfor.detach
  %11 = landingpad { i8*, i32 }
          cleanup
  %12 = extractvalue { i8*, i32 } %11, 0
  %13 = extractvalue { i8*, i32 } %11, 1
  sync within %syncreg, label %ehcleanup

sync.continue:                                    ; preds = %pfor.cond.cleanup
  %div32 = sdiv i32 %n, 100
  %add33 = add nsw i32 %div32, 1
  %conv34 = sext i32 %add33 to i64
  %mul35 = shl nsw i64 %conv34, 2
  %call36 = tail call noalias i8* @malloc(i64 %mul35) #2
  %14 = bitcast i8* %call36 to i32*
  %call39 = tail call noalias i8* @malloc(i64 %conv34) #2
  %cmp40294 = icmp sgt i32 %n, 0
  br i1 %cmp40294, label %while.body.preheader, label %while.end

while.body.preheader:                             ; preds = %sync.continue
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %sync.continue136
  %end.0295 = phi i32 [ %add140, %sync.continue136 ], [ %n, %while.body.preheader ]
  %div42287 = udiv i32 %end.0295, 100
  %add43 = add nuw nsw i32 %div42287, 1
  %sub44 = sub nsw i32 %end.0295, %add43
  %wide.trip.count = zext i32 %add43 to i64
  br label %pfor.detach56

pfor.cond.cleanup55:                              ; preds = %pfor.inc75
  sync within %syncreg45, label %pfor.cond.cleanup55.split

; CHECK: indvars.iv.next299.lcssa.escape = add i64 0, %cast.count

pfor.cond.cleanup55.split:                        ; preds = %pfor.cond.cleanup55
  br label %pfor.detach98

; CHECK: pfor.cond.cleanup55.split:
; CHECK: call fastcc void @_Z8randPermIiEvPT_i_pfor.detach98.ls2(i64 0, i64 %indvars.iv.next299.lcssa.escape

pfor.detach56:                                    ; preds = %pfor.inc75, %while.body
  %indvars.iv298 = phi i64 [ %indvars.iv.next299, %pfor.inc75 ], [ 0, %while.body ]
  %__begin47.0292 = phi i32 [ %inc76, %pfor.inc75 ], [ 0, %while.body ]
  detach within %syncreg45, label %pfor.body61, label %pfor.inc75 unwind label %lpad77

pfor.body61:                                      ; preds = %pfor.detach56
  %add62 = add nsw i32 %__begin47.0292, %sub44
  %idxprom63 = sext i32 %add62 to i64
  %arrayidx64 = getelementptr inbounds i32, i32* %0, i64 %idxprom63
  %15 = load i32, i32* %arrayidx64, align 4, !tbaa !42
  %idxprom65 = sext i32 %15 to i64
  %arrayidx66 = getelementptr inbounds i32, i32* %1, i64 %idxprom65
  %16 = load i32, i32* %arrayidx66, align 4, !tbaa !42
  %idxprom67 = sext i32 %16 to i64
  %arrayidx68 = getelementptr inbounds i32, i32* %2, i64 %idxprom67
  br label %do.body.i

do.body.i:                                        ; preds = %land.rhs.i, %pfor.body61
  %17 = load i32, i32* %arrayidx68, align 4, !tbaa !42
  %cmp.i = icmp slt i32 %17, %15
  br i1 %cmp.i, label %land.rhs.i, label %invoke.cont72

land.rhs.i:                                       ; preds = %do.body.i
  %18 = cmpxchg i32* %arrayidx68, i32 %17, i32 %15 seq_cst seq_cst
  %19 = extractvalue { i32, i1 } %18, 1
  br i1 %19, label %invoke.cont72, label %do.body.i

invoke.cont72:                                    ; preds = %land.rhs.i, %do.body.i
  reattach within %syncreg45, label %pfor.inc75

pfor.inc75:                                       ; preds = %pfor.detach56, %invoke.cont72
  %indvars.iv.next299 = add nuw nsw i64 %indvars.iv298, 1
  %inc76 = add nuw nsw i32 %__begin47.0292, 1
  %exitcond = icmp eq i64 %indvars.iv.next299, %wide.trip.count
  br i1 %exitcond, label %pfor.cond.cleanup55, label %pfor.detach56, !llvm.loop !137

lpad77:                                           ; preds = %pfor.detach56
  %20 = landingpad { i8*, i32 }
          cleanup
  %21 = extractvalue { i8*, i32 } %20, 0
  %22 = extractvalue { i8*, i32 } %20, 1
  sync within %syncreg45, label %ehcleanup

pfor.cond.cleanup97:                              ; preds = %pfor.inc134
  sync within %syncreg87, label %sync.continue136

pfor.detach98:                                    ; preds = %pfor.cond.cleanup55.split, %pfor.inc134
  %indvars.iv301 = phi i64 [ %indvars.iv.next302, %pfor.inc134 ], [ 0, %pfor.cond.cleanup55.split ]
  %__begin89.0293 = phi i32 [ %inc135, %pfor.inc134 ], [ 0, %pfor.cond.cleanup55.split ]
  detach within %syncreg87, label %pfor.body103, label %pfor.inc134

pfor.body103:                                     ; preds = %pfor.detach98
  %add105 = add nsw i32 %__begin89.0293, %sub44
  %idxprom106 = sext i32 %add105 to i64
  %arrayidx107 = getelementptr inbounds i32, i32* %0, i64 %idxprom106
  %23 = load i32, i32* %arrayidx107, align 4, !tbaa !42
  %idxprom109 = sext i32 %23 to i64
  %arrayidx110 = getelementptr inbounds i32, i32* %1, i64 %idxprom109
  %24 = load i32, i32* %arrayidx110, align 4, !tbaa !42
  %arrayidx112 = getelementptr inbounds i8, i8* %call39, i64 %indvars.iv301
  store i8 1, i8* %arrayidx112, align 1, !tbaa !7
  %arrayidx114 = getelementptr inbounds i32, i32* %14, i64 %indvars.iv301
  store i32 %23, i32* %arrayidx114, align 4, !tbaa !42
  %idxprom115 = sext i32 %24 to i64
  %arrayidx116 = getelementptr inbounds i32, i32* %2, i64 %idxprom115
  %25 = load i32, i32* %arrayidx116, align 4, !tbaa !42
  %cmp117 = icmp eq i32 %25, %23
  br i1 %cmp117, label %if.then118, label %if.end132

if.then118:                                       ; preds = %pfor.body103
  %arrayidx120 = getelementptr inbounds i32, i32* %2, i64 %idxprom109
  %26 = load i32, i32* %arrayidx120, align 4, !tbaa !42
  %cmp121 = icmp eq i32 %26, %23
  br i1 %cmp121, label %if.then122, label %if.end129

if.then122:                                       ; preds = %if.then118
  %arrayidx124 = getelementptr inbounds i32, i32* %A, i64 %idxprom109
  %arrayidx126 = getelementptr inbounds i32, i32* %A, i64 %idxprom115
  %27 = load i32, i32* %arrayidx124, align 4, !tbaa !42
  %28 = load i32, i32* %arrayidx126, align 4, !tbaa !42
  store i32 %28, i32* %arrayidx124, align 4, !tbaa !42
  store i32 %27, i32* %arrayidx126, align 4, !tbaa !42
  store i8 0, i8* %arrayidx112, align 1, !tbaa !7
  br label %if.end129

if.end129:                                        ; preds = %if.then122, %if.then118
  store i32 %24, i32* %arrayidx116, align 4, !tbaa !42
  br label %if.end132

if.end132:                                        ; preds = %if.end129, %pfor.body103
  reattach within %syncreg87, label %pfor.inc134

pfor.inc134:                                      ; preds = %if.end132, %pfor.detach98
  %indvars.iv.next302 = add nuw nsw i64 %indvars.iv301, 1
  %inc135 = add nuw nsw i32 %__begin89.0293, 1
  %exitcond305 = icmp eq i64 %indvars.iv.next302, %indvars.iv.next299
  br i1 %exitcond305, label %pfor.cond.cleanup97, label %pfor.detach98, !llvm.loop !138

sync.continue136:                                 ; preds = %pfor.cond.cleanup97
  %idx.ext = sext i32 %sub44 to i64
  %add.ptr = getelementptr inbounds i32, i32* %0, i64 %idx.ext
  %call.i = tail call { i32*, i64 } @_ZN8sequence4packIiiNS_4getAIiiEEEE4_seqIT_EPS4_PbT0_S8_T1_(i32* %add.ptr, i8* %call39, i32 0, i32 %add43, i32* %14)
  %29 = extractvalue { i32*, i64 } %call.i, 1
  %conv.i = trunc i64 %29 to i32
  %add140 = add nsw i32 %sub44, %conv.i
  %cmp40 = icmp sgt i32 %add140, 0
  br i1 %cmp40, label %while.body, label %while.end

while.end:                                        ; preds = %sync.continue136, %sync.continue
  tail call void @free(i8* %call3) #2
  tail call void @free(i8* %call) #2
  tail call void @free(i8* %call6) #2
  tail call void @free(i8* %call36) #2
  tail call void @free(i8* %call39) #2
  br label %cleanup

cleanup:                                          ; preds = %for.body, %if.then, %while.end
  ret void

ehcleanup:                                        ; preds = %lpad77, %lpad27
  %ehselector.slot29.0 = phi i32 [ %13, %lpad27 ], [ %22, %lpad77 ]
  %exn.slot28.0 = phi i8* [ %12, %lpad27 ], [ %21, %lpad77 ]
  %lpad.val149 = insertvalue { i8*, i32 } undef, i8* %exn.slot28.0, 0
  %lpad.val150 = insertvalue { i8*, i32 } %lpad.val149, i32 %ehselector.slot29.0, 1
  resume { i8*, i32 } %lpad.val150
}

; Function Attrs: uwtable
declare { i32*, i64 } @_ZN8sequence4packIiiNS_4getAIiiEEEE4_seqIT_EPS4_PbT0_S8_T1_(i32* %Out, i8* %Fl, i32 %s, i32 %e, i32* %f.coerce) local_unnamed_addr #4

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #5

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #5

declare i32 @__gxx_personality_v0(...)

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #5

; Function Attrs: argmemonly
declare void @llvm.detached.rethrow.sl_p0i8i32s(token, { i8*, i32 }) #6

; Function Attrs: nounwind
declare noalias i8* @malloc(i64) local_unnamed_addr #1

; Function Attrs: nounwind
declare void @free(i8* nocapture) local_unnamed_addr #1

!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C++ TBAA"}
!6 = !{!"tapir.loop.spawn.strategy", i32 1}
!7 = !{!8, !8, i64 0}
!8 = !{!"bool", !3, i64 0}
!27 = !{!"int", !3, i64 0}
!42 = !{!27, !27, i64 0}
!136 = distinct !{!136, !6}
!137 = distinct !{!137, !6}
!138 = distinct !{!138, !6}
