; RUN: llc %s -o - -verify-machineinstrs | FileCheck %s

target datalayout = "e-m:o-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "thumbv7s-apple-ios8.0.0"

%struct.cells = type { i32, i32, ptr }

@reg_len = external global i32, align 4

; The thumb2 size reduction pass commutes arguments to make the first src of an add the same as the dest.
; It needs to also move the internal flag when commuting arguments.

; CHECK-LABEL: @simulate

; Function Attrs: nounwind optsize ssp
define i32 @simulate(i32 %iterations, ptr nocapture %present, double %prob, ptr nocapture readonly %structure) {
entry:
  %0 = load i32, ptr @reg_len, align 4, !tbaa !3
  %sub = add nsw i32 %0, -1
  %div = sdiv i32 %sub, 31
  %rem2 = srem i32 %sub, 31
  %cmp35202 = icmp sgt i32 %rem2, 0
  br label %for.cond3.preheader

for.cond3.preheader:                              ; preds = %if.end85, %entry
  %call192 = tail call i32 @lrand48() #2
  br label %for.cond6.preheader

for.cond34.preheader:                             ; preds = %for.inc30
  br i1 %cmp35202, label %for.body37, label %for.end73

for.cond6.preheader:                              ; preds = %for.inc30, %for.cond3.preheader
  %call197 = phi i32 [ %call, %for.inc30 ], [ %call192, %for.cond3.preheader ]
  %i.0196 = phi i32 [ %inc31, %for.inc30 ], [ 0, %for.cond3.preheader ]
  %temp.1195 = phi ptr [ %5, %for.inc30 ], [ %present, %for.cond3.preheader ]
  %savefaulty.0194 = phi i32 [ %add12, %for.inc30 ], [ 0, %for.cond3.preheader ]
  %savef_free.0193 = phi i32 [ %add11, %for.inc30 ], [ 0, %for.cond3.preheader ]
  br label %for.body8

for.body8:                                        ; preds = %for.body8, %for.cond6.preheader
  %randv.0190 = phi i32 [ %call197, %for.cond6.preheader ], [ %shr, %for.body8 ]
  %j.0189 = phi i32 [ 0, %for.cond6.preheader ], [ %inc, %for.body8 ]
  %temp.2188 = phi ptr [ %temp.1195, %for.cond6.preheader ], [ %5, %for.body8 ]
  %savefaulty.1187 = phi i32 [ %savefaulty.0194, %for.cond6.preheader ], [ %add12, %for.body8 ]
  %savef_free.1186 = phi i32 [ %savef_free.0193, %for.cond6.preheader ], [ %add11, %for.body8 ]
  %1 = load i32, ptr %temp.2188, align 4, !tbaa !7
  %add11 = add nsw i32 %1, %savef_free.1186
  %faulty = getelementptr inbounds %struct.cells, ptr %temp.2188, i32 0, i32 1
  %2 = load i32, ptr %faulty, align 4, !tbaa !10
  %add12 = add nsw i32 %2, %savefaulty.1187
  %next = getelementptr inbounds %struct.cells, ptr %temp.2188, i32 0, i32 2
  %3 = load ptr, ptr %next, align 4, !tbaa !11
  %4 = load i32, ptr %3, align 4, !tbaa !7
  %add14 = add nsw i32 %4, %randv.0190
  %and = and i32 %add14, 1
  store i32 %and, ptr %temp.2188, align 4, !tbaa !7
  %call16 = tail call i32 @lrand48() #2
  %rem17 = srem i32 %call16, 1000
  %conv18 = sitofp i32 %rem17 to double
  %div19 = fdiv double %conv18, 1.000000e+03
  %cmp20 = fcmp olt double %div19, %prob
  %xor = zext i1 %cmp20 to i32
  %randv.1 = xor i32 %xor, %randv.0190
  %5 = load ptr, ptr %next, align 4, !tbaa !11
  %faulty25 = getelementptr inbounds %struct.cells, ptr %5, i32 0, i32 1
  %6 = load i32, ptr %faulty25, align 4, !tbaa !10
  %add26 = add nsw i32 %randv.1, %6
  %and27 = and i32 %add26, 1
  store i32 %and27, ptr %faulty, align 4, !tbaa !10
  %shr = ashr i32 %randv.0190, 1
  %inc = add nuw nsw i32 %j.0189, 1
  %exitcond = icmp eq i32 %inc, 31
  br i1 %exitcond, label %for.inc30, label %for.body8

for.inc30:                                        ; preds = %for.body8
  %inc31 = add nuw nsw i32 %i.0196, 1
  %cmp4 = icmp slt i32 %inc31, %div
  %call = tail call i32 @lrand48() #2
  br i1 %cmp4, label %for.cond6.preheader, label %for.cond34.preheader

for.body37:                                       ; preds = %for.body37, %for.cond34.preheader
  %randv.2207 = phi i32 [ %shr70, %for.body37 ], [ %call, %for.cond34.preheader ]
  %temp.3205 = phi ptr [ %9, %for.body37 ], [ %5, %for.cond34.preheader ]
  %.pre220 = getelementptr inbounds %struct.cells, ptr %temp.3205, i32 0, i32 1
  %next50 = getelementptr inbounds %struct.cells, ptr %temp.3205, i32 0, i32 2
  %7 = load ptr, ptr %next50, align 4, !tbaa !11
  %8 = load i32, ptr %7, align 4, !tbaa !7
  %add52 = add nsw i32 %8, %randv.2207
  %and53 = and i32 %add52, 1
  store i32 %and53, ptr %temp.3205, align 4, !tbaa !7
  %call55 = tail call i32 @lrand48() #2
  %rem56 = srem i32 %call55, 1000
  %conv57 = sitofp i32 %rem56 to double
  %div58 = fdiv double %conv57, 1.000000e+03
  %cmp59 = fcmp olt double %div58, %prob
  %xor62 = zext i1 %cmp59 to i32
  %randv.3 = xor i32 %xor62, %randv.2207
  %9 = load ptr, ptr %next50, align 4, !tbaa !11
  %faulty65 = getelementptr inbounds %struct.cells, ptr %9, i32 0, i32 1
  %10 = load i32, ptr %faulty65, align 4, !tbaa !10
  %add66 = add nsw i32 %randv.3, %10
  %and67 = and i32 %add66, 1
  store i32 %and67, ptr %.pre220, align 4, !tbaa !10
  %shr70 = ashr i32 %randv.2207, 1
  br label %for.body37

for.end73:                                        ; preds = %for.cond34.preheader
  %call74 = tail call i32 @lrand48() #2
  %11 = load i32, ptr @reg_len, align 4, !tbaa !3
  %sub75 = add nsw i32 %11, -1
  %arrayidx76 = getelementptr inbounds i8, ptr %structure, i32 %sub75
  %12 = load i8, ptr %arrayidx76, align 1, !tbaa !12
  %cmp78 = icmp eq i8 %12, 49
  br i1 %cmp78, label %if.then80, label %for.end73.if.end85_crit_edge

for.end73.if.end85_crit_edge:                     ; preds = %for.end73
  %.pre222 = getelementptr inbounds %struct.cells, ptr %5, i32 0, i32 1
  br label %if.end85

if.then80:                                        ; preds = %for.end73
  %13 = load i32, ptr %5, align 4, !tbaa !7
  %add82 = add nsw i32 %13, %add11
  %faulty83 = getelementptr inbounds %struct.cells, ptr %5, i32 0, i32 1
  %14 = load i32, ptr %faulty83, align 4, !tbaa !10
  %add84 = add nsw i32 %14, %add12
  br label %if.end85

if.end85:                                         ; preds = %if.then80, %for.end73.if.end85_crit_edge
  %faulty100.pre-phi = phi ptr [ %.pre222, %for.end73.if.end85_crit_edge ], [ %faulty83, %if.then80 ]
  %savef_free.5 = phi i32 [ %add11, %for.end73.if.end85_crit_edge ], [ %add82, %if.then80 ]
  %savefaulty.5 = phi i32 [ %add12, %for.end73.if.end85_crit_edge ], [ %add84, %if.then80 ]
  %add86 = add nsw i32 %savef_free.5, %call74
  %and87 = and i32 %add86, 1
  store i32 %and87, ptr %5, align 4, !tbaa !7
  %call89 = tail call i32 @lrand48() #2
  %rem90 = srem i32 %call89, 10000
  %conv91 = sitofp i32 %rem90 to double
  %div92 = fdiv double %conv91, 1.000000e+04
  %cmp93 = fcmp olt double %div92, %prob
  %xor96 = zext i1 %cmp93 to i32
  %randv.4 = xor i32 %xor96, %call74
  %add98 = add nsw i32 %randv.4, %savefaulty.5
  %and99 = and i32 %add98, 1
  store i32 %and99, ptr %faulty100.pre-phi, align 4, !tbaa !10
  br label %for.cond3.preheader
}

; Function Attrs: optsize
declare i32 @lrand48()

attributes #2 = { nounwind optsize }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"min_enum_size", i32 4}
!2 = !{!"clang version 3.7.0 (trunk 236243)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = !{!8, !4, i64 0}
!8 = !{!"cells", !4, i64 0, !4, i64 4, !9, i64 8}
!9 = !{!"any pointer", !5, i64 0}
!10 = !{!8, !4, i64 4}
!11 = !{!8, !9, i64 8}
!12 = !{!5, !5, i64 0}
