; RUN: opt -passes=loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -mtriple aarch64-unknown-linux-gnu -mattr=+sve -S < %s | FileCheck %s --check-prefix=CHECK-VF4UF1
; RUN: opt -passes=loop-vectorize -force-vector-width=4 -force-vector-interleave=2 -mtriple aarch64-unknown-linux-gnu -mattr=+sve -S < %s | FileCheck %s --check-prefix=CHECK-VF4UF2

; We vectorize this first order recurrence, with a set of insertelements for
; each unrolled part. Make sure these insertelements are generated in-order,
; because the shuffle of the first order recurrence will be added after the
; insertelement of the last part UF - 1, assuming the latter appears after the
; insertelements of all other parts.
;
; int PR33613(double *b, double j, int d) {
;   int a = 0;
;   for(int i = 0; i < 10240; i++, b+=25) {
;     double f = b[d]; // Scalarize to form insertelements
;     if (j * f)
;       a++;
;     j = f;
;   }
;   return a;
; }
;
define i32 @PR33613(ptr %b, double %j, i32 %d) #0 {
; CHECK-VF4UF2-LABEL: @PR33613
; CHECK-VF4UF2: vector.body
; CHECK-VF4UF2: %[[VEC_RECUR:.*]] = phi <vscale x 4 x double> [ {{.*}}, %vector.ph ], [ {{.*}}, %vector.body ]
; CHECK-VF4UF2: %[[SPLICE1:.*]] = call <vscale x 4 x double> @llvm.vector.splice.nxv4f64(<vscale x 4 x double> %[[VEC_RECUR]], <vscale x 4 x double> {{.*}}, i32 -1)
; CHECK-VF4UF2-NEXT: %[[SPLICE2:.*]] = call <vscale x 4 x double> @llvm.vector.splice.nxv4f64(<vscale x 4 x double> %{{.*}}, <vscale x 4 x double> %{{.*}}, i32 -1)
; CHECK-VF4UF2-NOT: insertelement <vscale x 4 x double>
; CHECK-VF4UF2: middle.block
entry:
  %idxprom = sext i32 %d to i64
  br label %for.body

for.cond.cleanup:
  %a.1.lcssa = phi i32 [ %a.1, %for.body ]
  ret i32 %a.1.lcssa

for.body:
  %b.addr.012 = phi ptr [ %b, %entry ], [ %add.ptr, %for.body ]
  %i.011 = phi i32 [ 0, %entry ], [ %inc1, %for.body ]
  %a.010 = phi i32 [ 0, %entry ], [ %a.1, %for.body ]
  %j.addr.09 = phi double [ %j, %entry ], [ %0, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %b.addr.012, i64 %idxprom
  %0 = load double, ptr %arrayidx, align 8
  %mul = fmul double %j.addr.09, %0
  %tobool = fcmp une double %mul, 0.000000e+00
  %inc = zext i1 %tobool to i32
  %a.1 = add nsw i32 %a.010, %inc
  %inc1 = add nuw nsw i32 %i.011, 1
  %add.ptr = getelementptr inbounds double, ptr %b.addr.012, i64 25
  %exitcond = icmp eq i32 %inc1, 10240
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !llvm.loop !0
}

; PR34711: given three consecutive instructions such that the first will be
; widened, the second is a cast that will be widened and needs to sink after the
; third, and the third is a first-order-recurring load that will be replicated
; instead of widened. Although the cast and the first instruction will both be
; widened, and are originally adjacent to each other, make sure the replicated
; load ends up appearing between them.
;
; void PR34711(short[2] *a, int *b, int *c, int n) {
;   for(int i = 0; i < n; i++) {
;     c[i] = 7;
;     b[i] = (a[i][0] * a[i][1]);
;   }
; }
;
; Check that the sext sank after the load in the vector loop.
define void @PR34711(ptr %a, ptr %b, ptr %c, i64 %n) #0 {
; CHECK-VF4UF1-LABEL: @PR34711
; CHECK-VF4UF1: vector.body
; CHECK-VF4UF1: %[[VEC_RECUR:.*]] = phi <vscale x 4 x i16> [ %vector.recur.init, %vector.ph ], [ %[[MGATHER:.*]], %vector.body ]
; CHECK-VF4UF1: %[[MGATHER]] = call <vscale x 4 x i16> @llvm.masked.gather.nxv4i16.nxv4p0(<vscale x 4 x ptr> {{.*}}, i32 2, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i16> poison)
; CHECK-VF4UF1-NEXT: %[[SPLICE:.*]] = call <vscale x 4 x i16> @llvm.vector.splice.nxv4i16(<vscale x 4 x i16> %[[VEC_RECUR]], <vscale x 4 x i16> %[[MGATHER]], i32 -1)
; CHECK-VF4UF1-NEXT: %[[SXT1:.*]] = sext <vscale x 4 x i16> %[[SPLICE]] to <vscale x 4 x i32>
; CHECK-VF4UF1-NEXT: %[[SXT2:.*]] = sext <vscale x 4 x i16> %[[MGATHER]] to <vscale x 4 x i32>
; CHECK-VF4UF1-NEXT: mul nsw <vscale x 4 x i32> %[[SXT2]], %[[SXT1]]
entry:
  %.pre = load i16, ptr %a
  br label %for.body

for.body:
  %0 = phi i16 [ %.pre, %entry ], [ %1, %for.body ]
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arraycidx = getelementptr inbounds i32, ptr %c, i64 %indvars.iv
  %cur.index = getelementptr inbounds [2 x i16], ptr %a, i64 %indvars.iv, i64 1
  store i32 7, ptr %arraycidx   ; 1st instruction, to be widened.
  %conv = sext i16 %0 to i32     ; 2nd, cast to sink after third.
  %1 = load i16, ptr %cur.index ; 3rd, first-order-recurring load not widened.
  %conv3 = sext i16 %1 to i32
  %mul = mul nsw i32 %conv3, %conv
  %arrayidx5 = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
  store i32 %mul, ptr %arrayidx5
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret void
}

attributes #0 = { vscale_range(1, 16) }
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
