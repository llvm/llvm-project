; RUN: opt %loadPolly -polly-delicm-max-ops=1000000 -polly-print-delicm -disable-output < %s | FileCheck %s
;
; This causes an assertion to fail on out-of-quota after 1000000 operations.
; (The error was specific to -polly-delicm-max-ops=1000000 and changes
;  in the implementation are likely to change the number of operations
;  up to the point where the error uses to occur)
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

define void @test(i32 %size, double* %lhs, i32 %lhsStride, double* %_rhs, double* %res, double %alpha) {
entry:
  br label %if.end

if.end:                                           ; preds = %entry
  %sub = add nsw i32 %size, -8
  %cmp.i.i.i = icmp sgt i32 %sub, 0
  %.sroa.speculated = select i1 %cmp.i.i.i, i32 %sub, i32 0
  %and36 = and i32 %.sroa.speculated, -2
  %cmp38463 = icmp sgt i32 %and36, 0
  br i1 %cmp38463, label %for.body40, label %for.cond177.preheader

for.cond177.preheader:                            ; preds = %for.cond.cleanup142, %if.end
  br label %for.body180

for.body40:                                       ; preds = %for.cond.cleanup142, %if.end
  %j.0464 = phi i32 [ %add55, %for.cond.cleanup142 ], [ 0, %if.end ]
  %mul41 = mul nsw i32 %j.0464, %lhsStride
  %add.ptr42 = getelementptr inbounds double, double* %lhs, i32 %mul41
  %add43 = or i32 %j.0464, 1
  %arrayidx46 = getelementptr inbounds double, double* %_rhs, i32 %j.0464
  %tmp = load double, double* %arrayidx46, align 8
  %mul47 = fmul double %tmp, %alpha
  %add55 = add nuw nsw i32 %j.0464, 2
  %arrayidx67 = getelementptr inbounds double, double* %res, i32 %j.0464
  %tmp1 = load double, double* %arrayidx67, align 8
  store double undef, double* %arrayidx67, align 8
  %arrayidx75 = getelementptr inbounds double, double* %res, i32 %add43
  %tmp2 = load double, double* %arrayidx75, align 8
  store double undef, double* %arrayidx75, align 8
  %arrayidx78 = getelementptr inbounds double, double* %add.ptr42, i32 %add43
  %tmp3 = load double, double* %arrayidx78, align 8
  %mul.i.i388 = fmul double %mul47, %tmp3
  %add88 = fadd double undef, 0.000000e+00
  %cmp120448 = icmp ult i32 %add55, %size
  br i1 %cmp120448, label %for.body122.preheader, label %for.cond.cleanup142

for.body122.preheader:                            ; preds = %for.body40
  %add.ptr117 = getelementptr inbounds double, double* %res, i32 %add55
  br label %for.body122

for.body122:                                      ; preds = %for.body122, %for.body122.preheader
  %i118.0455 = phi i32 [ %add137, %for.body122 ], [ %add55, %for.body122.preheader ]
  %resIt.0454 = phi double* [ %add.ptr135, %for.body122 ], [ %add.ptr117, %for.body122.preheader ]
  %ptmp2.0450 = phi double [ undef, %for.body122 ], [ 0.000000e+00, %for.body122.preheader ]
  %tmp4 = load double, double* %resIt.0454, align 8
  %add.i.i.i386 = fadd double undef, %tmp4
  %add.i.i.i384 = fadd double undef, %add.i.i.i386
  store double %add.i.i.i384, double* %resIt.0454, align 8
  %add.ptr135 = getelementptr inbounds double, double* %resIt.0454, i32 1
  %add137 = add nuw i32 %i118.0455, 1
  %exitcond469 = icmp eq i32 %add137, %size
  br i1 %exitcond469, label %for.cond.cleanup142.loopexit, label %for.body122

for.cond.cleanup142.loopexit:                     ; preds = %for.body122
  %.pre = load double, double* %arrayidx67, align 8
  %.pre471 = load double, double* %arrayidx75, align 8
  br label %for.cond.cleanup142

for.cond.cleanup142:                              ; preds = %for.cond.cleanup142.loopexit, %for.body40
  %tmp5 = phi double [ undef, %for.body40 ], [ %.pre471, %for.cond.cleanup142.loopexit ]
  %tmp6 = phi double [ undef, %for.body40 ], [ %.pre, %for.cond.cleanup142.loopexit ]
  %ptmp2.0.lcssa = phi double [ 0.000000e+00, %for.body40 ], [ undef, %for.cond.cleanup142.loopexit ]
  %add163 = fadd double %add88, %ptmp2.0.lcssa
  store double undef, double* %arrayidx67, align 8
  store double undef, double* %arrayidx75, align 8
  %cmp38 = icmp ult i32 %add55, %and36
  br i1 %cmp38, label %for.body40, label %for.cond177.preheader

for.cond.cleanup179:                              ; preds = %for.cond.cleanup198
  ret void

for.body180:                                      ; preds = %for.cond.cleanup198, %for.cond177.preheader
  %j176.0442 = phi i32 [ %add195, %for.cond.cleanup198 ], [ %and36, %for.cond177.preheader ]
  %add.ptr183 = getelementptr inbounds double, double* %lhs, i32 0
  %arrayidx185 = getelementptr inbounds double, double* %_rhs, i32 %j176.0442
  %tmp7 = load double, double* %arrayidx185, align 8
  %mul186 = fmul double %tmp7, %alpha
  %arrayidx189 = getelementptr inbounds double, double* %add.ptr183, i32 %j176.0442
  %tmp8 = load double, double* %arrayidx189, align 8
  %mul.i.i373 = fmul double %tmp8, %mul186
  %arrayidx192 = getelementptr inbounds double, double* %res, i32 %j176.0442
  %tmp9 = load double, double* %arrayidx192, align 8
  %add193 = fadd double %tmp9, %mul.i.i373
  store double %add193, double* %arrayidx192, align 8
  %add195 = add nuw nsw i32 %j176.0442, 1
  %cmp197438 = icmp slt i32 %add195, %size
  br i1 %cmp197438, label %for.body199, label %for.cond.cleanup198

for.cond.cleanup198.loopexit:                     ; preds = %for.body199
  %.pre472 = load double, double* %arrayidx192, align 8
  br label %for.cond.cleanup198

for.cond.cleanup198:                              ; preds = %for.cond.cleanup198.loopexit, %for.body180
  %tmp10 = phi double [ %add193, %for.body180 ], [ %.pre472, %for.cond.cleanup198.loopexit ]
  %t2187.0.lcssa = phi double [ 0.000000e+00, %for.body180 ], [ %add207, %for.cond.cleanup198.loopexit ]
  %add213 = fadd double %tmp10, undef
  store double %add213, double* %arrayidx192, align 8
  %exitcond468 = icmp eq i32 %add195, %size
  br i1 %exitcond468, label %for.cond.cleanup179, label %for.body180

for.body199:                                      ; preds = %for.body199, %for.body180
  %i194.0440 = phi i32 [ %inc209, %for.body199 ], [ %add195, %for.body180 ]
  %arrayidx200 = getelementptr inbounds double, double* %add.ptr183, i32 %i194.0440
  %tmp11 = load double, double* %arrayidx200, align 8
  %mul.i.i372 = fmul double %mul186, %tmp11
  %arrayidx202 = getelementptr inbounds double, double* %res, i32 %i194.0440
  %tmp12 = load double, double* %arrayidx202, align 8
  %add203 = fadd double %tmp12, %mul.i.i372
  store double %add203, double* %arrayidx202, align 8
  %arrayidx205 = getelementptr inbounds double, double* %_rhs, i32 %i194.0440
  %tmp13 = load double, double* %arrayidx200, align 8
  %tmp14 = load double, double* %arrayidx205, align 8
  %mul.i.i = fmul double %tmp13, %tmp14
  %add207 = fadd double undef, %mul.i.i
  %inc209 = add nuw nsw i32 %i194.0440, 1
  %exitcond = icmp eq i32 %inc209, %size
  br i1 %exitcond, label %for.cond.cleanup198.loopexit, label %for.body199
}


; CHECK: Zone not computed
