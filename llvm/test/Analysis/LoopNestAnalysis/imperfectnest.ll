; RUN: opt < %s -passes='print<loopnest>' -disable-output 2>&1 | FileCheck %s

; Test an imperfect 2-dim loop nest of the form:
;   for (int i = 0; i < nx; ++i) {
;     x[i] = i;
;     for (int j = 0; j < ny; ++j)
;       y[j][i] = x[i] + j;
;   }

define void @imperf_nest_1(i32 signext %nx, i32 signext %ny) {
; CHECK-LABEL: IsPerfect=false, Depth=2, OutermostLoop: imperf_nest_1_loop_i, Loops: ( imperf_nest_1_loop_i imperf_nest_1_loop_j )
entry:
  %0 = zext i32 %ny to i64
  %1 = zext i32 %nx to i64
  %2 = mul nuw i64 %0, %1
  %vla = alloca double, i64 %2, align 8
  %3 = zext i32 %ny to i64
  %vla1 = alloca double, i64 %3, align 8
  br label %imperf_nest_1_loop_i

imperf_nest_1_loop_i:
  %i2.0 = phi i32 [ 0, %entry ], [ %inc16, %for.inc15 ]
  %cmp = icmp slt i32 %i2.0, %nx
  br i1 %cmp, label %for.body, label %for.end17

for.body:
  %conv = sitofp i32 %i2.0 to double
  %idxprom = sext i32 %i2.0 to i64
  %arrayidx = getelementptr inbounds double, ptr %vla1, i64 %idxprom
  store double %conv, ptr %arrayidx, align 8
  br label %imperf_nest_1_loop_j

imperf_nest_1_loop_j:
  %j3.0 = phi i32 [ 0, %for.body ], [ %inc, %for.inc ]
  %cmp5 = icmp slt i32 %j3.0, %ny
  br i1 %cmp5, label %for.body7, label %for.end

for.body7:
  %idxprom8 = sext i32 %i2.0 to i64
  %arrayidx9 = getelementptr inbounds double, ptr %vla1, i64 %idxprom8
  %4 = load double, ptr %arrayidx9, align 8
  %conv10 = sitofp i32 %j3.0 to double
  %add = fadd double %4, %conv10
  %idxprom11 = sext i32 %j3.0 to i64
  %5 = mul nsw i64 %idxprom11, %1
  %arrayidx12 = getelementptr inbounds double, ptr %vla, i64 %5
  %idxprom13 = sext i32 %i2.0 to i64
  %arrayidx14 = getelementptr inbounds double, ptr %arrayidx12, i64 %idxprom13
  store double %add, ptr %arrayidx14, align 8
  br label %for.inc

for.inc:
  %inc = add nsw i32 %j3.0, 1
  br label %imperf_nest_1_loop_j

for.end:
  br label %for.inc15

for.inc15:
  %inc16 = add nsw i32 %i2.0, 1
  br label %imperf_nest_1_loop_i

for.end17:
  ret void
}

; Test an imperfect 2-dim loop nest of the form:
;   for (int i = 0; i < nx; ++i) {
;     for (int j = 0; j < ny; ++j)
;       y[j][i] = x[i] + j;
;     y[0][i] += i;
;   }

define void @imperf_nest_2(i32 signext %nx, i32 signext %ny) {
; CHECK-LABEL: IsPerfect=false, Depth=2, OutermostLoop: imperf_nest_2_loop_i, Loops: ( imperf_nest_2_loop_i imperf_nest_2_loop_j )
entry:
  %0 = zext i32 %ny to i64
  %1 = zext i32 %nx to i64
  %2 = mul nuw i64 %0, %1
  %vla = alloca double, i64 %2, align 8
  %3 = zext i32 %ny to i64
  %vla1 = alloca double, i64 %3, align 8
  br label %imperf_nest_2_loop_i

imperf_nest_2_loop_i:
  %i2.0 = phi i32 [ 0, %entry ], [ %inc17, %for.inc16 ]
  %cmp = icmp slt i32 %i2.0, %nx
  br i1 %cmp, label %for.body, label %for.end18
 
for.body:
  br label %imperf_nest_2_loop_j

imperf_nest_2_loop_j:
  %j3.0 = phi i32 [ 0, %for.body ], [ %inc, %for.inc ]
  %cmp5 = icmp slt i32 %j3.0, %ny
  br i1 %cmp5, label %for.body6, label %for.end

for.body6:
  %idxprom = sext i32 %i2.0 to i64
  %arrayidx = getelementptr inbounds double, ptr %vla1, i64 %idxprom
  %4 = load double, ptr %arrayidx, align 8
  %conv = sitofp i32 %j3.0 to double
  %add = fadd double %4, %conv
  %idxprom7 = sext i32 %j3.0 to i64
  %5 = mul nsw i64 %idxprom7, %1
  %arrayidx8 = getelementptr inbounds double, ptr %vla, i64 %5
  %idxprom9 = sext i32 %i2.0 to i64
  %arrayidx10 = getelementptr inbounds double, ptr %arrayidx8, i64 %idxprom9
  store double %add, ptr %arrayidx10, align 8
  br label %for.inc

for.inc:
  %inc = add nsw i32 %j3.0, 1
  br label %imperf_nest_2_loop_j

for.end:
  %conv11 = sitofp i32 %i2.0 to double
  %6 = mul nsw i64 0, %1
  %arrayidx12 = getelementptr inbounds double, ptr %vla, i64 %6
  %idxprom13 = sext i32 %i2.0 to i64
  %arrayidx14 = getelementptr inbounds double, ptr %arrayidx12, i64 %idxprom13
  %7 = load double, ptr %arrayidx14, align 8
  %add15 = fadd double %7, %conv11
  store double %add15, ptr %arrayidx14, align 8
  br label %for.inc16

for.inc16:
  %inc17 = add nsw i32 %i2.0, 1
  br label %imperf_nest_2_loop_i

for.end18:
  ret void
}

; Test an imperfect 2-dim loop nest of the form:
;   for (i = 0; i < nx; ++i) {
;     for (j = 0; j < ny-nk; ++j)
;       y[i][j] = x[i] + j;
;     for (j = ny-nk; j < ny; ++j)
;       y[i][j] = x[i] - j;
;   }

define void @imperf_nest_3(i32 signext %nx, i32 signext %ny, i32 signext %nk) {
; CHECK-LABEL: IsPerfect=false, Depth=2, OutermostLoop: imperf_nest_3_loop_i, Loops: ( imperf_nest_3_loop_i imperf_nest_3_loop_j imperf_nest_3_loop_k )
entry:
  %0 = zext i32 %nx to i64
  %1 = zext i32 %ny to i64
  %2 = mul nuw i64 %0, %1
  %vla = alloca double, i64 %2, align 8
  %3 = zext i32 %ny to i64
  %vla1 = alloca double, i64 %3, align 8
  br label %imperf_nest_3_loop_i

imperf_nest_3_loop_i:                                         ; preds = %for.inc25, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc26, %for.inc25 ]
  %cmp = icmp slt i32 %i.0, %nx
  br i1 %cmp, label %for.body, label %for.end27

for.body:                                         ; preds = %for.cond
  br label %imperf_nest_3_loop_j

imperf_nest_3_loop_j:                                        ; preds = %for.inc, %for.body
  %j.0 = phi i32 [ 0, %for.body ], [ %inc, %for.inc ]
  %sub = sub nsw i32 %ny, %nk
  %cmp3 = icmp slt i32 %j.0, %sub
  br i1 %cmp3, label %for.body4, label %for.end

for.body4:                                        ; preds = %imperf_nest_3_loop_j
  %idxprom = sext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds double, ptr %vla1, i64 %idxprom
  %4 = load double, ptr %arrayidx, align 8
  %conv = sitofp i32 %j.0 to double
  %add = fadd double %4, %conv
  %idxprom5 = sext i32 %i.0 to i64
  %5 = mul nsw i64 %idxprom5, %1
  %arrayidx6 = getelementptr inbounds double, ptr %vla, i64 %5
  %idxprom7 = sext i32 %j.0 to i64
  %arrayidx8 = getelementptr inbounds double, ptr %arrayidx6, i64 %idxprom7
  store double %add, ptr %arrayidx8, align 8
  br label %for.inc

for.inc:                                          ; preds = %for.body4
  %inc = add nsw i32 %j.0, 1
  br label %imperf_nest_3_loop_j

for.end:                                          ; preds = %imperf_nest_3_loop_j
  %sub9 = sub nsw i32 %ny, %nk
  br label %imperf_nest_3_loop_k

imperf_nest_3_loop_k:                                       ; preds = %for.inc22, %for.end
  %j.1 = phi i32 [ %sub9, %for.end ], [ %inc23, %for.inc22 ]
  %cmp11 = icmp slt i32 %j.1, %ny
  br i1 %cmp11, label %for.body13, label %for.end24

for.body13:                                       ; preds = %imperf_nest_3_loop_k
  %idxprom14 = sext i32 %i.0 to i64
  %arrayidx15 = getelementptr inbounds double, ptr %vla1, i64 %idxprom14
  %6 = load double, ptr %arrayidx15, align 8
  %conv16 = sitofp i32 %j.1 to double
  %sub17 = fsub double %6, %conv16
  %idxprom18 = sext i32 %i.0 to i64
  %7 = mul nsw i64 %idxprom18, %1
  %arrayidx19 = getelementptr inbounds double, ptr %vla, i64 %7
  %idxprom20 = sext i32 %j.1 to i64
  %arrayidx21 = getelementptr inbounds double, ptr %arrayidx19, i64 %idxprom20
  store double %sub17, ptr %arrayidx21, align 8
  br label %for.inc22

for.inc22:                                        ; preds = %for.body13
  %inc23 = add nsw i32 %j.1, 1
  br label %imperf_nest_3_loop_k

for.end24:                                        ; preds = %imperf_nest_3_loop_k
  br label %for.inc25

for.inc25:                                        ; preds = %for.end24
  %inc26 = add nsw i32 %i.0, 1
  br label %imperf_nest_3_loop_i

for.end27:                                        ; preds = %for.cond
  ret void
}

; Test an imperfect loop nest of the form:
;   for (i = 0; i < nx; ++i) {
;     for (j = 0; j < ny-nk; ++j)
;       for (k = 0; k < nk; ++k)
;         y[i][j][k] = x[i+j] + k;
;     for (j = ny-nk; j < ny; ++j)
;       y[i][j][0] = x[i] - j;
;   }

define void @imperf_nest_4(i32 signext %nx, i32 signext %ny, i32 signext %nk) {
; CHECK-LABEL: IsPerfect=false, Depth=2, OutermostLoop: imperf_nest_4_loop_j, Loops: ( imperf_nest_4_loop_j imperf_nest_4_loop_k )
; CHECK-LABEL: IsPerfect=false, Depth=3, OutermostLoop: imperf_nest_4_loop_i, Loops: ( imperf_nest_4_loop_i imperf_nest_4_loop_j imperf_nest_4_loop_j2 imperf_nest_4_loop_k )
entry:
  %0 = zext i32 %nx to i64
  %1 = zext i32 %ny to i64
  %2 = zext i32 %nk to i64
  %3 = mul nuw i64 %0, %1
  %4 = mul nuw i64 %3, %2
  %vla = alloca double, i64 %4, align 8
  %5 = zext i32 %ny to i64
  %vla1 = alloca double, i64 %5, align 8
  %cmp5 = icmp slt i32 0, %nx
  br i1 %cmp5, label %imperf_nest_4_loop_i.lr.ph, label %for.end37

imperf_nest_4_loop_i.lr.ph:
  br label %imperf_nest_4_loop_i

imperf_nest_4_loop_i:
  %i.0 = phi i32 [ 0, %imperf_nest_4_loop_i.lr.ph ], [ %inc36, %for.inc35 ]
  %sub2 = sub nsw i32 %ny, %nk
  %cmp33 = icmp slt i32 0, %sub2
  br i1 %cmp33, label %imperf_nest_4_loop_j.lr.ph, label %for.end17

imperf_nest_4_loop_j.lr.ph:
  br label %imperf_nest_4_loop_j

imperf_nest_4_loop_j:
  %j.0 = phi i32 [ 0, %imperf_nest_4_loop_j.lr.ph ], [ %inc16, %for.inc15 ]
  %cmp61 = icmp slt i32 0, %nk
  br i1 %cmp61, label %imperf_nest_4_loop_k.lr.ph, label %for.end

imperf_nest_4_loop_k.lr.ph:
  br label %imperf_nest_4_loop_k

imperf_nest_4_loop_k:
  %k.0 = phi i32 [ 0, %imperf_nest_4_loop_k.lr.ph ], [ %inc, %for.inc ]
  %add = add nsw i32 %i.0, %j.0
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds double, ptr %vla1, i64 %idxprom
  %6 = load double, ptr %arrayidx, align 8
  %conv = sitofp i32 %k.0 to double
  %add8 = fadd double %6, %conv
  %idxprom9 = sext i32 %i.0 to i64
  %7 = mul nuw i64 %1, %2
  %8 = mul nsw i64 %idxprom9, %7
  %arrayidx10 = getelementptr inbounds double, ptr %vla, i64 %8
  %idxprom11 = sext i32 %j.0 to i64
  %9 = mul nsw i64 %idxprom11, %2
  %arrayidx12 = getelementptr inbounds double, ptr %arrayidx10, i64 %9
  %idxprom13 = sext i32 %k.0 to i64
  %arrayidx14 = getelementptr inbounds double, ptr %arrayidx12, i64 %idxprom13
  store double %add8, ptr %arrayidx14, align 8
  br label %for.inc

for.inc:
  %inc = add nsw i32 %k.0, 1
  %cmp6 = icmp slt i32 %inc, %nk
  br i1 %cmp6, label %imperf_nest_4_loop_k, label %for.cond5.for.end_crit_edge

for.cond5.for.end_crit_edge:
  br label %for.end

for.end:
  br label %for.inc15

for.inc15:
  %inc16 = add nsw i32 %j.0, 1
  %sub = sub nsw i32 %ny, %nk
  %cmp3 = icmp slt i32 %inc16, %sub
  br i1 %cmp3, label %imperf_nest_4_loop_j, label %for.cond2.for.end17_crit_edge

for.cond2.for.end17_crit_edge:
  br label %for.end17

for.end17:
  %sub18 = sub nsw i32 %ny, %nk
  %cmp204 = icmp slt i32 %sub18, %ny
  br i1 %cmp204, label %imperf_nest_4_loop_j2.lr.ph, label %for.end34

imperf_nest_4_loop_j2.lr.ph:
  br label %imperf_nest_4_loop_j2

imperf_nest_4_loop_j2:
  %j.1 = phi i32 [ %sub18, %imperf_nest_4_loop_j2.lr.ph ], [ %inc33, %for.inc32 ]
  %idxprom23 = sext i32 %i.0 to i64
  %arrayidx24 = getelementptr inbounds double, ptr %vla1, i64 %idxprom23
  %10 = load double, ptr %arrayidx24, align 8
  %conv25 = sitofp i32 %j.1 to double
  %sub26 = fsub double %10, %conv25
  %idxprom27 = sext i32 %i.0 to i64
  %idxprom29 = sext i32 %j.1 to i64
  %11 = mul nsw i64 %idxprom29, %2
  %12 = mul nuw i64 %1, %2
  %13 = mul nsw i64 %idxprom27, %12
  %arrayidx28 = getelementptr inbounds double, ptr %vla, i64 %13
  %arrayidx30 = getelementptr inbounds double, ptr %arrayidx28, i64 %11
  %arrayidx31 = getelementptr inbounds double, ptr %arrayidx30, i64 0
  store double %sub26, ptr %arrayidx31, align 8
  br label %for.inc32

for.inc32:
  %inc33 = add nsw i32 %j.1, 1
  %cmp20 = icmp slt i32 %inc33, %ny
  br i1 %cmp20, label %imperf_nest_4_loop_j2, label %for.cond19.for.end34_crit_edge

for.cond19.for.end34_crit_edge:
  br label %for.end34

for.end34:                   
  br label %for.inc35

for.inc35:                   
  %inc36 = add nsw i32 %i.0, 1
  %cmp = icmp slt i32 %inc36, %nx
  br i1 %cmp, label %imperf_nest_4_loop_i, label %for.cond.for.end37_crit_edge

for.cond.for.end37_crit_edge:
  br label %for.end37

for.end37:
  ret void
}

; Test an imperfect loop nest of the form:
;   for (int i = 0; i < nx; ++i)
;     if (i > 5) {
;       for (int j = 0; j < ny; ++j)
;         y[j][i] = x[i][j] + j;
;     }

define void @imperf_nest_5(ptr %y, ptr %x, i32 signext %nx, i32 signext %ny) {
; CHECK-LABEL: IsPerfect=false, Depth=2, OutermostLoop: imperf_nest_5_loop_i, Loops: ( imperf_nest_5_loop_i imperf_nest_5_loop_j )
entry:
  %cmp2 = icmp slt i32 0, %nx
  br i1 %cmp2, label %imperf_nest_5_loop_i.lr.ph, label %for.end13

imperf_nest_5_loop_i.lr.ph:
  br label %imperf_nest_5_loop_i

imperf_nest_5_loop_i:      
  %i.0 = phi i32 [ 0, %imperf_nest_5_loop_i.lr.ph ], [ %inc12, %for.inc11 ]
  %cmp1 = icmp sgt i32 %i.0, 5
  br i1 %cmp1, label %if.then, label %if.end

if.then:         
  %cmp31 = icmp slt i32 0, %ny
  br i1 %cmp31, label %imperf_nest_5_loop_j.lr.ph, label %for.end

imperf_nest_5_loop_j.lr.ph:
  br label %imperf_nest_5_loop_j

imperf_nest_5_loop_j:      
  %j.0 = phi i32 [ 0, %imperf_nest_5_loop_j.lr.ph ], [ %inc, %for.inc ]
  %idxprom = sext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds ptr, ptr %x, i64 %idxprom
  %0 = load ptr, ptr %arrayidx, align 8
  %idxprom5 = sext i32 %j.0 to i64
  %arrayidx6 = getelementptr inbounds i32, ptr %0, i64 %idxprom5
  %1 = load i32, ptr %arrayidx6, align 4
  %add = add nsw i32 %1, %j.0
  %idxprom7 = sext i32 %j.0 to i64
  %arrayidx8 = getelementptr inbounds ptr, ptr %y, i64 %idxprom7
  %2 = load ptr, ptr %arrayidx8, align 8
  %idxprom9 = sext i32 %i.0 to i64
  %arrayidx10 = getelementptr inbounds i32, ptr %2, i64 %idxprom9
  store i32 %add, ptr %arrayidx10, align 4
  br label %for.inc

for.inc:
  %inc = add nsw i32 %j.0, 1
  %cmp3 = icmp slt i32 %inc, %ny
  br i1 %cmp3, label %imperf_nest_5_loop_j, label %for.cond2.for.end_crit_edge

for.cond2.for.end_crit_edge:
  br label %for.end

for.end:                    
  br label %if.end

if.end:                     
  br label %for.inc11

for.inc11:                  
  %inc12 = add nsw i32 %i.0, 1
  %cmp = icmp slt i32 %inc12, %nx
  br i1 %cmp, label %imperf_nest_5_loop_i, label %for.cond.for.end13_crit_edge

for.cond.for.end13_crit_edge:
  br label %for.end13

for.end13:                   
  ret void
}
