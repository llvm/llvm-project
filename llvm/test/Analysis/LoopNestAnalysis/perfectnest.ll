; RUN: opt < %s -passes='print<loopnest>' -disable-output 2>&1 | FileCheck %s

; Test a perfect 2-dim loop nest of the form:
;   for(i=0; i<nx; ++i)
;     for(j=0; j<nx; ++j)
;       y[i][j] = x[i][j];

define void @perf_nest_2D_1(i32** %y, i32** %x, i64 signext %nx, i64 signext %ny) {
; CHECK-LABEL: IsPerfect=true, Depth=1, OutermostLoop: perf_nest_2D_1_loop_j, Loops: ( perf_nest_2D_1_loop_j )
; CHECK-LABEL: IsPerfect=true, Depth=2, OutermostLoop: perf_nest_2D_1_loop_i, Loops: ( perf_nest_2D_1_loop_i perf_nest_2D_1_loop_j )
entry:
  br label %perf_nest_2D_1_loop_i

perf_nest_2D_1_loop_i:
  %i = phi i64 [ 0, %entry ], [ %inc13, %inc_i ]
  %cmp21 = icmp slt i64 0, %ny
  br i1 %cmp21, label %perf_nest_2D_1_loop_j, label %inc_i

perf_nest_2D_1_loop_j:
  %j = phi i64 [ 0, %perf_nest_2D_1_loop_i ], [ %inc, %inc_j ]
  %arrayidx = getelementptr inbounds i32*, i32** %x, i64 %j
  %0 = load i32*, i32** %arrayidx, align 8
  %arrayidx6 = getelementptr inbounds i32, i32* %0, i64 %j
  %1 = load i32, i32* %arrayidx6, align 4
  %arrayidx8 = getelementptr inbounds i32*, i32** %y, i64 %j
  %2 = load i32*, i32** %arrayidx8, align 8
  %arrayidx11 = getelementptr inbounds i32, i32* %2, i64 %i
  store i32 %1, i32* %arrayidx11, align 4
  br label %inc_j

inc_j:
  %inc = add nsw i64 %j, 1
  %cmp2 = icmp slt i64 %inc, %ny
  br i1 %cmp2, label %perf_nest_2D_1_loop_j, label %inc_i

inc_i:
  %inc13 = add nsw i64 %i, 1
  %cmp = icmp slt i64 %inc13, %nx
  br i1 %cmp, label %perf_nest_2D_1_loop_i, label %perf_nest_2D_1_loop_i_end

perf_nest_2D_1_loop_i_end:
  ret void
}

; Test a perfect 2-dim loop nest of the form:
;   for (i=0; i<100; ++i)
;     for (j=0; j<100; ++j)
;       y[i][j] = x[i][j];
define void @perf_nest_2D_2(i32** %y, i32** %x) {
; CHECK-LABEL: IsPerfect=true, Depth=1, OutermostLoop: perf_nest_2D_2_loop_j, Loops: ( perf_nest_2D_2_loop_j )
; CHECK-LABEL: IsPerfect=true, Depth=2, OutermostLoop: perf_nest_2D_2_loop_i, Loops: ( perf_nest_2D_2_loop_i perf_nest_2D_2_loop_j )
entry:
  br label %perf_nest_2D_2_loop_i

perf_nest_2D_2_loop_i:
  %i = phi i64 [ 0, %entry ], [ %inc13, %inc_i ]
  br label %perf_nest_2D_2_loop_j

perf_nest_2D_2_loop_j:
  %j = phi i64 [ 0, %perf_nest_2D_2_loop_i ], [ %inc, %inc_j ]
  %arrayidx = getelementptr inbounds i32*, i32** %x, i64 %j
  %0 = load i32*, i32** %arrayidx, align 8
  %arrayidx6 = getelementptr inbounds i32, i32* %0, i64 %j
  %1 = load i32, i32* %arrayidx6, align 4
  %arrayidx8 = getelementptr inbounds i32*, i32** %y, i64 %j
  %2 = load i32*, i32** %arrayidx8, align 8
  %arrayidx11 = getelementptr inbounds i32, i32* %2, i64 %i
  store i32 %1, i32* %arrayidx11, align 4
  br label %inc_j

inc_j:
  %inc = add nsw i64 %j, 1
  %cmp2 = icmp slt i64 %inc, 100
  br i1 %cmp2, label %perf_nest_2D_2_loop_j, label %loop_j_end

loop_j_end:
  br label %inc_i

inc_i:
  %inc13 = add nsw i64 %i, 1
  %cmp = icmp slt i64 %inc13, 100
  br i1 %cmp, label %perf_nest_2D_2_loop_i, label %perf_nest_2D_2_loop_i_end

perf_nest_2D_2_loop_i_end:
  ret void
}

; Test a perfect 3-dim loop nest of the form:
;   for (i=0; i<nx; ++i)
;     for (j=0; j<ny; ++j)
;       for (k=0; j<nk; ++k)
;          y[j][j][k] = x[i][j][k];
;

define void @perf_nest_3D_1(i32*** %y, i32*** %x, i32 signext %nx, i32 signext %ny, i32 signext %nk) {
; CHECK-LABEL: IsPerfect=true, Depth=1, OutermostLoop: perf_nest_3D_1_loop_k, Loops: ( perf_nest_3D_1_loop_k )
; CHECK-NEXT: IsPerfect=true, Depth=2, OutermostLoop: perf_nest_3D_1_loop_j, Loops: ( perf_nest_3D_1_loop_j perf_nest_3D_1_loop_k )
; CHECK-NEXT: IsPerfect=true, Depth=3, OutermostLoop: perf_nest_3D_1_loop_i, Loops: ( perf_nest_3D_1_loop_i perf_nest_3D_1_loop_j perf_nest_3D_1_loop_k )
entry:
  br label %perf_nest_3D_1_loop_i

perf_nest_3D_1_loop_i:
  %i = phi i32 [ 0, %entry ], [ %inci, %for.inci ]
  %cmp21 = icmp slt i32 0, %ny
  br i1 %cmp21, label %perf_nest_3D_1_loop_j, label %for.inci

perf_nest_3D_1_loop_j:
  %j = phi i32 [ 0, %perf_nest_3D_1_loop_i ], [ %incj, %for.incj ]
  %cmp22 = icmp slt i32 0, %nk
  br i1 %cmp22, label %perf_nest_3D_1_loop_k, label %for.incj

perf_nest_3D_1_loop_k:
  %k = phi i32 [ 0, %perf_nest_3D_1_loop_j ], [ %inck, %for.inck ]
  %idxprom = sext i32 %i to i64
  %arrayidx = getelementptr inbounds i32**, i32*** %x, i64 %idxprom
  %0 = load i32**, i32*** %arrayidx, align 8
  %idxprom7 = sext i32 %j to i64
  %arrayidx8 = getelementptr inbounds i32*, i32** %0, i64 %idxprom7
  %1 = load i32*, i32** %arrayidx8, align 8
  %idxprom9 = sext i32 %k to i64
  %arrayidx10 = getelementptr inbounds i32, i32* %1, i64 %idxprom9
  %2 = load i32, i32* %arrayidx10, align 4
  %idxprom11 = sext i32 %j to i64
  %arrayidx12 = getelementptr inbounds i32**, i32*** %y, i64 %idxprom11
  %3 = load i32**, i32*** %arrayidx12, align 8
  %idxprom13 = sext i32 %j to i64
  %arrayidx14 = getelementptr inbounds i32*, i32** %3, i64 %idxprom13
  %4 = load i32*, i32** %arrayidx14, align 8
  %idxprom15 = sext i32 %k to i64
  %arrayidx16 = getelementptr inbounds i32, i32* %4, i64 %idxprom15
  store i32 %2, i32* %arrayidx16, align 4
  br label %for.inck

for.inck:
  %inck = add nsw i32 %k, 1
  %cmp5 = icmp slt i32 %inck, %nk
  br i1 %cmp5, label %perf_nest_3D_1_loop_k, label %for.incj

for.incj:
  %incj = add nsw i32 %j, 1
  %cmp2 = icmp slt i32 %incj, %ny
  br i1 %cmp2, label %perf_nest_3D_1_loop_j, label %for.inci

for.inci:
  %inci = add nsw i32 %i, 1
  %cmp = icmp slt i32 %inci, %nx
  br i1 %cmp, label %perf_nest_3D_1_loop_i, label %perf_nest_3D_1_loop_i_end

perf_nest_3D_1_loop_i_end:
  ret void
}

; Test a perfect 3-dim loop nest of the form:
;   for (i=0; i<100; ++i)
;     for (j=0; j<100; ++j)
;       for (k=0; j<100; ++k)
;          y[j][j][k] = x[i][j][k];
;

define void @perf_nest_3D_2(i32*** %y, i32*** %x) {
; CHECK-LABEL: IsPerfect=true, Depth=1, OutermostLoop: perf_nest_3D_2_loop_k, Loops: ( perf_nest_3D_2_loop_k )
; CHECK-NEXT: IsPerfect=true, Depth=2, OutermostLoop: perf_nest_3D_2_loop_j, Loops: ( perf_nest_3D_2_loop_j perf_nest_3D_2_loop_k )
; CHECK-NEXT: IsPerfect=true, Depth=3, OutermostLoop: perf_nest_3D_2_loop_i, Loops: ( perf_nest_3D_2_loop_i perf_nest_3D_2_loop_j perf_nest_3D_2_loop_k )
entry:
  br label %perf_nest_3D_2_loop_i

perf_nest_3D_2_loop_i:
  %i = phi i32 [ 0, %entry ], [ %inci, %for.inci ]
  br label %perf_nest_3D_2_loop_j

perf_nest_3D_2_loop_j:
  %j = phi i32 [ 0, %perf_nest_3D_2_loop_i ], [ %incj, %for.incj ]
  br label %perf_nest_3D_2_loop_k

perf_nest_3D_2_loop_k:
  %k = phi i32 [ 0, %perf_nest_3D_2_loop_j ], [ %inck, %for.inck ]
  %idxprom = sext i32 %i to i64
  %arrayidx = getelementptr inbounds i32**, i32*** %x, i64 %idxprom
  %0 = load i32**, i32*** %arrayidx, align 8
  %idxprom7 = sext i32 %j to i64
  %arrayidx8 = getelementptr inbounds i32*, i32** %0, i64 %idxprom7
  %1 = load i32*, i32** %arrayidx8, align 8
  %idxprom9 = sext i32 %k to i64
  %arrayidx10 = getelementptr inbounds i32, i32* %1, i64 %idxprom9
  %2 = load i32, i32* %arrayidx10, align 4
  %idxprom11 = sext i32 %j to i64
  %arrayidx12 = getelementptr inbounds i32**, i32*** %y, i64 %idxprom11
  %3 = load i32**, i32*** %arrayidx12, align 8
  %idxprom13 = sext i32 %j to i64
  %arrayidx14 = getelementptr inbounds i32*, i32** %3, i64 %idxprom13
  %4 = load i32*, i32** %arrayidx14, align 8
  %idxprom15 = sext i32 %k to i64
  %arrayidx16 = getelementptr inbounds i32, i32* %4, i64 %idxprom15
  store i32 %2, i32* %arrayidx16, align 4
  br label %for.inck

for.inck:
  %inck = add nsw i32 %k, 1
  %cmp5 = icmp slt i32 %inck, 100
  br i1 %cmp5, label %perf_nest_3D_2_loop_k, label %loop_k_end

loop_k_end:
  br label %for.incj

for.incj:
  %incj = add nsw i32 %j, 1
  %cmp2 = icmp slt i32 %incj, 100
  br i1 %cmp2, label %perf_nest_3D_2_loop_j, label %loop_j_end

loop_j_end:
  br label %for.inci

for.inci:
  %inci = add nsw i32 %i, 1
  %cmp = icmp slt i32 %inci, 100
  br i1 %cmp, label %perf_nest_3D_2_loop_i, label %perf_nest_3D_2_loop_i_end

perf_nest_3D_2_loop_i_end:
  ret void
}

; Test a perfect loop nest with a live out reduction:
;   for (i = 0; i<ni; ++i)
;     if (0<nj) { // guard branch for the j-loop
;       for (j=0; j<nj; j+=1)
;         x+=(i+j);
;     }
;   return x;

define signext i32 @perf_nest_live_out(i32 signext %x, i32 signext %ni, i32 signext %nj) {
; CHECK-LABEL: IsPerfect=true, Depth=1, OutermostLoop: perf_nest_live_out_loop_j, Loops: ( perf_nest_live_out_loop_j )
; CHECK-LABEL: IsPerfect=true, Depth=2, OutermostLoop: perf_nest_live_out_loop_i, Loops: ( perf_nest_live_out_loop_i perf_nest_live_out_loop_j )
entry:
  %cmp4 = icmp slt i32 0, %ni
  br i1 %cmp4, label %perf_nest_live_out_loop_i.lr.ph, label %for.end7

perf_nest_live_out_loop_i.lr.ph:
  br label %perf_nest_live_out_loop_i

perf_nest_live_out_loop_i:
  %x.addr.06 = phi i32 [ %x, %perf_nest_live_out_loop_i.lr.ph ], [ %x.addr.1.lcssa, %for.inc5 ]
  %i.05 = phi i32 [ 0, %perf_nest_live_out_loop_i.lr.ph ], [ %inc6, %for.inc5 ]
  %cmp21 = icmp slt i32 0, %nj
  br i1 %cmp21, label %perf_nest_live_out_loop_j.lr.ph, label %for.inc5

perf_nest_live_out_loop_j.lr.ph:
  br label %perf_nest_live_out_loop_j

perf_nest_live_out_loop_j:
  %x.addr.13 = phi i32 [ %x.addr.06, %perf_nest_live_out_loop_j.lr.ph ], [ %add4, %perf_nest_live_out_loop_j ]
  %j.02 = phi i32 [ 0, %perf_nest_live_out_loop_j.lr.ph ], [ %inc, %perf_nest_live_out_loop_j ]
  %add = add nsw i32 %i.05, %j.02
  %add4 = add nsw i32 %x.addr.13, %add
  %inc = add nsw i32 %j.02, 1
  %cmp2 = icmp slt i32 %inc, %nj
  br i1 %cmp2, label %perf_nest_live_out_loop_j, label %for.cond1.for.inc5_crit_edge

for.cond1.for.inc5_crit_edge:
  %split = phi i32 [ %add4, %perf_nest_live_out_loop_j ]
  br label %for.inc5

for.inc5:
  %x.addr.1.lcssa = phi i32 [ %split, %for.cond1.for.inc5_crit_edge ], [ %x.addr.06, %perf_nest_live_out_loop_i ]
  %inc6 = add nsw i32 %i.05, 1
  %cmp = icmp slt i32 %inc6, %ni
  br i1 %cmp, label %perf_nest_live_out_loop_i, label %for.cond.for.end7_crit_edge

for.cond.for.end7_crit_edge:
  %split7 = phi i32 [ %x.addr.1.lcssa, %for.inc5 ]
  br label %for.end7

for.end7:
  %x.addr.0.lcssa = phi i32 [ %split7, %for.cond.for.end7_crit_edge ], [ %x, %entry ]
  ret i32 %x.addr.0.lcssa
}
