; RUN: opt -passes='loop-mssa(licm)' -S %s | FileCheck %s --check-prefixes CHECK,LICM
; RUN: opt -passes='loop-mssa(lnicm)' -S %s | FileCheck %s --check-prefixes CHECK,LNICM

; This test represents the following function:
;
; double sin(double);
; int abs(int);
; double test(double x, int y[10]) {
;   double t = 0; int s = 0;
;   for (int i = 0; i < 10; i++) {
;     for (int j = 0; j < 10; j++) {
;       t = sin(x);
;       s = abs(i);
;     }
;     y[i] = s;
;   }
;   return t;
; }
;
; We only want to sink the call of sin out of the loop nest.
; LICM also sinks the call of abs out of j-loop, but LNICM doesn't do so
; to try to make a perfect loop nest. (though y[i] = s; still prevents the 
; loop nest from being a perfect loop nest in this test case)

define dso_local double @test(double %x, i32* noalias %y) {
entry:
  br label %for.body

for.body:
  %i.02 = phi i32 [ 0, %entry ], [ %inc6, %for.end ]
  br label %for.body3

; CHECK: for.body3:
; LNICM: call i32 @abs(i32 %i.02)
; LICM-NOT: call i32 @abs(i32 %i.02)
for.body3:
  %j.01 = phi i32 [ 0, %for.body ], [ %inc, %for.body3 ]
  %call = call double @sin(double %x)
  %call4 = call i32 @abs(i32 %i.02)
  %inc = add nsw i32 %j.01, 1
  %cmp2 = icmp slt i32 %inc, 10
  br i1 %cmp2, label %for.body3, label %for.end

; CHECK: for.end:
; LICM: call i32 @abs(i32 %i.02)
; LNICM-NOT: call i32 @abs(i32 %i.02)
for.end:
  %s.1.lcssa = phi i32 [ %call4, %for.body3 ]
  %t.1.lcssa = phi double [ %call, %for.body3 ]
  %idxprom = sext i32 %i.02 to i64
  %arrayidx = getelementptr inbounds i32, i32* %y, i64 %idxprom
  store i32 %s.1.lcssa, i32* %arrayidx, align 4
  %inc6 = add nsw i32 %i.02, 1
  %cmp = icmp slt i32 %inc6, 10
  br i1 %cmp, label %for.body, label %for.end7

; CHECK: for.end7:
; CHECK: call double @sin(double %x)
for.end7:
  %t.0.lcssa = phi double [ %t.1.lcssa, %for.end ]
  ret double %t.0.lcssa
}

; double sin(double);
; int abs(int);
; double test(double x, int y[10]) {
;   double t = 0; int s = 0;
;   for (int i = 0; i < 10; i++) {
;     for (int k = 0; k < 10; k++) {
;       for (int j = 0; j < 10; j++) {
;         t = sin(x);
;         s = abs(i);
;       }
;     }
;     y[i] = s;
;   }
;   return t;
; }
;
define dso_local double @test2(double %x, i32* noalias %y) {
entry:
  br label %for.body

for.body:
  %i.02 = phi i32 [ 0, %entry ], [ %inc6, %for.end ]
  br label %for.k

for.k:
  %k = phi i64 [ 0, %for.body ], [ %inc.k, %for.end.k ]
  br label %for.body3

; CHECK: for.body3:
; LNICM: call i32 @abs(i32 %i.02)
; LICM-NOT: call i32 @abs(i32 %i.02)
for.body3:
  %j.01 = phi i32 [ 0, %for.k ], [ %inc, %for.body3 ]
  %call = call double @sin(double %x)
  %call4 = call i32 @abs(i32 %i.02)
  %inc = add nsw i32 %j.01, 1
  %cmp2 = icmp slt i32 %inc, 10
  br i1 %cmp2, label %for.body3, label %for.end.k

for.end.k:
  %s.lcssa.k = phi i32 [ %call4, %for.body3 ]
  %t.lcssa.k = phi double [ %call, %for.body3 ]
  %inc.k = add nsw i64 %k, 1
  %cmp.k = icmp slt i64 %inc.k, 10
  br i1 %cmp.k, label %for.k, label %for.end

; CHECK: for.end:
; LICM: call i32 @abs(i32 %i.02)
; LNICM-NOT: call i32 @abs(i32 %i.02)
for.end:
  %s.1.lcssa = phi i32 [ %s.lcssa.k, %for.end.k ]
  %t.1.lcssa = phi double [ %t.lcssa.k, %for.end.k ]
  %idxprom = sext i32 %i.02 to i64
  %arrayidx = getelementptr inbounds i32, i32* %y, i64 %idxprom
  store i32 %s.1.lcssa, i32* %arrayidx, align 4
  %inc6 = add nsw i32 %i.02, 1
  %cmp = icmp slt i32 %inc6, 10
  br i1 %cmp, label %for.body, label %for.end7

; CHECK: for.end7:
; CHECK: call double @sin(double %x)
for.end7:
  %t.0.lcssa = phi double [ %t.1.lcssa, %for.end ]
  ret double %t.0.lcssa
}

declare dso_local double @sin(double) #0

declare dso_local i32 @abs(i32) #0

attributes #0 = { nounwind readnone willreturn }
