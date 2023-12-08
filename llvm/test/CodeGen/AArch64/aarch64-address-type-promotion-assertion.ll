; RUN: llc -O3 -mcpu=cortex-a53 -mtriple=aarch64--linux-gnu %s -o - | FileCheck %s
; PR20188: don't crash when merging sexts.

; CHECK: foo:
define void @foo() unnamed_addr align 2 {
entry:
  br label %invoke.cont145

invoke.cont145:
  %or.cond = and i1 undef, false
  br i1 %or.cond, label %if.then274, label %invoke.cont145

if.then274:
  %0 = load i32, ptr null, align 4
  br i1 undef, label %invoke.cont291, label %if.else313

invoke.cont291:
  %idxprom.i.i.i605 = sext i32 %0 to i64
  %arrayidx.i.i.i607 = getelementptr inbounds double, ptr undef, i64 %idxprom.i.i.i605
  %idxprom.i.i.i596 = sext i32 %0 to i64
  %arrayidx.i.i.i598 = getelementptr inbounds double, ptr undef, i64 %idxprom.i.i.i596
  br label %if.end356

if.else313:
  %cmp314 = fcmp olt double undef, 0.000000e+00
  br i1 %cmp314, label %invoke.cont317, label %invoke.cont353

invoke.cont317:
  br i1 undef, label %invoke.cont326, label %invoke.cont334

invoke.cont326:
  %idxprom.i.i.i587 = sext i32 %0 to i64
  %arrayidx.i.i.i589 = getelementptr inbounds double, ptr undef, i64 %idxprom.i.i.i587
  %sub329 = fsub fast double undef, undef
  br label %invoke.cont334

invoke.cont334:
  %lo.1 = phi double [ %sub329, %invoke.cont326 ], [ undef, %invoke.cont317 ]
  br i1 undef, label %invoke.cont342, label %if.end356

invoke.cont342:
  %idxprom.i.i.i578 = sext i32 %0 to i64
  %arrayidx.i.i.i580 = getelementptr inbounds double, ptr undef, i64 %idxprom.i.i.i578
  br label %if.end356

invoke.cont353:
  %idxprom.i.i.i572 = sext i32 %0 to i64
  %arrayidx.i.i.i574 = getelementptr inbounds double, ptr undef, i64 %idxprom.i.i.i572
  br label %if.end356

if.end356:
  %lo.2 = phi double [ 0.000000e+00, %invoke.cont291 ], [ %lo.1, %invoke.cont342 ], [ undef, %invoke.cont353 ], [ %lo.1, %invoke.cont334 ]
  call void null(i32 %0, double %lo.2)
  unreachable
}
