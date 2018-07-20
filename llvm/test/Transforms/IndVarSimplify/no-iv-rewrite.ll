; RUN: opt < %s -indvars -S | FileCheck %s
;
; Make sure that indvars isn't inserting canonical IVs.
; This is kinda hard to do until linear function test replacement is removed.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define i32 @sum(i32* %arr, i32 %n) nounwind {
entry:
  %precond = icmp slt i32 0, %n
  br i1 %precond, label %ph, label %return

ph:
  br label %loop

; CHECK: loop:
;
; We should only have 2 IVs.
; CHECK: phi
; CHECK: phi
; CHECK-NOT: phi
;
; sext should be eliminated while preserving gep inboundsness.
; CHECK-NOT: sext
; CHECK: getelementptr inbounds
; CHECK: exit:
loop:
  %i.02 = phi i32 [ 0, %ph ], [ %iinc, %loop ]
  %s.01 = phi i32 [ 0, %ph ], [ %sinc, %loop ]
  %ofs = sext i32 %i.02 to i64
  %adr = getelementptr inbounds i32, i32* %arr, i64 %ofs
  %val = load i32, i32* %adr
  %sinc = add nsw i32 %s.01, %val
  %iinc = add nsw i32 %i.02, 1
  %cond = icmp slt i32 %iinc, %n
  br i1 %cond, label %loop, label %exit

exit:
  %s.lcssa = phi i32 [ %sinc, %loop ]
  br label %return

return:
  %s.0.lcssa = phi i32 [ %s.lcssa, %exit ], [ 0, %entry ]
  ret i32 %s.0.lcssa
}

define i64 @suml(i32* %arr, i32 %n) nounwind {
entry:
  %precond = icmp slt i32 0, %n
  br i1 %precond, label %ph, label %return

ph:
  br label %loop

; CHECK: loop:
;
; We should only have 2 IVs.
; CHECK: phi
; CHECK: phi
; CHECK-NOT: phi
;
; %ofs sext should be eliminated while preserving gep inboundsness.
; CHECK-NOT: sext
; CHECK: getelementptr inbounds
; %vall sext should obviously not be eliminated
; CHECK: sext
; CHECK: exit:
loop:
  %i.02 = phi i32 [ 0, %ph ], [ %iinc, %loop ]
  %s.01 = phi i64 [ 0, %ph ], [ %sinc, %loop ]
  %ofs = sext i32 %i.02 to i64
  %adr = getelementptr inbounds i32, i32* %arr, i64 %ofs
  %val = load i32, i32* %adr
  %vall = sext i32 %val to i64
  %sinc = add nsw i64 %s.01, %vall
  %iinc = add nsw i32 %i.02, 1
  %cond = icmp slt i32 %iinc, %n
  br i1 %cond, label %loop, label %exit

exit:
  %s.lcssa = phi i64 [ %sinc, %loop ]
  br label %return

return:
  %s.0.lcssa = phi i64 [ %s.lcssa, %exit ], [ 0, %entry ]
  ret i64 %s.0.lcssa
}

define void @outofbounds(i32* %first, i32* %last, i32 %idx) nounwind {
  %precond = icmp ne i32* %first, %last
  br i1 %precond, label %ph, label %return

; CHECK: ph:
; It's not indvars' job to perform LICM on %ofs
; CHECK-NOT: sext
ph:
  br label %loop

; CHECK: loop:
;
; Preserve exactly one pointer type IV.
; CHECK: phi i32*
; CHECK-NOT: phi
;
; Don't create any extra adds.
; CHECK-NOT: add
;
; Preserve gep inboundsness, and don't factor it.
; CHECK: getelementptr inbounds i32, i32* %ptriv, i32 1
; CHECK-NOT: add
; CHECK: exit:
loop:
  %ptriv = phi i32* [ %first, %ph ], [ %ptrpost, %loop ]
  %ofs = sext i32 %idx to i64
  %adr = getelementptr inbounds i32, i32* %ptriv, i64 %ofs
  store i32 3, i32* %adr
  %ptrpost = getelementptr inbounds i32, i32* %ptriv, i32 1
  %cond = icmp ne i32* %ptrpost, %last
  br i1 %cond, label %loop, label %exit

exit:
  br label %return

return:
  ret void
}

%structI = type { i32 }

define void @bitcastiv(i32 %start, i32 %limit, i32 %step, %structI* %base)
nounwind
{
entry:
  br label %loop

; CHECK: loop:
;
; Preserve casts
; CHECK: phi i32
; CHECK: bitcast
; CHECK: getelementptr
; CHECK: exit:
loop:
  %iv = phi i32 [%start, %entry], [%next, %loop]
  %p = phi %structI* [%base, %entry], [%pinc, %loop]
  %adr = getelementptr %structI, %structI* %p, i32 0, i32 0
  store i32 3, i32* %adr
  %pp = bitcast %structI* %p to i32*
  store i32 4, i32* %pp
  %pinc = getelementptr %structI, %structI* %p, i32 1
  %next = add i32 %iv, 1
  %cond = icmp ne i32 %next, %limit
  br i1 %cond, label %loop, label %exit

exit:
  ret void
}

define void @maxvisitor(i32 %limit, i32* %base) nounwind {
entry:
 br label %loop

; Test inserting a truncate at a phi use.
;
; CHECK: loop:
; CHECK: phi i64
; CHECK: trunc
; CHECK: exit:
loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %loop.inc ]
  %max = phi i32 [ 0, %entry ], [ %max.next, %loop.inc ]
  %idxprom = sext i32 %idx to i64
  %adr = getelementptr inbounds i32, i32* %base, i64 %idxprom
  %val = load i32, i32* %adr
  %cmp19 = icmp sgt i32 %val, %max
  br i1 %cmp19, label %if.then, label %if.else

if.then:
  br label %loop.inc

if.else:
  br label %loop.inc

loop.inc:
  %max.next = phi i32 [ %idx, %if.then ], [ %max, %if.else ]
  %idx.next = add nsw i32 %idx, 1
  %cmp = icmp slt i32 %idx.next, %limit
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define void @identityphi(i32 %limit) nounwind {
entry:
  br label %loop

; Test an edge case of removing an identity phi that directly feeds
; back to the loop iv.
;
; CHECK: loop:
; CHECK-NOT: phi
; CHECK: exit:
loop:
  %iv = phi i32 [ 0, %entry], [ %iv.next, %control ]
  br i1 undef, label %if.then, label %control

if.then:
  br label %control

control:
  %iv.next = phi i32 [ %iv, %loop ], [ undef, %if.then ]
  %cmp = icmp slt i32 %iv.next, %limit
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define i64 @cloneOr(i32 %limit, i64* %base) nounwind {
entry:
  ; ensure that the loop can't overflow
  %halfLim = ashr i32 %limit, 2
  br label %loop

; This test originally checked that the OR instruction was cloned. Now the
; ScalarEvolution is able to understand the loop evolution and that '%iv' at the
; end of the loop is an even value. Thus '%val' is computed at the end of the
; loop and the OR instruction is replaced by an ADD keeping the result
; equivalent.
;
; CHECK: sext
; CHECK: loop:
; CHECK: phi i64
; CHECK-NOT: sext
; CHECK: icmp slt i64
; CHECK: exit:
; CHECK: add i64
loop:
  %iv = phi i32 [ 0, %entry], [ %iv.next, %loop ]
  %t1 = sext i32 %iv to i64
  %adr = getelementptr i64, i64* %base, i64 %t1
  %val = load i64, i64* %adr
  %t2 = or i32 %iv, 1
  %t3 = sext i32 %t2 to i64
  %iv.next = add i32 %iv, 2
  %cmp = icmp slt i32 %iv.next, %halfLim
  br i1 %cmp, label %loop, label %exit

exit:
  %result = and i64 %val, %t3
  ret i64 %result
}

; The i induction variable looks like a wrap-around, but it really is just
; a simple affine IV.  Make sure that indvars simplifies through.
define i32 @indirectRecurrence() nounwind {
entry:
  br label %loop

; ReplaceLoopExitValue should fold the return value to constant 9.
; CHECK: loop:
; CHECK: phi i32
; CHECK: ret i32 9
loop:
  %j.0 = phi i32 [ 1, %entry ], [ %j.next, %cond_true ]
  %i.0 = phi i32 [ 0, %entry ], [ %j.0, %cond_true ]
  %tmp = icmp ne i32 %j.0, 10
  br i1 %tmp, label %cond_true, label %return

cond_true:
  %j.next = add i32 %j.0, 1
  br label %loop

return:
  ret i32 %i.0
}

; Eliminate the congruent phis j, k, and l.
; Eliminate the redundant IV increments k.next and l.next.
; Two phis should remain, one starting at %init, and one at %init1.
; Two increments should remain, one by %step and one by %step1.
; CHECK: loop:
; CHECK: phi i32
; CHECK: phi i32
; CHECK-NOT: phi
; CHECK: add i32
; CHECK: add i32
; CHECK: add i32
; CHECK-NOT: add
; CHECK: return:
;
; Five live-outs should remain.
; CHECK: lcssa = phi
; CHECK: lcssa = phi
; CHECK: lcssa = phi
; CHECK: lcssa = phi
; CHECK: lcssa = phi
; CHECK-NOT: phi
; CHECK: ret
define i32 @isomorphic(i32 %init, i32 %step, i32 %lim) nounwind {
entry:
  %step1 = add i32 %step, 1
  %init1 = add i32 %init, %step1
  %l.0 = sub i32 %init1, %step1
  br label %loop

loop:
  %ii = phi i32 [ %init1, %entry ], [ %ii.next, %loop ]
  %i = phi i32 [ %init, %entry ], [ %ii, %loop ]
  %j = phi i32 [ %init, %entry ], [ %j.next, %loop ]
  %k = phi i32 [ %init1, %entry ], [ %k.next, %loop ]
  %l = phi i32 [ %l.0, %entry ], [ %l.next, %loop ]
  %ii.next = add i32 %ii, %step1
  %j.next = add i32 %j, %step1
  %k.next = add i32 %k, %step1
  %l.step = add i32 %l, %step
  %l.next = add i32 %l.step, 1
  %cmp = icmp ne i32 %ii.next, %lim
  br i1 %cmp, label %loop, label %return

return:
  %sum1 = add i32 %i, %j.next
  %sum2 = add i32 %sum1, %k.next
  %sum3 = add i32 %sum1, %l.step
  %sum4 = add i32 %sum1, %l.next
  ret i32 %sum4
}

; Test a GEP IV that is derived from another GEP IV by a nop gep that
; lowers the type without changing the expression.
%structIF = type { i32, float }

define void @congruentgepiv(%structIF* %base) nounwind uwtable ssp {
entry:
  %first = getelementptr inbounds %structIF, %structIF* %base, i64 0, i32 0
  br label %loop

; CHECK: loop:
; CHECK: phi %structIF*
; CHECK-NOT: phi
; CHECK: getelementptr inbounds
; CHECK-NOT: getelementptr
; CHECK: exit:
loop:
  %ptr.iv = phi %structIF* [ %ptr.inc, %latch ], [ %base, %entry ]
  %next = phi i32* [ %next.inc, %latch ], [ %first, %entry ]
  store i32 4, i32* %next
  br i1 undef, label %latch, label %exit

latch:                         ; preds = %for.inc50.i
  %ptr.inc = getelementptr inbounds %structIF, %structIF* %ptr.iv, i64 1
  %next.inc = getelementptr inbounds %structIF, %structIF* %ptr.inc, i64 0, i32 0
  br label %loop

exit:
  ret void
}

declare void @use32(i32 %x)
declare void @use64(i64 %x)

; Test a widened IV that is used by a phi on different paths within the loop.
;
; CHECK: for.body:
; CHECK: phi i64
; CHECK: trunc i64
; CHECK: if.then:
; CHECK: for.inc:
; CHECK: phi i32
; CHECK: for.end:
define void @phiUsesTrunc() nounwind {
entry:
  br i1 undef, label %for.body, label %for.end

for.body:
  %iv = phi i32 [ %inc, %for.inc ], [ 1, %entry ]
  br i1 undef, label %if.then, label %if.else

if.then:
  br i1 undef, label %if.then33, label %for.inc

if.then33:
  br label %for.inc

if.else:
  br i1 undef, label %if.then97, label %for.inc

if.then97:
  %idxprom100 = sext i32 %iv to i64
  call void @use64(i64 %idxprom100)
  br label %for.inc

for.inc:
  %kmin.1 = phi i32 [ %iv, %if.then33 ], [ 0, %if.then ], [ %iv, %if.then97 ], [ 0, %if.else ]
  call void @use32(i32 %kmin.1)
  %inc = add nsw i32 %iv, 1
  br i1 undef, label %for.body, label %for.end

for.end:
  ret void
}
