; RUN: llc < %s -stress-early-ifcvt -aarch64-enable-atomic-cfg-tidy=0 | FileCheck %s
target triple = "arm64-apple-macosx"

; CHECK: mm2
define i32 @mm2(i32* nocapture %p, i32 %n) nounwind uwtable readonly ssp {
entry:
  br label %do.body

; CHECK: do.body
; Loop body has no branches before the backedge.
; CHECK-NOT: LBB
do.body:
  %max.0 = phi i32 [ 0, %entry ], [ %max.1, %do.cond ]
  %min.0 = phi i32 [ 0, %entry ], [ %min.1, %do.cond ]
  %n.addr.0 = phi i32 [ %n, %entry ], [ %dec, %do.cond ]
  %p.addr.0 = phi i32* [ %p, %entry ], [ %incdec.ptr, %do.cond ]
  %incdec.ptr = getelementptr inbounds i32, i32* %p.addr.0, i64 1
  %0 = load i32, i32* %p.addr.0, align 4
  %cmp = icmp sgt i32 %0, %max.0
  br i1 %cmp, label %do.cond, label %if.else

if.else:
  %cmp1 = icmp slt i32 %0, %min.0
  %.min.0 = select i1 %cmp1, i32 %0, i32 %min.0
  br label %do.cond

do.cond:
  %max.1 = phi i32 [ %0, %do.body ], [ %max.0, %if.else ]
  %min.1 = phi i32 [ %min.0, %do.body ], [ %.min.0, %if.else ]
; CHECK: cbnz
  %dec = add i32 %n.addr.0, -1
  %tobool = icmp eq i32 %dec, 0
  br i1 %tobool, label %do.end, label %do.body

do.end:
  %sub = sub nsw i32 %max.1, %min.1
  ret i32 %sub
}

; CHECK-LABEL: fold_inc_true_32:
; CHECK: {{subs.*wzr,|cmp}} w2, #1
; CHECK-NEXT: csinc w0, w1, w0, eq
; CHECK-NEXT: ret
define i32 @fold_inc_true_32(i32 %x, i32 %y, i32 %c) nounwind ssp {
entry:
  %tobool = icmp eq i32 %c, 1
  %inc = add nsw i32 %x, 1
  br i1 %tobool, label %eq_bb, label %done

eq_bb:
  br label %done

done:
  %cond = phi i32 [ %y, %eq_bb ], [ %inc, %entry ]
  ret i32 %cond
}

; CHECK-LABEL: fold_inc_true_64:
; CHECK: {{subs.*xzr,|cmp}} x2, #1
; CHECK-NEXT: csinc x0, x1, x0, eq
; CHECK-NEXT: ret
define i64 @fold_inc_true_64(i64 %x, i64 %y, i64 %c) nounwind ssp {
entry:
  %tobool = icmp eq i64 %c, 1
  %inc = add nsw i64 %x, 1
  br i1 %tobool, label %eq_bb, label %done

eq_bb:
  br label %done

done:
  %cond = phi i64 [ %y, %eq_bb ], [ %inc, %entry ]
  ret i64 %cond
}

; CHECK-LABEL: fold_inc_false_32:
; CHECK: {{subs.*wzr,|cmp}} w2, #1
; CHECK-NEXT: csinc w0, w1, w0, ne
; CHECK-NEXT: ret
define i32 @fold_inc_false_32(i32 %x, i32 %y, i32 %c) nounwind ssp {
entry:
  %tobool = icmp eq i32 %c, 1
  %inc = add nsw i32 %x, 1
  br i1 %tobool, label %eq_bb, label %done

eq_bb:
  br label %done

done:
  %cond = phi i32 [ %inc, %eq_bb ], [ %y, %entry ]
  ret i32 %cond
}

; CHECK-LABEL: fold_inc_false_64:
; CHECK: {{subs.*xzr,|cmp}} x2, #1
; CHECK-NEXT: csinc x0, x1, x0, ne
; CHECK-NEXT: ret
define i64 @fold_inc_false_64(i64 %x, i64 %y, i64 %c) nounwind ssp {
entry:
  %tobool = icmp eq i64 %c, 1
  %inc = add nsw i64 %x, 1
  br i1 %tobool, label %eq_bb, label %done

eq_bb:
  br label %done

done:
  %cond = phi i64 [ %inc, %eq_bb ], [ %y, %entry ]
  ret i64 %cond
}

; CHECK-LABEL: fold_inv_true_32:
; CHECK: {{subs.*wzr,|cmp}} w2, #1
; CHECK-NEXT: csinv w0, w1, w0, eq
; CHECK-NEXT: ret
define i32 @fold_inv_true_32(i32 %x, i32 %y, i32 %c) nounwind ssp {
entry:
  %tobool = icmp eq i32 %c, 1
  %inv = xor i32 %x, -1
  br i1 %tobool, label %eq_bb, label %done

eq_bb:
  br label %done

done:
  %cond = phi i32 [ %y, %eq_bb ], [ %inv, %entry ]
  ret i32 %cond
}

; CHECK-LABEL: fold_inv_true_64:
; CHECK: {{subs.*xzr,|cmp}} x2, #1
; CHECK-NEXT: csinv x0, x1, x0, eq
; CHECK-NEXT: ret
define i64 @fold_inv_true_64(i64 %x, i64 %y, i64 %c) nounwind ssp {
entry:
  %tobool = icmp eq i64 %c, 1
  %inv = xor i64 %x, -1
  br i1 %tobool, label %eq_bb, label %done

eq_bb:
  br label %done

done:
  %cond = phi i64 [ %y, %eq_bb ], [ %inv, %entry ]
  ret i64 %cond
}

; CHECK-LABEL: fold_inv_false_32:
; CHECK: {{subs.*wzr,|cmp}} w2, #1
; CHECK-NEXT: csinv w0, w1, w0, ne
; CHECK-NEXT: ret
define i32 @fold_inv_false_32(i32 %x, i32 %y, i32 %c) nounwind ssp {
entry:
  %tobool = icmp eq i32 %c, 1
  %inv = xor i32 %x, -1
  br i1 %tobool, label %eq_bb, label %done

eq_bb:
  br label %done

done:
  %cond = phi i32 [ %inv, %eq_bb ], [ %y, %entry ]
  ret i32 %cond
}

; CHECK-LABEL: fold_inv_false_64:
; CHECK: {{subs.*xzr,|cmp}} x2, #1
; CHECK-NEXT: csinv x0, x1, x0, ne
; CHECK-NEXT: ret
define i64 @fold_inv_false_64(i64 %x, i64 %y, i64 %c) nounwind ssp {
entry:
  %tobool = icmp eq i64 %c, 1
  %inv = xor i64 %x, -1
  br i1 %tobool, label %eq_bb, label %done

eq_bb:
  br label %done

done:
  %cond = phi i64 [ %inv, %eq_bb ], [ %y, %entry ]
  ret i64 %cond
}

; CHECK-LABEL: fold_neg_true_32:
; CHECK: {{subs.*wzr,|cmp}} w2, #1
; CHECK-NEXT: csneg w0, w1, w0, eq
; CHECK-NEXT: ret
define i32 @fold_neg_true_32(i32 %x, i32 %y, i32 %c) nounwind ssp {
entry:
  %tobool = icmp eq i32 %c, 1
  %neg = sub nsw i32 0, %x
  br i1 %tobool, label %eq_bb, label %done

eq_bb:
  br label %done

done:
  %cond = phi i32 [ %y, %eq_bb ], [ %neg, %entry ]
  ret i32 %cond
}

; CHECK-LABEL: fold_neg_true_64:
; CHECK: {{subs.*xzr,|cmp}} x2, #1
; CHECK-NEXT: csneg x0, x1, x0, eq
; CHECK-NEXT: ret
define i64 @fold_neg_true_64(i64 %x, i64 %y, i64 %c) nounwind ssp {
entry:
  %tobool = icmp eq i64 %c, 1
  %neg = sub nsw i64 0, %x
  br i1 %tobool, label %eq_bb, label %done

eq_bb:
  br label %done

done:
  %cond = phi i64 [ %y, %eq_bb ], [ %neg, %entry ]
  ret i64 %cond
}

; CHECK-LABEL: fold_neg_false_32:
; CHECK: {{subs.*wzr,|cmp}} w2, #1
; CHECK-NEXT: csneg w0, w1, w0, ne
; CHECK-NEXT: ret
define i32 @fold_neg_false_32(i32 %x, i32 %y, i32 %c) nounwind ssp {
entry:
  %tobool = icmp eq i32 %c, 1
  %neg = sub nsw i32 0, %x
  br i1 %tobool, label %eq_bb, label %done

eq_bb:
  br label %done

done:
  %cond = phi i32 [ %neg, %eq_bb ], [ %y, %entry ]
  ret i32 %cond
}

; CHECK-LABEL: fold_neg_false_64:
; CHECK: {{subs.*xzr,|cmp}} x2, #1
; CHECK-NEXT: csneg x0, x1, x0, ne
; CHECK-NEXT: ret
define i64 @fold_neg_false_64(i64 %x, i64 %y, i64 %c) nounwind ssp {
entry:
  %tobool = icmp eq i64 %c, 1
  %neg = sub nsw i64 0, %x
  br i1 %tobool, label %eq_bb, label %done

eq_bb:
  br label %done

done:
  %cond = phi i64 [ %neg, %eq_bb ], [ %y, %entry ]
  ret i64 %cond
}

; CHECK: cbnz_32
; CHECK: {{subs.*wzr,|cmp}} w2, #0
; CHECK-NEXT: csel w0, w1, w0, ne
; CHECK-NEXT: ret
define i32 @cbnz_32(i32 %x, i32 %y, i32 %c) nounwind ssp {
entry:
  %tobool = icmp eq i32 %c, 0
  br i1 %tobool, label %eq_bb, label %done

eq_bb:
  br label %done

done:
  %cond = phi i32 [ %x, %eq_bb ], [ %y, %entry ]
  ret i32 %cond
}

; CHECK: cbnz_64
; CHECK: {{subs.*xzr,|cmp}} x2, #0
; CHECK-NEXT: csel x0, x1, x0, ne
; CHECK-NEXT: ret
define i64 @cbnz_64(i64 %x, i64 %y, i64 %c) nounwind ssp {
entry:
  %tobool = icmp eq i64 %c, 0
  br i1 %tobool, label %eq_bb, label %done

eq_bb:
  br label %done

done:
  %cond = phi i64 [ %x, %eq_bb ], [ %y, %entry ]
  ret i64 %cond
}

; CHECK: cbz_32
; CHECK: {{subs.*wzr,|cmp}} w2, #0
; CHECK-NEXT: csel w0, w1, w0, eq
; CHECK-NEXT: ret
define i32 @cbz_32(i32 %x, i32 %y, i32 %c) nounwind ssp {
entry:
  %tobool = icmp ne i32 %c, 0
  br i1 %tobool, label %ne_bb, label %done

ne_bb:
  br label %done

done:
  %cond = phi i32 [ %x, %ne_bb ], [ %y, %entry ]
  ret i32 %cond
}

; CHECK: cbz_64
; CHECK: {{subs.*xzr,|cmp}} x2, #0
; CHECK-NEXT: csel x0, x1, x0, eq
; CHECK-NEXT: ret
define i64 @cbz_64(i64 %x, i64 %y, i64 %c) nounwind ssp {
entry:
  %tobool = icmp ne i64 %c, 0
  br i1 %tobool, label %ne_bb, label %done

ne_bb:
  br label %done

done:
  %cond = phi i64 [ %x, %ne_bb ], [ %y, %entry ]
  ret i64 %cond
}

; CHECK: tbnz_32
; CHECK: {{ands.*xzr,|tst}} w2, #0x80
; CHECK-NEXT: csel w0, w1, w0, ne
; CHECK-NEXT: ret
define i32 @tbnz_32(i32 %x, i32 %y, i32 %c) nounwind ssp {
entry:
  %mask = and i32 %c, 128
  %tobool = icmp eq i32 %mask, 0
  br i1 %tobool, label %eq_bb, label %done

eq_bb:
  br label %done

done:
  %cond = phi i32 [ %x, %eq_bb ], [ %y, %entry ]
  ret i32 %cond
}

; CHECK: tbnz_64
; CHECK: {{ands.*xzr,|tst}} x2, #0x8000000000000000
; CHECK-NEXT: csel x0, x1, x0, ne
; CHECK-NEXT: ret
define i64 @tbnz_64(i64 %x, i64 %y, i64 %c) nounwind ssp {
entry:
  %mask = and i64 %c, 9223372036854775808
  %tobool = icmp eq i64 %mask, 0
  br i1 %tobool, label %eq_bb, label %done

eq_bb:
  br label %done

done:
  %cond = phi i64 [ %x, %eq_bb ], [ %y, %entry ]
  ret i64 %cond
}

; CHECK: tbz_32
; CHECK: {{ands.*xzr,|tst}} w2, #0x80
; CHECK-NEXT: csel w0, w1, w0, eq
; CHECK-NEXT: ret
define i32 @tbz_32(i32 %x, i32 %y, i32 %c) nounwind ssp {
entry:
  %mask = and i32 %c, 128
  %tobool = icmp ne i32 %mask, 0
  br i1 %tobool, label %ne_bb, label %done

ne_bb:
  br label %done

done:
  %cond = phi i32 [ %x, %ne_bb ], [ %y, %entry ]
  ret i32 %cond
}

; CHECK: tbz_64
; CHECK: {{ands.*xzr,|tst}} x2, #0x8000000000000000
; CHECK-NEXT: csel x0, x1, x0, eq
; CHECK-NEXT: ret
define i64 @tbz_64(i64 %x, i64 %y, i64 %c) nounwind ssp {
entry:
  %mask = and i64 %c, 9223372036854775808
  %tobool = icmp ne i64 %mask, 0
  br i1 %tobool, label %ne_bb, label %done

ne_bb:
  br label %done

done:
  %cond = phi i64 [ %x, %ne_bb ], [ %y, %entry ]
  ret i64 %cond
}

; This function from 175.vpr folds an ADDWri into a CSINC.
; Remember to clear the kill flag on the ADDWri.
define i32 @get_ytrack_to_xtracks() nounwind ssp {
entry:
  br label %for.body

for.body:
  %x0 = load i32, i32* undef, align 4
  br i1 undef, label %if.then.i146, label %is_sbox.exit155

if.then.i146:
  %add8.i143 = add nsw i32 0, %x0
  %rem.i144 = srem i32 %add8.i143, %x0
  %add9.i145 = add i32 %rem.i144, 1
  br label %is_sbox.exit155

is_sbox.exit155:                                  ; preds = %if.then.i146, %for.body
  %seg_offset.0.i151 = phi i32 [ %add9.i145, %if.then.i146 ], [ undef, %for.body ]
  %idxprom15.i152 = sext i32 %seg_offset.0.i151 to i64
  %arrayidx18.i154 = getelementptr inbounds i32, i32* null, i64 %idxprom15.i152
  %x1 = load i32, i32* %arrayidx18.i154, align 4
  br i1 undef, label %for.body51, label %for.body

for.body51:                                       ; preds = %is_sbox.exit155
  call fastcc void @get_switch_type(i32 %x1, i32 undef, i16 signext undef, i16 signext undef, i16* undef)
  unreachable
}
declare fastcc void @get_switch_type(i32, i32, i16 signext, i16 signext, i16* nocapture) nounwind ssp
