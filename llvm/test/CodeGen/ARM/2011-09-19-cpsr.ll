; RUN: llc -march=thumb -mcpu=cortex-a8 < %s
; rdar://problem/10137436: sqlite3 miscompile
;
; CHECK: subs
; CHECK: cmp
; CHECK: it

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-ios4.0.0"

declare i8* @__memset_chk(i8*, i32, i32, i32) nounwind

define hidden fastcc i32 @sqlite3VdbeExec(i32* %p) nounwind {
entry:
  br label %sqlite3VarintLen.exit7424

sqlite3VarintLen.exit7424:                        ; preds = %do.body.i7423
  br label %do.body.i

do.body.i:                                        ; preds = %do.body.i, %sqlite3VarintLen.exit7424
  br i1 undef, label %do.body.i, label %sqlite3VarintLen.exit

sqlite3VarintLen.exit:                            ; preds = %do.body.i
  %sub2322 = add i64 undef, undef
  br i1 undef, label %too_big, label %if.end2327

if.end2327:                                       ; preds = %sqlite3VarintLen.exit
  br i1 undef, label %if.end2341, label %no_mem

if.end2341:                                       ; preds = %if.end2327
  br label %for.body2355

for.body2355:                                     ; preds = %for.body2355, %if.end2341
  %add2366 = add nsw i32 undef, undef
  br i1 undef, label %for.body2377, label %for.body2355

for.body2377:                                     ; preds = %for.body2355
  %conv23836154 = zext i32 %add2366 to i64
  %sub2384 = sub i64 %sub2322, %conv23836154
  %conv2385 = trunc i64 %sub2384 to i32
  %len.0.i = select i1 undef, i32 %conv2385, i32 undef
  %sub.i7384 = sub nsw i32 %len.0.i, 0
  %call.i.i7385 = call i8* @__memset_chk(i8* undef, i32 0, i32 %sub.i7384, i32 undef) nounwind
  unreachable

too_big:                                          ; preds = %sqlite3VarintLen.exit
  unreachable

no_mem:                                           ; preds = %if.end2327, %for.body, %entry.no_mem_crit_edge
  unreachable

sqlite3ErrStr.exit:                               ; preds = %if.then82
  unreachable
}
