; RUN: llc -mcpu=pwr7 < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind
define void @jbd2_journal_commit_transaction(i32 %input1, i32* %input2, i32* %input3, i8** %input4) #0 {
entry:
  br label %while.body392

while.body392:                                    ; preds = %wait_on_buffer.exit1319, %while.body392.lr.ph
  %0 = load i8*, i8** %input4, align 8
  %add.ptr399 = getelementptr inbounds i8, i8* %0, i64 -72
  %b_state.i.i1314 = bitcast i8* %add.ptr399 to i64*
  %ivar = add i32 %input1, 1
  %tobool.i1316 = icmp eq i32 %input1, 0
  br i1 %tobool.i1316, label %wait_on_buffer.exit1319, label %while.end418

wait_on_buffer.exit1319:                          ; preds = %while.body392
  %1 = load volatile i64, i64* %b_state.i.i1314, align 8
  %conv.i.i1322 = and i64 %1, 1
  %lnot404 = icmp eq i64 %conv.i.i1322, 0
  %.err.4 = select i1 %lnot404, i32 -5, i32 %input1
  %2 = call i64 asm sideeffect "1:.long 0x7c0000a8 $| ((($0) & 0x1f) << 21) $| (((0) & 0x1f) << 16) $| ((($3) & 0x1f) << 11) $| (((0) & 0x1) << 0) \0Aandc $0,$0,$2\0Astdcx. $0,0,$3\0Abne- 1b\0A", "=&r,=*m,r,r,*m,~{cc},~{memory}"(i64* %b_state.i.i1314, i64 262144, i64* %b_state.i.i1314, i64* %b_state.i.i1314) #0
  store i8* %0, i8** %input4, align 8
  %cmp.i1312 = icmp eq i32* %input2, %input3
  br i1 %cmp.i1312, label %while.end418, label %while.body392

while.end418:                                     ; preds = %wait_on_buffer.exit1319, %do.body378
  %err.4.lcssa = phi i32 [ %ivar, %while.body392 ], [ %.err.4, %wait_on_buffer.exit1319 ]
  %tobool419 = icmp eq i32 %err.4.lcssa, 0
  br i1 %tobool419, label %if.end421, label %if.then420

; CHECK-LABEL: @jbd2_journal_commit_transaction
; CHECK: andi.
; CHECK: crmove [[REG:[0-9]+]], 1
; CHECK: stdcx.
; CHECK: isel {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}, [[REG]]

if.then420:                                       ; preds = %while.end418
  unreachable

if.end421:                                        ; preds = %while.end418
  unreachable

}

attributes #0 = { nounwind }

