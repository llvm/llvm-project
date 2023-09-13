; RUN: llc -verify-machineinstrs -mcpu=pwr7 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -ppc-gen-isel=false < %s | FileCheck --check-prefix=CHECK-NO-ISEL %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind
define void @jbd2_journal_commit_transaction(ptr %journal, i64 %inp1, i32 %inp2,
                                             ptr %inp3, ptr %inp4,
                                             ptr %inp5, i1 %inp6,
                                             i1 %inp7, i1 %inp8) #0 {
entry:
  br i1 undef, label %do.body, label %if.then5

if.then5:                                         ; preds = %entry
  unreachable

do.body:                                          ; preds = %entry
  br i1 undef, label %do.body.i, label %trace_jbd2_start_commit.exit

do.body.i:                                        ; preds = %do.body
  unreachable

trace_jbd2_start_commit.exit:                     ; preds = %do.body
  br i1 undef, label %do.body.i1116, label %trace_jbd2_commit_locking.exit

do.body.i1116:                                    ; preds = %trace_jbd2_start_commit.exit
  br i1 undef, label %if.end.i1123, label %do.body5.i1122

do.body5.i1122:                                   ; preds = %do.body.i1116
  unreachable

if.end.i1123:                                     ; preds = %do.body.i1116
  br label %trace_jbd2_commit_locking.exit

trace_jbd2_commit_locking.exit:                   ; preds = %if.end.i1123, %trace_jbd2_start_commit.exit
  br i1 undef, label %spin_unlock.exit1146, label %if.then.i.i.i.i1144

if.then.i.i.i.i1144:                              ; preds = %trace_jbd2_commit_locking.exit
  unreachable

spin_unlock.exit1146:                             ; preds = %trace_jbd2_commit_locking.exit
  br i1 undef, label %spin_unlock.exit1154, label %if.then.i.i.i.i1152

if.then.i.i.i.i1152:                              ; preds = %spin_unlock.exit1146
  br label %spin_unlock.exit1154

spin_unlock.exit1154:                             ; preds = %if.then.i.i.i.i1152, %spin_unlock.exit1146
  br i1 undef, label %do.body.i1159, label %trace_jbd2_commit_flushing.exit

do.body.i1159:                                    ; preds = %spin_unlock.exit1154
  unreachable

trace_jbd2_commit_flushing.exit:                  ; preds = %spin_unlock.exit1154
  br i1 undef, label %for.end.i, label %for.body.lr.ph.i

for.body.lr.ph.i:                                 ; preds = %trace_jbd2_commit_flushing.exit
  br i1 undef, label %spin_unlock.exit.i, label %if.then.i.i.i.i.i

if.then.i.i.i.i.i:                                ; preds = %for.body.lr.ph.i
  unreachable

spin_unlock.exit.i:                               ; preds = %for.body.lr.ph.i
  unreachable

for.end.i:                                        ; preds = %trace_jbd2_commit_flushing.exit
  br i1 undef, label %journal_submit_data_buffers.exit, label %if.then.i.i.i.i31.i

if.then.i.i.i.i31.i:                              ; preds = %for.end.i
  unreachable

journal_submit_data_buffers.exit:                 ; preds = %for.end.i
  br i1 undef, label %if.end103, label %if.then102

if.then102:                                       ; preds = %journal_submit_data_buffers.exit
  unreachable

if.end103:                                        ; preds = %journal_submit_data_buffers.exit
  br i1 undef, label %do.body.i1182, label %trace_jbd2_commit_logging.exit

do.body.i1182:                                    ; preds = %if.end103
  unreachable

trace_jbd2_commit_logging.exit:                   ; preds = %if.end103
  br i1 undef, label %for.end.i1287, label %for.body.i1277

for.body.i1277:                                   ; preds = %trace_jbd2_commit_logging.exit
  unreachable

for.end.i1287:                                    ; preds = %trace_jbd2_commit_logging.exit
  br i1 undef, label %journal_finish_inode_data_buffers.exit, label %if.then.i.i.i.i84.i

if.then.i.i.i.i84.i:                              ; preds = %for.end.i1287
  unreachable

journal_finish_inode_data_buffers.exit:           ; preds = %for.end.i1287
  br i1 undef, label %if.end256, label %if.then249

if.then249:                                       ; preds = %journal_finish_inode_data_buffers.exit
  unreachable

if.end256:                                        ; preds = %journal_finish_inode_data_buffers.exit
  br i1 undef, label %do.body277, label %if.then260

if.then260:                                       ; preds = %if.end256
  br label %do.body277

do.body277:                                       ; preds = %if.then260, %if.end256
  br label %while.body318

while.body318:                                    ; preds = %wait_on_buffer.exit, %do.body277
  %tobool.i1295 = icmp eq i64 %inp1, 0
  br i1 %tobool.i1295, label %wait_on_buffer.exit, label %if.then.i1296

if.then.i1296:                                    ; preds = %while.body318
  unreachable

wait_on_buffer.exit:                              ; preds = %while.body318
  br i1 %inp6, label %do.body378, label %while.body318

do.body378:                                       ; preds = %wait_on_buffer.exit
  br i1 %inp7, label %while.end418, label %while.body392.lr.ph

while.body392.lr.ph:                              ; preds = %do.body378
  br label %while.body392

while.body392:                                    ; preds = %wait_on_buffer.exit1319, %while.body392.lr.ph
  %0 = load ptr, ptr undef, align 8
  %add.ptr399 = getelementptr inbounds i8, ptr %0, i64 -72
  %tobool.i1316 = icmp eq i64 %inp1, 0
  br i1 %tobool.i1316, label %wait_on_buffer.exit1319, label %if.then.i1317

if.then.i1317:                                    ; preds = %while.body392
  unreachable

wait_on_buffer.exit1319:                          ; preds = %while.body392
  %1 = load volatile i64, ptr %add.ptr399, align 8
  %conv.i.i1322 = and i64 %1, 1
  %lnot404 = icmp eq i64 %conv.i.i1322, 0
  %.err.4 = select i1 %lnot404, i32 -5, i32 %inp2
  %2 = call i64 asm sideeffect "1:.long 0x7c0000a8 $| ((($0) & 0x1f) << 21) $| (((0) & 0x1f) << 16) $| ((($3) & 0x1f) << 11) $| (((0) & 0x1) << 0) \0Aandc $0,$0,$2\0Astdcx. $0,0,$3\0Abne- 1b\0A", "=&r,=*m,r,r,*m,~{cc},~{memory}"(ptr elementtype(i64) %add.ptr399, i64 262144, ptr %add.ptr399, ptr elementtype(i64) %add.ptr399) #1
  %prev.i.i.i1325 = getelementptr inbounds i8, ptr %0, i64 8
  %3 = load ptr, ptr %inp4, align 8
  store ptr %3, ptr %inp5, align 8
  call void @__brelse(ptr %3) #1
  br i1 %inp8, label %while.end418, label %while.body392

; CHECK-LABEL: @jbd2_journal_commit_transaction
; CHECK-NO-ISEL-LABEL: @jbd2_journal_commit_transaction
; CHECK: andi.
; CHECK: crmove
; CHECK: stdcx.
; CHECK: iselgt {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK-NO-ISEL: bc 12, 1, [[TRUE:.LBB[0-9]+]]
; CHECK-NO-ISEL: ori 29, 3, 0
; CHECK-NO-ISEL: b [[SUCCESSOR:.LBB[0-9]+]]


while.end418:                                     ; preds = %wait_on_buffer.exit1319, %do.body378
  %err.4.lcssa = phi i32 [ %inp2, %do.body378 ], [ %.err.4, %wait_on_buffer.exit1319 ]
  br i1 %inp7, label %if.end421, label %if.then420

if.then420:                                       ; preds = %while.end418
  call void @jbd2_journal_abort(ptr %journal, i32 signext %err.4.lcssa) #1
  br label %if.end421

if.end421:                                        ; preds = %if.then420, %while.end418
  unreachable
}

declare void @jbd2_journal_abort(ptr, i32 signext)

declare void @__brelse(ptr)

attributes #0 = { nounwind }
attributes #1 = { nounwind }

