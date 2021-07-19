; RUN: llc < %s -O0 -asm-verbose=false -verify-machineinstrs -disable-block-placement -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s

; Test irreducible CFG handling.

target triple = "wasm32-unknown-unknown"

; A simple loop with two entries.

; CHECK-LABEL: test0:
; CHECK: f64.load
; CHECK: i32.const $[[REG:[^,]+]]=
; CHECK: br_table  $[[REG]],
define void @test0(double* %arg, i32 %arg1, i32 %arg2, i32 %arg3) {
bb:
  %tmp = icmp eq i32 %arg2, 0
  br i1 %tmp, label %bb6, label %bb3

bb3:                                              ; preds = %bb
  %tmp4 = getelementptr double, double* %arg, i32 %arg3
  %tmp5 = load double, double* %tmp4, align 4
  br label %bb13

bb6:                                              ; preds = %bb13, %bb
  %tmp7 = phi i32 [ %tmp18, %bb13 ], [ 0, %bb ]
  %tmp8 = icmp slt i32 %tmp7, %arg1
  br i1 %tmp8, label %bb9, label %bb19

bb9:                                              ; preds = %bb6
  %tmp10 = getelementptr double, double* %arg, i32 %tmp7
  %tmp11 = load double, double* %tmp10, align 4
  %tmp12 = fmul double %tmp11, 2.300000e+00
  store double %tmp12, double* %tmp10, align 4
  br label %bb13

bb13:                                             ; preds = %bb9, %bb3
  %tmp14 = phi double [ %tmp5, %bb3 ], [ %tmp12, %bb9 ]
  %tmp15 = phi i32 [ undef, %bb3 ], [ %tmp7, %bb9 ]
  %tmp16 = getelementptr double, double* %arg, i32 %tmp15
  %tmp17 = fadd double %tmp14, 1.300000e+00
  store double %tmp17, double* %tmp16, align 4
  %tmp18 = add nsw i32 %tmp15, 1
  br label %bb6

bb19:                                             ; preds = %bb6
  ret void
}

; A simple loop with two entries and an inner natural loop.

; CHECK-LABEL: test1:
; CHECK: f64.load
; CHECK: i32.const $[[REG:[^,]+]]=
; CHECK: br_table  $[[REG]],
define void @test1(double* %arg, i32 %arg1, i32 %arg2, i32 %arg3) {
bb:
  %tmp = icmp eq i32 %arg2, 0
  br i1 %tmp, label %bb6, label %bb3

bb3:                                              ; preds = %bb
  %tmp4 = getelementptr double, double* %arg, i32 %arg3
  %tmp5 = load double, double* %tmp4, align 4
  br label %bb13

bb6:                                              ; preds = %bb13, %bb
  %tmp7 = phi i32 [ %tmp18, %bb13 ], [ 0, %bb ]
  %tmp8 = icmp slt i32 %tmp7, %arg1
  br i1 %tmp8, label %bb9, label %bb19

bb9:                                              ; preds = %bb6
  %tmp10 = getelementptr double, double* %arg, i32 %tmp7
  %tmp11 = load double, double* %tmp10, align 4
  %tmp12 = fmul double %tmp11, 2.300000e+00
  store double %tmp12, double* %tmp10, align 4
  br label %bb10

bb10:                                             ; preds = %bb10, %bb9
  %p = phi i32 [ 0, %bb9 ], [ %pn, %bb10 ]
  %pn = add i32 %p, 1
  %c = icmp slt i32 %pn, 256
  br i1 %c, label %bb10, label %bb13

bb13:                                             ; preds = %bb10, %bb3
  %tmp14 = phi double [ %tmp5, %bb3 ], [ %tmp12, %bb10 ]
  %tmp15 = phi i32 [ undef, %bb3 ], [ %tmp7, %bb10 ]
  %tmp16 = getelementptr double, double* %arg, i32 %tmp15
  %tmp17 = fadd double %tmp14, 1.300000e+00
  store double %tmp17, double* %tmp16, align 4
  %tmp18 = add nsw i32 %tmp15, 1
  br label %bb6

bb19:                                             ; preds = %bb6
  ret void
}

; A simple loop 2 blocks that are both entries: A1 and A2.
; Even though A1 and A2 both have 3 predecessors (A0, A1, and A2), not 6 but
; only 4 new routing blocks to the dispatch block should be generated.

; CHECK-LABEL: test2:
; CHECK: br_if
; CHECK: i32.const $[[REG:[^,]+]]=
; CHECK: i32.const $[[REG]]=
; CHECK: br_table  $[[REG]],
; CHECK: i32.const $[[REG]]=
; CHECK: i32.const $[[REG]]=
; CHECK-NOT: i32.const $[[REG]]=
define i32 @test2(i32) {
entry:
  br label %A0

A0:                                               ; preds = %entry
  %a0a = tail call i32 @test2(i32 1)
  %a0b = icmp eq i32 %a0a, 0
  br i1 %a0b, label %A1, label %A2

A1:                                               ; preds = %A2, %A1, %A0
  %a1a = tail call i32 @test2(i32 2)
  %a1b = icmp eq i32 %a1a, 0
  br i1 %a1b, label %A1, label %A2

A2:                                               ; preds = %A2, %A1, %A0
  %a2a = tail call i32 @test2(i32 3)
  %a2b = icmp eq i32 %a2a, 0
  br i1 %a2b, label %A1, label %A2
}

; An interesting loop with inner loop and if-else structure too.

; CHECK-LABEL: test3:
; CHECK: br_if
define void @test3(i32 %ws) {
entry:
  %ws.addr = alloca i32, align 4
  store volatile i32 %ws, i32* %ws.addr, align 4
  %0 = load volatile i32, i32* %ws.addr, align 4
  %tobool = icmp ne i32 %0, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  br label %wynn

if.end:                                           ; preds = %entry
  %1 = load volatile i32, i32* %ws.addr, align 4
  %tobool1 = icmp ne i32 %1, 0
  br i1 %tobool1, label %if.end9, label %if.then2

if.then2:                                         ; preds = %if.end
  br label %for.cond

for.cond:                                         ; preds = %wynn, %if.then7, %if.then2
  %2 = load volatile i32, i32* %ws.addr, align 4
  %tobool3 = icmp ne i32 %2, 0
  br i1 %tobool3, label %if.then4, label %if.end5

if.then4:                                         ; preds = %for.cond
  br label %if.end5

if.end5:                                          ; preds = %if.then4, %for.cond
  %3 = load volatile i32, i32* %ws.addr, align 4
  %tobool6 = icmp ne i32 %3, 0
  br i1 %tobool6, label %if.then7, label %if.end8

if.then7:                                         ; preds = %if.end5
  br label %for.cond

if.end8:                                          ; preds = %if.end5
  br label %wynn

wynn:                                             ; preds = %if.end8, %if.then
  br label %for.cond

if.end9:                                          ; preds = %if.end
  ret void
}

; Multi-level irreducibility, after reducing in the main scope we must then
; reduce in the inner loop that we just created.
; CHECK: br_table
; CHECK: br_table
define void @pi_next() {
entry:
  br i1 undef, label %sw.bb5, label %return

sw.bb5:                                           ; preds = %entry
  br i1 undef, label %if.then.i49, label %if.else.i52

if.then.i49:                                      ; preds = %sw.bb5
  br label %for.inc197.i

if.else.i52:                                      ; preds = %sw.bb5
  br label %for.cond57.i

for.cond57.i:                                     ; preds = %for.inc205.i, %if.else.i52
  store i32 0, i32* undef, align 4
  br label %for.cond65.i

for.cond65.i:                                     ; preds = %for.inc201.i, %for.cond57.i
  br i1 undef, label %for.body70.i, label %for.inc205.i

for.body70.i:                                     ; preds = %for.cond65.i
  br label %for.cond76.i

for.cond76.i:                                     ; preds = %for.inc197.i, %for.body70.i
  %0 = phi i32 [ %inc199.i, %for.inc197.i ], [ 0, %for.body70.i ]
  %cmp81.i = icmp slt i32 %0, 0
  br i1 %cmp81.i, label %for.body82.i, label %for.inc201.i

for.body82.i:                                     ; preds = %for.cond76.i
  br label %for.inc197.i

for.inc197.i:                                     ; preds = %for.body82.i, %if.then.i49
  %inc199.i = add nsw i32 undef, 1
  br label %for.cond76.i

for.inc201.i:                                     ; preds = %for.cond76.i
  br label %for.cond65.i

for.inc205.i:                                     ; preds = %for.cond65.i
  br label %for.cond57.i

return:                                           ; preds = %entry
  ret void
}

; A more complx case of irreducible control flow, two interacting loops.
; CHECK: ps_hints_apply
; CHECK: br_table
define void @ps_hints_apply() {
entry:
  br label %psh

psh:                                              ; preds = %entry
  br i1 undef, label %for.cond, label %for.body

for.body:                                         ; preds = %psh
  br label %do.body

do.body:                                          ; preds = %do.cond, %for.body
  %cmp118 = icmp eq i32* undef, undef
  br i1 %cmp118, label %Skip, label %do.cond

do.cond:                                          ; preds = %do.body
  br label %do.body

for.cond:                                         ; preds = %Skip, %psh
  br label %for.body39

for.body39:                                       ; preds = %for.cond
  br i1 undef, label %Skip, label %do.body45

do.body45:                                        ; preds = %for.body39
  unreachable

Skip:                                             ; preds = %for.body39, %do.body
  br label %for.cond
}

; A simple sequence of loops with blocks in between, that should not be
; misinterpreted as irreducible control flow.
; CHECK: fannkuch_worker
; CHECK-NOT: br_table
define i32 @fannkuch_worker(i8* %_arg) {
for.cond:
  br label %do.body

do.body:                                          ; preds = %do.cond, %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.cond1, %do.body
  br i1 true, label %for.cond1, label %for.end

for.end:                                          ; preds = %for.cond1
  br label %do.cond

do.cond:                                          ; preds = %for.end
  br i1 true, label %do.body, label %do.end

do.end:                                           ; preds = %do.cond
  br label %for.cond2

for.cond2:                                        ; preds = %for.end6, %do.end
  br label %for.cond3

for.cond3:                                        ; preds = %for.cond3, %for.cond2
  br i1 true, label %for.cond3, label %for.end6

for.end6:                                         ; preds = %for.cond3
  br label %for.cond2

return:                                           ; No predecessors!
  ret i32 1
}

; Test an interesting pattern of nested irreducibility.

; CHECK: func_2:
; CHECK: br_table
define void @func_2() {
entry:
  br i1 undef, label %lbl_937, label %if.else787

lbl_937:                                          ; preds = %for.body978, %entry
  br label %if.end965

if.else787:                                       ; preds = %entry
  br label %if.end965

if.end965:                                        ; preds = %if.else787, %lbl_937
  br label %for.cond967

for.cond967:                                      ; preds = %for.end1035, %if.end965
  br label %for.cond975

for.cond975:                                      ; preds = %if.end984, %for.cond967
  br i1 undef, label %for.body978, label %for.end1035

for.body978:                                      ; preds = %for.cond975
  br i1 undef, label %lbl_937, label %if.end984

if.end984:                                        ; preds = %for.body978
  br label %for.cond975

for.end1035:                                      ; preds = %for.cond975
  br label %for.cond967
}
