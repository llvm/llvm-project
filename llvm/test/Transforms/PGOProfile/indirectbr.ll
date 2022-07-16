; RUN: llvm-profdata merge %S/Inputs/indirectbr.proftext -o %t.profdata
; RUN: opt < %s -passes=pgo-instr-use -pgo-instrument-entry=false -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefix=USE
; RUN: opt < %s -passes=pgo-instr-use -pgo-instrument-entry=false -pgo-test-profile-file=%t.profdata -S | opt -S -passes='print<branch-prob>' 2>&1 | FileCheck %s --check-prefix=BRANCHPROB
; RUN: llvm-profdata merge %S/Inputs/indirectbr_entry.proftext -o %t2.profdata

; RUN: opt < %s -passes=pgo-instr-use -pgo-instrument-entry=true -pgo-test-profile-file=%t2.profdata -S | FileCheck %s --check-prefix=USE
; RUN: opt < %s -passes=pgo-instr-use -pgo-instrument-entry=true -pgo-test-profile-file=%t2.profdata -S | opt -S -passes='print<branch-prob>' 2>&1 | FileCheck %s --check-prefix=BRANCHPROB

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@foo.table = internal unnamed_addr constant [3 x ptr] [ptr blockaddress(@foo, %return), ptr blockaddress(@foo, %label2), ptr blockaddress(@foo, %label3)], align 16

define i32 @foo(i32 %i) {
entry:
  %cmp = icmp ult i32 %i, 3
  br i1 %cmp, label %if.then, label %return

if.then:
  %idxprom = zext i32 %i to i64
  %arrayidx = getelementptr inbounds [3 x ptr], ptr @foo.table, i64 0, i64 %idxprom
  %0 = load ptr, ptr %arrayidx, align 8
  indirectbr ptr %0, [label %return, label %label2, label %label3]
; USE:  indirectbr ptr %0, [label %return, label %label2, label %label3]
; USE-SAME: !prof ![[BW_INDBR:[0-9]+]]
; USE: ![[BW_INDBR]] = !{!"branch_weights", i32 63, i32 20, i32 5}

label2:
  br label %return

label3:
  br label %return

return:
  %retval.0 = phi i32 [ 3, %label3 ], [ 2, %label2 ], [ 0, %entry ], [ 1, %if.then ]
  ret i32 %retval.0
}

; BRANCHPROB: Printing analysis {{.*}} for function 'foo':
; BRANCHPROB:---- Branch Probabilities ----
; BRANCHPROB:  edge entry -> if.then probability is 0x37c32b17 / 0x80000000 = 43.56%
; BRANCHPROB:  edge entry -> return.clone probability is 0x483cd4e9 / 0x80000000 = 56.44%
; BRANCHPROB:  edge if.then -> return probability is 0x5ba2e8ba / 0x80000000 = 71.59%
; BRANCHPROB:  edge if.then -> label2 probability is 0x1d1745d1 / 0x80000000 = 22.73%
; BRANCHPROB:  edge if.then -> label3 probability is 0x0745d174 / 0x80000000 = 5.68%
; BRANCHPROB:  edge label2 -> return.clone probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]
; BRANCHPROB:  edge label3 -> return.clone probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]
; BRANCHPROB:  edge return -> .split probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]
; BRANCHPROB:  edge return.clone -> .split probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]



