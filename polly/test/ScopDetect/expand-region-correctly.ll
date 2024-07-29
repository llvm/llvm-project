; RUN: opt %loadNPMPolly '-passes=print<polly-detect>' -disable-output < %s 2>&1 | FileCheck %s

; CHECK: Valid Region for Scop: if.end.1631 => for.cond.1647.outer

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @qtm_decompress() {
entry:
  br label %while.cond.outer.outer

while.cond.outer.outer:                           ; preds = %entry
  br label %if.end.1631

if.end.1631:                                      ; preds = %do.end.1721, %while.cond.outer.outer
  br i1 false, label %for.cond.1647.preheader, label %if.then.1635

if.then.1635:                                     ; preds = %if.end.1631
  br label %for.cond.1647.preheader

for.cond.1647.preheader:                          ; preds = %if.then.1635, %if.end.1631
  br label %for.cond.1647.outer

for.cond.1647.outer:                              ; preds = %do.end.1685, %for.cond.1647.preheader
  %bits_needed.5.ph = phi i8 [ 8, %for.cond.1647.preheader ], [ 0, %do.end.1685 ]
  br label %for.cond.1647

for.cond.1647:                                    ; preds = %do.cond.1718, %for.cond.1647.outer
  br i1 undef, label %do.cond.1718, label %if.then.1659

if.then.1659:                                     ; preds = %for.cond.1647
  br i1 false, label %do.end.1685, label %if.then.1662

if.then.1662:                                     ; preds = %if.then.1659
  unreachable

do.end.1685:                                      ; preds = %if.then.1659
  br label %for.cond.1647.outer

do.cond.1718:                                     ; preds = %for.cond.1647
  br i1 false, label %do.end.1721, label %for.cond.1647

do.end.1721:                                      ; preds = %do.cond.1718
  br label %if.end.1631
}
