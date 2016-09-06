; RUN: llc %s -o - | FileCheck %s
; This file checks some weird corner case in LiveRangeEdit.
; We used to do crash when we eliminate the definition
; of the product of splitting when the original live-range
; has already been removed.
; Basically, we have the following input.
; v1 = loadimm cst
; ...
; = use v1
;
; We split the live-range like this:
; v1 = loadimm cst
; ...
; v2 = copy v1
; ...
; = use v2
;
; We actually issue loadimm instead of the copy:
; v1 = loadimm cst
; ...
; v2 = loadimm cst 
; ...
; = use v2
;
; v1 is now dead so we remove its live-range.
; Actually, we shrink it to empty to keep the
; instruction around for futher remat opportunities
; (accessbile via the origin pointer.)
;
; Later v2 gets remove as well (e.g., because we
; remat it closer to its use) and the live-range
; gets eliminated. We used to crash at this point
; because we were looking for a VNI of origin (v1)
; at the slot index of the definition of v2. However,
; we do not have a VNI for v1 at this point, since the
; live-range is now empty... crash!
; PR27983

source_filename = "bugpoint-output-1e29d28.bc"
target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

@r = external global i32, align 4
@k = external global i32, align 4
@g = external global i32, align 4
@a = external global i16, align 2
@p = external global i32, align 4
@n = external global i16, align 2
@.str = external unnamed_addr constant [12 x i8], align 1
@.str.1 = external unnamed_addr constant [13 x i8], align 1
@s = external global i32, align 4
@z = external global i16, align 2

; CHECK-LABEL: fn1:
define void @fn1() #0 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %cleanup100, %for.end29, %entry
  %t7.0 = phi i16 [ undef, %entry ], [ %t7.1, %for.end29 ], [ %t7.19, %cleanup100 ]
  %t9.0 = phi i32 [ undef, %entry ], [ %t9.1, %for.end29 ], [ 0, %cleanup100 ]
  %t2.0 = phi i32 [ undef, %entry ], [ undef, %for.end29 ], [ %t2.18, %cleanup100 ]
  %tmp = load i32, i32* @r, align 4
  br i1 undef, label %if.then, label %if.end7

if.then:                                          ; preds = %for.cond
  %tobool = icmp ne i32 %tmp, 0
  %tobool1 = icmp ne i32 %t2.0, 0
  %tmp1 = and i1 %tobool1, %tobool
  %land.ext = zext i1 %tmp1 to i32
  %tmp2 = load i32, i32* @k, align 4
  %shr = lshr i32 %land.ext, %tmp2
  %tobool4 = icmp ne i32 %shr, 0
  %or.cond = and i1 false, %tobool4
  br i1 %or.cond, label %L6, label %if.end7

if.end7:                                          ; preds = %if.then, %for.cond
  %t2.1 = phi i32 [ %shr, %if.then ], [ %t2.0, %for.cond ]
  %tobool8 = icmp eq i32 undef, 0
  br i1 %tobool8, label %if.end11, label %for.cond10

for.cond10:                                       ; preds = %for.cond10, %if.end7
  br label %for.cond10

if.end11:                                         ; preds = %if.end7
  %tmp3 = load i32, i32* @g, align 4
  %tmp4 = load i16, i16* @a, align 2
  %conv = sext i16 %tmp4 to i32
  %div = sdiv i32 %tmp3, %conv
  %tobool12 = icmp eq i32 %div, 0
  br i1 %tobool12, label %for.cond15, label %L5

for.cond15:                                       ; preds = %for.cond17, %if.end11
  %t7.1 = phi i16 [ %t7.2, %for.cond17 ], [ %t7.0, %if.end11 ]
  %t9.1 = phi i32 [ %t9.2, %for.cond17 ], [ %t9.0, %if.end11 ]
  %tobool16 = icmp eq i32 undef, 0
  br i1 %tobool16, label %for.end29, label %for.cond17

for.cond17:                                       ; preds = %for.cond20, %for.cond15
  %t7.2 = phi i16 [ %t7.3, %for.cond20 ], [ %t7.1, %for.cond15 ]
  %t9.2 = phi i32 [ undef, %for.cond20 ], [ %t9.1, %for.cond15 ]
  %tobool18 = icmp eq i8 undef, 0
  br i1 %tobool18, label %for.cond15, label %for.cond20

for.cond20:                                       ; preds = %for.cond23, %for.cond17
  %t7.3 = phi i16 [ %t7.4, %for.cond23 ], [ %t7.2, %for.cond17 ]
  %tobool21 = icmp eq i32 undef, 0
  br i1 %tobool21, label %for.cond17, label %for.cond23

for.cond23:                                       ; preds = %L1, %for.cond20
  %t7.4 = phi i16 [ %t7.5, %L1 ], [ %t7.3, %for.cond20 ]
  %tobool24 = icmp eq i8 undef, 0
  br i1 %tobool24, label %for.cond20, label %L1

L1:                                               ; preds = %cleanup100, %for.cond23
  %t7.5 = phi i16 [ %t7.19, %cleanup100 ], [ %t7.4, %for.cond23 ]
  %conv26 = sext i16 undef to i64
  br label %for.cond23

for.end29:                                        ; preds = %for.cond15
  br i1 undef, label %for.cond, label %for.cond32thread-pre-split

for.cond32thread-pre-split:                       ; preds = %for.end29
  %.pr = load i32, i32* @p, align 4
  br label %for.cond32

for.cond32:                                       ; preds = %for.inc94, %for.cond32thread-pre-split
  %t7.6 = phi i16 [ %t7.1, %for.cond32thread-pre-split ], [ %t7.17, %for.inc94 ]
  %t3.4 = phi i64 [ 0, %for.cond32thread-pre-split ], [ 0, %for.inc94 ]
  %t9.6 = phi i32 [ %t9.1, %for.cond32thread-pre-split ], [ 0, %for.inc94 ]
  %t2.7 = phi i32 [ undef, %for.cond32thread-pre-split ], [ %t2.16, %for.inc94 ]
  %tobool33 = icmp eq i32 0, 0
  br i1 %tobool33, label %for.end95, label %for.body34

for.body34:                                       ; preds = %for.cond32
  %tobool35 = icmp eq i16 undef, 0
  br i1 %tobool35, label %for.inc94, label %if.then36

if.then36:                                        ; preds = %for.body34
  %tmp5 = load i16, i16* @n, align 2
  %tobool37 = icmp eq i32 undef, 0
  br i1 %tobool37, label %if.end78, label %if.then38

if.then38:                                        ; preds = %if.then36
  tail call void (i8*, ...) @printf(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str, i32 0, i32 0), i64 undef)
  %tobool40 = icmp eq i32 undef, 0
  br i1 %tobool40, label %L3, label %cleanup100

L3:                                               ; preds = %while.end.split, %if.then38
  %t7.7 = phi i16 [ %tmp5, %if.then38 ], [ %t7.15, %while.end.split ]
  %t3.5 = phi i64 [ %t3.4, %if.then38 ], [ %t3.11, %while.end.split ]
  %t2.8 = phi i32 [ %t2.7, %if.then38 ], [ %t2.14, %while.end.split ]
  %tobool43 = icmp eq i32 undef, 0
  br i1 %tobool43, label %if.end48, label %cleanup75

if.end48:                                         ; preds = %L3
  tail call void (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.1, i32 0, i32 0), i64 %t3.5)
  br i1 undef, label %if.end61, label %for.cond52.preheader

for.cond52.preheader:                             ; preds = %if.end48
  %tobool57 = icmp eq i16 undef, 0
  %.130 = select i1 %tobool57, i16 -8, i16 0
  br label %if.end61

if.end61:                                         ; preds = %for.cond52.preheader, %if.end48
  %t7.9 = phi i16 [ %t7.7, %if.end48 ], [ %.130, %for.cond52.preheader ]
  %tobool62 = icmp eq i32 undef, 0
  br i1 %tobool62, label %if.end71, label %if.then63

if.then63:                                        ; preds = %if.end61
  br i1 undef, label %if.end67, label %L5

L5:                                               ; preds = %cleanup100.L5_crit_edge, %if.then63, %if.end11
  %.pre = phi i32 [ %.pre.pre, %cleanup100.L5_crit_edge ], [ undef, %if.then63 ], [ %tmp, %if.end11 ]
  %t7.10 = phi i16 [ %t7.19, %cleanup100.L5_crit_edge ], [ %t7.9, %if.then63 ], [ %t7.0, %if.end11 ]
  %t3.6 = phi i64 [ 0, %cleanup100.L5_crit_edge ], [ %t3.5, %if.then63 ], [ 2, %if.end11 ]
  %t9.8 = phi i32 [ 0, %cleanup100.L5_crit_edge ], [ undef, %if.then63 ], [ %t9.0, %if.end11 ]
  %t2.9 = phi i32 [ %t2.18, %cleanup100.L5_crit_edge ], [ %t2.8, %if.then63 ], [ %t2.1, %if.end11 ]
  store i32 %t9.8, i32* @s, align 4
  br label %if.end67

if.end67:                                         ; preds = %L5, %if.then63
  %tmp6 = phi i32 [ %.pre, %L5 ], [ undef, %if.then63 ]
  %t7.11 = phi i16 [ %t7.10, %L5 ], [ %t7.9, %if.then63 ]
  %t3.7 = phi i64 [ %t3.6, %L5 ], [ %t3.5, %if.then63 ]
  %t9.9 = phi i32 [ %t9.8, %L5 ], [ undef, %if.then63 ]
  %t2.10 = phi i32 [ %t2.9, %L5 ], [ %t2.8, %if.then63 ]
  %tobool68 = icmp eq i32 %tmp6, 0
  br i1 %tobool68, label %if.end71, label %for.end95

if.end71:                                         ; preds = %if.end67, %if.end61
  %t7.12 = phi i16 [ %t7.11, %if.end67 ], [ %t7.9, %if.end61 ]
  %t3.8 = phi i64 [ %t3.7, %if.end67 ], [ %t3.5, %if.end61 ]
  %tobool72 = icmp eq i32 undef, 0
  br i1 %tobool72, label %cleanup75.thread128, label %if.then73

if.then73:                                        ; preds = %if.end71
  br label %cleanup100

cleanup75.thread128:                              ; preds = %if.end71
  br label %if.end78

cleanup75:                                        ; preds = %L3
  br i1 false, label %for.cond98, label %for.end95

if.end78:                                         ; preds = %cleanup75.thread128, %if.then36
  %t7.14 = phi i16 [ %tmp5, %if.then36 ], [ 0, %cleanup75.thread128 ]
  %t3.10 = phi i64 [ %t3.4, %if.then36 ], [ %t3.8, %cleanup75.thread128 ]
  %t9.12 = phi i32 [ %t9.6, %if.then36 ], [ undef, %cleanup75.thread128 ]
  %t2.13 = phi i32 [ %t2.7, %if.then36 ], [ undef, %cleanup75.thread128 ]
  store i16 %t7.14, i16* @z, align 2
  br label %L6

L6:                                               ; preds = %if.end78, %if.then
  %t7.15 = phi i16 [ %t7.0, %if.then ], [ %t7.14, %if.end78 ]
  %t3.11 = phi i64 [ 2, %if.then ], [ %t3.10, %if.end78 ]
  %t9.13 = phi i32 [ %t9.0, %if.then ], [ %t9.12, %if.end78 ]
  %t2.14 = phi i32 [ %shr, %if.then ], [ %t2.13, %if.end78 ]
  br i1 undef, label %while.condthread-pre-split, label %for.inc94

while.condthread-pre-split:                       ; preds = %L6
  %tobool83 = icmp eq i32 undef, 0
  br i1 %tobool83, label %while.end.split, label %while.cond

while.cond:                                       ; preds = %while.cond, %while.condthread-pre-split
  br label %while.cond

while.end.split:                                  ; preds = %while.condthread-pre-split
  %tobool84 = icmp eq i16 undef, 0
  br i1 %tobool84, label %for.inc94, label %L3

for.inc94:                                        ; preds = %while.end.split, %L6, %for.body34
  %t7.17 = phi i16 [ %t7.6, %for.body34 ], [ %t7.15, %L6 ], [ %t7.15, %while.end.split ]
  %t2.16 = phi i32 [ %t2.7, %for.body34 ], [ %t2.14, %L6 ], [ %t2.14, %while.end.split ]
  store i32 undef, i32* @p, align 4
  br label %for.cond32

for.end95:                                        ; preds = %cleanup75, %if.end67, %for.cond32
  %t7.18 = phi i16 [ %t7.6, %for.cond32 ], [ %t7.7, %cleanup75 ], [ %t7.11, %if.end67 ]
  %t2.17 = phi i32 [ %t2.7, %for.cond32 ], [ %t2.8, %cleanup75 ], [ %t2.10, %if.end67 ]
  %tobool96 = icmp eq i32 undef, 0
  br i1 %tobool96, label %cleanup100, label %for.cond98

for.cond98:                                       ; preds = %for.cond98, %for.end95, %cleanup75
  br label %for.cond98

cleanup100:                                       ; preds = %for.end95, %if.then73, %if.then38
  %t7.19 = phi i16 [ %t7.18, %for.end95 ], [ %tmp5, %if.then38 ], [ %t7.12, %if.then73 ]
  %t2.18 = phi i32 [ %t2.17, %for.end95 ], [ %t2.7, %if.then38 ], [ undef, %if.then73 ]
  switch i32 undef, label %unreachable [
    i32 0, label %for.cond
    i32 17, label %L1
    i32 7, label %cleanup100.L5_crit_edge
  ]

cleanup100.L5_crit_edge:                          ; preds = %cleanup100
  %.pre.pre = load i32, i32* @r, align 4
  br label %L5

unreachable:                                      ; preds = %cleanup100
  unreachable
}

; Function Attrs: nounwind
declare void @printf(i8* nocapture readonly, ...) #1

attributes #0 = { noreturn nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
