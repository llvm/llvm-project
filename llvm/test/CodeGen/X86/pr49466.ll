; RUN: opt < %s -O2 -codegenprepare -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@b = dso_local local_unnamed_addr global i64 0, align 8
@c = dso_local local_unnamed_addr global i64 0, align 8
@d = dso_local local_unnamed_addr global i64 0, align 8
@e = dso_local local_unnamed_addr global i64 0, align 8
@f = dso_local local_unnamed_addr global i64 0, align 8
@g = dso_local local_unnamed_addr global i64 0, align 8

; CHECK-LABEL: @m(

define dso_local i32 @m() local_unnamed_addr {
entry:
  %0 = load i64, i64* @f, align 8
  %1 = inttoptr i64 %0 to i32*
  %2 = load i64, i64* @c, align 8
  %conv18 = trunc i64 %2 to i32
  %cmp = icmp slt i32 %conv18, 3
  %3 = load i64, i64* @d, align 8
  %conv43 = trunc i64 %3 to i8
  %tobool40.not = icmp eq i8 %conv43, 0
  br label %for.cond

for.cond:                                         ; preds = %for.cond39.preheader, %entry
  %j.0 = phi i32 [ undef, %entry ], [ %j.1.lcssa, %for.cond39.preheader ]
  %p.0 = phi i64 [ undef, %entry ], [ %p.1.lcssa, %for.cond39.preheader ]
  %i.0 = phi i32 [ undef, %entry ], [ %i.1.lcssa, %for.cond39.preheader ]
  %cmp73 = icmp slt i32 %i.0, 3
  br i1 %cmp73, label %for.body.preheader, label %for.cond39.preheader

for.body.preheader:                               ; preds = %for.cond
  br label %for.body

for.cond1.loopexit:                               ; preds = %for.inc34.preheader, %for.end12
  br i1 %cmp, label %for.body, label %for.cond39.preheader.loopexit

for.cond39.preheader.loopexit:                    ; preds = %for.cond1.loopexit
  br label %for.cond39.preheader

for.cond39.preheader:                             ; preds = %for.cond39.preheader.loopexit, %for.cond
  %j.1.lcssa = phi i32 [ %j.0, %for.cond ], [ %conv18, %for.cond39.preheader.loopexit ]
  %p.1.lcssa = phi i64 [ %p.0, %for.cond ], [ 0, %for.cond39.preheader.loopexit ]
  %i.1.lcssa = phi i32 [ %i.0, %for.cond ], [ %conv18, %for.cond39.preheader.loopexit ]
  br i1 %tobool40.not, label %for.cond, label %for.inc42.preheader

for.inc42.preheader:                              ; preds = %for.cond39.preheader
  br label %for.inc42

for.body:                                         ; preds = %for.body.preheader, %for.cond1.loopexit
  %l.176 = phi i8 [ %sub, %for.cond1.loopexit ], [ 0, %for.body.preheader ]
  %p.175 = phi i64 [ 0, %for.cond1.loopexit ], [ %p.0, %for.body.preheader ]
  %j.174 = phi i32 [ %conv18, %for.cond1.loopexit ], [ %j.0, %for.body.preheader ]
  %tobool.not = icmp eq i32 %j.174, 0
  br i1 %tobool.not, label %cleanup45, label %for.cond2.preheader

for.cond2.preheader:                              ; preds = %for.body
  %tobool3.not69 = icmp eq i64 %p.175, 0
  %.pr.pre = load i64, i64* @e, align 8
  br i1 %tobool3.not69, label %for.end12, label %for.body4.preheader

for.body4.preheader:                              ; preds = %for.cond2.preheader
  %4 = sub i64 0, %p.175
  %xtraiter = and i64 %4, 7
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod.not, label %for.body4.prol.loopexit, label %for.body4.prol.preheader

for.body4.prol.preheader:                         ; preds = %for.body4.preheader
  %5 = mul nsw i64 %xtraiter, -1
  br label %for.body4.prol

for.body4.prol:                                   ; preds = %for.body4.prol.preheader, %for.body4.prol
  %lsr.iv = phi i64 [ 0, %for.body4.prol.preheader ], [ %lsr.iv.next, %for.body4.prol ]
  %lsr.iv.next = add nsw i64 %lsr.iv, -1
  %prol.iter.cmp.not = icmp eq i64 %5, %lsr.iv.next
  br i1 %prol.iter.cmp.not, label %for.body4.prol.loopexit.loopexit, label %for.body4.prol

for.body4.prol.loopexit.loopexit:                 ; preds = %for.body4.prol
  %6 = sub i64 %p.175, %lsr.iv.next
  br label %for.body4.prol.loopexit

for.body4.prol.loopexit:                          ; preds = %for.body4.prol.loopexit.loopexit, %for.body4.preheader
  %p.270.unr = phi i64 [ %p.175, %for.body4.preheader ], [ %6, %for.body4.prol.loopexit.loopexit ]
  %7 = icmp ugt i64 %p.175, -8
  br i1 %7, label %for.end12, label %for.body4.preheader89

for.body4.preheader89:                            ; preds = %for.body4.prol.loopexit
  br label %for.body4

for.body4:                                        ; preds = %for.body4.preheader89, %for.body4
  %p.270 = phi i64 [ %inc11.7, %for.body4 ], [ %p.270.unr, %for.body4.preheader89 ]
  %inc11.7 = add i64 %p.270, 8
  %tobool3.not.7 = icmp eq i64 %inc11.7, 0
  br i1 %tobool3.not.7, label %for.end12.loopexit, label %for.body4

for.end12.loopexit:                               ; preds = %for.body4
  br label %for.end12

for.end12:                                        ; preds = %for.end12.loopexit, %for.body4.prol.loopexit, %for.cond2.preheader
  %8 = load i32, i32* %1, align 4
  %conv23 = zext i32 %8 to i64
  %9 = load i64, i64* @b, align 8
  %div24 = udiv i64 %9, %conv23
  store i64 %div24, i64* @b, align 8
  %sub = add i8 %l.176, -1
  %tobool32.not72 = icmp eq i64 %.pr.pre, 0
  br i1 %tobool32.not72, label %for.cond1.loopexit, label %for.inc34.preheader

for.inc34.preheader:                              ; preds = %for.end12
  store i64 0, i64* @e, align 8
  br label %for.cond1.loopexit

for.inc42:                                        ; preds = %for.inc42.preheader, %for.inc42
  br label %for.inc42

cleanup45:                                        ; preds = %for.body
  %cmp13 = icmp ne i8 %l.176, 0
  %conv16 = zext i1 %cmp13 to i32
  ret i32 %conv16
}
