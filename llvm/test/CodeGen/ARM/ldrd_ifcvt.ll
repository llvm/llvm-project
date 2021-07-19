; RUN: llc -mtriple=armv7a-none-eabi %s -o -  -verify-machineinstrs | FileCheck %s
; RUN: llc -mtriple=thumbv7a-none-eabi %s -o -  -verify-machineinstrs | FileCheck %s
; RUN: llc -mtriple=thumbv7m-none-eabi %s -o -  -verify-machineinstrs | FileCheck %s

; Check we do not hit verifier errors from ifcvting volatile ldrd's
; CHECK: ldrdne
; CHECK: ldrdne
; CHECK: ldrdne
; CHECK: ldrdne

define void @c(i64* %b) noreturn nounwind {
entry:
  br label %for.cond

for.cond:                                         ; preds = %land.end.3, %entry
  %a.0 = phi i32 [ 0, %entry ], [ %conv2.3, %land.end.3 ]
  %tobool.not = icmp eq i32 %a.0, 0
  br i1 %tobool.not, label %land.end, label %land.rhs

land.rhs:                                         ; preds = %for.cond
  %0 = load volatile i64, i64* %b, align 8
  br label %land.end

land.end:                                         ; preds = %land.rhs, %for.cond
  %sub.i = add nuw nsw i32 %a.0, 65533
  %conv2 = and i32 %sub.i, 65535
  %tobool.not.1 = icmp eq i32 %conv2, 0
  br i1 %tobool.not.1, label %land.end.1, label %land.rhs.1

land.rhs.1:                                       ; preds = %land.end
  %1 = load volatile i64, i64* %b, align 8
  br label %land.end.1

land.end.1:                                       ; preds = %land.rhs.1, %land.end
  %sub.i.1 = add nuw nsw i32 %a.0, 65530
  %conv2.1 = and i32 %sub.i.1, 65535
  %tobool.not.2 = icmp eq i32 %conv2.1, 0
  br i1 %tobool.not.2, label %land.end.2, label %land.rhs.2

land.rhs.2:                                       ; preds = %land.end.1
  %2 = load volatile i64, i64* %b, align 8
  br label %land.end.2

land.end.2:                                       ; preds = %land.rhs.2, %land.end.1
  %sub.i.2 = add nuw nsw i32 %a.0, 65527
  %conv2.2 = and i32 %sub.i.2, 65535
  %tobool.not.3 = icmp eq i32 %conv2.2, 0
  br i1 %tobool.not.3, label %land.end.3, label %land.rhs.3

land.rhs.3:                                       ; preds = %land.end.2
  %3 = load volatile i64, i64* %b, align 8
  br label %land.end.3

land.end.3:                                       ; preds = %land.rhs.3, %land.end.2
  %sub.i.3 = add nuw nsw i32 %a.0, 65524
  %conv2.3 = and i32 %sub.i.3, 65535
  br label %for.cond
}
