; RUN: opt < %s -force-vector-width=4 -force-vector-interleave=1 -loop-vectorize -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; CHECK-LABEL: @PR34687(
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i32 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %[[LATCH:.*]] ]
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ [[TMP17:%.*]], %[[LATCH]] ]
; CHECK:       [[LATCH]]:
; CHECK:         [[TMP13:%.*]] = and <4 x i32> [[VEC_PHI]], <i32 255, i32 255, i32 255, i32 255>
; CHECK-NEXT:    [[TMP14:%.*]] = add nuw nsw <4 x i32> [[TMP13]], {{.*}}
; CHECK-NEXT:    [[INDEX_NEXT]] = add i32 [[INDEX]], 4
; CHECK:         [[TMP16:%.*]] = trunc <4 x i32> [[TMP14]] to <4 x i8>
; CHECK-NEXT:    [[TMP17]] = zext <4 x i8> [[TMP16]] to <4 x i32>
; CHECK-NEXT:    br i1 {{.*}}, label %middle.block, label %vector.body
;
define i8 @PR34687(i1 %c, i32 %x, i32 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %i.next, %if.end ]
  %r = phi i32 [ 0, %entry ], [ %r.next, %if.end ]
  br i1 %c, label %if.then, label %if.end

if.then:
  %tmp0 = sdiv i32 undef, undef
  br label %if.end

if.end:
  %tmp1 = and i32 %r, 255
  %i.next = add nsw i32 %i, 1
  %r.next = add nuw nsw i32 %tmp1, %x
  %cond = icmp eq i32 %i.next, %n
  br i1 %cond, label %for.end, label %for.body

for.end:
  %tmp2 = phi i32 [ %r.next, %if.end ]
  %tmp3 = trunc i32 %tmp2 to i8
  ret i8 %tmp3
}

; CHECK-LABEL: @PR35734(
; CHECK:       vector.ph:
; CHECK:         [[TMP3:%.*]] = insertelement <4 x i32> zeroinitializer, i32 %y, i32 0
; CHECK-NEXT:    br label %vector.body
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i32 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi <4 x i32> [ [[TMP3]], %vector.ph ], [ [[TMP9:%.*]], %vector.body ]
; CHECK:         [[TMP5:%.*]] = and <4 x i32> [[VEC_PHI]], <i32 1, i32 1, i32 1, i32 1>
; CHECK-NEXT:    [[TMP6:%.*]] = add <4 x i32> [[TMP5]], <i32 -1, i32 -1, i32 -1, i32 -1>
; CHECK-NEXT:    [[INDEX_NEXT]] = add i32 [[INDEX]], 4
; CHECK:         [[TMP8:%.*]] = trunc <4 x i32> [[TMP6]] to <4 x i1>
; CHECK-NEXT:    [[TMP9]] = sext <4 x i1> [[TMP8]] to <4 x i32>
; CHECK-NEXT:    br i1 {{.*}}, label %middle.block, label %vector.body
;
define i32 @PR35734(i32 %x, i32 %y) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ %x, %entry ], [ %i.next, %for.body ]
  %r = phi i32 [ %y, %entry ], [ %r.next, %for.body ]
  %tmp0 = and i32 %r, 1
  %r.next = add i32 %tmp0, -1
  %i.next = add nsw i32 %i, 1
  %cond = icmp sgt i32 %i, 77
  br i1 %cond, label %for.end, label %for.body

for.end:
  %tmp1 = phi i32 [ %r.next, %for.body ]
  ret i32 %tmp1
}
