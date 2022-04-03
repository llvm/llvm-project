; RUN: opt < %s  -loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -dce -instcombine -S
; RUN: opt < %s -debugify -loop-vectorize -S | FileCheck %s --check-prefix=DEBUGLOC

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

@a = common global i8 0, align 1
@b = common global i8 0, align 1

define void @f() nounwind uwtable ssp {
; Check that the induction phis and adds have debug location.
;
; DEBUGLOC-LABEL: vector.body:
; DEBUGLOC:         %vec.ind = phi {{.*}}, !dbg ![[DbgLoc:[0-9]+]]
; DEBUGLOC:         %vec.ind.next = add {{.*}}, !dbg ![[DbgLoc]]

scalar.ph:
  store i8 0, i8* inttoptr (i64 1 to i8*), align 1
  %0 = load i8, i8* @a, align 1
  br label %for.body

for.body:
  %mul16 = phi i8 [ 0, %scalar.ph ], [ %mul, %for.body ]              ; <------- i8 induction var.
  %c.015 = phi i8 [ undef, %scalar.ph ], [ %conv8, %for.body ]
  %conv2 = sext i8 %c.015 to i32
  %tobool = icmp ne i8 %c.015, 0
  %.sink = select i1 %tobool, i8 %c.015, i8 %0
  %mul = mul i8 %mul16, %.sink
  %add = add nsw i32 %conv2, 1
  %conv8 = trunc i32 %add to i8
  %sext = shl i32 %add, 24
  %phitmp14 = icmp slt i32 %sext, 268435456
  br i1 %phitmp14, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  store i8 %mul, i8* @b, align 1
  ret void
}

; Check that the location of the new phi comes from %c.015 = phi i8
; DEBUGLOC:         ![[DbgLoc]] = !DILocation(line: 5
