; RUN: opt -loop-reduce -S < %s | FileCheck %s
;
; PR11571: handle a postinc user outside of for.body7 that requires
; recursive expansion of a quadratic recurrence within for.body7. LSR
; needs to forget that for.body7 is a postinc loop during expansion.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32-S128"
target triple = "i386-unknown-freebsd10.0"

@b = external global [121 x i32]

; CHECK-LABEL: @vb(
;   Outer recurrence:
; CHECK: %lsr.iv1 = phi [121 x i32]*
;   Inner recurrence:
; CHECK: %lsr.iv = phi i32
;   Outer step (relative to inner recurrence):
; CHECK: %scevgep = getelementptr i1, i1* %{{.*}}, i32 %lsr.iv
;   Outer use:
; CHECK: %lsr.iv3 = phi [121 x i32]* [ %lsr.iv1, %for.body43.preheader ]
define void @vb() nounwind {
for.cond.preheader:
  br label %for.body7

for.body7:
  %indvars.iv77 = phi i32 [ %indvars.iv.next78, %for.body7 ], [ 1, %for.cond.preheader ]
  %bf.072 = phi i32 [ %t1, %for.body7 ], [ 0, %for.cond.preheader ]
  %t1 = add i32 %bf.072, %indvars.iv77
  %indvars.iv.next78 = add i32 %indvars.iv77, 1
  br i1 undef, label %for.body43, label %for.body7

for.body43:
  %bf.459 = phi i32 [ %inc44, %for.body43 ], [ %t1, %for.body7 ]
  %inc44 = add nsw i32 %bf.459, 1
  %arrayidx45 = getelementptr inbounds [121 x i32], [121 x i32]* @b, i32 0, i32 %bf.459
  %t2 = load i32, i32* %arrayidx45, align 4
  br label %for.body43
}

