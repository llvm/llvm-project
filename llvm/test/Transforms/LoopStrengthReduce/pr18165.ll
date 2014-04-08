; RUN: opt < %s -loop-reduce -S | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32-S128"
target triple = "x86_64-apple-macosx10.9.0"

; LSR shouldn't reuse IV if the resultant offset is not valid for the operand type.
; CHECK-NOT: trunc i32 %.ph to i8

%struct.anon = type { i32, i32, i32 }

@c = global i32 1, align 4
@.str = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
@b = common global i32 0, align 4
@a = common global %struct.anon zeroinitializer, align 4
@e = common global %struct.anon zeroinitializer, align 4
@d = common global i32 0, align 4
@f = common global i32 0, align 4
@g = common global i32 0, align 4
@h = common global i32 0, align 4

; Function Attrs: nounwind optsize ssp uwtable
define i32 @main() #0 {
entry:
  %0 = load i32* getelementptr inbounds (%struct.anon* @a, i64 0, i32 0), align 4, !tbaa !1
  %tobool7.i = icmp eq i32 %0, 0
  %.promoted.i = load i32* getelementptr inbounds (%struct.anon* @a, i64 0, i32 2), align 4, !tbaa !6
  %f.promoted.i = load i32* @f, align 4, !tbaa !7
  br label %for.body6.i.outer

for.body6.i.outer:                                ; preds = %entry, %lor.end.i
  %.ph = phi i32 [ %add.i, %lor.end.i ], [ 0, %entry ]
  %or1512.i.ph = phi i32 [ %or15.i, %lor.end.i ], [ %f.promoted.i, %entry ]
  %or1410.i.ph = phi i32 [ %or14.i, %lor.end.i ], [ %.promoted.i, %entry ]
  %p.addr.16.i.ph = phi i8 [ %inc10.i, %lor.end.i ], [ -128, %entry ]
  br i1 %tobool7.i, label %if.end9.i, label %lbl.loopexit.i

lbl.loopexit.i:                                   ; preds = %for.body6.i.outer, %lbl.loopexit.i
  br label %lbl.loopexit.i

if.end9.i:                                        ; preds = %for.body6.i.outer
  %inc10.i = add i8 %p.addr.16.i.ph, 1
  %tobool12.i = icmp eq i8 %p.addr.16.i.ph, 0
  br i1 %tobool12.i, label %lor.rhs.i, label %lor.end.i

lor.rhs.i:                                        ; preds = %if.end9.i
  %1 = load i32* @b, align 4, !tbaa !7
  %dec.i = add nsw i32 %1, -1
  store i32 %dec.i, i32* @b, align 4, !tbaa !7
  %tobool13.i = icmp ne i32 %1, 0
  br label %lor.end.i

lor.end.i:                                        ; preds = %lor.rhs.i, %if.end9.i
  %2 = phi i1 [ true, %if.end9.i ], [ %tobool13.i, %lor.rhs.i ]
  %lor.ext.i = zext i1 %2 to i32
  %or14.i = or i32 %lor.ext.i, %or1410.i.ph
  %or15.i = or i32 %or14.i, %or1512.i.ph
  %add.i = add nsw i32 %.ph, 2
  %cmp.i = icmp slt i32 %add.i, 21
  br i1 %cmp.i, label %for.body6.i.outer, label %fn1.exit

fn1.exit:                                         ; preds = %lor.end.i
  store i32 0, i32* @g, align 4, !tbaa !7
  store i32 %or14.i, i32* getelementptr inbounds (%struct.anon* @a, i64 0, i32 2), align 4, !tbaa !6
  store i32 %or15.i, i32* @f, align 4, !tbaa !7
  store i32 %add.i, i32* getelementptr inbounds (%struct.anon* @e, i64 0, i32 1), align 4, !tbaa !8
  store i32 0, i32* @h, align 4, !tbaa !7
  %3 = load i32* @b, align 4, !tbaa !7
  %call1 = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i64 0, i64 0), i32 %3) #2
  ret i32 0
}

; Function Attrs: nounwind optsize
declare i32 @printf(i8* nocapture readonly, ...) #1

attributes #0 = { nounwind optsize ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind optsize "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind optsize }

!llvm.ident = !{!0}

!0 = metadata !{metadata !"clang version 3.5 "}
!1 = metadata !{metadata !2, metadata !3, i64 0}
!2 = metadata !{metadata !"", metadata !3, i64 0, metadata !3, i64 4, metadata !3, i64 8}
!3 = metadata !{metadata !"int", metadata !4, i64 0}
!4 = metadata !{metadata !"omnipotent char", metadata !5, i64 0}
!5 = metadata !{metadata !"Simple C/C++ TBAA"}
!6 = metadata !{metadata !2, metadata !3, i64 8}
!7 = metadata !{metadata !3, metadata !3, i64 0}
!8 = metadata !{metadata !2, metadata !3, i64 4}
