; RUN: opt %loadPolly -polly-scops -disable-output < %s
;
; Check we do not crash.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.regnode_charclass_class.2.42.654.690.726.870.978.1770.1806.1842.2166.2274.2382.2598.2814.3030.3064 = type { i8, i8, i16, i32, [32 x i8], [4 x i8] }

; Function Attrs: nounwind uwtable
define void @S_cl_or(%struct.regnode_charclass_class.2.42.654.690.726.870.978.1770.1806.1842.2166.2274.2382.2598.2814.3030.3064* %cl, %struct.regnode_charclass_class.2.42.654.690.726.870.978.1770.1806.1842.2166.2274.2382.2598.2814.3030.3064* %or_with) #0 {
entry:
  %flags = getelementptr inbounds %struct.regnode_charclass_class.2.42.654.690.726.870.978.1770.1806.1842.2166.2274.2382.2598.2814.3030.3064, %struct.regnode_charclass_class.2.42.654.690.726.870.978.1770.1806.1842.2166.2274.2382.2598.2814.3030.3064* %or_with, i64 0, i32 0
  %0 = load i8, i8* %flags, align 4, !tbaa !1
  %conv = zext i8 %0 to i32
  %1 = load i8, i8* undef, align 4, !tbaa !1
  br label %land.lhs.true35

land.lhs.true35:                                  ; preds = %entry
  %and38 = and i32 %conv, 2
  %tobool39 = icmp ne i32 %and38, 0
  %and42 = and i8 %1, 2
  %tobool43 = icmp eq i8 %and42, 0
  %or.cond45 = and i1 %tobool39, %tobool43
  br i1 %or.cond45, label %if.end91, label %for.body49

for.body49:                                       ; preds = %land.lhs.true35
  %2 = load i8, i8* %flags, align 4, !tbaa !1
  %and65 = and i8 %2, 8
  %tobool66 = icmp eq i8 %and65, 0
  br i1 %tobool66, label %if.end91, label %for.body71

for.body71:                                       ; preds = %for.body71, %for.body49
  %arrayidx77 = getelementptr inbounds %struct.regnode_charclass_class.2.42.654.690.726.870.978.1770.1806.1842.2166.2274.2382.2598.2814.3030.3064, %struct.regnode_charclass_class.2.42.654.690.726.870.978.1770.1806.1842.2166.2274.2382.2598.2814.3030.3064* %cl, i64 0, i32 5, i64 0
  store i8 undef, i8* %arrayidx77, align 1, !tbaa !7
  br i1 false, label %for.body71, label %if.end91

if.end91:                                         ; preds = %for.body71, %for.body49, %land.lhs.true35
  ret void
}

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.9.0"}
!1 = !{!2, !3, i64 0}
!2 = !{!"regnode_charclass_class", !3, i64 0, !3, i64 1, !5, i64 2, !6, i64 4, !3, i64 8, !3, i64 40}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!"short", !3, i64 0}
!6 = !{!"int", !3, i64 0}
!7 = !{!3, !3, i64 0}
