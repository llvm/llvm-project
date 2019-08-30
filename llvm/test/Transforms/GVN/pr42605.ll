; RUN: opt -gvn %s -S | FileCheck %s
; PR42605. Check phi-translate won't translate the value number of a call
; to the value of another call with clobber in between.
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@global = dso_local local_unnamed_addr global i32 0, align 4
@.str = private unnamed_addr constant [8 x i8] c"%d, %d\0A\00", align 1

; Function Attrs: nofree nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr

; Function Attrs: noinline norecurse nounwind readonly uwtable
define dso_local i32 @_Z3gooi(i32 %i) local_unnamed_addr #0 {
entry:
  %t0 = load i32, i32* @global, align 4, !tbaa !2
  %add = add nsw i32 %t0, %i
  ret i32 %add
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @noclobber() local_unnamed_addr {
entry:
  %call = tail call i32 @_Z3gooi(i32 2)
  %add = add nsw i32 %call, 5
  %cmp = icmp sgt i32 %add, 2
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call1 = tail call i32 @_Z3gooi(i32 3)
  %add2 = add nsw i32 %call1, 5
  br label %if.end

; Check pre happens after phitranslate.
; CHECK-LABEL: @noclobber
; CHECK: %add4.pre-phi = phi i32 [ %add2, %if.then ], [ %add, %entry ]
; CHECK: printf(i8* getelementptr inbounds {{.*}}, i32 %add4.pre-phi)

if.end:                                           ; preds = %if.then, %entry
  %i.0 = phi i32 [ 3, %if.then ], [ 2, %entry ]
  %global2.0 = phi i32 [ %add2, %if.then ], [ %add, %entry ]
  %call3 = tail call i32 @_Z3gooi(i32 %i.0)
  %add4 = add nsw i32 %call3, 5
  %call5 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str, i64 0, i64 0), i32 %global2.0, i32 %add4)
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @hasclobber() local_unnamed_addr {
entry:
  %call = tail call i32 @_Z3gooi(i32 2)
  %add = add nsw i32 %call, 5
  %cmp = icmp sgt i32 %add, 2
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call1 = tail call i32 @_Z3gooi(i32 3)
  %add2 = add nsw i32 %call1, 5
  br label %if.end

; Check no pre happens.
; CHECK-LABEL: @hasclobber
; CHECK: %call3 = tail call i32 @_Z3gooi(i32 %i.0)
; CHECK-NEXT: %add4 = add nsw i32 %call3, 5
; CHECK-NEXT: printf(i8* getelementptr inbounds ({{.*}}, i32 %global2.0, i32 %add4)

if.end:                                           ; preds = %if.then, %entry
  %i.0 = phi i32 [ 3, %if.then ], [ 2, %entry ]
  %global2.0 = phi i32 [ %add2, %if.then ], [ %add, %entry ]
  store i32 5, i32* @global, align 4, !tbaa !2
  %call3 = tail call i32 @_Z3gooi(i32 %i.0)
  %add4 = add nsw i32 %call3, 5
  %call5 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str, i64 0, i64 0), i32 %global2.0, i32 %add4)
  ret void
}

attributes #0 = { noinline norecurse nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.0 (trunk 369798)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
