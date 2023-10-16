; RUN:opt -passes=licm -S < %s | FileCheck %s 

@stderr = external local_unnamed_addr global ptr, align 8
@.str = private unnamed_addr constant [17 x i8] c"COUNT|%ld|1|lps\0A\00", align 1
@Run_Index = dso_local local_unnamed_addr global i64 0, align 8
@.str.1 = private unnamed_addr constant [20 x i8] c"Usage: %s duration\0A\00", align 1

; Function Attrs: nounwind uwtable
define dso_local void @wake_me(i32 noundef %seconds, ptr noundef %func) local_unnamed_addr #0 {
entry:
  %call = tail call ptr @signal(i32 noundef 14, ptr noundef %func) #7
  %call1 = tail call i32 @alarm(i32 noundef %seconds) #7
  ret void
}

; Function Attrs: nounwind
declare ptr @signal(i32 noundef, ptr noundef) local_unnamed_addr #1

; Function Attrs: nounwind
declare i32 @alarm(i32 noundef) local_unnamed_addr #1

; Function Attrs: noreturn nounwind uwtable
define dso_local void @report() #2 {
entry:
  %0 = load ptr, ptr @stderr, align 8, !tbaa !6
  %1 = load i64, ptr @Run_Index, align 8, !tbaa !10
  %call = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %0, ptr noundef nonnull @.str, i64 noundef %1) #8
  tail call void @exit(i32 noundef 0) #9
  unreachable
}

; Function Attrs: nofree nounwind
declare noundef i32 @fprintf(ptr nocapture noundef, ptr nocapture noundef readonly, ...) local_unnamed_addr #3

; Function Attrs: noreturn nounwind
declare void @exit(i32 noundef) local_unnamed_addr #4

; Function Attrs: noreturn nounwind uwtable
define dso_local i32 @main(i32 noundef %argc, ptr nocapture noundef readonly %argv) local_unnamed_addr #2 {
entry:
  %cmp.not = icmp eq i32 %argc, 2
  br i1 %cmp.not, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %0 = load ptr, ptr @stderr, align 8, !tbaa !6
  %1 = load ptr, ptr %argv, align 8, !tbaa !6
  %call = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %0, ptr noundef nonnull @.str.1, ptr noundef %1) #8
  tail call void @exit(i32 noundef 1) #9
  unreachable

if.end:                                           ; preds = %entry
  %arrayidx1 = getelementptr inbounds ptr, ptr %argv, i64 1
  %2 = load ptr, ptr %arrayidx1, align 8, !tbaa !6
  %call.i = tail call i64 @strtol(ptr nocapture noundef nonnull %2, ptr noundef null, i32 noundef 10) #7
  %conv.i = trunc i64 %call.i to i32
  store i64 0, ptr @Run_Index, align 8, !tbaa !10
  %call.i4 = tail call ptr @signal(i32 noundef 14, ptr noundef nonnull @report) #7
  %call1.i = tail call i32 @alarm(i32 noundef %conv.i) #7
  store i64 1, ptr @Run_Index, align 8, !tbaa !10
  br label %for.cond

; CHECK-LABEL: for.cond
; CHECK: store
for.cond:                                         ; preds = %for.cond, %if.end
  %3 = load i64, ptr @Run_Index, align 8, !tbaa !10
  %inc = add i64 %3, 1
  store i64 %inc, ptr @Run_Index, align 8, !tbaa !10
  br label %for.cond
}

; Function Attrs: inlinehint mustprogress nofree nounwind readonly willreturn uwtable
define available_externally i32 @atoi(ptr noundef nonnull %__nptr) local_unnamed_addr #5 {
entry:
  %call = tail call i64 @strtol(ptr nocapture noundef nonnull %__nptr, ptr noundef null, i32 noundef 10) #7
  %conv = trunc i64 %call to i32
  ret i32 %conv
}

; Function Attrs: mustprogress nofree nounwind willreturn
declare i64 @strtol(ptr noundef readonly, ptr nocapture noundef, i32 noundef) local_unnamed_addr #6

attributes #0 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+neon,+v8a" }
attributes #1 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+neon,+v8a" }
attributes #2 = { noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+neon,+v8a" }
attributes #3 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+neon,+v8a" }
attributes #4 = { noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+neon,+v8a" }
attributes #5 = { inlinehint mustprogress nofree nounwind readonly willreturn uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+neon,+v8a" }
attributes #6 = { mustprogress nofree nounwind willreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+neon,+v8a" }
attributes #7 = { nounwind }
attributes #8 = { cold }
attributes #9 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"Huawei BiSheng Compiler clang version 15.0.4 (b1a488d0a16f)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"any pointer", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"long", !8, i64 0}

