; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr60017.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr60017.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.S0 = type { i16, i16 }

@x = dso_local local_unnamed_addr global { i8, [2 x [2 x i8]], i8, [2 x %struct.S0], [2 x i8] } { i8 1, [2 x [2 x i8]] [[2 x i8] c"\02\03", [2 x i8] c"\04\05"], i8 0, [2 x %struct.S0] [%struct.S0 { i16 6, i16 7 }, %struct.S0 { i16 8, i16 9 }], [2 x i8] zeroinitializer }, align 8

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local [2 x i64] @func() local_unnamed_addr #0 {
  %1 = load i64, ptr @x, align 8
  %2 = load i64, ptr getelementptr inbounds nuw (i8, ptr @x, i64 8), align 8
  %3 = insertvalue [2 x i64] poison, i64 %1, 0
  %4 = insertvalue [2 x i64] %3, i64 %2, 1
  ret [2 x i64] %4
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = load i64, ptr getelementptr inbounds nuw (i8, ptr @x, i64 8), align 8
  %2 = and i64 %1, 281470681743360
  %3 = icmp eq i64 %2, 38654705664
  br i1 %3, label %5, label %4

4:                                                ; preds = %0
  tail call void @abort() #3
  unreachable

5:                                                ; preds = %0
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
