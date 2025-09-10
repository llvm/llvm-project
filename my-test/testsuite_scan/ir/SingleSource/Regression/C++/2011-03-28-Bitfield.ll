; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C++/2011-03-28-Bitfield.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C++/2011-03-28-Bitfield.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct._operation = type { i8, [3 x i8] }

@op = dso_local local_unnamed_addr global %struct._operation zeroinitializer, align 4
@str = private unnamed_addr constant [21 x i8] c"Not 1,2,3 or 4: FAIL\00", align 4
@str.3 = private constant [8 x i8] c"4: PASS\00", align 4
@str.4 = private constant [16 x i8] c"1, 2 or 3: FAIL\00", align 4
@switch.table.main.rel = private unnamed_addr constant [4 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @str.4 to i64), i64 ptrtoint (ptr @switch.table.main.rel to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr @str.4 to i64), i64 ptrtoint (ptr @switch.table.main.rel to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr @str.4 to i64), i64 ptrtoint (ptr @switch.table.main.rel to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr @str.3 to i64), i64 ptrtoint (ptr @switch.table.main.rel to i64)) to i32)], align 4
@switch.table.main.5 = private unnamed_addr constant [4 x i32] [i32 -1, i32 -1, i32 -1, i32 0], align 4

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @_Z4initv() local_unnamed_addr #0 {
  %1 = load i8, ptr @op, align 4
  %2 = and i8 %1, -8
  %3 = or disjoint i8 %2, 4
  store i8 %3, ptr @op, align 4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nounwind uwtable
define dso_local noundef range(i32 -1, 1) i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #1 {
  tail call void @_Z4initv()
  %3 = load i8, ptr @op, align 4
  %4 = and i8 %3, 7
  %5 = add nsw i8 %4, -1
  %6 = icmp ult i8 %5, 4
  br i1 %6, label %7, label %14

7:                                                ; preds = %2
  %8 = zext nneg i8 %5 to i64
  %9 = shl i64 %8, 2
  %10 = call ptr @llvm.load.relative.i64(ptr @switch.table.main.rel, i64 %9)
  %11 = zext nneg i8 %5 to i64
  %12 = getelementptr inbounds nuw i32, ptr @switch.table.main.5, i64 %11
  %13 = load i32, ptr %12, align 4
  br label %14

14:                                               ; preds = %2, %7
  %15 = phi ptr [ %10, %7 ], [ @str, %2 ]
  %16 = phi i32 [ %13, %7 ], [ -1, %2 ]
  %17 = tail call i32 @puts(ptr nonnull dereferenceable(1) %15)
  ret i32 %16
}

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare ptr @llvm.load.relative.i64(ptr, i64) #3

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
