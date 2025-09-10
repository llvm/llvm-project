; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2007-03-02-VaCopy.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2007-03-02-VaCopy.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.__va_list = type { ptr, ptr, ptr, i32, i32 }

@.str = private unnamed_addr constant [11 x i8] c"string %s\0A\00", align 1
@.str.2 = private unnamed_addr constant [4 x i8] c"abc\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local void @testVaCopyArg(ptr readnone captures(none) %0, ...) local_unnamed_addr #0 {
  %2 = alloca %struct.__va_list, align 8
  %3 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #4
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #4
  call void @llvm.va_start.p0(ptr nonnull %2)
  call void @llvm.va_copy.p0(ptr nonnull %3, ptr nonnull %2)
  %4 = getelementptr inbounds nuw i8, ptr %3, i64 24
  %5 = load i32, ptr %4, align 8
  %6 = icmp sgt i32 %5, -1
  br i1 %6, label %15, label %7

7:                                                ; preds = %1
  %8 = add nsw i32 %5, 8
  store i32 %8, ptr %4, align 8
  %9 = icmp samesign ult i32 %5, -7
  br i1 %9, label %10, label %15

10:                                               ; preds = %7
  %11 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %12 = load ptr, ptr %11, align 8
  %13 = sext i32 %5 to i64
  %14 = getelementptr inbounds i8, ptr %12, i64 %13
  br label %18

15:                                               ; preds = %7, %1
  %16 = load ptr, ptr %3, align 8
  %17 = getelementptr inbounds nuw i8, ptr %16, i64 8
  store ptr %17, ptr %3, align 8
  br label %18

18:                                               ; preds = %15, %10
  %19 = phi ptr [ %14, %10 ], [ %16, %15 ]
  %20 = load ptr, ptr %19, align 8, !tbaa !6
  %21 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef %20)
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #4
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #4
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_start.p0(ptr) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_copy.p0(ptr, ptr) #2

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  tail call void (ptr, ...) @testVaCopyArg(ptr nonnull poison, ptr noundef nonnull @.str.2)
  ret i32 0
}

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #3 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 omnipotent char", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
