; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr88693.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr88693.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@bar.u = internal unnamed_addr constant [9 x i8] c"abcdefghi", align 1
@baz.u = internal unnamed_addr constant [9 x i8] c"jklmnopqr", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local void @foo(ptr noundef readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %0) #5
  %3 = icmp eq i64 %2, 9
  br i1 %3, label %5, label %4

4:                                                ; preds = %1
  tail call void @abort() #6
  unreachable

5:                                                ; preds = %1
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare i64 @strlen(ptr noundef captures(none)) local_unnamed_addr #1

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

; Function Attrs: nofree nounwind uwtable
define dso_local void @quux(ptr noundef readonly captures(none) %0) local_unnamed_addr #0 {
  br label %5

2:                                                ; preds = %5
  %3 = add nuw nsw i64 %6, 1
  %4 = icmp eq i64 %3, 100
  br i1 %4, label %11, label %5, !llvm.loop !6

5:                                                ; preds = %1, %2
  %6 = phi i64 [ 0, %1 ], [ %3, %2 ]
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 %6
  %8 = load i8, ptr %7, align 1, !tbaa !8
  %9 = icmp eq i8 %8, 120
  br i1 %9, label %2, label %10

10:                                               ; preds = %5
  tail call void @abort() #6
  unreachable

11:                                               ; preds = %2
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #3

; Function Attrs: nofree nounwind uwtable
define dso_local void @qux() local_unnamed_addr #0 {
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @bar() local_unnamed_addr #0 {
  %1 = alloca [100 x i8], align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #5
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(9) %1, ptr noundef nonnull align 1 dereferenceable(9) @bar.u, i64 9, i1 false)
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 9
  store i8 0, ptr %2, align 1, !tbaa !8
  %3 = call i64 @strlen(ptr noundef nonnull readonly dereferenceable(1) %1) #5
  %4 = icmp eq i64 %3, 9
  br i1 %4, label %6, label %5

5:                                                ; preds = %0
  tail call void @abort() #6
  unreachable

6:                                                ; preds = %0
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #5
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #4

; Function Attrs: nofree nounwind uwtable
define dso_local void @baz() local_unnamed_addr #0 {
  %1 = alloca [100 x i8], align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #5
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(9) %1, ptr noundef nonnull align 1 dereferenceable(9) @baz.u, i64 9, i1 false)
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 9
  store i8 0, ptr %2, align 1, !tbaa !8
  %3 = call i64 @strlen(ptr noundef nonnull readonly dereferenceable(1) %1) #5
  %4 = icmp eq i64 %3, 9
  br i1 %4, label %6, label %5

5:                                                ; preds = %0
  tail call void @abort() #6
  unreachable

6:                                                ; preds = %0
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #5
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = alloca [100 x i8], align 1
  %2 = alloca [100 x i8], align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #5
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(9) %2, ptr noundef nonnull align 1 dereferenceable(9) @bar.u, i64 9, i1 false)
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 9
  store i8 0, ptr %3, align 1, !tbaa !8
  %4 = call i64 @strlen(ptr noundef nonnull readonly dereferenceable(1) %2) #5
  %5 = icmp eq i64 %4, 9
  br i1 %5, label %7, label %6

6:                                                ; preds = %0
  tail call void @abort() #6
  unreachable

7:                                                ; preds = %0
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #5
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #5
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(9) %1, ptr noundef nonnull align 1 dereferenceable(9) @baz.u, i64 9, i1 false)
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 9
  store i8 0, ptr %8, align 1, !tbaa !8
  %9 = call i64 @strlen(ptr noundef nonnull readonly dereferenceable(1) %1) #5
  %10 = icmp eq i64 %9, 9
  br i1 %10, label %12, label %11

11:                                               ; preds = %7
  tail call void @abort() #6
  unreachable

12:                                               ; preds = %7
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #5
  ret i32 0
}

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { nounwind }
attributes #6 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = !{!9, !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
