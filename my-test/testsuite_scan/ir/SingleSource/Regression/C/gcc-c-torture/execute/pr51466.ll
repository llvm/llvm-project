; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr51466.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr51466.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: nofree noinline norecurse nounwind memory(inaccessiblemem: readwrite) uwtable
define dso_local noundef i32 @foo(i32 noundef %0) local_unnamed_addr #0 {
  %2 = alloca [4 x i32], align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #4
  %3 = sext i32 %0 to i64
  %4 = getelementptr inbounds i32, ptr %2, i64 %3
  store volatile i32 6, ptr %4, align 4, !tbaa !6
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #4
  ret i32 6
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nofree noinline norecurse nounwind memory(inaccessiblemem: readwrite) uwtable
define dso_local i32 @bar(i32 noundef %0) local_unnamed_addr #0 {
  %2 = alloca [4 x i32], align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #4
  %3 = sext i32 %0 to i64
  %4 = getelementptr inbounds i32, ptr %2, i64 %3
  store volatile i32 6, ptr %4, align 4, !tbaa !6
  store i32 8, ptr %4, align 4, !tbaa !6
  %5 = load volatile i32, ptr %4, align 4, !tbaa !6
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #4
  ret i32 %5
}

; Function Attrs: nofree noinline norecurse nounwind memory(inaccessiblemem: readwrite) uwtable
define dso_local i32 @baz(i32 noundef %0) local_unnamed_addr #0 {
  %2 = alloca [4 x i32], align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #4
  %3 = sext i32 %0 to i64
  %4 = getelementptr inbounds i32, ptr %2, i64 %3
  store volatile i32 6, ptr %4, align 4, !tbaa !6
  store i32 8, ptr %2, align 4, !tbaa !6
  %5 = load volatile i32, ptr %4, align 4, !tbaa !6
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #4
  ret i32 %5
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = tail call i32 @foo(i32 noundef 3)
  %2 = tail call i32 @bar(i32 noundef 2)
  %3 = icmp eq i32 %2, 8
  br i1 %3, label %4, label %10

4:                                                ; preds = %0
  %5 = tail call i32 @baz(i32 noundef 0)
  %6 = icmp eq i32 %5, 8
  br i1 %6, label %7, label %10

7:                                                ; preds = %4
  %8 = tail call i32 @baz(i32 noundef 1)
  %9 = icmp eq i32 %8, 6
  br i1 %9, label %11, label %10

10:                                               ; preds = %7, %4, %0
  tail call void @abort() #5
  unreachable

11:                                               ; preds = %7
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

attributes #0 = { nofree noinline norecurse nounwind memory(inaccessiblemem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nounwind }
attributes #5 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
