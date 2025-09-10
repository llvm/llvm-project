; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr67226.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr67226.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.assembly_operand = type { i32, i32, i32, i32, i32 }

@from_input = dso_local local_unnamed_addr global %struct.assembly_operand zeroinitializer, align 16
@to_input = dso_local local_unnamed_addr global %struct.assembly_operand zeroinitializer, align 16

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @assemblez_1(i32 %0, ptr dead_on_return noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load i32, ptr %1, align 4, !tbaa !6
  %4 = load i32, ptr @from_input, align 4, !tbaa !6
  %5 = icmp eq i32 %3, %4
  br i1 %5, label %7, label %6

6:                                                ; preds = %2
  tail call void @abort() #5
  unreachable

7:                                                ; preds = %2
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @t0(ptr dead_on_return noundef readonly captures(none) %0, ptr dead_on_return noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = alloca %struct.assembly_operand, align 4
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 4
  %5 = load i32, ptr %4, align 4, !tbaa !11
  %6 = icmp eq i32 %5, 0
  br i1 %6, label %7, label %8

7:                                                ; preds = %2
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(20) %3, ptr noundef nonnull align 4 dereferenceable(20) %1, i64 20, i1 false), !tbaa.struct !12
  call void @assemblez_1(i32 poison, ptr dead_on_return noundef nonnull %3)
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #6
  ret void

8:                                                ; preds = %2
  tail call void @abort() #5
  unreachable
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #2

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #2

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #4 {
  %1 = alloca %struct.assembly_operand, align 4
  %2 = alloca %struct.assembly_operand, align 4
  store <4 x i32> <i32 1, i32 0, i32 2, i32 3>, ptr @to_input, align 16, !tbaa !13
  store i32 4, ptr getelementptr inbounds nuw (i8, ptr @to_input, i64 16), align 16, !tbaa !14
  store <4 x i32> <i32 6, i32 5, i32 7, i32 8>, ptr @from_input, align 16, !tbaa !13
  store i32 9, ptr getelementptr inbounds nuw (i8, ptr @from_input, i64 16), align 16, !tbaa !14
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(20) %1, ptr noundef nonnull align 16 dereferenceable(20) @to_input, i64 20, i1 false), !tbaa.struct !12
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(20) %2, ptr noundef nonnull align 16 dereferenceable(20) @from_input, i64 20, i1 false), !tbaa.struct !12
  call void @t0(ptr dead_on_return noundef nonnull %1, ptr dead_on_return noundef nonnull %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #6
  ret i32 0
}

attributes #0 = { nofree noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { noreturn nounwind }
attributes #6 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !8, i64 0}
!7 = !{!"assembly_operand", !8, i64 0, !8, i64 4, !8, i64 8, !8, i64 12, !8, i64 16}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!7, !8, i64 4}
!12 = !{i64 0, i64 4, !13, i64 4, i64 4, !13, i64 8, i64 4, !13, i64 12, i64 4, !13, i64 16, i64 4, !13}
!13 = !{!8, !8, i64 0}
!14 = !{!7, !8, i64 16}
