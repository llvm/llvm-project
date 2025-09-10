; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr82954.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr82954.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@__const.main.p = private unnamed_addr constant [4 x i32] [i32 16, i32 32, i32 64, i32 128], align 4
@__const.main.q = private unnamed_addr constant [4 x i32] [i32 8, i32 4, i32 2, i32 1], align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @foo(ptr noalias noundef captures(none) %0, ptr noalias noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %4 = load <4 x i32>, ptr %0, align 4, !tbaa !6
  %5 = load <2 x i32>, ptr %3, align 4, !tbaa !6
  %6 = shufflevector <2 x i32> %5, <2 x i32> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %7 = shufflevector <4 x i32> <i32 1, i32 2, i32 poison, i32 poison>, <4 x i32> %6, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %8 = xor <4 x i32> %4, %7
  store <4 x i32> %8, ptr %0, align 4, !tbaa !6
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = alloca [4 x i32], align 16
  %2 = alloca [4 x i32], align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(16) %1, ptr noundef nonnull align 4 dereferenceable(16) @__const.main.p, i64 16, i1 false)
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(16) %2, ptr noundef nonnull align 4 dereferenceable(16) @__const.main.q, i64 16, i1 false)
  call void asm sideeffect "", "imr,imr,~{memory}"(ptr nonnull %1, ptr nonnull %2) #6, !srcloc !10
  call void @llvm.experimental.noalias.scope.decl(metadata !11)
  call void @llvm.experimental.noalias.scope.decl(metadata !14)
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %4 = load <4 x i32>, ptr %1, align 16, !tbaa !6, !alias.scope !11, !noalias !14
  %5 = freeze <4 x i32> %4
  %6 = load <2 x i32>, ptr %3, align 4, !tbaa !6, !alias.scope !14, !noalias !11
  %7 = shufflevector <2 x i32> %6, <2 x i32> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %8 = freeze <4 x i32> %7
  %9 = shufflevector <4 x i32> <i32 1, i32 2, i32 poison, i32 poison>, <4 x i32> %8, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %10 = xor <4 x i32> %5, %9
  store <4 x i32> %10, ptr %1, align 16, !tbaa !6, !alias.scope !11, !noalias !14
  %11 = shufflevector <4 x i32> %5, <4 x i32> %10, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %12 = icmp ne <4 x i32> %11, <i32 16, i32 32, i32 66, i32 129>
  %13 = bitcast <4 x i1> %12 to i4
  %14 = icmp eq i4 %13, 0
  br i1 %14, label %16, label %15

15:                                               ; preds = %0
  call void @abort() #7
  unreachable

16:                                               ; preds = %0
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #6
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #2

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #3

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #5

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #6 = { nounwind }
attributes #7 = { noreturn nounwind }

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
!10 = !{i64 287}
!11 = !{!12}
!12 = distinct !{!12, !13, !"foo: argument 0"}
!13 = distinct !{!13, !"foo"}
!14 = !{!15}
!15 = distinct !{!15, !13, !"foo: argument 1"}
