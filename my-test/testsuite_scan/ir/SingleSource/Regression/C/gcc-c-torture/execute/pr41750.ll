; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr41750.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr41750.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.foo_link_hash_table = type { %struct.bfd_link_hash_table, ptr, ptr }
%struct.bfd_link_hash_table = type { i32 }
%struct.foo_link_info = type { ptr }

@hash = dso_local global %struct.foo_link_hash_table zeroinitializer, align 8
@link_info = dso_local global %struct.foo_link_info zeroinitializer, align 8
@abfd = dso_local global i32 0, align 4

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, argmem: readwrite, inaccessiblemem: none) uwtable
define dso_local noundef i32 @foo_create_got_section(ptr noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load ptr, ptr %1, align 8, !tbaa !6
  %4 = getelementptr inbounds nuw i8, ptr %3, i64 16
  store ptr %0, ptr %4, align 8, !tbaa !12
  ret i32 1
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none) uwtable
define dso_local ptr @elf64_ia64_check_relocs(ptr noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #1 {
  %3 = load ptr, ptr %1, align 8, !tbaa !6
  %4 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %5 = load ptr, ptr %4, align 8, !tbaa !12
  %6 = icmp eq ptr %5, null
  br i1 %6, label %7, label %16

7:                                                ; preds = %2
  %8 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %9 = load ptr, ptr %8, align 8, !tbaa !17
  %10 = icmp eq ptr %9, null
  br i1 %10, label %11, label %12

11:                                               ; preds = %7
  store ptr %0, ptr %8, align 8, !tbaa !17
  br label %12

12:                                               ; preds = %11, %7
  %13 = phi ptr [ %9, %7 ], [ %0, %11 ]
  %14 = tail call i32 @foo_create_got_section(ptr noundef %13, ptr noundef nonnull readonly %1)
  %15 = load ptr, ptr %4, align 8, !tbaa !12
  br label %16

16:                                               ; preds = %2, %12
  %17 = phi ptr [ %5, %2 ], [ %15, %12 ]
  ret ptr %17
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  store ptr @hash, ptr @link_info, align 8, !tbaa !6
  %1 = tail call ptr @elf64_ia64_check_relocs(ptr noundef nonnull @abfd, ptr noundef nonnull @link_info)
  %2 = icmp eq ptr %1, @abfd
  br i1 %2, label %4, label %3

3:                                                ; preds = %0
  tail call void @abort() #4
  unreachable

4:                                                ; preds = %0
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, argmem: readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !8, i64 0}
!7 = !{!"foo_link_info", !8, i64 0}
!8 = !{!"p1 _ZTS19foo_link_hash_table", !9, i64 0}
!9 = !{!"any pointer", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
!12 = !{!13, !16, i64 16}
!13 = !{!"foo_link_hash_table", !14, i64 0, !16, i64 8, !16, i64 16}
!14 = !{!"bfd_link_hash_table", !15, i64 0}
!15 = !{!"int", !10, i64 0}
!16 = !{!"p1 int", !9, i64 0}
!17 = !{!13, !16, i64 8}
