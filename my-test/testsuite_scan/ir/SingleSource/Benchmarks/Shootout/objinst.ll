; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout/objinst.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout/objinst.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [5 x i8] c"true\00", align 1
@.str.1 = private unnamed_addr constant [6 x i8] c"false\00", align 1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local i8 @toggle_value(ptr noundef readonly captures(none) %0) #0 {
  %2 = load i8, ptr %0, align 8, !tbaa !6
  ret i8 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local noundef ptr @toggle_activate(ptr noundef returned captures(ret: address, provenance) %0) #1 {
  %2 = load i8, ptr %0, align 8, !tbaa !6
  %3 = icmp eq i8 %2, 0
  %4 = zext i1 %3 to i8
  store i8 %4, ptr %0, align 8, !tbaa !6
  ret ptr %0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local noundef ptr @init_Toggle(ptr noundef returned writeonly captures(ret: address, provenance) initializes((0, 1), (8, 24)) %0, i8 noundef %1) local_unnamed_addr #2 {
  store i8 %1, ptr %0, align 8, !tbaa !6
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store ptr @toggle_value, ptr %3, align 8, !tbaa !11
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store ptr @toggle_activate, ptr %4, align 8, !tbaa !12
  ret ptr %0
}

; Function Attrs: mustprogress nofree nounwind willreturn memory(write, argmem: none, inaccessiblemem: readwrite) uwtable
define dso_local noalias noundef ptr @new_Toggle(i8 noundef %0) local_unnamed_addr #3 {
  %2 = tail call noalias dereferenceable_or_null(24) ptr @malloc(i64 noundef 24) #10
  store i8 %0, ptr %2, align 8, !tbaa !6
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 8
  store ptr @toggle_value, ptr %3, align 8, !tbaa !11
  %4 = getelementptr inbounds nuw i8, ptr %2, i64 16
  store ptr @toggle_activate, ptr %4, align 8, !tbaa !12
  ret ptr %2
}

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr #4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local noundef ptr @nth_toggle_activate(ptr noundef returned captures(ret: address, provenance) %0) #1 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 28
  %3 = load i32, ptr %2, align 4, !tbaa !13
  %4 = add nsw i32 %3, 1
  store i32 %4, ptr %2, align 4, !tbaa !13
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %6 = load i32, ptr %5, align 8, !tbaa !16
  %7 = icmp slt i32 %4, %6
  br i1 %7, label %12, label %8

8:                                                ; preds = %1
  %9 = load i8, ptr %0, align 8, !tbaa !17
  %10 = icmp eq i8 %9, 0
  %11 = zext i1 %10 to i8
  store i8 %11, ptr %0, align 8, !tbaa !17
  store i32 0, ptr %2, align 4, !tbaa !13
  br label %12

12:                                               ; preds = %8, %1
  ret ptr %0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local noundef ptr @init_NthToggle(ptr noundef returned writeonly captures(ret: address, provenance) initializes((16, 32)) %0, i32 noundef %1) local_unnamed_addr #2 {
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 24
  store i32 %1, ptr %3, align 8, !tbaa !16
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 28
  store i32 0, ptr %4, align 4, !tbaa !13
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store ptr @nth_toggle_activate, ptr %5, align 8, !tbaa !18
  ret ptr %0
}

; Function Attrs: mustprogress nofree nounwind willreturn memory(write, argmem: none, inaccessiblemem: readwrite) uwtable
define dso_local noalias noundef ptr @new_NthToggle(i8 noundef %0, i32 noundef %1) local_unnamed_addr #3 {
  %3 = tail call noalias dereferenceable_or_null(32) ptr @malloc(i64 noundef 32) #10
  store i8 %0, ptr %3, align 8, !tbaa !6
  %4 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store ptr @toggle_value, ptr %4, align 8, !tbaa !11
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %6 = getelementptr inbounds nuw i8, ptr %3, i64 24
  store i32 %1, ptr %6, align 8, !tbaa !16
  %7 = getelementptr inbounds nuw i8, ptr %3, i64 28
  store i32 0, ptr %7, align 4, !tbaa !13
  store ptr @nth_toggle_activate, ptr %5, align 8, !tbaa !18
  ret ptr %3
}

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #5 {
  %3 = icmp eq i32 %0, 2
  br i1 %3, label %4, label %8

4:                                                ; preds = %2
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %6 = load ptr, ptr %5, align 8, !tbaa !19
  %7 = tail call i64 @strtol(ptr noundef nonnull captures(none) %6, ptr noundef null, i32 noundef 10) #11
  br label %8

8:                                                ; preds = %2, %4
  %9 = tail call noalias dereferenceable_or_null(24) ptr @malloc(i64 noundef 24) #10
  %10 = getelementptr inbounds nuw i8, ptr %9, i64 8
  store ptr @toggle_value, ptr %10, align 8, !tbaa !11
  %11 = getelementptr inbounds nuw i8, ptr %9, i64 16
  store ptr @toggle_activate, ptr %11, align 8, !tbaa !12
  %12 = tail call i32 @puts(ptr noundef nonnull dereferenceable(1) @.str.1)
  %13 = tail call i32 @puts(ptr noundef nonnull dereferenceable(1) @.str)
  store i8 0, ptr %9, align 8, !tbaa !6
  %14 = tail call i8 @toggle_value(ptr noundef nonnull %9) #11
  %15 = icmp eq i8 %14, 0
  %16 = select i1 %15, ptr @.str.1, ptr @.str
  %17 = tail call i32 @puts(ptr noundef nonnull dereferenceable(1) %16)
  %18 = load ptr, ptr %11, align 8, !tbaa !12
  %19 = tail call ptr %18(ptr noundef nonnull %9) #11
  %20 = getelementptr inbounds nuw i8, ptr %19, i64 8
  %21 = load ptr, ptr %20, align 8, !tbaa !11
  %22 = tail call i8 %21(ptr noundef nonnull %9) #11
  %23 = icmp eq i8 %22, 0
  %24 = select i1 %23, ptr @.str.1, ptr @.str
  %25 = tail call i32 @puts(ptr noundef nonnull dereferenceable(1) %24)
  %26 = load ptr, ptr %11, align 8, !tbaa !12
  %27 = tail call ptr %26(ptr noundef nonnull %9) #11
  %28 = getelementptr inbounds nuw i8, ptr %27, i64 8
  %29 = load ptr, ptr %28, align 8, !tbaa !11
  %30 = tail call i8 %29(ptr noundef nonnull %9) #11
  %31 = icmp eq i8 %30, 0
  %32 = select i1 %31, ptr @.str.1, ptr @.str
  %33 = tail call i32 @puts(ptr noundef nonnull dereferenceable(1) %32)
  tail call void @free(ptr noundef nonnull %9) #11
  %34 = tail call i32 @putchar(i32 10)
  %35 = tail call noalias dereferenceable_or_null(32) ptr @malloc(i64 noundef 32) #10
  %36 = getelementptr inbounds nuw i8, ptr %35, i64 8
  store ptr @toggle_value, ptr %36, align 8, !tbaa !11
  %37 = getelementptr inbounds nuw i8, ptr %35, i64 16
  %38 = getelementptr inbounds nuw i8, ptr %35, i64 24
  store i32 3, ptr %38, align 8, !tbaa !16
  %39 = getelementptr inbounds nuw i8, ptr %35, i64 28
  store ptr @nth_toggle_activate, ptr %37, align 8, !tbaa !18
  %40 = tail call i32 @puts(ptr noundef nonnull dereferenceable(1) @.str)
  %41 = tail call i32 @puts(ptr noundef nonnull dereferenceable(1) @.str)
  store i8 0, ptr %35, align 8, !tbaa !17
  store i32 0, ptr %39, align 4, !tbaa !13
  %42 = tail call i8 @toggle_value(ptr noundef nonnull %35) #11
  %43 = icmp eq i8 %42, 0
  %44 = select i1 %43, ptr @.str.1, ptr @.str
  %45 = tail call i32 @puts(ptr noundef nonnull dereferenceable(1) %44)
  %46 = load ptr, ptr %37, align 8, !tbaa !18
  %47 = tail call ptr %46(ptr noundef nonnull %35) #11
  %48 = getelementptr inbounds nuw i8, ptr %47, i64 8
  %49 = load ptr, ptr %48, align 8, !tbaa !11
  %50 = tail call i8 %49(ptr noundef nonnull %35) #11
  %51 = icmp eq i8 %50, 0
  %52 = select i1 %51, ptr @.str.1, ptr @.str
  %53 = tail call i32 @puts(ptr noundef nonnull dereferenceable(1) %52)
  %54 = load ptr, ptr %37, align 8, !tbaa !18
  %55 = tail call ptr %54(ptr noundef nonnull %35) #11
  %56 = getelementptr inbounds nuw i8, ptr %55, i64 8
  %57 = load ptr, ptr %56, align 8, !tbaa !11
  %58 = tail call i8 %57(ptr noundef nonnull %35) #11
  %59 = icmp eq i8 %58, 0
  %60 = select i1 %59, ptr @.str.1, ptr @.str
  %61 = tail call i32 @puts(ptr noundef nonnull dereferenceable(1) %60)
  %62 = load ptr, ptr %37, align 8, !tbaa !18
  %63 = tail call ptr %62(ptr noundef nonnull %35) #11
  %64 = getelementptr inbounds nuw i8, ptr %63, i64 8
  %65 = load ptr, ptr %64, align 8, !tbaa !11
  %66 = tail call i8 %65(ptr noundef nonnull %35) #11
  %67 = icmp eq i8 %66, 0
  %68 = select i1 %67, ptr @.str.1, ptr @.str
  %69 = tail call i32 @puts(ptr noundef nonnull dereferenceable(1) %68)
  %70 = load ptr, ptr %37, align 8, !tbaa !18
  %71 = tail call ptr %70(ptr noundef nonnull %35) #11
  %72 = getelementptr inbounds nuw i8, ptr %71, i64 8
  %73 = load ptr, ptr %72, align 8, !tbaa !11
  %74 = tail call i8 %73(ptr noundef nonnull %35) #11
  %75 = icmp eq i8 %74, 0
  %76 = select i1 %75, ptr @.str.1, ptr @.str
  %77 = tail call i32 @puts(ptr noundef nonnull dereferenceable(1) %76)
  %78 = load ptr, ptr %37, align 8, !tbaa !18
  %79 = tail call ptr %78(ptr noundef nonnull %35) #11
  %80 = getelementptr inbounds nuw i8, ptr %79, i64 8
  %81 = load ptr, ptr %80, align 8, !tbaa !11
  %82 = tail call i8 %81(ptr noundef nonnull %35) #11
  %83 = icmp eq i8 %82, 0
  %84 = select i1 %83, ptr @.str.1, ptr @.str
  %85 = tail call i32 @puts(ptr noundef nonnull dereferenceable(1) %84)
  tail call void @free(ptr noundef nonnull %35) #11
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #6

; Function Attrs: mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @free(ptr allocptr noundef captures(none)) local_unnamed_addr #7

; Function Attrs: mustprogress nocallback nofree nounwind willreturn
declare i64 @strtol(ptr noundef readonly, ptr noundef captures(none), i32 noundef) local_unnamed_addr #8

; Function Attrs: nofree nounwind
declare noundef i32 @putchar(i32 noundef) local_unnamed_addr #9

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nofree nounwind willreturn memory(write, argmem: none, inaccessiblemem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { mustprogress nocallback nofree nounwind willreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { nofree nounwind }
attributes #10 = { nounwind allocsize(0) }
attributes #11 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !8, i64 0}
!7 = !{!"Toggle", !8, i64 0, !10, i64 8, !10, i64 16}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!"any pointer", !8, i64 0}
!11 = !{!7, !10, i64 8}
!12 = !{!7, !10, i64 16}
!13 = !{!14, !15, i64 28}
!14 = !{!"NthToggle", !7, i64 0, !15, i64 24, !15, i64 28}
!15 = !{!"int", !8, i64 0}
!16 = !{!14, !15, i64 24}
!17 = !{!14, !8, i64 0}
!18 = !{!14, !10, i64 16}
!19 = !{!20, !20, i64 0}
!20 = !{!"p1 omnipotent char", !10, i64 0}
