; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr53688.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr53688.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.hdr = type { [9 x i8], [8 x i8] }

@p = dso_local local_unnamed_addr global %struct.hdr zeroinitializer, align 1
@.str = private unnamed_addr constant [10 x i8] c"FOOBARFOO\00", align 1
@headline = dso_local local_unnamed_addr global [256 x i8] zeroinitializer, align 1

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @init() local_unnamed_addr #0 {
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(9) @p, ptr noundef nonnull align 1 dereferenceable(9) @.str, i64 9, i1 false)
  store i64 6147487297207357523, ptr getelementptr inbounds nuw (i8, ptr @p, i64 9), align 1
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  tail call void @init()
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(9) @headline, ptr noundef nonnull align 1 dereferenceable(9) @p, i64 9, i1 false)
  store i8 32, ptr getelementptr inbounds nuw (i8, ptr @headline, i64 9), align 1
  %1 = load i64, ptr getelementptr inbounds nuw (i8, ptr @p, i64 9), align 1
  store i64 %1, ptr getelementptr inbounds nuw (i8, ptr @headline, i64 10), align 1
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(238) getelementptr inbounds nuw (i8, ptr @headline, i64 18), i8 32, i64 238, i1 false)
  %2 = and i64 %1, 255
  %3 = icmp eq i64 %2, 83
  br i1 %3, label %5, label %4

4:                                                ; preds = %0
  tail call void @abort() #5
  unreachable

5:                                                ; preds = %0
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #3

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #4

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #4 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
