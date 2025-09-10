; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/casts.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/casts.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [41 x i8] c"\0ACHAR             C = '%c' (%d)\09\09(0x%x)\0A\00", align 1
@.str.1 = private unnamed_addr constant [33 x i8] c"char to short   s1 = %d\09\09(0x%x)\0A\00", align 1
@.str.2 = private unnamed_addr constant [33 x i8] c"char to int     i1 = %d\09\09(0x%x)\0A\00", align 1
@.str.3 = private unnamed_addr constant [35 x i8] c"char to int64_t l1 = %ld\09\09(0x%lx)\0A\00", align 1
@.str.4 = private unnamed_addr constant [34 x i8] c"\0Achar to ubyte  uc1 = %u\09\09(0x%x)\0A\00", align 1
@.str.5 = private unnamed_addr constant [33 x i8] c"char to ushort us1 = %u\09\09(0x%x)\0A\00", align 1
@.str.6 = private unnamed_addr constant [33 x i8] c"char to uint   ui1 = %u\09\09(0x%x)\0A\00", align 1
@.str.7 = private unnamed_addr constant [37 x i8] c"char to uint64_t ul1 = %lu\09\09(0x%lx)\0A\00", align 1
@.str.8 = private unnamed_addr constant [35 x i8] c"\0A\0ASHORT            S = %d\09\09(0x%x)\0A\00", align 1
@.str.9 = private unnamed_addr constant [34 x i8] c"short to byte    c1 = %d\09\09(0x%x)\0A\00", align 1
@.str.10 = private unnamed_addr constant [34 x i8] c"short to int     i1 = %d\09\09(0x%x)\0A\00", align 1
@.str.11 = private unnamed_addr constant [36 x i8] c"short to int64_t l1 = %ld\09\09(0x%lx)\0A\00", align 1
@.str.12 = private unnamed_addr constant [35 x i8] c"\0Ashort to ubyte  uc1 = %u\09\09(0x%x)\0A\00", align 1
@.str.13 = private unnamed_addr constant [34 x i8] c"short to ushort us1 = %u\09\09(0x%x)\0A\00", align 1
@.str.14 = private unnamed_addr constant [34 x i8] c"short to uint   ui1 = %u\09\09(0x%x)\0A\00", align 1
@.str.15 = private unnamed_addr constant [38 x i8] c"short to uint64_t ul1 = %lu\09\09(0x%lx)\0A\00", align 1
@.str.16 = private unnamed_addr constant [36 x i8] c"\0A\0ALONG            L = %ld\09\09(0x%lx)\0A\00", align 1
@.str.17 = private unnamed_addr constant [33 x i8] c"long to byte    c1 = %d\09\09(0x%x)\0A\00", align 1
@.str.18 = private unnamed_addr constant [33 x i8] c"long to short   s1 = %d\09\09(0x%x)\0A\00", align 1
@.str.19 = private unnamed_addr constant [33 x i8] c"long to int     i1 = %d\09\09(0x%x)\0A\00", align 1
@.str.20 = private unnamed_addr constant [34 x i8] c"\0Along to ubyte  uc1 = %u\09\09(0x%x)\0A\00", align 1
@.str.21 = private unnamed_addr constant [33 x i8] c"long to ushort us1 = %u\09\09(0x%x)\0A\00", align 1
@.str.22 = private unnamed_addr constant [33 x i8] c"long to uint   ui1 = %u\09\09(0x%x)\0A\00", align 1
@.str.23 = private unnamed_addr constant [37 x i8] c"long to uint64_t ul1 = %lu\09\09(0x%lx)\0A\00", align 1
@.str.24 = private unnamed_addr constant [27 x i8] c"\0A\0AFLOAT            F = %f\0A\00", align 1
@.str.25 = private unnamed_addr constant [34 x i8] c"float to short   s1 = %d\09\09(0x%x)\0A\00", align 1
@.str.26 = private unnamed_addr constant [34 x i8] c"float to int     i1 = %d\09\09(0x%x)\0A\00", align 1
@.str.27 = private unnamed_addr constant [34 x i8] c"float to ushort us1 = %u\09\09(0x%x)\0A\00", align 1
@.str.28 = private unnamed_addr constant [34 x i8] c"float to uint   ui1 = %u\09\09(0x%x)\0A\00", align 1
@.str.29 = private unnamed_addr constant [38 x i8] c"float to uint64_t ul1 = %lu\09\09(0x%lx)\0A\00", align 1
@.str.30 = private unnamed_addr constant [28 x i8] c"\0A\0ADOUBLE            D = %f\0A\00", align 1
@.str.31 = private unnamed_addr constant [35 x i8] c"double to short   s1 = %d\09\09(0x%x)\0A\00", align 1
@.str.32 = private unnamed_addr constant [35 x i8] c"double to int     i1 = %d\09\09(0x%x)\0A\00", align 1
@.str.33 = private unnamed_addr constant [37 x i8] c"double to int64_t l1 = %ld\09\09(0x%lx)\0A\00", align 1
@.str.34 = private unnamed_addr constant [35 x i8] c"double to ushort us1 = %u\09\09(0x%x)\0A\00", align 1
@.str.35 = private unnamed_addr constant [35 x i8] c"double to uint   ui1 = %u\09\09(0x%x)\0A\00", align 1
@.str.36 = private unnamed_addr constant [39 x i8] c"double to uint64_t ul1 = %lu\09\09(0x%lx)\0A\00", align 1
@.str.37 = private unnamed_addr constant [28 x i8] c"double <- int64_t %ld = %f\0A\00", align 1
@.str.38 = private unnamed_addr constant [29 x i8] c"double <- uint64_t %lu = %f\0A\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = icmp sgt i32 %0, 1
  br i1 %3, label %4, label %22

4:                                                ; preds = %2
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %6 = load ptr, ptr %5, align 8, !tbaa !6
  %7 = tail call i64 @strtol(ptr noundef nonnull captures(none) %6, ptr noundef null, i32 noundef 10) #3
  %8 = trunc i64 %7 to i8
  %9 = icmp eq i32 %0, 2
  br i1 %9, label %22, label %10

10:                                               ; preds = %4
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %12 = load ptr, ptr %11, align 8, !tbaa !6
  %13 = tail call i64 @strtol(ptr noundef nonnull captures(none) %12, ptr noundef null, i32 noundef 10) #3
  %14 = trunc i64 %13 to i16
  %15 = icmp samesign ugt i32 %0, 3
  br i1 %15, label %16, label %22

16:                                               ; preds = %10
  %17 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %18 = load ptr, ptr %17, align 8, !tbaa !6
  %19 = tail call i64 @strtol(ptr noundef nonnull captures(none) %18, ptr noundef null, i32 noundef 10) #3
  %20 = shl i64 %19, 32
  %21 = ashr exact i64 %20, 32
  br label %22

22:                                               ; preds = %2, %4, %10, %16
  %23 = phi i1 [ true, %16 ], [ false, %10 ], [ false, %4 ], [ false, %2 ]
  %24 = phi i16 [ %14, %16 ], [ %14, %10 ], [ -769, %4 ], [ -769, %2 ]
  %25 = phi i8 [ %8, %16 ], [ %8, %10 ], [ %8, %4 ], [ 100, %2 ]
  %26 = phi i64 [ %21, %16 ], [ 179923220407203, %10 ], [ 179923220407203, %4 ], [ 179923220407203, %2 ]
  %27 = sext i8 %25 to i32
  %28 = sext i8 %25 to i64
  %29 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %27, i32 noundef %27, i32 noundef %27)
  %30 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %27, i32 noundef %27)
  %31 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %27, i32 noundef %27)
  %32 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i64 noundef %28, i64 noundef %28)
  %33 = zext i8 %25 to i32
  %34 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef %33, i32 noundef %33)
  %35 = and i32 %27, 65535
  %36 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, i32 noundef %35, i32 noundef %35)
  %37 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef %27, i32 noundef %27)
  %38 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i64 noundef %28, i64 noundef %28)
  %39 = zext i16 %24 to i32
  %40 = sext i16 %24 to i32
  %41 = sext i16 %24 to i64
  %42 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.8, i32 noundef %40, i32 noundef %40)
  %43 = shl i32 %39, 24
  %44 = ashr exact i32 %43, 24
  %45 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.9, i32 noundef %44, i32 noundef %44)
  %46 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.10, i32 noundef %40, i32 noundef %40)
  %47 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.11, i64 noundef %41, i64 noundef %41)
  %48 = and i16 %24, 255
  %49 = zext nneg i16 %48 to i32
  %50 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.12, i32 noundef %49, i32 noundef %49)
  %51 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.13, i32 noundef %39, i32 noundef %39)
  %52 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.14, i32 noundef %40, i32 noundef %40)
  %53 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.15, i64 noundef %41, i64 noundef %41)
  %54 = trunc i64 %26 to i32
  %55 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.16, i64 noundef %26, i64 noundef %26)
  %56 = shl i32 %54, 24
  %57 = ashr exact i32 %56, 24
  %58 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.17, i32 noundef %57, i32 noundef %57)
  %59 = shl i32 %54, 16
  %60 = ashr exact i32 %59, 16
  %61 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.18, i32 noundef %60, i32 noundef %60)
  %62 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.19, i32 noundef %54, i32 noundef %54)
  %63 = and i32 %54, 255
  %64 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.20, i32 noundef %63, i32 noundef %63)
  %65 = and i32 %54, 65535
  %66 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.21, i32 noundef %65, i32 noundef %65)
  %67 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.22, i32 noundef %54, i32 noundef %54)
  %68 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23, i64 noundef %26, i64 noundef %26)
  br i1 %23, label %69, label %74

69:                                               ; preds = %22
  %70 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %71 = load ptr, ptr %70, align 8, !tbaa !6
  %72 = tail call double @strtod(ptr noundef nonnull captures(none) %71, ptr noundef null) #3
  %73 = fptrunc double %72 to float
  br label %74

74:                                               ; preds = %22, %69
  %75 = phi float [ %73, %69 ], [ 1.000000e+00, %22 ]
  %76 = icmp sgt i32 %0, 4
  br i1 %76, label %77, label %81

77:                                               ; preds = %74
  %78 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %79 = load ptr, ptr %78, align 8, !tbaa !6
  %80 = tail call double @strtod(ptr noundef nonnull captures(none) %79, ptr noundef null) #3
  br label %81

81:                                               ; preds = %74, %77
  %82 = phi double [ %80, %77 ], [ 2.000000e+00, %74 ]
  %83 = fptoui float %75 to i16
  %84 = fptoui float %75 to i32
  %85 = fptoui float %75 to i64
  %86 = fptosi float %75 to i16
  %87 = fptosi float %75 to i32
  %88 = fpext float %75 to double
  %89 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.24, double noundef %88)
  %90 = sext i16 %86 to i32
  %91 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.25, i32 noundef %90, i32 noundef %90)
  %92 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.26, i32 noundef %87, i32 noundef %87)
  %93 = zext i16 %83 to i32
  %94 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.27, i32 noundef %93, i32 noundef %93)
  %95 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.28, i32 noundef %84, i32 noundef %84)
  %96 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.29, i64 noundef %85, i64 noundef %85)
  %97 = fptoui double %82 to i16
  %98 = fptoui double %82 to i32
  %99 = fptoui double %82 to i64
  %100 = fptosi double %82 to i16
  %101 = fptosi double %82 to i32
  %102 = fptosi double %82 to i64
  %103 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.30, double noundef %82)
  %104 = sext i16 %100 to i32
  %105 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.31, i32 noundef %104, i32 noundef %104)
  %106 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.32, i32 noundef %101, i32 noundef %101)
  %107 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.33, i64 noundef %102, i64 noundef %102)
  %108 = zext i16 %97 to i32
  %109 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.34, i32 noundef %108, i32 noundef %108)
  %110 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.35, i32 noundef %98, i32 noundef %98)
  %111 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.36, i64 noundef %99, i64 noundef %99)
  %112 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.37, i64 noundef 123, double noundef 1.230000e+02)
  %113 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.38, i64 noundef 123, double noundef 1.230000e+02)
  %114 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.37, i64 noundef -1, double noundef -1.000000e+00)
  %115 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.38, i64 noundef -1, double noundef 0x43F0000000000000)
  %116 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.37, i64 noundef -14, double noundef -1.400000e+01)
  %117 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.38, i64 noundef -14, double noundef 0x43F0000000000000)
  %118 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.37, i64 noundef 14, double noundef 1.400000e+01)
  %119 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.38, i64 noundef 14, double noundef 1.400000e+01)
  %120 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.37, i64 noundef -9223372036854775808, double noundef 0xC3E0000000000000)
  %121 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.38, i64 noundef -9223372036854775808, double noundef 0x43E0000000000000)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn
declare i64 @strtol(ptr noundef readonly, ptr noundef captures(none), i32 noundef) local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nounwind willreturn
declare double @strtod(ptr noundef readonly, ptr noundef captures(none)) local_unnamed_addr #2

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nounwind }

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
