; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/stkalign.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/stkalign.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.anon = type { i8, [63 x i8] }
%struct.anon.0 = type { i8 }

@test.s = internal global %struct.anon zeroinitializer, align 64
@test2.s = internal global %struct.anon.0 zeroinitializer, align 1

; Function Attrs: nounwind uwtable
define dso_local i32 @test(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #2
  call void asm "", "=*imr,=*m,0,*m"(ptr nonnull elementtype(i32) %3, ptr nonnull elementtype(%struct.anon) @test.s, ptr nonnull %3, ptr nonnull elementtype(%struct.anon) @test.s) #2, !srcloc !6
  %4 = icmp eq i32 %0, 0
  br i1 %4, label %9, label %5

5:                                                ; preds = %2
  %6 = add i32 %0, -1
  %7 = load i32, ptr %3, align 4, !tbaa !7
  %8 = call i32 @test(i32 noundef %6, i32 noundef %7)
  br label %12

9:                                                ; preds = %2
  %10 = load i32, ptr %3, align 4, !tbaa !7
  %11 = xor i32 %10, %1
  br label %12

12:                                               ; preds = %9, %5
  %13 = phi i32 [ %8, %5 ], [ %11, %9 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #2
  ret i32 %13
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nounwind uwtable
define dso_local i32 @test2(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #2
  call void asm "", "=*imr,=*m,0,*m"(ptr nonnull elementtype(i32) %3, ptr nonnull elementtype(%struct.anon.0) @test2.s, ptr nonnull %3, ptr nonnull elementtype(%struct.anon.0) @test2.s) #2, !srcloc !11
  %4 = icmp eq i32 %0, 0
  br i1 %4, label %9, label %5

5:                                                ; preds = %2
  %6 = add i32 %0, -1
  %7 = load i32, ptr %3, align 4, !tbaa !7
  %8 = call i32 @test2(i32 noundef %6, i32 noundef %7)
  br label %12

9:                                                ; preds = %2
  %10 = load i32, ptr %3, align 4, !tbaa !7
  %11 = xor i32 %10, %1
  br label %12

12:                                               ; preds = %9, %5
  %13 = phi i32 [ %8, %5 ], [ %11, %9 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #2
  ret i32 %13
}

; Function Attrs: nounwind uwtable
define dso_local range(i32 0, 2) i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #0 {
  %3 = tail call i32 @test(i32 noundef %0, i32 noundef 0)
  %4 = add nsw i32 %0, 1
  %5 = tail call i32 @test(i32 noundef %4, i32 noundef 0)
  %6 = or i32 %5, %3
  %7 = add nsw i32 %0, 2
  %8 = tail call i32 @test(i32 noundef %7, i32 noundef 0)
  %9 = or i32 %6, %8
  %10 = tail call i32 @test2(i32 noundef %0, i32 noundef 0)
  %11 = tail call i32 @test2(i32 noundef %4, i32 noundef 0)
  %12 = or i32 %11, %10
  %13 = tail call i32 @test2(i32 noundef %7, i32 noundef 0)
  %14 = or i32 %12, %13
  %15 = and i32 %9, 63
  %16 = icmp eq i32 %15, 0
  %17 = and i32 %14, 63
  %18 = icmp ne i32 %17, 0
  %19 = select i1 %16, i1 %18, i1 false
  %20 = zext i1 %19 to i32
  ret i32 %20
}

attributes #0 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{i64 344}
!7 = !{!8, !8, i64 0}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{i64 557}
