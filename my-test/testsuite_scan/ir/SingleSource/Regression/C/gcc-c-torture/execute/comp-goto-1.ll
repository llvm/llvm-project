; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/comp-goto-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/comp-goto-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%union.insn_t = type { %struct.anon }
%struct.anon = type { i64 }
%struct.tlb_entry_t = type { i32, i64 }
%struct.environment_t = type { ptr, [256 x i32], ptr, [256 x %struct.tlb_entry_t] }

@simulator_kernel.op_map = internal unnamed_addr constant [2 x ptr] [ptr blockaddress(@simulator_kernel, %72), ptr blockaddress(@simulator_kernel, %105)], align 8
@program = dso_local global [3 x %union.insn_t] zeroinitializer, align 16

; Function Attrs: cold nofree noreturn nounwind uwtable
define dso_local noundef i64 @f() local_unnamed_addr #0 {
  tail call void @abort() #7
  unreachable
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local i32 @simulator_kernel(i32 noundef %0, ptr noundef captures(none) %1) local_unnamed_addr #2 {
  %3 = load ptr, ptr %1, align 8, !tbaa !6
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 1040
  %6 = icmp eq i32 %0, 0
  br i1 %6, label %68, label %7

7:                                                ; preds = %2
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 1032
  %9 = load ptr, ptr %8, align 8, !tbaa !11
  %10 = icmp sgt i32 %0, 0
  br i1 %10, label %11, label %68

11:                                               ; preds = %7
  %12 = zext nneg i32 %0 to i64
  %13 = icmp eq i32 %0, 1
  br i1 %13, label %49, label %14

14:                                               ; preds = %11
  %15 = and i64 %12, 2147483646
  br label %16

16:                                               ; preds = %16, %14
  %17 = phi i64 [ 0, %14 ], [ %45, %16 ]
  %18 = getelementptr inbounds nuw %union.insn_t, ptr %9, i64 %17
  %19 = getelementptr inbounds nuw %union.insn_t, ptr %9, i64 %17
  %20 = getelementptr inbounds nuw i8, ptr %19, i64 8
  %21 = load i64, ptr %18, align 8
  %22 = load i64, ptr %20, align 8
  %23 = shl i64 %21, 46
  %24 = shl i64 %22, 46
  %25 = ashr exact i64 %23, 43
  %26 = ashr exact i64 %24, 43
  %27 = getelementptr inbounds i8, ptr @simulator_kernel.op_map, i64 %25
  %28 = getelementptr inbounds i8, ptr @simulator_kernel.op_map, i64 %26
  %29 = load ptr, ptr %27, align 8, !tbaa !12
  %30 = load ptr, ptr %28, align 8, !tbaa !12
  %31 = ptrtoint ptr %29 to i64
  %32 = ptrtoint ptr %30 to i64
  %33 = trunc i64 %31 to i32
  %34 = trunc i64 %32 to i32
  %35 = sub i32 %33, ptrtoint (ptr blockaddress(@simulator_kernel, %69) to i32)
  %36 = sub i32 %34, ptrtoint (ptr blockaddress(@simulator_kernel, %69) to i32)
  %37 = and i32 %35, 262143
  %38 = and i32 %36, 262143
  %39 = zext nneg i32 %37 to i64
  %40 = zext nneg i32 %38 to i64
  %41 = and i64 %21, -262144
  %42 = and i64 %22, -262144
  %43 = or disjoint i64 %41, %39
  %44 = or disjoint i64 %42, %40
  store i64 %43, ptr %18, align 8
  store i64 %44, ptr %20, align 8
  %45 = add nuw i64 %17, 2
  %46 = icmp eq i64 %45, %15
  br i1 %46, label %47, label %16, !llvm.loop !13

47:                                               ; preds = %16
  %48 = icmp eq i64 %15, %12
  br i1 %48, label %68, label %49

49:                                               ; preds = %11, %47
  %50 = phi i64 [ 0, %11 ], [ %15, %47 ]
  br label %51

51:                                               ; preds = %49, %51
  %52 = phi i64 [ %66, %51 ], [ %50, %49 ]
  %53 = getelementptr inbounds nuw %union.insn_t, ptr %9, i64 %52
  %54 = load i64, ptr %53, align 8
  %55 = shl i64 %54, 46
  %56 = ashr exact i64 %55, 43
  %57 = getelementptr inbounds i8, ptr @simulator_kernel.op_map, i64 %56
  %58 = load ptr, ptr %57, align 8, !tbaa !12
  %59 = ptrtoint ptr %58 to i64
  %60 = trunc i64 %59 to i32
  %61 = sub i32 %60, ptrtoint (ptr blockaddress(@simulator_kernel, %69) to i32)
  %62 = and i32 %61, 262143
  %63 = zext nneg i32 %62 to i64
  %64 = and i64 %54, -262144
  %65 = or disjoint i64 %64, %63
  store i64 %65, ptr %53, align 8
  %66 = add nuw nsw i64 %52, 1
  %67 = icmp eq i64 %66, %12
  br i1 %67, label %68, label %51, !llvm.loop !17

68:                                               ; preds = %51, %47, %7, %2
  br label %69

69:                                               ; preds = %68, %109
  %70 = phi ptr [ %115, %109 ], [ %3, %68 ]
  %71 = load i64, ptr %70, align 8, !tbaa !18
  br label %109

72:                                               ; preds = %109
  %73 = lshr i32 %122, 12
  %74 = load i64, ptr %115, align 8, !tbaa !18
  %75 = and i32 %73, 255
  %76 = zext nneg i32 %75 to i64
  %77 = getelementptr inbounds nuw %struct.tlb_entry_t, ptr %5, i64 %76
  %78 = load i32, ptr %77, align 8, !tbaa !19
  %79 = icmp eq i32 %78, %73
  br i1 %79, label %87, label %99

80:                                               ; preds = %99
  %81 = add nuw nsw i32 %101, 255
  %82 = and i32 %81, 255
  %83 = zext nneg i32 %82 to i64
  %84 = getelementptr inbounds nuw %struct.tlb_entry_t, ptr %5, i64 %83
  %85 = load i32, ptr %84, align 8, !tbaa !19
  %86 = icmp eq i32 %85, %73
  br i1 %86, label %87, label %99

87:                                               ; preds = %80, %72
  %88 = phi i64 [ %76, %72 ], [ %83, %80 ]
  %89 = shl nuw nsw i64 %88, 4
  %90 = getelementptr inbounds nuw i8, ptr %5, i64 %89
  %91 = getelementptr inbounds nuw i8, ptr %90, i64 8
  %92 = load i64, ptr %91, align 8, !tbaa !23
  %93 = zext i32 %122 to i64
  %94 = add i64 %92, %93
  %95 = inttoptr i64 %94 to ptr
  %96 = load i32, ptr %95, align 4, !tbaa !24
  %97 = zext nneg i32 %118 to i64
  %98 = getelementptr inbounds nuw i8, ptr %4, i64 %97
  store i32 %96, ptr %98, align 4, !tbaa !24
  br label %109

99:                                               ; preds = %72, %80
  %100 = phi i32 [ %85, %80 ], [ %78, %72 ]
  %101 = phi i32 [ %82, %80 ], [ %75, %72 ]
  %102 = icmp slt i32 %100, 0
  br i1 %102, label %103, label %80

103:                                              ; preds = %99
  %104 = tail call i64 @f()
  unreachable

105:                                              ; preds = %109
  %106 = zext nneg i32 %118 to i64
  %107 = getelementptr inbounds nuw i8, ptr %4, i64 %106
  %108 = load i32, ptr %107, align 4, !tbaa !24
  ret i32 %108

109:                                              ; preds = %87, %69
  %110 = phi i64 [ %71, %69 ], [ %74, %87 ]
  %111 = phi ptr [ %70, %69 ], [ %115, %87 ]
  %112 = shl i64 %110, 46
  %113 = ashr exact i64 %112, 46
  %114 = getelementptr inbounds i8, ptr blockaddress(@simulator_kernel, %69), i64 %113
  %115 = getelementptr inbounds nuw i8, ptr %111, i64 8
  %116 = trunc i64 %110 to i32
  %117 = lshr i32 %116, 20
  %118 = and i32 %117, 1020
  %119 = lshr i64 %110, 52
  %120 = and i64 %119, 1020
  %121 = getelementptr inbounds nuw i8, ptr %4, i64 %120
  %122 = load i32, ptr %121, align 4, !tbaa !24
  indirectbr ptr %114, [label %69, label %72, label %105]
}

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #3 {
  %1 = alloca %struct.environment_t, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #8
  %2 = tail call noalias dereferenceable_or_null(8192) ptr @malloc(i64 noundef 8192) #9
  %3 = ptrtoint ptr %2 to i64
  %4 = add i64 %3, 4095
  %5 = and i64 %4, -4096
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 1600
  store i32 291, ptr %6, align 8, !tbaa !19
  %7 = add i64 %5, -1191936
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 1608
  store i64 %7, ptr %8, align 8, !tbaa !23
  %9 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store i32 0, ptr %9, align 8, !tbaa !24
  %10 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store i32 1193040, ptr %10, align 8, !tbaa !24
  %11 = or disjoint i64 %5, 1104
  %12 = inttoptr i64 %11 to ptr
  store i32 88, ptr %12, align 16, !tbaa !24
  store <2 x i64> splat (i64 36028797018963968), ptr @program, align 16, !tbaa !18
  store i64 36028797018963969, ptr getelementptr inbounds nuw (i8, ptr @program, i64 16), align 16, !tbaa !18
  store ptr @program, ptr %1, align 8, !tbaa !6
  %13 = getelementptr inbounds nuw i8, ptr %1, i64 1032
  store ptr @program, ptr %13, align 8, !tbaa !11
  %14 = call i32 @simulator_kernel(i32 noundef 3, ptr noundef nonnull %1)
  %15 = icmp eq i32 %14, 88
  br i1 %15, label %17, label %16

16:                                               ; preds = %0
  tail call void @abort() #7
  unreachable

17:                                               ; preds = %0
  tail call void @exit(i32 noundef 0) #10
  unreachable
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #4

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr #5

; Function Attrs: nofree noreturn nounwind
declare void @exit(i32 noundef) local_unnamed_addr #6

attributes #0 = { cold nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { cold noreturn nounwind }
attributes #8 = { nounwind }
attributes #9 = { nounwind allocsize(0) }
attributes #10 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !8, i64 0}
!7 = !{!"", !8, i64 0, !9, i64 8, !8, i64 1032, !9, i64 1040}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!7, !8, i64 1032}
!12 = !{!8, !8, i64 0}
!13 = distinct !{!13, !14, !15, !16}
!14 = !{!"llvm.loop.mustprogress"}
!15 = !{!"llvm.loop.isvectorized", i32 1}
!16 = !{!"llvm.loop.unroll.runtime.disable"}
!17 = distinct !{!17, !14, !15}
!18 = !{!9, !9, i64 0}
!19 = !{!20, !21, i64 0}
!20 = !{!"", !21, i64 0, !22, i64 8}
!21 = !{!"int", !9, i64 0}
!22 = !{!"long", !9, i64 0}
!23 = !{!20, !22, i64 8}
!24 = !{!21, !21, i64 0}
