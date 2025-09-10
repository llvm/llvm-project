; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/memops-asm-lib.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/memops-asm-lib.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@inside_main = external local_unnamed_addr global i32, align 4
@llvm.compiler.used = appending global [5 x ptr] [ptr @my_bcopy, ptr @my_bzero, ptr @my_memcpy, ptr @my_memmove, ptr @my_memset], section "llvm.metadata"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local noundef ptr @my_memcpy(ptr noundef returned writeonly captures(ret: address, provenance) %0, ptr noundef readonly captures(none) %1, i64 noundef %2) #0 {
  %4 = icmp eq i64 %2, 0
  br i1 %4, label %62, label %5

5:                                                ; preds = %3
  %6 = ptrtoint ptr %0 to i64
  %7 = ptrtoint ptr %1 to i64
  %8 = icmp ult i64 %2, 8
  %9 = sub i64 %6, %7
  %10 = icmp ult i64 %9, 32
  %11 = or i1 %8, %10
  br i1 %11, label %49, label %12

12:                                               ; preds = %5
  %13 = icmp ult i64 %2, 32
  br i1 %13, label %34, label %14

14:                                               ; preds = %12
  %15 = and i64 %2, -32
  br label %16

16:                                               ; preds = %16, %14
  %17 = phi i64 [ 0, %14 ], [ %24, %16 ]
  %18 = getelementptr i8, ptr %1, i64 %17
  %19 = getelementptr i8, ptr %0, i64 %17
  %20 = getelementptr i8, ptr %18, i64 16
  %21 = load <16 x i8>, ptr %18, align 1, !tbaa !6
  %22 = load <16 x i8>, ptr %20, align 1, !tbaa !6
  %23 = getelementptr i8, ptr %19, i64 16
  store <16 x i8> %21, ptr %19, align 1, !tbaa !6
  store <16 x i8> %22, ptr %23, align 1, !tbaa !6
  %24 = add nuw i64 %17, 32
  %25 = icmp eq i64 %24, %15
  br i1 %25, label %26, label %16, !llvm.loop !9

26:                                               ; preds = %16
  %27 = icmp eq i64 %2, %15
  br i1 %27, label %62, label %28

28:                                               ; preds = %26
  %29 = getelementptr i8, ptr %1, i64 %15
  %30 = getelementptr i8, ptr %0, i64 %15
  %31 = and i64 %2, 31
  %32 = and i64 %2, 24
  %33 = icmp eq i64 %32, 0
  br i1 %33, label %49, label %34

34:                                               ; preds = %28, %12
  %35 = phi i64 [ %15, %28 ], [ 0, %12 ]
  %36 = and i64 %2, -8
  %37 = getelementptr i8, ptr %1, i64 %36
  %38 = getelementptr i8, ptr %0, i64 %36
  %39 = and i64 %2, 7
  br label %40

40:                                               ; preds = %40, %34
  %41 = phi i64 [ %35, %34 ], [ %45, %40 ]
  %42 = getelementptr i8, ptr %1, i64 %41
  %43 = getelementptr i8, ptr %0, i64 %41
  %44 = load <8 x i8>, ptr %42, align 1, !tbaa !6
  store <8 x i8> %44, ptr %43, align 1, !tbaa !6
  %45 = add nuw i64 %41, 8
  %46 = icmp eq i64 %45, %36
  br i1 %46, label %47, label %40, !llvm.loop !13

47:                                               ; preds = %40
  %48 = icmp eq i64 %2, %36
  br i1 %48, label %62, label %49

49:                                               ; preds = %28, %47, %5
  %50 = phi ptr [ %1, %5 ], [ %29, %28 ], [ %37, %47 ]
  %51 = phi ptr [ %0, %5 ], [ %30, %28 ], [ %38, %47 ]
  %52 = phi i64 [ %2, %5 ], [ %31, %28 ], [ %39, %47 ]
  br label %53

53:                                               ; preds = %49, %53
  %54 = phi ptr [ %58, %53 ], [ %50, %49 ]
  %55 = phi ptr [ %60, %53 ], [ %51, %49 ]
  %56 = phi i64 [ %57, %53 ], [ %52, %49 ]
  %57 = add i64 %56, -1
  %58 = getelementptr inbounds nuw i8, ptr %54, i64 1
  %59 = load i8, ptr %54, align 1, !tbaa !6
  %60 = getelementptr inbounds nuw i8, ptr %55, i64 1
  store i8 %59, ptr %55, align 1, !tbaa !6
  %61 = icmp eq i64 %57, 0
  br i1 %61, label %62, label %53, !llvm.loop !14

62:                                               ; preds = %53, %26, %47, %3
  ret ptr %0
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @my_bcopy(ptr noundef readonly captures(address) %0, ptr noundef writeonly captures(address) %1, i64 noundef %2) #0 {
  %4 = ptrtoint ptr %1 to i64
  %5 = ptrtoint ptr %0 to i64
  %6 = ptrtoint ptr %1 to i64
  %7 = icmp ult ptr %0, %1
  %8 = icmp eq i64 %2, 0
  br i1 %7, label %65, label %9

9:                                                ; preds = %3
  br i1 %8, label %135, label %10

10:                                               ; preds = %9
  %11 = icmp ult i64 %2, 8
  %12 = sub i64 %6, %5
  %13 = icmp ult i64 %12, 32
  %14 = or i1 %11, %13
  br i1 %14, label %52, label %15

15:                                               ; preds = %10
  %16 = icmp ult i64 %2, 32
  br i1 %16, label %37, label %17

17:                                               ; preds = %15
  %18 = and i64 %2, -32
  br label %19

19:                                               ; preds = %19, %17
  %20 = phi i64 [ 0, %17 ], [ %27, %19 ]
  %21 = getelementptr i8, ptr %0, i64 %20
  %22 = getelementptr i8, ptr %1, i64 %20
  %23 = getelementptr i8, ptr %21, i64 16
  %24 = load <16 x i8>, ptr %21, align 1, !tbaa !6
  %25 = load <16 x i8>, ptr %23, align 1, !tbaa !6
  %26 = getelementptr i8, ptr %22, i64 16
  store <16 x i8> %24, ptr %22, align 1, !tbaa !6
  store <16 x i8> %25, ptr %26, align 1, !tbaa !6
  %27 = add nuw i64 %20, 32
  %28 = icmp eq i64 %27, %18
  br i1 %28, label %29, label %19, !llvm.loop !15

29:                                               ; preds = %19
  %30 = icmp eq i64 %2, %18
  br i1 %30, label %135, label %31

31:                                               ; preds = %29
  %32 = getelementptr i8, ptr %0, i64 %18
  %33 = getelementptr i8, ptr %1, i64 %18
  %34 = and i64 %2, 31
  %35 = and i64 %2, 24
  %36 = icmp eq i64 %35, 0
  br i1 %36, label %52, label %37

37:                                               ; preds = %31, %15
  %38 = phi i64 [ %18, %31 ], [ 0, %15 ]
  %39 = and i64 %2, -8
  %40 = getelementptr i8, ptr %0, i64 %39
  %41 = getelementptr i8, ptr %1, i64 %39
  %42 = and i64 %2, 7
  br label %43

43:                                               ; preds = %43, %37
  %44 = phi i64 [ %38, %37 ], [ %48, %43 ]
  %45 = getelementptr i8, ptr %0, i64 %44
  %46 = getelementptr i8, ptr %1, i64 %44
  %47 = load <8 x i8>, ptr %45, align 1, !tbaa !6
  store <8 x i8> %47, ptr %46, align 1, !tbaa !6
  %48 = add nuw i64 %44, 8
  %49 = icmp eq i64 %48, %39
  br i1 %49, label %50, label %43, !llvm.loop !16

50:                                               ; preds = %43
  %51 = icmp eq i64 %2, %39
  br i1 %51, label %135, label %52

52:                                               ; preds = %31, %50, %10
  %53 = phi ptr [ %0, %10 ], [ %32, %31 ], [ %40, %50 ]
  %54 = phi ptr [ %1, %10 ], [ %33, %31 ], [ %41, %50 ]
  %55 = phi i64 [ %2, %10 ], [ %34, %31 ], [ %42, %50 ]
  br label %56

56:                                               ; preds = %52, %56
  %57 = phi ptr [ %61, %56 ], [ %53, %52 ]
  %58 = phi ptr [ %63, %56 ], [ %54, %52 ]
  %59 = phi i64 [ %60, %56 ], [ %55, %52 ]
  %60 = add i64 %59, -1
  %61 = getelementptr inbounds nuw i8, ptr %57, i64 1
  %62 = load i8, ptr %57, align 1, !tbaa !6
  %63 = getelementptr inbounds nuw i8, ptr %58, i64 1
  store i8 %62, ptr %58, align 1, !tbaa !6
  %64 = icmp eq i64 %60, 0
  br i1 %64, label %135, label %56, !llvm.loop !17

65:                                               ; preds = %3
  br i1 %8, label %135, label %66

66:                                               ; preds = %65
  %67 = getelementptr inbounds nuw i8, ptr %0, i64 %2
  %68 = getelementptr inbounds nuw i8, ptr %1, i64 %2
  %69 = icmp ult i64 %2, 8
  %70 = sub i64 %5, %4
  %71 = icmp ult i64 %70, 32
  %72 = or i1 %69, %71
  br i1 %72, label %122, label %73

73:                                               ; preds = %66
  %74 = icmp ult i64 %2, 32
  br i1 %74, label %101, label %75

75:                                               ; preds = %73
  %76 = and i64 %2, -32
  br label %77

77:                                               ; preds = %77, %75
  %78 = phi i64 [ 0, %75 ], [ %89, %77 ]
  %79 = sub i64 0, %78
  %80 = getelementptr i8, ptr %67, i64 %79
  %81 = sub i64 0, %78
  %82 = getelementptr i8, ptr %68, i64 %81
  %83 = getelementptr inbounds i8, ptr %80, i64 -16
  %84 = getelementptr inbounds i8, ptr %80, i64 -32
  %85 = load <16 x i8>, ptr %83, align 1, !tbaa !6
  %86 = load <16 x i8>, ptr %84, align 1, !tbaa !6
  %87 = getelementptr inbounds i8, ptr %82, i64 -16
  %88 = getelementptr inbounds i8, ptr %82, i64 -32
  store <16 x i8> %85, ptr %87, align 1, !tbaa !6
  store <16 x i8> %86, ptr %88, align 1, !tbaa !6
  %89 = add nuw i64 %78, 32
  %90 = icmp eq i64 %89, %76
  br i1 %90, label %91, label %77, !llvm.loop !18

91:                                               ; preds = %77
  %92 = icmp eq i64 %2, %76
  br i1 %92, label %135, label %93

93:                                               ; preds = %91
  %94 = sub i64 0, %76
  %95 = getelementptr i8, ptr %67, i64 %94
  %96 = sub i64 0, %76
  %97 = getelementptr i8, ptr %68, i64 %96
  %98 = and i64 %2, 31
  %99 = and i64 %2, 24
  %100 = icmp eq i64 %99, 0
  br i1 %100, label %122, label %101

101:                                              ; preds = %93, %73
  %102 = phi i64 [ %76, %93 ], [ 0, %73 ]
  %103 = and i64 %2, -8
  %104 = sub i64 0, %103
  %105 = getelementptr i8, ptr %67, i64 %104
  %106 = sub i64 0, %103
  %107 = getelementptr i8, ptr %68, i64 %106
  %108 = and i64 %2, 7
  br label %109

109:                                              ; preds = %109, %101
  %110 = phi i64 [ %102, %101 ], [ %118, %109 ]
  %111 = sub i64 0, %110
  %112 = getelementptr i8, ptr %67, i64 %111
  %113 = sub i64 0, %110
  %114 = getelementptr i8, ptr %68, i64 %113
  %115 = getelementptr inbounds i8, ptr %112, i64 -8
  %116 = load <8 x i8>, ptr %115, align 1, !tbaa !6
  %117 = getelementptr inbounds i8, ptr %114, i64 -8
  store <8 x i8> %116, ptr %117, align 1, !tbaa !6
  %118 = add nuw i64 %110, 8
  %119 = icmp eq i64 %118, %103
  br i1 %119, label %120, label %109, !llvm.loop !19

120:                                              ; preds = %109
  %121 = icmp eq i64 %2, %103
  br i1 %121, label %135, label %122

122:                                              ; preds = %93, %120, %66
  %123 = phi ptr [ %67, %66 ], [ %95, %93 ], [ %105, %120 ]
  %124 = phi ptr [ %68, %66 ], [ %97, %93 ], [ %107, %120 ]
  %125 = phi i64 [ %2, %66 ], [ %98, %93 ], [ %108, %120 ]
  br label %126

126:                                              ; preds = %122, %126
  %127 = phi ptr [ %131, %126 ], [ %123, %122 ]
  %128 = phi ptr [ %133, %126 ], [ %124, %122 ]
  %129 = phi i64 [ %130, %126 ], [ %125, %122 ]
  %130 = add i64 %129, -1
  %131 = getelementptr inbounds i8, ptr %127, i64 -1
  %132 = load i8, ptr %131, align 1, !tbaa !6
  %133 = getelementptr inbounds i8, ptr %128, i64 -1
  store i8 %132, ptr %133, align 1, !tbaa !6
  %134 = icmp eq i64 %130, 0
  br i1 %134, label %135, label %126, !llvm.loop !20

135:                                              ; preds = %56, %126, %29, %50, %91, %120, %9, %65
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local noundef ptr @my_memmove(ptr noundef returned writeonly captures(address, ret: address, provenance) %0, ptr noundef readonly captures(address) %1, i64 noundef %2) #0 {
  %4 = ptrtoint ptr %0 to i64
  %5 = ptrtoint ptr %1 to i64
  %6 = ptrtoint ptr %0 to i64
  %7 = icmp ult ptr %1, %0
  %8 = icmp eq i64 %2, 0
  br i1 %7, label %65, label %9

9:                                                ; preds = %3
  br i1 %8, label %135, label %10

10:                                               ; preds = %9
  %11 = icmp ult i64 %2, 8
  %12 = sub i64 %6, %5
  %13 = icmp ult i64 %12, 32
  %14 = or i1 %11, %13
  br i1 %14, label %52, label %15

15:                                               ; preds = %10
  %16 = icmp ult i64 %2, 32
  br i1 %16, label %37, label %17

17:                                               ; preds = %15
  %18 = and i64 %2, -32
  br label %19

19:                                               ; preds = %19, %17
  %20 = phi i64 [ 0, %17 ], [ %27, %19 ]
  %21 = getelementptr i8, ptr %1, i64 %20
  %22 = getelementptr i8, ptr %0, i64 %20
  %23 = getelementptr i8, ptr %21, i64 16
  %24 = load <16 x i8>, ptr %21, align 1, !tbaa !6
  %25 = load <16 x i8>, ptr %23, align 1, !tbaa !6
  %26 = getelementptr i8, ptr %22, i64 16
  store <16 x i8> %24, ptr %22, align 1, !tbaa !6
  store <16 x i8> %25, ptr %26, align 1, !tbaa !6
  %27 = add nuw i64 %20, 32
  %28 = icmp eq i64 %27, %18
  br i1 %28, label %29, label %19, !llvm.loop !21

29:                                               ; preds = %19
  %30 = icmp eq i64 %2, %18
  br i1 %30, label %135, label %31

31:                                               ; preds = %29
  %32 = getelementptr i8, ptr %1, i64 %18
  %33 = getelementptr i8, ptr %0, i64 %18
  %34 = and i64 %2, 31
  %35 = and i64 %2, 24
  %36 = icmp eq i64 %35, 0
  br i1 %36, label %52, label %37

37:                                               ; preds = %31, %15
  %38 = phi i64 [ %18, %31 ], [ 0, %15 ]
  %39 = and i64 %2, -8
  %40 = getelementptr i8, ptr %1, i64 %39
  %41 = getelementptr i8, ptr %0, i64 %39
  %42 = and i64 %2, 7
  br label %43

43:                                               ; preds = %43, %37
  %44 = phi i64 [ %38, %37 ], [ %48, %43 ]
  %45 = getelementptr i8, ptr %1, i64 %44
  %46 = getelementptr i8, ptr %0, i64 %44
  %47 = load <8 x i8>, ptr %45, align 1, !tbaa !6
  store <8 x i8> %47, ptr %46, align 1, !tbaa !6
  %48 = add nuw i64 %44, 8
  %49 = icmp eq i64 %48, %39
  br i1 %49, label %50, label %43, !llvm.loop !22

50:                                               ; preds = %43
  %51 = icmp eq i64 %2, %39
  br i1 %51, label %135, label %52

52:                                               ; preds = %31, %50, %10
  %53 = phi ptr [ %1, %10 ], [ %32, %31 ], [ %40, %50 ]
  %54 = phi ptr [ %0, %10 ], [ %33, %31 ], [ %41, %50 ]
  %55 = phi i64 [ %2, %10 ], [ %34, %31 ], [ %42, %50 ]
  br label %56

56:                                               ; preds = %52, %56
  %57 = phi ptr [ %61, %56 ], [ %53, %52 ]
  %58 = phi ptr [ %63, %56 ], [ %54, %52 ]
  %59 = phi i64 [ %60, %56 ], [ %55, %52 ]
  %60 = add i64 %59, -1
  %61 = getelementptr inbounds nuw i8, ptr %57, i64 1
  %62 = load i8, ptr %57, align 1, !tbaa !6
  %63 = getelementptr inbounds nuw i8, ptr %58, i64 1
  store i8 %62, ptr %58, align 1, !tbaa !6
  %64 = icmp eq i64 %60, 0
  br i1 %64, label %135, label %56, !llvm.loop !23

65:                                               ; preds = %3
  br i1 %8, label %135, label %66

66:                                               ; preds = %65
  %67 = getelementptr inbounds nuw i8, ptr %1, i64 %2
  %68 = getelementptr inbounds nuw i8, ptr %0, i64 %2
  %69 = icmp ult i64 %2, 8
  %70 = sub i64 %5, %4
  %71 = icmp ult i64 %70, 32
  %72 = or i1 %69, %71
  br i1 %72, label %122, label %73

73:                                               ; preds = %66
  %74 = icmp ult i64 %2, 32
  br i1 %74, label %101, label %75

75:                                               ; preds = %73
  %76 = and i64 %2, -32
  br label %77

77:                                               ; preds = %77, %75
  %78 = phi i64 [ 0, %75 ], [ %89, %77 ]
  %79 = sub i64 0, %78
  %80 = getelementptr i8, ptr %67, i64 %79
  %81 = sub i64 0, %78
  %82 = getelementptr i8, ptr %68, i64 %81
  %83 = getelementptr inbounds i8, ptr %80, i64 -16
  %84 = getelementptr inbounds i8, ptr %80, i64 -32
  %85 = load <16 x i8>, ptr %83, align 1, !tbaa !6
  %86 = load <16 x i8>, ptr %84, align 1, !tbaa !6
  %87 = getelementptr inbounds i8, ptr %82, i64 -16
  %88 = getelementptr inbounds i8, ptr %82, i64 -32
  store <16 x i8> %85, ptr %87, align 1, !tbaa !6
  store <16 x i8> %86, ptr %88, align 1, !tbaa !6
  %89 = add nuw i64 %78, 32
  %90 = icmp eq i64 %89, %76
  br i1 %90, label %91, label %77, !llvm.loop !24

91:                                               ; preds = %77
  %92 = icmp eq i64 %2, %76
  br i1 %92, label %135, label %93

93:                                               ; preds = %91
  %94 = sub i64 0, %76
  %95 = getelementptr i8, ptr %67, i64 %94
  %96 = sub i64 0, %76
  %97 = getelementptr i8, ptr %68, i64 %96
  %98 = and i64 %2, 31
  %99 = and i64 %2, 24
  %100 = icmp eq i64 %99, 0
  br i1 %100, label %122, label %101

101:                                              ; preds = %93, %73
  %102 = phi i64 [ %76, %93 ], [ 0, %73 ]
  %103 = and i64 %2, -8
  %104 = sub i64 0, %103
  %105 = getelementptr i8, ptr %67, i64 %104
  %106 = sub i64 0, %103
  %107 = getelementptr i8, ptr %68, i64 %106
  %108 = and i64 %2, 7
  br label %109

109:                                              ; preds = %109, %101
  %110 = phi i64 [ %102, %101 ], [ %118, %109 ]
  %111 = sub i64 0, %110
  %112 = getelementptr i8, ptr %67, i64 %111
  %113 = sub i64 0, %110
  %114 = getelementptr i8, ptr %68, i64 %113
  %115 = getelementptr inbounds i8, ptr %112, i64 -8
  %116 = load <8 x i8>, ptr %115, align 1, !tbaa !6
  %117 = getelementptr inbounds i8, ptr %114, i64 -8
  store <8 x i8> %116, ptr %117, align 1, !tbaa !6
  %118 = add nuw i64 %110, 8
  %119 = icmp eq i64 %118, %103
  br i1 %119, label %120, label %109, !llvm.loop !25

120:                                              ; preds = %109
  %121 = icmp eq i64 %2, %103
  br i1 %121, label %135, label %122

122:                                              ; preds = %93, %120, %66
  %123 = phi ptr [ %67, %66 ], [ %95, %93 ], [ %105, %120 ]
  %124 = phi ptr [ %68, %66 ], [ %97, %93 ], [ %107, %120 ]
  %125 = phi i64 [ %2, %66 ], [ %98, %93 ], [ %108, %120 ]
  br label %126

126:                                              ; preds = %122, %126
  %127 = phi ptr [ %131, %126 ], [ %123, %122 ]
  %128 = phi ptr [ %133, %126 ], [ %124, %122 ]
  %129 = phi i64 [ %130, %126 ], [ %125, %122 ]
  %130 = add i64 %129, -1
  %131 = getelementptr inbounds i8, ptr %127, i64 -1
  %132 = load i8, ptr %131, align 1, !tbaa !6
  %133 = getelementptr inbounds i8, ptr %128, i64 -1
  store i8 %132, ptr %133, align 1, !tbaa !6
  %134 = icmp eq i64 %130, 0
  br i1 %134, label %135, label %126, !llvm.loop !26

135:                                              ; preds = %56, %126, %29, %50, %91, %120, %9, %65
  ret ptr %0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local noundef ptr @my_memset(ptr noundef returned writeonly captures(ret: address, provenance) %0, i32 noundef %1, i64 noundef %2) #1 {
  %4 = icmp eq i64 %2, 0
  br i1 %4, label %7, label %5

5:                                                ; preds = %3
  %6 = trunc i32 %1 to i8
  tail call void @llvm.memset.p0.i64(ptr align 1 %0, i8 %6, i64 %2, i1 false), !tbaa !6
  br label %7

7:                                                ; preds = %5, %3
  ret ptr %0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @my_bzero(ptr noundef writeonly captures(none) %0, i64 noundef %1) #1 {
  %3 = icmp eq i64 %1, 0
  br i1 %3, label %5, label %4

4:                                                ; preds = %2
  tail call void @llvm.memset.p0.i64(ptr align 1 %0, i8 0, i64 %1, i1 false), !tbaa !6
  br label %5

5:                                                ; preds = %4, %2
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef ptr @memcpy(ptr noundef returned writeonly captures(ret: address, provenance) %0, ptr noundef readonly captures(none) %1, i64 noundef %2) local_unnamed_addr #2 {
  %4 = icmp eq i64 %2, 0
  br i1 %4, label %62, label %5

5:                                                ; preds = %3
  %6 = ptrtoint ptr %0 to i64
  %7 = ptrtoint ptr %1 to i64
  %8 = icmp ult i64 %2, 8
  %9 = sub i64 %6, %7
  %10 = icmp ult i64 %9, 32
  %11 = or i1 %8, %10
  br i1 %11, label %49, label %12

12:                                               ; preds = %5
  %13 = icmp ult i64 %2, 32
  br i1 %13, label %34, label %14

14:                                               ; preds = %12
  %15 = and i64 %2, -32
  br label %16

16:                                               ; preds = %16, %14
  %17 = phi i64 [ 0, %14 ], [ %24, %16 ]
  %18 = getelementptr i8, ptr %1, i64 %17
  %19 = getelementptr i8, ptr %0, i64 %17
  %20 = getelementptr i8, ptr %18, i64 16
  %21 = load <16 x i8>, ptr %18, align 1, !tbaa !6
  %22 = load <16 x i8>, ptr %20, align 1, !tbaa !6
  %23 = getelementptr i8, ptr %19, i64 16
  store <16 x i8> %21, ptr %19, align 1, !tbaa !6
  store <16 x i8> %22, ptr %23, align 1, !tbaa !6
  %24 = add nuw i64 %17, 32
  %25 = icmp eq i64 %24, %15
  br i1 %25, label %26, label %16, !llvm.loop !27

26:                                               ; preds = %16
  %27 = icmp eq i64 %2, %15
  br i1 %27, label %62, label %28

28:                                               ; preds = %26
  %29 = getelementptr i8, ptr %1, i64 %15
  %30 = getelementptr i8, ptr %0, i64 %15
  %31 = and i64 %2, 31
  %32 = and i64 %2, 24
  %33 = icmp eq i64 %32, 0
  br i1 %33, label %49, label %34

34:                                               ; preds = %28, %12
  %35 = phi i64 [ %15, %28 ], [ 0, %12 ]
  %36 = and i64 %2, -8
  %37 = getelementptr i8, ptr %1, i64 %36
  %38 = getelementptr i8, ptr %0, i64 %36
  %39 = and i64 %2, 7
  br label %40

40:                                               ; preds = %40, %34
  %41 = phi i64 [ %35, %34 ], [ %45, %40 ]
  %42 = getelementptr i8, ptr %1, i64 %41
  %43 = getelementptr i8, ptr %0, i64 %41
  %44 = load <8 x i8>, ptr %42, align 1, !tbaa !6
  store <8 x i8> %44, ptr %43, align 1, !tbaa !6
  %45 = add nuw i64 %41, 8
  %46 = icmp eq i64 %45, %36
  br i1 %46, label %47, label %40, !llvm.loop !28

47:                                               ; preds = %40
  %48 = icmp eq i64 %2, %36
  br i1 %48, label %62, label %49

49:                                               ; preds = %28, %47, %5
  %50 = phi ptr [ %1, %5 ], [ %29, %28 ], [ %37, %47 ]
  %51 = phi ptr [ %0, %5 ], [ %30, %28 ], [ %38, %47 ]
  %52 = phi i64 [ %2, %5 ], [ %31, %28 ], [ %39, %47 ]
  br label %53

53:                                               ; preds = %49, %53
  %54 = phi ptr [ %58, %53 ], [ %50, %49 ]
  %55 = phi ptr [ %60, %53 ], [ %51, %49 ]
  %56 = phi i64 [ %57, %53 ], [ %52, %49 ]
  %57 = add i64 %56, -1
  %58 = getelementptr inbounds nuw i8, ptr %54, i64 1
  %59 = load i8, ptr %54, align 1, !tbaa !6
  %60 = getelementptr inbounds nuw i8, ptr %55, i64 1
  store i8 %59, ptr %55, align 1, !tbaa !6
  %61 = icmp eq i64 %57, 0
  br i1 %61, label %62, label %53, !llvm.loop !29

62:                                               ; preds = %53, %26, %47, %3
  %63 = load i32, ptr @inside_main, align 4, !tbaa !30
  %64 = icmp eq i32 %63, 0
  br i1 %64, label %66, label %65

65:                                               ; preds = %62
  tail call void @abort() #5
  unreachable

66:                                               ; preds = %62
  ret ptr %0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

; Function Attrs: nofree nounwind uwtable
define dso_local void @bcopy(ptr noundef readonly captures(address) %0, ptr noundef writeonly captures(address) %1, i64 noundef %2) local_unnamed_addr #2 {
  %4 = ptrtoint ptr %1 to i64
  %5 = ptrtoint ptr %0 to i64
  %6 = ptrtoint ptr %1 to i64
  %7 = icmp ult ptr %0, %1
  %8 = icmp eq i64 %2, 0
  br i1 %7, label %65, label %9

9:                                                ; preds = %3
  br i1 %8, label %135, label %10

10:                                               ; preds = %9
  %11 = icmp ult i64 %2, 8
  %12 = sub i64 %6, %5
  %13 = icmp ult i64 %12, 32
  %14 = or i1 %11, %13
  br i1 %14, label %52, label %15

15:                                               ; preds = %10
  %16 = icmp ult i64 %2, 32
  br i1 %16, label %37, label %17

17:                                               ; preds = %15
  %18 = and i64 %2, -32
  br label %19

19:                                               ; preds = %19, %17
  %20 = phi i64 [ 0, %17 ], [ %27, %19 ]
  %21 = getelementptr i8, ptr %0, i64 %20
  %22 = getelementptr i8, ptr %1, i64 %20
  %23 = getelementptr i8, ptr %21, i64 16
  %24 = load <16 x i8>, ptr %21, align 1, !tbaa !6
  %25 = load <16 x i8>, ptr %23, align 1, !tbaa !6
  %26 = getelementptr i8, ptr %22, i64 16
  store <16 x i8> %24, ptr %22, align 1, !tbaa !6
  store <16 x i8> %25, ptr %26, align 1, !tbaa !6
  %27 = add nuw i64 %20, 32
  %28 = icmp eq i64 %27, %18
  br i1 %28, label %29, label %19, !llvm.loop !32

29:                                               ; preds = %19
  %30 = icmp eq i64 %2, %18
  br i1 %30, label %135, label %31

31:                                               ; preds = %29
  %32 = getelementptr i8, ptr %0, i64 %18
  %33 = getelementptr i8, ptr %1, i64 %18
  %34 = and i64 %2, 31
  %35 = and i64 %2, 24
  %36 = icmp eq i64 %35, 0
  br i1 %36, label %52, label %37

37:                                               ; preds = %31, %15
  %38 = phi i64 [ %18, %31 ], [ 0, %15 ]
  %39 = and i64 %2, -8
  %40 = getelementptr i8, ptr %0, i64 %39
  %41 = getelementptr i8, ptr %1, i64 %39
  %42 = and i64 %2, 7
  br label %43

43:                                               ; preds = %43, %37
  %44 = phi i64 [ %38, %37 ], [ %48, %43 ]
  %45 = getelementptr i8, ptr %0, i64 %44
  %46 = getelementptr i8, ptr %1, i64 %44
  %47 = load <8 x i8>, ptr %45, align 1, !tbaa !6
  store <8 x i8> %47, ptr %46, align 1, !tbaa !6
  %48 = add nuw i64 %44, 8
  %49 = icmp eq i64 %48, %39
  br i1 %49, label %50, label %43, !llvm.loop !33

50:                                               ; preds = %43
  %51 = icmp eq i64 %2, %39
  br i1 %51, label %135, label %52

52:                                               ; preds = %31, %50, %10
  %53 = phi ptr [ %0, %10 ], [ %32, %31 ], [ %40, %50 ]
  %54 = phi ptr [ %1, %10 ], [ %33, %31 ], [ %41, %50 ]
  %55 = phi i64 [ %2, %10 ], [ %34, %31 ], [ %42, %50 ]
  br label %56

56:                                               ; preds = %52, %56
  %57 = phi ptr [ %61, %56 ], [ %53, %52 ]
  %58 = phi ptr [ %63, %56 ], [ %54, %52 ]
  %59 = phi i64 [ %60, %56 ], [ %55, %52 ]
  %60 = add i64 %59, -1
  %61 = getelementptr inbounds nuw i8, ptr %57, i64 1
  %62 = load i8, ptr %57, align 1, !tbaa !6
  %63 = getelementptr inbounds nuw i8, ptr %58, i64 1
  store i8 %62, ptr %58, align 1, !tbaa !6
  %64 = icmp eq i64 %60, 0
  br i1 %64, label %135, label %56, !llvm.loop !34

65:                                               ; preds = %3
  br i1 %8, label %135, label %66

66:                                               ; preds = %65
  %67 = getelementptr inbounds nuw i8, ptr %0, i64 %2
  %68 = getelementptr inbounds nuw i8, ptr %1, i64 %2
  %69 = icmp ult i64 %2, 8
  %70 = sub i64 %5, %4
  %71 = icmp ult i64 %70, 32
  %72 = or i1 %69, %71
  br i1 %72, label %122, label %73

73:                                               ; preds = %66
  %74 = icmp ult i64 %2, 32
  br i1 %74, label %101, label %75

75:                                               ; preds = %73
  %76 = and i64 %2, -32
  br label %77

77:                                               ; preds = %77, %75
  %78 = phi i64 [ 0, %75 ], [ %89, %77 ]
  %79 = sub i64 0, %78
  %80 = getelementptr i8, ptr %67, i64 %79
  %81 = sub i64 0, %78
  %82 = getelementptr i8, ptr %68, i64 %81
  %83 = getelementptr inbounds i8, ptr %80, i64 -16
  %84 = getelementptr inbounds i8, ptr %80, i64 -32
  %85 = load <16 x i8>, ptr %83, align 1, !tbaa !6
  %86 = load <16 x i8>, ptr %84, align 1, !tbaa !6
  %87 = getelementptr inbounds i8, ptr %82, i64 -16
  %88 = getelementptr inbounds i8, ptr %82, i64 -32
  store <16 x i8> %85, ptr %87, align 1, !tbaa !6
  store <16 x i8> %86, ptr %88, align 1, !tbaa !6
  %89 = add nuw i64 %78, 32
  %90 = icmp eq i64 %89, %76
  br i1 %90, label %91, label %77, !llvm.loop !35

91:                                               ; preds = %77
  %92 = icmp eq i64 %2, %76
  br i1 %92, label %135, label %93

93:                                               ; preds = %91
  %94 = sub i64 0, %76
  %95 = getelementptr i8, ptr %67, i64 %94
  %96 = sub i64 0, %76
  %97 = getelementptr i8, ptr %68, i64 %96
  %98 = and i64 %2, 31
  %99 = and i64 %2, 24
  %100 = icmp eq i64 %99, 0
  br i1 %100, label %122, label %101

101:                                              ; preds = %93, %73
  %102 = phi i64 [ %76, %93 ], [ 0, %73 ]
  %103 = and i64 %2, -8
  %104 = sub i64 0, %103
  %105 = getelementptr i8, ptr %67, i64 %104
  %106 = sub i64 0, %103
  %107 = getelementptr i8, ptr %68, i64 %106
  %108 = and i64 %2, 7
  br label %109

109:                                              ; preds = %109, %101
  %110 = phi i64 [ %102, %101 ], [ %118, %109 ]
  %111 = sub i64 0, %110
  %112 = getelementptr i8, ptr %67, i64 %111
  %113 = sub i64 0, %110
  %114 = getelementptr i8, ptr %68, i64 %113
  %115 = getelementptr inbounds i8, ptr %112, i64 -8
  %116 = load <8 x i8>, ptr %115, align 1, !tbaa !6
  %117 = getelementptr inbounds i8, ptr %114, i64 -8
  store <8 x i8> %116, ptr %117, align 1, !tbaa !6
  %118 = add nuw i64 %110, 8
  %119 = icmp eq i64 %118, %103
  br i1 %119, label %120, label %109, !llvm.loop !36

120:                                              ; preds = %109
  %121 = icmp eq i64 %2, %103
  br i1 %121, label %135, label %122

122:                                              ; preds = %93, %120, %66
  %123 = phi ptr [ %67, %66 ], [ %95, %93 ], [ %105, %120 ]
  %124 = phi ptr [ %68, %66 ], [ %97, %93 ], [ %107, %120 ]
  %125 = phi i64 [ %2, %66 ], [ %98, %93 ], [ %108, %120 ]
  br label %126

126:                                              ; preds = %122, %126
  %127 = phi ptr [ %131, %126 ], [ %123, %122 ]
  %128 = phi ptr [ %133, %126 ], [ %124, %122 ]
  %129 = phi i64 [ %130, %126 ], [ %125, %122 ]
  %130 = add i64 %129, -1
  %131 = getelementptr inbounds i8, ptr %127, i64 -1
  %132 = load i8, ptr %131, align 1, !tbaa !6
  %133 = getelementptr inbounds i8, ptr %128, i64 -1
  store i8 %132, ptr %133, align 1, !tbaa !6
  %134 = icmp eq i64 %130, 0
  br i1 %134, label %135, label %126, !llvm.loop !37

135:                                              ; preds = %56, %126, %29, %50, %91, %120, %9, %65
  %136 = load i32, ptr @inside_main, align 4, !tbaa !30
  %137 = icmp eq i32 %136, 0
  br i1 %137, label %139, label %138

138:                                              ; preds = %135
  tail call void @abort() #5
  unreachable

139:                                              ; preds = %135
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef ptr @memset(ptr noundef returned writeonly captures(ret: address, provenance) %0, i32 noundef %1, i64 noundef %2) local_unnamed_addr #2 {
  %4 = icmp eq i64 %2, 0
  br i1 %4, label %7, label %5

5:                                                ; preds = %3
  %6 = trunc i32 %1 to i8
  tail call void @llvm.memset.p0.i64(ptr align 1 %0, i8 %6, i64 %2, i1 false), !tbaa !6
  br label %7

7:                                                ; preds = %3, %5
  %8 = load i32, ptr @inside_main, align 4, !tbaa !30
  %9 = icmp eq i32 %8, 0
  br i1 %9, label %11, label %10

10:                                               ; preds = %7
  tail call void @abort() #5
  unreachable

11:                                               ; preds = %7
  ret ptr %0
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @bzero(ptr noundef writeonly captures(none) %0, i64 noundef %1) local_unnamed_addr #2 {
  %3 = icmp eq i64 %1, 0
  br i1 %3, label %5, label %4

4:                                                ; preds = %2
  tail call void @llvm.memset.p0.i64(ptr align 1 %0, i8 0, i64 %1, i1 false), !tbaa !6
  br label %5

5:                                                ; preds = %2, %4
  %6 = load i32, ptr @inside_main, align 4, !tbaa !30
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %9, label %8

8:                                                ; preds = %5
  tail call void @abort() #5
  unreachable

9:                                                ; preds = %5
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #4

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nocallback nofree nounwind willreturn memory(argmem: write) }
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
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = distinct !{!9, !10, !11, !12}
!10 = !{!"llvm.loop.mustprogress"}
!11 = !{!"llvm.loop.isvectorized", i32 1}
!12 = !{!"llvm.loop.unroll.runtime.disable"}
!13 = distinct !{!13, !10, !11, !12}
!14 = distinct !{!14, !10, !11}
!15 = distinct !{!15, !10, !11, !12}
!16 = distinct !{!16, !10, !11, !12}
!17 = distinct !{!17, !10, !11}
!18 = distinct !{!18, !10, !11, !12}
!19 = distinct !{!19, !10, !11, !12}
!20 = distinct !{!20, !10, !11}
!21 = distinct !{!21, !10, !11, !12}
!22 = distinct !{!22, !10, !11, !12}
!23 = distinct !{!23, !10, !11}
!24 = distinct !{!24, !10, !11, !12}
!25 = distinct !{!25, !10, !11, !12}
!26 = distinct !{!26, !10, !11}
!27 = distinct !{!27, !10, !11, !12}
!28 = distinct !{!28, !10, !11, !12}
!29 = distinct !{!29, !10, !11}
!30 = !{!31, !31, i64 0}
!31 = !{!"int", !7, i64 0}
!32 = distinct !{!32, !10, !11, !12}
!33 = distinct !{!33, !10, !11, !12}
!34 = distinct !{!34, !10, !11}
!35 = distinct !{!35, !10, !11, !12}
!36 = distinct !{!36, !10, !11, !12}
!37 = distinct !{!37, !10, !11}
