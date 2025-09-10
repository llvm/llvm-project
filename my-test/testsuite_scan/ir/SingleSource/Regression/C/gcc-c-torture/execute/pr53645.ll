; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr53645.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr53645.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@u = dso_local global [2 x <4 x i32>] [<4 x i32> <i32 73, i32 65531, i32 0, i32 174>, <4 x i32> <i32 1, i32 8173, i32 -1, i32 -64>], align 16
@s = dso_local global [2 x <4 x i32>] [<4 x i32> <i32 73, i32 -9123, i32 32761, i32 8191>, <4 x i32> <i32 9903, i32 -1, i32 -7323, i32 0>], align 16

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @uq4444(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %4 = lshr <4 x i32> %3, splat (i32 2)
  store <4 x i32> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @ur4444(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %4 = and <4 x i32> %3, splat (i32 3)
  store <4 x i32> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @sq4444(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %4 = sdiv <4 x i32> %3, splat (i32 4)
  store <4 x i32> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @sr4444(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %4 = srem <4 x i32> %3, splat (i32 4)
  store <4 x i32> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @uq1428(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %4 = lshr <4 x i32> %3, <i32 0, i32 2, i32 1, i32 3>
  store <4 x i32> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @ur1428(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %4 = and <4 x i32> %3, <i32 0, i32 3, i32 1, i32 7>
  store <4 x i32> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @sq1428(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %4 = sdiv <4 x i32> %3, <i32 1, i32 4, i32 2, i32 8>
  store <4 x i32> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @sr1428(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %4 = srem <4 x i32> %3, <i32 1, i32 4, i32 2, i32 8>
  store <4 x i32> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @uq3333(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %4 = udiv <4 x i32> %3, splat (i32 3)
  store <4 x i32> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @ur3333(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %4 = urem <4 x i32> %3, splat (i32 3)
  store <4 x i32> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @sq3333(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %4 = sdiv <4 x i32> %3, splat (i32 3)
  store <4 x i32> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @sr3333(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %4 = srem <4 x i32> %3, splat (i32 3)
  store <4 x i32> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @uq6565(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %4 = udiv <4 x i32> %3, <i32 6, i32 5, i32 6, i32 5>
  store <4 x i32> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @ur6565(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %4 = urem <4 x i32> %3, <i32 6, i32 5, i32 6, i32 5>
  store <4 x i32> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @sq6565(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %4 = sdiv <4 x i32> %3, <i32 6, i32 5, i32 6, i32 5>
  store <4 x i32> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @sr6565(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %4 = srem <4 x i32> %3, <i32 6, i32 5, i32 6, i32 5>
  store <4 x i32> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @uq1414146(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %4 = udiv <4 x i32> %3, <i32 14, i32 14, i32 14, i32 6>
  store <4 x i32> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @ur1414146(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %4 = urem <4 x i32> %3, <i32 14, i32 14, i32 14, i32 6>
  store <4 x i32> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @sq1414146(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %4 = sdiv <4 x i32> %3, <i32 14, i32 14, i32 14, i32 6>
  store <4 x i32> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @sr1414146(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %4 = srem <4 x i32> %3, <i32 14, i32 14, i32 14, i32 6>
  store <4 x i32> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @uq7777(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %4 = udiv <4 x i32> %3, splat (i32 7)
  store <4 x i32> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @ur7777(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %4 = urem <4 x i32> %3, splat (i32 7)
  store <4 x i32> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @sq7777(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %4 = sdiv <4 x i32> %3, splat (i32 7)
  store <4 x i32> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @sr7777(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %4 = srem <4 x i32> %3, splat (i32 7)
  store <4 x i32> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = alloca <4 x i32>, align 16
  %2 = alloca <4 x i32>, align 16
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #4
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #4
  br label %3

3:                                                ; preds = %0, %314
  %4 = phi i1 [ true, %0 ], [ false, %314 ]
  %5 = phi i64 [ 0, %0 ], [ 1, %314 ]
  %6 = getelementptr inbounds nuw <4 x i32>, ptr @u, i64 %5
  call void @uq4444(ptr noundef nonnull %1, ptr noundef nonnull %6)
  %7 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %8 = extractelement <4 x i32> %7, i64 0
  %9 = load <4 x i32>, ptr %6, align 16, !tbaa !6
  %10 = extractelement <4 x i32> %9, i64 0
  %11 = lshr i32 %10, 2
  %12 = icmp eq i32 %8, %11
  br i1 %12, label %13, label %18

13:                                               ; preds = %3
  %14 = extractelement <4 x i32> %7, i64 3
  %15 = extractelement <4 x i32> %9, i64 3
  %16 = lshr i32 %15, 2
  %17 = icmp eq i32 %14, %16
  br i1 %17, label %19, label %18

18:                                               ; preds = %13, %3
  call void @abort() #5
  unreachable

19:                                               ; preds = %13
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !9
  %20 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %21 = extractelement <4 x i32> %20, i64 2
  %22 = load <4 x i32>, ptr %6, align 16, !tbaa !6
  %23 = extractelement <4 x i32> %22, i64 2
  %24 = lshr i32 %23, 2
  %25 = icmp eq i32 %21, %24
  br i1 %25, label %26, label %31

26:                                               ; preds = %19
  %27 = extractelement <4 x i32> %20, i64 1
  %28 = extractelement <4 x i32> %22, i64 1
  %29 = lshr i32 %28, 2
  %30 = icmp eq i32 %27, %29
  br i1 %30, label %32, label %31

31:                                               ; preds = %26, %19
  call void @abort() #5
  unreachable

32:                                               ; preds = %26
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !10
  call void @ur4444(ptr noundef nonnull %1, ptr noundef nonnull %6)
  %33 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %34 = extractelement <4 x i32> %33, i64 0
  %35 = load <4 x i32>, ptr %6, align 16, !tbaa !6
  %36 = extractelement <4 x i32> %35, i64 0
  %37 = and i32 %36, 3
  %38 = icmp eq i32 %34, %37
  br i1 %38, label %39, label %44

39:                                               ; preds = %32
  %40 = extractelement <4 x i32> %33, i64 3
  %41 = extractelement <4 x i32> %35, i64 3
  %42 = and i32 %41, 3
  %43 = icmp eq i32 %40, %42
  br i1 %43, label %45, label %44

44:                                               ; preds = %39, %32
  call void @abort() #5
  unreachable

45:                                               ; preds = %39
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !11
  %46 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %47 = extractelement <4 x i32> %46, i64 2
  %48 = load <4 x i32>, ptr %6, align 16, !tbaa !6
  %49 = extractelement <4 x i32> %48, i64 2
  %50 = and i32 %49, 3
  %51 = icmp eq i32 %47, %50
  br i1 %51, label %52, label %57

52:                                               ; preds = %45
  %53 = extractelement <4 x i32> %46, i64 1
  %54 = extractelement <4 x i32> %48, i64 1
  %55 = and i32 %54, 3
  %56 = icmp eq i32 %53, %55
  br i1 %56, label %58, label %57

57:                                               ; preds = %52, %45
  call void @abort() #5
  unreachable

58:                                               ; preds = %52
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !12
  call void @uq1428(ptr noundef nonnull %1, ptr noundef nonnull %6)
  %59 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %60 = load <4 x i32>, ptr %6, align 16, !tbaa !6
  %61 = icmp eq <4 x i32> %59, %60
  %62 = extractelement <4 x i1> %61, i64 0
  br i1 %62, label %63, label %68

63:                                               ; preds = %58
  %64 = extractelement <4 x i32> %59, i64 3
  %65 = extractelement <4 x i32> %60, i64 3
  %66 = lshr i32 %65, 3
  %67 = icmp eq i32 %64, %66
  br i1 %67, label %69, label %68

68:                                               ; preds = %63, %58
  call void @abort() #5
  unreachable

69:                                               ; preds = %63
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !13
  %70 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %71 = extractelement <4 x i32> %70, i64 2
  %72 = load <4 x i32>, ptr %6, align 16, !tbaa !6
  %73 = extractelement <4 x i32> %72, i64 2
  %74 = lshr i32 %73, 1
  %75 = icmp eq i32 %71, %74
  br i1 %75, label %76, label %81

76:                                               ; preds = %69
  %77 = extractelement <4 x i32> %70, i64 1
  %78 = extractelement <4 x i32> %72, i64 1
  %79 = lshr i32 %78, 2
  %80 = icmp eq i32 %77, %79
  br i1 %80, label %82, label %81

81:                                               ; preds = %76, %69
  call void @abort() #5
  unreachable

82:                                               ; preds = %76
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !14
  call void @ur1428(ptr noundef nonnull %1, ptr noundef nonnull %6)
  %83 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %84 = extractelement <4 x i32> %83, i64 0
  %85 = icmp eq i32 %84, 0
  br i1 %85, label %86, label %92

86:                                               ; preds = %82
  %87 = extractelement <4 x i32> %83, i64 3
  %88 = getelementptr inbounds nuw i8, ptr %6, i64 12
  %89 = load i32, ptr %88, align 4, !tbaa !6
  %90 = and i32 %89, 7
  %91 = icmp eq i32 %87, %90
  br i1 %91, label %93, label %92

92:                                               ; preds = %86, %82
  call void @abort() #5
  unreachable

93:                                               ; preds = %86
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !15
  %94 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %95 = extractelement <4 x i32> %94, i64 2
  %96 = load <4 x i32>, ptr %6, align 16, !tbaa !6
  %97 = extractelement <4 x i32> %96, i64 2
  %98 = and i32 %97, 1
  %99 = icmp eq i32 %95, %98
  br i1 %99, label %100, label %105

100:                                              ; preds = %93
  %101 = extractelement <4 x i32> %94, i64 1
  %102 = extractelement <4 x i32> %96, i64 1
  %103 = and i32 %102, 3
  %104 = icmp eq i32 %101, %103
  br i1 %104, label %106, label %105

105:                                              ; preds = %100, %93
  call void @abort() #5
  unreachable

106:                                              ; preds = %100
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !16
  call void @uq3333(ptr noundef nonnull %1, ptr noundef nonnull %6)
  %107 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %108 = extractelement <4 x i32> %107, i64 0
  %109 = load <4 x i32>, ptr %6, align 16, !tbaa !6
  %110 = extractelement <4 x i32> %109, i64 0
  %111 = udiv i32 %110, 3
  %112 = icmp eq i32 %108, %111
  br i1 %112, label %113, label %118

113:                                              ; preds = %106
  %114 = extractelement <4 x i32> %107, i64 3
  %115 = extractelement <4 x i32> %109, i64 3
  %116 = udiv i32 %115, 3
  %117 = icmp eq i32 %114, %116
  br i1 %117, label %119, label %118

118:                                              ; preds = %113, %106
  call void @abort() #5
  unreachable

119:                                              ; preds = %113
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !17
  %120 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %121 = extractelement <4 x i32> %120, i64 2
  %122 = load <4 x i32>, ptr %6, align 16, !tbaa !6
  %123 = extractelement <4 x i32> %122, i64 2
  %124 = udiv i32 %123, 3
  %125 = icmp eq i32 %121, %124
  br i1 %125, label %126, label %131

126:                                              ; preds = %119
  %127 = extractelement <4 x i32> %120, i64 1
  %128 = extractelement <4 x i32> %122, i64 1
  %129 = udiv i32 %128, 3
  %130 = icmp eq i32 %127, %129
  br i1 %130, label %132, label %131

131:                                              ; preds = %126, %119
  call void @abort() #5
  unreachable

132:                                              ; preds = %126
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !18
  call void @ur3333(ptr noundef nonnull %1, ptr noundef nonnull %6)
  %133 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %134 = extractelement <4 x i32> %133, i64 0
  %135 = load <4 x i32>, ptr %6, align 16, !tbaa !6
  %136 = extractelement <4 x i32> %135, i64 0
  %137 = urem i32 %136, 3
  %138 = icmp eq i32 %134, %137
  br i1 %138, label %139, label %144

139:                                              ; preds = %132
  %140 = extractelement <4 x i32> %133, i64 3
  %141 = extractelement <4 x i32> %135, i64 3
  %142 = urem i32 %141, 3
  %143 = icmp eq i32 %140, %142
  br i1 %143, label %145, label %144

144:                                              ; preds = %139, %132
  call void @abort() #5
  unreachable

145:                                              ; preds = %139
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !19
  %146 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %147 = extractelement <4 x i32> %146, i64 2
  %148 = load <4 x i32>, ptr %6, align 16, !tbaa !6
  %149 = extractelement <4 x i32> %148, i64 2
  %150 = urem i32 %149, 3
  %151 = icmp eq i32 %147, %150
  br i1 %151, label %152, label %157

152:                                              ; preds = %145
  %153 = extractelement <4 x i32> %146, i64 1
  %154 = extractelement <4 x i32> %148, i64 1
  %155 = urem i32 %154, 3
  %156 = icmp eq i32 %153, %155
  br i1 %156, label %158, label %157

157:                                              ; preds = %152, %145
  call void @abort() #5
  unreachable

158:                                              ; preds = %152
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !20
  call void @uq6565(ptr noundef nonnull %1, ptr noundef nonnull %6)
  %159 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %160 = extractelement <4 x i32> %159, i64 0
  %161 = load <4 x i32>, ptr %6, align 16, !tbaa !6
  %162 = extractelement <4 x i32> %161, i64 0
  %163 = udiv i32 %162, 6
  %164 = icmp eq i32 %160, %163
  br i1 %164, label %165, label %170

165:                                              ; preds = %158
  %166 = extractelement <4 x i32> %159, i64 3
  %167 = extractelement <4 x i32> %161, i64 3
  %168 = udiv i32 %167, 5
  %169 = icmp eq i32 %166, %168
  br i1 %169, label %171, label %170

170:                                              ; preds = %165, %158
  call void @abort() #5
  unreachable

171:                                              ; preds = %165
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !21
  %172 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %173 = extractelement <4 x i32> %172, i64 2
  %174 = load <4 x i32>, ptr %6, align 16, !tbaa !6
  %175 = extractelement <4 x i32> %174, i64 2
  %176 = udiv i32 %175, 6
  %177 = icmp eq i32 %173, %176
  br i1 %177, label %178, label %183

178:                                              ; preds = %171
  %179 = extractelement <4 x i32> %172, i64 1
  %180 = extractelement <4 x i32> %174, i64 1
  %181 = udiv i32 %180, 5
  %182 = icmp eq i32 %179, %181
  br i1 %182, label %184, label %183

183:                                              ; preds = %178, %171
  call void @abort() #5
  unreachable

184:                                              ; preds = %178
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !22
  call void @ur6565(ptr noundef nonnull %1, ptr noundef nonnull %6)
  %185 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %186 = extractelement <4 x i32> %185, i64 0
  %187 = load <4 x i32>, ptr %6, align 16, !tbaa !6
  %188 = extractelement <4 x i32> %187, i64 0
  %189 = urem i32 %188, 6
  %190 = icmp eq i32 %186, %189
  br i1 %190, label %191, label %196

191:                                              ; preds = %184
  %192 = extractelement <4 x i32> %185, i64 3
  %193 = extractelement <4 x i32> %187, i64 3
  %194 = urem i32 %193, 5
  %195 = icmp eq i32 %192, %194
  br i1 %195, label %197, label %196

196:                                              ; preds = %191, %184
  call void @abort() #5
  unreachable

197:                                              ; preds = %191
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !23
  %198 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %199 = extractelement <4 x i32> %198, i64 2
  %200 = load <4 x i32>, ptr %6, align 16, !tbaa !6
  %201 = extractelement <4 x i32> %200, i64 2
  %202 = urem i32 %201, 6
  %203 = icmp eq i32 %199, %202
  br i1 %203, label %204, label %209

204:                                              ; preds = %197
  %205 = extractelement <4 x i32> %198, i64 1
  %206 = extractelement <4 x i32> %200, i64 1
  %207 = urem i32 %206, 5
  %208 = icmp eq i32 %205, %207
  br i1 %208, label %210, label %209

209:                                              ; preds = %204, %197
  call void @abort() #5
  unreachable

210:                                              ; preds = %204
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !24
  call void @uq1414146(ptr noundef nonnull %1, ptr noundef nonnull %6)
  %211 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %212 = extractelement <4 x i32> %211, i64 0
  %213 = load <4 x i32>, ptr %6, align 16, !tbaa !6
  %214 = extractelement <4 x i32> %213, i64 0
  %215 = udiv i32 %214, 14
  %216 = icmp eq i32 %212, %215
  br i1 %216, label %217, label %222

217:                                              ; preds = %210
  %218 = extractelement <4 x i32> %211, i64 3
  %219 = extractelement <4 x i32> %213, i64 3
  %220 = udiv i32 %219, 6
  %221 = icmp eq i32 %218, %220
  br i1 %221, label %223, label %222

222:                                              ; preds = %217, %210
  call void @abort() #5
  unreachable

223:                                              ; preds = %217
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !25
  %224 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %225 = extractelement <4 x i32> %224, i64 2
  %226 = load <4 x i32>, ptr %6, align 16, !tbaa !6
  %227 = extractelement <4 x i32> %226, i64 2
  %228 = udiv i32 %227, 14
  %229 = icmp eq i32 %225, %228
  br i1 %229, label %230, label %235

230:                                              ; preds = %223
  %231 = extractelement <4 x i32> %224, i64 1
  %232 = extractelement <4 x i32> %226, i64 1
  %233 = udiv i32 %232, 14
  %234 = icmp eq i32 %231, %233
  br i1 %234, label %236, label %235

235:                                              ; preds = %230, %223
  call void @abort() #5
  unreachable

236:                                              ; preds = %230
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !26
  call void @ur1414146(ptr noundef nonnull %1, ptr noundef nonnull %6)
  %237 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %238 = extractelement <4 x i32> %237, i64 0
  %239 = load <4 x i32>, ptr %6, align 16, !tbaa !6
  %240 = extractelement <4 x i32> %239, i64 0
  %241 = urem i32 %240, 14
  %242 = icmp eq i32 %238, %241
  br i1 %242, label %243, label %248

243:                                              ; preds = %236
  %244 = extractelement <4 x i32> %237, i64 3
  %245 = extractelement <4 x i32> %239, i64 3
  %246 = urem i32 %245, 6
  %247 = icmp eq i32 %244, %246
  br i1 %247, label %249, label %248

248:                                              ; preds = %243, %236
  call void @abort() #5
  unreachable

249:                                              ; preds = %243
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !27
  %250 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %251 = extractelement <4 x i32> %250, i64 2
  %252 = load <4 x i32>, ptr %6, align 16, !tbaa !6
  %253 = extractelement <4 x i32> %252, i64 2
  %254 = urem i32 %253, 14
  %255 = icmp eq i32 %251, %254
  br i1 %255, label %256, label %261

256:                                              ; preds = %249
  %257 = extractelement <4 x i32> %250, i64 1
  %258 = extractelement <4 x i32> %252, i64 1
  %259 = urem i32 %258, 14
  %260 = icmp eq i32 %257, %259
  br i1 %260, label %262, label %261

261:                                              ; preds = %256, %249
  call void @abort() #5
  unreachable

262:                                              ; preds = %256
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !28
  call void @uq7777(ptr noundef nonnull %1, ptr noundef nonnull %6)
  %263 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %264 = extractelement <4 x i32> %263, i64 0
  %265 = load <4 x i32>, ptr %6, align 16, !tbaa !6
  %266 = extractelement <4 x i32> %265, i64 0
  %267 = udiv i32 %266, 7
  %268 = icmp eq i32 %264, %267
  br i1 %268, label %269, label %274

269:                                              ; preds = %262
  %270 = extractelement <4 x i32> %263, i64 3
  %271 = extractelement <4 x i32> %265, i64 3
  %272 = udiv i32 %271, 7
  %273 = icmp eq i32 %270, %272
  br i1 %273, label %275, label %274

274:                                              ; preds = %269, %262
  call void @abort() #5
  unreachable

275:                                              ; preds = %269
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !29
  %276 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %277 = extractelement <4 x i32> %276, i64 2
  %278 = load <4 x i32>, ptr %6, align 16, !tbaa !6
  %279 = extractelement <4 x i32> %278, i64 2
  %280 = udiv i32 %279, 7
  %281 = icmp eq i32 %277, %280
  br i1 %281, label %282, label %287

282:                                              ; preds = %275
  %283 = extractelement <4 x i32> %276, i64 1
  %284 = extractelement <4 x i32> %278, i64 1
  %285 = udiv i32 %284, 7
  %286 = icmp eq i32 %283, %285
  br i1 %286, label %288, label %287

287:                                              ; preds = %282, %275
  call void @abort() #5
  unreachable

288:                                              ; preds = %282
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !30
  call void @ur7777(ptr noundef nonnull %1, ptr noundef nonnull %6)
  %289 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %290 = extractelement <4 x i32> %289, i64 0
  %291 = load <4 x i32>, ptr %6, align 16, !tbaa !6
  %292 = extractelement <4 x i32> %291, i64 0
  %293 = urem i32 %292, 7
  %294 = icmp eq i32 %290, %293
  br i1 %294, label %295, label %300

295:                                              ; preds = %288
  %296 = extractelement <4 x i32> %289, i64 3
  %297 = extractelement <4 x i32> %291, i64 3
  %298 = urem i32 %297, 7
  %299 = icmp eq i32 %296, %298
  br i1 %299, label %301, label %300

300:                                              ; preds = %295, %288
  call void @abort() #5
  unreachable

301:                                              ; preds = %295
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !31
  %302 = load <4 x i32>, ptr %1, align 16, !tbaa !6
  %303 = extractelement <4 x i32> %302, i64 2
  %304 = load <4 x i32>, ptr %6, align 16, !tbaa !6
  %305 = extractelement <4 x i32> %304, i64 2
  %306 = urem i32 %305, 7
  %307 = icmp eq i32 %303, %306
  br i1 %307, label %308, label %313

308:                                              ; preds = %301
  %309 = extractelement <4 x i32> %302, i64 1
  %310 = extractelement <4 x i32> %304, i64 1
  %311 = urem i32 %310, 7
  %312 = icmp eq i32 %309, %311
  br i1 %312, label %314, label %313

313:                                              ; preds = %308, %301
  call void @abort() #5
  unreachable

314:                                              ; preds = %308
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !32
  br i1 %4, label %3, label %315, !llvm.loop !33

315:                                              ; preds = %314, %626
  %316 = phi i1 [ false, %626 ], [ true, %314 ]
  %317 = phi i64 [ 1, %626 ], [ 0, %314 ]
  %318 = getelementptr inbounds nuw <4 x i32>, ptr @s, i64 %317
  call void @sq4444(ptr noundef nonnull %2, ptr noundef nonnull %318)
  %319 = load <4 x i32>, ptr %2, align 16, !tbaa !6
  %320 = extractelement <4 x i32> %319, i64 0
  %321 = load <4 x i32>, ptr %318, align 16, !tbaa !6
  %322 = extractelement <4 x i32> %321, i64 0
  %323 = sdiv i32 %322, 4
  %324 = icmp eq i32 %320, %323
  br i1 %324, label %325, label %330

325:                                              ; preds = %315
  %326 = extractelement <4 x i32> %319, i64 3
  %327 = extractelement <4 x i32> %321, i64 3
  %328 = sdiv i32 %327, 4
  %329 = icmp eq i32 %326, %328
  br i1 %329, label %331, label %330

330:                                              ; preds = %325, %315
  call void @abort() #5
  unreachable

331:                                              ; preds = %325
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !35
  %332 = load <4 x i32>, ptr %2, align 16, !tbaa !6
  %333 = extractelement <4 x i32> %332, i64 2
  %334 = load <4 x i32>, ptr %318, align 16, !tbaa !6
  %335 = extractelement <4 x i32> %334, i64 2
  %336 = sdiv i32 %335, 4
  %337 = icmp eq i32 %333, %336
  br i1 %337, label %338, label %343

338:                                              ; preds = %331
  %339 = extractelement <4 x i32> %332, i64 1
  %340 = extractelement <4 x i32> %334, i64 1
  %341 = sdiv i32 %340, 4
  %342 = icmp eq i32 %339, %341
  br i1 %342, label %344, label %343

343:                                              ; preds = %338, %331
  call void @abort() #5
  unreachable

344:                                              ; preds = %338
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !36
  call void @sr4444(ptr noundef nonnull %2, ptr noundef nonnull %318)
  %345 = load <4 x i32>, ptr %2, align 16, !tbaa !6
  %346 = extractelement <4 x i32> %345, i64 0
  %347 = load <4 x i32>, ptr %318, align 16, !tbaa !6
  %348 = extractelement <4 x i32> %347, i64 0
  %349 = srem i32 %348, 4
  %350 = icmp eq i32 %346, %349
  br i1 %350, label %351, label %356

351:                                              ; preds = %344
  %352 = extractelement <4 x i32> %345, i64 3
  %353 = extractelement <4 x i32> %347, i64 3
  %354 = srem i32 %353, 4
  %355 = icmp eq i32 %352, %354
  br i1 %355, label %357, label %356

356:                                              ; preds = %351, %344
  call void @abort() #5
  unreachable

357:                                              ; preds = %351
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !37
  %358 = load <4 x i32>, ptr %2, align 16, !tbaa !6
  %359 = extractelement <4 x i32> %358, i64 2
  %360 = load <4 x i32>, ptr %318, align 16, !tbaa !6
  %361 = extractelement <4 x i32> %360, i64 2
  %362 = srem i32 %361, 4
  %363 = icmp eq i32 %359, %362
  br i1 %363, label %364, label %369

364:                                              ; preds = %357
  %365 = extractelement <4 x i32> %358, i64 1
  %366 = extractelement <4 x i32> %360, i64 1
  %367 = srem i32 %366, 4
  %368 = icmp eq i32 %365, %367
  br i1 %368, label %370, label %369

369:                                              ; preds = %364, %357
  call void @abort() #5
  unreachable

370:                                              ; preds = %364
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !38
  call void @sq1428(ptr noundef nonnull %2, ptr noundef nonnull %318)
  %371 = load <4 x i32>, ptr %2, align 16, !tbaa !6
  %372 = load <4 x i32>, ptr %318, align 16, !tbaa !6
  %373 = icmp eq <4 x i32> %371, %372
  %374 = extractelement <4 x i1> %373, i64 0
  br i1 %374, label %375, label %380

375:                                              ; preds = %370
  %376 = extractelement <4 x i32> %371, i64 3
  %377 = extractelement <4 x i32> %372, i64 3
  %378 = sdiv i32 %377, 8
  %379 = icmp eq i32 %376, %378
  br i1 %379, label %381, label %380

380:                                              ; preds = %375, %370
  call void @abort() #5
  unreachable

381:                                              ; preds = %375
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !39
  %382 = load <4 x i32>, ptr %2, align 16, !tbaa !6
  %383 = extractelement <4 x i32> %382, i64 2
  %384 = load <4 x i32>, ptr %318, align 16, !tbaa !6
  %385 = extractelement <4 x i32> %384, i64 2
  %386 = sdiv i32 %385, 2
  %387 = icmp eq i32 %383, %386
  br i1 %387, label %388, label %393

388:                                              ; preds = %381
  %389 = extractelement <4 x i32> %382, i64 1
  %390 = extractelement <4 x i32> %384, i64 1
  %391 = sdiv i32 %390, 4
  %392 = icmp eq i32 %389, %391
  br i1 %392, label %394, label %393

393:                                              ; preds = %388, %381
  call void @abort() #5
  unreachable

394:                                              ; preds = %388
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !40
  call void @sr1428(ptr noundef nonnull %2, ptr noundef nonnull %318)
  %395 = load <4 x i32>, ptr %2, align 16, !tbaa !6
  %396 = extractelement <4 x i32> %395, i64 0
  %397 = icmp eq i32 %396, 0
  br i1 %397, label %398, label %404

398:                                              ; preds = %394
  %399 = extractelement <4 x i32> %395, i64 3
  %400 = getelementptr inbounds nuw i8, ptr %318, i64 12
  %401 = load i32, ptr %400, align 4, !tbaa !6
  %402 = srem i32 %401, 8
  %403 = icmp eq i32 %399, %402
  br i1 %403, label %405, label %404

404:                                              ; preds = %398, %394
  call void @abort() #5
  unreachable

405:                                              ; preds = %398
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !41
  %406 = load <4 x i32>, ptr %2, align 16, !tbaa !6
  %407 = extractelement <4 x i32> %406, i64 2
  %408 = load <4 x i32>, ptr %318, align 16, !tbaa !6
  %409 = extractelement <4 x i32> %408, i64 2
  %410 = srem i32 %409, 2
  %411 = icmp eq i32 %407, %410
  br i1 %411, label %412, label %417

412:                                              ; preds = %405
  %413 = extractelement <4 x i32> %406, i64 1
  %414 = extractelement <4 x i32> %408, i64 1
  %415 = srem i32 %414, 4
  %416 = icmp eq i32 %413, %415
  br i1 %416, label %418, label %417

417:                                              ; preds = %412, %405
  call void @abort() #5
  unreachable

418:                                              ; preds = %412
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !42
  call void @sq3333(ptr noundef nonnull %2, ptr noundef nonnull %318)
  %419 = load <4 x i32>, ptr %2, align 16, !tbaa !6
  %420 = extractelement <4 x i32> %419, i64 0
  %421 = load <4 x i32>, ptr %318, align 16, !tbaa !6
  %422 = extractelement <4 x i32> %421, i64 0
  %423 = sdiv i32 %422, 3
  %424 = icmp eq i32 %420, %423
  br i1 %424, label %425, label %430

425:                                              ; preds = %418
  %426 = extractelement <4 x i32> %419, i64 3
  %427 = extractelement <4 x i32> %421, i64 3
  %428 = sdiv i32 %427, 3
  %429 = icmp eq i32 %426, %428
  br i1 %429, label %431, label %430

430:                                              ; preds = %425, %418
  call void @abort() #5
  unreachable

431:                                              ; preds = %425
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !43
  %432 = load <4 x i32>, ptr %2, align 16, !tbaa !6
  %433 = extractelement <4 x i32> %432, i64 2
  %434 = load <4 x i32>, ptr %318, align 16, !tbaa !6
  %435 = extractelement <4 x i32> %434, i64 2
  %436 = sdiv i32 %435, 3
  %437 = icmp eq i32 %433, %436
  br i1 %437, label %438, label %443

438:                                              ; preds = %431
  %439 = extractelement <4 x i32> %432, i64 1
  %440 = extractelement <4 x i32> %434, i64 1
  %441 = sdiv i32 %440, 3
  %442 = icmp eq i32 %439, %441
  br i1 %442, label %444, label %443

443:                                              ; preds = %438, %431
  call void @abort() #5
  unreachable

444:                                              ; preds = %438
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !44
  call void @sr3333(ptr noundef nonnull %2, ptr noundef nonnull %318)
  %445 = load <4 x i32>, ptr %2, align 16, !tbaa !6
  %446 = extractelement <4 x i32> %445, i64 0
  %447 = load <4 x i32>, ptr %318, align 16, !tbaa !6
  %448 = extractelement <4 x i32> %447, i64 0
  %449 = srem i32 %448, 3
  %450 = icmp eq i32 %446, %449
  br i1 %450, label %451, label %456

451:                                              ; preds = %444
  %452 = extractelement <4 x i32> %445, i64 3
  %453 = extractelement <4 x i32> %447, i64 3
  %454 = srem i32 %453, 3
  %455 = icmp eq i32 %452, %454
  br i1 %455, label %457, label %456

456:                                              ; preds = %451, %444
  call void @abort() #5
  unreachable

457:                                              ; preds = %451
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !45
  %458 = load <4 x i32>, ptr %2, align 16, !tbaa !6
  %459 = extractelement <4 x i32> %458, i64 2
  %460 = load <4 x i32>, ptr %318, align 16, !tbaa !6
  %461 = extractelement <4 x i32> %460, i64 2
  %462 = srem i32 %461, 3
  %463 = icmp eq i32 %459, %462
  br i1 %463, label %464, label %469

464:                                              ; preds = %457
  %465 = extractelement <4 x i32> %458, i64 1
  %466 = extractelement <4 x i32> %460, i64 1
  %467 = srem i32 %466, 3
  %468 = icmp eq i32 %465, %467
  br i1 %468, label %470, label %469

469:                                              ; preds = %464, %457
  call void @abort() #5
  unreachable

470:                                              ; preds = %464
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !46
  call void @sq6565(ptr noundef nonnull %2, ptr noundef nonnull %318)
  %471 = load <4 x i32>, ptr %2, align 16, !tbaa !6
  %472 = extractelement <4 x i32> %471, i64 0
  %473 = load <4 x i32>, ptr %318, align 16, !tbaa !6
  %474 = extractelement <4 x i32> %473, i64 0
  %475 = sdiv i32 %474, 6
  %476 = icmp eq i32 %472, %475
  br i1 %476, label %477, label %482

477:                                              ; preds = %470
  %478 = extractelement <4 x i32> %471, i64 3
  %479 = extractelement <4 x i32> %473, i64 3
  %480 = sdiv i32 %479, 5
  %481 = icmp eq i32 %478, %480
  br i1 %481, label %483, label %482

482:                                              ; preds = %477, %470
  call void @abort() #5
  unreachable

483:                                              ; preds = %477
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !47
  %484 = load <4 x i32>, ptr %2, align 16, !tbaa !6
  %485 = extractelement <4 x i32> %484, i64 2
  %486 = load <4 x i32>, ptr %318, align 16, !tbaa !6
  %487 = extractelement <4 x i32> %486, i64 2
  %488 = sdiv i32 %487, 6
  %489 = icmp eq i32 %485, %488
  br i1 %489, label %490, label %495

490:                                              ; preds = %483
  %491 = extractelement <4 x i32> %484, i64 1
  %492 = extractelement <4 x i32> %486, i64 1
  %493 = sdiv i32 %492, 5
  %494 = icmp eq i32 %491, %493
  br i1 %494, label %496, label %495

495:                                              ; preds = %490, %483
  call void @abort() #5
  unreachable

496:                                              ; preds = %490
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !48
  call void @sr6565(ptr noundef nonnull %2, ptr noundef nonnull %318)
  %497 = load <4 x i32>, ptr %2, align 16, !tbaa !6
  %498 = extractelement <4 x i32> %497, i64 0
  %499 = load <4 x i32>, ptr %318, align 16, !tbaa !6
  %500 = extractelement <4 x i32> %499, i64 0
  %501 = srem i32 %500, 6
  %502 = icmp eq i32 %498, %501
  br i1 %502, label %503, label %508

503:                                              ; preds = %496
  %504 = extractelement <4 x i32> %497, i64 3
  %505 = extractelement <4 x i32> %499, i64 3
  %506 = srem i32 %505, 5
  %507 = icmp eq i32 %504, %506
  br i1 %507, label %509, label %508

508:                                              ; preds = %503, %496
  call void @abort() #5
  unreachable

509:                                              ; preds = %503
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !49
  %510 = load <4 x i32>, ptr %2, align 16, !tbaa !6
  %511 = extractelement <4 x i32> %510, i64 2
  %512 = load <4 x i32>, ptr %318, align 16, !tbaa !6
  %513 = extractelement <4 x i32> %512, i64 2
  %514 = srem i32 %513, 6
  %515 = icmp eq i32 %511, %514
  br i1 %515, label %516, label %521

516:                                              ; preds = %509
  %517 = extractelement <4 x i32> %510, i64 1
  %518 = extractelement <4 x i32> %512, i64 1
  %519 = srem i32 %518, 5
  %520 = icmp eq i32 %517, %519
  br i1 %520, label %522, label %521

521:                                              ; preds = %516, %509
  call void @abort() #5
  unreachable

522:                                              ; preds = %516
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !50
  call void @sq1414146(ptr noundef nonnull %2, ptr noundef nonnull %318)
  %523 = load <4 x i32>, ptr %2, align 16, !tbaa !6
  %524 = extractelement <4 x i32> %523, i64 0
  %525 = load <4 x i32>, ptr %318, align 16, !tbaa !6
  %526 = extractelement <4 x i32> %525, i64 0
  %527 = sdiv i32 %526, 14
  %528 = icmp eq i32 %524, %527
  br i1 %528, label %529, label %534

529:                                              ; preds = %522
  %530 = extractelement <4 x i32> %523, i64 3
  %531 = extractelement <4 x i32> %525, i64 3
  %532 = sdiv i32 %531, 6
  %533 = icmp eq i32 %530, %532
  br i1 %533, label %535, label %534

534:                                              ; preds = %529, %522
  call void @abort() #5
  unreachable

535:                                              ; preds = %529
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !51
  %536 = load <4 x i32>, ptr %2, align 16, !tbaa !6
  %537 = extractelement <4 x i32> %536, i64 2
  %538 = load <4 x i32>, ptr %318, align 16, !tbaa !6
  %539 = extractelement <4 x i32> %538, i64 2
  %540 = sdiv i32 %539, 14
  %541 = icmp eq i32 %537, %540
  br i1 %541, label %542, label %547

542:                                              ; preds = %535
  %543 = extractelement <4 x i32> %536, i64 1
  %544 = extractelement <4 x i32> %538, i64 1
  %545 = sdiv i32 %544, 14
  %546 = icmp eq i32 %543, %545
  br i1 %546, label %548, label %547

547:                                              ; preds = %542, %535
  call void @abort() #5
  unreachable

548:                                              ; preds = %542
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !52
  call void @sr1414146(ptr noundef nonnull %2, ptr noundef nonnull %318)
  %549 = load <4 x i32>, ptr %2, align 16, !tbaa !6
  %550 = extractelement <4 x i32> %549, i64 0
  %551 = load <4 x i32>, ptr %318, align 16, !tbaa !6
  %552 = extractelement <4 x i32> %551, i64 0
  %553 = srem i32 %552, 14
  %554 = icmp eq i32 %550, %553
  br i1 %554, label %555, label %560

555:                                              ; preds = %548
  %556 = extractelement <4 x i32> %549, i64 3
  %557 = extractelement <4 x i32> %551, i64 3
  %558 = srem i32 %557, 6
  %559 = icmp eq i32 %556, %558
  br i1 %559, label %561, label %560

560:                                              ; preds = %555, %548
  call void @abort() #5
  unreachable

561:                                              ; preds = %555
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !53
  %562 = load <4 x i32>, ptr %2, align 16, !tbaa !6
  %563 = extractelement <4 x i32> %562, i64 2
  %564 = load <4 x i32>, ptr %318, align 16, !tbaa !6
  %565 = extractelement <4 x i32> %564, i64 2
  %566 = srem i32 %565, 14
  %567 = icmp eq i32 %563, %566
  br i1 %567, label %568, label %573

568:                                              ; preds = %561
  %569 = extractelement <4 x i32> %562, i64 1
  %570 = extractelement <4 x i32> %564, i64 1
  %571 = srem i32 %570, 14
  %572 = icmp eq i32 %569, %571
  br i1 %572, label %574, label %573

573:                                              ; preds = %568, %561
  call void @abort() #5
  unreachable

574:                                              ; preds = %568
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !54
  call void @sq7777(ptr noundef nonnull %2, ptr noundef nonnull %318)
  %575 = load <4 x i32>, ptr %2, align 16, !tbaa !6
  %576 = extractelement <4 x i32> %575, i64 0
  %577 = load <4 x i32>, ptr %318, align 16, !tbaa !6
  %578 = extractelement <4 x i32> %577, i64 0
  %579 = sdiv i32 %578, 7
  %580 = icmp eq i32 %576, %579
  br i1 %580, label %581, label %586

581:                                              ; preds = %574
  %582 = extractelement <4 x i32> %575, i64 3
  %583 = extractelement <4 x i32> %577, i64 3
  %584 = sdiv i32 %583, 7
  %585 = icmp eq i32 %582, %584
  br i1 %585, label %587, label %586

586:                                              ; preds = %581, %574
  call void @abort() #5
  unreachable

587:                                              ; preds = %581
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !55
  %588 = load <4 x i32>, ptr %2, align 16, !tbaa !6
  %589 = extractelement <4 x i32> %588, i64 2
  %590 = load <4 x i32>, ptr %318, align 16, !tbaa !6
  %591 = extractelement <4 x i32> %590, i64 2
  %592 = sdiv i32 %591, 7
  %593 = icmp eq i32 %589, %592
  br i1 %593, label %594, label %599

594:                                              ; preds = %587
  %595 = extractelement <4 x i32> %588, i64 1
  %596 = extractelement <4 x i32> %590, i64 1
  %597 = sdiv i32 %596, 7
  %598 = icmp eq i32 %595, %597
  br i1 %598, label %600, label %599

599:                                              ; preds = %594, %587
  call void @abort() #5
  unreachable

600:                                              ; preds = %594
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !56
  call void @sr7777(ptr noundef nonnull %2, ptr noundef nonnull %318)
  %601 = load <4 x i32>, ptr %2, align 16, !tbaa !6
  %602 = extractelement <4 x i32> %601, i64 0
  %603 = load <4 x i32>, ptr %318, align 16, !tbaa !6
  %604 = extractelement <4 x i32> %603, i64 0
  %605 = srem i32 %604, 7
  %606 = icmp eq i32 %602, %605
  br i1 %606, label %607, label %612

607:                                              ; preds = %600
  %608 = extractelement <4 x i32> %601, i64 3
  %609 = extractelement <4 x i32> %603, i64 3
  %610 = srem i32 %609, 7
  %611 = icmp eq i32 %608, %610
  br i1 %611, label %613, label %612

612:                                              ; preds = %607, %600
  call void @abort() #5
  unreachable

613:                                              ; preds = %607
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !57
  %614 = load <4 x i32>, ptr %2, align 16, !tbaa !6
  %615 = extractelement <4 x i32> %614, i64 2
  %616 = load <4 x i32>, ptr %318, align 16, !tbaa !6
  %617 = extractelement <4 x i32> %616, i64 2
  %618 = srem i32 %617, 7
  %619 = icmp eq i32 %615, %618
  br i1 %619, label %620, label %625

620:                                              ; preds = %613
  %621 = extractelement <4 x i32> %614, i64 1
  %622 = extractelement <4 x i32> %616, i64 1
  %623 = srem i32 %622, 7
  %624 = icmp eq i32 %621, %623
  br i1 %624, label %626, label %625

625:                                              ; preds = %620, %613
  call void @abort() #5
  unreachable

626:                                              ; preds = %620
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !58
  br i1 %316, label %315, label %627, !llvm.loop !59

627:                                              ; preds = %626
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #4
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #4
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #2

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #2

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
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
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = !{i64 2147508546}
!10 = !{i64 2147508675}
!11 = !{i64 2147508841}
!12 = !{i64 2147508970}
!13 = !{i64 2147509175}
!14 = !{i64 2147509304}
!15 = !{i64 2147509470}
!16 = !{i64 2147509599}
!17 = !{i64 2147509804}
!18 = !{i64 2147509933}
!19 = !{i64 2147510099}
!20 = !{i64 2147510228}
!21 = !{i64 2147510433}
!22 = !{i64 2147510562}
!23 = !{i64 2147510728}
!24 = !{i64 2147510857}
!25 = !{i64 2147511062}
!26 = !{i64 2147511191}
!27 = !{i64 2147511357}
!28 = !{i64 2147511486}
!29 = !{i64 2147511709}
!30 = !{i64 2147511838}
!31 = !{i64 2147512004}
!32 = !{i64 2147512133}
!33 = distinct !{!33, !34}
!34 = !{!"llvm.loop.mustprogress"}
!35 = !{i64 2147512471}
!36 = !{i64 2147512600}
!37 = !{i64 2147512766}
!38 = !{i64 2147512895}
!39 = !{i64 2147513100}
!40 = !{i64 2147513229}
!41 = !{i64 2147513395}
!42 = !{i64 2147513524}
!43 = !{i64 2147513729}
!44 = !{i64 2147513858}
!45 = !{i64 2147514024}
!46 = !{i64 2147514153}
!47 = !{i64 2147514358}
!48 = !{i64 2147514487}
!49 = !{i64 2147514653}
!50 = !{i64 2147514782}
!51 = !{i64 2147514987}
!52 = !{i64 2147515116}
!53 = !{i64 2147515282}
!54 = !{i64 2147515411}
!55 = !{i64 2147515634}
!56 = !{i64 2147515763}
!57 = !{i64 2147515929}
!58 = !{i64 2147516058}
!59 = distinct !{!59, !34}
