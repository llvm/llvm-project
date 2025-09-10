; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr53645-2.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr53645-2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@u = dso_local global [2 x <8 x i16>] [<8 x i16> <i16 73, i16 -5, i16 0, i16 174, i16 921, i16 -1, i16 17, i16 178>, <8 x i16> <i16 1, i16 8173, i16 -1, i16 -64, i16 12, i16 29612, i16 128, i16 8912>], align 16
@s = dso_local global [2 x <8 x i16>] [<8 x i16> <i16 73, i16 -9123, i16 32761, i16 8191, i16 16371, i16 1201, i16 12701, i16 9999>, <8 x i16> <i16 9903, i16 -1, i16 -7323, i16 0, i16 -7, i16 -323, i16 9124, i16 -9199>], align 16

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @uq44444444(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %4 = lshr <8 x i16> %3, splat (i16 2)
  store <8 x i16> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @ur44444444(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %4 = and <8 x i16> %3, splat (i16 3)
  store <8 x i16> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @sq44444444(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %4 = sdiv <8 x i16> %3, splat (i16 4)
  store <8 x i16> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @sr44444444(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %4 = srem <8 x i16> %3, splat (i16 4)
  store <8 x i16> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @uq1428166432128(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %4 = lshr <8 x i16> %3, <i16 0, i16 2, i16 1, i16 3, i16 4, i16 6, i16 5, i16 7>
  store <8 x i16> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @ur1428166432128(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %4 = and <8 x i16> %3, <i16 0, i16 3, i16 1, i16 7, i16 15, i16 63, i16 31, i16 127>
  store <8 x i16> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @sq1428166432128(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %4 = sdiv <8 x i16> %3, <i16 1, i16 4, i16 2, i16 8, i16 16, i16 64, i16 32, i16 128>
  store <8 x i16> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @sr1428166432128(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %4 = srem <8 x i16> %3, <i16 1, i16 4, i16 2, i16 8, i16 16, i16 64, i16 32, i16 128>
  store <8 x i16> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @uq33333333(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %4 = udiv <8 x i16> %3, splat (i16 3)
  store <8 x i16> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @ur33333333(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %4 = urem <8 x i16> %3, splat (i16 3)
  store <8 x i16> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @sq33333333(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %4 = sdiv <8 x i16> %3, splat (i16 3)
  store <8 x i16> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @sr33333333(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %4 = srem <8 x i16> %3, splat (i16 3)
  store <8 x i16> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @uq65656565(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %4 = udiv <8 x i16> %3, <i16 6, i16 5, i16 6, i16 5, i16 6, i16 5, i16 6, i16 5>
  store <8 x i16> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @ur65656565(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %4 = urem <8 x i16> %3, <i16 6, i16 5, i16 6, i16 5, i16 6, i16 5, i16 6, i16 5>
  store <8 x i16> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @sq65656565(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %4 = sdiv <8 x i16> %3, <i16 6, i16 5, i16 6, i16 5, i16 6, i16 5, i16 6, i16 5>
  store <8 x i16> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @sr65656565(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %4 = srem <8 x i16> %3, <i16 6, i16 5, i16 6, i16 5, i16 6, i16 5, i16 6, i16 5>
  store <8 x i16> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @uq14141461461414(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %4 = udiv <8 x i16> %3, <i16 14, i16 14, i16 14, i16 6, i16 14, i16 6, i16 14, i16 14>
  store <8 x i16> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @ur14141461461414(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %4 = urem <8 x i16> %3, <i16 14, i16 14, i16 14, i16 6, i16 14, i16 6, i16 14, i16 14>
  store <8 x i16> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @sq14141461461414(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %4 = sdiv <8 x i16> %3, <i16 14, i16 14, i16 14, i16 6, i16 14, i16 6, i16 14, i16 14>
  store <8 x i16> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @sr14141461461414(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %4 = srem <8 x i16> %3, <i16 14, i16 14, i16 14, i16 6, i16 14, i16 6, i16 14, i16 14>
  store <8 x i16> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @uq77777777(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %4 = udiv <8 x i16> %3, splat (i16 7)
  store <8 x i16> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @ur77777777(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %4 = urem <8 x i16> %3, splat (i16 7)
  store <8 x i16> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @sq77777777(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %4 = sdiv <8 x i16> %3, splat (i16 7)
  store <8 x i16> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @sr77777777(ptr noundef writeonly captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %4 = srem <8 x i16> %3, splat (i16 7)
  store <8 x i16> %4, ptr %0, align 16, !tbaa !6
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = alloca <8 x i16>, align 16
  %2 = alloca <8 x i16>, align 16
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #4
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #4
  br label %3

3:                                                ; preds = %0, %626
  %4 = phi i1 [ true, %0 ], [ false, %626 ]
  %5 = phi i64 [ 0, %0 ], [ 1, %626 ]
  %6 = getelementptr inbounds nuw <8 x i16>, ptr @u, i64 %5
  call void @uq44444444(ptr noundef nonnull %1, ptr noundef nonnull %6)
  %7 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %8 = extractelement <8 x i16> %7, i64 0
  %9 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %10 = extractelement <8 x i16> %9, i64 0
  %11 = lshr i16 %10, 2
  %12 = icmp eq i16 %8, %11
  br i1 %12, label %13, label %18

13:                                               ; preds = %3
  %14 = extractelement <8 x i16> %7, i64 3
  %15 = extractelement <8 x i16> %9, i64 3
  %16 = lshr i16 %15, 2
  %17 = icmp eq i16 %14, %16
  br i1 %17, label %19, label %18

18:                                               ; preds = %13, %3
  call void @abort() #5
  unreachable

19:                                               ; preds = %13
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !9
  %20 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %21 = extractelement <8 x i16> %20, i64 2
  %22 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %23 = extractelement <8 x i16> %22, i64 2
  %24 = lshr i16 %23, 2
  %25 = icmp eq i16 %21, %24
  br i1 %25, label %26, label %31

26:                                               ; preds = %19
  %27 = extractelement <8 x i16> %20, i64 1
  %28 = extractelement <8 x i16> %22, i64 1
  %29 = lshr i16 %28, 2
  %30 = icmp eq i16 %27, %29
  br i1 %30, label %32, label %31

31:                                               ; preds = %26, %19
  call void @abort() #5
  unreachable

32:                                               ; preds = %26
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !10
  %33 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %34 = extractelement <8 x i16> %33, i64 4
  %35 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %36 = extractelement <8 x i16> %35, i64 4
  %37 = lshr i16 %36, 2
  %38 = icmp eq i16 %34, %37
  br i1 %38, label %39, label %44

39:                                               ; preds = %32
  %40 = extractelement <8 x i16> %33, i64 7
  %41 = extractelement <8 x i16> %35, i64 7
  %42 = lshr i16 %41, 2
  %43 = icmp eq i16 %40, %42
  br i1 %43, label %45, label %44

44:                                               ; preds = %39, %32
  call void @abort() #5
  unreachable

45:                                               ; preds = %39
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !11
  %46 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %47 = extractelement <8 x i16> %46, i64 6
  %48 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %49 = extractelement <8 x i16> %48, i64 6
  %50 = lshr i16 %49, 2
  %51 = icmp eq i16 %47, %50
  br i1 %51, label %52, label %57

52:                                               ; preds = %45
  %53 = extractelement <8 x i16> %46, i64 5
  %54 = extractelement <8 x i16> %48, i64 5
  %55 = lshr i16 %54, 2
  %56 = icmp eq i16 %53, %55
  br i1 %56, label %58, label %57

57:                                               ; preds = %52, %45
  call void @abort() #5
  unreachable

58:                                               ; preds = %52
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !12
  call void @ur44444444(ptr noundef nonnull %1, ptr noundef nonnull %6)
  %59 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %60 = extractelement <8 x i16> %59, i64 0
  %61 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %62 = extractelement <8 x i16> %61, i64 0
  %63 = and i16 %62, 3
  %64 = icmp eq i16 %60, %63
  br i1 %64, label %65, label %70

65:                                               ; preds = %58
  %66 = extractelement <8 x i16> %59, i64 3
  %67 = extractelement <8 x i16> %61, i64 3
  %68 = and i16 %67, 3
  %69 = icmp eq i16 %66, %68
  br i1 %69, label %71, label %70

70:                                               ; preds = %65, %58
  call void @abort() #5
  unreachable

71:                                               ; preds = %65
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !13
  %72 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %73 = extractelement <8 x i16> %72, i64 2
  %74 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %75 = extractelement <8 x i16> %74, i64 2
  %76 = and i16 %75, 3
  %77 = icmp eq i16 %73, %76
  br i1 %77, label %78, label %83

78:                                               ; preds = %71
  %79 = extractelement <8 x i16> %72, i64 1
  %80 = extractelement <8 x i16> %74, i64 1
  %81 = and i16 %80, 3
  %82 = icmp eq i16 %79, %81
  br i1 %82, label %84, label %83

83:                                               ; preds = %78, %71
  call void @abort() #5
  unreachable

84:                                               ; preds = %78
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !14
  %85 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %86 = extractelement <8 x i16> %85, i64 4
  %87 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %88 = extractelement <8 x i16> %87, i64 4
  %89 = and i16 %88, 3
  %90 = icmp eq i16 %86, %89
  br i1 %90, label %91, label %96

91:                                               ; preds = %84
  %92 = extractelement <8 x i16> %85, i64 7
  %93 = extractelement <8 x i16> %87, i64 7
  %94 = and i16 %93, 3
  %95 = icmp eq i16 %92, %94
  br i1 %95, label %97, label %96

96:                                               ; preds = %91, %84
  call void @abort() #5
  unreachable

97:                                               ; preds = %91
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !15
  %98 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %99 = extractelement <8 x i16> %98, i64 6
  %100 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %101 = extractelement <8 x i16> %100, i64 6
  %102 = and i16 %101, 3
  %103 = icmp eq i16 %99, %102
  br i1 %103, label %104, label %109

104:                                              ; preds = %97
  %105 = extractelement <8 x i16> %98, i64 5
  %106 = extractelement <8 x i16> %100, i64 5
  %107 = and i16 %106, 3
  %108 = icmp eq i16 %105, %107
  br i1 %108, label %110, label %109

109:                                              ; preds = %104, %97
  call void @abort() #5
  unreachable

110:                                              ; preds = %104
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !16
  call void @uq1428166432128(ptr noundef nonnull %1, ptr noundef nonnull %6)
  %111 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %112 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %113 = icmp eq <8 x i16> %111, %112
  %114 = extractelement <8 x i1> %113, i64 0
  br i1 %114, label %115, label %120

115:                                              ; preds = %110
  %116 = extractelement <8 x i16> %111, i64 3
  %117 = extractelement <8 x i16> %112, i64 3
  %118 = lshr i16 %117, 3
  %119 = icmp eq i16 %116, %118
  br i1 %119, label %121, label %120

120:                                              ; preds = %115, %110
  call void @abort() #5
  unreachable

121:                                              ; preds = %115
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !17
  %122 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %123 = extractelement <8 x i16> %122, i64 2
  %124 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %125 = extractelement <8 x i16> %124, i64 2
  %126 = lshr i16 %125, 1
  %127 = icmp eq i16 %123, %126
  br i1 %127, label %128, label %133

128:                                              ; preds = %121
  %129 = extractelement <8 x i16> %122, i64 1
  %130 = extractelement <8 x i16> %124, i64 1
  %131 = lshr i16 %130, 2
  %132 = icmp eq i16 %129, %131
  br i1 %132, label %134, label %133

133:                                              ; preds = %128, %121
  call void @abort() #5
  unreachable

134:                                              ; preds = %128
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !18
  %135 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %136 = extractelement <8 x i16> %135, i64 4
  %137 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %138 = extractelement <8 x i16> %137, i64 4
  %139 = lshr i16 %138, 4
  %140 = icmp eq i16 %136, %139
  br i1 %140, label %141, label %146

141:                                              ; preds = %134
  %142 = extractelement <8 x i16> %135, i64 7
  %143 = extractelement <8 x i16> %137, i64 7
  %144 = lshr i16 %143, 7
  %145 = icmp eq i16 %142, %144
  br i1 %145, label %147, label %146

146:                                              ; preds = %141, %134
  call void @abort() #5
  unreachable

147:                                              ; preds = %141
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !19
  %148 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %149 = extractelement <8 x i16> %148, i64 6
  %150 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %151 = extractelement <8 x i16> %150, i64 6
  %152 = lshr i16 %151, 5
  %153 = icmp eq i16 %149, %152
  br i1 %153, label %154, label %159

154:                                              ; preds = %147
  %155 = extractelement <8 x i16> %148, i64 5
  %156 = extractelement <8 x i16> %150, i64 5
  %157 = lshr i16 %156, 6
  %158 = icmp eq i16 %155, %157
  br i1 %158, label %160, label %159

159:                                              ; preds = %154, %147
  call void @abort() #5
  unreachable

160:                                              ; preds = %154
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !20
  call void @ur1428166432128(ptr noundef nonnull %1, ptr noundef nonnull %6)
  %161 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %162 = extractelement <8 x i16> %161, i64 0
  %163 = icmp eq i16 %162, 0
  br i1 %163, label %164, label %170

164:                                              ; preds = %160
  %165 = extractelement <8 x i16> %161, i64 3
  %166 = getelementptr inbounds nuw i8, ptr %6, i64 6
  %167 = load i16, ptr %166, align 2, !tbaa !6
  %168 = and i16 %167, 7
  %169 = icmp eq i16 %165, %168
  br i1 %169, label %171, label %170

170:                                              ; preds = %164, %160
  call void @abort() #5
  unreachable

171:                                              ; preds = %164
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !21
  %172 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %173 = extractelement <8 x i16> %172, i64 2
  %174 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %175 = extractelement <8 x i16> %174, i64 2
  %176 = and i16 %175, 1
  %177 = icmp eq i16 %173, %176
  br i1 %177, label %178, label %183

178:                                              ; preds = %171
  %179 = extractelement <8 x i16> %172, i64 1
  %180 = extractelement <8 x i16> %174, i64 1
  %181 = and i16 %180, 3
  %182 = icmp eq i16 %179, %181
  br i1 %182, label %184, label %183

183:                                              ; preds = %178, %171
  call void @abort() #5
  unreachable

184:                                              ; preds = %178
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !22
  %185 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %186 = extractelement <8 x i16> %185, i64 4
  %187 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %188 = extractelement <8 x i16> %187, i64 4
  %189 = and i16 %188, 15
  %190 = icmp eq i16 %186, %189
  br i1 %190, label %191, label %196

191:                                              ; preds = %184
  %192 = extractelement <8 x i16> %185, i64 7
  %193 = extractelement <8 x i16> %187, i64 7
  %194 = and i16 %193, 127
  %195 = icmp eq i16 %192, %194
  br i1 %195, label %197, label %196

196:                                              ; preds = %191, %184
  call void @abort() #5
  unreachable

197:                                              ; preds = %191
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !23
  %198 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %199 = extractelement <8 x i16> %198, i64 6
  %200 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %201 = extractelement <8 x i16> %200, i64 6
  %202 = and i16 %201, 31
  %203 = icmp eq i16 %199, %202
  br i1 %203, label %204, label %209

204:                                              ; preds = %197
  %205 = extractelement <8 x i16> %198, i64 5
  %206 = extractelement <8 x i16> %200, i64 5
  %207 = and i16 %206, 63
  %208 = icmp eq i16 %205, %207
  br i1 %208, label %210, label %209

209:                                              ; preds = %204, %197
  call void @abort() #5
  unreachable

210:                                              ; preds = %204
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !24
  call void @uq33333333(ptr noundef nonnull %1, ptr noundef nonnull %6)
  %211 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %212 = extractelement <8 x i16> %211, i64 0
  %213 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %214 = extractelement <8 x i16> %213, i64 0
  %215 = udiv i16 %214, 3
  %216 = icmp eq i16 %212, %215
  br i1 %216, label %217, label %222

217:                                              ; preds = %210
  %218 = extractelement <8 x i16> %211, i64 3
  %219 = extractelement <8 x i16> %213, i64 3
  %220 = udiv i16 %219, 3
  %221 = icmp eq i16 %218, %220
  br i1 %221, label %223, label %222

222:                                              ; preds = %217, %210
  call void @abort() #5
  unreachable

223:                                              ; preds = %217
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !25
  %224 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %225 = extractelement <8 x i16> %224, i64 2
  %226 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %227 = extractelement <8 x i16> %226, i64 2
  %228 = udiv i16 %227, 3
  %229 = icmp eq i16 %225, %228
  br i1 %229, label %230, label %235

230:                                              ; preds = %223
  %231 = extractelement <8 x i16> %224, i64 1
  %232 = extractelement <8 x i16> %226, i64 1
  %233 = udiv i16 %232, 3
  %234 = icmp eq i16 %231, %233
  br i1 %234, label %236, label %235

235:                                              ; preds = %230, %223
  call void @abort() #5
  unreachable

236:                                              ; preds = %230
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !26
  %237 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %238 = extractelement <8 x i16> %237, i64 4
  %239 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %240 = extractelement <8 x i16> %239, i64 4
  %241 = udiv i16 %240, 3
  %242 = icmp eq i16 %238, %241
  br i1 %242, label %243, label %248

243:                                              ; preds = %236
  %244 = extractelement <8 x i16> %237, i64 7
  %245 = extractelement <8 x i16> %239, i64 7
  %246 = udiv i16 %245, 3
  %247 = icmp eq i16 %244, %246
  br i1 %247, label %249, label %248

248:                                              ; preds = %243, %236
  call void @abort() #5
  unreachable

249:                                              ; preds = %243
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !27
  %250 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %251 = extractelement <8 x i16> %250, i64 6
  %252 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %253 = extractelement <8 x i16> %252, i64 6
  %254 = udiv i16 %253, 3
  %255 = icmp eq i16 %251, %254
  br i1 %255, label %256, label %261

256:                                              ; preds = %249
  %257 = extractelement <8 x i16> %250, i64 5
  %258 = extractelement <8 x i16> %252, i64 5
  %259 = udiv i16 %258, 3
  %260 = icmp eq i16 %257, %259
  br i1 %260, label %262, label %261

261:                                              ; preds = %256, %249
  call void @abort() #5
  unreachable

262:                                              ; preds = %256
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !28
  call void @ur33333333(ptr noundef nonnull %1, ptr noundef nonnull %6)
  %263 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %264 = extractelement <8 x i16> %263, i64 0
  %265 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %266 = extractelement <8 x i16> %265, i64 0
  %267 = urem i16 %266, 3
  %268 = icmp eq i16 %264, %267
  br i1 %268, label %269, label %274

269:                                              ; preds = %262
  %270 = extractelement <8 x i16> %263, i64 3
  %271 = extractelement <8 x i16> %265, i64 3
  %272 = urem i16 %271, 3
  %273 = icmp eq i16 %270, %272
  br i1 %273, label %275, label %274

274:                                              ; preds = %269, %262
  call void @abort() #5
  unreachable

275:                                              ; preds = %269
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !29
  %276 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %277 = extractelement <8 x i16> %276, i64 2
  %278 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %279 = extractelement <8 x i16> %278, i64 2
  %280 = urem i16 %279, 3
  %281 = icmp eq i16 %277, %280
  br i1 %281, label %282, label %287

282:                                              ; preds = %275
  %283 = extractelement <8 x i16> %276, i64 1
  %284 = extractelement <8 x i16> %278, i64 1
  %285 = urem i16 %284, 3
  %286 = icmp eq i16 %283, %285
  br i1 %286, label %288, label %287

287:                                              ; preds = %282, %275
  call void @abort() #5
  unreachable

288:                                              ; preds = %282
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !30
  %289 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %290 = extractelement <8 x i16> %289, i64 4
  %291 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %292 = extractelement <8 x i16> %291, i64 4
  %293 = urem i16 %292, 3
  %294 = icmp eq i16 %290, %293
  br i1 %294, label %295, label %300

295:                                              ; preds = %288
  %296 = extractelement <8 x i16> %289, i64 7
  %297 = extractelement <8 x i16> %291, i64 7
  %298 = urem i16 %297, 3
  %299 = icmp eq i16 %296, %298
  br i1 %299, label %301, label %300

300:                                              ; preds = %295, %288
  call void @abort() #5
  unreachable

301:                                              ; preds = %295
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !31
  %302 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %303 = extractelement <8 x i16> %302, i64 6
  %304 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %305 = extractelement <8 x i16> %304, i64 6
  %306 = urem i16 %305, 3
  %307 = icmp eq i16 %303, %306
  br i1 %307, label %308, label %313

308:                                              ; preds = %301
  %309 = extractelement <8 x i16> %302, i64 5
  %310 = extractelement <8 x i16> %304, i64 5
  %311 = urem i16 %310, 3
  %312 = icmp eq i16 %309, %311
  br i1 %312, label %314, label %313

313:                                              ; preds = %308, %301
  call void @abort() #5
  unreachable

314:                                              ; preds = %308
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !32
  call void @uq65656565(ptr noundef nonnull %1, ptr noundef nonnull %6)
  %315 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %316 = extractelement <8 x i16> %315, i64 0
  %317 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %318 = extractelement <8 x i16> %317, i64 0
  %319 = udiv i16 %318, 6
  %320 = icmp eq i16 %316, %319
  br i1 %320, label %321, label %326

321:                                              ; preds = %314
  %322 = extractelement <8 x i16> %315, i64 3
  %323 = extractelement <8 x i16> %317, i64 3
  %324 = udiv i16 %323, 5
  %325 = icmp eq i16 %322, %324
  br i1 %325, label %327, label %326

326:                                              ; preds = %321, %314
  call void @abort() #5
  unreachable

327:                                              ; preds = %321
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !33
  %328 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %329 = extractelement <8 x i16> %328, i64 2
  %330 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %331 = extractelement <8 x i16> %330, i64 2
  %332 = udiv i16 %331, 6
  %333 = icmp eq i16 %329, %332
  br i1 %333, label %334, label %339

334:                                              ; preds = %327
  %335 = extractelement <8 x i16> %328, i64 1
  %336 = extractelement <8 x i16> %330, i64 1
  %337 = udiv i16 %336, 5
  %338 = icmp eq i16 %335, %337
  br i1 %338, label %340, label %339

339:                                              ; preds = %334, %327
  call void @abort() #5
  unreachable

340:                                              ; preds = %334
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !34
  %341 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %342 = extractelement <8 x i16> %341, i64 4
  %343 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %344 = extractelement <8 x i16> %343, i64 4
  %345 = udiv i16 %344, 6
  %346 = icmp eq i16 %342, %345
  br i1 %346, label %347, label %352

347:                                              ; preds = %340
  %348 = extractelement <8 x i16> %341, i64 7
  %349 = extractelement <8 x i16> %343, i64 7
  %350 = udiv i16 %349, 5
  %351 = icmp eq i16 %348, %350
  br i1 %351, label %353, label %352

352:                                              ; preds = %347, %340
  call void @abort() #5
  unreachable

353:                                              ; preds = %347
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !35
  %354 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %355 = extractelement <8 x i16> %354, i64 6
  %356 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %357 = extractelement <8 x i16> %356, i64 6
  %358 = udiv i16 %357, 6
  %359 = icmp eq i16 %355, %358
  br i1 %359, label %360, label %365

360:                                              ; preds = %353
  %361 = extractelement <8 x i16> %354, i64 5
  %362 = extractelement <8 x i16> %356, i64 5
  %363 = udiv i16 %362, 5
  %364 = icmp eq i16 %361, %363
  br i1 %364, label %366, label %365

365:                                              ; preds = %360, %353
  call void @abort() #5
  unreachable

366:                                              ; preds = %360
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !36
  call void @ur65656565(ptr noundef nonnull %1, ptr noundef nonnull %6)
  %367 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %368 = extractelement <8 x i16> %367, i64 0
  %369 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %370 = extractelement <8 x i16> %369, i64 0
  %371 = urem i16 %370, 6
  %372 = icmp eq i16 %368, %371
  br i1 %372, label %373, label %378

373:                                              ; preds = %366
  %374 = extractelement <8 x i16> %367, i64 3
  %375 = extractelement <8 x i16> %369, i64 3
  %376 = urem i16 %375, 5
  %377 = icmp eq i16 %374, %376
  br i1 %377, label %379, label %378

378:                                              ; preds = %373, %366
  call void @abort() #5
  unreachable

379:                                              ; preds = %373
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !37
  %380 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %381 = extractelement <8 x i16> %380, i64 2
  %382 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %383 = extractelement <8 x i16> %382, i64 2
  %384 = urem i16 %383, 6
  %385 = icmp eq i16 %381, %384
  br i1 %385, label %386, label %391

386:                                              ; preds = %379
  %387 = extractelement <8 x i16> %380, i64 1
  %388 = extractelement <8 x i16> %382, i64 1
  %389 = urem i16 %388, 5
  %390 = icmp eq i16 %387, %389
  br i1 %390, label %392, label %391

391:                                              ; preds = %386, %379
  call void @abort() #5
  unreachable

392:                                              ; preds = %386
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !38
  %393 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %394 = extractelement <8 x i16> %393, i64 4
  %395 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %396 = extractelement <8 x i16> %395, i64 4
  %397 = urem i16 %396, 6
  %398 = icmp eq i16 %394, %397
  br i1 %398, label %399, label %404

399:                                              ; preds = %392
  %400 = extractelement <8 x i16> %393, i64 7
  %401 = extractelement <8 x i16> %395, i64 7
  %402 = urem i16 %401, 5
  %403 = icmp eq i16 %400, %402
  br i1 %403, label %405, label %404

404:                                              ; preds = %399, %392
  call void @abort() #5
  unreachable

405:                                              ; preds = %399
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !39
  %406 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %407 = extractelement <8 x i16> %406, i64 6
  %408 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %409 = extractelement <8 x i16> %408, i64 6
  %410 = urem i16 %409, 6
  %411 = icmp eq i16 %407, %410
  br i1 %411, label %412, label %417

412:                                              ; preds = %405
  %413 = extractelement <8 x i16> %406, i64 5
  %414 = extractelement <8 x i16> %408, i64 5
  %415 = urem i16 %414, 5
  %416 = icmp eq i16 %413, %415
  br i1 %416, label %418, label %417

417:                                              ; preds = %412, %405
  call void @abort() #5
  unreachable

418:                                              ; preds = %412
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !40
  call void @uq14141461461414(ptr noundef nonnull %1, ptr noundef nonnull %6)
  %419 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %420 = extractelement <8 x i16> %419, i64 0
  %421 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %422 = extractelement <8 x i16> %421, i64 0
  %423 = udiv i16 %422, 14
  %424 = icmp eq i16 %420, %423
  br i1 %424, label %425, label %430

425:                                              ; preds = %418
  %426 = extractelement <8 x i16> %419, i64 3
  %427 = extractelement <8 x i16> %421, i64 3
  %428 = udiv i16 %427, 6
  %429 = icmp eq i16 %426, %428
  br i1 %429, label %431, label %430

430:                                              ; preds = %425, %418
  call void @abort() #5
  unreachable

431:                                              ; preds = %425
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !41
  %432 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %433 = extractelement <8 x i16> %432, i64 2
  %434 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %435 = extractelement <8 x i16> %434, i64 2
  %436 = udiv i16 %435, 14
  %437 = icmp eq i16 %433, %436
  br i1 %437, label %438, label %443

438:                                              ; preds = %431
  %439 = extractelement <8 x i16> %432, i64 1
  %440 = extractelement <8 x i16> %434, i64 1
  %441 = udiv i16 %440, 14
  %442 = icmp eq i16 %439, %441
  br i1 %442, label %444, label %443

443:                                              ; preds = %438, %431
  call void @abort() #5
  unreachable

444:                                              ; preds = %438
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !42
  %445 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %446 = extractelement <8 x i16> %445, i64 4
  %447 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %448 = extractelement <8 x i16> %447, i64 4
  %449 = udiv i16 %448, 14
  %450 = icmp eq i16 %446, %449
  br i1 %450, label %451, label %456

451:                                              ; preds = %444
  %452 = extractelement <8 x i16> %445, i64 7
  %453 = extractelement <8 x i16> %447, i64 7
  %454 = udiv i16 %453, 14
  %455 = icmp eq i16 %452, %454
  br i1 %455, label %457, label %456

456:                                              ; preds = %451, %444
  call void @abort() #5
  unreachable

457:                                              ; preds = %451
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !43
  %458 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %459 = extractelement <8 x i16> %458, i64 6
  %460 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %461 = extractelement <8 x i16> %460, i64 6
  %462 = udiv i16 %461, 14
  %463 = icmp eq i16 %459, %462
  br i1 %463, label %464, label %469

464:                                              ; preds = %457
  %465 = extractelement <8 x i16> %458, i64 5
  %466 = extractelement <8 x i16> %460, i64 5
  %467 = udiv i16 %466, 6
  %468 = icmp eq i16 %465, %467
  br i1 %468, label %470, label %469

469:                                              ; preds = %464, %457
  call void @abort() #5
  unreachable

470:                                              ; preds = %464
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !44
  call void @ur14141461461414(ptr noundef nonnull %1, ptr noundef nonnull %6)
  %471 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %472 = extractelement <8 x i16> %471, i64 0
  %473 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %474 = extractelement <8 x i16> %473, i64 0
  %475 = urem i16 %474, 14
  %476 = icmp eq i16 %472, %475
  br i1 %476, label %477, label %482

477:                                              ; preds = %470
  %478 = extractelement <8 x i16> %471, i64 3
  %479 = extractelement <8 x i16> %473, i64 3
  %480 = urem i16 %479, 6
  %481 = icmp eq i16 %478, %480
  br i1 %481, label %483, label %482

482:                                              ; preds = %477, %470
  call void @abort() #5
  unreachable

483:                                              ; preds = %477
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !45
  %484 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %485 = extractelement <8 x i16> %484, i64 2
  %486 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %487 = extractelement <8 x i16> %486, i64 2
  %488 = urem i16 %487, 14
  %489 = icmp eq i16 %485, %488
  br i1 %489, label %490, label %495

490:                                              ; preds = %483
  %491 = extractelement <8 x i16> %484, i64 1
  %492 = extractelement <8 x i16> %486, i64 1
  %493 = urem i16 %492, 14
  %494 = icmp eq i16 %491, %493
  br i1 %494, label %496, label %495

495:                                              ; preds = %490, %483
  call void @abort() #5
  unreachable

496:                                              ; preds = %490
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !46
  %497 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %498 = extractelement <8 x i16> %497, i64 4
  %499 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %500 = extractelement <8 x i16> %499, i64 4
  %501 = urem i16 %500, 14
  %502 = icmp eq i16 %498, %501
  br i1 %502, label %503, label %508

503:                                              ; preds = %496
  %504 = extractelement <8 x i16> %497, i64 7
  %505 = extractelement <8 x i16> %499, i64 7
  %506 = urem i16 %505, 14
  %507 = icmp eq i16 %504, %506
  br i1 %507, label %509, label %508

508:                                              ; preds = %503, %496
  call void @abort() #5
  unreachable

509:                                              ; preds = %503
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !47
  %510 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %511 = extractelement <8 x i16> %510, i64 6
  %512 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %513 = extractelement <8 x i16> %512, i64 6
  %514 = urem i16 %513, 14
  %515 = icmp eq i16 %511, %514
  br i1 %515, label %516, label %521

516:                                              ; preds = %509
  %517 = extractelement <8 x i16> %510, i64 5
  %518 = extractelement <8 x i16> %512, i64 5
  %519 = urem i16 %518, 6
  %520 = icmp eq i16 %517, %519
  br i1 %520, label %522, label %521

521:                                              ; preds = %516, %509
  call void @abort() #5
  unreachable

522:                                              ; preds = %516
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !48
  call void @uq77777777(ptr noundef nonnull %1, ptr noundef nonnull %6)
  %523 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %524 = extractelement <8 x i16> %523, i64 0
  %525 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %526 = extractelement <8 x i16> %525, i64 0
  %527 = udiv i16 %526, 7
  %528 = icmp eq i16 %524, %527
  br i1 %528, label %529, label %534

529:                                              ; preds = %522
  %530 = extractelement <8 x i16> %523, i64 3
  %531 = extractelement <8 x i16> %525, i64 3
  %532 = udiv i16 %531, 7
  %533 = icmp eq i16 %530, %532
  br i1 %533, label %535, label %534

534:                                              ; preds = %529, %522
  call void @abort() #5
  unreachable

535:                                              ; preds = %529
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !49
  %536 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %537 = extractelement <8 x i16> %536, i64 2
  %538 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %539 = extractelement <8 x i16> %538, i64 2
  %540 = udiv i16 %539, 7
  %541 = icmp eq i16 %537, %540
  br i1 %541, label %542, label %547

542:                                              ; preds = %535
  %543 = extractelement <8 x i16> %536, i64 1
  %544 = extractelement <8 x i16> %538, i64 1
  %545 = udiv i16 %544, 7
  %546 = icmp eq i16 %543, %545
  br i1 %546, label %548, label %547

547:                                              ; preds = %542, %535
  call void @abort() #5
  unreachable

548:                                              ; preds = %542
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !50
  %549 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %550 = extractelement <8 x i16> %549, i64 4
  %551 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %552 = extractelement <8 x i16> %551, i64 4
  %553 = udiv i16 %552, 7
  %554 = icmp eq i16 %550, %553
  br i1 %554, label %555, label %560

555:                                              ; preds = %548
  %556 = extractelement <8 x i16> %549, i64 7
  %557 = extractelement <8 x i16> %551, i64 7
  %558 = udiv i16 %557, 7
  %559 = icmp eq i16 %556, %558
  br i1 %559, label %561, label %560

560:                                              ; preds = %555, %548
  call void @abort() #5
  unreachable

561:                                              ; preds = %555
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !51
  %562 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %563 = extractelement <8 x i16> %562, i64 6
  %564 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %565 = extractelement <8 x i16> %564, i64 6
  %566 = udiv i16 %565, 7
  %567 = icmp eq i16 %563, %566
  br i1 %567, label %568, label %573

568:                                              ; preds = %561
  %569 = extractelement <8 x i16> %562, i64 5
  %570 = extractelement <8 x i16> %564, i64 5
  %571 = udiv i16 %570, 7
  %572 = icmp eq i16 %569, %571
  br i1 %572, label %574, label %573

573:                                              ; preds = %568, %561
  call void @abort() #5
  unreachable

574:                                              ; preds = %568
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !52
  call void @ur77777777(ptr noundef nonnull %1, ptr noundef nonnull %6)
  %575 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %576 = extractelement <8 x i16> %575, i64 0
  %577 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %578 = extractelement <8 x i16> %577, i64 0
  %579 = urem i16 %578, 7
  %580 = icmp eq i16 %576, %579
  br i1 %580, label %581, label %586

581:                                              ; preds = %574
  %582 = extractelement <8 x i16> %575, i64 3
  %583 = extractelement <8 x i16> %577, i64 3
  %584 = urem i16 %583, 7
  %585 = icmp eq i16 %582, %584
  br i1 %585, label %587, label %586

586:                                              ; preds = %581, %574
  call void @abort() #5
  unreachable

587:                                              ; preds = %581
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !53
  %588 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %589 = extractelement <8 x i16> %588, i64 2
  %590 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %591 = extractelement <8 x i16> %590, i64 2
  %592 = urem i16 %591, 7
  %593 = icmp eq i16 %589, %592
  br i1 %593, label %594, label %599

594:                                              ; preds = %587
  %595 = extractelement <8 x i16> %588, i64 1
  %596 = extractelement <8 x i16> %590, i64 1
  %597 = urem i16 %596, 7
  %598 = icmp eq i16 %595, %597
  br i1 %598, label %600, label %599

599:                                              ; preds = %594, %587
  call void @abort() #5
  unreachable

600:                                              ; preds = %594
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !54
  %601 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %602 = extractelement <8 x i16> %601, i64 4
  %603 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %604 = extractelement <8 x i16> %603, i64 4
  %605 = urem i16 %604, 7
  %606 = icmp eq i16 %602, %605
  br i1 %606, label %607, label %612

607:                                              ; preds = %600
  %608 = extractelement <8 x i16> %601, i64 7
  %609 = extractelement <8 x i16> %603, i64 7
  %610 = urem i16 %609, 7
  %611 = icmp eq i16 %608, %610
  br i1 %611, label %613, label %612

612:                                              ; preds = %607, %600
  call void @abort() #5
  unreachable

613:                                              ; preds = %607
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !55
  %614 = load <8 x i16>, ptr %1, align 16, !tbaa !6
  %615 = extractelement <8 x i16> %614, i64 6
  %616 = load <8 x i16>, ptr %6, align 16, !tbaa !6
  %617 = extractelement <8 x i16> %616, i64 6
  %618 = urem i16 %617, 7
  %619 = icmp eq i16 %615, %618
  br i1 %619, label %620, label %625

620:                                              ; preds = %613
  %621 = extractelement <8 x i16> %614, i64 5
  %622 = extractelement <8 x i16> %616, i64 5
  %623 = urem i16 %622, 7
  %624 = icmp eq i16 %621, %623
  br i1 %624, label %626, label %625

625:                                              ; preds = %620, %613
  call void @abort() #5
  unreachable

626:                                              ; preds = %620
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #4, !srcloc !56
  br i1 %4, label %3, label %627, !llvm.loop !57

627:                                              ; preds = %626, %1250
  %628 = phi i1 [ false, %1250 ], [ true, %626 ]
  %629 = phi i64 [ 1, %1250 ], [ 0, %626 ]
  %630 = getelementptr inbounds nuw <8 x i16>, ptr @s, i64 %629
  call void @sq44444444(ptr noundef nonnull %2, ptr noundef nonnull %630)
  %631 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %632 = extractelement <8 x i16> %631, i64 0
  %633 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %634 = extractelement <8 x i16> %633, i64 0
  %635 = sdiv i16 %634, 4
  %636 = icmp eq i16 %632, %635
  br i1 %636, label %637, label %642

637:                                              ; preds = %627
  %638 = extractelement <8 x i16> %631, i64 3
  %639 = extractelement <8 x i16> %633, i64 3
  %640 = sdiv i16 %639, 4
  %641 = icmp eq i16 %638, %640
  br i1 %641, label %643, label %642

642:                                              ; preds = %637, %627
  call void @abort() #5
  unreachable

643:                                              ; preds = %637
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !59
  %644 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %645 = extractelement <8 x i16> %644, i64 2
  %646 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %647 = extractelement <8 x i16> %646, i64 2
  %648 = sdiv i16 %647, 4
  %649 = icmp eq i16 %645, %648
  br i1 %649, label %650, label %655

650:                                              ; preds = %643
  %651 = extractelement <8 x i16> %644, i64 1
  %652 = extractelement <8 x i16> %646, i64 1
  %653 = sdiv i16 %652, 4
  %654 = icmp eq i16 %651, %653
  br i1 %654, label %656, label %655

655:                                              ; preds = %650, %643
  call void @abort() #5
  unreachable

656:                                              ; preds = %650
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !60
  %657 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %658 = extractelement <8 x i16> %657, i64 4
  %659 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %660 = extractelement <8 x i16> %659, i64 4
  %661 = sdiv i16 %660, 4
  %662 = icmp eq i16 %658, %661
  br i1 %662, label %663, label %668

663:                                              ; preds = %656
  %664 = extractelement <8 x i16> %657, i64 7
  %665 = extractelement <8 x i16> %659, i64 7
  %666 = sdiv i16 %665, 4
  %667 = icmp eq i16 %664, %666
  br i1 %667, label %669, label %668

668:                                              ; preds = %663, %656
  call void @abort() #5
  unreachable

669:                                              ; preds = %663
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !61
  %670 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %671 = extractelement <8 x i16> %670, i64 6
  %672 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %673 = extractelement <8 x i16> %672, i64 6
  %674 = sdiv i16 %673, 4
  %675 = icmp eq i16 %671, %674
  br i1 %675, label %676, label %681

676:                                              ; preds = %669
  %677 = extractelement <8 x i16> %670, i64 5
  %678 = extractelement <8 x i16> %672, i64 5
  %679 = sdiv i16 %678, 4
  %680 = icmp eq i16 %677, %679
  br i1 %680, label %682, label %681

681:                                              ; preds = %676, %669
  call void @abort() #5
  unreachable

682:                                              ; preds = %676
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !62
  call void @sr44444444(ptr noundef nonnull %2, ptr noundef nonnull %630)
  %683 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %684 = extractelement <8 x i16> %683, i64 0
  %685 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %686 = extractelement <8 x i16> %685, i64 0
  %687 = srem i16 %686, 4
  %688 = icmp eq i16 %687, %684
  br i1 %688, label %689, label %694

689:                                              ; preds = %682
  %690 = extractelement <8 x i16> %683, i64 3
  %691 = extractelement <8 x i16> %685, i64 3
  %692 = srem i16 %691, 4
  %693 = icmp eq i16 %692, %690
  br i1 %693, label %695, label %694

694:                                              ; preds = %689, %682
  call void @abort() #5
  unreachable

695:                                              ; preds = %689
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !63
  %696 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %697 = extractelement <8 x i16> %696, i64 2
  %698 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %699 = extractelement <8 x i16> %698, i64 2
  %700 = srem i16 %699, 4
  %701 = icmp eq i16 %700, %697
  br i1 %701, label %702, label %707

702:                                              ; preds = %695
  %703 = extractelement <8 x i16> %696, i64 1
  %704 = extractelement <8 x i16> %698, i64 1
  %705 = srem i16 %704, 4
  %706 = icmp eq i16 %705, %703
  br i1 %706, label %708, label %707

707:                                              ; preds = %702, %695
  call void @abort() #5
  unreachable

708:                                              ; preds = %702
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !64
  %709 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %710 = extractelement <8 x i16> %709, i64 4
  %711 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %712 = extractelement <8 x i16> %711, i64 4
  %713 = srem i16 %712, 4
  %714 = icmp eq i16 %713, %710
  br i1 %714, label %715, label %720

715:                                              ; preds = %708
  %716 = extractelement <8 x i16> %709, i64 7
  %717 = extractelement <8 x i16> %711, i64 7
  %718 = srem i16 %717, 4
  %719 = icmp eq i16 %718, %716
  br i1 %719, label %721, label %720

720:                                              ; preds = %715, %708
  call void @abort() #5
  unreachable

721:                                              ; preds = %715
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !65
  %722 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %723 = extractelement <8 x i16> %722, i64 6
  %724 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %725 = extractelement <8 x i16> %724, i64 6
  %726 = srem i16 %725, 4
  %727 = icmp eq i16 %726, %723
  br i1 %727, label %728, label %733

728:                                              ; preds = %721
  %729 = extractelement <8 x i16> %722, i64 5
  %730 = extractelement <8 x i16> %724, i64 5
  %731 = srem i16 %730, 4
  %732 = icmp eq i16 %731, %729
  br i1 %732, label %734, label %733

733:                                              ; preds = %728, %721
  call void @abort() #5
  unreachable

734:                                              ; preds = %728
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !66
  call void @sq1428166432128(ptr noundef nonnull %2, ptr noundef nonnull %630)
  %735 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %736 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %737 = icmp eq <8 x i16> %735, %736
  %738 = extractelement <8 x i1> %737, i64 0
  br i1 %738, label %739, label %744

739:                                              ; preds = %734
  %740 = extractelement <8 x i16> %735, i64 3
  %741 = extractelement <8 x i16> %736, i64 3
  %742 = sdiv i16 %741, 8
  %743 = icmp eq i16 %740, %742
  br i1 %743, label %745, label %744

744:                                              ; preds = %739, %734
  call void @abort() #5
  unreachable

745:                                              ; preds = %739
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !67
  %746 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %747 = extractelement <8 x i16> %746, i64 2
  %748 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %749 = extractelement <8 x i16> %748, i64 2
  %750 = sdiv i16 %749, 2
  %751 = icmp eq i16 %747, %750
  br i1 %751, label %752, label %757

752:                                              ; preds = %745
  %753 = extractelement <8 x i16> %746, i64 1
  %754 = extractelement <8 x i16> %748, i64 1
  %755 = sdiv i16 %754, 4
  %756 = icmp eq i16 %753, %755
  br i1 %756, label %758, label %757

757:                                              ; preds = %752, %745
  call void @abort() #5
  unreachable

758:                                              ; preds = %752
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !68
  %759 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %760 = extractelement <8 x i16> %759, i64 4
  %761 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %762 = extractelement <8 x i16> %761, i64 4
  %763 = sdiv i16 %762, 16
  %764 = icmp eq i16 %760, %763
  br i1 %764, label %765, label %770

765:                                              ; preds = %758
  %766 = extractelement <8 x i16> %759, i64 7
  %767 = extractelement <8 x i16> %761, i64 7
  %768 = sdiv i16 %767, 128
  %769 = icmp eq i16 %766, %768
  br i1 %769, label %771, label %770

770:                                              ; preds = %765, %758
  call void @abort() #5
  unreachable

771:                                              ; preds = %765
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !69
  %772 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %773 = extractelement <8 x i16> %772, i64 6
  %774 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %775 = extractelement <8 x i16> %774, i64 6
  %776 = sdiv i16 %775, 32
  %777 = icmp eq i16 %773, %776
  br i1 %777, label %778, label %783

778:                                              ; preds = %771
  %779 = extractelement <8 x i16> %772, i64 5
  %780 = extractelement <8 x i16> %774, i64 5
  %781 = sdiv i16 %780, 64
  %782 = icmp eq i16 %779, %781
  br i1 %782, label %784, label %783

783:                                              ; preds = %778, %771
  call void @abort() #5
  unreachable

784:                                              ; preds = %778
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !70
  call void @sr1428166432128(ptr noundef nonnull %2, ptr noundef nonnull %630)
  %785 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %786 = extractelement <8 x i16> %785, i64 0
  %787 = icmp eq i16 %786, 0
  br i1 %787, label %788, label %794

788:                                              ; preds = %784
  %789 = extractelement <8 x i16> %785, i64 3
  %790 = getelementptr inbounds nuw i8, ptr %630, i64 6
  %791 = load i16, ptr %790, align 2, !tbaa !6
  %792 = srem i16 %791, 8
  %793 = icmp eq i16 %792, %789
  br i1 %793, label %795, label %794

794:                                              ; preds = %788, %784
  call void @abort() #5
  unreachable

795:                                              ; preds = %788
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !71
  %796 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %797 = extractelement <8 x i16> %796, i64 2
  %798 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %799 = extractelement <8 x i16> %798, i64 2
  %800 = srem i16 %799, 2
  %801 = icmp eq i16 %800, %797
  br i1 %801, label %802, label %807

802:                                              ; preds = %795
  %803 = extractelement <8 x i16> %796, i64 1
  %804 = extractelement <8 x i16> %798, i64 1
  %805 = srem i16 %804, 4
  %806 = icmp eq i16 %805, %803
  br i1 %806, label %808, label %807

807:                                              ; preds = %802, %795
  call void @abort() #5
  unreachable

808:                                              ; preds = %802
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !72
  %809 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %810 = extractelement <8 x i16> %809, i64 4
  %811 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %812 = extractelement <8 x i16> %811, i64 4
  %813 = srem i16 %812, 16
  %814 = icmp eq i16 %813, %810
  br i1 %814, label %815, label %820

815:                                              ; preds = %808
  %816 = extractelement <8 x i16> %809, i64 7
  %817 = extractelement <8 x i16> %811, i64 7
  %818 = srem i16 %817, 128
  %819 = icmp eq i16 %818, %816
  br i1 %819, label %821, label %820

820:                                              ; preds = %815, %808
  call void @abort() #5
  unreachable

821:                                              ; preds = %815
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !73
  %822 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %823 = extractelement <8 x i16> %822, i64 6
  %824 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %825 = extractelement <8 x i16> %824, i64 6
  %826 = srem i16 %825, 32
  %827 = icmp eq i16 %826, %823
  br i1 %827, label %828, label %833

828:                                              ; preds = %821
  %829 = extractelement <8 x i16> %822, i64 5
  %830 = extractelement <8 x i16> %824, i64 5
  %831 = srem i16 %830, 64
  %832 = icmp eq i16 %831, %829
  br i1 %832, label %834, label %833

833:                                              ; preds = %828, %821
  call void @abort() #5
  unreachable

834:                                              ; preds = %828
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !74
  call void @sq33333333(ptr noundef nonnull %2, ptr noundef nonnull %630)
  %835 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %836 = extractelement <8 x i16> %835, i64 0
  %837 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %838 = extractelement <8 x i16> %837, i64 0
  %839 = sdiv i16 %838, 3
  %840 = icmp eq i16 %836, %839
  br i1 %840, label %841, label %846

841:                                              ; preds = %834
  %842 = extractelement <8 x i16> %835, i64 3
  %843 = extractelement <8 x i16> %837, i64 3
  %844 = sdiv i16 %843, 3
  %845 = icmp eq i16 %842, %844
  br i1 %845, label %847, label %846

846:                                              ; preds = %841, %834
  call void @abort() #5
  unreachable

847:                                              ; preds = %841
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !75
  %848 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %849 = extractelement <8 x i16> %848, i64 2
  %850 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %851 = extractelement <8 x i16> %850, i64 2
  %852 = sdiv i16 %851, 3
  %853 = icmp eq i16 %849, %852
  br i1 %853, label %854, label %859

854:                                              ; preds = %847
  %855 = extractelement <8 x i16> %848, i64 1
  %856 = extractelement <8 x i16> %850, i64 1
  %857 = sdiv i16 %856, 3
  %858 = icmp eq i16 %855, %857
  br i1 %858, label %860, label %859

859:                                              ; preds = %854, %847
  call void @abort() #5
  unreachable

860:                                              ; preds = %854
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !76
  %861 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %862 = extractelement <8 x i16> %861, i64 4
  %863 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %864 = extractelement <8 x i16> %863, i64 4
  %865 = sdiv i16 %864, 3
  %866 = icmp eq i16 %862, %865
  br i1 %866, label %867, label %872

867:                                              ; preds = %860
  %868 = extractelement <8 x i16> %861, i64 7
  %869 = extractelement <8 x i16> %863, i64 7
  %870 = sdiv i16 %869, 3
  %871 = icmp eq i16 %868, %870
  br i1 %871, label %873, label %872

872:                                              ; preds = %867, %860
  call void @abort() #5
  unreachable

873:                                              ; preds = %867
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !77
  %874 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %875 = extractelement <8 x i16> %874, i64 6
  %876 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %877 = extractelement <8 x i16> %876, i64 6
  %878 = sdiv i16 %877, 3
  %879 = icmp eq i16 %875, %878
  br i1 %879, label %880, label %885

880:                                              ; preds = %873
  %881 = extractelement <8 x i16> %874, i64 5
  %882 = extractelement <8 x i16> %876, i64 5
  %883 = sdiv i16 %882, 3
  %884 = icmp eq i16 %881, %883
  br i1 %884, label %886, label %885

885:                                              ; preds = %880, %873
  call void @abort() #5
  unreachable

886:                                              ; preds = %880
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !78
  call void @sr33333333(ptr noundef nonnull %2, ptr noundef nonnull %630)
  %887 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %888 = extractelement <8 x i16> %887, i64 0
  %889 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %890 = extractelement <8 x i16> %889, i64 0
  %891 = srem i16 %890, 3
  %892 = icmp eq i16 %891, %888
  br i1 %892, label %893, label %898

893:                                              ; preds = %886
  %894 = extractelement <8 x i16> %887, i64 3
  %895 = extractelement <8 x i16> %889, i64 3
  %896 = srem i16 %895, 3
  %897 = icmp eq i16 %896, %894
  br i1 %897, label %899, label %898

898:                                              ; preds = %893, %886
  call void @abort() #5
  unreachable

899:                                              ; preds = %893
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !79
  %900 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %901 = extractelement <8 x i16> %900, i64 2
  %902 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %903 = extractelement <8 x i16> %902, i64 2
  %904 = srem i16 %903, 3
  %905 = icmp eq i16 %904, %901
  br i1 %905, label %906, label %911

906:                                              ; preds = %899
  %907 = extractelement <8 x i16> %900, i64 1
  %908 = extractelement <8 x i16> %902, i64 1
  %909 = srem i16 %908, 3
  %910 = icmp eq i16 %909, %907
  br i1 %910, label %912, label %911

911:                                              ; preds = %906, %899
  call void @abort() #5
  unreachable

912:                                              ; preds = %906
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !80
  %913 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %914 = extractelement <8 x i16> %913, i64 4
  %915 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %916 = extractelement <8 x i16> %915, i64 4
  %917 = srem i16 %916, 3
  %918 = icmp eq i16 %917, %914
  br i1 %918, label %919, label %924

919:                                              ; preds = %912
  %920 = extractelement <8 x i16> %913, i64 7
  %921 = extractelement <8 x i16> %915, i64 7
  %922 = srem i16 %921, 3
  %923 = icmp eq i16 %922, %920
  br i1 %923, label %925, label %924

924:                                              ; preds = %919, %912
  call void @abort() #5
  unreachable

925:                                              ; preds = %919
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !81
  %926 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %927 = extractelement <8 x i16> %926, i64 6
  %928 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %929 = extractelement <8 x i16> %928, i64 6
  %930 = srem i16 %929, 3
  %931 = icmp eq i16 %930, %927
  br i1 %931, label %932, label %937

932:                                              ; preds = %925
  %933 = extractelement <8 x i16> %926, i64 5
  %934 = extractelement <8 x i16> %928, i64 5
  %935 = srem i16 %934, 3
  %936 = icmp eq i16 %935, %933
  br i1 %936, label %938, label %937

937:                                              ; preds = %932, %925
  call void @abort() #5
  unreachable

938:                                              ; preds = %932
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !82
  call void @sq65656565(ptr noundef nonnull %2, ptr noundef nonnull %630)
  %939 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %940 = extractelement <8 x i16> %939, i64 0
  %941 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %942 = extractelement <8 x i16> %941, i64 0
  %943 = sdiv i16 %942, 6
  %944 = icmp eq i16 %940, %943
  br i1 %944, label %945, label %950

945:                                              ; preds = %938
  %946 = extractelement <8 x i16> %939, i64 3
  %947 = extractelement <8 x i16> %941, i64 3
  %948 = sdiv i16 %947, 5
  %949 = icmp eq i16 %946, %948
  br i1 %949, label %951, label %950

950:                                              ; preds = %945, %938
  call void @abort() #5
  unreachable

951:                                              ; preds = %945
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !83
  %952 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %953 = extractelement <8 x i16> %952, i64 2
  %954 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %955 = extractelement <8 x i16> %954, i64 2
  %956 = sdiv i16 %955, 6
  %957 = icmp eq i16 %953, %956
  br i1 %957, label %958, label %963

958:                                              ; preds = %951
  %959 = extractelement <8 x i16> %952, i64 1
  %960 = extractelement <8 x i16> %954, i64 1
  %961 = sdiv i16 %960, 5
  %962 = icmp eq i16 %959, %961
  br i1 %962, label %964, label %963

963:                                              ; preds = %958, %951
  call void @abort() #5
  unreachable

964:                                              ; preds = %958
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !84
  %965 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %966 = extractelement <8 x i16> %965, i64 4
  %967 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %968 = extractelement <8 x i16> %967, i64 4
  %969 = sdiv i16 %968, 6
  %970 = icmp eq i16 %966, %969
  br i1 %970, label %971, label %976

971:                                              ; preds = %964
  %972 = extractelement <8 x i16> %965, i64 7
  %973 = extractelement <8 x i16> %967, i64 7
  %974 = sdiv i16 %973, 5
  %975 = icmp eq i16 %972, %974
  br i1 %975, label %977, label %976

976:                                              ; preds = %971, %964
  call void @abort() #5
  unreachable

977:                                              ; preds = %971
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !85
  %978 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %979 = extractelement <8 x i16> %978, i64 6
  %980 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %981 = extractelement <8 x i16> %980, i64 6
  %982 = sdiv i16 %981, 6
  %983 = icmp eq i16 %979, %982
  br i1 %983, label %984, label %989

984:                                              ; preds = %977
  %985 = extractelement <8 x i16> %978, i64 5
  %986 = extractelement <8 x i16> %980, i64 5
  %987 = sdiv i16 %986, 5
  %988 = icmp eq i16 %985, %987
  br i1 %988, label %990, label %989

989:                                              ; preds = %984, %977
  call void @abort() #5
  unreachable

990:                                              ; preds = %984
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !86
  call void @sr65656565(ptr noundef nonnull %2, ptr noundef nonnull %630)
  %991 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %992 = extractelement <8 x i16> %991, i64 0
  %993 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %994 = extractelement <8 x i16> %993, i64 0
  %995 = srem i16 %994, 6
  %996 = icmp eq i16 %995, %992
  br i1 %996, label %997, label %1002

997:                                              ; preds = %990
  %998 = extractelement <8 x i16> %991, i64 3
  %999 = extractelement <8 x i16> %993, i64 3
  %1000 = srem i16 %999, 5
  %1001 = icmp eq i16 %1000, %998
  br i1 %1001, label %1003, label %1002

1002:                                             ; preds = %997, %990
  call void @abort() #5
  unreachable

1003:                                             ; preds = %997
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !87
  %1004 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %1005 = extractelement <8 x i16> %1004, i64 2
  %1006 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %1007 = extractelement <8 x i16> %1006, i64 2
  %1008 = srem i16 %1007, 6
  %1009 = icmp eq i16 %1008, %1005
  br i1 %1009, label %1010, label %1015

1010:                                             ; preds = %1003
  %1011 = extractelement <8 x i16> %1004, i64 1
  %1012 = extractelement <8 x i16> %1006, i64 1
  %1013 = srem i16 %1012, 5
  %1014 = icmp eq i16 %1013, %1011
  br i1 %1014, label %1016, label %1015

1015:                                             ; preds = %1010, %1003
  call void @abort() #5
  unreachable

1016:                                             ; preds = %1010
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !88
  %1017 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %1018 = extractelement <8 x i16> %1017, i64 4
  %1019 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %1020 = extractelement <8 x i16> %1019, i64 4
  %1021 = srem i16 %1020, 6
  %1022 = icmp eq i16 %1021, %1018
  br i1 %1022, label %1023, label %1028

1023:                                             ; preds = %1016
  %1024 = extractelement <8 x i16> %1017, i64 7
  %1025 = extractelement <8 x i16> %1019, i64 7
  %1026 = srem i16 %1025, 5
  %1027 = icmp eq i16 %1026, %1024
  br i1 %1027, label %1029, label %1028

1028:                                             ; preds = %1023, %1016
  call void @abort() #5
  unreachable

1029:                                             ; preds = %1023
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !89
  %1030 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %1031 = extractelement <8 x i16> %1030, i64 6
  %1032 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %1033 = extractelement <8 x i16> %1032, i64 6
  %1034 = srem i16 %1033, 6
  %1035 = icmp eq i16 %1034, %1031
  br i1 %1035, label %1036, label %1041

1036:                                             ; preds = %1029
  %1037 = extractelement <8 x i16> %1030, i64 5
  %1038 = extractelement <8 x i16> %1032, i64 5
  %1039 = srem i16 %1038, 5
  %1040 = icmp eq i16 %1039, %1037
  br i1 %1040, label %1042, label %1041

1041:                                             ; preds = %1036, %1029
  call void @abort() #5
  unreachable

1042:                                             ; preds = %1036
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !90
  call void @sq14141461461414(ptr noundef nonnull %2, ptr noundef nonnull %630)
  %1043 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %1044 = extractelement <8 x i16> %1043, i64 0
  %1045 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %1046 = extractelement <8 x i16> %1045, i64 0
  %1047 = sdiv i16 %1046, 14
  %1048 = icmp eq i16 %1044, %1047
  br i1 %1048, label %1049, label %1054

1049:                                             ; preds = %1042
  %1050 = extractelement <8 x i16> %1043, i64 3
  %1051 = extractelement <8 x i16> %1045, i64 3
  %1052 = sdiv i16 %1051, 6
  %1053 = icmp eq i16 %1050, %1052
  br i1 %1053, label %1055, label %1054

1054:                                             ; preds = %1049, %1042
  call void @abort() #5
  unreachable

1055:                                             ; preds = %1049
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !91
  %1056 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %1057 = extractelement <8 x i16> %1056, i64 2
  %1058 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %1059 = extractelement <8 x i16> %1058, i64 2
  %1060 = sdiv i16 %1059, 14
  %1061 = icmp eq i16 %1057, %1060
  br i1 %1061, label %1062, label %1067

1062:                                             ; preds = %1055
  %1063 = extractelement <8 x i16> %1056, i64 1
  %1064 = extractelement <8 x i16> %1058, i64 1
  %1065 = sdiv i16 %1064, 14
  %1066 = icmp eq i16 %1063, %1065
  br i1 %1066, label %1068, label %1067

1067:                                             ; preds = %1062, %1055
  call void @abort() #5
  unreachable

1068:                                             ; preds = %1062
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !92
  %1069 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %1070 = extractelement <8 x i16> %1069, i64 4
  %1071 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %1072 = extractelement <8 x i16> %1071, i64 4
  %1073 = sdiv i16 %1072, 14
  %1074 = icmp eq i16 %1070, %1073
  br i1 %1074, label %1075, label %1080

1075:                                             ; preds = %1068
  %1076 = extractelement <8 x i16> %1069, i64 7
  %1077 = extractelement <8 x i16> %1071, i64 7
  %1078 = sdiv i16 %1077, 14
  %1079 = icmp eq i16 %1076, %1078
  br i1 %1079, label %1081, label %1080

1080:                                             ; preds = %1075, %1068
  call void @abort() #5
  unreachable

1081:                                             ; preds = %1075
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !93
  %1082 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %1083 = extractelement <8 x i16> %1082, i64 6
  %1084 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %1085 = extractelement <8 x i16> %1084, i64 6
  %1086 = sdiv i16 %1085, 14
  %1087 = icmp eq i16 %1083, %1086
  br i1 %1087, label %1088, label %1093

1088:                                             ; preds = %1081
  %1089 = extractelement <8 x i16> %1082, i64 5
  %1090 = extractelement <8 x i16> %1084, i64 5
  %1091 = sdiv i16 %1090, 6
  %1092 = icmp eq i16 %1089, %1091
  br i1 %1092, label %1094, label %1093

1093:                                             ; preds = %1088, %1081
  call void @abort() #5
  unreachable

1094:                                             ; preds = %1088
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !94
  call void @sr14141461461414(ptr noundef nonnull %2, ptr noundef nonnull %630)
  %1095 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %1096 = extractelement <8 x i16> %1095, i64 0
  %1097 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %1098 = extractelement <8 x i16> %1097, i64 0
  %1099 = srem i16 %1098, 14
  %1100 = icmp eq i16 %1099, %1096
  br i1 %1100, label %1101, label %1106

1101:                                             ; preds = %1094
  %1102 = extractelement <8 x i16> %1095, i64 3
  %1103 = extractelement <8 x i16> %1097, i64 3
  %1104 = srem i16 %1103, 6
  %1105 = icmp eq i16 %1104, %1102
  br i1 %1105, label %1107, label %1106

1106:                                             ; preds = %1101, %1094
  call void @abort() #5
  unreachable

1107:                                             ; preds = %1101
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !95
  %1108 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %1109 = extractelement <8 x i16> %1108, i64 2
  %1110 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %1111 = extractelement <8 x i16> %1110, i64 2
  %1112 = srem i16 %1111, 14
  %1113 = icmp eq i16 %1112, %1109
  br i1 %1113, label %1114, label %1119

1114:                                             ; preds = %1107
  %1115 = extractelement <8 x i16> %1108, i64 1
  %1116 = extractelement <8 x i16> %1110, i64 1
  %1117 = srem i16 %1116, 14
  %1118 = icmp eq i16 %1117, %1115
  br i1 %1118, label %1120, label %1119

1119:                                             ; preds = %1114, %1107
  call void @abort() #5
  unreachable

1120:                                             ; preds = %1114
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !96
  %1121 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %1122 = extractelement <8 x i16> %1121, i64 4
  %1123 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %1124 = extractelement <8 x i16> %1123, i64 4
  %1125 = srem i16 %1124, 14
  %1126 = icmp eq i16 %1125, %1122
  br i1 %1126, label %1127, label %1132

1127:                                             ; preds = %1120
  %1128 = extractelement <8 x i16> %1121, i64 7
  %1129 = extractelement <8 x i16> %1123, i64 7
  %1130 = srem i16 %1129, 14
  %1131 = icmp eq i16 %1130, %1128
  br i1 %1131, label %1133, label %1132

1132:                                             ; preds = %1127, %1120
  call void @abort() #5
  unreachable

1133:                                             ; preds = %1127
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !97
  %1134 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %1135 = extractelement <8 x i16> %1134, i64 6
  %1136 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %1137 = extractelement <8 x i16> %1136, i64 6
  %1138 = srem i16 %1137, 14
  %1139 = icmp eq i16 %1138, %1135
  br i1 %1139, label %1140, label %1145

1140:                                             ; preds = %1133
  %1141 = extractelement <8 x i16> %1134, i64 5
  %1142 = extractelement <8 x i16> %1136, i64 5
  %1143 = srem i16 %1142, 6
  %1144 = icmp eq i16 %1143, %1141
  br i1 %1144, label %1146, label %1145

1145:                                             ; preds = %1140, %1133
  call void @abort() #5
  unreachable

1146:                                             ; preds = %1140
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !98
  call void @sq77777777(ptr noundef nonnull %2, ptr noundef nonnull %630)
  %1147 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %1148 = extractelement <8 x i16> %1147, i64 0
  %1149 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %1150 = extractelement <8 x i16> %1149, i64 0
  %1151 = sdiv i16 %1150, 7
  %1152 = icmp eq i16 %1148, %1151
  br i1 %1152, label %1153, label %1158

1153:                                             ; preds = %1146
  %1154 = extractelement <8 x i16> %1147, i64 3
  %1155 = extractelement <8 x i16> %1149, i64 3
  %1156 = sdiv i16 %1155, 7
  %1157 = icmp eq i16 %1154, %1156
  br i1 %1157, label %1159, label %1158

1158:                                             ; preds = %1153, %1146
  call void @abort() #5
  unreachable

1159:                                             ; preds = %1153
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !99
  %1160 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %1161 = extractelement <8 x i16> %1160, i64 2
  %1162 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %1163 = extractelement <8 x i16> %1162, i64 2
  %1164 = sdiv i16 %1163, 7
  %1165 = icmp eq i16 %1161, %1164
  br i1 %1165, label %1166, label %1171

1166:                                             ; preds = %1159
  %1167 = extractelement <8 x i16> %1160, i64 1
  %1168 = extractelement <8 x i16> %1162, i64 1
  %1169 = sdiv i16 %1168, 7
  %1170 = icmp eq i16 %1167, %1169
  br i1 %1170, label %1172, label %1171

1171:                                             ; preds = %1166, %1159
  call void @abort() #5
  unreachable

1172:                                             ; preds = %1166
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !100
  %1173 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %1174 = extractelement <8 x i16> %1173, i64 4
  %1175 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %1176 = extractelement <8 x i16> %1175, i64 4
  %1177 = sdiv i16 %1176, 7
  %1178 = icmp eq i16 %1174, %1177
  br i1 %1178, label %1179, label %1184

1179:                                             ; preds = %1172
  %1180 = extractelement <8 x i16> %1173, i64 7
  %1181 = extractelement <8 x i16> %1175, i64 7
  %1182 = sdiv i16 %1181, 7
  %1183 = icmp eq i16 %1180, %1182
  br i1 %1183, label %1185, label %1184

1184:                                             ; preds = %1179, %1172
  call void @abort() #5
  unreachable

1185:                                             ; preds = %1179
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !101
  %1186 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %1187 = extractelement <8 x i16> %1186, i64 6
  %1188 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %1189 = extractelement <8 x i16> %1188, i64 6
  %1190 = sdiv i16 %1189, 7
  %1191 = icmp eq i16 %1187, %1190
  br i1 %1191, label %1192, label %1197

1192:                                             ; preds = %1185
  %1193 = extractelement <8 x i16> %1186, i64 5
  %1194 = extractelement <8 x i16> %1188, i64 5
  %1195 = sdiv i16 %1194, 7
  %1196 = icmp eq i16 %1193, %1195
  br i1 %1196, label %1198, label %1197

1197:                                             ; preds = %1192, %1185
  call void @abort() #5
  unreachable

1198:                                             ; preds = %1192
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !102
  call void @sr77777777(ptr noundef nonnull %2, ptr noundef nonnull %630)
  %1199 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %1200 = extractelement <8 x i16> %1199, i64 0
  %1201 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %1202 = extractelement <8 x i16> %1201, i64 0
  %1203 = srem i16 %1202, 7
  %1204 = icmp eq i16 %1203, %1200
  br i1 %1204, label %1205, label %1210

1205:                                             ; preds = %1198
  %1206 = extractelement <8 x i16> %1199, i64 3
  %1207 = extractelement <8 x i16> %1201, i64 3
  %1208 = srem i16 %1207, 7
  %1209 = icmp eq i16 %1208, %1206
  br i1 %1209, label %1211, label %1210

1210:                                             ; preds = %1205, %1198
  call void @abort() #5
  unreachable

1211:                                             ; preds = %1205
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !103
  %1212 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %1213 = extractelement <8 x i16> %1212, i64 2
  %1214 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %1215 = extractelement <8 x i16> %1214, i64 2
  %1216 = srem i16 %1215, 7
  %1217 = icmp eq i16 %1216, %1213
  br i1 %1217, label %1218, label %1223

1218:                                             ; preds = %1211
  %1219 = extractelement <8 x i16> %1212, i64 1
  %1220 = extractelement <8 x i16> %1214, i64 1
  %1221 = srem i16 %1220, 7
  %1222 = icmp eq i16 %1221, %1219
  br i1 %1222, label %1224, label %1223

1223:                                             ; preds = %1218, %1211
  call void @abort() #5
  unreachable

1224:                                             ; preds = %1218
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !104
  %1225 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %1226 = extractelement <8 x i16> %1225, i64 4
  %1227 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %1228 = extractelement <8 x i16> %1227, i64 4
  %1229 = srem i16 %1228, 7
  %1230 = icmp eq i16 %1229, %1226
  br i1 %1230, label %1231, label %1236

1231:                                             ; preds = %1224
  %1232 = extractelement <8 x i16> %1225, i64 7
  %1233 = extractelement <8 x i16> %1227, i64 7
  %1234 = srem i16 %1233, 7
  %1235 = icmp eq i16 %1234, %1232
  br i1 %1235, label %1237, label %1236

1236:                                             ; preds = %1231, %1224
  call void @abort() #5
  unreachable

1237:                                             ; preds = %1231
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !105
  %1238 = load <8 x i16>, ptr %2, align 16, !tbaa !6
  %1239 = extractelement <8 x i16> %1238, i64 6
  %1240 = load <8 x i16>, ptr %630, align 16, !tbaa !6
  %1241 = extractelement <8 x i16> %1240, i64 6
  %1242 = srem i16 %1241, 7
  %1243 = icmp eq i16 %1242, %1239
  br i1 %1243, label %1244, label %1249

1244:                                             ; preds = %1237
  %1245 = extractelement <8 x i16> %1238, i64 5
  %1246 = extractelement <8 x i16> %1240, i64 5
  %1247 = srem i16 %1246, 7
  %1248 = icmp eq i16 %1247, %1245
  br i1 %1248, label %1250, label %1249

1249:                                             ; preds = %1244, %1237
  call void @abort() #5
  unreachable

1250:                                             ; preds = %1244
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %2) #4, !srcloc !106
  br i1 %628, label %627, label %1251, !llvm.loop !107

1251:                                             ; preds = %1250
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
!9 = !{i64 2147511316}
!10 = !{i64 2147511445}
!11 = !{i64 2147511574}
!12 = !{i64 2147511703}
!13 = !{i64 2147511880}
!14 = !{i64 2147512009}
!15 = !{i64 2147512138}
!16 = !{i64 2147512267}
!17 = !{i64 2147512523}
!18 = !{i64 2147512652}
!19 = !{i64 2147512781}
!20 = !{i64 2147512910}
!21 = !{i64 2147513087}
!22 = !{i64 2147513216}
!23 = !{i64 2147513345}
!24 = !{i64 2147513474}
!25 = !{i64 2147513760}
!26 = !{i64 2147513889}
!27 = !{i64 2147514018}
!28 = !{i64 2147514147}
!29 = !{i64 2147514324}
!30 = !{i64 2147514453}
!31 = !{i64 2147514582}
!32 = !{i64 2147514711}
!33 = !{i64 2147514967}
!34 = !{i64 2147515096}
!35 = !{i64 2147515225}
!36 = !{i64 2147515354}
!37 = !{i64 2147515531}
!38 = !{i64 2147515660}
!39 = !{i64 2147515789}
!40 = !{i64 2147515918}
!41 = !{i64 2147516174}
!42 = !{i64 2147516303}
!43 = !{i64 2147516432}
!44 = !{i64 2147516561}
!45 = !{i64 2147516738}
!46 = !{i64 2147516867}
!47 = !{i64 2147516996}
!48 = !{i64 2147517125}
!49 = !{i64 2147517417}
!50 = !{i64 2147517546}
!51 = !{i64 2147517675}
!52 = !{i64 2147517804}
!53 = !{i64 2147517981}
!54 = !{i64 2147518110}
!55 = !{i64 2147518239}
!56 = !{i64 2147518368}
!57 = distinct !{!57, !58}
!58 = !{!"llvm.loop.mustprogress"}
!59 = !{i64 2147518830}
!60 = !{i64 2147518959}
!61 = !{i64 2147519088}
!62 = !{i64 2147519217}
!63 = !{i64 2147519394}
!64 = !{i64 2147519523}
!65 = !{i64 2147519652}
!66 = !{i64 2147519781}
!67 = !{i64 2147520037}
!68 = !{i64 2147520166}
!69 = !{i64 2147520295}
!70 = !{i64 2147520424}
!71 = !{i64 2147520601}
!72 = !{i64 2147520730}
!73 = !{i64 2147520859}
!74 = !{i64 2147520988}
!75 = !{i64 2147521274}
!76 = !{i64 2147521403}
!77 = !{i64 2147521532}
!78 = !{i64 2147521661}
!79 = !{i64 2147521838}
!80 = !{i64 2147521967}
!81 = !{i64 2147522096}
!82 = !{i64 2147522225}
!83 = !{i64 2147522481}
!84 = !{i64 2147522610}
!85 = !{i64 2147522739}
!86 = !{i64 2147522868}
!87 = !{i64 2147523045}
!88 = !{i64 2147523174}
!89 = !{i64 2147523303}
!90 = !{i64 2147523432}
!91 = !{i64 2147523688}
!92 = !{i64 2147523817}
!93 = !{i64 2147523946}
!94 = !{i64 2147524075}
!95 = !{i64 2147524252}
!96 = !{i64 2147524381}
!97 = !{i64 2147524510}
!98 = !{i64 2147524639}
!99 = !{i64 2147524931}
!100 = !{i64 2147525060}
!101 = !{i64 2147525189}
!102 = !{i64 2147525318}
!103 = !{i64 2147525495}
!104 = !{i64 2147525624}
!105 = !{i64 2147525753}
!106 = !{i64 2147525882}
!107 = distinct !{!107, !58}
