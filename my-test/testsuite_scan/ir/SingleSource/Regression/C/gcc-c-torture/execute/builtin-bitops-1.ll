; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtin-bitops-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtin-bitops-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@ints = dso_local local_unnamed_addr global [13 x i32] [i32 0, i32 1, i32 -2147483648, i32 2, i32 1073741824, i32 65536, i32 32768, i32 -1515870811, i32 1515870810, i32 -889323520, i32 13303296, i32 51966, i32 -1], align 4
@longs = dso_local local_unnamed_addr global [13 x i64] [i64 0, i64 1, i64 -9223372036854775808, i64 2, i64 4611686018427387904, i64 4294967296, i64 2147483648, i64 -6510615555426900571, i64 6510615555426900570, i64 -3819392241693097984, i64 223195676147712, i64 3405695742, i64 -1], align 8
@longlongs = dso_local local_unnamed_addr global [13 x i64] [i64 0, i64 1, i64 -9223372036854775808, i64 2, i64 4611686018427387904, i64 4294967296, i64 2147483648, i64 -6510615555426900571, i64 6510615555426900570, i64 -3819392241693097984, i64 223195676147712, i64 3405695742, i64 -1], align 8

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local range(i32 0, 33) i32 @my_ffs(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp eq i32 %0, 0
  br i1 %2, label %97, label %3

3:                                                ; preds = %1
  %4 = and i32 %0, 1
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %6, label %97

6:                                                ; preds = %3
  %7 = and i32 %0, 2
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %9, label %97

9:                                                ; preds = %6
  %10 = and i32 %0, 4
  %11 = icmp eq i32 %10, 0
  br i1 %11, label %12, label %97

12:                                               ; preds = %9
  %13 = and i32 %0, 8
  %14 = icmp eq i32 %13, 0
  br i1 %14, label %15, label %97

15:                                               ; preds = %12
  %16 = and i32 %0, 16
  %17 = icmp eq i32 %16, 0
  br i1 %17, label %18, label %97

18:                                               ; preds = %15
  %19 = and i32 %0, 32
  %20 = icmp eq i32 %19, 0
  br i1 %20, label %21, label %97

21:                                               ; preds = %18
  %22 = and i32 %0, 64
  %23 = icmp eq i32 %22, 0
  br i1 %23, label %24, label %97

24:                                               ; preds = %21
  %25 = and i32 %0, 128
  %26 = icmp eq i32 %25, 0
  br i1 %26, label %27, label %97

27:                                               ; preds = %24
  %28 = and i32 %0, 256
  %29 = icmp eq i32 %28, 0
  br i1 %29, label %30, label %97

30:                                               ; preds = %27
  %31 = and i32 %0, 512
  %32 = icmp eq i32 %31, 0
  br i1 %32, label %33, label %97

33:                                               ; preds = %30
  %34 = and i32 %0, 1024
  %35 = icmp eq i32 %34, 0
  br i1 %35, label %36, label %97

36:                                               ; preds = %33
  %37 = and i32 %0, 2048
  %38 = icmp eq i32 %37, 0
  br i1 %38, label %39, label %97

39:                                               ; preds = %36
  %40 = and i32 %0, 4096
  %41 = icmp eq i32 %40, 0
  br i1 %41, label %42, label %97

42:                                               ; preds = %39
  %43 = and i32 %0, 8192
  %44 = icmp eq i32 %43, 0
  br i1 %44, label %45, label %97

45:                                               ; preds = %42
  %46 = and i32 %0, 16384
  %47 = icmp eq i32 %46, 0
  br i1 %47, label %48, label %97

48:                                               ; preds = %45
  %49 = and i32 %0, 32768
  %50 = icmp eq i32 %49, 0
  br i1 %50, label %51, label %97

51:                                               ; preds = %48
  %52 = and i32 %0, 65536
  %53 = icmp eq i32 %52, 0
  br i1 %53, label %54, label %97

54:                                               ; preds = %51
  %55 = and i32 %0, 131072
  %56 = icmp eq i32 %55, 0
  br i1 %56, label %57, label %97

57:                                               ; preds = %54
  %58 = and i32 %0, 262144
  %59 = icmp eq i32 %58, 0
  br i1 %59, label %60, label %97

60:                                               ; preds = %57
  %61 = and i32 %0, 524288
  %62 = icmp eq i32 %61, 0
  br i1 %62, label %63, label %97

63:                                               ; preds = %60
  %64 = and i32 %0, 1048576
  %65 = icmp eq i32 %64, 0
  br i1 %65, label %66, label %97

66:                                               ; preds = %63
  %67 = and i32 %0, 2097152
  %68 = icmp eq i32 %67, 0
  br i1 %68, label %69, label %97

69:                                               ; preds = %66
  %70 = and i32 %0, 4194304
  %71 = icmp eq i32 %70, 0
  br i1 %71, label %72, label %97

72:                                               ; preds = %69
  %73 = and i32 %0, 8388608
  %74 = icmp eq i32 %73, 0
  br i1 %74, label %75, label %97

75:                                               ; preds = %72
  %76 = and i32 %0, 16777216
  %77 = icmp eq i32 %76, 0
  br i1 %77, label %78, label %97

78:                                               ; preds = %75
  %79 = and i32 %0, 33554432
  %80 = icmp eq i32 %79, 0
  br i1 %80, label %81, label %97

81:                                               ; preds = %78
  %82 = and i32 %0, 67108864
  %83 = icmp eq i32 %82, 0
  br i1 %83, label %84, label %97

84:                                               ; preds = %81
  %85 = and i32 %0, 134217728
  %86 = icmp eq i32 %85, 0
  br i1 %86, label %87, label %97

87:                                               ; preds = %84
  %88 = and i32 %0, 268435456
  %89 = icmp eq i32 %88, 0
  br i1 %89, label %90, label %97

90:                                               ; preds = %87
  %91 = and i32 %0, 536870912
  %92 = icmp eq i32 %91, 0
  br i1 %92, label %93, label %97

93:                                               ; preds = %90
  %94 = and i32 %0, 1073741824
  %95 = icmp eq i32 %94, 0
  %96 = select i1 %95, i32 32, i32 31
  br label %97

97:                                               ; preds = %93, %3, %6, %9, %12, %15, %18, %21, %24, %27, %30, %33, %36, %39, %42, %45, %48, %51, %54, %57, %60, %63, %66, %69, %72, %75, %78, %81, %84, %87, %90, %1
  %98 = phi i32 [ 0, %1 ], [ 1, %3 ], [ 2, %6 ], [ 3, %9 ], [ 4, %12 ], [ 5, %15 ], [ 6, %18 ], [ 7, %21 ], [ 8, %24 ], [ 9, %27 ], [ 10, %30 ], [ 11, %33 ], [ 12, %36 ], [ 13, %39 ], [ 14, %42 ], [ 15, %45 ], [ 16, %48 ], [ 17, %51 ], [ 18, %54 ], [ 19, %57 ], [ 20, %60 ], [ 21, %63 ], [ 22, %66 ], [ 23, %69 ], [ 24, %72 ], [ 25, %75 ], [ 26, %78 ], [ 27, %81 ], [ 28, %84 ], [ 29, %87 ], [ 30, %90 ], [ %96, %93 ]
  ret i32 %98
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local range(i32 0, 33) i32 @my_ctz(i32 noundef %0) local_unnamed_addr #0 {
  %2 = and i32 %0, 1
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %4, label %97

4:                                                ; preds = %1
  %5 = and i32 %0, 2
  %6 = icmp eq i32 %5, 0
  br i1 %6, label %7, label %97

7:                                                ; preds = %4
  %8 = and i32 %0, 4
  %9 = icmp eq i32 %8, 0
  br i1 %9, label %10, label %97

10:                                               ; preds = %7
  %11 = and i32 %0, 8
  %12 = icmp eq i32 %11, 0
  br i1 %12, label %13, label %97

13:                                               ; preds = %10
  %14 = and i32 %0, 16
  %15 = icmp eq i32 %14, 0
  br i1 %15, label %16, label %97

16:                                               ; preds = %13
  %17 = and i32 %0, 32
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %19, label %97

19:                                               ; preds = %16
  %20 = and i32 %0, 64
  %21 = icmp eq i32 %20, 0
  br i1 %21, label %22, label %97

22:                                               ; preds = %19
  %23 = and i32 %0, 128
  %24 = icmp eq i32 %23, 0
  br i1 %24, label %25, label %97

25:                                               ; preds = %22
  %26 = and i32 %0, 256
  %27 = icmp eq i32 %26, 0
  br i1 %27, label %28, label %97

28:                                               ; preds = %25
  %29 = and i32 %0, 512
  %30 = icmp eq i32 %29, 0
  br i1 %30, label %31, label %97

31:                                               ; preds = %28
  %32 = and i32 %0, 1024
  %33 = icmp eq i32 %32, 0
  br i1 %33, label %34, label %97

34:                                               ; preds = %31
  %35 = and i32 %0, 2048
  %36 = icmp eq i32 %35, 0
  br i1 %36, label %37, label %97

37:                                               ; preds = %34
  %38 = and i32 %0, 4096
  %39 = icmp eq i32 %38, 0
  br i1 %39, label %40, label %97

40:                                               ; preds = %37
  %41 = and i32 %0, 8192
  %42 = icmp eq i32 %41, 0
  br i1 %42, label %43, label %97

43:                                               ; preds = %40
  %44 = and i32 %0, 16384
  %45 = icmp eq i32 %44, 0
  br i1 %45, label %46, label %97

46:                                               ; preds = %43
  %47 = and i32 %0, 32768
  %48 = icmp eq i32 %47, 0
  br i1 %48, label %49, label %97

49:                                               ; preds = %46
  %50 = and i32 %0, 65536
  %51 = icmp eq i32 %50, 0
  br i1 %51, label %52, label %97

52:                                               ; preds = %49
  %53 = and i32 %0, 131072
  %54 = icmp eq i32 %53, 0
  br i1 %54, label %55, label %97

55:                                               ; preds = %52
  %56 = and i32 %0, 262144
  %57 = icmp eq i32 %56, 0
  br i1 %57, label %58, label %97

58:                                               ; preds = %55
  %59 = and i32 %0, 524288
  %60 = icmp eq i32 %59, 0
  br i1 %60, label %61, label %97

61:                                               ; preds = %58
  %62 = and i32 %0, 1048576
  %63 = icmp eq i32 %62, 0
  br i1 %63, label %64, label %97

64:                                               ; preds = %61
  %65 = and i32 %0, 2097152
  %66 = icmp eq i32 %65, 0
  br i1 %66, label %67, label %97

67:                                               ; preds = %64
  %68 = and i32 %0, 4194304
  %69 = icmp eq i32 %68, 0
  br i1 %69, label %70, label %97

70:                                               ; preds = %67
  %71 = and i32 %0, 8388608
  %72 = icmp eq i32 %71, 0
  br i1 %72, label %73, label %97

73:                                               ; preds = %70
  %74 = and i32 %0, 16777216
  %75 = icmp eq i32 %74, 0
  br i1 %75, label %76, label %97

76:                                               ; preds = %73
  %77 = and i32 %0, 33554432
  %78 = icmp eq i32 %77, 0
  br i1 %78, label %79, label %97

79:                                               ; preds = %76
  %80 = and i32 %0, 67108864
  %81 = icmp eq i32 %80, 0
  br i1 %81, label %82, label %97

82:                                               ; preds = %79
  %83 = and i32 %0, 134217728
  %84 = icmp eq i32 %83, 0
  br i1 %84, label %85, label %97

85:                                               ; preds = %82
  %86 = and i32 %0, 268435456
  %87 = icmp eq i32 %86, 0
  br i1 %87, label %88, label %97

88:                                               ; preds = %85
  %89 = and i32 %0, 536870912
  %90 = icmp eq i32 %89, 0
  br i1 %90, label %91, label %97

91:                                               ; preds = %88
  %92 = and i32 %0, 1073741824
  %93 = icmp eq i32 %92, 0
  br i1 %93, label %94, label %97

94:                                               ; preds = %91
  %95 = icmp eq i32 %0, 0
  %96 = select i1 %95, i32 32, i32 31
  br label %97

97:                                               ; preds = %94, %91, %88, %85, %82, %79, %76, %73, %70, %67, %64, %61, %58, %55, %52, %49, %46, %43, %40, %37, %34, %31, %28, %25, %22, %19, %16, %13, %10, %7, %4, %1
  %98 = phi i32 [ 0, %1 ], [ 1, %4 ], [ 2, %7 ], [ 3, %10 ], [ 4, %13 ], [ 5, %16 ], [ 6, %19 ], [ 7, %22 ], [ 8, %25 ], [ 9, %28 ], [ 10, %31 ], [ 11, %34 ], [ 12, %37 ], [ 13, %40 ], [ 14, %43 ], [ 15, %46 ], [ 16, %49 ], [ 17, %52 ], [ 18, %55 ], [ 19, %58 ], [ 20, %61 ], [ 21, %64 ], [ 22, %67 ], [ 23, %70 ], [ 24, %73 ], [ 25, %76 ], [ 26, %79 ], [ 27, %82 ], [ 28, %85 ], [ 29, %88 ], [ 30, %91 ], [ %96, %94 ]
  ret i32 %98
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local range(i32 0, 33) i32 @my_clz(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp sgt i32 %0, -1
  br i1 %2, label %3, label %66

3:                                                ; preds = %1
  %4 = icmp samesign ult i32 %0, 1073741824
  br i1 %4, label %5, label %66

5:                                                ; preds = %3
  %6 = icmp samesign ult i32 %0, 536870912
  br i1 %6, label %7, label %66

7:                                                ; preds = %5
  %8 = icmp samesign ult i32 %0, 268435456
  br i1 %8, label %9, label %66

9:                                                ; preds = %7
  %10 = icmp samesign ult i32 %0, 134217728
  br i1 %10, label %11, label %66

11:                                               ; preds = %9
  %12 = icmp samesign ult i32 %0, 67108864
  br i1 %12, label %13, label %66

13:                                               ; preds = %11
  %14 = icmp samesign ult i32 %0, 33554432
  br i1 %14, label %15, label %66

15:                                               ; preds = %13
  %16 = icmp samesign ult i32 %0, 16777216
  br i1 %16, label %17, label %66

17:                                               ; preds = %15
  %18 = icmp samesign ult i32 %0, 8388608
  br i1 %18, label %19, label %66

19:                                               ; preds = %17
  %20 = icmp samesign ult i32 %0, 4194304
  br i1 %20, label %21, label %66

21:                                               ; preds = %19
  %22 = icmp samesign ult i32 %0, 2097152
  br i1 %22, label %23, label %66

23:                                               ; preds = %21
  %24 = icmp samesign ult i32 %0, 1048576
  br i1 %24, label %25, label %66

25:                                               ; preds = %23
  %26 = icmp samesign ult i32 %0, 524288
  br i1 %26, label %27, label %66

27:                                               ; preds = %25
  %28 = icmp samesign ult i32 %0, 262144
  br i1 %28, label %29, label %66

29:                                               ; preds = %27
  %30 = icmp samesign ult i32 %0, 131072
  br i1 %30, label %31, label %66

31:                                               ; preds = %29
  %32 = icmp samesign ult i32 %0, 65536
  br i1 %32, label %33, label %66

33:                                               ; preds = %31
  %34 = icmp samesign ult i32 %0, 32768
  br i1 %34, label %35, label %66

35:                                               ; preds = %33
  %36 = icmp samesign ult i32 %0, 16384
  br i1 %36, label %37, label %66

37:                                               ; preds = %35
  %38 = icmp samesign ult i32 %0, 8192
  br i1 %38, label %39, label %66

39:                                               ; preds = %37
  %40 = icmp samesign ult i32 %0, 4096
  br i1 %40, label %41, label %66

41:                                               ; preds = %39
  %42 = icmp samesign ult i32 %0, 2048
  br i1 %42, label %43, label %66

43:                                               ; preds = %41
  %44 = icmp samesign ult i32 %0, 1024
  br i1 %44, label %45, label %66

45:                                               ; preds = %43
  %46 = icmp samesign ult i32 %0, 512
  br i1 %46, label %47, label %66

47:                                               ; preds = %45
  %48 = icmp samesign ult i32 %0, 256
  br i1 %48, label %49, label %66

49:                                               ; preds = %47
  %50 = icmp samesign ult i32 %0, 128
  br i1 %50, label %51, label %66

51:                                               ; preds = %49
  %52 = icmp samesign ult i32 %0, 64
  br i1 %52, label %53, label %66

53:                                               ; preds = %51
  %54 = icmp samesign ult i32 %0, 32
  br i1 %54, label %55, label %66

55:                                               ; preds = %53
  %56 = icmp samesign ult i32 %0, 16
  br i1 %56, label %57, label %66

57:                                               ; preds = %55
  %58 = icmp samesign ult i32 %0, 8
  br i1 %58, label %59, label %66

59:                                               ; preds = %57
  %60 = icmp samesign ult i32 %0, 4
  br i1 %60, label %61, label %66

61:                                               ; preds = %59
  %62 = icmp samesign ult i32 %0, 2
  br i1 %62, label %63, label %66

63:                                               ; preds = %61
  %64 = icmp eq i32 %0, 0
  %65 = select i1 %64, i32 32, i32 31
  br label %66

66:                                               ; preds = %63, %61, %59, %57, %55, %53, %51, %49, %47, %45, %43, %41, %39, %37, %35, %33, %31, %29, %27, %25, %23, %21, %19, %17, %15, %13, %11, %9, %7, %5, %3, %1
  %67 = phi i32 [ 0, %1 ], [ 1, %3 ], [ 2, %5 ], [ 3, %7 ], [ 4, %9 ], [ 5, %11 ], [ 6, %13 ], [ 7, %15 ], [ 8, %17 ], [ 9, %19 ], [ 10, %21 ], [ 11, %23 ], [ 12, %25 ], [ 13, %27 ], [ 14, %29 ], [ 15, %31 ], [ 16, %33 ], [ 17, %35 ], [ 18, %37 ], [ 19, %39 ], [ 20, %41 ], [ 21, %43 ], [ 22, %45 ], [ 23, %47 ], [ 24, %49 ], [ 25, %51 ], [ 26, %53 ], [ 27, %55 ], [ 28, %57 ], [ 29, %59 ], [ 30, %61 ], [ %65, %63 ]
  ret i32 %67
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local range(i32 0, 32) i32 @my_clrsb(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 31
  %3 = lshr i32 %0, 30
  %4 = and i32 %3, 1
  %5 = icmp eq i32 %4, %2
  br i1 %5, label %6, label %126

6:                                                ; preds = %1
  %7 = lshr i32 %0, 29
  %8 = and i32 %7, 1
  %9 = icmp eq i32 %8, %2
  br i1 %9, label %10, label %126

10:                                               ; preds = %6
  %11 = lshr i32 %0, 28
  %12 = and i32 %11, 1
  %13 = icmp eq i32 %12, %2
  br i1 %13, label %14, label %126

14:                                               ; preds = %10
  %15 = lshr i32 %0, 27
  %16 = and i32 %15, 1
  %17 = icmp eq i32 %16, %2
  br i1 %17, label %18, label %126

18:                                               ; preds = %14
  %19 = lshr i32 %0, 26
  %20 = and i32 %19, 1
  %21 = icmp eq i32 %20, %2
  br i1 %21, label %22, label %126

22:                                               ; preds = %18
  %23 = lshr i32 %0, 25
  %24 = and i32 %23, 1
  %25 = icmp eq i32 %24, %2
  br i1 %25, label %26, label %126

26:                                               ; preds = %22
  %27 = lshr i32 %0, 24
  %28 = and i32 %27, 1
  %29 = icmp eq i32 %28, %2
  br i1 %29, label %30, label %126

30:                                               ; preds = %26
  %31 = lshr i32 %0, 23
  %32 = and i32 %31, 1
  %33 = icmp eq i32 %32, %2
  br i1 %33, label %34, label %126

34:                                               ; preds = %30
  %35 = lshr i32 %0, 22
  %36 = and i32 %35, 1
  %37 = icmp eq i32 %36, %2
  br i1 %37, label %38, label %126

38:                                               ; preds = %34
  %39 = lshr i32 %0, 21
  %40 = and i32 %39, 1
  %41 = icmp eq i32 %40, %2
  br i1 %41, label %42, label %126

42:                                               ; preds = %38
  %43 = lshr i32 %0, 20
  %44 = and i32 %43, 1
  %45 = icmp eq i32 %44, %2
  br i1 %45, label %46, label %126

46:                                               ; preds = %42
  %47 = lshr i32 %0, 19
  %48 = and i32 %47, 1
  %49 = icmp eq i32 %48, %2
  br i1 %49, label %50, label %126

50:                                               ; preds = %46
  %51 = lshr i32 %0, 18
  %52 = and i32 %51, 1
  %53 = icmp eq i32 %52, %2
  br i1 %53, label %54, label %126

54:                                               ; preds = %50
  %55 = lshr i32 %0, 17
  %56 = and i32 %55, 1
  %57 = icmp eq i32 %56, %2
  br i1 %57, label %58, label %126

58:                                               ; preds = %54
  %59 = lshr i32 %0, 16
  %60 = and i32 %59, 1
  %61 = icmp eq i32 %60, %2
  br i1 %61, label %62, label %126

62:                                               ; preds = %58
  %63 = lshr i32 %0, 15
  %64 = and i32 %63, 1
  %65 = icmp eq i32 %64, %2
  br i1 %65, label %66, label %126

66:                                               ; preds = %62
  %67 = lshr i32 %0, 14
  %68 = and i32 %67, 1
  %69 = icmp eq i32 %68, %2
  br i1 %69, label %70, label %126

70:                                               ; preds = %66
  %71 = lshr i32 %0, 13
  %72 = and i32 %71, 1
  %73 = icmp eq i32 %72, %2
  br i1 %73, label %74, label %126

74:                                               ; preds = %70
  %75 = lshr i32 %0, 12
  %76 = and i32 %75, 1
  %77 = icmp eq i32 %76, %2
  br i1 %77, label %78, label %126

78:                                               ; preds = %74
  %79 = lshr i32 %0, 11
  %80 = and i32 %79, 1
  %81 = icmp eq i32 %80, %2
  br i1 %81, label %82, label %126

82:                                               ; preds = %78
  %83 = lshr i32 %0, 10
  %84 = and i32 %83, 1
  %85 = icmp eq i32 %84, %2
  br i1 %85, label %86, label %126

86:                                               ; preds = %82
  %87 = lshr i32 %0, 9
  %88 = and i32 %87, 1
  %89 = icmp eq i32 %88, %2
  br i1 %89, label %90, label %126

90:                                               ; preds = %86
  %91 = lshr i32 %0, 8
  %92 = and i32 %91, 1
  %93 = icmp eq i32 %92, %2
  br i1 %93, label %94, label %126

94:                                               ; preds = %90
  %95 = lshr i32 %0, 7
  %96 = and i32 %95, 1
  %97 = icmp eq i32 %96, %2
  br i1 %97, label %98, label %126

98:                                               ; preds = %94
  %99 = lshr i32 %0, 6
  %100 = and i32 %99, 1
  %101 = icmp eq i32 %100, %2
  br i1 %101, label %102, label %126

102:                                              ; preds = %98
  %103 = lshr i32 %0, 5
  %104 = and i32 %103, 1
  %105 = icmp eq i32 %104, %2
  br i1 %105, label %106, label %126

106:                                              ; preds = %102
  %107 = lshr i32 %0, 4
  %108 = and i32 %107, 1
  %109 = icmp eq i32 %108, %2
  br i1 %109, label %110, label %126

110:                                              ; preds = %106
  %111 = lshr i32 %0, 3
  %112 = and i32 %111, 1
  %113 = icmp eq i32 %112, %2
  br i1 %113, label %114, label %126

114:                                              ; preds = %110
  %115 = lshr i32 %0, 2
  %116 = and i32 %115, 1
  %117 = icmp eq i32 %116, %2
  br i1 %117, label %118, label %126

118:                                              ; preds = %114
  %119 = lshr i32 %0, 1
  %120 = and i32 %119, 1
  %121 = icmp eq i32 %120, %2
  br i1 %121, label %122, label %126

122:                                              ; preds = %118
  %123 = and i32 %0, 1
  %124 = icmp eq i32 %123, %2
  %125 = select i1 %124, i32 31, i32 30
  br label %126

126:                                              ; preds = %122, %118, %114, %110, %106, %102, %98, %94, %90, %86, %82, %78, %74, %70, %66, %62, %58, %54, %50, %46, %42, %38, %34, %30, %26, %22, %18, %14, %10, %6, %1
  %127 = phi i32 [ 0, %1 ], [ 1, %6 ], [ 2, %10 ], [ 3, %14 ], [ 4, %18 ], [ 5, %22 ], [ 6, %26 ], [ 7, %30 ], [ 8, %34 ], [ 9, %38 ], [ 10, %42 ], [ 11, %46 ], [ 12, %50 ], [ 13, %54 ], [ 14, %58 ], [ 15, %62 ], [ 16, %66 ], [ 17, %70 ], [ 18, %74 ], [ 19, %78 ], [ 20, %82 ], [ 21, %86 ], [ 22, %90 ], [ 23, %94 ], [ 24, %98 ], [ 25, %102 ], [ 26, %106 ], [ 27, %110 ], [ 28, %114 ], [ 29, %118 ], [ %125, %122 ]
  ret i32 %127
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local range(i32 0, 33) i32 @my_popcount(i32 noundef %0) local_unnamed_addr #0 {
  %2 = insertelement <28 x i32> poison, i32 %0, i64 0
  %3 = shufflevector <28 x i32> %2, <28 x i32> poison, <28 x i32> zeroinitializer
  %4 = lshr <28 x i32> %3, <i32 1, i32 0, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
  %5 = and <28 x i32> %4, splat (i32 1)
  %6 = lshr i32 %0, 28
  %7 = and i32 %6, 1
  %8 = lshr i32 %0, 29
  %9 = and i32 %8, 1
  %10 = lshr i32 %0, 30
  %11 = and i32 %10, 1
  %12 = lshr i32 %0, 31
  %13 = tail call i32 @llvm.vector.reduce.add.v28i32(<28 x i32> %5)
  %14 = add i32 %13, %7
  %15 = add nuw nsw i32 %9, %11
  %16 = add i32 %14, %15
  %17 = add i32 %16, %12
  ret i32 %17
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local range(i32 0, 2) i32 @my_parity(i32 noundef %0) local_unnamed_addr #0 {
  %2 = insertelement <28 x i32> poison, i32 %0, i64 0
  %3 = shufflevector <28 x i32> %2, <28 x i32> poison, <28 x i32> zeroinitializer
  %4 = lshr <28 x i32> %3, <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
  %5 = lshr i32 %0, 29
  %6 = lshr i32 %0, 30
  %7 = lshr i32 %0, 31
  %8 = tail call i32 @llvm.vector.reduce.add.v28i32(<28 x i32> %4)
  %9 = add i32 %8, %5
  %10 = add nuw nsw i32 %6, %7
  %11 = add i32 %9, %10
  %12 = add i32 %11, %0
  %13 = and i32 %12, 1
  ret i32 %13
}

; Function Attrs: nofree norecurse nosync nounwind memory(none) uwtable
define dso_local i32 @my_ffsl(i64 noundef %0) local_unnamed_addr #1 {
  %2 = icmp eq i64 %0, 0
  br i1 %2, label %14, label %3

3:                                                ; preds = %1, %8
  %4 = phi i64 [ %9, %8 ], [ 0, %1 ]
  %5 = shl nuw i64 1, %4
  %6 = and i64 %5, %0
  %7 = icmp eq i64 %6, 0
  br i1 %7, label %8, label %11

8:                                                ; preds = %3
  %9 = add nuw nsw i64 %4, 1
  %10 = icmp eq i64 %9, 64
  br i1 %10, label %14, label %3, !llvm.loop !6

11:                                               ; preds = %3
  %12 = trunc nuw nsw i64 %4 to i32
  %13 = add nuw nsw i32 %12, 1
  br label %14

14:                                               ; preds = %8, %11, %1
  %15 = phi i32 [ 0, %1 ], [ %13, %11 ], [ 65, %8 ]
  ret i32 %15
}

; Function Attrs: nofree norecurse nosync nounwind memory(none) uwtable
define dso_local i32 @my_ctzl(i64 noundef %0) local_unnamed_addr #1 {
  br label %2

2:                                                ; preds = %1, %7
  %3 = phi i64 [ 0, %1 ], [ %8, %7 ]
  %4 = shl nuw i64 1, %3
  %5 = and i64 %4, %0
  %6 = icmp eq i64 %5, 0
  br i1 %6, label %7, label %10

7:                                                ; preds = %2
  %8 = add nuw nsw i64 %3, 1
  %9 = icmp eq i64 %8, 64
  br i1 %9, label %12, label %2, !llvm.loop !8

10:                                               ; preds = %2
  %11 = trunc nuw nsw i64 %3 to i32
  br label %12

12:                                               ; preds = %7, %10
  %13 = phi i32 [ %11, %10 ], [ 64, %7 ]
  ret i32 %13
}

; Function Attrs: nofree norecurse nosync nounwind memory(none) uwtable
define dso_local i32 @my_clzl(i64 noundef %0) local_unnamed_addr #1 {
  br label %2

2:                                                ; preds = %1, %7
  %3 = phi i64 [ 0, %1 ], [ %8, %7 ]
  %4 = lshr exact i64 -9223372036854775808, %3
  %5 = and i64 %4, %0
  %6 = icmp eq i64 %5, 0
  br i1 %6, label %7, label %10

7:                                                ; preds = %2
  %8 = add nuw nsw i64 %3, 1
  %9 = icmp eq i64 %8, 64
  br i1 %9, label %12, label %2, !llvm.loop !9

10:                                               ; preds = %2
  %11 = trunc nuw nsw i64 %3 to i32
  br label %12

12:                                               ; preds = %7, %10
  %13 = phi i32 [ %11, %10 ], [ 64, %7 ]
  ret i32 %13
}

; Function Attrs: nofree norecurse nosync nounwind memory(none) uwtable
define dso_local range(i32 0, -1) i32 @my_clrsbl(i64 noundef %0) local_unnamed_addr #1 {
  %2 = lshr i64 %0, 63
  br label %3

3:                                                ; preds = %1, %9
  %4 = phi i64 [ 1, %1 ], [ %10, %9 ]
  %5 = sub nsw i64 63, %4
  %6 = lshr i64 %0, %5
  %7 = and i64 %6, 1
  %8 = icmp eq i64 %7, %2
  br i1 %8, label %9, label %12

9:                                                ; preds = %3
  %10 = add nuw nsw i64 %4, 1
  %11 = icmp eq i64 %10, 64
  br i1 %11, label %15, label %3, !llvm.loop !10

12:                                               ; preds = %3
  %13 = trunc nuw nsw i64 %4 to i32
  %14 = add nsw i32 %13, -1
  br label %15

15:                                               ; preds = %9, %12
  %16 = phi i32 [ %14, %12 ], [ 63, %9 ]
  ret i32 %16
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local range(i32 0, 65) i32 @my_popcountl(i64 noundef %0) local_unnamed_addr #0 {
  %2 = trunc i64 %0 to i32
  %3 = lshr i32 %2, 31
  %4 = insertelement <2 x i64> poison, i64 %0, i64 0
  %5 = shufflevector <2 x i64> %4, <2 x i64> poison, <2 x i32> zeroinitializer
  %6 = lshr <2 x i64> %5, <i64 0, i64 32>
  %7 = shufflevector <2 x i64> %6, <2 x i64> poison, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1>
  %8 = trunc <32 x i64> %7 to <32 x i32>
  %9 = lshr <32 x i32> %8, <i32 1, i32 0, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 0>
  %10 = and <32 x i32> %9, splat (i32 1)
  %11 = insertelement <16 x i64> poison, i64 %0, i64 0
  %12 = shufflevector <16 x i64> %11, <16 x i64> poison, <16 x i32> zeroinitializer
  %13 = lshr <16 x i64> %12, <i64 33, i64 34, i64 35, i64 36, i64 37, i64 38, i64 39, i64 40, i64 41, i64 42, i64 43, i64 44, i64 45, i64 46, i64 47, i64 48>
  %14 = trunc nuw <16 x i64> %13 to <16 x i32>
  %15 = and <16 x i32> %14, splat (i32 1)
  %16 = insertelement <8 x i64> poison, i64 %0, i64 0
  %17 = shufflevector <8 x i64> %16, <8 x i64> poison, <8 x i32> zeroinitializer
  %18 = lshr <8 x i64> %17, <i64 49, i64 50, i64 51, i64 52, i64 53, i64 54, i64 55, i64 56>
  %19 = trunc nuw nsw <8 x i64> %18 to <8 x i32>
  %20 = and <8 x i32> %19, splat (i32 1)
  %21 = insertelement <4 x i64> poison, i64 %0, i64 0
  %22 = shufflevector <4 x i64> %21, <4 x i64> poison, <4 x i32> zeroinitializer
  %23 = lshr <4 x i64> %22, <i64 57, i64 58, i64 59, i64 60>
  %24 = trunc nuw nsw <4 x i64> %23 to <4 x i32>
  %25 = and <4 x i32> %24, splat (i32 1)
  %26 = lshr i64 %0, 61
  %27 = trunc nuw nsw i64 %26 to i32
  %28 = and i32 %27, 1
  %29 = lshr i64 %0, 62
  %30 = trunc nuw nsw i64 %29 to i32
  %31 = and i32 %30, 1
  %32 = lshr i64 %0, 63
  %33 = trunc nuw nsw i64 %32 to i32
  %34 = shufflevector <32 x i32> %10, <32 x i32> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %35 = add nuw nsw <16 x i32> %34, %15
  %36 = shufflevector <16 x i32> %35, <16 x i32> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %37 = shufflevector <32 x i32> %36, <32 x i32> %10, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %38 = shufflevector <16 x i32> %35, <16 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %39 = add nuw nsw <8 x i32> %38, %20
  %40 = shufflevector <8 x i32> %39, <8 x i32> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %41 = shufflevector <32 x i32> %40, <32 x i32> %37, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 4, i32 5, i32 6, i32 7, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %42 = shufflevector <8 x i32> %39, <8 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %43 = add nuw nsw <4 x i32> %42, %25
  %44 = shufflevector <4 x i32> %43, <4 x i32> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %45 = shufflevector <32 x i32> %44, <32 x i32> %41, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %46 = tail call i32 @llvm.vector.reduce.add.v32i32(<32 x i32> %45)
  %47 = add i32 %46, %28
  %48 = add nuw nsw i32 %31, %33
  %49 = add i32 %47, %48
  %50 = add i32 %49, %3
  ret i32 %50
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local range(i32 0, 2) i32 @my_parityl(i64 noundef %0) local_unnamed_addr #0 {
  %2 = trunc i64 %0 to i32
  %3 = insertelement <32 x i64> poison, i64 %0, i64 0
  %4 = shufflevector <32 x i64> %3, <32 x i64> poison, <32 x i32> zeroinitializer
  %5 = lshr <32 x i64> %4, <i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9, i64 10, i64 11, i64 12, i64 13, i64 14, i64 15, i64 16, i64 17, i64 18, i64 19, i64 20, i64 21, i64 22, i64 23, i64 24, i64 25, i64 26, i64 27, i64 28, i64 29, i64 30, i64 31, i64 32>
  %6 = trunc <32 x i64> %5 to <32 x i32>
  %7 = insertelement <16 x i64> poison, i64 %0, i64 0
  %8 = shufflevector <16 x i64> %7, <16 x i64> poison, <16 x i32> zeroinitializer
  %9 = lshr <16 x i64> %8, <i64 33, i64 34, i64 35, i64 36, i64 37, i64 38, i64 39, i64 40, i64 41, i64 42, i64 43, i64 44, i64 45, i64 46, i64 47, i64 48>
  %10 = trunc nuw <16 x i64> %9 to <16 x i32>
  %11 = insertelement <8 x i64> poison, i64 %0, i64 0
  %12 = shufflevector <8 x i64> %11, <8 x i64> poison, <8 x i32> zeroinitializer
  %13 = lshr <8 x i64> %12, <i64 49, i64 50, i64 51, i64 52, i64 53, i64 54, i64 55, i64 56>
  %14 = trunc nuw nsw <8 x i64> %13 to <8 x i32>
  %15 = insertelement <4 x i64> poison, i64 %0, i64 0
  %16 = shufflevector <4 x i64> %15, <4 x i64> poison, <4 x i32> zeroinitializer
  %17 = lshr <4 x i64> %16, <i64 57, i64 58, i64 59, i64 60>
  %18 = trunc nuw nsw <4 x i64> %17 to <4 x i32>
  %19 = lshr i64 %0, 61
  %20 = trunc nuw nsw i64 %19 to i32
  %21 = lshr i64 %0, 62
  %22 = trunc nuw nsw i64 %21 to i32
  %23 = lshr i64 %0, 63
  %24 = trunc nuw nsw i64 %23 to i32
  %25 = shufflevector <32 x i32> %6, <32 x i32> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %26 = add <16 x i32> %25, %10
  %27 = shufflevector <16 x i32> %26, <16 x i32> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %28 = shufflevector <32 x i32> %27, <32 x i32> %6, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %29 = shufflevector <16 x i32> %26, <16 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %30 = add <8 x i32> %29, %14
  %31 = shufflevector <8 x i32> %30, <8 x i32> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %32 = shufflevector <32 x i32> %31, <32 x i32> %28, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 4, i32 5, i32 6, i32 7, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %33 = shufflevector <8 x i32> %30, <8 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %34 = add <4 x i32> %33, %18
  %35 = shufflevector <4 x i32> %34, <4 x i32> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %36 = shufflevector <32 x i32> %35, <32 x i32> %32, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %37 = tail call i32 @llvm.vector.reduce.add.v32i32(<32 x i32> %36)
  %38 = add i32 %37, %20
  %39 = add nuw nsw i32 %22, %24
  %40 = add i32 %38, %39
  %41 = add i32 %40, %2
  %42 = and i32 %41, 1
  ret i32 %42
}

; Function Attrs: nofree norecurse nosync nounwind memory(none) uwtable
define dso_local i32 @my_ffsll(i64 noundef %0) local_unnamed_addr #1 {
  %2 = icmp eq i64 %0, 0
  br i1 %2, label %14, label %3

3:                                                ; preds = %1, %8
  %4 = phi i64 [ %9, %8 ], [ 0, %1 ]
  %5 = shl nuw i64 1, %4
  %6 = and i64 %5, %0
  %7 = icmp eq i64 %6, 0
  br i1 %7, label %8, label %11

8:                                                ; preds = %3
  %9 = add nuw nsw i64 %4, 1
  %10 = icmp eq i64 %9, 64
  br i1 %10, label %14, label %3, !llvm.loop !11

11:                                               ; preds = %3
  %12 = trunc nuw nsw i64 %4 to i32
  %13 = add nuw nsw i32 %12, 1
  br label %14

14:                                               ; preds = %8, %11, %1
  %15 = phi i32 [ 0, %1 ], [ %13, %11 ], [ 65, %8 ]
  ret i32 %15
}

; Function Attrs: nofree norecurse nosync nounwind memory(none) uwtable
define dso_local i32 @my_ctzll(i64 noundef %0) local_unnamed_addr #1 {
  br label %2

2:                                                ; preds = %1, %7
  %3 = phi i64 [ 0, %1 ], [ %8, %7 ]
  %4 = shl nuw i64 1, %3
  %5 = and i64 %4, %0
  %6 = icmp eq i64 %5, 0
  br i1 %6, label %7, label %10

7:                                                ; preds = %2
  %8 = add nuw nsw i64 %3, 1
  %9 = icmp eq i64 %8, 64
  br i1 %9, label %12, label %2, !llvm.loop !12

10:                                               ; preds = %2
  %11 = trunc nuw nsw i64 %3 to i32
  br label %12

12:                                               ; preds = %7, %10
  %13 = phi i32 [ %11, %10 ], [ 64, %7 ]
  ret i32 %13
}

; Function Attrs: nofree norecurse nosync nounwind memory(none) uwtable
define dso_local i32 @my_clzll(i64 noundef %0) local_unnamed_addr #1 {
  br label %2

2:                                                ; preds = %1, %7
  %3 = phi i64 [ 0, %1 ], [ %8, %7 ]
  %4 = lshr exact i64 -9223372036854775808, %3
  %5 = and i64 %4, %0
  %6 = icmp eq i64 %5, 0
  br i1 %6, label %7, label %10

7:                                                ; preds = %2
  %8 = add nuw nsw i64 %3, 1
  %9 = icmp eq i64 %8, 64
  br i1 %9, label %12, label %2, !llvm.loop !13

10:                                               ; preds = %2
  %11 = trunc nuw nsw i64 %3 to i32
  br label %12

12:                                               ; preds = %7, %10
  %13 = phi i32 [ %11, %10 ], [ 64, %7 ]
  ret i32 %13
}

; Function Attrs: nofree norecurse nosync nounwind memory(none) uwtable
define dso_local range(i32 0, -1) i32 @my_clrsbll(i64 noundef %0) local_unnamed_addr #1 {
  %2 = lshr i64 %0, 63
  br label %3

3:                                                ; preds = %1, %9
  %4 = phi i64 [ 1, %1 ], [ %10, %9 ]
  %5 = sub nsw i64 63, %4
  %6 = lshr i64 %0, %5
  %7 = and i64 %6, 1
  %8 = icmp eq i64 %7, %2
  br i1 %8, label %9, label %12

9:                                                ; preds = %3
  %10 = add nuw nsw i64 %4, 1
  %11 = icmp eq i64 %10, 64
  br i1 %11, label %15, label %3, !llvm.loop !14

12:                                               ; preds = %3
  %13 = trunc nuw nsw i64 %4 to i32
  %14 = add nsw i32 %13, -1
  br label %15

15:                                               ; preds = %9, %12
  %16 = phi i32 [ %14, %12 ], [ 63, %9 ]
  ret i32 %16
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local range(i32 0, 65) i32 @my_popcountll(i64 noundef %0) local_unnamed_addr #0 {
  %2 = trunc i64 %0 to i32
  %3 = lshr i32 %2, 31
  %4 = insertelement <2 x i64> poison, i64 %0, i64 0
  %5 = shufflevector <2 x i64> %4, <2 x i64> poison, <2 x i32> zeroinitializer
  %6 = lshr <2 x i64> %5, <i64 0, i64 32>
  %7 = shufflevector <2 x i64> %6, <2 x i64> poison, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1>
  %8 = trunc <32 x i64> %7 to <32 x i32>
  %9 = lshr <32 x i32> %8, <i32 1, i32 0, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 0>
  %10 = and <32 x i32> %9, splat (i32 1)
  %11 = insertelement <16 x i64> poison, i64 %0, i64 0
  %12 = shufflevector <16 x i64> %11, <16 x i64> poison, <16 x i32> zeroinitializer
  %13 = lshr <16 x i64> %12, <i64 33, i64 34, i64 35, i64 36, i64 37, i64 38, i64 39, i64 40, i64 41, i64 42, i64 43, i64 44, i64 45, i64 46, i64 47, i64 48>
  %14 = trunc nuw <16 x i64> %13 to <16 x i32>
  %15 = and <16 x i32> %14, splat (i32 1)
  %16 = insertelement <8 x i64> poison, i64 %0, i64 0
  %17 = shufflevector <8 x i64> %16, <8 x i64> poison, <8 x i32> zeroinitializer
  %18 = lshr <8 x i64> %17, <i64 49, i64 50, i64 51, i64 52, i64 53, i64 54, i64 55, i64 56>
  %19 = trunc nuw nsw <8 x i64> %18 to <8 x i32>
  %20 = and <8 x i32> %19, splat (i32 1)
  %21 = insertelement <4 x i64> poison, i64 %0, i64 0
  %22 = shufflevector <4 x i64> %21, <4 x i64> poison, <4 x i32> zeroinitializer
  %23 = lshr <4 x i64> %22, <i64 57, i64 58, i64 59, i64 60>
  %24 = trunc nuw nsw <4 x i64> %23 to <4 x i32>
  %25 = and <4 x i32> %24, splat (i32 1)
  %26 = lshr i64 %0, 61
  %27 = trunc nuw nsw i64 %26 to i32
  %28 = and i32 %27, 1
  %29 = lshr i64 %0, 62
  %30 = trunc nuw nsw i64 %29 to i32
  %31 = and i32 %30, 1
  %32 = lshr i64 %0, 63
  %33 = trunc nuw nsw i64 %32 to i32
  %34 = shufflevector <32 x i32> %10, <32 x i32> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %35 = add nuw nsw <16 x i32> %34, %15
  %36 = shufflevector <16 x i32> %35, <16 x i32> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %37 = shufflevector <32 x i32> %36, <32 x i32> %10, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %38 = shufflevector <16 x i32> %35, <16 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %39 = add nuw nsw <8 x i32> %38, %20
  %40 = shufflevector <8 x i32> %39, <8 x i32> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %41 = shufflevector <32 x i32> %40, <32 x i32> %37, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 4, i32 5, i32 6, i32 7, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %42 = shufflevector <8 x i32> %39, <8 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %43 = add nuw nsw <4 x i32> %42, %25
  %44 = shufflevector <4 x i32> %43, <4 x i32> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %45 = shufflevector <32 x i32> %44, <32 x i32> %41, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %46 = tail call i32 @llvm.vector.reduce.add.v32i32(<32 x i32> %45)
  %47 = add i32 %46, %28
  %48 = add nuw nsw i32 %31, %33
  %49 = add i32 %47, %48
  %50 = add i32 %49, %3
  ret i32 %50
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local range(i32 0, 2) i32 @my_parityll(i64 noundef %0) local_unnamed_addr #0 {
  %2 = trunc i64 %0 to i32
  %3 = insertelement <32 x i64> poison, i64 %0, i64 0
  %4 = shufflevector <32 x i64> %3, <32 x i64> poison, <32 x i32> zeroinitializer
  %5 = lshr <32 x i64> %4, <i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9, i64 10, i64 11, i64 12, i64 13, i64 14, i64 15, i64 16, i64 17, i64 18, i64 19, i64 20, i64 21, i64 22, i64 23, i64 24, i64 25, i64 26, i64 27, i64 28, i64 29, i64 30, i64 31, i64 32>
  %6 = trunc <32 x i64> %5 to <32 x i32>
  %7 = insertelement <16 x i64> poison, i64 %0, i64 0
  %8 = shufflevector <16 x i64> %7, <16 x i64> poison, <16 x i32> zeroinitializer
  %9 = lshr <16 x i64> %8, <i64 33, i64 34, i64 35, i64 36, i64 37, i64 38, i64 39, i64 40, i64 41, i64 42, i64 43, i64 44, i64 45, i64 46, i64 47, i64 48>
  %10 = trunc nuw <16 x i64> %9 to <16 x i32>
  %11 = insertelement <8 x i64> poison, i64 %0, i64 0
  %12 = shufflevector <8 x i64> %11, <8 x i64> poison, <8 x i32> zeroinitializer
  %13 = lshr <8 x i64> %12, <i64 49, i64 50, i64 51, i64 52, i64 53, i64 54, i64 55, i64 56>
  %14 = trunc nuw nsw <8 x i64> %13 to <8 x i32>
  %15 = insertelement <4 x i64> poison, i64 %0, i64 0
  %16 = shufflevector <4 x i64> %15, <4 x i64> poison, <4 x i32> zeroinitializer
  %17 = lshr <4 x i64> %16, <i64 57, i64 58, i64 59, i64 60>
  %18 = trunc nuw nsw <4 x i64> %17 to <4 x i32>
  %19 = lshr i64 %0, 61
  %20 = trunc nuw nsw i64 %19 to i32
  %21 = lshr i64 %0, 62
  %22 = trunc nuw nsw i64 %21 to i32
  %23 = lshr i64 %0, 63
  %24 = trunc nuw nsw i64 %23 to i32
  %25 = shufflevector <32 x i32> %6, <32 x i32> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %26 = add <16 x i32> %25, %10
  %27 = shufflevector <16 x i32> %26, <16 x i32> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %28 = shufflevector <32 x i32> %27, <32 x i32> %6, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %29 = shufflevector <16 x i32> %26, <16 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %30 = add <8 x i32> %29, %14
  %31 = shufflevector <8 x i32> %30, <8 x i32> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %32 = shufflevector <32 x i32> %31, <32 x i32> %28, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 4, i32 5, i32 6, i32 7, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %33 = shufflevector <8 x i32> %30, <8 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %34 = add <4 x i32> %33, %18
  %35 = shufflevector <4 x i32> %34, <4 x i32> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %36 = shufflevector <32 x i32> %35, <32 x i32> %32, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %37 = tail call i32 @llvm.vector.reduce.add.v32i32(<32 x i32> %36)
  %38 = add i32 %37, %20
  %39 = add nuw nsw i32 %22, %24
  %40 = add i32 %38, %39
  %41 = add i32 %40, %2
  %42 = and i32 %41, 1
  ret i32 %42
}

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  br label %4

1:                                                ; preds = %38
  %2 = add nuw nsw i64 %5, 1
  %3 = icmp eq i64 %2, 13
  br i1 %3, label %57, label %4, !llvm.loop !15

4:                                                ; preds = %0, %1
  %5 = phi i64 [ 0, %0 ], [ %2, %1 ]
  %6 = getelementptr inbounds nuw i32, ptr @ints, i64 %5
  %7 = load i32, ptr %6, align 4, !tbaa !16
  %8 = tail call range(i32 0, 33) i32 @llvm.cttz.i32(i32 %7, i1 true)
  %9 = add nuw nsw i32 %8, 1
  %10 = icmp eq i32 %7, 0
  %11 = select i1 %10, i32 0, i32 %9
  %12 = tail call i32 @my_ffs(i32 noundef %7)
  %13 = icmp eq i32 %11, %12
  br i1 %13, label %15, label %14

14:                                               ; preds = %4
  tail call void @abort() #7
  unreachable

15:                                               ; preds = %4
  br i1 %10, label %25, label %16

16:                                               ; preds = %15
  %17 = tail call range(i32 0, 33) i32 @llvm.ctlz.i32(i32 %7, i1 true)
  %18 = tail call i32 @my_clz(i32 noundef %7)
  %19 = icmp eq i32 %17, %18
  br i1 %19, label %21, label %20

20:                                               ; preds = %16
  tail call void @abort() #7
  unreachable

21:                                               ; preds = %16
  %22 = tail call i32 @my_ctz(i32 noundef %7)
  %23 = icmp eq i32 %8, %22
  br i1 %23, label %25, label %24

24:                                               ; preds = %21
  tail call void @abort() #7
  unreachable

25:                                               ; preds = %15, %21
  %26 = ashr i32 %7, 31
  %27 = xor i32 %26, %7
  %28 = tail call range(i32 0, 33) i32 @llvm.ctlz.i32(i32 %27, i1 false)
  %29 = add nsw i32 %28, -1
  %30 = tail call i32 @my_clrsb(i32 noundef %7)
  %31 = icmp eq i32 %29, %30
  br i1 %31, label %33, label %32

32:                                               ; preds = %25
  tail call void @abort() #7
  unreachable

33:                                               ; preds = %25
  %34 = tail call range(i32 0, 33) i32 @llvm.ctpop.i32(i32 %7)
  %35 = tail call i32 @my_popcount(i32 noundef %7)
  %36 = icmp eq i32 %34, %35
  br i1 %36, label %38, label %37

37:                                               ; preds = %33
  tail call void @abort() #7
  unreachable

38:                                               ; preds = %33
  %39 = insertelement <28 x i32> poison, i32 %7, i64 0
  %40 = shufflevector <28 x i32> %39, <28 x i32> poison, <28 x i32> zeroinitializer
  %41 = lshr <28 x i32> %40, <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
  %42 = lshr i32 %7, 29
  %43 = lshr i32 %7, 30
  %44 = lshr i32 %7, 31
  %45 = tail call i32 @llvm.vector.reduce.add.v28i32(<28 x i32> %41)
  %46 = add i32 %45, %42
  %47 = add nuw nsw i32 %43, %44
  %48 = add i32 %46, %47
  %49 = add i32 %48, %7
  %50 = xor i32 %49, %34
  %51 = and i32 %50, 1
  %52 = icmp eq i32 %51, 0
  br i1 %52, label %1, label %53

53:                                               ; preds = %38
  tail call void @abort() #7
  unreachable

54:                                               ; preds = %133
  %55 = add nuw nsw i64 %58, 1
  %56 = icmp eq i64 %55, 13
  br i1 %56, label %180, label %57, !llvm.loop !20

57:                                               ; preds = %1, %54
  %58 = phi i64 [ %55, %54 ], [ 0, %1 ]
  %59 = getelementptr inbounds nuw i64, ptr @longs, i64 %58
  %60 = load i64, ptr %59, align 8, !tbaa !21
  %61 = tail call range(i64 0, 65) i64 @llvm.cttz.i64(i64 %60, i1 true)
  %62 = icmp eq i64 %60, 0
  %63 = trunc nuw nsw i64 %61 to i32
  %64 = add nuw nsw i32 %63, 1
  br i1 %62, label %104, label %65

65:                                               ; preds = %57, %70
  %66 = phi i64 [ %71, %70 ], [ 0, %57 ]
  %67 = shl nuw i64 1, %66
  %68 = and i64 %67, %60
  %69 = icmp eq i64 %68, 0
  br i1 %69, label %70, label %73

70:                                               ; preds = %65
  %71 = add nuw nsw i64 %66, 1
  %72 = icmp eq i64 %71, 64
  br i1 %72, label %76, label %65, !llvm.loop !6

73:                                               ; preds = %65
  %74 = trunc nuw nsw i64 %66 to i32
  %75 = add nuw nsw i32 %74, 1
  br label %76

76:                                               ; preds = %70, %73
  %77 = phi i32 [ %75, %73 ], [ 65, %70 ]
  %78 = icmp eq i32 %64, %77
  br i1 %78, label %80, label %79

79:                                               ; preds = %76
  tail call void @abort() #7
  unreachable

80:                                               ; preds = %76
  %81 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %60, i1 true)
  br label %82

82:                                               ; preds = %87, %80
  %83 = phi i64 [ 0, %80 ], [ %88, %87 ]
  %84 = lshr exact i64 -9223372036854775808, %83
  %85 = and i64 %84, %60
  %86 = icmp eq i64 %85, 0
  br i1 %86, label %87, label %90

87:                                               ; preds = %82
  %88 = add nuw nsw i64 %83, 1
  %89 = icmp eq i64 %88, 64
  br i1 %89, label %92, label %82, !llvm.loop !9

90:                                               ; preds = %82
  %91 = icmp eq i64 %83, %81
  br i1 %91, label %93, label %92

92:                                               ; preds = %90, %87
  tail call void @abort() #7
  unreachable

93:                                               ; preds = %90, %98
  %94 = phi i64 [ %99, %98 ], [ 0, %90 ]
  %95 = shl nuw i64 1, %94
  %96 = and i64 %95, %60
  %97 = icmp eq i64 %96, 0
  br i1 %97, label %98, label %101

98:                                               ; preds = %93
  %99 = add nuw nsw i64 %94, 1
  %100 = icmp eq i64 %99, 64
  br i1 %100, label %103, label %93, !llvm.loop !8

101:                                              ; preds = %93
  %102 = icmp eq i64 %94, %61
  br i1 %102, label %104, label %103

103:                                              ; preds = %101, %98
  tail call void @abort() #7
  unreachable

104:                                              ; preds = %57, %101
  %105 = ashr i64 %60, 63
  %106 = xor i64 %105, %60
  %107 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %106, i1 false)
  %108 = trunc nuw nsw i64 %107 to i32
  %109 = add nsw i32 %108, -1
  %110 = lshr i64 %60, 63
  br label %111

111:                                              ; preds = %117, %104
  %112 = phi i64 [ 1, %104 ], [ %118, %117 ]
  %113 = sub nuw nsw i64 63, %112
  %114 = lshr i64 %60, %113
  %115 = and i64 %114, 1
  %116 = icmp eq i64 %115, %110
  br i1 %116, label %117, label %120

117:                                              ; preds = %111
  %118 = add nuw nsw i64 %112, 1
  %119 = icmp eq i64 %118, 64
  br i1 %119, label %123, label %111, !llvm.loop !10

120:                                              ; preds = %111
  %121 = trunc nuw nsw i64 %112 to i32
  %122 = add nsw i32 %121, -1
  br label %123

123:                                              ; preds = %117, %120
  %124 = phi i32 [ %122, %120 ], [ 63, %117 ]
  %125 = icmp eq i32 %109, %124
  br i1 %125, label %127, label %126

126:                                              ; preds = %123
  tail call void @abort() #7
  unreachable

127:                                              ; preds = %123
  %128 = tail call range(i64 0, 65) i64 @llvm.ctpop.i64(i64 %60)
  %129 = trunc nuw nsw i64 %128 to i32
  %130 = tail call i32 @my_popcountl(i64 noundef %60)
  %131 = icmp eq i32 %130, %129
  br i1 %131, label %133, label %132

132:                                              ; preds = %127
  tail call void @abort() #7
  unreachable

133:                                              ; preds = %127
  %134 = trunc i64 %60 to i32
  %135 = insertelement <32 x i64> poison, i64 %60, i64 0
  %136 = shufflevector <32 x i64> %135, <32 x i64> poison, <32 x i32> zeroinitializer
  %137 = lshr <32 x i64> %136, <i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9, i64 10, i64 11, i64 12, i64 13, i64 14, i64 15, i64 16, i64 17, i64 18, i64 19, i64 20, i64 21, i64 22, i64 23, i64 24, i64 25, i64 26, i64 27, i64 28, i64 29, i64 30, i64 31, i64 32>
  %138 = trunc <32 x i64> %137 to <32 x i32>
  %139 = insertelement <16 x i64> poison, i64 %60, i64 0
  %140 = shufflevector <16 x i64> %139, <16 x i64> poison, <16 x i32> zeroinitializer
  %141 = lshr <16 x i64> %140, <i64 33, i64 34, i64 35, i64 36, i64 37, i64 38, i64 39, i64 40, i64 41, i64 42, i64 43, i64 44, i64 45, i64 46, i64 47, i64 48>
  %142 = trunc nuw <16 x i64> %141 to <16 x i32>
  %143 = insertelement <8 x i64> poison, i64 %60, i64 0
  %144 = shufflevector <8 x i64> %143, <8 x i64> poison, <8 x i32> zeroinitializer
  %145 = lshr <8 x i64> %144, <i64 49, i64 50, i64 51, i64 52, i64 53, i64 54, i64 55, i64 56>
  %146 = trunc nuw nsw <8 x i64> %145 to <8 x i32>
  %147 = insertelement <4 x i64> poison, i64 %60, i64 0
  %148 = shufflevector <4 x i64> %147, <4 x i64> poison, <4 x i32> zeroinitializer
  %149 = lshr <4 x i64> %148, <i64 57, i64 58, i64 59, i64 60>
  %150 = trunc nuw nsw <4 x i64> %149 to <4 x i32>
  %151 = lshr i64 %60, 61
  %152 = trunc nuw nsw i64 %151 to i32
  %153 = lshr i64 %60, 62
  %154 = trunc nuw nsw i64 %153 to i32
  %155 = trunc nuw nsw i64 %110 to i32
  %156 = shufflevector <32 x i32> %138, <32 x i32> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %157 = add <16 x i32> %156, %142
  %158 = shufflevector <16 x i32> %157, <16 x i32> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %159 = shufflevector <32 x i32> %158, <32 x i32> %138, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %160 = shufflevector <16 x i32> %157, <16 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %161 = add <8 x i32> %160, %146
  %162 = shufflevector <8 x i32> %161, <8 x i32> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %163 = shufflevector <32 x i32> %162, <32 x i32> %159, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 4, i32 5, i32 6, i32 7, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %164 = shufflevector <8 x i32> %161, <8 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %165 = add <4 x i32> %164, %150
  %166 = shufflevector <4 x i32> %165, <4 x i32> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %167 = shufflevector <32 x i32> %166, <32 x i32> %163, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %168 = tail call i32 @llvm.vector.reduce.add.v32i32(<32 x i32> %167)
  %169 = add i32 %168, %152
  %170 = add nuw nsw i32 %154, %155
  %171 = add i32 %169, %170
  %172 = add i32 %171, %134
  %173 = xor i32 %172, %129
  %174 = and i32 %173, 1
  %175 = icmp eq i32 %174, 0
  br i1 %175, label %54, label %176

176:                                              ; preds = %133
  tail call void @abort() #7
  unreachable

177:                                              ; preds = %256
  %178 = add nuw nsw i64 %181, 1
  %179 = icmp eq i64 %178, 13
  br i1 %179, label %300, label %180, !llvm.loop !23

180:                                              ; preds = %54, %177
  %181 = phi i64 [ %178, %177 ], [ 0, %54 ]
  %182 = getelementptr inbounds nuw i64, ptr @longlongs, i64 %181
  %183 = load i64, ptr %182, align 8, !tbaa !24
  %184 = tail call range(i64 0, 65) i64 @llvm.cttz.i64(i64 %183, i1 true)
  %185 = icmp eq i64 %183, 0
  %186 = trunc nuw nsw i64 %184 to i32
  %187 = add nuw nsw i32 %186, 1
  br i1 %185, label %227, label %188

188:                                              ; preds = %180, %193
  %189 = phi i64 [ %194, %193 ], [ 0, %180 ]
  %190 = shl nuw i64 1, %189
  %191 = and i64 %190, %183
  %192 = icmp eq i64 %191, 0
  br i1 %192, label %193, label %196

193:                                              ; preds = %188
  %194 = add nuw nsw i64 %189, 1
  %195 = icmp eq i64 %194, 64
  br i1 %195, label %199, label %188, !llvm.loop !11

196:                                              ; preds = %188
  %197 = trunc nuw nsw i64 %189 to i32
  %198 = add nuw nsw i32 %197, 1
  br label %199

199:                                              ; preds = %193, %196
  %200 = phi i32 [ %198, %196 ], [ 65, %193 ]
  %201 = icmp eq i32 %187, %200
  br i1 %201, label %203, label %202

202:                                              ; preds = %199
  tail call void @abort() #7
  unreachable

203:                                              ; preds = %199
  %204 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %183, i1 true)
  br label %205

205:                                              ; preds = %210, %203
  %206 = phi i64 [ 0, %203 ], [ %211, %210 ]
  %207 = lshr exact i64 -9223372036854775808, %206
  %208 = and i64 %207, %183
  %209 = icmp eq i64 %208, 0
  br i1 %209, label %210, label %213

210:                                              ; preds = %205
  %211 = add nuw nsw i64 %206, 1
  %212 = icmp eq i64 %211, 64
  br i1 %212, label %215, label %205, !llvm.loop !13

213:                                              ; preds = %205
  %214 = icmp eq i64 %206, %204
  br i1 %214, label %216, label %215

215:                                              ; preds = %213, %210
  tail call void @abort() #7
  unreachable

216:                                              ; preds = %213, %221
  %217 = phi i64 [ %222, %221 ], [ 0, %213 ]
  %218 = shl nuw i64 1, %217
  %219 = and i64 %218, %183
  %220 = icmp eq i64 %219, 0
  br i1 %220, label %221, label %224

221:                                              ; preds = %216
  %222 = add nuw nsw i64 %217, 1
  %223 = icmp eq i64 %222, 64
  br i1 %223, label %226, label %216, !llvm.loop !12

224:                                              ; preds = %216
  %225 = icmp eq i64 %217, %184
  br i1 %225, label %227, label %226

226:                                              ; preds = %224, %221
  tail call void @abort() #7
  unreachable

227:                                              ; preds = %180, %224
  %228 = ashr i64 %183, 63
  %229 = xor i64 %228, %183
  %230 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %229, i1 false)
  %231 = trunc nuw nsw i64 %230 to i32
  %232 = add nsw i32 %231, -1
  %233 = lshr i64 %183, 63
  br label %234

234:                                              ; preds = %240, %227
  %235 = phi i64 [ 1, %227 ], [ %241, %240 ]
  %236 = sub nuw nsw i64 63, %235
  %237 = lshr i64 %183, %236
  %238 = and i64 %237, 1
  %239 = icmp eq i64 %238, %233
  br i1 %239, label %240, label %243

240:                                              ; preds = %234
  %241 = add nuw nsw i64 %235, 1
  %242 = icmp eq i64 %241, 64
  br i1 %242, label %246, label %234, !llvm.loop !14

243:                                              ; preds = %234
  %244 = trunc nuw nsw i64 %235 to i32
  %245 = add nsw i32 %244, -1
  br label %246

246:                                              ; preds = %240, %243
  %247 = phi i32 [ %245, %243 ], [ 63, %240 ]
  %248 = icmp eq i32 %232, %247
  br i1 %248, label %250, label %249

249:                                              ; preds = %246
  tail call void @abort() #7
  unreachable

250:                                              ; preds = %246
  %251 = tail call range(i64 0, 65) i64 @llvm.ctpop.i64(i64 %183)
  %252 = trunc nuw nsw i64 %251 to i32
  %253 = tail call i32 @my_popcountll(i64 noundef %183)
  %254 = icmp eq i32 %253, %252
  br i1 %254, label %256, label %255

255:                                              ; preds = %250
  tail call void @abort() #7
  unreachable

256:                                              ; preds = %250
  %257 = trunc i64 %183 to i32
  %258 = insertelement <32 x i64> poison, i64 %183, i64 0
  %259 = shufflevector <32 x i64> %258, <32 x i64> poison, <32 x i32> zeroinitializer
  %260 = lshr <32 x i64> %259, <i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9, i64 10, i64 11, i64 12, i64 13, i64 14, i64 15, i64 16, i64 17, i64 18, i64 19, i64 20, i64 21, i64 22, i64 23, i64 24, i64 25, i64 26, i64 27, i64 28, i64 29, i64 30, i64 31, i64 32>
  %261 = trunc <32 x i64> %260 to <32 x i32>
  %262 = insertelement <16 x i64> poison, i64 %183, i64 0
  %263 = shufflevector <16 x i64> %262, <16 x i64> poison, <16 x i32> zeroinitializer
  %264 = lshr <16 x i64> %263, <i64 33, i64 34, i64 35, i64 36, i64 37, i64 38, i64 39, i64 40, i64 41, i64 42, i64 43, i64 44, i64 45, i64 46, i64 47, i64 48>
  %265 = trunc nuw <16 x i64> %264 to <16 x i32>
  %266 = insertelement <8 x i64> poison, i64 %183, i64 0
  %267 = shufflevector <8 x i64> %266, <8 x i64> poison, <8 x i32> zeroinitializer
  %268 = lshr <8 x i64> %267, <i64 49, i64 50, i64 51, i64 52, i64 53, i64 54, i64 55, i64 56>
  %269 = trunc nuw nsw <8 x i64> %268 to <8 x i32>
  %270 = insertelement <4 x i64> poison, i64 %183, i64 0
  %271 = shufflevector <4 x i64> %270, <4 x i64> poison, <4 x i32> zeroinitializer
  %272 = lshr <4 x i64> %271, <i64 57, i64 58, i64 59, i64 60>
  %273 = trunc nuw nsw <4 x i64> %272 to <4 x i32>
  %274 = lshr i64 %183, 61
  %275 = trunc nuw nsw i64 %274 to i32
  %276 = lshr i64 %183, 62
  %277 = trunc nuw nsw i64 %276 to i32
  %278 = trunc nuw nsw i64 %233 to i32
  %279 = shufflevector <32 x i32> %261, <32 x i32> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %280 = add <16 x i32> %279, %265
  %281 = shufflevector <16 x i32> %280, <16 x i32> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %282 = shufflevector <32 x i32> %281, <32 x i32> %261, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %283 = shufflevector <16 x i32> %280, <16 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %284 = add <8 x i32> %283, %269
  %285 = shufflevector <8 x i32> %284, <8 x i32> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %286 = shufflevector <32 x i32> %285, <32 x i32> %282, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 4, i32 5, i32 6, i32 7, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %287 = shufflevector <8 x i32> %284, <8 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %288 = add <4 x i32> %287, %273
  %289 = shufflevector <4 x i32> %288, <4 x i32> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %290 = shufflevector <32 x i32> %289, <32 x i32> %286, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %291 = tail call i32 @llvm.vector.reduce.add.v32i32(<32 x i32> %290)
  %292 = add i32 %291, %275
  %293 = add nuw nsw i32 %277, %278
  %294 = add i32 %292, %293
  %295 = add i32 %294, %257
  %296 = xor i32 %295, %252
  %297 = and i32 %296, 1
  %298 = icmp eq i32 %297, 0
  br i1 %298, label %177, label %299

299:                                              ; preds = %256
  tail call void @abort() #7
  unreachable

300:                                              ; preds = %177
  tail call void @exit(i32 noundef 0) #7
  unreachable
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.cttz.i32(i32, i1 immarg) #3

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.ctlz.i32(i32, i1 immarg) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.ctpop.i32(i32) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.cttz.i64(i64, i1 immarg) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.ctlz.i64(i64, i1 immarg) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.ctpop.i64(i64) #3

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #5

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.vector.reduce.add.v28i32(<28 x i32>) #6

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.vector.reduce.add.v32i32(<32 x i32>) #6

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree norecurse nosync nounwind memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nofree noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #7 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = distinct !{!8, !7}
!9 = distinct !{!9, !7}
!10 = distinct !{!10, !7}
!11 = distinct !{!11, !7}
!12 = distinct !{!12, !7}
!13 = distinct !{!13, !7}
!14 = distinct !{!14, !7}
!15 = distinct !{!15, !7}
!16 = !{!17, !17, i64 0}
!17 = !{!"int", !18, i64 0}
!18 = !{!"omnipotent char", !19, i64 0}
!19 = !{!"Simple C/C++ TBAA"}
!20 = distinct !{!20, !7}
!21 = !{!22, !22, i64 0}
!22 = !{!"long", !18, i64 0}
!23 = distinct !{!23, !7}
!24 = !{!25, !25, i64 0}
!25 = !{!"long long", !18, i64 0}
