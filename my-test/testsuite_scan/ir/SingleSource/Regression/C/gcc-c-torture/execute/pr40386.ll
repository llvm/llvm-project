; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr40386.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr40386.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@c = dso_local local_unnamed_addr global i8 52, align 4
@s = dso_local local_unnamed_addr global i16 -3532, align 4
@i = dso_local local_unnamed_addr global i32 62004, align 4
@l = dso_local local_unnamed_addr global i64 4063516280, align 8
@ll = dso_local local_unnamed_addr global i64 1090791845765373680, align 8
@shift1 = dso_local local_unnamed_addr global i32 4, align 4
@shift2 = dso_local local_unnamed_addr global i32 60, align 4

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load i8, ptr @c, align 4, !tbaa !6
  %2 = zext i8 %1 to i32
  %3 = load i32, ptr @shift1, align 4, !tbaa !9
  %4 = lshr i32 %2, %3
  %5 = sext i32 %3 to i64
  %6 = sub i32 8, %3
  %7 = shl i32 %2, %6
  %8 = or i32 %7, %4
  %9 = icmp eq i32 %8, 835
  br i1 %9, label %11, label %10

10:                                               ; preds = %0
  tail call void @abort() #3
  unreachable

11:                                               ; preds = %0
  %12 = lshr i32 %2, 4
  %13 = shl nuw nsw i32 %2, 4
  %14 = or disjoint i32 %12, %13
  %15 = icmp eq i32 %14, 835
  br i1 %15, label %17, label %16

16:                                               ; preds = %11
  tail call void @abort() #3
  unreachable

17:                                               ; preds = %11
  %18 = load i16, ptr @s, align 4, !tbaa !11
  %19 = sext i16 %18 to i32
  %20 = ashr i32 %19, %3
  %21 = sub i32 16, %3
  %22 = shl i32 %19, %21
  %23 = or i32 %20, %22
  %24 = icmp eq i32 %23, -221
  br i1 %24, label %26, label %25

25:                                               ; preds = %17
  tail call void @abort() #3
  unreachable

26:                                               ; preds = %17
  %27 = ashr i32 %19, 4
  %28 = shl nsw i32 %19, 12
  %29 = or i32 %27, %28
  %30 = icmp eq i32 %29, -221
  br i1 %30, label %32, label %31

31:                                               ; preds = %26
  tail call void @abort() #3
  unreachable

32:                                               ; preds = %26
  %33 = load i32, ptr @i, align 4, !tbaa !9
  %34 = ashr i32 %33, %3
  %35 = sub i32 32, %3
  %36 = shl i32 %33, %35
  %37 = or i32 %34, %36
  %38 = icmp eq i32 %37, 1073745699
  br i1 %38, label %40, label %39

39:                                               ; preds = %32
  tail call void @abort() #3
  unreachable

40:                                               ; preds = %32
  %41 = ashr i32 %33, 4
  %42 = shl i32 %33, 28
  %43 = or i32 %41, %42
  %44 = icmp eq i32 %43, 1073745699
  br i1 %44, label %46, label %45

45:                                               ; preds = %40
  tail call void @abort() #3
  unreachable

46:                                               ; preds = %40
  %47 = load i64, ptr @l, align 8, !tbaa !13
  %48 = zext i32 %3 to i64
  %49 = ashr i64 %47, %48
  %50 = sub nsw i64 64, %5
  %51 = shl i64 %47, %50
  %52 = or i64 %49, %51
  %53 = icmp eq i64 %52, -9223372036600806041
  br i1 %53, label %55, label %54

54:                                               ; preds = %46
  tail call void @abort() #3
  unreachable

55:                                               ; preds = %46
  %56 = ashr i64 %47, 4
  %57 = shl i64 %47, 60
  %58 = or i64 %56, %57
  %59 = icmp eq i64 %58, -9223372036600806041
  br i1 %59, label %61, label %60

60:                                               ; preds = %55
  tail call void @abort() #3
  unreachable

61:                                               ; preds = %55
  %62 = load i64, ptr @ll, align 8, !tbaa !15
  %63 = ashr i64 %62, %48
  %64 = shl i64 %62, %50
  %65 = or i64 %63, %64
  %66 = icmp eq i64 %65, 68174490360335855
  br i1 %66, label %68, label %67

67:                                               ; preds = %61
  tail call void @abort() #3
  unreachable

68:                                               ; preds = %61
  %69 = ashr i64 %62, 4
  %70 = shl i64 %62, 60
  %71 = or i64 %69, %70
  %72 = icmp eq i64 %71, 68174490360335855
  br i1 %72, label %74, label %73

73:                                               ; preds = %68
  tail call void @abort() #3
  unreachable

74:                                               ; preds = %68
  %75 = load i32, ptr @shift2, align 4, !tbaa !9
  %76 = zext i32 %75 to i64
  %77 = ashr i64 %62, %76
  %78 = sext i32 %75 to i64
  %79 = sub nsw i64 64, %78
  %80 = shl i64 %62, %79
  %81 = or i64 %80, %77
  %82 = icmp eq i64 %81, -994074541463572736
  br i1 %82, label %84, label %83

83:                                               ; preds = %74
  tail call void @abort() #3
  unreachable

84:                                               ; preds = %74
  %85 = ashr i64 %62, 60
  %86 = shl i64 %62, 4
  %87 = or i64 %85, %86
  %88 = icmp eq i64 %87, -994074541463572736
  br i1 %88, label %90, label %89

89:                                               ; preds = %84
  tail call void @abort() #3
  unreachable

90:                                               ; preds = %84
  %91 = shl i32 %2, %3
  %92 = lshr i32 %2, %6
  %93 = or i32 %92, %91
  %94 = icmp eq i32 %93, 835
  br i1 %94, label %96, label %95

95:                                               ; preds = %90
  tail call void @abort() #3
  unreachable

96:                                               ; preds = %90
  %97 = shl i32 %19, %3
  %98 = ashr i32 %19, %21
  %99 = or i32 %97, %98
  %100 = icmp eq i32 %99, -1
  br i1 %100, label %102, label %101

101:                                              ; preds = %96
  tail call void @abort() #3
  unreachable

102:                                              ; preds = %96
  %103 = shl nsw i32 %19, 4
  %104 = ashr i32 %19, 12
  %105 = or i32 %103, %104
  %106 = icmp eq i32 %105, -1
  br i1 %106, label %108, label %107

107:                                              ; preds = %102
  tail call void @abort() #3
  unreachable

108:                                              ; preds = %102
  %109 = shl i32 %33, %3
  %110 = ashr i32 %33, %35
  %111 = or i32 %109, %110
  %112 = icmp eq i32 %111, 992064
  br i1 %112, label %114, label %113

113:                                              ; preds = %108
  tail call void @abort() #3
  unreachable

114:                                              ; preds = %108
  %115 = shl i32 %33, 4
  %116 = ashr i32 %33, 28
  %117 = or i32 %115, %116
  %118 = icmp eq i32 %117, 992064
  br i1 %118, label %120, label %119

119:                                              ; preds = %114
  tail call void @abort() #3
  unreachable

120:                                              ; preds = %114
  %121 = shl i64 %47, %48
  %122 = ashr i64 %47, %50
  %123 = or i64 %121, %122
  %124 = icmp eq i64 %123, 65016260480
  br i1 %124, label %126, label %125

125:                                              ; preds = %120
  tail call void @abort() #3
  unreachable

126:                                              ; preds = %120
  %127 = shl i64 %47, 4
  %128 = ashr i64 %47, 60
  %129 = or i64 %127, %128
  %130 = icmp eq i64 %129, 65016260480
  br i1 %130, label %132, label %131

131:                                              ; preds = %126
  tail call void @abort() #3
  unreachable

132:                                              ; preds = %126
  %133 = shl i64 %62, %48
  %134 = ashr i64 %62, %50
  %135 = or i64 %133, %134
  %136 = icmp eq i64 %135, -994074541463572736
  br i1 %136, label %138, label %137

137:                                              ; preds = %132
  tail call void @abort() #3
  unreachable

138:                                              ; preds = %132
  %139 = shl i64 %62, %76
  %140 = ashr i64 %62, %79
  %141 = or i64 %140, %139
  %142 = icmp eq i64 %141, 68174490360335855
  br i1 %142, label %144, label %143

143:                                              ; preds = %138
  tail call void @abort() #3
  unreachable

144:                                              ; preds = %138
  tail call void @exit(i32 noundef 0) #3
  unreachable
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #2

attributes #0 = { nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { noreturn nounwind }

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
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !7, i64 0}
!11 = !{!12, !12, i64 0}
!12 = !{!"short", !7, i64 0}
!13 = !{!14, !14, i64 0}
!14 = !{!"long", !7, i64 0}
!15 = !{!16, !16, i64 0}
!16 = !{!"long long", !7, i64 0}
