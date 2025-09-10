; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr39501.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr39501.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef float @float_min1(float noundef %0, float noundef %1) local_unnamed_addr #0 {
  %3 = fcmp olt float %0, %1
  %4 = select i1 %3, float %0, float %1
  ret float %4
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef float @float_min2(float noundef %0, float noundef %1) local_unnamed_addr #0 {
  %3 = fcmp ole float %0, %1
  %4 = select i1 %3, float %0, float %1
  ret float %4
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef float @float_max1(float noundef %0, float noundef %1) local_unnamed_addr #0 {
  %3 = fcmp ogt float %0, %1
  %4 = select i1 %3, float %0, float %1
  ret float %4
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef float @float_max2(float noundef %0, float noundef %1) local_unnamed_addr #0 {
  %3 = fcmp oge float %0, %1
  %4 = select i1 %3, float %0, float %1
  ret float %4
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef double @double_min1(double noundef %0, double noundef %1) local_unnamed_addr #0 {
  %3 = fcmp olt double %0, %1
  %4 = select i1 %3, double %0, double %1
  ret double %4
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef double @double_min2(double noundef %0, double noundef %1) local_unnamed_addr #0 {
  %3 = fcmp ole double %0, %1
  %4 = select i1 %3, double %0, double %1
  ret double %4
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef double @double_max1(double noundef %0, double noundef %1) local_unnamed_addr #0 {
  %3 = fcmp ogt double %0, %1
  %4 = select i1 %3, double %0, double %1
  ret double %4
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef double @double_max2(double noundef %0, double noundef %1) local_unnamed_addr #0 {
  %3 = fcmp oge double %0, %1
  %4 = select i1 %3, double %0, double %1
  ret double %4
}

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = tail call float @float_min1(float noundef 0.000000e+00, float noundef -1.000000e+00)
  %2 = fcmp une float %1, -1.000000e+00
  br i1 %2, label %3, label %4

3:                                                ; preds = %0
  tail call void @abort() #4
  unreachable

4:                                                ; preds = %0
  %5 = tail call float @float_min1(float noundef -1.000000e+00, float noundef 0.000000e+00)
  %6 = fcmp une float %5, -1.000000e+00
  br i1 %6, label %7, label %8

7:                                                ; preds = %4
  tail call void @abort() #4
  unreachable

8:                                                ; preds = %4
  %9 = tail call float @float_min1(float noundef 0.000000e+00, float noundef 1.000000e+00)
  %10 = fcmp une float %9, 0.000000e+00
  br i1 %10, label %11, label %12

11:                                               ; preds = %8
  tail call void @abort() #4
  unreachable

12:                                               ; preds = %8
  %13 = tail call float @float_min1(float noundef 1.000000e+00, float noundef 0.000000e+00)
  %14 = fcmp une float %13, 0.000000e+00
  br i1 %14, label %15, label %16

15:                                               ; preds = %12
  tail call void @abort() #4
  unreachable

16:                                               ; preds = %12
  %17 = tail call float @float_min1(float noundef -1.000000e+00, float noundef 1.000000e+00)
  %18 = fcmp une float %17, -1.000000e+00
  br i1 %18, label %19, label %20

19:                                               ; preds = %16
  tail call void @abort() #4
  unreachable

20:                                               ; preds = %16
  %21 = tail call float @float_min1(float noundef 1.000000e+00, float noundef -1.000000e+00)
  %22 = fcmp une float %21, -1.000000e+00
  br i1 %22, label %23, label %24

23:                                               ; preds = %20
  tail call void @abort() #4
  unreachable

24:                                               ; preds = %20
  %25 = tail call float @float_max1(float noundef 0.000000e+00, float noundef -1.000000e+00)
  %26 = fcmp une float %25, 0.000000e+00
  br i1 %26, label %27, label %28

27:                                               ; preds = %24
  tail call void @abort() #4
  unreachable

28:                                               ; preds = %24
  %29 = tail call float @float_max1(float noundef -1.000000e+00, float noundef 0.000000e+00)
  %30 = fcmp une float %29, 0.000000e+00
  br i1 %30, label %31, label %32

31:                                               ; preds = %28
  tail call void @abort() #4
  unreachable

32:                                               ; preds = %28
  %33 = tail call float @float_max1(float noundef 0.000000e+00, float noundef 1.000000e+00)
  %34 = fcmp une float %33, 1.000000e+00
  br i1 %34, label %35, label %36

35:                                               ; preds = %32
  tail call void @abort() #4
  unreachable

36:                                               ; preds = %32
  %37 = tail call float @float_max1(float noundef 1.000000e+00, float noundef 0.000000e+00)
  %38 = fcmp une float %37, 1.000000e+00
  br i1 %38, label %39, label %40

39:                                               ; preds = %36
  tail call void @abort() #4
  unreachable

40:                                               ; preds = %36
  %41 = tail call float @float_max1(float noundef -1.000000e+00, float noundef 1.000000e+00)
  %42 = fcmp une float %41, 1.000000e+00
  br i1 %42, label %43, label %44

43:                                               ; preds = %40
  tail call void @abort() #4
  unreachable

44:                                               ; preds = %40
  %45 = tail call float @float_max1(float noundef 1.000000e+00, float noundef -1.000000e+00)
  %46 = fcmp une float %45, 1.000000e+00
  br i1 %46, label %47, label %48

47:                                               ; preds = %44
  tail call void @abort() #4
  unreachable

48:                                               ; preds = %44
  %49 = tail call float @float_min2(float noundef 0.000000e+00, float noundef -1.000000e+00)
  %50 = fcmp une float %49, -1.000000e+00
  br i1 %50, label %51, label %52

51:                                               ; preds = %48
  tail call void @abort() #4
  unreachable

52:                                               ; preds = %48
  %53 = tail call float @float_min2(float noundef -1.000000e+00, float noundef 0.000000e+00)
  %54 = fcmp une float %53, -1.000000e+00
  br i1 %54, label %55, label %56

55:                                               ; preds = %52
  tail call void @abort() #4
  unreachable

56:                                               ; preds = %52
  %57 = tail call float @float_min2(float noundef 0.000000e+00, float noundef 1.000000e+00)
  %58 = fcmp une float %57, 0.000000e+00
  br i1 %58, label %59, label %60

59:                                               ; preds = %56
  tail call void @abort() #4
  unreachable

60:                                               ; preds = %56
  %61 = tail call float @float_min2(float noundef 1.000000e+00, float noundef 0.000000e+00)
  %62 = fcmp une float %61, 0.000000e+00
  br i1 %62, label %63, label %64

63:                                               ; preds = %60
  tail call void @abort() #4
  unreachable

64:                                               ; preds = %60
  %65 = tail call float @float_min2(float noundef -1.000000e+00, float noundef 1.000000e+00)
  %66 = fcmp une float %65, -1.000000e+00
  br i1 %66, label %67, label %68

67:                                               ; preds = %64
  tail call void @abort() #4
  unreachable

68:                                               ; preds = %64
  %69 = tail call float @float_min2(float noundef 1.000000e+00, float noundef -1.000000e+00)
  %70 = fcmp une float %69, -1.000000e+00
  br i1 %70, label %71, label %72

71:                                               ; preds = %68
  tail call void @abort() #4
  unreachable

72:                                               ; preds = %68
  %73 = tail call float @float_max2(float noundef 0.000000e+00, float noundef -1.000000e+00)
  %74 = fcmp une float %73, 0.000000e+00
  br i1 %74, label %75, label %76

75:                                               ; preds = %72
  tail call void @abort() #4
  unreachable

76:                                               ; preds = %72
  %77 = tail call float @float_max2(float noundef -1.000000e+00, float noundef 0.000000e+00)
  %78 = fcmp une float %77, 0.000000e+00
  br i1 %78, label %79, label %80

79:                                               ; preds = %76
  tail call void @abort() #4
  unreachable

80:                                               ; preds = %76
  %81 = tail call float @float_max2(float noundef 0.000000e+00, float noundef 1.000000e+00)
  %82 = fcmp une float %81, 1.000000e+00
  br i1 %82, label %83, label %84

83:                                               ; preds = %80
  tail call void @abort() #4
  unreachable

84:                                               ; preds = %80
  %85 = tail call float @float_max2(float noundef 1.000000e+00, float noundef 0.000000e+00)
  %86 = fcmp une float %85, 1.000000e+00
  br i1 %86, label %87, label %88

87:                                               ; preds = %84
  tail call void @abort() #4
  unreachable

88:                                               ; preds = %84
  %89 = tail call float @float_max2(float noundef -1.000000e+00, float noundef 1.000000e+00)
  %90 = fcmp une float %89, 1.000000e+00
  br i1 %90, label %91, label %92

91:                                               ; preds = %88
  tail call void @abort() #4
  unreachable

92:                                               ; preds = %88
  %93 = tail call float @float_max2(float noundef 1.000000e+00, float noundef -1.000000e+00)
  %94 = fcmp une float %93, 1.000000e+00
  br i1 %94, label %95, label %96

95:                                               ; preds = %92
  tail call void @abort() #4
  unreachable

96:                                               ; preds = %92
  %97 = tail call double @double_min1(double noundef 0.000000e+00, double noundef -1.000000e+00)
  %98 = fcmp une double %97, -1.000000e+00
  br i1 %98, label %99, label %100

99:                                               ; preds = %96
  tail call void @abort() #4
  unreachable

100:                                              ; preds = %96
  %101 = tail call double @double_min1(double noundef -1.000000e+00, double noundef 0.000000e+00)
  %102 = fcmp une double %101, -1.000000e+00
  br i1 %102, label %103, label %104

103:                                              ; preds = %100
  tail call void @abort() #4
  unreachable

104:                                              ; preds = %100
  %105 = tail call double @double_min1(double noundef 0.000000e+00, double noundef 1.000000e+00)
  %106 = fcmp une double %105, 0.000000e+00
  br i1 %106, label %107, label %108

107:                                              ; preds = %104
  tail call void @abort() #4
  unreachable

108:                                              ; preds = %104
  %109 = tail call double @double_min1(double noundef 1.000000e+00, double noundef 0.000000e+00)
  %110 = fcmp une double %109, 0.000000e+00
  br i1 %110, label %111, label %112

111:                                              ; preds = %108
  tail call void @abort() #4
  unreachable

112:                                              ; preds = %108
  %113 = tail call double @double_min1(double noundef -1.000000e+00, double noundef 1.000000e+00)
  %114 = fcmp une double %113, -1.000000e+00
  br i1 %114, label %115, label %116

115:                                              ; preds = %112
  tail call void @abort() #4
  unreachable

116:                                              ; preds = %112
  %117 = tail call double @double_min1(double noundef 1.000000e+00, double noundef -1.000000e+00)
  %118 = fcmp une double %117, -1.000000e+00
  br i1 %118, label %119, label %120

119:                                              ; preds = %116
  tail call void @abort() #4
  unreachable

120:                                              ; preds = %116
  %121 = tail call double @double_max1(double noundef 0.000000e+00, double noundef -1.000000e+00)
  %122 = fcmp une double %121, 0.000000e+00
  br i1 %122, label %123, label %124

123:                                              ; preds = %120
  tail call void @abort() #4
  unreachable

124:                                              ; preds = %120
  %125 = tail call double @double_max1(double noundef -1.000000e+00, double noundef 0.000000e+00)
  %126 = fcmp une double %125, 0.000000e+00
  br i1 %126, label %127, label %128

127:                                              ; preds = %124
  tail call void @abort() #4
  unreachable

128:                                              ; preds = %124
  %129 = tail call double @double_max1(double noundef 0.000000e+00, double noundef 1.000000e+00)
  %130 = fcmp une double %129, 1.000000e+00
  br i1 %130, label %131, label %132

131:                                              ; preds = %128
  tail call void @abort() #4
  unreachable

132:                                              ; preds = %128
  %133 = tail call double @double_max1(double noundef 1.000000e+00, double noundef 0.000000e+00)
  %134 = fcmp une double %133, 1.000000e+00
  br i1 %134, label %135, label %136

135:                                              ; preds = %132
  tail call void @abort() #4
  unreachable

136:                                              ; preds = %132
  %137 = tail call double @double_max1(double noundef -1.000000e+00, double noundef 1.000000e+00)
  %138 = fcmp une double %137, 1.000000e+00
  br i1 %138, label %139, label %140

139:                                              ; preds = %136
  tail call void @abort() #4
  unreachable

140:                                              ; preds = %136
  %141 = tail call double @double_max1(double noundef 1.000000e+00, double noundef -1.000000e+00)
  %142 = fcmp une double %141, 1.000000e+00
  br i1 %142, label %143, label %144

143:                                              ; preds = %140
  tail call void @abort() #4
  unreachable

144:                                              ; preds = %140
  %145 = tail call double @double_min2(double noundef 0.000000e+00, double noundef -1.000000e+00)
  %146 = fcmp une double %145, -1.000000e+00
  br i1 %146, label %147, label %148

147:                                              ; preds = %144
  tail call void @abort() #4
  unreachable

148:                                              ; preds = %144
  %149 = tail call double @double_min2(double noundef -1.000000e+00, double noundef 0.000000e+00)
  %150 = fcmp une double %149, -1.000000e+00
  br i1 %150, label %151, label %152

151:                                              ; preds = %148
  tail call void @abort() #4
  unreachable

152:                                              ; preds = %148
  %153 = tail call double @double_min2(double noundef 0.000000e+00, double noundef 1.000000e+00)
  %154 = fcmp une double %153, 0.000000e+00
  br i1 %154, label %155, label %156

155:                                              ; preds = %152
  tail call void @abort() #4
  unreachable

156:                                              ; preds = %152
  %157 = tail call double @double_min2(double noundef 1.000000e+00, double noundef 0.000000e+00)
  %158 = fcmp une double %157, 0.000000e+00
  br i1 %158, label %159, label %160

159:                                              ; preds = %156
  tail call void @abort() #4
  unreachable

160:                                              ; preds = %156
  %161 = tail call double @double_min2(double noundef -1.000000e+00, double noundef 1.000000e+00)
  %162 = fcmp une double %161, -1.000000e+00
  br i1 %162, label %163, label %164

163:                                              ; preds = %160
  tail call void @abort() #4
  unreachable

164:                                              ; preds = %160
  %165 = tail call double @double_min2(double noundef 1.000000e+00, double noundef -1.000000e+00)
  %166 = fcmp une double %165, -1.000000e+00
  br i1 %166, label %167, label %168

167:                                              ; preds = %164
  tail call void @abort() #4
  unreachable

168:                                              ; preds = %164
  %169 = tail call double @double_max2(double noundef 0.000000e+00, double noundef -1.000000e+00)
  %170 = fcmp une double %169, 0.000000e+00
  br i1 %170, label %171, label %172

171:                                              ; preds = %168
  tail call void @abort() #4
  unreachable

172:                                              ; preds = %168
  %173 = tail call double @double_max2(double noundef -1.000000e+00, double noundef 0.000000e+00)
  %174 = fcmp une double %173, 0.000000e+00
  br i1 %174, label %175, label %176

175:                                              ; preds = %172
  tail call void @abort() #4
  unreachable

176:                                              ; preds = %172
  %177 = tail call double @double_max2(double noundef 0.000000e+00, double noundef 1.000000e+00)
  %178 = fcmp une double %177, 1.000000e+00
  br i1 %178, label %179, label %180

179:                                              ; preds = %176
  tail call void @abort() #4
  unreachable

180:                                              ; preds = %176
  %181 = tail call double @double_max2(double noundef 1.000000e+00, double noundef 0.000000e+00)
  %182 = fcmp une double %181, 1.000000e+00
  br i1 %182, label %183, label %184

183:                                              ; preds = %180
  tail call void @abort() #4
  unreachable

184:                                              ; preds = %180
  %185 = tail call double @double_max2(double noundef -1.000000e+00, double noundef 1.000000e+00)
  %186 = fcmp une double %185, 1.000000e+00
  br i1 %186, label %187, label %188

187:                                              ; preds = %184
  tail call void @abort() #4
  unreachable

188:                                              ; preds = %184
  %189 = tail call double @double_max2(double noundef 1.000000e+00, double noundef -1.000000e+00)
  %190 = fcmp une double %189, 1.000000e+00
  br i1 %190, label %191, label %192

191:                                              ; preds = %188
  tail call void @abort() #4
  unreachable

192:                                              ; preds = %188
  tail call void @exit(i32 noundef 0) #4
  unreachable
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #3

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
