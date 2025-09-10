; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20030323-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20030323-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local ptr @NSReturnAddress(i32 noundef %0) local_unnamed_addr #0 {
  switch i32 %0, label %202 [
    i32 0, label %2
    i32 1, label %4
    i32 2, label %6
    i32 3, label %8
    i32 4, label %10
    i32 5, label %12
    i32 6, label %14
    i32 7, label %16
    i32 8, label %18
    i32 9, label %20
    i32 10, label %22
    i32 11, label %24
    i32 12, label %26
    i32 13, label %28
    i32 14, label %30
    i32 15, label %32
    i32 16, label %34
    i32 17, label %36
    i32 18, label %38
    i32 19, label %40
    i32 20, label %42
    i32 21, label %44
    i32 22, label %46
    i32 23, label %48
    i32 24, label %50
    i32 25, label %52
    i32 26, label %54
    i32 27, label %56
    i32 28, label %58
    i32 29, label %60
    i32 30, label %62
    i32 31, label %64
    i32 32, label %66
    i32 33, label %68
    i32 34, label %70
    i32 35, label %72
    i32 36, label %74
    i32 37, label %76
    i32 38, label %78
    i32 39, label %80
    i32 40, label %82
    i32 41, label %84
    i32 42, label %86
    i32 43, label %88
    i32 44, label %90
    i32 45, label %92
    i32 46, label %94
    i32 47, label %96
    i32 48, label %98
    i32 49, label %100
    i32 50, label %102
    i32 51, label %104
    i32 52, label %106
    i32 53, label %108
    i32 54, label %110
    i32 55, label %112
    i32 56, label %114
    i32 57, label %116
    i32 58, label %118
    i32 59, label %120
    i32 60, label %122
    i32 61, label %124
    i32 62, label %126
    i32 63, label %128
    i32 64, label %130
    i32 65, label %132
    i32 66, label %134
    i32 67, label %136
    i32 68, label %138
    i32 69, label %140
    i32 70, label %142
    i32 71, label %144
    i32 72, label %146
    i32 73, label %148
    i32 74, label %150
    i32 75, label %152
    i32 76, label %154
    i32 77, label %156
    i32 78, label %158
    i32 79, label %160
    i32 80, label %162
    i32 81, label %164
    i32 82, label %166
    i32 83, label %168
    i32 84, label %170
    i32 85, label %172
    i32 86, label %174
    i32 87, label %176
    i32 88, label %178
    i32 89, label %180
    i32 90, label %182
    i32 91, label %184
    i32 92, label %186
    i32 93, label %188
    i32 94, label %190
    i32 95, label %192
    i32 96, label %194
    i32 97, label %196
    i32 98, label %198
    i32 99, label %200
  ]

2:                                                ; preds = %1
  %3 = tail call ptr @llvm.returnaddress(i32 1)
  br label %202

4:                                                ; preds = %1
  %5 = tail call ptr @llvm.returnaddress(i32 2)
  br label %202

6:                                                ; preds = %1
  %7 = tail call ptr @llvm.returnaddress(i32 3)
  br label %202

8:                                                ; preds = %1
  %9 = tail call ptr @llvm.returnaddress(i32 4)
  br label %202

10:                                               ; preds = %1
  %11 = tail call ptr @llvm.returnaddress(i32 5)
  br label %202

12:                                               ; preds = %1
  %13 = tail call ptr @llvm.returnaddress(i32 6)
  br label %202

14:                                               ; preds = %1
  %15 = tail call ptr @llvm.returnaddress(i32 7)
  br label %202

16:                                               ; preds = %1
  %17 = tail call ptr @llvm.returnaddress(i32 8)
  br label %202

18:                                               ; preds = %1
  %19 = tail call ptr @llvm.returnaddress(i32 9)
  br label %202

20:                                               ; preds = %1
  %21 = tail call ptr @llvm.returnaddress(i32 10)
  br label %202

22:                                               ; preds = %1
  %23 = tail call ptr @llvm.returnaddress(i32 11)
  br label %202

24:                                               ; preds = %1
  %25 = tail call ptr @llvm.returnaddress(i32 12)
  br label %202

26:                                               ; preds = %1
  %27 = tail call ptr @llvm.returnaddress(i32 13)
  br label %202

28:                                               ; preds = %1
  %29 = tail call ptr @llvm.returnaddress(i32 14)
  br label %202

30:                                               ; preds = %1
  %31 = tail call ptr @llvm.returnaddress(i32 15)
  br label %202

32:                                               ; preds = %1
  %33 = tail call ptr @llvm.returnaddress(i32 16)
  br label %202

34:                                               ; preds = %1
  %35 = tail call ptr @llvm.returnaddress(i32 17)
  br label %202

36:                                               ; preds = %1
  %37 = tail call ptr @llvm.returnaddress(i32 18)
  br label %202

38:                                               ; preds = %1
  %39 = tail call ptr @llvm.returnaddress(i32 19)
  br label %202

40:                                               ; preds = %1
  %41 = tail call ptr @llvm.returnaddress(i32 20)
  br label %202

42:                                               ; preds = %1
  %43 = tail call ptr @llvm.returnaddress(i32 21)
  br label %202

44:                                               ; preds = %1
  %45 = tail call ptr @llvm.returnaddress(i32 22)
  br label %202

46:                                               ; preds = %1
  %47 = tail call ptr @llvm.returnaddress(i32 23)
  br label %202

48:                                               ; preds = %1
  %49 = tail call ptr @llvm.returnaddress(i32 24)
  br label %202

50:                                               ; preds = %1
  %51 = tail call ptr @llvm.returnaddress(i32 25)
  br label %202

52:                                               ; preds = %1
  %53 = tail call ptr @llvm.returnaddress(i32 26)
  br label %202

54:                                               ; preds = %1
  %55 = tail call ptr @llvm.returnaddress(i32 27)
  br label %202

56:                                               ; preds = %1
  %57 = tail call ptr @llvm.returnaddress(i32 28)
  br label %202

58:                                               ; preds = %1
  %59 = tail call ptr @llvm.returnaddress(i32 29)
  br label %202

60:                                               ; preds = %1
  %61 = tail call ptr @llvm.returnaddress(i32 30)
  br label %202

62:                                               ; preds = %1
  %63 = tail call ptr @llvm.returnaddress(i32 31)
  br label %202

64:                                               ; preds = %1
  %65 = tail call ptr @llvm.returnaddress(i32 32)
  br label %202

66:                                               ; preds = %1
  %67 = tail call ptr @llvm.returnaddress(i32 33)
  br label %202

68:                                               ; preds = %1
  %69 = tail call ptr @llvm.returnaddress(i32 34)
  br label %202

70:                                               ; preds = %1
  %71 = tail call ptr @llvm.returnaddress(i32 35)
  br label %202

72:                                               ; preds = %1
  %73 = tail call ptr @llvm.returnaddress(i32 36)
  br label %202

74:                                               ; preds = %1
  %75 = tail call ptr @llvm.returnaddress(i32 37)
  br label %202

76:                                               ; preds = %1
  %77 = tail call ptr @llvm.returnaddress(i32 38)
  br label %202

78:                                               ; preds = %1
  %79 = tail call ptr @llvm.returnaddress(i32 39)
  br label %202

80:                                               ; preds = %1
  %81 = tail call ptr @llvm.returnaddress(i32 40)
  br label %202

82:                                               ; preds = %1
  %83 = tail call ptr @llvm.returnaddress(i32 41)
  br label %202

84:                                               ; preds = %1
  %85 = tail call ptr @llvm.returnaddress(i32 42)
  br label %202

86:                                               ; preds = %1
  %87 = tail call ptr @llvm.returnaddress(i32 43)
  br label %202

88:                                               ; preds = %1
  %89 = tail call ptr @llvm.returnaddress(i32 44)
  br label %202

90:                                               ; preds = %1
  %91 = tail call ptr @llvm.returnaddress(i32 45)
  br label %202

92:                                               ; preds = %1
  %93 = tail call ptr @llvm.returnaddress(i32 46)
  br label %202

94:                                               ; preds = %1
  %95 = tail call ptr @llvm.returnaddress(i32 47)
  br label %202

96:                                               ; preds = %1
  %97 = tail call ptr @llvm.returnaddress(i32 48)
  br label %202

98:                                               ; preds = %1
  %99 = tail call ptr @llvm.returnaddress(i32 49)
  br label %202

100:                                              ; preds = %1
  %101 = tail call ptr @llvm.returnaddress(i32 50)
  br label %202

102:                                              ; preds = %1
  %103 = tail call ptr @llvm.returnaddress(i32 51)
  br label %202

104:                                              ; preds = %1
  %105 = tail call ptr @llvm.returnaddress(i32 52)
  br label %202

106:                                              ; preds = %1
  %107 = tail call ptr @llvm.returnaddress(i32 53)
  br label %202

108:                                              ; preds = %1
  %109 = tail call ptr @llvm.returnaddress(i32 54)
  br label %202

110:                                              ; preds = %1
  %111 = tail call ptr @llvm.returnaddress(i32 55)
  br label %202

112:                                              ; preds = %1
  %113 = tail call ptr @llvm.returnaddress(i32 56)
  br label %202

114:                                              ; preds = %1
  %115 = tail call ptr @llvm.returnaddress(i32 57)
  br label %202

116:                                              ; preds = %1
  %117 = tail call ptr @llvm.returnaddress(i32 58)
  br label %202

118:                                              ; preds = %1
  %119 = tail call ptr @llvm.returnaddress(i32 59)
  br label %202

120:                                              ; preds = %1
  %121 = tail call ptr @llvm.returnaddress(i32 60)
  br label %202

122:                                              ; preds = %1
  %123 = tail call ptr @llvm.returnaddress(i32 61)
  br label %202

124:                                              ; preds = %1
  %125 = tail call ptr @llvm.returnaddress(i32 62)
  br label %202

126:                                              ; preds = %1
  %127 = tail call ptr @llvm.returnaddress(i32 63)
  br label %202

128:                                              ; preds = %1
  %129 = tail call ptr @llvm.returnaddress(i32 64)
  br label %202

130:                                              ; preds = %1
  %131 = tail call ptr @llvm.returnaddress(i32 65)
  br label %202

132:                                              ; preds = %1
  %133 = tail call ptr @llvm.returnaddress(i32 66)
  br label %202

134:                                              ; preds = %1
  %135 = tail call ptr @llvm.returnaddress(i32 67)
  br label %202

136:                                              ; preds = %1
  %137 = tail call ptr @llvm.returnaddress(i32 68)
  br label %202

138:                                              ; preds = %1
  %139 = tail call ptr @llvm.returnaddress(i32 69)
  br label %202

140:                                              ; preds = %1
  %141 = tail call ptr @llvm.returnaddress(i32 70)
  br label %202

142:                                              ; preds = %1
  %143 = tail call ptr @llvm.returnaddress(i32 71)
  br label %202

144:                                              ; preds = %1
  %145 = tail call ptr @llvm.returnaddress(i32 72)
  br label %202

146:                                              ; preds = %1
  %147 = tail call ptr @llvm.returnaddress(i32 73)
  br label %202

148:                                              ; preds = %1
  %149 = tail call ptr @llvm.returnaddress(i32 74)
  br label %202

150:                                              ; preds = %1
  %151 = tail call ptr @llvm.returnaddress(i32 75)
  br label %202

152:                                              ; preds = %1
  %153 = tail call ptr @llvm.returnaddress(i32 76)
  br label %202

154:                                              ; preds = %1
  %155 = tail call ptr @llvm.returnaddress(i32 77)
  br label %202

156:                                              ; preds = %1
  %157 = tail call ptr @llvm.returnaddress(i32 78)
  br label %202

158:                                              ; preds = %1
  %159 = tail call ptr @llvm.returnaddress(i32 79)
  br label %202

160:                                              ; preds = %1
  %161 = tail call ptr @llvm.returnaddress(i32 80)
  br label %202

162:                                              ; preds = %1
  %163 = tail call ptr @llvm.returnaddress(i32 81)
  br label %202

164:                                              ; preds = %1
  %165 = tail call ptr @llvm.returnaddress(i32 82)
  br label %202

166:                                              ; preds = %1
  %167 = tail call ptr @llvm.returnaddress(i32 83)
  br label %202

168:                                              ; preds = %1
  %169 = tail call ptr @llvm.returnaddress(i32 84)
  br label %202

170:                                              ; preds = %1
  %171 = tail call ptr @llvm.returnaddress(i32 85)
  br label %202

172:                                              ; preds = %1
  %173 = tail call ptr @llvm.returnaddress(i32 86)
  br label %202

174:                                              ; preds = %1
  %175 = tail call ptr @llvm.returnaddress(i32 87)
  br label %202

176:                                              ; preds = %1
  %177 = tail call ptr @llvm.returnaddress(i32 88)
  br label %202

178:                                              ; preds = %1
  %179 = tail call ptr @llvm.returnaddress(i32 89)
  br label %202

180:                                              ; preds = %1
  %181 = tail call ptr @llvm.returnaddress(i32 90)
  br label %202

182:                                              ; preds = %1
  %183 = tail call ptr @llvm.returnaddress(i32 91)
  br label %202

184:                                              ; preds = %1
  %185 = tail call ptr @llvm.returnaddress(i32 92)
  br label %202

186:                                              ; preds = %1
  %187 = tail call ptr @llvm.returnaddress(i32 93)
  br label %202

188:                                              ; preds = %1
  %189 = tail call ptr @llvm.returnaddress(i32 94)
  br label %202

190:                                              ; preds = %1
  %191 = tail call ptr @llvm.returnaddress(i32 95)
  br label %202

192:                                              ; preds = %1
  %193 = tail call ptr @llvm.returnaddress(i32 96)
  br label %202

194:                                              ; preds = %1
  %195 = tail call ptr @llvm.returnaddress(i32 97)
  br label %202

196:                                              ; preds = %1
  %197 = tail call ptr @llvm.returnaddress(i32 98)
  br label %202

198:                                              ; preds = %1
  %199 = tail call ptr @llvm.returnaddress(i32 99)
  br label %202

200:                                              ; preds = %1
  %201 = tail call ptr @llvm.returnaddress(i32 100)
  br label %202

202:                                              ; preds = %1, %200, %198, %196, %194, %192, %190, %188, %186, %184, %182, %180, %178, %176, %174, %172, %170, %168, %166, %164, %162, %160, %158, %156, %154, %152, %150, %148, %146, %144, %142, %140, %138, %136, %134, %132, %130, %128, %126, %124, %122, %120, %118, %116, %114, %112, %110, %108, %106, %104, %102, %100, %98, %96, %94, %92, %90, %88, %86, %84, %82, %80, %78, %76, %74, %72, %70, %68, %66, %64, %62, %60, %58, %56, %54, %52, %50, %48, %46, %44, %42, %40, %38, %36, %34, %32, %30, %28, %26, %24, %22, %20, %18, %16, %14, %12, %10, %8, %6, %4, %2
  %203 = phi ptr [ %3, %2 ], [ %5, %4 ], [ %7, %6 ], [ %9, %8 ], [ %11, %10 ], [ %13, %12 ], [ %15, %14 ], [ %17, %16 ], [ %19, %18 ], [ %21, %20 ], [ %23, %22 ], [ %25, %24 ], [ %27, %26 ], [ %29, %28 ], [ %31, %30 ], [ %33, %32 ], [ %35, %34 ], [ %37, %36 ], [ %39, %38 ], [ %41, %40 ], [ %43, %42 ], [ %45, %44 ], [ %47, %46 ], [ %49, %48 ], [ %51, %50 ], [ %53, %52 ], [ %55, %54 ], [ %57, %56 ], [ %59, %58 ], [ %61, %60 ], [ %63, %62 ], [ %65, %64 ], [ %67, %66 ], [ %69, %68 ], [ %71, %70 ], [ %73, %72 ], [ %75, %74 ], [ %77, %76 ], [ %79, %78 ], [ %81, %80 ], [ %83, %82 ], [ %85, %84 ], [ %87, %86 ], [ %89, %88 ], [ %91, %90 ], [ %93, %92 ], [ %95, %94 ], [ %97, %96 ], [ %99, %98 ], [ %101, %100 ], [ %103, %102 ], [ %105, %104 ], [ %107, %106 ], [ %109, %108 ], [ %111, %110 ], [ %113, %112 ], [ %115, %114 ], [ %117, %116 ], [ %119, %118 ], [ %121, %120 ], [ %123, %122 ], [ %125, %124 ], [ %127, %126 ], [ %129, %128 ], [ %131, %130 ], [ %133, %132 ], [ %135, %134 ], [ %137, %136 ], [ %139, %138 ], [ %141, %140 ], [ %143, %142 ], [ %145, %144 ], [ %147, %146 ], [ %149, %148 ], [ %151, %150 ], [ %153, %152 ], [ %155, %154 ], [ %157, %156 ], [ %159, %158 ], [ %161, %160 ], [ %163, %162 ], [ %165, %164 ], [ %167, %166 ], [ %169, %168 ], [ %171, %170 ], [ %173, %172 ], [ %175, %174 ], [ %177, %176 ], [ %179, %178 ], [ %181, %180 ], [ %183, %182 ], [ %185, %184 ], [ %187, %186 ], [ %189, %188 ], [ %191, %190 ], [ %193, %192 ], [ %195, %194 ], [ %197, %196 ], [ %199, %198 ], [ %201, %200 ], [ null, %1 ]
  ret ptr %203
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.returnaddress(i32 immarg) #1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  ret i32 0
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
