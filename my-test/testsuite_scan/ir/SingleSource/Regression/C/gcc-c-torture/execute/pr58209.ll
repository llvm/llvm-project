; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr58209.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr58209.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@buf = dso_local global [1024 x i64] zeroinitializer, align 8

; Function Attrs: nofree nosync nounwind memory(none) uwtable
define dso_local ptr @foo(i64 noundef %0) local_unnamed_addr #0 {
  %2 = icmp eq i64 %0, 0
  br i1 %2, label %3, label %5

3:                                                ; preds = %1, %5
  %4 = phi ptr [ %10, %5 ], [ @buf, %1 ]
  ret ptr %4

5:                                                ; preds = %1
  %6 = add nsw i64 %0, -1
  %7 = tail call ptr @foo(i64 noundef %6)
  %8 = ptrtoint ptr %7 to i64
  %9 = add i64 %8, 8
  %10 = inttoptr i64 %9 to ptr
  br label %3
}

; Function Attrs: nofree nosync nounwind memory(none) uwtable
define dso_local nonnull ptr @bar(i64 noundef %0) local_unnamed_addr #0 {
  %2 = icmp eq i64 %0, 0
  br i1 %2, label %7, label %3

3:                                                ; preds = %1
  %4 = add nsw i64 %0, -1
  %5 = tail call ptr @foo(i64 noundef %4)
  %6 = getelementptr inbounds nuw i8, ptr %5, i64 8
  br label %7

7:                                                ; preds = %1, %3
  %8 = phi ptr [ %6, %3 ], [ @buf, %1 ]
  ret ptr %8
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = tail call ptr @foo(i64 noundef 0)
  %2 = icmp eq ptr %1, @buf
  br i1 %2, label %179, label %186

3:                                                ; preds = %182
  %4 = tail call ptr @foo(i64 noundef 2)
  %5 = icmp eq ptr %4, getelementptr inbounds nuw (i8, ptr @buf, i64 16)
  br i1 %5, label %6, label %186

6:                                                ; preds = %3
  %7 = tail call ptr @foo(i64 noundef 1)
  %8 = getelementptr inbounds nuw i8, ptr %7, i64 8
  %9 = icmp eq ptr %8, %4
  br i1 %9, label %10, label %186

10:                                               ; preds = %6
  %11 = tail call ptr @foo(i64 noundef 3)
  %12 = icmp eq ptr %11, getelementptr inbounds nuw (i8, ptr @buf, i64 24)
  br i1 %12, label %13, label %186

13:                                               ; preds = %10
  %14 = tail call ptr @foo(i64 noundef 2)
  %15 = getelementptr inbounds nuw i8, ptr %14, i64 8
  %16 = icmp eq ptr %15, %11
  br i1 %16, label %17, label %186

17:                                               ; preds = %13
  %18 = tail call ptr @foo(i64 noundef 4)
  %19 = icmp eq ptr %18, getelementptr inbounds nuw (i8, ptr @buf, i64 32)
  br i1 %19, label %20, label %186

20:                                               ; preds = %17
  %21 = tail call ptr @foo(i64 noundef 3)
  %22 = getelementptr inbounds nuw i8, ptr %21, i64 8
  %23 = icmp eq ptr %22, %18
  br i1 %23, label %24, label %186

24:                                               ; preds = %20
  %25 = tail call ptr @foo(i64 noundef 5)
  %26 = icmp eq ptr %25, getelementptr inbounds nuw (i8, ptr @buf, i64 40)
  br i1 %26, label %27, label %186

27:                                               ; preds = %24
  %28 = tail call ptr @foo(i64 noundef 4)
  %29 = getelementptr inbounds nuw i8, ptr %28, i64 8
  %30 = icmp eq ptr %29, %25
  br i1 %30, label %31, label %186

31:                                               ; preds = %27
  %32 = tail call ptr @foo(i64 noundef 6)
  %33 = icmp eq ptr %32, getelementptr inbounds nuw (i8, ptr @buf, i64 48)
  br i1 %33, label %34, label %186

34:                                               ; preds = %31
  %35 = tail call ptr @foo(i64 noundef 5)
  %36 = getelementptr inbounds nuw i8, ptr %35, i64 8
  %37 = icmp eq ptr %36, %32
  br i1 %37, label %38, label %186

38:                                               ; preds = %34
  %39 = tail call ptr @foo(i64 noundef 7)
  %40 = icmp eq ptr %39, getelementptr inbounds nuw (i8, ptr @buf, i64 56)
  br i1 %40, label %41, label %186

41:                                               ; preds = %38
  %42 = tail call ptr @foo(i64 noundef 6)
  %43 = getelementptr inbounds nuw i8, ptr %42, i64 8
  %44 = icmp eq ptr %43, %39
  br i1 %44, label %45, label %186

45:                                               ; preds = %41
  %46 = tail call ptr @foo(i64 noundef 8)
  %47 = icmp eq ptr %46, getelementptr inbounds nuw (i8, ptr @buf, i64 64)
  br i1 %47, label %48, label %186

48:                                               ; preds = %45
  %49 = tail call ptr @foo(i64 noundef 7)
  %50 = getelementptr inbounds nuw i8, ptr %49, i64 8
  %51 = icmp eq ptr %50, %46
  br i1 %51, label %52, label %186

52:                                               ; preds = %48
  %53 = tail call ptr @foo(i64 noundef 9)
  %54 = icmp eq ptr %53, getelementptr inbounds nuw (i8, ptr @buf, i64 72)
  br i1 %54, label %55, label %186

55:                                               ; preds = %52
  %56 = tail call ptr @foo(i64 noundef 8)
  %57 = getelementptr inbounds nuw i8, ptr %56, i64 8
  %58 = icmp eq ptr %57, %53
  br i1 %58, label %59, label %186

59:                                               ; preds = %55
  %60 = tail call ptr @foo(i64 noundef 10)
  %61 = icmp eq ptr %60, getelementptr inbounds nuw (i8, ptr @buf, i64 80)
  br i1 %61, label %62, label %186

62:                                               ; preds = %59
  %63 = tail call ptr @foo(i64 noundef 9)
  %64 = getelementptr inbounds nuw i8, ptr %63, i64 8
  %65 = icmp eq ptr %64, %60
  br i1 %65, label %66, label %186

66:                                               ; preds = %62
  %67 = tail call ptr @foo(i64 noundef 11)
  %68 = icmp eq ptr %67, getelementptr inbounds nuw (i8, ptr @buf, i64 88)
  br i1 %68, label %69, label %186

69:                                               ; preds = %66
  %70 = tail call ptr @foo(i64 noundef 10)
  %71 = getelementptr inbounds nuw i8, ptr %70, i64 8
  %72 = icmp eq ptr %71, %67
  br i1 %72, label %73, label %186

73:                                               ; preds = %69
  %74 = tail call ptr @foo(i64 noundef 12)
  %75 = icmp eq ptr %74, getelementptr inbounds nuw (i8, ptr @buf, i64 96)
  br i1 %75, label %76, label %186

76:                                               ; preds = %73
  %77 = tail call ptr @foo(i64 noundef 11)
  %78 = getelementptr inbounds nuw i8, ptr %77, i64 8
  %79 = icmp eq ptr %78, %74
  br i1 %79, label %80, label %186

80:                                               ; preds = %76
  %81 = tail call ptr @foo(i64 noundef 13)
  %82 = icmp eq ptr %81, getelementptr inbounds nuw (i8, ptr @buf, i64 104)
  br i1 %82, label %83, label %186

83:                                               ; preds = %80
  %84 = tail call ptr @foo(i64 noundef 12)
  %85 = getelementptr inbounds nuw i8, ptr %84, i64 8
  %86 = icmp eq ptr %85, %81
  br i1 %86, label %87, label %186

87:                                               ; preds = %83
  %88 = tail call ptr @foo(i64 noundef 14)
  %89 = icmp eq ptr %88, getelementptr inbounds nuw (i8, ptr @buf, i64 112)
  br i1 %89, label %90, label %186

90:                                               ; preds = %87
  %91 = tail call ptr @foo(i64 noundef 13)
  %92 = getelementptr inbounds nuw i8, ptr %91, i64 8
  %93 = icmp eq ptr %92, %88
  br i1 %93, label %94, label %186

94:                                               ; preds = %90
  %95 = tail call ptr @foo(i64 noundef 15)
  %96 = icmp eq ptr %95, getelementptr inbounds nuw (i8, ptr @buf, i64 120)
  br i1 %96, label %97, label %186

97:                                               ; preds = %94
  %98 = tail call ptr @foo(i64 noundef 14)
  %99 = getelementptr inbounds nuw i8, ptr %98, i64 8
  %100 = icmp eq ptr %99, %95
  br i1 %100, label %101, label %186

101:                                              ; preds = %97
  %102 = tail call ptr @foo(i64 noundef 16)
  %103 = icmp eq ptr %102, getelementptr inbounds nuw (i8, ptr @buf, i64 128)
  br i1 %103, label %104, label %186

104:                                              ; preds = %101
  %105 = tail call ptr @foo(i64 noundef 15)
  %106 = getelementptr inbounds nuw i8, ptr %105, i64 8
  %107 = icmp eq ptr %106, %102
  br i1 %107, label %108, label %186

108:                                              ; preds = %104
  %109 = tail call ptr @foo(i64 noundef 17)
  %110 = icmp eq ptr %109, getelementptr inbounds nuw (i8, ptr @buf, i64 136)
  br i1 %110, label %111, label %186

111:                                              ; preds = %108
  %112 = tail call ptr @foo(i64 noundef 16)
  %113 = getelementptr inbounds nuw i8, ptr %112, i64 8
  %114 = icmp eq ptr %113, %109
  br i1 %114, label %115, label %186

115:                                              ; preds = %111
  %116 = tail call ptr @foo(i64 noundef 18)
  %117 = icmp eq ptr %116, getelementptr inbounds nuw (i8, ptr @buf, i64 144)
  br i1 %117, label %118, label %186

118:                                              ; preds = %115
  %119 = tail call ptr @foo(i64 noundef 17)
  %120 = getelementptr inbounds nuw i8, ptr %119, i64 8
  %121 = icmp eq ptr %120, %116
  br i1 %121, label %122, label %186

122:                                              ; preds = %118
  %123 = tail call ptr @foo(i64 noundef 19)
  %124 = icmp eq ptr %123, getelementptr inbounds nuw (i8, ptr @buf, i64 152)
  br i1 %124, label %125, label %186

125:                                              ; preds = %122
  %126 = tail call ptr @foo(i64 noundef 18)
  %127 = getelementptr inbounds nuw i8, ptr %126, i64 8
  %128 = icmp eq ptr %127, %123
  br i1 %128, label %129, label %186

129:                                              ; preds = %125
  %130 = tail call ptr @foo(i64 noundef 20)
  %131 = icmp eq ptr %130, getelementptr inbounds nuw (i8, ptr @buf, i64 160)
  br i1 %131, label %132, label %186

132:                                              ; preds = %129
  %133 = tail call ptr @foo(i64 noundef 19)
  %134 = getelementptr inbounds nuw i8, ptr %133, i64 8
  %135 = icmp eq ptr %134, %130
  br i1 %135, label %136, label %186

136:                                              ; preds = %132
  %137 = tail call ptr @foo(i64 noundef 21)
  %138 = icmp eq ptr %137, getelementptr inbounds nuw (i8, ptr @buf, i64 168)
  br i1 %138, label %139, label %186

139:                                              ; preds = %136
  %140 = tail call ptr @foo(i64 noundef 20)
  %141 = getelementptr inbounds nuw i8, ptr %140, i64 8
  %142 = icmp eq ptr %141, %137
  br i1 %142, label %143, label %186

143:                                              ; preds = %139
  %144 = tail call ptr @foo(i64 noundef 22)
  %145 = icmp eq ptr %144, getelementptr inbounds nuw (i8, ptr @buf, i64 176)
  br i1 %145, label %146, label %186

146:                                              ; preds = %143
  %147 = tail call ptr @foo(i64 noundef 21)
  %148 = getelementptr inbounds nuw i8, ptr %147, i64 8
  %149 = icmp eq ptr %148, %144
  br i1 %149, label %150, label %186

150:                                              ; preds = %146
  %151 = tail call ptr @foo(i64 noundef 23)
  %152 = icmp eq ptr %151, getelementptr inbounds nuw (i8, ptr @buf, i64 184)
  br i1 %152, label %153, label %186

153:                                              ; preds = %150
  %154 = tail call ptr @foo(i64 noundef 22)
  %155 = getelementptr inbounds nuw i8, ptr %154, i64 8
  %156 = icmp eq ptr %155, %151
  br i1 %156, label %157, label %186

157:                                              ; preds = %153
  %158 = tail call ptr @foo(i64 noundef 24)
  %159 = icmp eq ptr %158, getelementptr inbounds nuw (i8, ptr @buf, i64 192)
  br i1 %159, label %160, label %186

160:                                              ; preds = %157
  %161 = tail call ptr @foo(i64 noundef 23)
  %162 = getelementptr inbounds nuw i8, ptr %161, i64 8
  %163 = icmp eq ptr %162, %158
  br i1 %163, label %164, label %186

164:                                              ; preds = %160
  %165 = tail call ptr @foo(i64 noundef 25)
  %166 = icmp eq ptr %165, getelementptr inbounds nuw (i8, ptr @buf, i64 200)
  br i1 %166, label %167, label %186

167:                                              ; preds = %164
  %168 = tail call ptr @foo(i64 noundef 24)
  %169 = getelementptr inbounds nuw i8, ptr %168, i64 8
  %170 = icmp eq ptr %169, %165
  br i1 %170, label %171, label %186

171:                                              ; preds = %167
  %172 = tail call ptr @foo(i64 noundef 26)
  %173 = icmp eq ptr %172, getelementptr inbounds nuw (i8, ptr @buf, i64 208)
  br i1 %173, label %174, label %186

174:                                              ; preds = %171
  %175 = tail call ptr @foo(i64 noundef 25)
  %176 = getelementptr inbounds nuw i8, ptr %175, i64 8
  %177 = icmp eq ptr %176, %172
  br i1 %177, label %178, label %186

178:                                              ; preds = %174
  ret i32 0

179:                                              ; preds = %0
  %180 = tail call ptr @foo(i64 noundef 1)
  %181 = icmp eq ptr %180, getelementptr inbounds nuw (i8, ptr @buf, i64 8)
  br i1 %181, label %182, label %186

182:                                              ; preds = %179
  %183 = tail call ptr @foo(i64 noundef 0)
  %184 = getelementptr inbounds nuw i8, ptr %183, i64 8
  %185 = icmp eq ptr %184, %180
  br i1 %185, label %3, label %186

186:                                              ; preds = %182, %179, %3, %6, %10, %13, %17, %20, %24, %27, %31, %34, %38, %41, %45, %48, %52, %55, %59, %62, %66, %69, %73, %76, %80, %83, %87, %90, %94, %97, %101, %104, %108, %111, %115, %118, %122, %125, %129, %132, %136, %139, %143, %146, %150, %153, %157, %160, %164, %167, %171, %174, %0
  tail call void @abort() #3
  unreachable
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

attributes #0 = { nofree nosync nounwind memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
