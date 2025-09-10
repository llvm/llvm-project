; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr64979.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr64979.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.__va_list = type { ptr, ptr, ptr, i32, i32 }

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @bar(i32 %0, ptr noundef captures(address_is_null) %1) local_unnamed_addr #0 {
  %3 = icmp eq ptr %1, null
  br i1 %3, label %206, label %4

4:                                                ; preds = %2
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %6 = load i32, ptr %5, align 8
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %8 = icmp sgt i32 %6, -1
  br i1 %8, label %180, label %173

9:                                                ; preds = %184
  %10 = icmp sgt i32 %185, -1
  br i1 %10, label %18, label %11

11:                                               ; preds = %9
  %12 = add nsw i32 %185, 8
  store i32 %12, ptr %5, align 8
  %13 = icmp samesign ult i32 %185, -7
  br i1 %13, label %14, label %18

14:                                               ; preds = %11
  %15 = load ptr, ptr %7, align 8
  %16 = sext i32 %185 to i64
  %17 = getelementptr inbounds i8, ptr %15, i64 %16
  br label %22

18:                                               ; preds = %11, %9
  %19 = phi i32 [ %12, %11 ], [ %185, %9 ]
  %20 = load ptr, ptr %1, align 8
  %21 = getelementptr inbounds nuw i8, ptr %20, i64 8
  store ptr %21, ptr %1, align 8
  br label %22

22:                                               ; preds = %18, %14
  %23 = phi i32 [ %12, %14 ], [ %19, %18 ]
  %24 = phi ptr [ %17, %14 ], [ %20, %18 ]
  %25 = load i32, ptr %24, align 8, !tbaa !6
  %26 = icmp eq i32 %25, 1
  br i1 %26, label %27, label %189

27:                                               ; preds = %22
  %28 = icmp sgt i32 %23, -1
  br i1 %28, label %36, label %29

29:                                               ; preds = %27
  %30 = add nsw i32 %23, 8
  store i32 %30, ptr %5, align 8
  %31 = icmp samesign ult i32 %23, -7
  br i1 %31, label %32, label %36

32:                                               ; preds = %29
  %33 = load ptr, ptr %7, align 8
  %34 = sext i32 %23 to i64
  %35 = getelementptr inbounds i8, ptr %33, i64 %34
  br label %40

36:                                               ; preds = %29, %27
  %37 = phi i32 [ %30, %29 ], [ %23, %27 ]
  %38 = load ptr, ptr %1, align 8
  %39 = getelementptr inbounds nuw i8, ptr %38, i64 8
  store ptr %39, ptr %1, align 8
  br label %40

40:                                               ; preds = %36, %32
  %41 = phi i32 [ %30, %32 ], [ %37, %36 ]
  %42 = phi ptr [ %35, %32 ], [ %38, %36 ]
  %43 = load i32, ptr %42, align 8, !tbaa !6
  %44 = icmp eq i32 %43, 2
  br i1 %44, label %45, label %189

45:                                               ; preds = %40
  %46 = icmp sgt i32 %41, -1
  br i1 %46, label %54, label %47

47:                                               ; preds = %45
  %48 = add nsw i32 %41, 8
  store i32 %48, ptr %5, align 8
  %49 = icmp samesign ult i32 %41, -7
  br i1 %49, label %50, label %54

50:                                               ; preds = %47
  %51 = load ptr, ptr %7, align 8
  %52 = sext i32 %41 to i64
  %53 = getelementptr inbounds i8, ptr %51, i64 %52
  br label %58

54:                                               ; preds = %47, %45
  %55 = phi i32 [ %48, %47 ], [ %41, %45 ]
  %56 = load ptr, ptr %1, align 8
  %57 = getelementptr inbounds nuw i8, ptr %56, i64 8
  store ptr %57, ptr %1, align 8
  br label %58

58:                                               ; preds = %54, %50
  %59 = phi i32 [ %48, %50 ], [ %55, %54 ]
  %60 = phi ptr [ %53, %50 ], [ %56, %54 ]
  %61 = load i32, ptr %60, align 8, !tbaa !6
  %62 = icmp eq i32 %61, 3
  br i1 %62, label %63, label %189

63:                                               ; preds = %58
  %64 = icmp sgt i32 %59, -1
  br i1 %64, label %72, label %65

65:                                               ; preds = %63
  %66 = add nsw i32 %59, 8
  store i32 %66, ptr %5, align 8
  %67 = icmp samesign ult i32 %59, -7
  br i1 %67, label %68, label %72

68:                                               ; preds = %65
  %69 = load ptr, ptr %7, align 8
  %70 = sext i32 %59 to i64
  %71 = getelementptr inbounds i8, ptr %69, i64 %70
  br label %76

72:                                               ; preds = %65, %63
  %73 = phi i32 [ %66, %65 ], [ %59, %63 ]
  %74 = load ptr, ptr %1, align 8
  %75 = getelementptr inbounds nuw i8, ptr %74, i64 8
  store ptr %75, ptr %1, align 8
  br label %76

76:                                               ; preds = %72, %68
  %77 = phi i32 [ %66, %68 ], [ %73, %72 ]
  %78 = phi ptr [ %71, %68 ], [ %74, %72 ]
  %79 = load i32, ptr %78, align 8, !tbaa !6
  %80 = icmp eq i32 %79, 4
  br i1 %80, label %81, label %189

81:                                               ; preds = %76
  %82 = icmp sgt i32 %77, -1
  br i1 %82, label %90, label %83

83:                                               ; preds = %81
  %84 = add nsw i32 %77, 8
  store i32 %84, ptr %5, align 8
  %85 = icmp samesign ult i32 %77, -7
  br i1 %85, label %86, label %90

86:                                               ; preds = %83
  %87 = load ptr, ptr %7, align 8
  %88 = sext i32 %77 to i64
  %89 = getelementptr inbounds i8, ptr %87, i64 %88
  br label %94

90:                                               ; preds = %83, %81
  %91 = phi i32 [ %84, %83 ], [ %77, %81 ]
  %92 = load ptr, ptr %1, align 8
  %93 = getelementptr inbounds nuw i8, ptr %92, i64 8
  store ptr %93, ptr %1, align 8
  br label %94

94:                                               ; preds = %90, %86
  %95 = phi i32 [ %84, %86 ], [ %91, %90 ]
  %96 = phi ptr [ %89, %86 ], [ %92, %90 ]
  %97 = load i32, ptr %96, align 8, !tbaa !6
  %98 = icmp eq i32 %97, 5
  br i1 %98, label %99, label %189

99:                                               ; preds = %94
  %100 = icmp sgt i32 %95, -1
  br i1 %100, label %108, label %101

101:                                              ; preds = %99
  %102 = add nsw i32 %95, 8
  store i32 %102, ptr %5, align 8
  %103 = icmp samesign ult i32 %95, -7
  br i1 %103, label %104, label %108

104:                                              ; preds = %101
  %105 = load ptr, ptr %7, align 8
  %106 = sext i32 %95 to i64
  %107 = getelementptr inbounds i8, ptr %105, i64 %106
  br label %112

108:                                              ; preds = %101, %99
  %109 = phi i32 [ %102, %101 ], [ %95, %99 ]
  %110 = load ptr, ptr %1, align 8
  %111 = getelementptr inbounds nuw i8, ptr %110, i64 8
  store ptr %111, ptr %1, align 8
  br label %112

112:                                              ; preds = %108, %104
  %113 = phi i32 [ %102, %104 ], [ %109, %108 ]
  %114 = phi ptr [ %107, %104 ], [ %110, %108 ]
  %115 = load i32, ptr %114, align 8, !tbaa !6
  %116 = icmp eq i32 %115, 6
  br i1 %116, label %117, label %189

117:                                              ; preds = %112
  %118 = icmp sgt i32 %113, -1
  br i1 %118, label %126, label %119

119:                                              ; preds = %117
  %120 = add nsw i32 %113, 8
  store i32 %120, ptr %5, align 8
  %121 = icmp samesign ult i32 %113, -7
  br i1 %121, label %122, label %126

122:                                              ; preds = %119
  %123 = load ptr, ptr %7, align 8
  %124 = sext i32 %113 to i64
  %125 = getelementptr inbounds i8, ptr %123, i64 %124
  br label %130

126:                                              ; preds = %119, %117
  %127 = phi i32 [ %120, %119 ], [ %113, %117 ]
  %128 = load ptr, ptr %1, align 8
  %129 = getelementptr inbounds nuw i8, ptr %128, i64 8
  store ptr %129, ptr %1, align 8
  br label %130

130:                                              ; preds = %126, %122
  %131 = phi i32 [ %120, %122 ], [ %127, %126 ]
  %132 = phi ptr [ %125, %122 ], [ %128, %126 ]
  %133 = load i32, ptr %132, align 8, !tbaa !6
  %134 = icmp eq i32 %133, 7
  br i1 %134, label %135, label %189

135:                                              ; preds = %130
  %136 = icmp sgt i32 %131, -1
  br i1 %136, label %144, label %137

137:                                              ; preds = %135
  %138 = add nsw i32 %131, 8
  store i32 %138, ptr %5, align 8
  %139 = icmp samesign ult i32 %131, -7
  br i1 %139, label %140, label %144

140:                                              ; preds = %137
  %141 = load ptr, ptr %7, align 8
  %142 = sext i32 %131 to i64
  %143 = getelementptr inbounds i8, ptr %141, i64 %142
  br label %148

144:                                              ; preds = %137, %135
  %145 = phi i32 [ %138, %137 ], [ %131, %135 ]
  %146 = load ptr, ptr %1, align 8
  %147 = getelementptr inbounds nuw i8, ptr %146, i64 8
  store ptr %147, ptr %1, align 8
  br label %148

148:                                              ; preds = %144, %140
  %149 = phi i32 [ %138, %140 ], [ %145, %144 ]
  %150 = phi ptr [ %143, %140 ], [ %146, %144 ]
  %151 = load i32, ptr %150, align 8, !tbaa !6
  %152 = icmp eq i32 %151, 8
  br i1 %152, label %153, label %189

153:                                              ; preds = %148
  %154 = icmp sgt i32 %149, -1
  br i1 %154, label %162, label %155

155:                                              ; preds = %153
  %156 = add nsw i32 %149, 8
  store i32 %156, ptr %5, align 8
  %157 = icmp samesign ult i32 %149, -7
  br i1 %157, label %158, label %162

158:                                              ; preds = %155
  %159 = load ptr, ptr %7, align 8
  %160 = sext i32 %149 to i64
  %161 = getelementptr inbounds i8, ptr %159, i64 %160
  br label %165

162:                                              ; preds = %155, %153
  %163 = load ptr, ptr %1, align 8
  %164 = getelementptr inbounds nuw i8, ptr %163, i64 8
  store ptr %164, ptr %1, align 8
  br label %165

165:                                              ; preds = %162, %158
  %166 = phi ptr [ %161, %158 ], [ %163, %162 ]
  %167 = load i32, ptr %166, align 8, !tbaa !6
  %168 = icmp eq i32 %167, 9
  br i1 %168, label %169, label %189

169:                                              ; preds = %165
  %170 = getelementptr inbounds nuw i8, ptr %1, i64 28
  %171 = load i32, ptr %170, align 4
  %172 = icmp sgt i32 %171, -1
  br i1 %172, label %198, label %190

173:                                              ; preds = %4
  %174 = add nsw i32 %6, 8
  store i32 %174, ptr %5, align 8
  %175 = icmp samesign ult i32 %6, -7
  br i1 %175, label %176, label %180

176:                                              ; preds = %173
  %177 = load ptr, ptr %7, align 8
  %178 = sext i32 %6 to i64
  %179 = getelementptr inbounds i8, ptr %177, i64 %178
  br label %184

180:                                              ; preds = %173, %4
  %181 = phi i32 [ %174, %173 ], [ %6, %4 ]
  %182 = load ptr, ptr %1, align 8
  %183 = getelementptr inbounds nuw i8, ptr %182, i64 8
  store ptr %183, ptr %1, align 8
  br label %184

184:                                              ; preds = %180, %176
  %185 = phi i32 [ %174, %176 ], [ %181, %180 ]
  %186 = phi ptr [ %179, %176 ], [ %182, %180 ]
  %187 = load i32, ptr %186, align 8, !tbaa !6
  %188 = icmp eq i32 %187, 0
  br i1 %188, label %9, label %189

189:                                              ; preds = %165, %148, %130, %112, %94, %76, %58, %40, %22, %184
  tail call void @abort() #5
  unreachable

190:                                              ; preds = %169
  %191 = add nsw i32 %171, 16
  store i32 %191, ptr %170, align 4
  %192 = icmp samesign ult i32 %171, -15
  br i1 %192, label %193, label %198

193:                                              ; preds = %190
  %194 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %195 = load ptr, ptr %194, align 8
  %196 = sext i32 %171 to i64
  %197 = getelementptr inbounds i8, ptr %195, i64 %196
  br label %201

198:                                              ; preds = %190, %169
  %199 = load ptr, ptr %1, align 8
  %200 = getelementptr inbounds nuw i8, ptr %199, i64 8
  store ptr %200, ptr %1, align 8
  br label %201

201:                                              ; preds = %198, %193
  %202 = phi ptr [ %197, %193 ], [ %199, %198 ]
  %203 = load double, ptr %202, align 8, !tbaa !10
  %204 = fcmp une double %203, 5.000000e-01
  br i1 %204, label %205, label %206

205:                                              ; preds = %201
  tail call void @abort() #5
  unreachable

206:                                              ; preds = %201, %2
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @foo(i32 %0, ...) local_unnamed_addr #0 {
  %2 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #6
  call void @llvm.va_start.p0(ptr nonnull %2)
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %4 = load i32, ptr %3, align 8
  %5 = icmp sgt i32 %4, -1
  br i1 %5, label %14, label %6

6:                                                ; preds = %1
  %7 = add nsw i32 %4, 8
  store i32 %7, ptr %3, align 8
  %8 = icmp samesign ult i32 %4, -7
  br i1 %8, label %9, label %14

9:                                                ; preds = %6
  %10 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %11 = load ptr, ptr %10, align 8
  %12 = sext i32 %4 to i64
  %13 = getelementptr inbounds i8, ptr %11, i64 %12
  br label %17

14:                                               ; preds = %6, %1
  %15 = load ptr, ptr %2, align 8
  %16 = getelementptr inbounds nuw i8, ptr %15, i64 8
  store ptr %16, ptr %2, align 8
  br label %17

17:                                               ; preds = %14, %9
  %18 = phi ptr [ %13, %9 ], [ %15, %14 ]
  %19 = load i32, ptr %18, align 8, !tbaa !6
  %20 = icmp eq i32 %19, 0
  %21 = select i1 %20, ptr null, ptr %2
  call void @bar(i32 poison, ptr noundef %21)
  call void @llvm.va_end.p0(ptr nonnull %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #6
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_start.p0(ptr) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_end.p0(ptr) #3

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #4 {
  tail call void (i32, ...) @foo(i32 poison, i32 noundef 1, i32 noundef 0, i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, i32 noundef 6, i32 noundef 7, i32 noundef 8, i32 noundef 9, double noundef 5.000000e-01)
  ret i32 0
}

attributes #0 = { nofree noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #4 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { noreturn nounwind }
attributes #6 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"double", !8, i64 0}
