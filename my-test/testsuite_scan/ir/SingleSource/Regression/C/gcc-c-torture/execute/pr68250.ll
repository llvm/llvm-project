; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr68250.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr68250.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@b = dso_local local_unnamed_addr global i8 0, align 1
@a = dso_local local_unnamed_addr global i8 0, align 1
@o = dso_local local_unnamed_addr global i8 0, align 4
@d = dso_local local_unnamed_addr global i16 0, align 4
@n = dso_local local_unnamed_addr global i16 0, align 4
@j = dso_local local_unnamed_addr global i32 0, align 4
@c = dso_local local_unnamed_addr global i16 0, align 4
@m = dso_local local_unnamed_addr global i8 0, align 4
@f = dso_local local_unnamed_addr global i32 0, align 4
@l = dso_local local_unnamed_addr global i8 0, align 4
@h = dso_local local_unnamed_addr global i8 0, align 4
@k = dso_local local_unnamed_addr global i8 0, align 4
@e = dso_local local_unnamed_addr global i32 0, align 4
@q = dso_local local_unnamed_addr global i32 0, align 4
@g = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn1() local_unnamed_addr #0 {
  %1 = load i8, ptr @o, align 4, !tbaa !6
  %2 = sext i8 %1 to i32
  %3 = icmp sgt i8 %1, 0
  %4 = load i16, ptr @d, align 4, !tbaa !9
  br i1 %3, label %12, label %5

5:                                                ; preds = %0
  %6 = sext i16 %4 to i32
  %7 = lshr i32 1, %2
  %8 = icmp slt i32 %7, %6
  br i1 %8, label %12, label %9

9:                                                ; preds = %5
  %10 = shl nuw nsw i32 %6, %2
  %11 = trunc i32 %10 to i16
  br label %12

12:                                               ; preds = %0, %5, %9
  %13 = phi i16 [ %11, %9 ], [ %4, %5 ], [ %4, %0 ]
  store i16 %13, ptr @n, align 4, !tbaa !9
  %14 = load i32, ptr @j, align 4, !tbaa !11
  %15 = icmp eq i32 %14, 0
  br i1 %15, label %65, label %16

16:                                               ; preds = %12
  %17 = load i8, ptr @m, align 4
  %18 = load i16, ptr @c, align 4, !tbaa !9
  %19 = freeze i16 %18
  %20 = icmp ne i16 %19, 0
  %21 = freeze i8 %17
  %22 = icmp ne i8 %21, 0
  %23 = sub i32 0, %14
  %24 = icmp ult i32 %23, 8
  br i1 %24, label %53, label %25

25:                                               ; preds = %16
  %26 = icmp ult i32 %23, 64
  br i1 %26, label %40, label %27

27:                                               ; preds = %25
  %28 = and i32 %23, -64
  br label %29

29:                                               ; preds = %29, %27
  %30 = phi i32 [ 0, %27 ], [ %31, %29 ]
  %31 = add nuw i32 %30, 64
  %32 = icmp eq i32 %31, %28
  br i1 %32, label %33, label %29, !llvm.loop !13

33:                                               ; preds = %29
  %34 = or i1 %20, %22
  %35 = icmp eq i32 %28, %23
  br i1 %35, label %62, label %36

36:                                               ; preds = %33
  %37 = add i32 %14, %28
  %38 = and i32 %23, 56
  %39 = icmp eq i32 %38, 0
  br i1 %39, label %53, label %40

40:                                               ; preds = %36, %25
  %41 = phi i32 [ %28, %36 ], [ 0, %25 ]
  %42 = phi i1 [ %34, %36 ], [ %22, %25 ]
  %43 = and i32 %23, -8
  %44 = add i32 %14, %43
  br label %45

45:                                               ; preds = %45, %40
  %46 = phi i32 [ %41, %40 ], [ %47, %45 ]
  %47 = add nuw i32 %46, 8
  %48 = icmp eq i32 %47, %43
  br i1 %48, label %49, label %45, !llvm.loop !17

49:                                               ; preds = %45
  %50 = or i1 %42, %20
  %51 = or i1 %50, %22
  %52 = icmp eq i32 %43, %23
  br i1 %52, label %62, label %53

53:                                               ; preds = %36, %49, %16
  %54 = phi i32 [ %14, %16 ], [ %37, %36 ], [ %44, %49 ]
  %55 = phi i1 [ %22, %16 ], [ %34, %36 ], [ %51, %49 ]
  br label %56

56:                                               ; preds = %53, %56
  %57 = phi i32 [ %60, %56 ], [ %54, %53 ]
  %58 = phi i1 [ %59, %56 ], [ %55, %53 ]
  %59 = select i1 %20, i1 true, i1 %58
  %60 = add nsw i32 %57, 1
  %61 = icmp eq i32 %60, 0
  br i1 %61, label %62, label %56, !llvm.loop !18

62:                                               ; preds = %56, %49, %33
  %63 = phi i1 [ %34, %33 ], [ %51, %49 ], [ %59, %56 ]
  %64 = zext i1 %63 to i8
  store i8 %64, ptr @m, align 4, !tbaa !6
  store i32 0, ptr @j, align 4, !tbaa !11
  br label %65

65:                                               ; preds = %62, %12
  %66 = load i32, ptr @f, align 4, !tbaa !11
  %67 = trunc i32 %66 to i8
  %68 = add i8 %67, 1
  store i8 %68, ptr @l, align 4, !tbaa !6
  %69 = icmp slt i32 %66, 1
  br i1 %69, label %70, label %73

70:                                               ; preds = %65
  %71 = load i8, ptr @h, align 4, !tbaa !6
  %72 = add i8 %71, 1
  store i8 %72, ptr @k, align 4, !tbaa !6
  store i32 1, ptr @f, align 4, !tbaa !11
  br label %73

73:                                               ; preds = %70, %65
  ret void
}

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @fn2(i32 noundef %0) local_unnamed_addr #1 {
  %2 = icmp eq i32 %0, 1
  br i1 %2, label %4, label %3

3:                                                ; preds = %1
  tail call void @abort() #4
  unreachable

4:                                                ; preds = %1
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #3 {
  %1 = load i32, ptr @e, align 4, !tbaa !11
  %2 = load i32, ptr @j, align 4
  %3 = load i8, ptr @m, align 4
  %4 = freeze i8 %3
  %5 = load i32, ptr @f, align 4
  %6 = load i8, ptr @k, align 4
  %7 = icmp slt i32 %1, 1
  br i1 %7, label %8, label %232

8:                                                ; preds = %0
  %9 = load i8, ptr @o, align 4, !tbaa !6
  %10 = sext i8 %9 to i32
  %11 = icmp sgt i8 %9, 0
  %12 = load i16, ptr @d, align 4, !tbaa !9
  %13 = sext i16 %12 to i32
  %14 = shl nuw nsw i32 %13, %10
  %15 = trunc i32 %14 to i16
  %16 = load i16, ptr @c, align 4
  %17 = freeze i16 %16
  %18 = icmp ne i16 %17, 0
  %19 = load i8, ptr @h, align 4
  %20 = add i8 %19, 1
  %21 = load i32, ptr @q, align 4, !tbaa !11
  br i1 %11, label %22, label %89

22:                                               ; preds = %8, %86
  %23 = phi i8 [ %82, %86 ], [ undef, %8 ]
  %24 = phi i32 [ %87, %86 ], [ %1, %8 ]
  %25 = phi i32 [ 0, %86 ], [ %2, %8 ]
  %26 = phi i8 [ %75, %86 ], [ %4, %8 ]
  %27 = phi i32 [ %80, %86 ], [ %5, %8 ]
  %28 = phi i8 [ %79, %86 ], [ %6, %8 ]
  %29 = icmp eq i32 %25, 0
  br i1 %29, label %74, label %30

30:                                               ; preds = %22
  %31 = icmp ne i8 %26, 0
  %32 = sub i32 0, %25
  %33 = icmp ult i32 %32, 8
  br i1 %33, label %62, label %34

34:                                               ; preds = %30
  %35 = icmp ult i32 %32, 64
  br i1 %35, label %49, label %36

36:                                               ; preds = %34
  %37 = and i32 %32, -64
  br label %38

38:                                               ; preds = %38, %36
  %39 = phi i32 [ 0, %36 ], [ %40, %38 ]
  %40 = add nuw i32 %39, 64
  %41 = icmp eq i32 %40, %37
  br i1 %41, label %42, label %38, !llvm.loop !19

42:                                               ; preds = %38
  %43 = select i1 %18, i1 true, i1 %31
  %44 = icmp eq i32 %37, %32
  br i1 %44, label %71, label %45

45:                                               ; preds = %42
  %46 = add i32 %25, %37
  %47 = and i32 %32, 56
  %48 = icmp eq i32 %47, 0
  br i1 %48, label %62, label %49

49:                                               ; preds = %45, %34
  %50 = phi i32 [ %37, %45 ], [ 0, %34 ]
  %51 = phi i1 [ %43, %45 ], [ %31, %34 ]
  %52 = and i32 %32, -8
  %53 = add i32 %25, %52
  br label %54

54:                                               ; preds = %54, %49
  %55 = phi i32 [ %50, %49 ], [ %56, %54 ]
  %56 = add nuw i32 %55, 8
  %57 = icmp eq i32 %56, %52
  br i1 %57, label %58, label %54, !llvm.loop !20

58:                                               ; preds = %54
  %59 = or i1 %51, %18
  %60 = or i1 %59, %31
  %61 = icmp eq i32 %52, %32
  br i1 %61, label %71, label %62

62:                                               ; preds = %45, %58, %30
  %63 = phi i32 [ %25, %30 ], [ %46, %45 ], [ %53, %58 ]
  %64 = phi i1 [ %31, %30 ], [ %43, %45 ], [ %60, %58 ]
  br label %65

65:                                               ; preds = %62, %65
  %66 = phi i32 [ %69, %65 ], [ %63, %62 ]
  %67 = phi i1 [ %68, %65 ], [ %64, %62 ]
  %68 = select i1 %18, i1 true, i1 %67
  %69 = add nsw i32 %66, 1
  %70 = icmp eq i32 %69, 0
  br i1 %70, label %71, label %65, !llvm.loop !21

71:                                               ; preds = %65, %58, %42
  %72 = phi i1 [ %43, %42 ], [ %60, %58 ], [ %68, %65 ]
  %73 = zext i1 %72 to i8
  store i8 %73, ptr @m, align 4, !tbaa !6
  store i32 0, ptr @j, align 4, !tbaa !11
  br label %74

74:                                               ; preds = %71, %22
  %75 = phi i8 [ %73, %71 ], [ %26, %22 ]
  %76 = icmp slt i32 %27, 1
  br i1 %76, label %77, label %78

77:                                               ; preds = %74
  store i8 %20, ptr @k, align 4, !tbaa !6
  store i32 1, ptr @f, align 4, !tbaa !11
  br label %78

78:                                               ; preds = %77, %74
  %79 = phi i8 [ %28, %74 ], [ %20, %77 ]
  %80 = phi i32 [ %27, %74 ], [ 1, %77 ]
  %81 = icmp eq i8 %79, 0
  %82 = select i1 %81, i8 %23, i8 %79
  %83 = sext i8 %82 to i32
  %84 = icmp slt i32 %21, %83
  br i1 %84, label %85, label %86

85:                                               ; preds = %78
  store i32 0, ptr @g, align 4, !tbaa !11
  br label %86

86:                                               ; preds = %85, %78
  %87 = add nsw i32 %24, 1
  %88 = icmp eq i32 %24, 0
  br i1 %88, label %226, label %22, !llvm.loop !22

89:                                               ; preds = %8
  %90 = lshr i32 1, %10
  %91 = icmp slt i32 %90, %13
  br i1 %91, label %92, label %159

92:                                               ; preds = %89, %156
  %93 = phi i8 [ %152, %156 ], [ undef, %89 ]
  %94 = phi i32 [ %157, %156 ], [ %1, %89 ]
  %95 = phi i32 [ 0, %156 ], [ %2, %89 ]
  %96 = phi i8 [ %145, %156 ], [ %4, %89 ]
  %97 = phi i32 [ %150, %156 ], [ %5, %89 ]
  %98 = phi i8 [ %149, %156 ], [ %6, %89 ]
  %99 = icmp eq i32 %95, 0
  br i1 %99, label %144, label %100

100:                                              ; preds = %92
  %101 = icmp ne i8 %96, 0
  %102 = sub i32 0, %95
  %103 = icmp ult i32 %102, 8
  br i1 %103, label %132, label %104

104:                                              ; preds = %100
  %105 = icmp ult i32 %102, 64
  br i1 %105, label %119, label %106

106:                                              ; preds = %104
  %107 = and i32 %102, -64
  br label %108

108:                                              ; preds = %108, %106
  %109 = phi i32 [ 0, %106 ], [ %110, %108 ]
  %110 = add nuw i32 %109, 64
  %111 = icmp eq i32 %110, %107
  br i1 %111, label %112, label %108, !llvm.loop !23

112:                                              ; preds = %108
  %113 = select i1 %18, i1 true, i1 %101
  %114 = icmp eq i32 %107, %102
  br i1 %114, label %141, label %115

115:                                              ; preds = %112
  %116 = add i32 %95, %107
  %117 = and i32 %102, 56
  %118 = icmp eq i32 %117, 0
  br i1 %118, label %132, label %119

119:                                              ; preds = %115, %104
  %120 = phi i32 [ %107, %115 ], [ 0, %104 ]
  %121 = phi i1 [ %113, %115 ], [ %101, %104 ]
  %122 = and i32 %102, -8
  %123 = add i32 %95, %122
  br label %124

124:                                              ; preds = %124, %119
  %125 = phi i32 [ %120, %119 ], [ %126, %124 ]
  %126 = add nuw i32 %125, 8
  %127 = icmp eq i32 %126, %122
  br i1 %127, label %128, label %124, !llvm.loop !24

128:                                              ; preds = %124
  %129 = or i1 %121, %18
  %130 = or i1 %129, %101
  %131 = icmp eq i32 %122, %102
  br i1 %131, label %141, label %132

132:                                              ; preds = %115, %128, %100
  %133 = phi i32 [ %95, %100 ], [ %116, %115 ], [ %123, %128 ]
  %134 = phi i1 [ %101, %100 ], [ %113, %115 ], [ %130, %128 ]
  br label %135

135:                                              ; preds = %132, %135
  %136 = phi i32 [ %139, %135 ], [ %133, %132 ]
  %137 = phi i1 [ %138, %135 ], [ %134, %132 ]
  %138 = select i1 %18, i1 true, i1 %137
  %139 = add nsw i32 %136, 1
  %140 = icmp eq i32 %139, 0
  br i1 %140, label %141, label %135, !llvm.loop !25

141:                                              ; preds = %135, %128, %112
  %142 = phi i1 [ %113, %112 ], [ %130, %128 ], [ %138, %135 ]
  %143 = zext i1 %142 to i8
  store i8 %143, ptr @m, align 4, !tbaa !6
  store i32 0, ptr @j, align 4, !tbaa !11
  br label %144

144:                                              ; preds = %141, %92
  %145 = phi i8 [ %143, %141 ], [ %96, %92 ]
  %146 = icmp slt i32 %97, 1
  br i1 %146, label %147, label %148

147:                                              ; preds = %144
  store i8 %20, ptr @k, align 4, !tbaa !6
  store i32 1, ptr @f, align 4, !tbaa !11
  br label %148

148:                                              ; preds = %147, %144
  %149 = phi i8 [ %98, %144 ], [ %20, %147 ]
  %150 = phi i32 [ %97, %144 ], [ 1, %147 ]
  %151 = icmp eq i8 %149, 0
  %152 = select i1 %151, i8 %93, i8 %149
  %153 = sext i8 %152 to i32
  %154 = icmp slt i32 %21, %153
  br i1 %154, label %155, label %156

155:                                              ; preds = %148
  store i32 0, ptr @g, align 4, !tbaa !11
  br label %156

156:                                              ; preds = %155, %148
  %157 = add nsw i32 %94, 1
  %158 = icmp eq i32 %94, 0
  br i1 %158, label %226, label %92, !llvm.loop !22

159:                                              ; preds = %89, %223
  %160 = phi i8 [ %219, %223 ], [ undef, %89 ]
  %161 = phi i32 [ %224, %223 ], [ %1, %89 ]
  %162 = phi i32 [ 0, %223 ], [ %2, %89 ]
  %163 = phi i8 [ %212, %223 ], [ %4, %89 ]
  %164 = phi i32 [ %217, %223 ], [ %5, %89 ]
  %165 = phi i8 [ %216, %223 ], [ %6, %89 ]
  %166 = icmp eq i32 %162, 0
  br i1 %166, label %211, label %167

167:                                              ; preds = %159
  %168 = icmp ne i8 %163, 0
  %169 = sub i32 0, %162
  %170 = icmp ult i32 %169, 8
  br i1 %170, label %199, label %171

171:                                              ; preds = %167
  %172 = icmp ult i32 %169, 64
  br i1 %172, label %186, label %173

173:                                              ; preds = %171
  %174 = and i32 %169, -64
  br label %175

175:                                              ; preds = %175, %173
  %176 = phi i32 [ 0, %173 ], [ %177, %175 ]
  %177 = add nuw i32 %176, 64
  %178 = icmp eq i32 %177, %174
  br i1 %178, label %179, label %175, !llvm.loop !26

179:                                              ; preds = %175
  %180 = select i1 %18, i1 true, i1 %168
  %181 = icmp eq i32 %174, %169
  br i1 %181, label %208, label %182

182:                                              ; preds = %179
  %183 = add i32 %162, %174
  %184 = and i32 %169, 56
  %185 = icmp eq i32 %184, 0
  br i1 %185, label %199, label %186

186:                                              ; preds = %182, %171
  %187 = phi i32 [ %174, %182 ], [ 0, %171 ]
  %188 = phi i1 [ %180, %182 ], [ %168, %171 ]
  %189 = and i32 %169, -8
  %190 = add i32 %162, %189
  br label %191

191:                                              ; preds = %191, %186
  %192 = phi i32 [ %187, %186 ], [ %193, %191 ]
  %193 = add nuw i32 %192, 8
  %194 = icmp eq i32 %193, %189
  br i1 %194, label %195, label %191, !llvm.loop !27

195:                                              ; preds = %191
  %196 = or i1 %188, %18
  %197 = or i1 %196, %168
  %198 = icmp eq i32 %189, %169
  br i1 %198, label %208, label %199

199:                                              ; preds = %182, %195, %167
  %200 = phi i32 [ %162, %167 ], [ %183, %182 ], [ %190, %195 ]
  %201 = phi i1 [ %168, %167 ], [ %180, %182 ], [ %197, %195 ]
  br label %202

202:                                              ; preds = %199, %202
  %203 = phi i32 [ %206, %202 ], [ %200, %199 ]
  %204 = phi i1 [ %205, %202 ], [ %201, %199 ]
  %205 = select i1 %18, i1 true, i1 %204
  %206 = add nsw i32 %203, 1
  %207 = icmp eq i32 %206, 0
  br i1 %207, label %208, label %202, !llvm.loop !28

208:                                              ; preds = %202, %195, %179
  %209 = phi i1 [ %180, %179 ], [ %197, %195 ], [ %205, %202 ]
  %210 = zext i1 %209 to i8
  store i8 %210, ptr @m, align 4, !tbaa !6
  store i32 0, ptr @j, align 4, !tbaa !11
  br label %211

211:                                              ; preds = %208, %159
  %212 = phi i8 [ %210, %208 ], [ %163, %159 ]
  %213 = icmp slt i32 %164, 1
  br i1 %213, label %214, label %215

214:                                              ; preds = %211
  store i8 %20, ptr @k, align 4, !tbaa !6
  store i32 1, ptr @f, align 4, !tbaa !11
  br label %215

215:                                              ; preds = %211, %214
  %216 = phi i8 [ %165, %211 ], [ %20, %214 ]
  %217 = phi i32 [ %164, %211 ], [ 1, %214 ]
  %218 = icmp eq i8 %216, 0
  %219 = select i1 %218, i8 %160, i8 %216
  %220 = sext i8 %219 to i32
  %221 = icmp slt i32 %21, %220
  br i1 %221, label %222, label %223

222:                                              ; preds = %215
  store i32 0, ptr @g, align 4, !tbaa !11
  br label %223

223:                                              ; preds = %215, %222
  %224 = add nsw i32 %161, 1
  %225 = icmp eq i32 %161, 0
  br i1 %225, label %226, label %159, !llvm.loop !22

226:                                              ; preds = %223, %156, %86
  %227 = phi i8 [ %79, %86 ], [ %149, %156 ], [ %216, %223 ]
  %228 = phi i32 [ %27, %86 ], [ %97, %156 ], [ %164, %223 ]
  %229 = phi i16 [ %12, %86 ], [ %12, %156 ], [ %15, %223 ]
  %230 = trunc i32 %228 to i8
  %231 = add i8 %230, 1
  store i16 %229, ptr @n, align 4, !tbaa !9
  store i8 %231, ptr @l, align 4, !tbaa !6
  store i32 1, ptr @e, align 4, !tbaa !11
  br label %232

232:                                              ; preds = %226, %0
  %233 = phi i8 [ %227, %226 ], [ %6, %0 ]
  %234 = sext i8 %233 to i32
  tail call void @fn2(i32 noundef %234)
  ret i32 0
}

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { noreturn nounwind }

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
!10 = !{!"short", !7, i64 0}
!11 = !{!12, !12, i64 0}
!12 = !{!"int", !7, i64 0}
!13 = distinct !{!13, !14, !15, !16}
!14 = !{!"llvm.loop.mustprogress"}
!15 = !{!"llvm.loop.isvectorized", i32 1}
!16 = !{!"llvm.loop.unroll.runtime.disable"}
!17 = distinct !{!17, !14, !15, !16}
!18 = distinct !{!18, !14, !16, !15}
!19 = distinct !{!19, !14, !15, !16}
!20 = distinct !{!20, !14, !15, !16}
!21 = distinct !{!21, !14, !16, !15}
!22 = distinct !{!22, !14}
!23 = distinct !{!23, !14, !15, !16}
!24 = distinct !{!24, !14, !15, !16}
!25 = distinct !{!25, !14, !16, !15}
!26 = distinct !{!26, !14, !15, !16}
!27 = distinct !{!27, !14, !15, !16}
!28 = distinct !{!28, !14, !16, !15}
