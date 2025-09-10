; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/strlen-6.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/strlen-6.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@i0 = dso_local global i32 0, align 4
@ca = dso_local constant [2 x [3 x i8]] [[3 x i8] c"12\00", [3 x i8] zeroinitializer], align 1
@cb = dso_local constant [2 x [3 x i8]] [[3 x i8] c"123", [3 x i8] c"4\00\00"], align 1
@va = dso_local global [2 x [3 x i8]] [[3 x i8] c"123", [3 x i8] zeroinitializer], align 1
@vb = dso_local global [2 x [3 x i8]] [[3 x i8] c"123", [3 x i8] c"45\00"], align 1
@.str = private unnamed_addr constant [7 x i8] c"123456\00", align 1
@s = dso_local local_unnamed_addr global ptr @.str, align 8
@pca = dso_local local_unnamed_addr global ptr @ca, align 8
@pcb = dso_local local_unnamed_addr global ptr @cb, align 8
@pva = dso_local local_unnamed_addr global ptr @va, align 8
@pvb = dso_local local_unnamed_addr global ptr @vb, align 8
@nfails = dso_local local_unnamed_addr global i32 0, align 4
@.str.1 = private unnamed_addr constant [2 x i8] c"1\00", align 1
@.str.2 = private unnamed_addr constant [46 x i8] c"line %i: strlen ((%s) = (\22%s\22)) == %u failed\0A\00", align 1
@.str.3 = private unnamed_addr constant [17 x i8] c"i0 ? \221\22 : ca[0]\00", align 1
@.str.4 = private unnamed_addr constant [4 x i8] c"123\00", align 1
@.str.5 = private unnamed_addr constant [19 x i8] c"i0 ? ca[0] : \22123\22\00", align 1
@.str.6 = private unnamed_addr constant [17 x i8] c"i0 ? \221\22 : cb[0]\00", align 1
@.str.7 = private unnamed_addr constant [3 x i8] c"12\00", align 1
@.str.8 = private unnamed_addr constant [18 x i8] c"i0 ? cb[0] : \2212\22\00", align 1
@.str.9 = private unnamed_addr constant [17 x i8] c"i0 ? \221\22 : va[0]\00", align 1
@.str.10 = private unnamed_addr constant [5 x i8] c"1234\00", align 1
@.str.11 = private unnamed_addr constant [20 x i8] c"i0 ? va[0] : \221234\22\00", align 1
@.str.12 = private unnamed_addr constant [17 x i8] c"i0 ? \221\22 : vb[0]\00", align 1
@.str.13 = private unnamed_addr constant [18 x i8] c"i0 ? vb[0] : \2212\22\00", align 1
@__const.test_binary_cond_expr_local.lva = private unnamed_addr constant [2 x [3 x i8]] [[3 x i8] c"123", [3 x i8] zeroinitializer], align 1
@__const.test_binary_cond_expr_local.lvb = private unnamed_addr constant [2 x [3 x i8]] [[3 x i8] c"123", [3 x i8] c"45\00"], align 1
@.str.14 = private unnamed_addr constant [18 x i8] c"i0 ? \221\22 : lca[0]\00", align 1
@.str.15 = private unnamed_addr constant [20 x i8] c"i0 ? lca[0] : \22123\22\00", align 1
@.str.16 = private unnamed_addr constant [18 x i8] c"i0 ? \221\22 : lcb[0]\00", align 1
@.str.17 = private unnamed_addr constant [19 x i8] c"i0 ? lcb[0] : \2212\22\00", align 1
@.str.18 = private unnamed_addr constant [18 x i8] c"i0 ? \221\22 : lva[0]\00", align 1
@.str.19 = private unnamed_addr constant [21 x i8] c"i0 ? lva[0] : \221234\22\00", align 1
@.str.20 = private unnamed_addr constant [18 x i8] c"i0 ? \221\22 : lvb[0]\00", align 1
@.str.21 = private unnamed_addr constant [19 x i8] c"i0 ? lvb[0] : \2212\22\00", align 1
@.str.22 = private unnamed_addr constant [38 x i8] c"i0 == 0 ? s : i0 == 1 ? vb[0] : \22123\22\00", align 1
@.str.23 = private unnamed_addr constant [38 x i8] c"i0 == 0 ? vb[0] : i0 == 1 ? s : \22123\22\00", align 1
@.str.24 = private unnamed_addr constant [38 x i8] c"i0 == 0 ? \22123\22 : i0 == 1 ? s : vb[0]\00", align 1
@.str.25 = private unnamed_addr constant [17 x i8] c"i0 ? *pca : *pcb\00", align 1
@.str.26 = private unnamed_addr constant [17 x i8] c"i0 ? *pcb : *pca\00", align 1
@.str.27 = private unnamed_addr constant [17 x i8] c"i0 ? *pva : *pvb\00", align 1
@.str.28 = private unnamed_addr constant [17 x i8] c"i0 ? *pvb : *pva\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = alloca [2 x [3 x i8]], align 1
  %2 = alloca [2 x [3 x i8]], align 1
  %3 = alloca [2 x [3 x i8]], align 1
  %4 = alloca [2 x [3 x i8]], align 1
  %5 = load volatile i32, ptr @i0, align 4, !tbaa !6
  %6 = icmp eq i32 %5, 0
  br i1 %6, label %11, label %7

7:                                                ; preds = %0
  %8 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 35, ptr noundef nonnull @.str.3, ptr noundef nonnull @.str.1, i32 noundef 2) #6
  %9 = load i32, ptr @nfails, align 4, !tbaa !6
  %10 = add i32 %9, 1
  store i32 %10, ptr @nfails, align 4, !tbaa !6
  br label %11

11:                                               ; preds = %7, %0
  %12 = load volatile i32, ptr @i0, align 4, !tbaa !6
  %13 = icmp eq i32 %12, 0
  br i1 %13, label %18, label %14

14:                                               ; preds = %11
  %15 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 36, ptr noundef nonnull @.str.5, ptr noundef nonnull @ca, i32 noundef 3) #6
  %16 = load i32, ptr @nfails, align 4, !tbaa !6
  %17 = add i32 %16, 1
  store i32 %17, ptr @nfails, align 4, !tbaa !6
  br label %18

18:                                               ; preds = %14, %11
  %19 = load volatile i32, ptr @i0, align 4, !tbaa !6
  %20 = icmp eq i32 %19, 0
  br i1 %20, label %25, label %21

21:                                               ; preds = %18
  %22 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 43, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.1, i32 noundef 4) #6
  %23 = load i32, ptr @nfails, align 4, !tbaa !6
  %24 = add i32 %23, 1
  store i32 %24, ptr @nfails, align 4, !tbaa !6
  br label %25

25:                                               ; preds = %21, %18
  %26 = load volatile i32, ptr @i0, align 4, !tbaa !6
  %27 = icmp eq i32 %26, 0
  br i1 %27, label %32, label %28

28:                                               ; preds = %25
  %29 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 44, ptr noundef nonnull @.str.8, ptr noundef nonnull @cb, i32 noundef 2) #6
  %30 = load i32, ptr @nfails, align 4, !tbaa !6
  %31 = add i32 %30, 1
  store i32 %31, ptr @nfails, align 4, !tbaa !6
  br label %32

32:                                               ; preds = %28, %25
  %33 = load volatile i32, ptr @i0, align 4, !tbaa !6
  %34 = icmp eq i32 %33, 0
  %35 = select i1 %34, ptr @va, ptr @.str.1
  %36 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %35) #6
  %37 = and i64 %36, 4294967295
  %38 = icmp eq i64 %37, 3
  br i1 %38, label %43, label %39

39:                                               ; preds = %32
  %40 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 46, ptr noundef nonnull @.str.9, ptr noundef nonnull %35, i32 noundef 3) #6
  %41 = load i32, ptr @nfails, align 4, !tbaa !6
  %42 = add i32 %41, 1
  store i32 %42, ptr @nfails, align 4, !tbaa !6
  br label %43

43:                                               ; preds = %39, %32
  %44 = load volatile i32, ptr @i0, align 4, !tbaa !6
  %45 = icmp eq i32 %44, 0
  %46 = select i1 %45, ptr @.str.10, ptr @va
  %47 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %46) #6
  %48 = and i64 %47, 4294967295
  %49 = icmp eq i64 %48, 4
  br i1 %49, label %54, label %50

50:                                               ; preds = %43
  %51 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 47, ptr noundef nonnull @.str.11, ptr noundef nonnull %46, i32 noundef 4) #6
  %52 = load i32, ptr @nfails, align 4, !tbaa !6
  %53 = add i32 %52, 1
  store i32 %53, ptr @nfails, align 4, !tbaa !6
  br label %54

54:                                               ; preds = %50, %43
  %55 = load volatile i32, ptr @i0, align 4, !tbaa !6
  %56 = icmp eq i32 %55, 0
  %57 = select i1 %56, ptr @vb, ptr @.str.1
  %58 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %57) #6
  %59 = and i64 %58, 4294967295
  %60 = icmp eq i64 %59, 5
  br i1 %60, label %65, label %61

61:                                               ; preds = %54
  %62 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 49, ptr noundef nonnull @.str.12, ptr noundef nonnull %57, i32 noundef 5) #6
  %63 = load i32, ptr @nfails, align 4, !tbaa !6
  %64 = add i32 %63, 1
  store i32 %64, ptr @nfails, align 4, !tbaa !6
  br label %65

65:                                               ; preds = %61, %54
  %66 = load volatile i32, ptr @i0, align 4, !tbaa !6
  %67 = icmp eq i32 %66, 0
  %68 = select i1 %67, ptr @.str.7, ptr @vb
  %69 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %68) #6
  %70 = and i64 %69, 4294967295
  %71 = icmp eq i64 %70, 2
  br i1 %71, label %76, label %72

72:                                               ; preds = %65
  %73 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 50, ptr noundef nonnull @.str.13, ptr noundef nonnull %68, i32 noundef 2) #6
  %74 = load i32, ptr @nfails, align 4, !tbaa !6
  %75 = add i32 %74, 1
  store i32 %75, ptr @nfails, align 4, !tbaa !6
  br label %76

76:                                               ; preds = %65, %72
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(6) %1, ptr noundef nonnull align 1 dereferenceable(6) @ca, i64 6, i1 false)
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(6) %2, ptr noundef nonnull align 1 dereferenceable(6) @cb, i64 6, i1 false)
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(6) %3, ptr noundef nonnull align 1 dereferenceable(6) @__const.test_binary_cond_expr_local.lva, i64 6, i1 false)
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(6) %4, ptr noundef nonnull align 1 dereferenceable(6) @__const.test_binary_cond_expr_local.lvb, i64 6, i1 false)
  %77 = load volatile i32, ptr @i0, align 4, !tbaa !6
  %78 = icmp eq i32 %77, 0
  %79 = select i1 %78, ptr %1, ptr @.str.1
  %80 = call i64 @strlen(ptr noundef nonnull dereferenceable(1) %79) #6
  %81 = and i64 %80, 4294967295
  %82 = icmp eq i64 %81, 2
  br i1 %82, label %87, label %83

83:                                               ; preds = %76
  %84 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 63, ptr noundef nonnull @.str.14, ptr noundef nonnull %79, i32 noundef 2) #6
  %85 = load i32, ptr @nfails, align 4, !tbaa !6
  %86 = add i32 %85, 1
  store i32 %86, ptr @nfails, align 4, !tbaa !6
  br label %87

87:                                               ; preds = %83, %76
  %88 = load volatile i32, ptr @i0, align 4, !tbaa !6
  %89 = icmp eq i32 %88, 0
  %90 = select i1 %89, ptr @.str.4, ptr %1
  %91 = call i64 @strlen(ptr noundef nonnull dereferenceable(1) %90) #6
  %92 = and i64 %91, 4294967295
  %93 = icmp eq i64 %92, 3
  br i1 %93, label %98, label %94

94:                                               ; preds = %87
  %95 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 64, ptr noundef nonnull @.str.15, ptr noundef nonnull %90, i32 noundef 3) #6
  %96 = load i32, ptr @nfails, align 4, !tbaa !6
  %97 = add i32 %96, 1
  store i32 %97, ptr @nfails, align 4, !tbaa !6
  br label %98

98:                                               ; preds = %94, %87
  %99 = load volatile i32, ptr @i0, align 4, !tbaa !6
  %100 = icmp eq i32 %99, 0
  %101 = select i1 %100, ptr %2, ptr @.str.1
  %102 = call i64 @strlen(ptr noundef nonnull dereferenceable(1) %101) #6
  %103 = and i64 %102, 4294967295
  %104 = icmp eq i64 %103, 4
  br i1 %104, label %109, label %105

105:                                              ; preds = %98
  %106 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 66, ptr noundef nonnull @.str.16, ptr noundef nonnull %101, i32 noundef 4) #6
  %107 = load i32, ptr @nfails, align 4, !tbaa !6
  %108 = add i32 %107, 1
  store i32 %108, ptr @nfails, align 4, !tbaa !6
  br label %109

109:                                              ; preds = %105, %98
  %110 = load volatile i32, ptr @i0, align 4, !tbaa !6
  %111 = icmp eq i32 %110, 0
  %112 = select i1 %111, ptr @.str.7, ptr %2
  %113 = call i64 @strlen(ptr noundef nonnull dereferenceable(1) %112) #6
  %114 = and i64 %113, 4294967295
  %115 = icmp eq i64 %114, 2
  br i1 %115, label %120, label %116

116:                                              ; preds = %109
  %117 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 67, ptr noundef nonnull @.str.17, ptr noundef nonnull %112, i32 noundef 2) #6
  %118 = load i32, ptr @nfails, align 4, !tbaa !6
  %119 = add i32 %118, 1
  store i32 %119, ptr @nfails, align 4, !tbaa !6
  br label %120

120:                                              ; preds = %116, %109
  %121 = load volatile i32, ptr @i0, align 4, !tbaa !6
  %122 = icmp eq i32 %121, 0
  %123 = select i1 %122, ptr %3, ptr @.str.1
  %124 = call i64 @strlen(ptr noundef nonnull dereferenceable(1) %123) #6
  %125 = and i64 %124, 4294967295
  %126 = icmp eq i64 %125, 3
  br i1 %126, label %131, label %127

127:                                              ; preds = %120
  %128 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 69, ptr noundef nonnull @.str.18, ptr noundef nonnull %123, i32 noundef 3) #6
  %129 = load i32, ptr @nfails, align 4, !tbaa !6
  %130 = add i32 %129, 1
  store i32 %130, ptr @nfails, align 4, !tbaa !6
  br label %131

131:                                              ; preds = %127, %120
  %132 = load volatile i32, ptr @i0, align 4, !tbaa !6
  %133 = icmp eq i32 %132, 0
  %134 = select i1 %133, ptr @.str.10, ptr %3
  %135 = call i64 @strlen(ptr noundef nonnull dereferenceable(1) %134) #6
  %136 = and i64 %135, 4294967295
  %137 = icmp eq i64 %136, 4
  br i1 %137, label %142, label %138

138:                                              ; preds = %131
  %139 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 70, ptr noundef nonnull @.str.19, ptr noundef nonnull %134, i32 noundef 4) #6
  %140 = load i32, ptr @nfails, align 4, !tbaa !6
  %141 = add i32 %140, 1
  store i32 %141, ptr @nfails, align 4, !tbaa !6
  br label %142

142:                                              ; preds = %138, %131
  %143 = load volatile i32, ptr @i0, align 4, !tbaa !6
  %144 = icmp eq i32 %143, 0
  %145 = select i1 %144, ptr %4, ptr @.str.1
  %146 = call i64 @strlen(ptr noundef nonnull dereferenceable(1) %145) #6
  %147 = and i64 %146, 4294967295
  %148 = icmp eq i64 %147, 5
  br i1 %148, label %153, label %149

149:                                              ; preds = %142
  %150 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 72, ptr noundef nonnull @.str.20, ptr noundef nonnull %145, i32 noundef 5) #6
  %151 = load i32, ptr @nfails, align 4, !tbaa !6
  %152 = add i32 %151, 1
  store i32 %152, ptr @nfails, align 4, !tbaa !6
  br label %153

153:                                              ; preds = %149, %142
  %154 = load volatile i32, ptr @i0, align 4, !tbaa !6
  %155 = icmp eq i32 %154, 0
  %156 = select i1 %155, ptr @.str.7, ptr %4
  %157 = call i64 @strlen(ptr noundef nonnull dereferenceable(1) %156) #6
  %158 = and i64 %157, 4294967295
  %159 = icmp eq i64 %158, 2
  br i1 %159, label %164, label %160

160:                                              ; preds = %153
  %161 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 73, ptr noundef nonnull @.str.21, ptr noundef nonnull %156, i32 noundef 2) #6
  %162 = load i32, ptr @nfails, align 4, !tbaa !6
  %163 = add i32 %162, 1
  store i32 %163, ptr @nfails, align 4, !tbaa !6
  br label %164

164:                                              ; preds = %153, %160
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #6
  %165 = load volatile i32, ptr @i0, align 4, !tbaa !6
  %166 = icmp eq i32 %165, 0
  br i1 %166, label %167, label %169

167:                                              ; preds = %164
  %168 = load ptr, ptr @s, align 8, !tbaa !10
  br label %173

169:                                              ; preds = %164
  %170 = load volatile i32, ptr @i0, align 4, !tbaa !6
  %171 = icmp eq i32 %170, 1
  %172 = select i1 %171, ptr @vb, ptr @.str.4
  br label %173

173:                                              ; preds = %169, %167
  %174 = phi ptr [ %168, %167 ], [ %172, %169 ]
  %175 = call i64 @strlen(ptr noundef nonnull dereferenceable(1) %174) #6
  %176 = and i64 %175, 4294967295
  %177 = icmp eq i64 %176, 6
  br i1 %177, label %182, label %178

178:                                              ; preds = %173
  %179 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 80, ptr noundef nonnull @.str.22, ptr noundef nonnull %174, i32 noundef 6) #6
  %180 = load i32, ptr @nfails, align 4, !tbaa !6
  %181 = add i32 %180, 1
  store i32 %181, ptr @nfails, align 4, !tbaa !6
  br label %182

182:                                              ; preds = %178, %173
  %183 = load volatile i32, ptr @i0, align 4, !tbaa !6
  %184 = icmp eq i32 %183, 0
  br i1 %184, label %190, label %185

185:                                              ; preds = %182
  %186 = load volatile i32, ptr @i0, align 4, !tbaa !6
  %187 = icmp eq i32 %186, 1
  %188 = load ptr, ptr @s, align 8
  %189 = select i1 %187, ptr %188, ptr @.str.4
  br label %190

190:                                              ; preds = %185, %182
  %191 = phi ptr [ %189, %185 ], [ @vb, %182 ]
  %192 = call i64 @strlen(ptr noundef nonnull dereferenceable(1) %191) #6
  %193 = and i64 %192, 4294967295
  %194 = icmp eq i64 %193, 5
  br i1 %194, label %199, label %195

195:                                              ; preds = %190
  %196 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 81, ptr noundef nonnull @.str.23, ptr noundef nonnull %191, i32 noundef 5) #6
  %197 = load i32, ptr @nfails, align 4, !tbaa !6
  %198 = add i32 %197, 1
  store i32 %198, ptr @nfails, align 4, !tbaa !6
  br label %199

199:                                              ; preds = %195, %190
  %200 = load volatile i32, ptr @i0, align 4, !tbaa !6
  %201 = icmp eq i32 %200, 0
  br i1 %201, label %207, label %202

202:                                              ; preds = %199
  %203 = load volatile i32, ptr @i0, align 4, !tbaa !6
  %204 = icmp eq i32 %203, 1
  %205 = load ptr, ptr @s, align 8
  %206 = select i1 %204, ptr %205, ptr @vb
  br label %207

207:                                              ; preds = %202, %199
  %208 = phi ptr [ %206, %202 ], [ @.str.4, %199 ]
  %209 = call i64 @strlen(ptr noundef nonnull dereferenceable(1) %208) #6
  %210 = and i64 %209, 4294967295
  %211 = icmp eq i64 %210, 3
  br i1 %211, label %216, label %212

212:                                              ; preds = %207
  %213 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 82, ptr noundef nonnull @.str.24, ptr noundef nonnull %208, i32 noundef 3) #6
  %214 = load i32, ptr @nfails, align 4, !tbaa !6
  %215 = add i32 %214, 1
  store i32 %215, ptr @nfails, align 4, !tbaa !6
  br label %216

216:                                              ; preds = %207, %212
  %217 = load volatile i32, ptr @i0, align 4, !tbaa !6
  %218 = icmp eq i32 %217, 0
  %219 = load ptr, ptr @pca, align 8
  %220 = load ptr, ptr @pcb, align 8
  %221 = select i1 %218, ptr %220, ptr %219
  %222 = call i64 @strlen(ptr noundef nonnull dereferenceable(1) %221) #6
  %223 = and i64 %222, 4294967295
  %224 = icmp eq i64 %223, 4
  br i1 %224, label %231, label %225

225:                                              ; preds = %216
  %226 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 95, ptr noundef nonnull @.str.25, ptr noundef nonnull %221, i32 noundef 4) #6
  %227 = load i32, ptr @nfails, align 4, !tbaa !6
  %228 = add i32 %227, 1
  store i32 %228, ptr @nfails, align 4, !tbaa !6
  %229 = load ptr, ptr @pcb, align 8
  %230 = load ptr, ptr @pca, align 8
  br label %231

231:                                              ; preds = %225, %216
  %232 = phi ptr [ %219, %216 ], [ %230, %225 ]
  %233 = phi ptr [ %220, %216 ], [ %229, %225 ]
  %234 = load volatile i32, ptr @i0, align 4, !tbaa !6
  %235 = icmp eq i32 %234, 0
  %236 = select i1 %235, ptr %232, ptr %233
  %237 = call i64 @strlen(ptr noundef nonnull dereferenceable(1) %236) #6
  %238 = and i64 %237, 4294967295
  %239 = icmp eq i64 %238, 2
  br i1 %239, label %244, label %240

240:                                              ; preds = %231
  %241 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 96, ptr noundef nonnull @.str.26, ptr noundef nonnull %236, i32 noundef 2) #6
  %242 = load i32, ptr @nfails, align 4, !tbaa !6
  %243 = add i32 %242, 1
  store i32 %243, ptr @nfails, align 4, !tbaa !6
  br label %244

244:                                              ; preds = %240, %231
  %245 = load volatile i32, ptr @i0, align 4, !tbaa !6
  %246 = icmp eq i32 %245, 0
  %247 = load ptr, ptr @pva, align 8
  %248 = load ptr, ptr @pvb, align 8
  %249 = select i1 %246, ptr %248, ptr %247
  %250 = call i64 @strlen(ptr noundef nonnull dereferenceable(1) %249) #6
  %251 = and i64 %250, 4294967295
  %252 = icmp eq i64 %251, 5
  br i1 %252, label %259, label %253

253:                                              ; preds = %244
  %254 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 98, ptr noundef nonnull @.str.27, ptr noundef nonnull %249, i32 noundef 5) #6
  %255 = load i32, ptr @nfails, align 4, !tbaa !6
  %256 = add i32 %255, 1
  store i32 %256, ptr @nfails, align 4, !tbaa !6
  %257 = load ptr, ptr @pvb, align 8
  %258 = load ptr, ptr @pva, align 8
  br label %259

259:                                              ; preds = %253, %244
  %260 = phi ptr [ %247, %244 ], [ %258, %253 ]
  %261 = phi ptr [ %248, %244 ], [ %257, %253 ]
  %262 = load volatile i32, ptr @i0, align 4, !tbaa !6
  %263 = icmp eq i32 %262, 0
  %264 = select i1 %263, ptr %260, ptr %261
  %265 = call i64 @strlen(ptr noundef nonnull dereferenceable(1) %264) #6
  %266 = and i64 %265, 4294967295
  %267 = icmp eq i64 %266, 3
  br i1 %267, label %272, label %268

268:                                              ; preds = %259
  %269 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 99, ptr noundef nonnull @.str.28, ptr noundef nonnull %264, i32 noundef 3) #6
  %270 = load i32, ptr @nfails, align 4, !tbaa !6
  %271 = add i32 %270, 1
  store i32 %271, ptr @nfails, align 4, !tbaa !6
  br label %274

272:                                              ; preds = %259
  %273 = load i32, ptr @nfails, align 4, !tbaa !6
  br label %274

274:                                              ; preds = %272, %268
  %275 = phi i32 [ %273, %272 ], [ %271, %268 ]
  %276 = icmp eq i32 %275, 0
  br i1 %276, label %278, label %277

277:                                              ; preds = %274
  call void @abort() #7
  unreachable

278:                                              ; preds = %274
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #2

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare i64 @strlen(ptr noundef captures(none)) local_unnamed_addr #3

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #2

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #5

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #6 = { nounwind }
attributes #7 = { noreturn nounwind }

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
!11 = !{!"p1 omnipotent char", !12, i64 0}
!12 = !{!"any pointer", !8, i64 0}
