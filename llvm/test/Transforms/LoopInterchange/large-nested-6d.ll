; RUN: opt < %s -passes=loop-interchange -cache-line-size=64 -pass-remarks='loop-interchange' -pass-remarks-missed='loop-interchange' -pass-remarks-output=%t -disable-output -S
; RUN: FileCheck --input-file=%t %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"

; The IR test case below is a full and representative motivating example
; for loop-interchange containing a more complex loop nest structure that
; corresponds to this pseudo-code:
;
;      for L=1 to NX
;       for M=1 to NY
;        for i=1 to NX
;         for j=1 to NY
;          for IL=1 to NX
;           load GlobC(i,IL,L)
;           load GlobG(i,IL,L)
;           load GlobE(i,IL,L)
;           load GlobI(i,IL,L)
;           for JL=1 to NY
;            load GlobD(j,JL,M)
;            load GlobH(j,JL,M)
;            load GlobF(j,JL,M)
;            load GlobJ(j,JL,M)
;            store GlobL(NY*i+j,NY*IL+JL)
;           End
;          End
;         End
;        End
;        // Stmt 2
;        // Stmt 3
;        // Stmt 4
;      End
;     End
;
; It is important to note here that this comes from Fortran code, which uses a
; column-major data layout, so loops 'j' and 'JL' should be interchanged. I.e.
; in the IR below, basic block JL.body is part of the loop that we would like
; like to see interchanged as there are 4 loads and 1 store that are
; unit-strided over 'j', so making 'j' loop the innermost is preferable here.
;
; TODO:
;
; There are a few issues that prevent loop-interchange to perform its
; transformation on this test case:
;
; 1. LoopNest checks: the first check that is perform is whether loop 'L.header'
;    and 'M.header' are perfectly nested, which they are not. It needs to be
;    investigate why the whole loop nest rooted under L is rejected as a
;    candidate.
;
; 2. DependenceAnalysis: it finds this dependency:
;
;    Found output dependency between Src and Dst
;      Src:  store double %46, ptr %48, align 8
;      Dst:  store double %46, ptr %48, align 8
;
;
; CHECK:       --- !Missed
; CHECK-NEXT:  Pass:            loop-interchange
; CHECK-NEXT:  Name:            UnsupportedLoopNestDepth
; CHECK-NEXT:  Function:        test
; CHECK-NEXT:  Args:
; CHECK-NEXT:    - String:          'Unsupported depth of loop nest, the supported range is ['
; CHECK-NEXT:    - String:          '2'
; CHECK-NEXT:    - String:          ', '
; CHECK-NEXT:    - String:          '10'
; CHECK-NEXT:    - String:          "].\n"
; CHECK-NEXT:  ...
; CHECK-NEXT:  --- !Analysis
; CHECK-NEXT:  Pass:            loop-interchange
; CHECK-NEXT:  Name:            Dependence
; CHECK-NEXT:  Function:        test
; CHECK-NEXT:  Args:
; CHECK-NEXT:    - String:          Computed dependence info, invoking the transform.
; CHECK-NEXT:  ...
; CHECK-NEXT:  --- !Missed
; CHECK-NEXT:  Pass:            loop-interchange
; CHECK-NEXT:  Name:            Dependence
; CHECK-NEXT:  Function:        test
; CHECK-NEXT:  Args:
; CHECK-NEXT:    - String:          Cannot interchange loops due to dependences.
; CHECK-NEXT:  ...
; CHECK-NEXT:  --- !Missed
; CHECK-NEXT:  Pass:            loop-interchange
; CHECK-NEXT:  Name:            UnsupportedLoopNestDepth
; CHECK-NEXT:  Function:        test
; CHECK-NEXT:  Args:
; CHECK-NEXT:    - String:          'Unsupported depth of loop nest, the supported range is ['
; CHECK-NEXT:    - String:          '2'
; CHECK-NEXT:    - String:          ', '
; CHECK-NEXT:    - String:          '10'
; CHECK-NEXT:    - String:          "].\n"
; CHECK-NEXT:  ...
; CHECK-NEXT:  --- !Analysis
; CHECK-NEXT:  Pass:            loop-interchange
; CHECK-NEXT:  Name:            Dependence
; CHECK-NEXT:  Function:        test
; CHECK-NEXT:  Args:
; CHECK-NEXT:    - String:          Computed dependence info, invoking the transform.
; CHECK-NEXT:  ...
; CHECK-NEXT:  --- !Missed
; CHECK-NEXT:  Pass:            loop-interchange
; CHECK-NEXT:  Name:            NotTightlyNested
; CHECK-NEXT:  Function:        test
; CHECK-NEXT:  Args:
; CHECK-NEXT:    - String:          Cannot interchange loops because they are not tightly nested.
; CHECK-NEXT:  ...
; CHECK-NEXT:  --- !Missed
; CHECK-NEXT:  Pass:            loop-interchange
; CHECK-NEXT:  Name:            Dependence
; CHECK-NEXT:  Function:        test
; CHECK-NEXT:  Args:
; CHECK-NEXT:    - String:          Cannot interchange loops due to dependences.
; CHECK-NEXT:  ...
; CHECK-NEXT:  --- !Analysis
; CHECK-NEXT:  Pass:            loop-interchange
; CHECK-NEXT:  Name:            Dependence
; CHECK-NEXT:  Function:        test
; CHECK-NEXT:  Args:
; CHECK-NEXT:    - String:          Computed dependence info, invoking the transform.
; CHECK-NEXT:  ...
; CHECK-NEXT:  --- !Missed
; CHECK-NEXT:  Pass:            loop-interchange
; CHECK-NEXT:  Name:            Dependence
; CHECK-NEXT:  Function:        test
; CHECK-NEXT:  Args:
; CHECK-NEXT:    - String:          All loops have dependencies in all directions.
; CHECK-NEXT:  ...

@GlobC = local_unnamed_addr global [54 x [54 x [54 x double]]] zeroinitializer
@GlobD = local_unnamed_addr global [54 x [54 x [54 x double]]] zeroinitializer
@GlobE = local_unnamed_addr global [54 x [54 x [54 x double]]] zeroinitializer
@GlobF = local_unnamed_addr global [54 x [54 x [54 x double]]] zeroinitializer
@GlobG = local_unnamed_addr global [54 x [54 x [54 x double]]] zeroinitializer
@GlobH = local_unnamed_addr global [54 x [54 x [54 x double]]] zeroinitializer
@GlobI = local_unnamed_addr global [54 x [54 x [54 x double]]] zeroinitializer
@GlobJ = local_unnamed_addr global [54 x [54 x [54 x double]]] zeroinitializer
@GlobK = local_unnamed_addr global [1000 x [1000 x double]] zeroinitializer
@GlobL = local_unnamed_addr global [1000 x [1000 x double]] zeroinitializer
@GlobM = local_unnamed_addr global [2500 x double] zeroinitializer

define void @test(ptr noalias readonly captures(none) %0, ptr noalias readonly captures(none) %1, ptr noalias captures(none) %2, ptr noalias captures(none) %3, ptr noalias readonly captures(none) %4, ptr noalias readonly captures(none) %5, ptr noalias readonly captures(none) %6, ptr noalias readonly captures(none) %7, ptr noalias readonly captures(none) %8, ptr noalias readonly captures(none) %9) {
  %11 = alloca [2500 x double], align 8
  %12 = load i32, ptr %4, align 4
  %13 = tail call i32 @llvm.smax.i32(i32 %12, i32 0)
  %14 = zext nneg i32 %13 to i64
  %15 = load i32, ptr %9, align 4
  %.not = icmp eq i32 %15, 1
  br i1 %.not, label %171, label %16

16:
  %17 = load i32, ptr %7, align 4
  %18 = sext i32 %17 to i64
  %19 = icmp sgt i32 %17, 0
  br i1 %19, label %.lr.ph286, label %._crit_edge287

.lr.ph286:
  %20 = load i32, ptr %8, align 4
  %21 = sext i32 %20 to i64
  %22 = icmp sgt i32 %20, 0
  br i1 %22, label %preheader.L, label %._crit_edge287

preheader.L:
  %23 = load i32, ptr %5, align 4
  %24 = tail call i32 @llvm.smax.i32(i32 %23, i32 0)
  %25 = zext nneg i32 %24 to i64
  %26 = load i32, ptr %6, align 4
  %27 = sext i32 %26 to i64
  %28 = getelementptr double, ptr %1, i64 %27
  %.not241270.us = icmp slt i32 %23, 1
  %29 = shl nuw nsw i64 %25, 3
  %30 = add nuw nsw i64 %25, 2
  %31 = icmp sgt i32 %23, 0
  %.neg = sext i1 %31 to i64
  %32 = add nsw i64 %30, %.neg
  br label %L.header

L.header:
  %L = phi i64 [ %L.next, %L.latch ], [ 1, %preheader.L ]
  %33 = mul nuw nsw i64 %L, 2916
  %34 = add nsw i64 %33, -2971
  %35 = add nsw i64 %L, -1
  %36 = mul nsw i64 %35, %21
  br label %M.header

exit.i:
  br i1 %.not241270.us, label %._crit_edge275.us.thread, label %.preheader258.us.preheader

.lr.ph274.us:
  %37 = phi i64 [ %48, %.lr.ph274.us ], [ %25, %.preheader260.us ]
  %38 = phi double [ %46, %.lr.ph274.us ], [ 0.000000e+00, %.preheader260.us ]
  %39 = phi i64 [ %47, %.lr.ph274.us ], [ 1, %.preheader260.us ]
  %40 = add nsw i64 %39, -1
  %41 = getelementptr double, ptr %28, i64 %40
  %42 = load double, ptr %41, align 8
  %43 = getelementptr double, ptr @GlobM, i64 %40
  %44 = load double, ptr %43, align 8
  %45 = fmul fast double %44, %42
  %46 = fadd fast double %45, %38
  %47 = add nuw nsw i64 %39, 1
  %48 = add nsw i64 %37, -1
  %.not242.us = icmp eq i64 %48, 0
  br i1 %.not242.us, label %.lr.ph278.us.preheader, label %.lr.ph274.us

.lr.ph278.us.preheader:
  %.lcssa = phi double [ %46, %.lr.ph274.us ]
  %49 = add nsw i64 %M, %36
  %50 = getelementptr double, ptr %11, i64 %49
  %51 = getelementptr i8, ptr %50, i64 -8
  store double %.lcssa, ptr %51, align 8
  %52 = getelementptr double, ptr @GlobK, i64 %49
  %53 = getelementptr i8, ptr %52, i64 -8
  br label %.lr.ph278.us

latch.M.loopexit:
  br label %latch.M

latch.M:
  %M.next = add nuw nsw i64 %M, 1
  %exitcond335.not = icmp eq i64 %M, %21
  br i1 %exitcond335.not, label %L.latch, label %M.header

.lr.ph278.us:
  %54 = phi i64 [ %133, %._crit_edge279.us ], [ 1, %.lr.ph278.us.preheader ]
  %55 = add nsw i64 %54, -1
  %.idx244.us = mul nuw nsw i64 %55, 8000
  %56 = getelementptr i8, ptr @GlobL, i64 %.idx244.us
  br label %57

57:
  %58 = phi i64 [ %25, %.lr.ph278.us ], [ %69, %57 ]
  %59 = phi double [ 0.000000e+00, %.lr.ph278.us ], [ %67, %57 ]
  %60 = phi i64 [ 1, %.lr.ph278.us ], [ %68, %57 ]
  %61 = add nsw i64 %60, -1
  %62 = getelementptr double, ptr %56, i64 %61
  %63 = load double, ptr %62, align 8
  %64 = getelementptr double, ptr %28, i64 %61
  %65 = load double, ptr %64, align 8
  %66 = fmul fast double %65, %63
  %67 = fadd fast double %66, %59
  %68 = add nuw nsw i64 %60, 1
  %69 = add nsw i64 %58, -1
  %.not243.us = icmp eq i64 %69, 0
  br i1 %.not243.us, label %._crit_edge279.us, label %57

70:
  %71 = phi i64 [ %25, %.preheader258.us ], [ %81, %70 ]
  %72 = phi i64 [ 1, %.preheader258.us ], [ %80, %70 ]
  %73 = add nsw i64 %72, -1
  %74 = getelementptr double, ptr @GlobM, i64 %73
  %75 = load double, ptr %74, align 8
  %76 = getelementptr double, ptr %84, i64 %73
  %77 = load double, ptr %76, align 8
  %78 = fmul fast double %86, %77
  %79 = fadd fast double %78, %75
  store double %79, ptr %74, align 8
  %80 = add nuw nsw i64 %72, 1
  %81 = add nsw i64 %71, -1
  %.not245.us = icmp eq i64 %81, 0
  br i1 %.not245.us, label %._crit_edge.us, label %70

.preheader258.us:
  %82 = phi i64 [ %128, %._crit_edge.us ], [ 1, %.preheader258.us.preheader ]
  %83 = add nsw i64 %82, -1
  %.idx246.us = mul nuw nsw i64 %83, 8000
  %84 = getelementptr i8, ptr @GlobL, i64 %.idx246.us
  %85 = getelementptr double, ptr %28, i64 %83
  %86 = load double, ptr %85, align 8
  br label %70

.preheader260.us:
  br label %.lr.ph274.us

._crit_edge275.us.thread:
  %87 = getelementptr double, ptr %11, i64 %M
  %88 = getelementptr double, ptr %87, i64 %36
  %89 = getelementptr i8, ptr %88, i64 -8
  store double 0.000000e+00, ptr %89, align 8
  br label %latch.M

.preheader258.us.preheader:
  call void @llvm.memset.p0.i64(ptr nonnull align 16 @GlobM, i8 0, i64 %29, i1 false)
  br label %.preheader258.us

M.header:
  %M = phi i64 [ 1, %L.header ], [ %M.next, %latch.M ]
  %90 = mul nuw nsw i64 %M, 2916
  %91 = add nsw i64 %90, -2971
  br label %i.header

i.header:
  %i = phi i64 [ %i.next, %i.latch ], [ 1, %M.header ]
  %92 = add nsw i64 %34, %i
  %93 = add nsw i64 %i, -1
  %94 = mul nsw i64 %93, %21
  %invariant.gep = getelementptr double, ptr @GlobL, i64 %94
  br label %j.header

j.header:
  %j = phi i64 [ %j.next, %j.latch ], [ 1, %i.header ]
  %95 = add nsw i64 %91, %j
  %gep358 = getelementptr double, ptr %invariant.gep, i64 %j
  br label %IL.header

IL.header:
  %IL = phi i64 [ %IL.next, %IL.latch ], [ 1, %j.header ]
  %96 = mul nuw nsw i64 %IL, 54
  %97 = add nsw i64 %92, %96
  %98 = getelementptr double, ptr @GlobC, i64 %97
  %99 = load double, ptr %98, align 8
  %100 = getelementptr double, ptr @GlobG, i64 %97
  %101 = load double, ptr %100, align 8
  %102 = getelementptr double, ptr @GlobE, i64 %97
  %103 = load double, ptr %102, align 8
  %104 = getelementptr double, ptr @GlobI, i64 %97
  %105 = load double, ptr %104, align 8
  %106 = add nsw i64 %IL, -1
  %107 = mul nsw i64 %106, %21
  br label %JL.body

JL.body:
  %JL = phi i64 [ %JL.next, %JL.body ], [ 1, %IL.header ]
  %109 = mul nuw nsw i64 %JL, 54
  %110 = add nsw i64 %95, %109
  %111 = getelementptr double, ptr @GlobD, i64 %110
  %112 = load double, ptr %111, align 8
  %113 = fmul fast double %112, %99
  %114 = getelementptr double, ptr @GlobH, i64 %110
  %115 = load double, ptr %114, align 8
  %116 = fmul fast double %115, %101
  %117 = fadd fast double %116, %113
  %118 = getelementptr double, ptr @GlobF, i64 %110
  %119 = load double, ptr %118, align 8
  %120 = fmul fast double %119, %103
  %121 = fadd fast double %117, %120
  %122 = getelementptr double, ptr @GlobJ, i64 %110
  %123 = load double, ptr %122, align 8
  %124 = fmul fast double %123, %105
  %125 = fadd fast double %121, %124
  %126 = add nsw i64 %JL, %107
  %.idx247.us.us.us.us.us.us = mul nsw i64 %126, 8000
  %gep.us.us.us.us.us.us = getelementptr i8, ptr %gep358, i64 %.idx247.us.us.us.us.us.us
  %127 = getelementptr i8, ptr %gep.us.us.us.us.us.us, i64 -8008
  store double %125, ptr %127, align 8
  %JL.next = add nuw nsw i64 %JL, 1
  %exitcond.not = icmp eq i64 %JL, %21
  br i1 %exitcond.not, label %IL.latch, label %JL.body

IL.latch:
  %IL.next = add nuw nsw i64 %IL, 1
  %exitcond320.not = icmp eq i64 %IL, %18
  br i1 %exitcond320.not, label %j.latch, label %IL.header

j.latch:
  %j.next = add nuw nsw i64 %j, 1
  %exitcond324.not = icmp eq i64 %j, %21
  br i1 %exitcond324.not, label %i.latch, label %j.header

i.latch:
  %i.next = add nuw nsw i64 %i, 1
  %exitcond328.not = icmp eq i64 %i, %18
  br i1 %exitcond328.not, label %exit.i, label %i.header

._crit_edge.us:
  %128 = add nuw nsw i64 %82, 1
  %exitcond329.not = icmp eq i64 %128, %32
  br i1 %exitcond329.not, label %.preheader260.us, label %.preheader258.us

._crit_edge279.us:
  %.lcssa360 = phi double [ %67, %57 ]
  %129 = getelementptr double, ptr @GlobM, i64 %55
  %130 = load double, ptr %129, align 8
  %131 = fadd fast double %130, %.lcssa360
  %132 = getelementptr i8, ptr %53, i64 %.idx244.us
  store double %131, ptr %132, align 8
  %133 = add nuw nsw i64 %54, 1
  %exitcond331.not = icmp eq i64 %133, %32
  br i1 %exitcond331.not, label %latch.M.loopexit, label %.lr.ph278.us

L.latch:
  %L.next = add nuw nsw i64 %L, 1
  %exitcond339.not = icmp eq i64 %L, %18
  br i1 %exitcond339.not, label %exit.L, label %L.header

exit.L:
  br label %._crit_edge287

._crit_edge287:
  %134 = load i32, ptr %6, align 4
  %135 = load i32, ptr %5, align 4
  %136 = tail call i32 @llvm.smax.i32(i32 %135, i32 0)
  %137 = zext nneg i32 %136 to i64
  %138 = sext i32 %134 to i64
  %139 = getelementptr double, ptr %2, i64 %138
  %140 = shl nuw nsw i64 %137, 3
  %.not236 = icmp slt i32 %135, 1
  %141 = select i1 %.not236, i64 1, i64 %140
  %142 = tail call ptr @malloc(i64 %141)
  br i1 %.not236, label %._crit_edge294, label %.preheader254.preheader

.preheader254.preheader:
  call void @llvm.memset.p0.i64(ptr align 8 %142, i8 0, i64 %140, i1 false)
  br label %.preheader254

.preheader254:
  %143 = phi i64 [ %160, %._crit_edge ], [ 1, %.preheader254.preheader ]
  %144 = add nsw i64 %143, -1
  %.idx240 = mul nuw nsw i64 %144, 8000
  %145 = getelementptr i8, ptr %0, i64 %.idx240
  %146 = getelementptr double, ptr %11, i64 %144
  %147 = load double, ptr %146, align 8
  br label %148

.preheader253:
  br label %.lr.ph293

148:
  %149 = phi i64 [ %137, %.preheader254 ], [ %159, %148 ]
  %150 = phi i64 [ 1, %.preheader254 ], [ %158, %148 ]
  %151 = add nsw i64 %150, -1
  %152 = getelementptr double, ptr %142, i64 %151
  %153 = load double, ptr %152, align 8
  %154 = getelementptr double, ptr %145, i64 %151
  %155 = load double, ptr %154, align 8
  %156 = fmul fast double %147, %155
  %157 = fadd fast double %156, %153
  store double %157, ptr %152, align 8
  %158 = add nuw nsw i64 %150, 1
  %159 = add nsw i64 %149, -1
  %.not239 = icmp eq i64 %159, 0
  br i1 %.not239, label %._crit_edge, label %148

._crit_edge:
  %160 = add nuw nsw i64 %143, 1
  %exitcond341.not = icmp eq i64 %143, %137
  br i1 %exitcond341.not, label %.preheader253, label %.preheader254

.lr.ph293:
  %161 = phi i64 [ %170, %.lr.ph293 ], [ %137, %.preheader253 ]
  %162 = phi i64 [ %169, %.lr.ph293 ], [ 1, %.preheader253 ]
  %163 = add nsw i64 %162, -1
  %164 = getelementptr double, ptr %139, i64 %163
  %165 = getelementptr double, ptr %142, i64 %163
  %166 = load double, ptr %165, align 8
  %167 = load double, ptr %164, align 8
  %168 = fsub fast double %167, %166
  store double %168, ptr %164, align 8
  %169 = add nuw nsw i64 %162, 1
  %170 = add nsw i64 %161, -1
  %.not238 = icmp eq i64 %170, 0
  br i1 %.not238, label %._crit_edge294.loopexit359, label %.lr.ph293

171:
  %172 = load i32, ptr %6, align 4
  %173 = load i32, ptr %5, align 4
  %174 = tail call i32 @llvm.smax.i32(i32 %173, i32 0)
  %175 = zext nneg i32 %174 to i64
  %176 = shl nuw nsw i64 %175, 3
  %177 = mul i64 %176, %175
  %178 = tail call i64 @llvm.smax.i64(i64 %177, i64 1)
  %179 = tail call ptr @malloc(i64 %178)
  %.not311 = icmp slt i32 %173, 1
  br i1 %.not311, label %._crit_edge294, label %.preheader250.us.preheader

.preheader250.us.preheader:
  %180 = mul nuw nsw i64 %175, %175
  %181 = shl i64 %180, 3
  call void @llvm.memset.p0.i64(ptr align 8 %179, i8 0, i64 %181, i1 false)
  br label %.preheader250.us

.preheader250.us:
  %182 = phi i64 [ %203, %._crit_edge301.split.us ], [ 1, %.preheader250.us.preheader ]
  %183 = add nsw i64 %182, -1
  %.idx.us = mul nuw nsw i64 %183, 8000
  %184 = getelementptr i8, ptr %0, i64 %.idx.us
  %invariant.gep.us = getelementptr double, ptr @GlobK, i64 %183
  br label %.preheader249.us

185:
  %186 = phi i64 [ %175, %.preheader249.us ], [ %196, %185 ]
  %187 = phi i64 [ 1, %.preheader249.us ], [ %195, %185 ]
  %188 = add nsw i64 %187, -1
  %189 = getelementptr double, ptr %200, i64 %188
  %190 = load double, ptr %189, align 8
  %191 = getelementptr double, ptr %184, i64 %188
  %192 = load double, ptr %191, align 8
  %193 = fmul fast double %201, %192
  %194 = fadd fast double %193, %190
  store double %194, ptr %189, align 8
  %195 = add nuw nsw i64 %187, 1
  %196 = add nsw i64 %186, -1
  %.not233.us = icmp eq i64 %196, 0
  br i1 %.not233.us, label %._crit_edge300.us, label %185

.preheader249.us:
  %197 = phi i64 [ 1, %.preheader250.us ], [ %202, %._crit_edge300.us ]
  %198 = add nsw i64 %197, -1
  %199 = mul nuw nsw i64 %198, %175
  %200 = getelementptr double, ptr %179, i64 %199
  %.idx234.us = mul nuw nsw i64 %198, 8000
  %gep.us = getelementptr i8, ptr %invariant.gep.us, i64 %.idx234.us
  %201 = load double, ptr %gep.us, align 8
  br label %185

._crit_edge300.us:
  %202 = add nuw nsw i64 %197, 1
  %exitcond344.not = icmp eq i64 %197, %175
  br i1 %exitcond344.not, label %._crit_edge301.split.us, label %.preheader249.us

._crit_edge301.split.us:
  %203 = add nuw nsw i64 %182, 1
  %exitcond345.not = icmp eq i64 %182, %175
  br i1 %exitcond345.not, label %.preheader248, label %.preheader250.us

.preheader248:
  br label %.preheader.lr.ph

.preheader.lr.ph:
  %204 = sext i32 %172 to i64
  %invariant.gep306 = getelementptr double, ptr %3, i64 %204
  br label %.preheader

.preheader:
  %205 = phi i64 [ 1, %.preheader.lr.ph ], [ %221, %._crit_edge304 ]
  %206 = add nsw i64 %205, -1
  %207 = add nsw i64 %206, %204
  %208 = mul nsw i64 %207, %14
  %gep307 = getelementptr double, ptr %invariant.gep306, i64 %208
  %209 = mul nuw nsw i64 %206, %175
  %210 = getelementptr double, ptr %179, i64 %209
  br label %211

211:
  %212 = phi i64 [ %175, %.preheader ], [ %220, %211 ]
  %213 = phi i64 [ 1, %.preheader ], [ %219, %211 ]
  %214 = add nsw i64 %213, -1
  %gep = getelementptr double, ptr %gep307, i64 %214
  %215 = getelementptr double, ptr %210, i64 %214
  %216 = load double, ptr %215, align 8
  %217 = load double, ptr %gep, align 8
  %218 = fsub fast double %217, %216
  store double %218, ptr %gep, align 8
  %219 = add nuw nsw i64 %213, 1
  %220 = add nsw i64 %212, -1
  %.not232 = icmp eq i64 %220, 0
  br i1 %.not232, label %._crit_edge304, label %211

._crit_edge304:
  %221 = add nuw nsw i64 %205, 1
  %exitcond347.not = icmp eq i64 %205, %175
  br i1 %exitcond347.not, label %._crit_edge294.loopexit, label %.preheader

._crit_edge294.loopexit:
  br label %._crit_edge294

._crit_edge294.loopexit359:
  br label %._crit_edge294

._crit_edge294:
  %.sink = phi ptr [ %142, %._crit_edge287 ], [ %179, %171 ], [ %179, %._crit_edge294.loopexit ], [ %142, %._crit_edge294.loopexit359 ]
  tail call void @free(ptr %.sink)
  ret void
}

declare i64 @llvm.smax.i64(i64, i64)
declare i32 @llvm.smax.i32(i32, i32)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare void @free(ptr allocptr noundef captures(none)) local_unnamed_addr
declare noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr
