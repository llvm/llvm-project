; ModuleID = 'accumulators.c'
source_filename = "accumulators.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: nofree noinline norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local double @accumulators(ptr noalias noundef readonly captures(none) %a0, ptr noalias noundef readonly captures(none) %a1, ptr noalias noundef writeonly captures(none) %out) local_unnamed_addr #0 {
entry:
  %s = alloca [256 x double], align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %s) #4
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(2048) %s, i8 0, i64 2048, i1 false), !tbaa !6
  br label %vector.ph159

vector.ph159:                                     ; preds = %for.cond.cleanup18.7, %entry
  %indvars.iv72 = phi i64 [ 0, %entry ], [ %indvars.iv.next73, %for.cond.cleanup18.7 ]
  %arrayidx11 = getelementptr inbounds nuw double, ptr %a0, i64 %indvars.iv72
  %0 = load double, ptr %arrayidx11, align 8, !tbaa !6
  %arrayidx14 = getelementptr inbounds nuw double, ptr %a1, i64 %indvars.iv72
  %1 = load double, ptr %arrayidx14, align 8, !tbaa !6
  %add15 = fadd double %0, %1
  %broadcast.splatinsert160 = insertelement <2 x double> poison, double %add15, i64 0
  %broadcast.splat161 = shufflevector <2 x double> %broadcast.splatinsert160, <2 x double> poison, <2 x i32> zeroinitializer
  br label %vector.body162

vector.body162:                                   ; preds = %vector.body162, %vector.ph159
  %index163 = phi i64 [ 0, %vector.ph159 ], [ %index.next168, %vector.body162 ]
  %vec.ind164 = phi <2 x i64> [ <i64 0, i64 1>, %vector.ph159 ], [ %vec.ind.next169, %vector.body162 ]
  %2 = trunc <2 x i64> %vec.ind164 to <2 x i32>
  %3 = add <2 x i32> %2, splat (i32 1)
  %4 = trunc <2 x i64> %vec.ind164 to <2 x i32>
  %5 = add <2 x i32> %4, splat (i32 3)
  %6 = uitofp nneg <2 x i32> %3 to <2 x double>
  %7 = uitofp nneg <2 x i32> %5 to <2 x double>
  %8 = getelementptr inbounds nuw double, ptr %s, i64 %index163
  %9 = getelementptr inbounds nuw i8, ptr %8, i64 16
  %wide.load166 = load <2 x double>, ptr %8, align 8, !tbaa !6
  %wide.load167 = load <2 x double>, ptr %9, align 8, !tbaa !6
  %10 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %broadcast.splat161, <2 x double> %6, <2 x double> %wide.load166)
  %11 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %broadcast.splat161, <2 x double> %7, <2 x double> %wide.load167)
  store <2 x double> %10, ptr %8, align 8, !tbaa !6
  store <2 x double> %11, ptr %9, align 8, !tbaa !6
  %index.next168 = add nuw i64 %index163, 4
  %vec.ind.next169 = add <2 x i64> %vec.ind164, splat (i64 4)
  %12 = icmp eq i64 %index.next168, 256
  br i1 %12, label %vector.ph146, label %vector.body162, !llvm.loop !10

vector.ph146:                                     ; preds = %vector.body162
  %13 = or disjoint i64 %indvars.iv72, 1
  %arrayidx11.1 = getelementptr inbounds nuw double, ptr %a0, i64 %13
  %14 = load double, ptr %arrayidx11.1, align 8, !tbaa !6
  %arrayidx14.1 = getelementptr inbounds nuw double, ptr %a1, i64 %13
  %15 = load double, ptr %arrayidx14.1, align 8, !tbaa !6
  %add15.1 = fadd double %14, %15
  %broadcast.splatinsert147 = insertelement <2 x double> poison, double %add15.1, i64 0
  %broadcast.splat148 = shufflevector <2 x double> %broadcast.splatinsert147, <2 x double> poison, <2 x i32> zeroinitializer
  br label %vector.body149

vector.body149:                                   ; preds = %vector.body149, %vector.ph146
  %index150 = phi i64 [ 0, %vector.ph146 ], [ %index.next155, %vector.body149 ]
  %vec.ind151 = phi <2 x i64> [ <i64 0, i64 1>, %vector.ph146 ], [ %vec.ind.next156, %vector.body149 ]
  %16 = trunc <2 x i64> %vec.ind151 to <2 x i32>
  %17 = add <2 x i32> %16, splat (i32 1)
  %18 = trunc <2 x i64> %vec.ind151 to <2 x i32>
  %19 = add <2 x i32> %18, splat (i32 3)
  %20 = uitofp nneg <2 x i32> %17 to <2 x double>
  %21 = uitofp nneg <2 x i32> %19 to <2 x double>
  %22 = getelementptr inbounds nuw double, ptr %s, i64 %index150
  %23 = getelementptr inbounds nuw i8, ptr %22, i64 16
  %wide.load153 = load <2 x double>, ptr %22, align 8, !tbaa !6
  %wide.load154 = load <2 x double>, ptr %23, align 8, !tbaa !6
  %24 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %broadcast.splat148, <2 x double> %20, <2 x double> %wide.load153)
  %25 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %broadcast.splat148, <2 x double> %21, <2 x double> %wide.load154)
  store <2 x double> %24, ptr %22, align 8, !tbaa !6
  store <2 x double> %25, ptr %23, align 8, !tbaa !6
  %index.next155 = add nuw i64 %index150, 4
  %vec.ind.next156 = add <2 x i64> %vec.ind151, splat (i64 4)
  %26 = icmp eq i64 %index.next155, 256
  br i1 %26, label %vector.ph133, label %vector.body149, !llvm.loop !14

vector.ph133:                                     ; preds = %vector.body149
  %27 = or disjoint i64 %indvars.iv72, 2
  %arrayidx11.2 = getelementptr inbounds nuw double, ptr %a0, i64 %27
  %28 = load double, ptr %arrayidx11.2, align 8, !tbaa !6
  %arrayidx14.2 = getelementptr inbounds nuw double, ptr %a1, i64 %27
  %29 = load double, ptr %arrayidx14.2, align 8, !tbaa !6
  %add15.2 = fadd double %28, %29
  %broadcast.splatinsert134 = insertelement <2 x double> poison, double %add15.2, i64 0
  %broadcast.splat135 = shufflevector <2 x double> %broadcast.splatinsert134, <2 x double> poison, <2 x i32> zeroinitializer
  br label %vector.body136

vector.body136:                                   ; preds = %vector.body136, %vector.ph133
  %index137 = phi i64 [ 0, %vector.ph133 ], [ %index.next142, %vector.body136 ]
  %vec.ind138 = phi <2 x i64> [ <i64 0, i64 1>, %vector.ph133 ], [ %vec.ind.next143, %vector.body136 ]
  %30 = trunc <2 x i64> %vec.ind138 to <2 x i32>
  %31 = add <2 x i32> %30, splat (i32 1)
  %32 = trunc <2 x i64> %vec.ind138 to <2 x i32>
  %33 = add <2 x i32> %32, splat (i32 3)
  %34 = uitofp nneg <2 x i32> %31 to <2 x double>
  %35 = uitofp nneg <2 x i32> %33 to <2 x double>
  %36 = getelementptr inbounds nuw double, ptr %s, i64 %index137
  %37 = getelementptr inbounds nuw i8, ptr %36, i64 16
  %wide.load140 = load <2 x double>, ptr %36, align 8, !tbaa !6
  %wide.load141 = load <2 x double>, ptr %37, align 8, !tbaa !6
  %38 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %broadcast.splat135, <2 x double> %34, <2 x double> %wide.load140)
  %39 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %broadcast.splat135, <2 x double> %35, <2 x double> %wide.load141)
  store <2 x double> %38, ptr %36, align 8, !tbaa !6
  store <2 x double> %39, ptr %37, align 8, !tbaa !6
  %index.next142 = add nuw i64 %index137, 4
  %vec.ind.next143 = add <2 x i64> %vec.ind138, splat (i64 4)
  %40 = icmp eq i64 %index.next142, 256
  br i1 %40, label %vector.ph120, label %vector.body136, !llvm.loop !15

vector.ph120:                                     ; preds = %vector.body136
  %41 = or disjoint i64 %indvars.iv72, 3
  %arrayidx11.3 = getelementptr inbounds nuw double, ptr %a0, i64 %41
  %42 = load double, ptr %arrayidx11.3, align 8, !tbaa !6
  %arrayidx14.3 = getelementptr inbounds nuw double, ptr %a1, i64 %41
  %43 = load double, ptr %arrayidx14.3, align 8, !tbaa !6
  %add15.3 = fadd double %42, %43
  %broadcast.splatinsert121 = insertelement <2 x double> poison, double %add15.3, i64 0
  %broadcast.splat122 = shufflevector <2 x double> %broadcast.splatinsert121, <2 x double> poison, <2 x i32> zeroinitializer
  br label %vector.body123

vector.body123:                                   ; preds = %vector.body123, %vector.ph120
  %index124 = phi i64 [ 0, %vector.ph120 ], [ %index.next129, %vector.body123 ]
  %vec.ind125 = phi <2 x i64> [ <i64 0, i64 1>, %vector.ph120 ], [ %vec.ind.next130, %vector.body123 ]
  %44 = trunc <2 x i64> %vec.ind125 to <2 x i32>
  %45 = add <2 x i32> %44, splat (i32 1)
  %46 = trunc <2 x i64> %vec.ind125 to <2 x i32>
  %47 = add <2 x i32> %46, splat (i32 3)
  %48 = uitofp nneg <2 x i32> %45 to <2 x double>
  %49 = uitofp nneg <2 x i32> %47 to <2 x double>
  %50 = getelementptr inbounds nuw double, ptr %s, i64 %index124
  %51 = getelementptr inbounds nuw i8, ptr %50, i64 16
  %wide.load127 = load <2 x double>, ptr %50, align 8, !tbaa !6
  %wide.load128 = load <2 x double>, ptr %51, align 8, !tbaa !6
  %52 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %broadcast.splat122, <2 x double> %48, <2 x double> %wide.load127)
  %53 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %broadcast.splat122, <2 x double> %49, <2 x double> %wide.load128)
  store <2 x double> %52, ptr %50, align 8, !tbaa !6
  store <2 x double> %53, ptr %51, align 8, !tbaa !6
  %index.next129 = add nuw i64 %index124, 4
  %vec.ind.next130 = add <2 x i64> %vec.ind125, splat (i64 4)
  %54 = icmp eq i64 %index.next129, 256
  br i1 %54, label %vector.ph107, label %vector.body123, !llvm.loop !16

vector.ph107:                                     ; preds = %vector.body123
  %55 = or disjoint i64 %indvars.iv72, 4
  %arrayidx11.4 = getelementptr inbounds nuw double, ptr %a0, i64 %55
  %56 = load double, ptr %arrayidx11.4, align 8, !tbaa !6
  %arrayidx14.4 = getelementptr inbounds nuw double, ptr %a1, i64 %55
  %57 = load double, ptr %arrayidx14.4, align 8, !tbaa !6
  %add15.4 = fadd double %56, %57
  %broadcast.splatinsert108 = insertelement <2 x double> poison, double %add15.4, i64 0
  %broadcast.splat109 = shufflevector <2 x double> %broadcast.splatinsert108, <2 x double> poison, <2 x i32> zeroinitializer
  br label %vector.body110

vector.body110:                                   ; preds = %vector.body110, %vector.ph107
  %index111 = phi i64 [ 0, %vector.ph107 ], [ %index.next116, %vector.body110 ]
  %vec.ind112 = phi <2 x i64> [ <i64 0, i64 1>, %vector.ph107 ], [ %vec.ind.next117, %vector.body110 ]
  %58 = trunc <2 x i64> %vec.ind112 to <2 x i32>
  %59 = add <2 x i32> %58, splat (i32 1)
  %60 = trunc <2 x i64> %vec.ind112 to <2 x i32>
  %61 = add <2 x i32> %60, splat (i32 3)
  %62 = uitofp nneg <2 x i32> %59 to <2 x double>
  %63 = uitofp nneg <2 x i32> %61 to <2 x double>
  %64 = getelementptr inbounds nuw double, ptr %s, i64 %index111
  %65 = getelementptr inbounds nuw i8, ptr %64, i64 16
  %wide.load114 = load <2 x double>, ptr %64, align 8, !tbaa !6
  %wide.load115 = load <2 x double>, ptr %65, align 8, !tbaa !6
  %66 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %broadcast.splat109, <2 x double> %62, <2 x double> %wide.load114)
  %67 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %broadcast.splat109, <2 x double> %63, <2 x double> %wide.load115)
  store <2 x double> %66, ptr %64, align 8, !tbaa !6
  store <2 x double> %67, ptr %65, align 8, !tbaa !6
  %index.next116 = add nuw i64 %index111, 4
  %vec.ind.next117 = add <2 x i64> %vec.ind112, splat (i64 4)
  %68 = icmp eq i64 %index.next116, 256
  br i1 %68, label %vector.ph94, label %vector.body110, !llvm.loop !17

vector.ph94:                                      ; preds = %vector.body110
  %69 = or disjoint i64 %indvars.iv72, 5
  %arrayidx11.5 = getelementptr inbounds nuw double, ptr %a0, i64 %69
  %70 = load double, ptr %arrayidx11.5, align 8, !tbaa !6
  %arrayidx14.5 = getelementptr inbounds nuw double, ptr %a1, i64 %69
  %71 = load double, ptr %arrayidx14.5, align 8, !tbaa !6
  %add15.5 = fadd double %70, %71
  %broadcast.splatinsert95 = insertelement <2 x double> poison, double %add15.5, i64 0
  %broadcast.splat96 = shufflevector <2 x double> %broadcast.splatinsert95, <2 x double> poison, <2 x i32> zeroinitializer
  br label %vector.body97

vector.body97:                                    ; preds = %vector.body97, %vector.ph94
  %index98 = phi i64 [ 0, %vector.ph94 ], [ %index.next103, %vector.body97 ]
  %vec.ind99 = phi <2 x i64> [ <i64 0, i64 1>, %vector.ph94 ], [ %vec.ind.next104, %vector.body97 ]
  %72 = trunc <2 x i64> %vec.ind99 to <2 x i32>
  %73 = add <2 x i32> %72, splat (i32 1)
  %74 = trunc <2 x i64> %vec.ind99 to <2 x i32>
  %75 = add <2 x i32> %74, splat (i32 3)
  %76 = uitofp nneg <2 x i32> %73 to <2 x double>
  %77 = uitofp nneg <2 x i32> %75 to <2 x double>
  %78 = getelementptr inbounds nuw double, ptr %s, i64 %index98
  %79 = getelementptr inbounds nuw i8, ptr %78, i64 16
  %wide.load101 = load <2 x double>, ptr %78, align 8, !tbaa !6
  %wide.load102 = load <2 x double>, ptr %79, align 8, !tbaa !6
  %80 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %broadcast.splat96, <2 x double> %76, <2 x double> %wide.load101)
  %81 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %broadcast.splat96, <2 x double> %77, <2 x double> %wide.load102)
  store <2 x double> %80, ptr %78, align 8, !tbaa !6
  store <2 x double> %81, ptr %79, align 8, !tbaa !6
  %index.next103 = add nuw i64 %index98, 4
  %vec.ind.next104 = add <2 x i64> %vec.ind99, splat (i64 4)
  %82 = icmp eq i64 %index.next103, 256
  br i1 %82, label %vector.ph81, label %vector.body97, !llvm.loop !18

vector.ph81:                                      ; preds = %vector.body97
  %83 = or disjoint i64 %indvars.iv72, 6
  %arrayidx11.6 = getelementptr inbounds nuw double, ptr %a0, i64 %83
  %84 = load double, ptr %arrayidx11.6, align 8, !tbaa !6
  %arrayidx14.6 = getelementptr inbounds nuw double, ptr %a1, i64 %83
  %85 = load double, ptr %arrayidx14.6, align 8, !tbaa !6
  %add15.6 = fadd double %84, %85
  %broadcast.splatinsert82 = insertelement <2 x double> poison, double %add15.6, i64 0
  %broadcast.splat83 = shufflevector <2 x double> %broadcast.splatinsert82, <2 x double> poison, <2 x i32> zeroinitializer
  br label %vector.body84

vector.body84:                                    ; preds = %vector.body84, %vector.ph81
  %index85 = phi i64 [ 0, %vector.ph81 ], [ %index.next90, %vector.body84 ]
  %vec.ind86 = phi <2 x i64> [ <i64 0, i64 1>, %vector.ph81 ], [ %vec.ind.next91, %vector.body84 ]
  %86 = trunc <2 x i64> %vec.ind86 to <2 x i32>
  %87 = add <2 x i32> %86, splat (i32 1)
  %88 = trunc <2 x i64> %vec.ind86 to <2 x i32>
  %89 = add <2 x i32> %88, splat (i32 3)
  %90 = uitofp nneg <2 x i32> %87 to <2 x double>
  %91 = uitofp nneg <2 x i32> %89 to <2 x double>
  %92 = getelementptr inbounds nuw double, ptr %s, i64 %index85
  %93 = getelementptr inbounds nuw i8, ptr %92, i64 16
  %wide.load88 = load <2 x double>, ptr %92, align 8, !tbaa !6
  %wide.load89 = load <2 x double>, ptr %93, align 8, !tbaa !6
  %94 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %broadcast.splat83, <2 x double> %90, <2 x double> %wide.load88)
  %95 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %broadcast.splat83, <2 x double> %91, <2 x double> %wide.load89)
  store <2 x double> %94, ptr %92, align 8, !tbaa !6
  store <2 x double> %95, ptr %93, align 8, !tbaa !6
  %index.next90 = add nuw i64 %index85, 4
  %vec.ind.next91 = add <2 x i64> %vec.ind86, splat (i64 4)
  %96 = icmp eq i64 %index.next90, 256
  br i1 %96, label %vector.ph, label %vector.body84, !llvm.loop !19

vector.ph:                                        ; preds = %vector.body84
  %97 = or disjoint i64 %indvars.iv72, 7
  %arrayidx11.7 = getelementptr inbounds nuw double, ptr %a0, i64 %97
  %98 = load double, ptr %arrayidx11.7, align 8, !tbaa !6
  %arrayidx14.7 = getelementptr inbounds nuw double, ptr %a1, i64 %97
  %99 = load double, ptr %arrayidx14.7, align 8, !tbaa !6
  %add15.7 = fadd double %98, %99
  %broadcast.splatinsert = insertelement <2 x double> poison, double %add15.7, i64 0
  %broadcast.splat = shufflevector <2 x double> %broadcast.splatinsert, <2 x double> poison, <2 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %vec.ind = phi <2 x i64> [ <i64 0, i64 1>, %vector.ph ], [ %vec.ind.next, %vector.body ]
  %100 = trunc <2 x i64> %vec.ind to <2 x i32>
  %101 = add <2 x i32> %100, splat (i32 1)
  %102 = trunc <2 x i64> %vec.ind to <2 x i32>
  %103 = add <2 x i32> %102, splat (i32 3)
  %104 = uitofp nneg <2 x i32> %101 to <2 x double>
  %105 = uitofp nneg <2 x i32> %103 to <2 x double>
  %106 = getelementptr inbounds nuw double, ptr %s, i64 %index
  %107 = getelementptr inbounds nuw i8, ptr %106, i64 16
  %wide.load = load <2 x double>, ptr %106, align 8, !tbaa !6
  %wide.load79 = load <2 x double>, ptr %107, align 8, !tbaa !6
  %108 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %broadcast.splat, <2 x double> %104, <2 x double> %wide.load)
  %109 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %broadcast.splat, <2 x double> %105, <2 x double> %wide.load79)
  store <2 x double> %108, ptr %106, align 8, !tbaa !6
  store <2 x double> %109, ptr %107, align 8, !tbaa !6
  %index.next = add nuw i64 %index, 4
  %vec.ind.next = add <2 x i64> %vec.ind, splat (i64 4)
  %110 = icmp eq i64 %index.next, 256
  br i1 %110, label %for.cond.cleanup18.7, label %vector.body, !llvm.loop !20

for.cond.cleanup18.7:                             ; preds = %vector.body
  %indvars.iv.next73 = add nuw nsw i64 %indvars.iv72, 8
  %cmp3 = icmp samesign ult i64 %indvars.iv72, 199992
  br i1 %cmp3, label %vector.ph159, label %vector.body173, !llvm.loop !21

vector.body173:                                   ; preds = %for.cond.cleanup18.7, %vector.body173
  %index174 = phi i64 [ %index.next177, %vector.body173 ], [ 0, %for.cond.cleanup18.7 ]
  %vec.phi = phi double [ %114, %vector.body173 ], [ 0.000000e+00, %for.cond.cleanup18.7 ]
  %111 = getelementptr inbounds nuw double, ptr %s, i64 %index174
  %112 = getelementptr inbounds nuw i8, ptr %111, i64 16
  %wide.load175 = load <2 x double>, ptr %111, align 8, !tbaa !6
  %wide.load176 = load <2 x double>, ptr %112, align 8, !tbaa !6
  %113 = tail call double @llvm.vector.reduce.fadd.v2f64(double %vec.phi, <2 x double> %wide.load175)
  %114 = tail call double @llvm.vector.reduce.fadd.v2f64(double %113, <2 x double> %wide.load176)
  %index.next177 = add nuw i64 %index174, 4
  %115 = icmp eq i64 %index.next177, 256
  br i1 %115, label %for.cond.cleanup36, label %vector.body173, !llvm.loop !22

for.cond.cleanup36:                               ; preds = %vector.body173
  store double %114, ptr %out, align 8, !tbaa !6
  call void @llvm.lifetime.end.p0(ptr nonnull %s) #4
  ret double %114
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x double> @llvm.fmuladd.v2f64(<2 x double>, <2 x double>, <2 x double>) #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.vector.reduce.fadd.v2f64(double, <2 x double>) #3

attributes #0 = { nofree noinline norecurse nosync nounwind memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git 9f790e9e900f8dab0e35b49a5844c2900865231e)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"double", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = distinct !{!10, !11, !12, !13}
!11 = !{!"llvm.loop.mustprogress"}
!12 = !{!"llvm.loop.isvectorized", i32 1}
!13 = !{!"llvm.loop.unroll.runtime.disable"}
!14 = distinct !{!14, !11, !12, !13}
!15 = distinct !{!15, !11, !12, !13}
!16 = distinct !{!16, !11, !12, !13}
!17 = distinct !{!17, !11, !12, !13}
!18 = distinct !{!18, !11, !12, !13}
!19 = distinct !{!19, !11, !12, !13}
!20 = distinct !{!20, !11, !12, !13}
!21 = distinct !{!21, !11}
!22 = distinct !{!22, !11, !12, !13}
