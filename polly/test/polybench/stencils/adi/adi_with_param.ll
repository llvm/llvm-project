; RUN: opt %loadPolly  %defaultOpts -polly-analyze-ir  -print-top-scop-only -analyze %s | FileCheck %s
; XFAIL: *
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@X = common global [1024 x [1024 x double]] zeroinitializer, align 32
@A = common global [1024 x [1024 x double]] zeroinitializer, align 32
@B = common global [1024 x [1024 x double]] zeroinitializer, align 32
@stderr = external global %struct._IO_FILE*
@.str = private constant [8 x i8] c"%0.2lf \00", align 1

define void @scop_func(i64 %n) nounwind {
bb.nph81:
  %0 = icmp sgt i64 %n, 0
  %1 = icmp sgt i64 %n, 1
  %2 = add nsw i64 %n, -2
  %3 = icmp sgt i64 %2, 0
  %4 = add nsw i64 %n, -3
  %tmp = add i64 %n, -1
  br label %bb5.preheader

bb.nph:                                           ; preds = %bb.nph.preheader, %bb4
  %storemerge112 = phi i64 [ %16, %bb4 ], [ 0, %bb.nph.preheader ]
  %scevgep86.phi.trans.insert = getelementptr [1024 x [1024 x double]]* @B, i64 0, i64 %storemerge112, i64 0
  %scevgep85.phi.trans.insert = getelementptr [1024 x [1024 x double]]* @X, i64 0, i64 %storemerge112, i64 0
  %.pre = load double* %scevgep85.phi.trans.insert, align 32
  %.pre149 = load double* %scevgep86.phi.trans.insert, align 32
  br label %bb2

bb2:                                              ; preds = %bb2, %bb.nph
  %5 = phi double [ %.pre149, %bb.nph ], [ %15, %bb2 ]
  %6 = phi double [ %.pre, %bb.nph ], [ %11, %bb2 ]
  %indvar = phi i64 [ 0, %bb.nph ], [ %tmp90, %bb2 ]
  %tmp42 = add i64 %indvar, 1
  %scevgep84 = getelementptr [1024 x [1024 x double]]* @B, i64 0, i64 %storemerge112, i64 %tmp42
  %scevgep83 = getelementptr [1024 x [1024 x double]]* @A, i64 0, i64 %storemerge112, i64 %tmp42
  %scevgep = getelementptr [1024 x [1024 x double]]* @X, i64 0, i64 %storemerge112, i64 %tmp42
  %tmp90 = add i64 %indvar, 1
  %7 = load double* %scevgep, align 8
  %8 = load double* %scevgep83, align 8
  %9 = fmul double %6, %8
  %10 = fdiv double %9, %5
  %11 = fsub double %7, %10
  store double %11, double* %scevgep, align 8
  %12 = load double* %scevgep84, align 8
  %13 = fmul double %8, %8
  %14 = fdiv double %13, %5
  %15 = fsub double %12, %14
  store double %15, double* %scevgep84, align 8
  %exitcond37 = icmp eq i64 %tmp90, %tmp
  br i1 %exitcond37, label %bb4, label %bb2

bb4:                                              ; preds = %bb2
  %16 = add nsw i64 %storemerge112, 1
  %exitcond = icmp eq i64 %16, %n
  br i1 %exitcond, label %bb8.loopexit.loopexit, label %bb.nph

bb.nph16:                                         ; preds = %bb5.preheader
  br i1 %1, label %bb.nph.preheader, label %bb8.loopexit

bb.nph.preheader:                                 ; preds = %bb.nph16
  br label %bb.nph

bb7:                                              ; preds = %bb7.preheader, %bb7
  %storemerge217 = phi i64 [ %20, %bb7 ], [ 0, %bb7.preheader ]
  %scevgep96 = getelementptr [1024 x [1024 x double]]* @B, i64 0, i64 %storemerge217, i64 %tmp
  %scevgep95 = getelementptr [1024 x [1024 x double]]* @X, i64 0, i64 %storemerge217, i64 %tmp
  %17 = load double* %scevgep95, align 8
  %18 = load double* %scevgep96, align 8
  %19 = fdiv double %17, %18
  store double %19, double* %scevgep95, align 8
  %20 = add nsw i64 %storemerge217, 1
  %exitcond18 = icmp eq i64 %20, %n
  br i1 %exitcond18, label %bb14.loopexit, label %bb7

bb8.loopexit.loopexit:                            ; preds = %bb4
  br label %bb8.loopexit

bb8.loopexit:                                     ; preds = %bb8.loopexit.loopexit, %bb.nph16
  br i1 %0, label %bb7.preheader, label %bb20.loopexit

bb7.preheader:                                    ; preds = %bb8.loopexit
  br label %bb7

bb11:                                             ; preds = %bb12.preheader, %bb11
  %storemerge920 = phi i64 [ %28, %bb11 ], [ 0, %bb12.preheader ]
  %tmp30 = mul i64 %storemerge920, -1
  %tmp31 = add i64 %4, %tmp30
  %scevgep104 = getelementptr [1024 x [1024 x double]]* @B, i64 0, i64 %storemerge323, i64 %tmp31
  %scevgep103 = getelementptr [1024 x [1024 x double]]* @A, i64 0, i64 %storemerge323, i64 %tmp31
  %scevgep102 = getelementptr [1024 x [1024 x double]]* @X, i64 0, i64 %storemerge323, i64 %tmp31
  %tmp35 = add i64 %2, %tmp30
  %scevgep100 = getelementptr [1024 x [1024 x double]]* @X, i64 0, i64 %storemerge323, i64 %tmp35
  %21 = load double* %scevgep100, align 8
  %22 = load double* %scevgep102, align 8
  %23 = load double* %scevgep103, align 8
  %24 = fmul double %22, %23
  %25 = fsub double %21, %24
  %26 = load double* %scevgep104, align 8
  %27 = fdiv double %25, %26
  store double %27, double* %scevgep100, align 8
  %28 = add nsw i64 %storemerge920, 1
  %exitcond21 = icmp eq i64 %28, %2
  br i1 %exitcond21, label %bb13, label %bb11

bb13:                                             ; preds = %bb11
  %29 = add nsw i64 %storemerge323, 1
  %exitcond29 = icmp eq i64 %29, %n
  br i1 %exitcond29, label %bb20.loopexit.loopexit, label %bb12.preheader

bb14.loopexit:                                    ; preds = %bb7
  %.not = xor i1 %0, true
  %.not150 = xor i1 %3, true
  %brmerge = or i1 %.not, %.not150
  br i1 %brmerge, label %bb20.loopexit, label %bb12.preheader.preheader

bb12.preheader.preheader:                         ; preds = %bb14.loopexit
  br label %bb12.preheader

bb12.preheader:                                   ; preds = %bb12.preheader.preheader, %bb13
  %storemerge323 = phi i64 [ %29, %bb13 ], [ 0, %bb12.preheader.preheader ]
  br label %bb11

bb17:                                             ; preds = %bb18.preheader, %bb17
  %storemerge828 = phi i64 [ %41, %bb17 ], [ 0, %bb18.preheader ]
  %scevgep119 = getelementptr [1024 x [1024 x double]]* @B, i64 0, i64 %indvar114, i64 %storemerge828
  %scevgep118 = getelementptr [1024 x [1024 x double]]* @X, i64 0, i64 %indvar114, i64 %storemerge828
  %scevgep121 = getelementptr [1024 x [1024 x double]]* @B, i64 0, i64 %tmp11, i64 %storemerge828
  %scevgep120 = getelementptr [1024 x [1024 x double]]* @A, i64 0, i64 %tmp11, i64 %storemerge828
  %scevgep117 = getelementptr [1024 x [1024 x double]]* @X, i64 0, i64 %tmp11, i64 %storemerge828
  %30 = load double* %scevgep117, align 8
  %31 = load double* %scevgep118, align 8
  %32 = load double* %scevgep120, align 8
  %33 = fmul double %31, %32
  %34 = load double* %scevgep119, align 8
  %35 = fdiv double %33, %34
  %36 = fsub double %30, %35
  store double %36, double* %scevgep117, align 8
  %37 = load double* %scevgep121, align 8
  %38 = fmul double %32, %32
  %39 = fdiv double %38, %34
  %40 = fsub double %37, %39
  store double %40, double* %scevgep121, align 8
  %41 = add nsw i64 %storemerge828, 1
  %exitcond1 = icmp eq i64 %41, %n
  br i1 %exitcond1, label %bb19, label %bb17

bb19:                                             ; preds = %bb17
  %tmp125 = add i64 %indvar114, 1
  %exitcond8 = icmp eq i64 %tmp125, %tmp
  br i1 %exitcond8, label %bb23.loopexit.loopexit, label %bb18.preheader

bb20.loopexit.loopexit:                           ; preds = %bb13
  br label %bb20.loopexit

bb20.loopexit:                                    ; preds = %bb20.loopexit.loopexit, %bb5.preheader, %bb14.loopexit, %bb8.loopexit
  br i1 %1, label %bb.nph34, label %bb23.loopexit

bb.nph34:                                         ; preds = %bb20.loopexit
  br i1 %0, label %bb18.preheader.preheader, label %bb29.loopexit

bb18.preheader.preheader:                         ; preds = %bb.nph34
  br label %bb18.preheader

bb18.preheader:                                   ; preds = %bb18.preheader.preheader, %bb19
  %indvar114 = phi i64 [ %tmp125, %bb19 ], [ 0, %bb18.preheader.preheader ]
  %tmp11 = add i64 %indvar114, 1
  br label %bb17

bb22:                                             ; preds = %bb22.preheader, %bb22
  %storemerge535 = phi i64 [ %45, %bb22 ], [ 0, %bb22.preheader ]
  %scevgep131 = getelementptr [1024 x [1024 x double]]* @B, i64 0, i64 %tmp, i64 %storemerge535
  %scevgep130 = getelementptr [1024 x [1024 x double]]* @X, i64 0, i64 %tmp, i64 %storemerge535
  %42 = load double* %scevgep130, align 8
  %43 = load double* %scevgep131, align 8
  %44 = fdiv double %42, %43
  store double %44, double* %scevgep130, align 8
  %45 = add nsw i64 %storemerge535, 1
  %exitcond15 = icmp eq i64 %45, %n
  br i1 %exitcond15, label %bb29.loopexit.loopexit, label %bb22

bb23.loopexit.loopexit:                           ; preds = %bb19
  br label %bb23.loopexit

bb23.loopexit:                                    ; preds = %bb23.loopexit.loopexit, %bb20.loopexit
  br i1 %0, label %bb22.preheader, label %bb29.loopexit

bb22.preheader:                                   ; preds = %bb23.loopexit
  br label %bb22

bb26:                                             ; preds = %bb27.preheader, %bb26
  %storemerge737 = phi i64 [ %53, %bb26 ], [ 0, %bb27.preheader ]
  %scevgep138 = getelementptr [1024 x [1024 x double]]* @A, i64 0, i64 %tmp58, i64 %storemerge737
  %scevgep137 = getelementptr [1024 x [1024 x double]]* @X, i64 0, i64 %tmp58, i64 %storemerge737
  %scevgep139 = getelementptr [1024 x [1024 x double]]* @B, i64 0, i64 %tmp61, i64 %storemerge737
  %scevgep135 = getelementptr [1024 x [1024 x double]]* @X, i64 0, i64 %tmp61, i64 %storemerge737
  %46 = load double* %scevgep135, align 8
  %47 = load double* %scevgep137, align 8
  %48 = load double* %scevgep138, align 8
  %49 = fmul double %47, %48
  %50 = fsub double %46, %49
  %51 = load double* %scevgep139, align 8
  %52 = fdiv double %50, %51
  store double %52, double* %scevgep135, align 8
  %53 = add nsw i64 %storemerge737, 1
  %exitcond48 = icmp eq i64 %53, %n
  br i1 %exitcond48, label %bb28, label %bb26

bb28:                                             ; preds = %bb26
  %54 = add nsw i64 %storemerge640, 1
  %exitcond56 = icmp eq i64 %54, %2
  br i1 %exitcond56, label %bb30.loopexit, label %bb27.preheader

bb29.loopexit.loopexit:                           ; preds = %bb22
  br label %bb29.loopexit

bb29.loopexit:                                    ; preds = %bb29.loopexit.loopexit, %bb23.loopexit, %bb.nph34
  %.not151 = xor i1 %3, true
  %.not152 = xor i1 %0, true
  %brmerge153 = or i1 %.not151, %.not152
  br i1 %brmerge153, label %bb30, label %bb27.preheader.preheader

bb27.preheader.preheader:                         ; preds = %bb29.loopexit
  br label %bb27.preheader

bb27.preheader:                                   ; preds = %bb27.preheader.preheader, %bb28
  %storemerge640 = phi i64 [ %54, %bb28 ], [ 0, %bb27.preheader.preheader ]
  %tmp57 = mul i64 %storemerge640, -1
  %tmp58 = add i64 %4, %tmp57
  %tmp61 = add i64 %2, %tmp57
  br label %bb26

bb30.loopexit:                                    ; preds = %bb28
  br label %bb30

bb30:                                             ; preds = %bb30.loopexit, %bb29.loopexit
  %55 = add nsw i64 %storemerge46, 1
  %exitcond64 = icmp eq i64 %55, 10
  br i1 %exitcond64, label %return, label %bb5.preheader

bb5.preheader:                                    ; preds = %bb30, %bb.nph81
  %storemerge46 = phi i64 [ 0, %bb.nph81 ], [ %55, %bb30 ]
  br i1 %0, label %bb.nph16, label %bb20.loopexit

return:                                           ; preds = %bb30
  ret void
}
