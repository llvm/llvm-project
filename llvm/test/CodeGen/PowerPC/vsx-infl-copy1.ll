; RUN: llc -verify-machineinstrs -mcpu=pwr7 < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

@ub = external global [1024 x i32], align 4
@uc = external global [1024 x i32], align 4

; Function Attrs: noinline nounwind
define <4 x i32> @_Z8example9Pj(ptr %addr1, i64 %input1, i64 %input2) #0 {
entry:
  br label %vector.body

; CHECK-LABEL: @_Z8example9Pj
; CHECK: xxlxor 
; CHECK: xxlxor 
; CHECK: xxlxor 
; CHECK: xxlxor 
; CHECK: xxlxor 

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %vec.phi = phi <4 x i32> [ zeroinitializer, %entry ], [ %43, %vector.body ]
  %vec.phi20 = phi <4 x i32> [ zeroinitializer, %entry ], [ %44, %vector.body ]
  %vec.phi21 = phi <4 x i32> [ zeroinitializer, %entry ], [ %45, %vector.body ]
  %vec.phi23 = phi <4 x i32> [ zeroinitializer, %entry ], [ %46, %vector.body ]
  %vec.phi24 = phi <4 x i32> [ zeroinitializer, %entry ], [ %47, %vector.body ]
  %vec.phi25 = phi <4 x i32> [ zeroinitializer, %entry ], [ %48, %vector.body ]
  %vec.phi26 = phi <4 x i32> [ zeroinitializer, %entry ], [ %49, %vector.body ]
  %vec.phi27 = phi <4 x i32> [ zeroinitializer, %entry ], [ %50, %vector.body ]
  %vec.phi28 = phi <4 x i32> [ zeroinitializer, %entry ], [ %51, %vector.body ]
  %vec.phi29 = phi <4 x i32> [ zeroinitializer, %entry ], [ %52, %vector.body ]
  %vec.phi30 = phi <4 x i32> [ zeroinitializer, %entry ], [ %53, %vector.body ]
  %wide.load32 = load <4 x i32>, ptr null, align 4
  %.sum82 = add i64 %index, 24
  %0 = getelementptr [1024 x i32], ptr @ub, i64 0, i64 %.sum82
  %1 = bitcast ptr %0 to ptr
  %wide.load36 = load <4 x i32>, ptr %1, align 4
  %wide.load37 = load <4 x i32>, ptr %addr1, align 4
  %.sum84 = add i64 %index, 32
  %2 = getelementptr [1024 x i32], ptr @ub, i64 0, i64 %.sum84
  %3 = bitcast ptr %2 to ptr
  %wide.load38 = load <4 x i32>, ptr %3, align 4
  %.sum85 = add i64 %index, 36
  %4 = getelementptr [1024 x i32], ptr @ub, i64 0, i64 %.sum85
  %5 = bitcast ptr %4 to ptr
  %wide.load39 = load <4 x i32>, ptr %5, align 4
  %6 = getelementptr [1024 x i32], ptr @ub, i64 0, i64 %input1
  %7 = bitcast ptr %6 to ptr
  %wide.load40 = load <4 x i32>, ptr %7, align 4
  %.sum87 = add i64 %index, 44
  %8 = getelementptr [1024 x i32], ptr @ub, i64 0, i64 %.sum87
  %9 = bitcast ptr %8 to ptr
  %wide.load41 = load <4 x i32>, ptr %9, align 4
  %10 = getelementptr inbounds [1024 x i32], ptr @uc, i64 0, i64 %index
  %11 = bitcast ptr %10 to ptr
  %wide.load42 = load <4 x i32>, ptr %11, align 4
  %.sum8889 = or i64 %index, 4
  %12 = getelementptr [1024 x i32], ptr @uc, i64 0, i64 %.sum8889
  %13 = bitcast ptr %12 to ptr
  %wide.load43 = load <4 x i32>, ptr %13, align 4
  %.sum9091 = or i64 %index, 8
  %14 = getelementptr [1024 x i32], ptr @uc, i64 0, i64 %.sum9091
  %15 = bitcast ptr %14 to ptr
  %wide.load44 = load <4 x i32>, ptr %15, align 4
  %.sum94 = add i64 %index, 16
  %16 = getelementptr [1024 x i32], ptr @uc, i64 0, i64 %.sum94
  %17 = bitcast ptr %16 to ptr
  %wide.load46 = load <4 x i32>, ptr %17, align 4
  %.sum95 = add i64 %index, 20
  %18 = getelementptr [1024 x i32], ptr @uc, i64 0, i64 %.sum95
  %19 = bitcast ptr %18 to ptr
  %wide.load47 = load <4 x i32>, ptr %19, align 4
  %20 = getelementptr [1024 x i32], ptr @uc, i64 0, i64 %input2
  %21 = bitcast ptr %20 to ptr
  %wide.load48 = load <4 x i32>, ptr %21, align 4
  %.sum97 = add i64 %index, 28
  %22 = getelementptr [1024 x i32], ptr @uc, i64 0, i64 %.sum97
  %23 = bitcast ptr %22 to ptr
  %wide.load49 = load <4 x i32>, ptr %23, align 4
  %.sum98 = add i64 %index, 32
  %24 = getelementptr [1024 x i32], ptr @uc, i64 0, i64 %.sum98
  %25 = bitcast ptr %24 to ptr
  %wide.load50 = load <4 x i32>, ptr %25, align 4
  %.sum99 = add i64 %index, 36
  %26 = getelementptr [1024 x i32], ptr @uc, i64 0, i64 %.sum99
  %27 = bitcast ptr %26 to ptr
  %wide.load51 = load <4 x i32>, ptr %27, align 4
  %.sum100 = add i64 %index, 40
  %28 = getelementptr [1024 x i32], ptr @uc, i64 0, i64 %.sum100
  %29 = bitcast ptr %28 to ptr
  %wide.load52 = load <4 x i32>, ptr %29, align 4
  %.sum101 = add i64 %index, 44
  %30 = getelementptr [1024 x i32], ptr @uc, i64 0, i64 %.sum101
  %31 = bitcast ptr %30 to ptr
  %wide.load53 = load <4 x i32>, ptr %31, align 4
  %32 = add <4 x i32> zeroinitializer, %vec.phi
  %33 = add <4 x i32> zeroinitializer, %vec.phi20
  %34 = add <4 x i32> %wide.load32, %vec.phi21
  %35 = add <4 x i32> zeroinitializer, %vec.phi23
  %36 = add <4 x i32> zeroinitializer, %vec.phi24
  %37 = add <4 x i32> %wide.load36, %vec.phi25
  %38 = add <4 x i32> %wide.load37, %vec.phi26
  %39 = add <4 x i32> %wide.load38, %vec.phi27
  %40 = add <4 x i32> %wide.load39, %vec.phi28
  %41 = add <4 x i32> %wide.load40, %vec.phi29
  %42 = add <4 x i32> %wide.load41, %vec.phi30
  %43 = sub <4 x i32> %32, %wide.load42
  %44 = sub <4 x i32> %33, %wide.load43
  %45 = sub <4 x i32> %34, %wide.load44
  %46 = sub <4 x i32> %35, %wide.load46
  %47 = sub <4 x i32> %36, %wide.load47
  %48 = sub <4 x i32> %37, %wide.load48
  %49 = sub <4 x i32> %38, %wide.load49
  %50 = sub <4 x i32> %39, %wide.load50
  %51 = sub <4 x i32> %40, %wide.load51
  %52 = sub <4 x i32> %41, %wide.load52
  %53 = sub <4 x i32> %42, %wide.load53
  %index.next = add i64 %index, 48
  br i1 false, label %middle.block, label %vector.body

middle.block:                                     ; preds = %vector.body
  %.lcssa112 = phi <4 x i32> [ %53, %vector.body ]
  %.lcssa111 = phi <4 x i32> [ %52, %vector.body ]
  %.lcssa110 = phi <4 x i32> [ %51, %vector.body ]
  %.lcssa109 = phi <4 x i32> [ %50, %vector.body ]
  %.lcssa108 = phi <4 x i32> [ %49, %vector.body ]
  %.lcssa107 = phi <4 x i32> [ %48, %vector.body ]
  %.lcssa106 = phi <4 x i32> [ %47, %vector.body ]
  %.lcssa105 = phi <4 x i32> [ %46, %vector.body ]
  %.lcssa103 = phi <4 x i32> [ %45, %vector.body ]
  %.lcssa102 = phi <4 x i32> [ %44, %vector.body ]
  %.lcssa = phi <4 x i32> [ %43, %vector.body ]
  %54 = add <4 x i32> %.lcssa112, %.lcssa111
  %55 = add <4 x i32> %.lcssa110, %54
  %56 = add <4 x i32> %.lcssa109, %55
  %57 = add <4 x i32> %.lcssa108, %56
  %58 = add <4 x i32> %.lcssa107, %57
  %59 = add <4 x i32> %.lcssa106, %58
  %60 = add <4 x i32> %.lcssa105, %59
  %61 = add <4 x i32> %.lcssa103, %60
  %62 = add <4 x i32> %.lcssa102, %61
  ret <4 x i32> %62
}

attributes #0 = { noinline nounwind }

