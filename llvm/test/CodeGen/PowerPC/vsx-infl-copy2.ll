; RUN: llc -verify-machineinstrs -mcpu=pwr7 < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind
define void @_Z28test_goto_loop_unroll_factorILi22EiEvPKT0_iPKc(i32* nocapture readonly %first) #0 {
entry:
  br i1 false, label %loop2_start, label %if.end5

; CHECK-LABEL: @_Z28test_goto_loop_unroll_factorILi22EiEvPKT0_iPKc

loop2_start:                                      ; preds = %loop2_start, %entry
  br i1 undef, label %loop2_start, label %if.then.i31

if.end5:                                          ; preds = %entry
  br i1 undef, label %loop_start.preheader, label %if.then.i31

loop_start.preheader:                             ; preds = %if.end5
  br i1 false, label %middle.block, label %vector.body

vector.body:                                      ; preds = %vector.body, %loop_start.preheader
  %vec.phi61 = phi <4 x i32> [ %34, %vector.body ], [ zeroinitializer, %loop_start.preheader ]
  %vec.phi62 = phi <4 x i32> [ %35, %vector.body ], [ zeroinitializer, %loop_start.preheader ]
  %vec.phi63 = phi <4 x i32> [ %36, %vector.body ], [ zeroinitializer, %loop_start.preheader ]
  %vec.phi65 = phi <4 x i32> [ %37, %vector.body ], [ zeroinitializer, %loop_start.preheader ]
  %vec.phi67 = phi <4 x i32> [ %38, %vector.body ], [ zeroinitializer, %loop_start.preheader ]
  %vec.phi68 = phi <4 x i32> [ %39, %vector.body ], [ zeroinitializer, %loop_start.preheader ]
  %vec.phi69 = phi <4 x i32> [ %40, %vector.body ], [ zeroinitializer, %loop_start.preheader ]
  %vec.phi70 = phi <4 x i32> [ %41, %vector.body ], [ zeroinitializer, %loop_start.preheader ]
  %vec.phi71 = phi <4 x i32> [ %42, %vector.body ], [ zeroinitializer, %loop_start.preheader ]
  %.sum = add i64 0, 4
  %wide.load72 = load <4 x i32>, <4 x i32>* null, align 4
  %.sum109 = add i64 0, 8
  %0 = getelementptr i32, i32* %first, i64 %.sum109
  %1 = bitcast i32* %0 to <4 x i32>*
  %wide.load73 = load <4 x i32>, <4 x i32>* %1, align 4
  %.sum110 = add i64 0, 12
  %2 = getelementptr i32, i32* %first, i64 %.sum110
  %3 = bitcast i32* %2 to <4 x i32>*
  %wide.load74 = load <4 x i32>, <4 x i32>* %3, align 4
  %.sum112 = add i64 0, 20
  %4 = getelementptr i32, i32* %first, i64 %.sum112
  %5 = bitcast i32* %4 to <4 x i32>*
  %wide.load76 = load <4 x i32>, <4 x i32>* %5, align 4
  %.sum114 = add i64 0, 28
  %6 = getelementptr i32, i32* %first, i64 %.sum114
  %7 = bitcast i32* %6 to <4 x i32>*
  %wide.load78 = load <4 x i32>, <4 x i32>* %7, align 4
  %.sum115 = add i64 0, 32
  %8 = getelementptr i32, i32* %first, i64 %.sum115
  %9 = bitcast i32* %8 to <4 x i32>*
  %wide.load79 = load <4 x i32>, <4 x i32>* %9, align 4
  %.sum116 = add i64 0, 36
  %10 = getelementptr i32, i32* %first, i64 %.sum116
  %11 = bitcast i32* %10 to <4 x i32>*
  %wide.load80 = load <4 x i32>, <4 x i32>* %11, align 4
  %.sum117 = add i64 0, 40
  %12 = getelementptr i32, i32* %first, i64 %.sum117
  %13 = bitcast i32* %12 to <4 x i32>*
  %wide.load81 = load <4 x i32>, <4 x i32>* %13, align 4
  %.sum118 = add i64 0, 44
  %14 = getelementptr i32, i32* %first, i64 %.sum118
  %15 = bitcast i32* %14 to <4 x i32>*
  %wide.load82 = load <4 x i32>, <4 x i32>* %15, align 4
  %16 = mul <4 x i32> %wide.load72, <i32 269850533, i32 269850533, i32 269850533, i32 269850533>
  %17 = mul <4 x i32> %wide.load73, <i32 269850533, i32 269850533, i32 269850533, i32 269850533>
  %18 = mul <4 x i32> %wide.load74, <i32 269850533, i32 269850533, i32 269850533, i32 269850533>
  %19 = mul <4 x i32> %wide.load76, <i32 269850533, i32 269850533, i32 269850533, i32 269850533>
  %20 = mul <4 x i32> %wide.load78, <i32 269850533, i32 269850533, i32 269850533, i32 269850533>
  %21 = mul <4 x i32> %wide.load79, <i32 269850533, i32 269850533, i32 269850533, i32 269850533>
  %22 = mul <4 x i32> %wide.load80, <i32 269850533, i32 269850533, i32 269850533, i32 269850533>
  %23 = mul <4 x i32> %wide.load81, <i32 269850533, i32 269850533, i32 269850533, i32 269850533>
  %24 = mul <4 x i32> %wide.load82, <i32 269850533, i32 269850533, i32 269850533, i32 269850533>
  %25 = add <4 x i32> %16, <i32 -1138325064, i32 -1138325064, i32 -1138325064, i32 -1138325064>
  %26 = add <4 x i32> %17, <i32 -1138325064, i32 -1138325064, i32 -1138325064, i32 -1138325064>
  %27 = add <4 x i32> %18, <i32 -1138325064, i32 -1138325064, i32 -1138325064, i32 -1138325064>
  %28 = add <4 x i32> %19, <i32 -1138325064, i32 -1138325064, i32 -1138325064, i32 -1138325064>
  %29 = add <4 x i32> %20, <i32 -1138325064, i32 -1138325064, i32 -1138325064, i32 -1138325064>
  %30 = add <4 x i32> %21, <i32 -1138325064, i32 -1138325064, i32 -1138325064, i32 -1138325064>
  %31 = add <4 x i32> %22, <i32 -1138325064, i32 -1138325064, i32 -1138325064, i32 -1138325064>
  %32 = add <4 x i32> %23, <i32 -1138325064, i32 -1138325064, i32 -1138325064, i32 -1138325064>
  %33 = add <4 x i32> %24, <i32 -1138325064, i32 -1138325064, i32 -1138325064, i32 -1138325064>
  %34 = add nsw <4 x i32> %25, %vec.phi61
  %35 = add nsw <4 x i32> %26, %vec.phi62
  %36 = add nsw <4 x i32> %27, %vec.phi63
  %37 = add nsw <4 x i32> %28, %vec.phi65
  %38 = add nsw <4 x i32> %29, %vec.phi67
  %39 = add nsw <4 x i32> %30, %vec.phi68
  %40 = add nsw <4 x i32> %31, %vec.phi69
  %41 = add nsw <4 x i32> %32, %vec.phi70
  %42 = add nsw <4 x i32> %33, %vec.phi71
  br i1 false, label %middle.block, label %vector.body

middle.block:                                     ; preds = %vector.body, %loop_start.preheader
  %rdx.vec.exit.phi85 = phi <4 x i32> [ zeroinitializer, %loop_start.preheader ], [ %34, %vector.body ]
  %rdx.vec.exit.phi86 = phi <4 x i32> [ zeroinitializer, %loop_start.preheader ], [ %35, %vector.body ]
  %rdx.vec.exit.phi87 = phi <4 x i32> [ zeroinitializer, %loop_start.preheader ], [ %36, %vector.body ]
  %rdx.vec.exit.phi89 = phi <4 x i32> [ zeroinitializer, %loop_start.preheader ], [ %37, %vector.body ]
  %rdx.vec.exit.phi91 = phi <4 x i32> [ zeroinitializer, %loop_start.preheader ], [ %38, %vector.body ]
  %rdx.vec.exit.phi92 = phi <4 x i32> [ zeroinitializer, %loop_start.preheader ], [ %39, %vector.body ]
  %rdx.vec.exit.phi93 = phi <4 x i32> [ zeroinitializer, %loop_start.preheader ], [ %40, %vector.body ]
  %rdx.vec.exit.phi94 = phi <4 x i32> [ zeroinitializer, %loop_start.preheader ], [ %41, %vector.body ]
  %rdx.vec.exit.phi95 = phi <4 x i32> [ zeroinitializer, %loop_start.preheader ], [ %42, %vector.body ]
  br i1 false, label %if.then.i31, label %loop_start.prol

loop_start.prol:                                  ; preds = %loop_start.prol, %middle.block
  br label %loop_start.prol

if.then.i31:                                      ; preds = %middle.block, %if.end5, %loop2_start
  unreachable
}

attributes #0 = { nounwind }

