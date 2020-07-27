; RUN: llc -verify-machineinstrs < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind
define void @foo(double* noalias nocapture %a, double* noalias nocapture readonly %b) #0 {
entry:
  br label %vector.body

; CHECK-LABEL: @foo
; Make sure that the offset constants we use are all even (only the last should be odd).
; CHECK-DAG: li {{[0-9]+}}, 1056
; CHECK-DAG: li {{[0-9]+}}, 1088
; CHECK-DAG: li {{[0-9]+}}, 1152
; CHECK-DAG: li {{[0-9]+}}, 1216
; CHECK-DAG: li {{[0-9]+}}, 1280
; CHECK-DAG: li {{[0-9]+}}, 1344
; CHECK-DAG: li {{[0-9]+}}, 1408
; CHECK-DAG: li {{[0-9]+}}, 1472
; CHECK-DAG: li {{[0-9]+}}, 1536
; CHECK-DAG: li {{[0-9]+}}, 1600
; CHECK-DAG: li {{[0-9]+}}, 1568
; CHECK-DAG: li {{[0-9]+}}, 1664
; CHECK-DAG: li {{[0-9]+}}, 1632
; CHECK-DAG: li {{[0-9]+}}, 1728
; CHECK-DAG: li {{[0-9]+}}, 1696
; CHECK-DAG: li {{[0-9]+}}, 1792
; CHECK-DAG: li {{[0-9]+}}, 1760
; CHECK-DAG: li {{[0-9]+}}, 1856
; CHECK-DAG: li {{[0-9]+}}, 1824
; CHECK-DAG: li {{[0-9]+}}, 1920
; CHECK-DAG: li {{[0-9]+}}, 1888
; CHECK-DAG: li {{[0-9]+}}, 1984
; CHECK-DAG: li {{[0-9]+}}, 1952
; CHECK-DAG: li {{[0-9]+}}, 2016
; CHECK-DAG: li {{[0-9]+}}, 1024
; CHECK-DAG: li {{[0-9]+}}, 1120
; CHECK-DAG: li {{[0-9]+}}, 1184
; CHECK-DAG: li {{[0-9]+}}, 1248
; CHECK-DAG: li {{[0-9]+}}, 1312
; CHECK-DAG: li {{[0-9]+}}, 1376
; CHECK-DAG: li {{[0-9]+}}, 1440
; CHECK-DAG: li {{[0-9]+}}, 1504
; CHECK-DAG: li {{[0-9]+}}, 2047
; CHECK: blr

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i64 [ 0, %entry ], [ %index.next.15, %vector.body ]
  %0 = shl i64 %index, 1
  %1 = getelementptr inbounds double, double* %b, i64 %0
  %2 = bitcast double* %1 to <8 x double>*
  %wide.vec = load <8 x double>, <8 x double>* %2, align 8
  %strided.vec = shufflevector <8 x double> %wide.vec, <8 x double> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %3 = fadd <4 x double> %strided.vec, <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>
  %4 = getelementptr inbounds double, double* %a, i64 %index
  %5 = bitcast double* %4 to <4 x double>*
  store <4 x double> %3, <4 x double>* %5, align 8
  %index.next = or i64 %index, 4
  %6 = shl i64 %index.next, 1
  %7 = getelementptr inbounds double, double* %b, i64 %6
  %8 = bitcast double* %7 to <8 x double>*
  %wide.vec.1 = load <8 x double>, <8 x double>* %8, align 8
  %strided.vec.1 = shufflevector <8 x double> %wide.vec.1, <8 x double> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %9 = fadd <4 x double> %strided.vec.1, <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>
  %10 = getelementptr inbounds double, double* %a, i64 %index.next
  %11 = bitcast double* %10 to <4 x double>*
  store <4 x double> %9, <4 x double>* %11, align 8
  %index.next.1 = or i64 %index, 8
  %12 = shl i64 %index.next.1, 1
  %13 = getelementptr inbounds double, double* %b, i64 %12
  %14 = bitcast double* %13 to <8 x double>*
  %wide.vec.2 = load <8 x double>, <8 x double>* %14, align 8
  %strided.vec.2 = shufflevector <8 x double> %wide.vec.2, <8 x double> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %15 = fadd <4 x double> %strided.vec.2, <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>
  %16 = getelementptr inbounds double, double* %a, i64 %index.next.1
  %17 = bitcast double* %16 to <4 x double>*
  store <4 x double> %15, <4 x double>* %17, align 8
  %index.next.2 = or i64 %index, 12
  %18 = shl i64 %index.next.2, 1
  %19 = getelementptr inbounds double, double* %b, i64 %18
  %20 = bitcast double* %19 to <8 x double>*
  %wide.vec.3 = load <8 x double>, <8 x double>* %20, align 8
  %strided.vec.3 = shufflevector <8 x double> %wide.vec.3, <8 x double> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %21 = fadd <4 x double> %strided.vec.3, <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>
  %22 = getelementptr inbounds double, double* %a, i64 %index.next.2
  %23 = bitcast double* %22 to <4 x double>*
  store <4 x double> %21, <4 x double>* %23, align 8
  %index.next.3 = or i64 %index, 16
  %24 = shl i64 %index.next.3, 1
  %25 = getelementptr inbounds double, double* %b, i64 %24
  %26 = bitcast double* %25 to <8 x double>*
  %wide.vec.4 = load <8 x double>, <8 x double>* %26, align 8
  %strided.vec.4 = shufflevector <8 x double> %wide.vec.4, <8 x double> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %27 = fadd <4 x double> %strided.vec.4, <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>
  %28 = getelementptr inbounds double, double* %a, i64 %index.next.3
  %29 = bitcast double* %28 to <4 x double>*
  store <4 x double> %27, <4 x double>* %29, align 8
  %index.next.4 = or i64 %index, 20
  %30 = shl i64 %index.next.4, 1
  %31 = getelementptr inbounds double, double* %b, i64 %30
  %32 = bitcast double* %31 to <8 x double>*
  %wide.vec.5 = load <8 x double>, <8 x double>* %32, align 8
  %strided.vec.5 = shufflevector <8 x double> %wide.vec.5, <8 x double> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %33 = fadd <4 x double> %strided.vec.5, <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>
  %34 = getelementptr inbounds double, double* %a, i64 %index.next.4
  %35 = bitcast double* %34 to <4 x double>*
  store <4 x double> %33, <4 x double>* %35, align 8
  %index.next.5 = or i64 %index, 24
  %36 = shl i64 %index.next.5, 1
  %37 = getelementptr inbounds double, double* %b, i64 %36
  %38 = bitcast double* %37 to <8 x double>*
  %wide.vec.6 = load <8 x double>, <8 x double>* %38, align 8
  %strided.vec.6 = shufflevector <8 x double> %wide.vec.6, <8 x double> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %39 = fadd <4 x double> %strided.vec.6, <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>
  %40 = getelementptr inbounds double, double* %a, i64 %index.next.5
  %41 = bitcast double* %40 to <4 x double>*
  store <4 x double> %39, <4 x double>* %41, align 8
  %index.next.6 = or i64 %index, 28
  %42 = shl i64 %index.next.6, 1
  %43 = getelementptr inbounds double, double* %b, i64 %42
  %44 = bitcast double* %43 to <8 x double>*
  %wide.vec.7 = load <8 x double>, <8 x double>* %44, align 8
  %strided.vec.7 = shufflevector <8 x double> %wide.vec.7, <8 x double> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %45 = fadd <4 x double> %strided.vec.7, <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>
  %46 = getelementptr inbounds double, double* %a, i64 %index.next.6
  %47 = bitcast double* %46 to <4 x double>*
  store <4 x double> %45, <4 x double>* %47, align 8
  %index.next.7 = or i64 %index, 32
  %48 = shl i64 %index.next.7, 1
  %49 = getelementptr inbounds double, double* %b, i64 %48
  %50 = bitcast double* %49 to <8 x double>*
  %wide.vec.8 = load <8 x double>, <8 x double>* %50, align 8
  %strided.vec.8 = shufflevector <8 x double> %wide.vec.8, <8 x double> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %51 = fadd <4 x double> %strided.vec.8, <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>
  %52 = getelementptr inbounds double, double* %a, i64 %index.next.7
  %53 = bitcast double* %52 to <4 x double>*
  store <4 x double> %51, <4 x double>* %53, align 8
  %index.next.8 = or i64 %index, 36
  %54 = shl i64 %index.next.8, 1
  %55 = getelementptr inbounds double, double* %b, i64 %54
  %56 = bitcast double* %55 to <8 x double>*
  %wide.vec.9 = load <8 x double>, <8 x double>* %56, align 8
  %strided.vec.9 = shufflevector <8 x double> %wide.vec.9, <8 x double> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %57 = fadd <4 x double> %strided.vec.9, <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>
  %58 = getelementptr inbounds double, double* %a, i64 %index.next.8
  %59 = bitcast double* %58 to <4 x double>*
  store <4 x double> %57, <4 x double>* %59, align 8
  %index.next.9 = or i64 %index, 40
  %60 = shl i64 %index.next.9, 1
  %61 = getelementptr inbounds double, double* %b, i64 %60
  %62 = bitcast double* %61 to <8 x double>*
  %wide.vec.10 = load <8 x double>, <8 x double>* %62, align 8
  %strided.vec.10 = shufflevector <8 x double> %wide.vec.10, <8 x double> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %63 = fadd <4 x double> %strided.vec.10, <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>
  %64 = getelementptr inbounds double, double* %a, i64 %index.next.9
  %65 = bitcast double* %64 to <4 x double>*
  store <4 x double> %63, <4 x double>* %65, align 8
  %index.next.10 = or i64 %index, 44
  %66 = shl i64 %index.next.10, 1
  %67 = getelementptr inbounds double, double* %b, i64 %66
  %68 = bitcast double* %67 to <8 x double>*
  %wide.vec.11 = load <8 x double>, <8 x double>* %68, align 8
  %strided.vec.11 = shufflevector <8 x double> %wide.vec.11, <8 x double> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %69 = fadd <4 x double> %strided.vec.11, <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>
  %70 = getelementptr inbounds double, double* %a, i64 %index.next.10
  %71 = bitcast double* %70 to <4 x double>*
  store <4 x double> %69, <4 x double>* %71, align 8
  %index.next.11 = or i64 %index, 48
  %72 = shl i64 %index.next.11, 1
  %73 = getelementptr inbounds double, double* %b, i64 %72
  %74 = bitcast double* %73 to <8 x double>*
  %wide.vec.12 = load <8 x double>, <8 x double>* %74, align 8
  %strided.vec.12 = shufflevector <8 x double> %wide.vec.12, <8 x double> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %75 = fadd <4 x double> %strided.vec.12, <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>
  %76 = getelementptr inbounds double, double* %a, i64 %index.next.11
  %77 = bitcast double* %76 to <4 x double>*
  store <4 x double> %75, <4 x double>* %77, align 8
  %index.next.12 = or i64 %index, 52
  %78 = shl i64 %index.next.12, 1
  %79 = getelementptr inbounds double, double* %b, i64 %78
  %80 = bitcast double* %79 to <8 x double>*
  %wide.vec.13 = load <8 x double>, <8 x double>* %80, align 8
  %strided.vec.13 = shufflevector <8 x double> %wide.vec.13, <8 x double> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %81 = fadd <4 x double> %strided.vec.13, <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>
  %82 = getelementptr inbounds double, double* %a, i64 %index.next.12
  %83 = bitcast double* %82 to <4 x double>*
  store <4 x double> %81, <4 x double>* %83, align 8
  %index.next.13 = or i64 %index, 56
  %84 = shl i64 %index.next.13, 1
  %85 = getelementptr inbounds double, double* %b, i64 %84
  %86 = bitcast double* %85 to <8 x double>*
  %wide.vec.14 = load <8 x double>, <8 x double>* %86, align 8
  %strided.vec.14 = shufflevector <8 x double> %wide.vec.14, <8 x double> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %87 = fadd <4 x double> %strided.vec.14, <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>
  %88 = getelementptr inbounds double, double* %a, i64 %index.next.13
  %89 = bitcast double* %88 to <4 x double>*
  store <4 x double> %87, <4 x double>* %89, align 8
  %index.next.14 = or i64 %index, 60
  %90 = shl i64 %index.next.14, 1
  %91 = getelementptr inbounds double, double* %b, i64 %90
  %92 = bitcast double* %91 to <8 x double>*
  %wide.vec.15 = load <8 x double>, <8 x double>* %92, align 8
  %strided.vec.15 = shufflevector <8 x double> %wide.vec.15, <8 x double> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %93 = fadd <4 x double> %strided.vec.15, <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>
  %94 = getelementptr inbounds double, double* %a, i64 %index.next.14
  %95 = bitcast double* %94 to <4 x double>*
  store <4 x double> %93, <4 x double>* %95, align 8
  %index.next.15 = add nsw i64 %index, 64
  %96 = icmp eq i64 %index.next.15, 1600
  br i1 %96, label %for.cond.cleanup, label %vector.body

for.cond.cleanup:                                 ; preds = %vector.body
  ret void
}

attributes #0 = { nounwind "target-cpu"="a2q" }

