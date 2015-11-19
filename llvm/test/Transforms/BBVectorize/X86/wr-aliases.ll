; RUN: opt -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7-avx -disable-basicaa -bb-vectorize -S < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%class.QBezier.15 = type { double, double, double, double, double, double, double, double }

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1) #0

; Function Attrs: uwtable
declare fastcc void @_ZL12printQBezier7QBezier(%class.QBezier.15* byval nocapture readonly align 8) #1

; Function Attrs: nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture) #0

; Function Attrs: nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) #0

define void @main_arrayctor.cont([10 x %class.QBezier.15]* %beziers, %class.QBezier.15* %agg.tmp.i, %class.QBezier.15* %agg.tmp55.i, %class.QBezier.15* %agg.tmp56.i) {
newFuncRoot:
  br label %arrayctor.cont

arrayctor.cont.ret.exitStub:                      ; preds = %arrayctor.cont
  ret void

; CHECK-LABEL: @main_arrayctor.cont
; CHECK: <2 x double>
; CHECK: @_ZL12printQBezier7QBezier
; CHECK: store double %mul8.i, double* %x3.i, align 16
; CHECK: load double, double* %x3.i, align 16
; CHECK: ret

arrayctor.cont:                                   ; preds = %newFuncRoot
  %ref.tmp.sroa.0.0.idx = getelementptr inbounds [10 x %class.QBezier.15], [10 x %class.QBezier.15]* %beziers, i64 0, i64 0, i32 0
  store double 1.000000e+01, double* %ref.tmp.sroa.0.0.idx, align 16
  %ref.tmp.sroa.2.0.idx1 = getelementptr inbounds [10 x %class.QBezier.15], [10 x %class.QBezier.15]* %beziers, i64 0, i64 0, i32 1
  store double 2.000000e+01, double* %ref.tmp.sroa.2.0.idx1, align 8
  %ref.tmp.sroa.3.0.idx2 = getelementptr inbounds [10 x %class.QBezier.15], [10 x %class.QBezier.15]* %beziers, i64 0, i64 0, i32 2
  store double 3.000000e+01, double* %ref.tmp.sroa.3.0.idx2, align 16
  %ref.tmp.sroa.4.0.idx3 = getelementptr inbounds [10 x %class.QBezier.15], [10 x %class.QBezier.15]* %beziers, i64 0, i64 0, i32 3
  store double 4.000000e+01, double* %ref.tmp.sroa.4.0.idx3, align 8
  %ref.tmp.sroa.5.0.idx4 = getelementptr inbounds [10 x %class.QBezier.15], [10 x %class.QBezier.15]* %beziers, i64 0, i64 0, i32 4
  store double 5.000000e+01, double* %ref.tmp.sroa.5.0.idx4, align 16
  %ref.tmp.sroa.6.0.idx5 = getelementptr inbounds [10 x %class.QBezier.15], [10 x %class.QBezier.15]* %beziers, i64 0, i64 0, i32 5
  store double 6.000000e+01, double* %ref.tmp.sroa.6.0.idx5, align 8
  %ref.tmp.sroa.7.0.idx6 = getelementptr inbounds [10 x %class.QBezier.15], [10 x %class.QBezier.15]* %beziers, i64 0, i64 0, i32 6
  store double 7.000000e+01, double* %ref.tmp.sroa.7.0.idx6, align 16
  %ref.tmp.sroa.8.0.idx7 = getelementptr inbounds [10 x %class.QBezier.15], [10 x %class.QBezier.15]* %beziers, i64 0, i64 0, i32 7
  store double 8.000000e+01, double* %ref.tmp.sroa.8.0.idx7, align 8
  %add.ptr = getelementptr inbounds [10 x %class.QBezier.15], [10 x %class.QBezier.15]* %beziers, i64 0, i64 1
  %v0 = bitcast %class.QBezier.15* %agg.tmp.i to i8*
  call void @llvm.lifetime.start(i64 64, i8* %v0)
  %v1 = bitcast %class.QBezier.15* %agg.tmp55.i to i8*
  call void @llvm.lifetime.start(i64 64, i8* %v1)
  %v2 = bitcast %class.QBezier.15* %agg.tmp56.i to i8*
  call void @llvm.lifetime.start(i64 64, i8* %v2)
  %v3 = bitcast [10 x %class.QBezier.15]* %beziers to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %v0, i8* %v3, i64 64, i1 false)
  call fastcc void @_ZL12printQBezier7QBezier(%class.QBezier.15* byval align 8 %agg.tmp.i)
  %x2.i = getelementptr inbounds [10 x %class.QBezier.15], [10 x %class.QBezier.15]* %beziers, i64 0, i64 0, i32 2
  %v4 = load double, double* %x2.i, align 16
  %x3.i = getelementptr inbounds [10 x %class.QBezier.15], [10 x %class.QBezier.15]* %beziers, i64 0, i64 0, i32 4
  %v5 = load double, double* %x3.i, align 16
  %add.i = fadd double %v4, %v5
  %mul.i = fmul double 5.000000e-01, %add.i
  %x1.i = getelementptr inbounds [10 x %class.QBezier.15], [10 x %class.QBezier.15]* %beziers, i64 0, i64 0, i32 0
  %v6 = load double, double* %x1.i, align 16
  %add3.i = fadd double %v4, %v6
  %mul4.i = fmul double 5.000000e-01, %add3.i
  %x25.i = getelementptr inbounds [10 x %class.QBezier.15], [10 x %class.QBezier.15]* %beziers, i64 0, i64 1, i32 2
  store double %mul4.i, double* %x25.i, align 16
  %v7 = load double, double* %x3.i, align 16
  %x4.i = getelementptr inbounds [10 x %class.QBezier.15], [10 x %class.QBezier.15]* %beziers, i64 0, i64 0, i32 6
  %v8 = load double, double* %x4.i, align 16
  %add7.i = fadd double %v7, %v8
  %mul8.i = fmul double 5.000000e-01, %add7.i
  store double %mul8.i, double* %x3.i, align 16
  %v9 = load double, double* %x1.i, align 16
  %x111.i = getelementptr inbounds %class.QBezier.15, %class.QBezier.15* %add.ptr, i64 0, i32 0
  store double %v9, double* %x111.i, align 16
  %v10 = load double, double* %x25.i, align 16
  %add15.i = fadd double %mul.i, %v10
  %mul16.i = fmul double 5.000000e-01, %add15.i
  %x317.i = getelementptr inbounds [10 x %class.QBezier.15], [10 x %class.QBezier.15]* %beziers, i64 0, i64 1, i32 4
  store double %mul16.i, double* %x317.i, align 16
  %v11 = load double, double* %x3.i, align 16
  %add19.i = fadd double %mul.i, %v11
  %mul20.i = fmul double 5.000000e-01, %add19.i
  store double %mul20.i, double* %x2.i, align 16
  %v12 = load double, double* %x317.i, align 16
  %add24.i = fadd double %v12, %mul20.i
  %mul25.i = fmul double 5.000000e-01, %add24.i
  store double %mul25.i, double* %x1.i, align 16
  %x427.i = getelementptr inbounds [10 x %class.QBezier.15], [10 x %class.QBezier.15]* %beziers, i64 0, i64 1, i32 6
  store double %mul25.i, double* %x427.i, align 16
  %y2.i = getelementptr inbounds [10 x %class.QBezier.15], [10 x %class.QBezier.15]* %beziers, i64 0, i64 0, i32 3
  %v13 = load double, double* %y2.i, align 8
  %y3.i = getelementptr inbounds [10 x %class.QBezier.15], [10 x %class.QBezier.15]* %beziers, i64 0, i64 0, i32 5
  %v14 = load double, double* %y3.i, align 8
  %add28.i = fadd double %v13, %v14
  %div.i = fmul double 5.000000e-01, %add28.i
  %y1.i = getelementptr inbounds [10 x %class.QBezier.15], [10 x %class.QBezier.15]* %beziers, i64 0, i64 0, i32 1
  %v15 = load double, double* %y1.i, align 8
  %add30.i = fadd double %v13, %v15
  %mul31.i = fmul double 5.000000e-01, %add30.i
  %y232.i = getelementptr inbounds [10 x %class.QBezier.15], [10 x %class.QBezier.15]* %beziers, i64 0, i64 1, i32 3
  store double %mul31.i, double* %y232.i, align 8
  %v16 = load double, double* %y3.i, align 8
  %y4.i = getelementptr inbounds [10 x %class.QBezier.15], [10 x %class.QBezier.15]* %beziers, i64 0, i64 0, i32 7
  %v17 = load double, double* %y4.i, align 8
  %add34.i = fadd double %v16, %v17
  %mul35.i = fmul double 5.000000e-01, %add34.i
  store double %mul35.i, double* %y3.i, align 8
  %v18 = load double, double* %y1.i, align 8
  %y138.i = getelementptr inbounds [10 x %class.QBezier.15], [10 x %class.QBezier.15]* %beziers, i64 0, i64 1, i32 1
  store double %v18, double* %y138.i, align 8
  %v19 = load double, double* %y232.i, align 8
  %add42.i = fadd double %div.i, %v19
  %mul43.i = fmul double 5.000000e-01, %add42.i
  %y344.i = getelementptr inbounds [10 x %class.QBezier.15], [10 x %class.QBezier.15]* %beziers, i64 0, i64 1, i32 5
  store double %mul43.i, double* %y344.i, align 8
  %v20 = load double, double* %y3.i, align 8
  %add46.i = fadd double %div.i, %v20
  %mul47.i = fmul double 5.000000e-01, %add46.i
  store double %mul47.i, double* %y2.i, align 8
  %v21 = load double, double* %y344.i, align 8
  %add51.i = fadd double %v21, %mul47.i
  %mul52.i = fmul double 5.000000e-01, %add51.i
  store double %mul52.i, double* %y1.i, align 8
  %y454.i = getelementptr inbounds [10 x %class.QBezier.15], [10 x %class.QBezier.15]* %beziers, i64 0, i64 1, i32 7
  store double %mul52.i, double* %y454.i, align 8
  %v22 = bitcast %class.QBezier.15* %add.ptr to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %v1, i8* %v22, i64 64, i1 false)
  call fastcc void @_ZL12printQBezier7QBezier(%class.QBezier.15* byval align 8 %agg.tmp55.i)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %v2, i8* %v3, i64 64, i1 false)
  call fastcc void @_ZL12printQBezier7QBezier(%class.QBezier.15* byval align 8 %agg.tmp56.i)
  call void @llvm.lifetime.end(i64 64, i8* %v0)
  call void @llvm.lifetime.end(i64 64, i8* %v1)
  call void @llvm.lifetime.end(i64 64, i8* %v2)
  br label %arrayctor.cont.ret.exitStub
}

attributes #0 = { nounwind }
attributes #1 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
