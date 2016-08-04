; RUN: llc -verify-machineinstrs < %s -mcpu=ppc | FileCheck %s

target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32"
target triple = "powerpc-unknown-linux-gnu"

define i64 @foo(double* nocapture %n) nounwind readonly {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.06 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %x.05 = phi i64 [ 0, %entry ], [ %conv1, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %n, i32 %i.06
  %0 = load double, double* %arrayidx, align 8
  %conv = sitofp i64 %x.05 to double
  %add = fadd double %conv, %0
  %conv1 = fptosi double %add to i64
  %inc = add nsw i32 %i.06, 1
  %exitcond = icmp eq i32 %inc, 2048
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i64 %conv1
}

; CHECK: @foo
; CHECK-NOT: mtctr

@init_value = global double 1.000000e+00, align 8
@data64 = global [8000 x i64] zeroinitializer, align 8

define i32 @main(i32 %argc, i8** nocapture %argv) {
entry:
  %0 = load double, double* @init_value, align 8
  %conv = fptosi double %0 to i64
  %broadcast.splatinsert.i = insertelement <2 x i64> undef, i64 %conv, i32 0
  %broadcast.splat.i = shufflevector <2 x i64> %broadcast.splatinsert.i, <2 x i64> undef, <2 x i32> zeroinitializer
  br label %vector.body.i

vector.body.i:                                    ; preds = %vector.body.i, %entry
  %index.i = phi i32 [ 0, %entry ], [ %index.next.i, %vector.body.i ]
  %next.gep.i = getelementptr [8000 x i64], [8000 x i64]* @data64, i32 0, i32 %index.i
  %1 = bitcast i64* %next.gep.i to <2 x i64>*
  store <2 x i64> %broadcast.splat.i, <2 x i64>* %1, align 8
  %next.gep.sum24.i = or i32 %index.i, 2
  %2 = getelementptr [8000 x i64], [8000 x i64]* @data64, i32 0, i32 %next.gep.sum24.i
  %3 = bitcast i64* %2 to <2 x i64>*
  store <2 x i64> %broadcast.splat.i, <2 x i64>* %3, align 8
  %index.next.i = add i32 %index.i, 4
  %4 = icmp eq i32 %index.next.i, 8000
  br i1 %4, label %_Z4fillIPxxEvT_S1_T0_.exit, label %vector.body.i

_Z4fillIPxxEvT_S1_T0_.exit:                       ; preds = %vector.body.i
  ret i32 0
}

; CHECK: @main
; CHECK: __fixdfdi
; CHECK: mtctr

