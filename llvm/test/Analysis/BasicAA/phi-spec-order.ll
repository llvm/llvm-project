target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64le-unknown-linux"
; RUN: opt < %s -basic-aa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

@X = external global [16000 x double], align 32
@Y = external global [16000 x double], align 32

; CHECK: NoAlias: [16000 x double]* %lsr.iv1, [16000 x double]* %lsr.iv4
; CHECK: NoAlias: <4 x double>* %scevgep11, <4 x double>* %scevgep7
; CHECK: NoAlias: <4 x double>* %scevgep10, <4 x double>* %scevgep7
; CHECK: NoAlias: <4 x double>* %scevgep7, <4 x double>* %scevgep9
; CHECK: NoAlias: <4 x double>* %scevgep11, <4 x double>* %scevgep3
; CHECK: NoAlias: <4 x double>* %scevgep10, <4 x double>* %scevgep3
; CHECK: NoAlias: <4 x double>* %scevgep3, <4 x double>* %scevgep9
; CHECK: NoAlias: double* %scevgep, double* %scevgep5
define signext i32 @s000() nounwind {
entry:
  br label %for.cond2.preheader

for.cond2.preheader:                              ; preds = %for.end, %entry
  %nl.018 = phi i32 [ 0, %entry ], [ %inc9, %for.end ]
  br label %for.body4

for.body4:                                        ; preds = %for.body4, %for.cond2.preheader
  %lsr.iv4 = phi [16000 x double]* [ %i11, %for.body4 ], [ bitcast (double* getelementptr inbounds ([16000 x double], [16000 x double]* @Y, i64 0, i64 8)
 to [16000 x double]*), %for.cond2.preheader ]
  %lsr.iv1 = phi [16000 x double]* [ %i10, %for.body4 ], [ @X, %for.cond2.preheader ]

  %lsr.iv = phi i32 [ %lsr.iv.next, %for.body4 ], [ 16000, %for.cond2.preheader ]
  %lsr.iv46 = bitcast [16000 x double]* %lsr.iv4 to <4 x double>*
  %lsr.iv12 = bitcast [16000 x double]* %lsr.iv1 to <4 x double>*
  %scevgep11 = getelementptr <4 x double>, <4 x double>* %lsr.iv46, i64 -2
  %i6 = load <4 x double>, <4 x double>* %scevgep11, align 32
  %add = fadd <4 x double> %i6, <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>
  store <4 x double> %add, <4 x double>* %lsr.iv12, align 32
  %scevgep10 = getelementptr <4 x double>, <4 x double>* %lsr.iv46, i64 -1
  %i7 = load <4 x double>, <4 x double>* %scevgep10, align 32
  %add.4 = fadd <4 x double> %i7, <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>
  %scevgep9 = getelementptr <4 x double>, <4 x double>* %lsr.iv12, i64 1
  store <4 x double> %add.4, <4 x double>* %scevgep9, align 32
  %i8 = load <4 x double>, <4 x double>* %lsr.iv46, align 32
  %add.8 = fadd <4 x double> %i8, <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>
  %scevgep8 = getelementptr <4 x double>, <4 x double>* %lsr.iv12, i64 2
  store <4 x double> %add.8, <4 x double>* %scevgep8, align 32
  %scevgep7 = getelementptr <4 x double>, <4 x double>* %lsr.iv46, i64 1
  %i9 = load <4 x double>, <4 x double>* %scevgep7, align 32
  %add.12 = fadd <4 x double> %i9, <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>
  %scevgep3 = getelementptr <4 x double>, <4 x double>* %lsr.iv12, i64 3
  store <4 x double> %add.12, <4 x double>* %scevgep3, align 32

  %lsr.iv.next = add i32 %lsr.iv, -16
  %scevgep = getelementptr [16000 x double], [16000 x double]* %lsr.iv1, i64 0, i64 16
  %i10 = bitcast double* %scevgep to [16000 x double]*
  %scevgep5 = getelementptr [16000 x double], [16000 x double]* %lsr.iv4, i64 0, i64 16
  %i11 = bitcast double* %scevgep5 to [16000 x double]*
  %exitcond.15 = icmp eq i32 %lsr.iv.next, 0
  br i1 %exitcond.15, label %for.end, label %for.body4

for.end:                                          ; preds = %for.body4
  %inc9 = add nsw i32 %nl.018, 1
  %exitcond = icmp eq i32 %inc9, 400000
  br i1 %exitcond, label %for.end10, label %for.cond2.preheader

for.end10:                                        ; preds = %for.end
  ret i32 0
}
