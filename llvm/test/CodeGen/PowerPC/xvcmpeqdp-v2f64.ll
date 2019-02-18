; RUN: llc < %s -verify-machineinstrs | FileCheck %s
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

; Function Attrs: nounwind
define void @__fmax_double3_3D_exec(<3 x double> %input1, <3 x i64> %input2, 
                                    <3 x i1> %input3, <3 x i64> %input4,
                                    <3 x i64> %input5,  <4 x double>* %input6) #0 {
entry:
  br i1 undef, label %if.then.i, label %fmax_double3.exit

if.then.i:                                        ; preds = %entry
  %cmp24.i.i = fcmp ord <3 x double> %input1, zeroinitializer
  %sext25.i.i = sext <3 x i1> %cmp24.i.i to <3 x i64>
  %neg.i.i = xor <3 x i64> %sext25.i.i, <i64 -1, i64 -1, i64 -1>
  %or.i.i = or <3 x i64> %input2, %neg.i.i
  %neg.i.i.i = select <3 x i1> %input3, <3 x i64> zeroinitializer, <3 x i64> %sext25.i.i
  %and.i.i.i = and <3 x i64> %input4, %neg.i.i.i
  %and26.i.i.i = and <3 x i64> %input5, %or.i.i
  %or.i.i.i = or <3 x i64> %and.i.i.i, %and26.i.i.i
  %astype32.i.i.i = bitcast <3 x i64> %or.i.i.i to <3 x double>
  %extractVec33.i.i.i = shufflevector <3 x double> %astype32.i.i.i, <3 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 undef>
  store <4 x double> %extractVec33.i.i.i, <4 x double>* %input6, align 32
  br label %fmax_double3.exit

; CHECK-LABEL: @__fmax_double3_3D_exec
; CHECK: xvcmpeqdp

fmax_double3.exit:                                ; preds = %if.then.i, %entry
  br i1 undef, label %if.then, label %do.end

if.then:                                          ; preds = %fmax_double3.exit
  unreachable

do.end:                                           ; preds = %fmax_double3.exit
  ret void
}

attributes #0 = { nounwind }

