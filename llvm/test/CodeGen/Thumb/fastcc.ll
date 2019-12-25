; RUN: llc < %s -mcpu=arm926ej-s -mattr=+vfp2

; This is a regression test, to ensure that fastcc functions are correctly
; handled when compiling for a processor which has a floating-point unit which
; is not accessible from the selected instruction set.

target datalayout = "e-m:e-p:32:32-i1:8:32-i8:8:32-i16:16:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv5e-none-linux-gnueabi"

; Function Attrs: optsize
define fastcc void @_foo(float %walpha) #0 {
entry:
  br label %for.body13

for.body13:                                       ; preds = %for.body13, %entry
  br i1 undef, label %for.end182.critedge, label %for.body13

for.end182.critedge:                              ; preds = %for.body13
  %conv183 = fpext float %walpha to double
  %mul184 = fmul double %conv183, 8.200000e-01
  %conv185 = fptrunc double %mul184 to float
  %conv188 = fpext float %conv185 to double
  %mul189 = fmul double %conv188, 6.000000e-01
  %conv190 = fptrunc double %mul189 to float
  br label %for.body193

for.body193:                                      ; preds = %for.body193, %for.end182.critedge
  %mul195 = fmul float %conv190, undef
  br label %for.body193
}

attributes #0 = { optsize "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.5.0 "}
