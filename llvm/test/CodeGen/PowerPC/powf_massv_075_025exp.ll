; RUN: llc -vector-library=MASSV < %s -mtriple=powerpc64le-unknown-unknown -mcpu=pwr9 | FileCheck -check-prefixes=CHECK-PWR9 %s
; RUN: llc -vector-library=MASSV < %s -mtriple=powerpc64le-unknown-unknown -mcpu=pwr8 | FileCheck -check-prefixes=CHECK-PWR8 %s

; Exponent is a variable
define void @vspow_var(float* nocapture %z, float* nocapture readonly %y, float* nocapture readonly %x)  {
; CHECK-LABEL:       @vspow_var
; CHECK-PWR9:        bl __powf4_P9
; CHECK-PWR8:        bl __powf4_P8
; CHECK:             blr
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %next.gep = getelementptr float, float* %z, i64 %index
  %next.gep31 = getelementptr float, float* %y, i64 %index
  %next.gep32 = getelementptr float, float* %x, i64 %index
  %0 = bitcast float* %next.gep32 to <4 x float>*
  %wide.load = load <4 x float>, <4 x float>* %0, align 4
  %1 = bitcast float* %next.gep31 to <4 x float>*
  %wide.load33 = load <4 x float>, <4 x float>* %1, align 4
  %2 = call ninf afn nsz <4 x float> @__powf4_P8(<4 x float> %wide.load, <4 x float> %wide.load33)
  %3 = bitcast float* %next.gep to <4 x float>*
  store <4 x float> %2, <4 x float>* %3, align 4
  %index.next = add i64 %index, 4
  %4 = icmp eq i64 %index.next, 1024
  br i1 %4, label %for.end, label %vector.body

for.end:
  ret void
}

; Exponent is a constant != 0.75 and !=0.25
define void @vspow_const(float* nocapture %y, float* nocapture readonly %x)  {
; CHECK-LABEL:       @vspow_const
; CHECK-PWR9:        bl __powf4_P9
; CHECK-PWR8:        bl __powf4_P8
; CHECK:             blr
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %next.gep = getelementptr float, float* %y, i64 %index
  %next.gep19 = getelementptr float, float* %x, i64 %index
  %0 = bitcast float* %next.gep19 to <4 x float>*
  %wide.load = load <4 x float>, <4 x float>* %0, align 4
  %1 = call ninf afn nsz <4 x float> @__powf4_P8(<4 x float> %wide.load, <4 x float> <float 0x3FE851EB80000000, float 0x3FE851EB80000000, float 0x3FE851EB80000000, float 0x3FE851EB80000000>)
  %2 = bitcast float* %next.gep to <4 x float>*
  store <4 x float> %1, <4 x float>* %2, align 4
  %index.next = add i64 %index, 4
  %3 = icmp eq i64 %index.next, 1024
  br i1 %3, label %for.end, label %vector.body

for.end:
  ret void
}

; Exponent is a constant != 0.75 and !=0.25 and they are different 
define void @vspow_neq_const(float* nocapture %y, float* nocapture readonly %x)  {
; CHECK-LABEL:       @vspow_neq_const
; CHECK-PWR9:        bl __powf4_P9
; CHECK-PWR8:        bl __powf4_P8
; CHECK:             blr
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %next.gep = getelementptr float, float* %y, i64 %index
  %next.gep19 = getelementptr float, float* %x, i64 %index
  %0 = bitcast float* %next.gep19 to <4 x float>*
  %wide.load = load <4 x float>, <4 x float>* %0, align 4
  %1 = call ninf afn nsz <4 x float> @__powf4_P8(<4 x float> %wide.load, <4 x float> <float 0x3FE861EB80000000, float 0x3FE871EB80000000, float 0x3FE851EB80000000, float 0x3FE851EB80000000>)
  %2 = bitcast float* %next.gep to <4 x float>*
  store <4 x float> %1, <4 x float>* %2, align 4
  %index.next = add i64 %index, 4
  %3 = icmp eq i64 %index.next, 1024
  br i1 %3, label %for.end, label %vector.body

for.end:
  ret void
}

; Exponent is a constant != 0.75 and !=0.25
define void @vspow_neq075_const(float* nocapture %y, float* nocapture readonly %x)  {
; CHECK-LABEL:       @vspow_neq075_const
; CHECK-PWR9:        bl __powf4_P9
; CHECK-PWR8:        bl __powf4_P8
; CHECK:             blr
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %next.gep = getelementptr float, float* %y, i64 %index
  %next.gep19 = getelementptr float, float* %x, i64 %index
  %0 = bitcast float* %next.gep19 to <4 x float>*
  %wide.load = load <4 x float>, <4 x float>* %0, align 4
  %1 = call ninf afn nsz <4 x float> @__powf4_P8(<4 x float> %wide.load, <4 x float> <float 7.500000e-01, float 7.500000e-01, float 7.500000e-01, float 0x3FE851EB80000000>)
  %2 = bitcast float* %next.gep to <4 x float>*
  store <4 x float> %1, <4 x float>* %2, align 4
  %index.next = add i64 %index, 4
  %3 = icmp eq i64 %index.next, 1024
  br i1 %3, label %for.end, label %vector.body

for.end:
  ret void
}

; Exponent is a constant != 0.75 and !=0.25
define void @vspow_neq025_const(float* nocapture %y, float* nocapture readonly %x)  {
; CHECK-LABEL:       @vspow_neq025_const
; CHECK-PWR9:        bl __powf4_P9
; CHECK-PWR8:        bl __powf4_P8
; CHECK:             blr
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %next.gep = getelementptr float, float* %y, i64 %index
  %next.gep19 = getelementptr float, float* %x, i64 %index
  %0 = bitcast float* %next.gep19 to <4 x float>*
  %wide.load = load <4 x float>, <4 x float>* %0, align 4
  %1 = call ninf afn nsz <4 x float> @__powf4_P8(<4 x float> %wide.load, <4 x float> <float 0x3FE851EB80000000, float 2.500000e-01, float 0x3FE851EB80000000, float 2.500000e-01>)
  %2 = bitcast float* %next.gep to <4 x float>*
  store <4 x float> %1, <4 x float>* %2, align 4
  %index.next = add i64 %index, 4
  %3 = icmp eq i64 %index.next, 1024
  br i1 %3, label %for.end, label %vector.body

for.end:
  ret void
}

; Exponent is 0.75
define void @vspow_075(float* nocapture %y, float* nocapture readonly %x)  {
; CHECK-LABEL:       @vspow_075
; CHECK-NOT:         bl __powf4_P{{[8,9]}}
; CHECK:             xvrsqrtesp
; CHECK:             blr
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %next.gep = getelementptr float, float* %y, i64 %index
  %next.gep19 = getelementptr float, float* %x, i64 %index
  %0 = bitcast float* %next.gep19 to <4 x float>*
  %wide.load = load <4 x float>, <4 x float>* %0, align 4
  %1 = call ninf afn <4 x float> @__powf4_P8(<4 x float> %wide.load, <4 x float> <float 7.500000e-01, float 7.500000e-01, float 7.500000e-01, float 7.500000e-01>)
  %2 = bitcast float* %next.gep to <4 x float>*
  store <4 x float> %1, <4 x float>* %2, align 4
  %index.next = add i64 %index, 4
  %3 = icmp eq i64 %index.next, 1024
  br i1 %3, label %for.end, label %vector.body

for.end:
  ret void
}

; Exponent is 0.25
define void @vspow_025(float* nocapture %y, float* nocapture readonly %x)  {
; CHECK-LABEL:       @vspow_025
; CHECK-NOT:         bl __powf4_P{{[8,9]}}
; CHECK:             xvrsqrtesp
; CHECK:             blr
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %next.gep = getelementptr float, float* %y, i64 %index
  %next.gep19 = getelementptr float, float* %x, i64 %index
  %0 = bitcast float* %next.gep19 to <4 x float>*
  %wide.load = load <4 x float>, <4 x float>* %0, align 4
  %1 = call ninf afn nsz <4 x float> @__powf4_P8(<4 x float> %wide.load, <4 x float> <float 2.500000e-01, float 2.500000e-01, float 2.500000e-01, float 2.500000e-01>)
  %2 = bitcast float* %next.gep to <4 x float>*
  store <4 x float> %1, <4 x float>* %2, align 4
  %index.next = add i64 %index, 4
  %3 = icmp eq i64 %index.next, 1024
  br i1 %3, label %for.end, label %vector.body

for.end:
  ret void
}

; Exponent is 0.75 but no proper fast-math flags
define void @vspow_075_nofast(float* nocapture %y, float* nocapture readonly %x)  {
; CHECK-LABEL:       @vspow_075_nofast
; CHECK-PWR9:        bl __powf4_P9
; CHECK-PWR8:        bl __powf4_P8
; CHECK-NOT:         xvrsqrtesp
; CHECK:             blr
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %next.gep = getelementptr float, float* %y, i64 %index
  %next.gep19 = getelementptr float, float* %x, i64 %index
  %0 = bitcast float* %next.gep19 to <4 x float>*
  %wide.load = load <4 x float>, <4 x float>* %0, align 4
  %1 = call <4 x float> @__powf4_P8(<4 x float> %wide.load, <4 x float> <float 7.500000e-01, float 7.500000e-01, float 7.500000e-01, float 7.500000e-01>)
  %2 = bitcast float* %next.gep to <4 x float>*
  store <4 x float> %1, <4 x float>* %2, align 4
  %index.next = add i64 %index, 4
  %3 = icmp eq i64 %index.next, 1024
  br i1 %3, label %for.end, label %vector.body

for.end:
  ret void
}

; Exponent is 0.25 but no proper fast-math flags
define void @vspow_025_nofast(float* nocapture %y, float* nocapture readonly %x)  {
; CHECK-LABEL:       @vspow_025_nofast
; CHECK-PWR9:        bl __powf4_P9
; CHECK-PWR8:        bl __powf4_P8
; CHECK-NOT:         xvrsqrtesp
; CHECK:             blr
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %next.gep = getelementptr float, float* %y, i64 %index
  %next.gep19 = getelementptr float, float* %x, i64 %index
  %0 = bitcast float* %next.gep19 to <4 x float>*
  %wide.load = load <4 x float>, <4 x float>* %0, align 4
  %1 = call <4 x float> @__powf4_P8(<4 x float> %wide.load, <4 x float> <float 2.500000e-01, float 2.500000e-01, float 2.500000e-01, float 2.500000e-01>)
  %2 = bitcast float* %next.gep to <4 x float>*
  store <4 x float> %1, <4 x float>* %2, align 4
  %index.next = add i64 %index, 4
  %3 = icmp eq i64 %index.next, 1024
  br i1 %3, label %for.end, label %vector.body

for.end:
  ret void
}

; Function Attrs: nounwind readnone speculatable willreturn
declare <4 x float> @__powf4_P8(<4 x float>, <4 x float>)
