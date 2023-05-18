; RUN: llc -vector-library=MASSV < %s -mtriple=powerpc64le-unknown-unknown -mcpu=pwr10 | FileCheck -check-prefixes=CHECK-PWR9 %s
; RUN: llc -vector-library=MASSV < %s -mtriple=powerpc64le-unknown-unknown -mcpu=pwr9 | FileCheck -check-prefixes=CHECK-PWR9 %s
; RUN: llc -vector-library=MASSV < %s -mtriple=powerpc64le-unknown-unknown -mcpu=pwr8 | FileCheck -check-prefixes=CHECK-PWR8 %s
; RUN: llc -vector-library=MASSV < %s -mtriple=powerpc-ibm-aix-xcoff -mcpu=pwr10 | FileCheck -check-prefixes=CHECK-PWR10 %s
; RUN: llc -vector-library=MASSV < %s -mtriple=powerpc-ibm-aix-xcoff -mcpu=pwr9 | FileCheck -check-prefixes=CHECK-PWR9 %s
; RUN: llc -vector-library=MASSV < %s -mtriple=powerpc-ibm-aix-xcoff -mcpu=pwr8 | FileCheck -check-prefixes=CHECK-PWR8 %s
; RUN: llc -vector-library=MASSV < %s -mtriple=powerpc-ibm-aix-xcoff -mcpu=pwr7 | FileCheck -check-prefixes=CHECK-PWR7 %s

; Exponent is a variable
define void @vspow_var(ptr nocapture %z, ptr nocapture readonly %y, ptr nocapture readonly %x)  {
; CHECK-LABEL:       @vspow_var
; CHECK-PWR10:       __powf4_P10
; CHECK-PWR9:        __powf4_P9
; CHECK-PWR8:        __powf4_P8
; CHECK-PWR7:        __powf4_P7
; CHECK:             blr
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %next.gep = getelementptr float, ptr %z, i64 %index
  %next.gep31 = getelementptr float, ptr %y, i64 %index
  %next.gep32 = getelementptr float, ptr %x, i64 %index
  %wide.load = load <4 x float>, ptr %next.gep32, align 4
  %wide.load33 = load <4 x float>, ptr %next.gep31, align 4
  %0 = call ninf afn nsz <4 x float> @__powf4(<4 x float> %wide.load, <4 x float> %wide.load33)
  store <4 x float> %0, ptr %next.gep, align 4
  %index.next = add i64 %index, 4
  %1 = icmp eq i64 %index.next, 1024
  br i1 %1, label %for.end, label %vector.body

for.end:
  ret void
}

; Exponent is a constant != 0.75 and !=0.25
define void @vspow_const(ptr nocapture %y, ptr nocapture readonly %x)  {
; CHECK-LABEL:       @vspow_const
; CHECK-PWR10:       __powf4_P10
; CHECK-PWR9:        __powf4_P9
; CHECK-PWR8:        __powf4_P8
; CHECK-PWR7:        __powf4_P7
; CHECK:             blr
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %next.gep = getelementptr float, ptr %y, i64 %index
  %next.gep19 = getelementptr float, ptr %x, i64 %index
  %wide.load = load <4 x float>, ptr %next.gep19, align 4
  %0 = call ninf afn nsz <4 x float> @__powf4(<4 x float> %wide.load, <4 x float> <float 0x3FE851EB80000000, float 0x3FE851EB80000000, float 0x3FE851EB80000000, float 0x3FE851EB80000000>)
  store <4 x float> %0, ptr %next.gep, align 4
  %index.next = add i64 %index, 4
  %1 = icmp eq i64 %index.next, 1024
  br i1 %1, label %for.end, label %vector.body

for.end:
  ret void
}

; Exponent is a constant != 0.75 and !=0.25 and they are different 
define void @vspow_neq_const(ptr nocapture %y, ptr nocapture readonly %x)  {
; CHECK-LABEL:       @vspow_neq_const
; CHECK-PWR10:       __powf4_P10
; CHECK-PWR9:        __powf4_P9
; CHECK-PWR8:        __powf4_P8
; CHECK-PWR7:        __powf4_P7
; CHECK:             blr
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %next.gep = getelementptr float, ptr %y, i64 %index
  %next.gep19 = getelementptr float, ptr %x, i64 %index
  %wide.load = load <4 x float>, ptr %next.gep19, align 4
  %0 = call ninf afn nsz <4 x float> @__powf4(<4 x float> %wide.load, <4 x float> <float 0x3FE861EB80000000, float 0x3FE871EB80000000, float 0x3FE851EB80000000, float 0x3FE851EB80000000>)
  store <4 x float> %0, ptr %next.gep, align 4
  %index.next = add i64 %index, 4
  %1 = icmp eq i64 %index.next, 1024
  br i1 %1, label %for.end, label %vector.body

for.end:
  ret void
}

; Exponent is a constant != 0.75 and !=0.25
define void @vspow_neq075_const(ptr nocapture %y, ptr nocapture readonly %x)  {
; CHECK-LABEL:       @vspow_neq075_const
; CHECK-PWR10:       __powf4_P10
; CHECK-PWR9:        __powf4_P9
; CHECK-PWR8:        __powf4_P8
; CHECK-PWR7:        __powf4_P7
; CHECK:             blr
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %next.gep = getelementptr float, ptr %y, i64 %index
  %next.gep19 = getelementptr float, ptr %x, i64 %index
  %wide.load = load <4 x float>, ptr %next.gep19, align 4
  %0 = call ninf afn nsz <4 x float> @__powf4(<4 x float> %wide.load, <4 x float> <float 7.500000e-01, float 7.500000e-01, float 7.500000e-01, float 0x3FE851EB80000000>)
  store <4 x float> %0, ptr %next.gep, align 4
  %index.next = add i64 %index, 4
  %1 = icmp eq i64 %index.next, 1024
  br i1 %1, label %for.end, label %vector.body

for.end:
  ret void
}

; Exponent is a constant != 0.75 and !=0.25
define void @vspow_neq025_const(ptr nocapture %y, ptr nocapture readonly %x)  {
; CHECK-LABEL:       @vspow_neq025_const
; CHECK-PWR10:       __powf4_P10
; CHECK-PWR9:        __powf4_P9
; CHECK-PWR8:        __powf4_P8
; CHECK-PWR7:        __powf4_P7
; CHECK:             blr
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %next.gep = getelementptr float, ptr %y, i64 %index
  %next.gep19 = getelementptr float, ptr %x, i64 %index
  %wide.load = load <4 x float>, ptr %next.gep19, align 4
  %0 = call ninf afn nsz <4 x float> @__powf4(<4 x float> %wide.load, <4 x float> <float 0x3FE851EB80000000, float 2.500000e-01, float 0x3FE851EB80000000, float 2.500000e-01>)
  store <4 x float> %0, ptr %next.gep, align 4
  %index.next = add i64 %index, 4
  %1 = icmp eq i64 %index.next, 1024
  br i1 %1, label %for.end, label %vector.body

for.end:
  ret void
}

; Exponent is 0.75
define void @vspow_075(ptr nocapture %y, ptr nocapture readonly %x)  {
; CHECK-LABEL:       @vspow_075
; CHECK-NOT:         __powf4_P{{[7,8,9,10]}}
; CHECK:             xvrsqrtesp
; CHECK:             blr
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %next.gep = getelementptr float, ptr %y, i64 %index
  %next.gep19 = getelementptr float, ptr %x, i64 %index
  %wide.load = load <4 x float>, ptr %next.gep19, align 4
  %0 = call ninf afn <4 x float> @__powf4(<4 x float> %wide.load, <4 x float> <float 7.500000e-01, float 7.500000e-01, float 7.500000e-01, float 7.500000e-01>)
  store <4 x float> %0, ptr %next.gep, align 4
  %index.next = add i64 %index, 4
  %1 = icmp eq i64 %index.next, 1024
  br i1 %1, label %for.end, label %vector.body

for.end:
  ret void
}

; Exponent is 0.25
define void @vspow_025(ptr nocapture %y, ptr nocapture readonly %x)  {
; CHECK-LABEL:       @vspow_025
; CHECK-NOT:         __powf4_P{{[7,8,9,10]}}
; CHECK:             xvrsqrtesp
; CHECK:             blr
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %next.gep = getelementptr float, ptr %y, i64 %index
  %next.gep19 = getelementptr float, ptr %x, i64 %index
  %wide.load = load <4 x float>, ptr %next.gep19, align 4
  %0 = call ninf afn nsz <4 x float> @__powf4(<4 x float> %wide.load, <4 x float> <float 2.500000e-01, float 2.500000e-01, float 2.500000e-01, float 2.500000e-01>)
  store <4 x float> %0, ptr %next.gep, align 4
  %index.next = add i64 %index, 4
  %1 = icmp eq i64 %index.next, 1024
  br i1 %1, label %for.end, label %vector.body

for.end:
  ret void
}

; Exponent is 0.75 but no proper fast-math flags
define void @vspow_075_nofast(ptr nocapture %y, ptr nocapture readonly %x)  {
; CHECK-LABEL:       @vspow_075_nofast
; CHECK-PWR10:       __powf4_P10
; CHECK-PWR9:        __powf4_P9
; CHECK-PWR8:        __powf4_P8
; CHECK-PWR7:        __powf4_P7
; CHECK-NOT:         xvrsqrtesp
; CHECK:             blr
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %next.gep = getelementptr float, ptr %y, i64 %index
  %next.gep19 = getelementptr float, ptr %x, i64 %index
  %wide.load = load <4 x float>, ptr %next.gep19, align 4
  %0 = call <4 x float> @__powf4(<4 x float> %wide.load, <4 x float> <float 7.500000e-01, float 7.500000e-01, float 7.500000e-01, float 7.500000e-01>)
  store <4 x float> %0, ptr %next.gep, align 4
  %index.next = add i64 %index, 4
  %1 = icmp eq i64 %index.next, 1024
  br i1 %1, label %for.end, label %vector.body

for.end:
  ret void
}

; Exponent is 0.25 but no proper fast-math flags
define void @vspow_025_nofast(ptr nocapture %y, ptr nocapture readonly %x)  {
; CHECK-LABEL:       @vspow_025_nofast
; CHECK-PWR10:       __powf4_P10
; CHECK-PWR9:        __powf4_P9
; CHECK-PWR8:        __powf4_P8
; CHECK-PWR7:        __powf4_P7
; CHECK-NOT:         xvrsqrtesp
; CHECK:             blr
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %next.gep = getelementptr float, ptr %y, i64 %index
  %next.gep19 = getelementptr float, ptr %x, i64 %index
  %wide.load = load <4 x float>, ptr %next.gep19, align 4
  %0 = call <4 x float> @__powf4(<4 x float> %wide.load, <4 x float> <float 2.500000e-01, float 2.500000e-01, float 2.500000e-01, float 2.500000e-01>)
  store <4 x float> %0, ptr %next.gep, align 4
  %index.next = add i64 %index, 4
  %1 = icmp eq i64 %index.next, 1024
  br i1 %1, label %for.end, label %vector.body

for.end:
  ret void
}

; Function Attrs: nounwind readnone speculatable willreturn
declare <4 x float> @__powf4(<4 x float>, <4 x float>)
