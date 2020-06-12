; RUN: llc -vector-library=MASSV < %s -mtriple=powerpc64le-unknown-unknown -mcpu=pwr9 | FileCheck -check-prefixes=CHECK-PWR9 %s
; RUN: llc -vector-library=MASSV < %s -mtriple=powerpc64le-unknown-unknown -mcpu=pwr8 | FileCheck -check-prefixes=CHECK-PWR8 %s

; Exponent is a variable
define void @my_vpow_var(double* nocapture %z, double* nocapture readonly %y, double* nocapture readonly %x) {
; CHECK-LABEL:       @vspow_var
; CHECK-PWR9:        bl __powd2_P9
; CHECK-PWR8:        bl __powd2_P8
; CHECK:             blr
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %next.gep = getelementptr double, double* %z, i64 %index
  %next.gep31 = getelementptr double, double* %y, i64 %index
  %next.gep32 = getelementptr double, double* %x, i64 %index
  %0 = bitcast double* %next.gep32 to <2 x double>*
  %wide.load = load <2 x double>, <2 x double>* %0, align 8
  %1 = bitcast double* %next.gep31 to <2 x double>*
  %wide.load33 = load <2 x double>, <2 x double>* %1, align 8
  %2 = call ninf afn nsz <2 x double> @__powd2_massv(<2 x double> %wide.load, <2 x double> %wide.load33)
  %3 = bitcast double* %next.gep to <2 x double>*
  store <2 x double> %2, <2 x double>* %3, align 8
  %index.next = add i64 %index, 2
  %4 = icmp eq i64 %index.next, 1024
  br i1 %4, label %for.end, label %vector.body

for.end:
  ret void
}

; Exponent is a constant != 0.75 and !=0.25
define void @my_vpow_const(double* nocapture %y, double* nocapture readonly %x) {
; CHECK-LABEL:       @vspow_const
; CHECK-PWR9:        bl __powd2_P9
; CHECK-PWR8:        bl __powd2_P8
; CHECK:             blr
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %next.gep = getelementptr double, double* %y, i64 %index
  %next.gep19 = getelementptr double, double* %x, i64 %index
  %0 = bitcast double* %next.gep19 to <2 x double>*
  %wide.load = load <2 x double>, <2 x double>* %0, align 8
  %1 = call ninf afn nsz <2 x double> @__powd2_massv(<2 x double> %wide.load, <2 x double> <double 7.600000e-01, double 7.600000e-01>)
  %2 = bitcast double* %next.gep to <2 x double>*
  store <2 x double> %1, <2 x double>* %2, align 8
  %index.next = add i64 %index, 2
  %3 = icmp eq i64 %index.next, 1024
  br i1 %3, label %for.end, label %vector.body

for.end:
  ret void
}

; Exponent is 0.75
define void @my_vpow_075(double* nocapture %y, double* nocapture readonly %x) {
; CHECK-LABEL:       @vspow_075
; CHECK-NOT:         bl __powd2_P{{[8,9]}}
; CHECK:             xvrsqrtesp
; CHECK:             blr
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %next.gep = getelementptr double, double* %y, i64 %index
  %next.gep19 = getelementptr double, double* %x, i64 %index
  %0 = bitcast double* %next.gep19 to <2 x double>*
  %wide.load = load <2 x double>, <2 x double>* %0, align 8
  %1 = call ninf afn <2 x double> @__powd2_massv(<2 x double> %wide.load, <2 x double> <double 7.500000e-01, double 7.500000e-01>)
  %2 = bitcast double* %next.gep to <2 x double>*
  store <2 x double> %1, <2 x double>* %2, align 8
  %index.next = add i64 %index, 2
  %3 = icmp eq i64 %index.next, 1024
  br i1 %3, label %for.end, label %vector.body

for.end:
  ret void
}

; Exponent is 0.25
define void @my_vpow_025(double* nocapture %y, double* nocapture readonly %x) {
; CHECK-LABEL:       @vspow_025
; CHECK-NOT:         bl __powd2_P{{[8,9]}}
; CHECK:             xvrsqrtesp
; CHECK:             blr
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %next.gep = getelementptr double, double* %y, i64 %index
  %next.gep19 = getelementptr double, double* %x, i64 %index
  %0 = bitcast double* %next.gep19 to <2 x double>*
  %wide.load = load <2 x double>, <2 x double>* %0, align 8
  %1 = call ninf afn nsz <2 x double> @__powd2_massv(<2 x double> %wide.load, <2 x double> <double 2.500000e-01, double 2.500000e-01>)
  %2 = bitcast double* %next.gep to <2 x double>*
  store <2 x double> %1, <2 x double>* %2, align 8
  %index.next = add i64 %index, 2
  %3 = icmp eq i64 %index.next, 1024
  br i1 %3, label %for.end, label %vector.body

for.end:
  ret void
}

; Exponent is 0.75 but no proper fast-math flags
define void @my_vpow_075_nofast(double* nocapture %y, double* nocapture readonly %x) {
; CHECK-LABEL:       @vspow_075_nofast
; CHECK-PWR9:        bl __powd2_P9
; CHECK-PWR8:        bl __powd2_P8
; CHECK-NOT:         xvrsqrtesp
; CHECK:             blr
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %next.gep = getelementptr double, double* %y, i64 %index
  %next.gep19 = getelementptr double, double* %x, i64 %index
  %0 = bitcast double* %next.gep19 to <2 x double>*
  %wide.load = load <2 x double>, <2 x double>* %0, align 8
  %1 = call <2 x double> @__powd2_massv(<2 x double> %wide.load, <2 x double> <double 7.500000e-01, double 7.500000e-01>)
  %2 = bitcast double* %next.gep to <2 x double>*
  store <2 x double> %1, <2 x double>* %2, align 8
  %index.next = add i64 %index, 2
  %3 = icmp eq i64 %index.next, 1024
  br i1 %3, label %for.end, label %vector.body

for.end:
  ret void
}

; Exponent is 0.25 but no proper fast-math flags
define void @my_vpow_025_nofast(double* nocapture %y, double* nocapture readonly %x) {
; CHECK-LABEL:       @vspow_025_nofast
; CHECK-PWR9:        bl __powd2_P9
; CHECK-PWR8:        bl __powd2_P8
; CHECK-NOT:         xvrsqrtesp
; CHECK:             blr
entry:
  br label %vector.body

vector.body:
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %entry ]
  %next.gep = getelementptr double, double* %y, i64 %index
  %next.gep19 = getelementptr double, double* %x, i64 %index
  %0 = bitcast double* %next.gep19 to <2 x double>*
  %wide.load = load <2 x double>, <2 x double>* %0, align 8
  %1 = call <2 x double> @__powd2_massv(<2 x double> %wide.load, <2 x double> <double 2.500000e-01, double 2.500000e-01>)
  %2 = bitcast double* %next.gep to <2 x double>*
  store <2 x double> %1, <2 x double>* %2, align 8
  %index.next = add i64 %index, 2
  %3 = icmp eq i64 %index.next, 1024
  br i1 %3, label %for.end, label %vector.body

for.end:
  ret void
}

; Function Attrs: nounwind readnone speculatable willreturn
declare <2 x double> @__powd2_massv(<2 x double>, <2 x double>) #1
