; RUN: llc -mtriple=hexagon-unknown-elf -mcpu=hexagonv67 < %s | FileCheck %s


; CHECK-LABEL: t1
; CHECK-NOT: dfmax

define dso_local double @t1(double %a, double %b) local_unnamed_addr {
entry:
  %0 = tail call double @llvm.maxnum.f64(double %a, double %b)
  ret double %0
}

; CHECK-LABEL: t2
; CHECK-NOT: dfmin

define dso_local double @t2(double %a, double %b) local_unnamed_addr {
entry:
  %0 = tail call double @llvm.minnum.f64(double %a, double %b)
  ret double %0
}

; CHECK-LABEL: t3
; CHECK-NOT: sfmax

define dso_local float @t3(float %a, float %b) local_unnamed_addr {
entry:
  %0 = tail call float @llvm.maxnum.f32(float %a, float %b)
  ret float %0
}

; CHECK-LABEL: t4
; CHECK-NOT: sfmin

define dso_local float @t4(float %a, float %b) local_unnamed_addr {
entry:
  %0 = tail call float @llvm.minnum.f32(float %a, float %b)
  ret float %0
}

; CHECK-LABEL: t1num
; CHECK: dfmax

define dso_local double @t1num(double %a, double %b) local_unnamed_addr {
entry:
  %0 = tail call double @llvm.maximumnum.f64(double %a, double %b)
  ret double %0
}

; CHECK-LABEL: t2num
; CHECK: dfmin

define dso_local double @t2num(double %a, double %b) local_unnamed_addr {
entry:
  %0 = tail call double @llvm.minimumnum.f64(double %a, double %b)
  ret double %0
}

; CHECK-LABEL: t3num
; CHECK: sfmax

define dso_local float @t3num(float %a, float %b) local_unnamed_addr {
entry:
  %0 = tail call float @llvm.maximumnum.f32(float %a, float %b)
  ret float %0
}

; CHECK-LABEL: t4num
; CHECK: sfmin

define dso_local float @t4num(float %a, float %b) local_unnamed_addr {
entry:
  %0 = tail call float @llvm.minimumnum.f32(float %a, float %b)
  ret float %0
}


declare double @llvm.minnum.f64(double, double) #1
declare double @llvm.maxnum.f64(double, double) #1
declare float @llvm.maxnum.f32(float, float) #1
declare float @llvm.minnum.f32(float, float) #1

attributes #1 = { nounwind readnone speculatable }
