; RUN: llc -relocation-model=pic -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu \
; RUN:   -mcpu=pwr8 -mattr=-direct-move | FileCheck %s
; RUN: llc -relocation-model=pic -verify-machineinstrs < %s -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr8 -mattr=-direct-move | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr9 -mattr=-direct-move | FileCheck %s -check-prefix=CHECK-P9

@d = common global double 0.000000e+00, align 8
@f = common global float 0.000000e+00, align 4
@i = common global i32 0, align 4
@ui = common global i32 0, align 4

; Function Attrs: nounwind
define void @dblToInt() #0 {
entry:
  %ii = alloca i32, align 4
  %0 = load double, ptr @d, align 8
  %conv = fptosi double %0 to i32
  store volatile i32 %conv, ptr %ii, align 4
  ret void
; CHECK-LABEL: @dblToInt
; CHECK: xscvdpsxws [[REGCONV1:[0-9]+]],
; CHECK: stfiwx [[REGCONV1]],
}

; Function Attrs: nounwind
define void @fltToInt() #0 {
entry:
  %ii = alloca i32, align 4
  %0 = load float, ptr @f, align 4
  %conv = fptosi float %0 to i32
  store volatile i32 %conv, ptr %ii, align 4
  ret void
; CHECK-LABEL: @fltToInt
; CHECK: xscvdpsxws [[REGCONV2:[0-9]+]],
; CHECK: stfiwx [[REGCONV2]],
}

; Function Attrs: nounwind
define void @intToDbl() #0 {
entry:
  %dd = alloca double, align 8
  %0 = load i32, ptr @i, align 4
  %conv = sitofp i32 %0 to double
  store volatile double %conv, ptr %dd, align 8
  ret void
; CHECK-LABEL: @intToDbl
; CHECK: lfiwax [[REGLD1:[0-9]+]],
; CHECK: xscvsxddp {{[0-9]+}}, [[REGLD1]]
}

; Function Attrs: nounwind
define void @intToFlt() #0 {
entry:
  %ff = alloca float, align 4
  %0 = load i32, ptr @i, align 4
  %conv = sitofp i32 %0 to float
  store volatile float %conv, ptr %ff, align 4
  ret void
; CHECK-LABEL: @intToFlt
; CHECK: lfiwax [[REGLD2:[0-9]+]],
; CHECK: xscvsxdsp {{[0-9]}}, [[REGLD2]]
}

; Function Attrs: nounwind
define void @dblToUInt() #0 {
entry:
  %uiui = alloca i32, align 4
  %0 = load double, ptr @d, align 8
  %conv = fptoui double %0 to i32
  store volatile i32 %conv, ptr %uiui, align 4
  ret void
; CHECK-LABEL: @dblToUInt
; CHECK: xscvdpuxws [[REGCONV3:[0-9]+]],
; CHECK: stfiwx [[REGCONV3]],
}

; Function Attrs: nounwind
define void @fltToUInt() #0 {
entry:
  %uiui = alloca i32, align 4
  %0 = load float, ptr @f, align 4
  %conv = fptoui float %0 to i32
  store volatile i32 %conv, ptr %uiui, align 4
  ret void
; CHECK-LABEL: @fltToUInt
; CHECK: xscvdpuxws [[REGCONV4:[0-9]+]],
; CHECK: stfiwx [[REGCONV4]],
}

; Function Attrs: nounwind
define void @uIntToDbl() #0 {
entry:
  %dd = alloca double, align 8
  %0 = load i32, ptr @ui, align 4
  %conv = uitofp i32 %0 to double
  store volatile double %conv, ptr %dd, align 8
  ret void
; CHECK-LABEL: @uIntToDbl
; CHECK: lfiwzx [[REGLD3:[0-9]+]],
; CHECK: xscvuxddp {{[0-9]+}}, [[REGLD3]]
}

; Function Attrs: nounwind
define void @uIntToFlt() #0 {
entry:
  %ff = alloca float, align 4
  %0 = load i32, ptr @ui, align 4
  %conv = uitofp i32 %0 to float
  store volatile float %conv, ptr %ff, align 4
  ret void
; CHECK-LABEL: @uIntToFlt
; CHECK: lfiwzx [[REGLD4:[0-9]+]],
; CHECK: xscvuxdsp {{[0-9]+}}, [[REGLD4]]
}

; Function Attrs: nounwind
define void @dblToFloat() #0 {
entry:
  %ff = alloca float, align 4
  %0 = load double, ptr @d, align 8
  %conv = fptrunc double %0 to float
  store volatile float %conv, ptr %ff, align 4
  ret void
; CHECK-LABEL: @dblToFloat
; CHECK: lfd [[REGLD5:[0-9]+]],
; CHECK: stfs [[REGLD5]],
; CHECK-P9-LABEL: @dblToFloat
; CHECK-P9: lfd [[REGLD5:[0-9]+]],
; CHECK-P9: stfs [[REGLD5]],
}

; Function Attrs: nounwind
define void @floatToDbl() #0 {
entry:
  %dd = alloca double, align 8
  %0 = load float, ptr @f, align 4
  %conv = fpext float %0 to double
  store volatile double %conv, ptr %dd, align 8
  ret void
; CHECK-LABEL: @floatToDbl
; CHECK: lfs [[REGLD5:[0-9]+]],
; CHECK: stfd [[REGLD5]],
; CHECK-P9-LABEL: @floatToDbl
; CHECK-P9: lfs [[REGLD5:[0-9]+]],
; CHECK-P9: stfd [[REGLD5]],
}
