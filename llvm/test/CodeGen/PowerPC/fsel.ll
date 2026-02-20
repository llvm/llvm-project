; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -mattr=-vsx | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -enable-no-nans-fp-math -mattr=+vsx | FileCheck -check-prefix=CHECK-VSX %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define double @zerocmp1(double %a, double %y, double %z) #0 {
entry:
  %cmp = fcmp ult double %a, 0.000000e+00
  %z.y = select i1 %cmp, double %z, double %y
  ret double %z.y

; CHECK: @zerocmp1
; CHECK-NOT: fsel
; CHECK: blr
}

define double @zerocmp1_finite(double %a, double %y, double %z) #0 {
entry:
  %cmp = fcmp ninf nnan ult double %a, 0.000000e+00
  %z.y = select i1 %cmp, double %z, double %y
  ret double %z.y

; CHECK: @zerocmp1_finite
; CHECK: fsel 1, 1, 2, 3
; CHECK: blr

; CHECK-VSX: @zerocmp1_finite
; CHECK-VSX: fsel 1, 1, 2, 3
; CHECK-VSX: blr
}

define double @zerocmp2(double %a, double %y, double %z) #0 {
entry:
  %cmp = fcmp ogt double %a, 0.000000e+00
  %y.z = select i1 %cmp, double %y, double %z
  ret double %y.z

; CHECK: @zerocmp2
; CHECK-NOT: fsel
; CHECK: blr
}

define double @zerocmp2_finite(double %a, double %y, double %z) #0 {
entry:
  %cmp = fcmp ninf nnan ogt double %a, 0.000000e+00
  %y.z = select i1 %cmp, double %y, double %z
  ret double %y.z

; CHECK: @zerocmp2_finite
; CHECK: fneg [[REG:[0-9]+]], 1
; CHECK: fsel 1, [[REG]], 3, 2
; CHECK: blr

; CHECK-VSX: @zerocmp2_finite
; CHECK-VSX: xsnegdp [[REG:[0-9]+]], 1
; CHECK-VSX: fsel 1, [[REG]], 3, 2
; CHECK-VSX: blr
}

define double @zerocmp3(double %a, double %y, double %z) #0 {
entry:
  %cmp = fcmp oeq double %a, 0.000000e+00
  %y.z = select i1 %cmp, double %y, double %z
  ret double %y.z

; CHECK: @zerocmp3
; CHECK-NOT: fsel
; CHECK: blr
}

define double @zerocmp3_finite(double %a, double %y, double %z) #0 {
entry:
  %cmp = fcmp ninf nnan oeq double %a, 0.000000e+00
  %y.z = select i1 %cmp, double %y, double %z
  ret double %y.z

; CHECK: @zerocmp3_finite
; CHECK: fsel [[REG:[0-9]+]], 1, 2, 3
; CHECK: fneg [[REG2:[0-9]+]], 1
; CHECK: fsel 1, [[REG2]], [[REG]], 3
; CHECK: blr

; CHECK-VSX: @zerocmp3_finite
; CHECK-VSX: fsel [[REG:[0-9]+]], 1, 2, 3
; CHECK-VSX: xsnegdp [[REG2:[0-9]+]], 1
; CHECK-VSX: fsel 1, [[REG2]], [[REG]], 3
; CHECK-VSX: blr
}

define double @min1(double %a, double %b) #0 {
entry:
  %cmp = fcmp ole double %a, %b
  %cond = select i1 %cmp, double %a, double %b
  ret double %cond

; CHECK: @min1
; CHECK-NOT: fsel
; CHECK: blr
}

define double @min1_finite(double %a, double %b) #0 {
entry:
  %cmp = fcmp ninf nnan ole double %a, %b
  %cond = select i1 %cmp, double %a, double %b
  ret double %cond

; CHECK: @min1_finite
; CHECK: fsub [[REG:[0-9]+]], 2, 1
; CHECK: fsel 1, [[REG]], 1, 2
; CHECK: blr

; CHECK-VSX: @min1_finite
; CHECK-VSX: xssubdp [[REG:[0-9]+]], 2, 1
; CHECK-VSX: fsel 1, [[REG]], 1, 2
; CHECK-VSX: blr
}

define double @max1(double %a, double %b) #0 {
entry:
  %cmp = fcmp oge double %a, %b
  %cond = select i1 %cmp, double %a, double %b
  ret double %cond

; CHECK: @max1
; CHECK-NOT: fsel
; CHECK: blr
}

define double @max1_finite(double %a, double %b) #0 {
entry:
  %cmp = fcmp ninf nnan oge double %a, %b
  %cond = select i1 %cmp, double %a, double %b
  ret double %cond

; CHECK: @max1_finite
; CHECK: fsub [[REG:[0-9]+]], 1, 2
; CHECK: fsel 1, [[REG]], 1, 2
; CHECK: blr

; CHECK-VSX: @max1_finite
; CHECK-VSX: xssubdp [[REG:[0-9]+]], 1, 2
; CHECK-VSX: fsel 1, [[REG]], 1, 2
; CHECK-VSX: blr
}

define double @cmp1(double %a, double %b, double %y, double %z) #0 {
entry:
  %cmp = fcmp ult double %a, %b
  %z.y = select i1 %cmp, double %z, double %y
  ret double %z.y

; CHECK: @cmp1
; CHECK-NOT: fsel
; CHECK: blr
}

define double @cmp1_finite(double %a, double %b, double %y, double %z) #0 {
entry:
  %cmp = fcmp ninf nnan ult double %a, %b
  %z.y = select i1 %cmp, double %z, double %y
  ret double %z.y

; CHECK: @cmp1_finite
; CHECK: fsub [[REG:[0-9]+]], 1, 2
; CHECK: fsel 1, [[REG]], 3, 4
; CHECK: blr

; CHECK-VSX: @cmp1_finite
; CHECK-VSX: xssubdp [[REG:[0-9]+]], 1, 2
; CHECK-VSX: fsel 1, [[REG]], 3, 4
; CHECK-VSX: blr
}

define double @cmp2(double %a, double %b, double %y, double %z) #0 {
entry:
  %cmp = fcmp ogt double %a, %b
  %y.z = select i1 %cmp, double %y, double %z
  ret double %y.z

; CHECK: @cmp2
; CHECK-NOT: fsel
; CHECK: blr
}

define double @cmp2_finite(double %a, double %b, double %y, double %z) #0 {
entry:
  %cmp = fcmp ninf nnan ogt double %a, %b
  %y.z = select i1 %cmp, double %y, double %z
  ret double %y.z

; CHECK: @cmp2_finite
; CHECK: fsub [[REG:[0-9]+]], 2, 1
; CHECK: fsel 1, [[REG]], 4, 3
; CHECK: blr

; CHECK-VSX: @cmp2_finite
; CHECK-VSX: xssubdp [[REG:[0-9]+]], 2, 1
; CHECK-VSX: fsel 1, [[REG]], 4, 3
; CHECK-VSX: blr
}

define double @cmp3(double %a, double %b, double %y, double %z) #0 {
entry:
  %cmp = fcmp oeq double %a, %b
  %y.z = select i1 %cmp, double %y, double %z
  ret double %y.z

; CHECK: @cmp3
; CHECK-NOT: fsel
; CHECK: blr
}

define double @cmp3_finite(double %a, double %b, double %y, double %z) #0 {
entry:
  %cmp = fcmp ninf nnan oeq double %a, %b
  %y.z = select i1 %cmp, double %y, double %z
  ret double %y.z

; CHECK: @cmp3_finite
; CHECK: fsub [[REG:[0-9]+]], 1, 2
; CHECK: fsel [[REG2:[0-9]+]], [[REG]], 3, 4
; CHECK: fneg [[REG3:[0-9]+]], [[REG]]
; CHECK: fsel 1, [[REG3]], [[REG2]], 4
; CHECK: blr

; CHECK-VSX: @cmp3_finite
; CHECK-VSX: xssubdp [[REG:[0-9]+]], 1, 2
; CHECK-VSX: fsel [[REG2:[0-9]+]], [[REG]], 3, 4
; CHECK-VSX: xsnegdp [[REG3:[0-9]+]], [[REG]]
; CHECK-VSX: fsel 1, [[REG3]], [[REG2]], 4
; CHECK-VSX: blr
}

attributes #0 = { nounwind readnone }