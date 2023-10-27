; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -mattr=-vsx | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -enable-no-infs-fp-math -enable-no-nans-fp-math -mattr=-vsx | FileCheck -check-prefix=CHECK-FM %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -enable-no-infs-fp-math -enable-no-nans-fp-math -mattr=+vsx | FileCheck -check-prefix=CHECK-FM-VSX %s
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

; CHECK-FM: @zerocmp1
; CHECK-FM: fsel 1, 1, 2, 3
; CHECK-FM: blr

; CHECK-FM-VSX: @zerocmp1
; CHECK-FM-VSX: fsel 1, 1, 2, 3
; CHECK-FM-VSX: blr
}

define double @zerocmp2(double %a, double %y, double %z) #0 {
entry:
  %cmp = fcmp ogt double %a, 0.000000e+00
  %y.z = select i1 %cmp, double %y, double %z
  ret double %y.z

; CHECK: @zerocmp2
; CHECK-NOT: fsel
; CHECK: blr

; CHECK-FM: @zerocmp2
; CHECK-FM: fneg [[REG:[0-9]+]], 1
; CHECK-FM: fsel 1, [[REG]], 3, 2
; CHECK-FM: blr

; CHECK-FM-VSX: @zerocmp2
; CHECK-FM-VSX: xsnegdp [[REG:[0-9]+]], 1
; CHECK-FM-VSX: fsel 1, [[REG]], 3, 2
; CHECK-FM-VSX: blr
}

define double @zerocmp3(double %a, double %y, double %z) #0 {
entry:
  %cmp = fcmp oeq double %a, 0.000000e+00
  %y.z = select i1 %cmp, double %y, double %z
  ret double %y.z

; CHECK: @zerocmp3
; CHECK-NOT: fsel
; CHECK: blr

; CHECK-FM: @zerocmp3
; CHECK-FM: fsel [[REG:[0-9]+]], 1, 2, 3
; CHECK-FM: fneg [[REG2:[0-9]+]], 1
; CHECK-FM: fsel 1, [[REG2]], [[REG]], 3
; CHECK-FM: blr

; CHECK-FM-VSX: @zerocmp3
; CHECK-FM-VSX: fsel [[REG:[0-9]+]], 1, 2, 3
; CHECK-FM-VSX: xsnegdp [[REG2:[0-9]+]], 1
; CHECK-FM-VSX: fsel 1, [[REG2]], [[REG]], 3
; CHECK-FM-VSX: blr
}

define double @min1(double %a, double %b) #0 {
entry:
  %cmp = fcmp ole double %a, %b
  %cond = select i1 %cmp, double %a, double %b
  ret double %cond

; CHECK: @min1
; CHECK-NOT: fsel
; CHECK: blr

; CHECK-FM: @min1
; CHECK-FM: fsub [[REG:[0-9]+]], 2, 1
; CHECK-FM: fsel 1, [[REG]], 1, 2
; CHECK-FM: blr

; CHECK-FM-VSX: @min1
; CHECK-FM-VSX: xssubdp [[REG:[0-9]+]], 2, 1
; CHECK-FM-VSX: fsel 1, [[REG]], 1, 2
; CHECK-FM-VSX: blr
}

define double @max1(double %a, double %b) #0 {
entry:
  %cmp = fcmp oge double %a, %b
  %cond = select i1 %cmp, double %a, double %b
  ret double %cond

; CHECK: @max1
; CHECK-NOT: fsel
; CHECK: blr

; CHECK-FM: @max1
; CHECK-FM: fsub [[REG:[0-9]+]], 1, 2
; CHECK-FM: fsel 1, [[REG]], 1, 2
; CHECK-FM: blr

; CHECK-FM-VSX: @max1
; CHECK-FM-VSX: xssubdp [[REG:[0-9]+]], 1, 2
; CHECK-FM-VSX: fsel 1, [[REG]], 1, 2
; CHECK-FM-VSX: blr
}

define double @cmp1(double %a, double %b, double %y, double %z) #0 {
entry:
  %cmp = fcmp ult double %a, %b
  %z.y = select i1 %cmp, double %z, double %y
  ret double %z.y

; CHECK: @cmp1
; CHECK-NOT: fsel
; CHECK: blr

; CHECK-FM: @cmp1
; CHECK-FM: fsub [[REG:[0-9]+]], 1, 2
; CHECK-FM: fsel 1, [[REG]], 3, 4
; CHECK-FM: blr

; CHECK-FM-VSX: @cmp1
; CHECK-FM-VSX: xssubdp [[REG:[0-9]+]], 1, 2
; CHECK-FM-VSX: fsel 1, [[REG]], 3, 4
; CHECK-FM-VSX: blr
}

define double @cmp2(double %a, double %b, double %y, double %z) #0 {
entry:
  %cmp = fcmp ogt double %a, %b
  %y.z = select i1 %cmp, double %y, double %z
  ret double %y.z

; CHECK: @cmp2
; CHECK-NOT: fsel
; CHECK: blr

; CHECK-FM: @cmp2
; CHECK-FM: fsub [[REG:[0-9]+]], 2, 1
; CHECK-FM: fsel 1, [[REG]], 4, 3
; CHECK-FM: blr

; CHECK-FM-VSX: @cmp2
; CHECK-FM-VSX: xssubdp [[REG:[0-9]+]], 2, 1
; CHECK-FM-VSX: fsel 1, [[REG]], 4, 3
; CHECK-FM-VSX: blr
}

define double @cmp3(double %a, double %b, double %y, double %z) #0 {
entry:
  %cmp = fcmp oeq double %a, %b
  %y.z = select i1 %cmp, double %y, double %z
  ret double %y.z

; CHECK: @cmp3
; CHECK-NOT: fsel
; CHECK: blr

; CHECK-FM: @cmp3
; CHECK-FM: fsub [[REG:[0-9]+]], 1, 2
; CHECK-FM: fsel [[REG2:[0-9]+]], [[REG]], 3, 4
; CHECK-FM: fneg [[REG3:[0-9]+]], [[REG]]
; CHECK-FM: fsel 1, [[REG3]], [[REG2]], 4
; CHECK-FM: blr

; CHECK-FM-VSX: @cmp3
; CHECK-FM-VSX: xssubdp [[REG:[0-9]+]], 1, 2
; CHECK-FM-VSX: fsel [[REG2:[0-9]+]], [[REG]], 3, 4
; CHECK-FM-VSX: xsnegdp [[REG3:[0-9]+]], [[REG]]
; CHECK-FM-VSX: fsel 1, [[REG3]], [[REG2]], 4
; CHECK-FM-VSX: blr
}

attributes #0 = { nounwind readnone }

