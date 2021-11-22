; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu \
; RUN:   -mcpu=a2 | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu \
; RUN:   -mcpu=pwr7 -mattr=+vsx | FileCheck -check-prefix=CHECK-VSX %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu \
; RUN:   -mcpu=g5
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu \
; RUN:   -mcpu=pwr9 -mattr=-direct-move | FileCheck -check-prefix=CHECK-P9 %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define i64 @foo(float %a) nounwind {
  %x = fptosi float %a to i64
  ret i64 %x

; CHECK: @foo
; CHECK: fctidz [[REG:[0-9]+]], 1
; CHECK: stfd [[REG]],
; CHECK: ld 3,
; CHECK: blr

; CHECK-VSX: @foo
; CHECK-VSX: xscvdpsxds [[REG:[0-9]+]], 1
; CHECK-VSX: stfd [[REG]],
; CHECK-VSX: ld 3,
; CHECK-VSX: blr

; CHECK-P9-LABEL: @foo
; CHECK-P9: xscvdpsxds [[REG:[0-9]+]], 1
; CHECK-P9: stfd [[REG]],
; CHECK-P9: ld 3,
; CHECK-P9: blr
}

define i64 @foo2(double %a) nounwind {
  %x = fptosi double %a to i64
  ret i64 %x

; CHECK: @foo2
; CHECK: fctidz [[REG:[0-9]+]], 1
; CHECK: stfd [[REG]],
; CHECK: ld 3,
; CHECK: blr

; CHECK-VSX: @foo2
; CHECK-VSX: xscvdpsxds [[REG:[0-9]+]], 1
; CHECK-VSX: stfd [[REG]],
; CHECK-VSX: ld 3,
; CHECK-VSX: blr

; CHECK-P9-LABEL: @foo2
; CHECK-P9: xscvdpsxds [[REG:[0-9]+]], 1
; CHECK-P9: stfd [[REG]],
; CHECK-P9: ld 3,
; CHECK-P9: blr
}

define i64 @foo3(float %a) nounwind {
  %x = fptoui float %a to i64
  ret i64 %x

; CHECK: @foo3
; CHECK: fctiduz [[REG:[0-9]+]], 1
; CHECK: stfd [[REG]],
; CHECK: ld 3,
; CHECK: blr

; CHECK-VSX: @foo3
; CHECK-VSX: xscvdpuxds [[REG:[0-9]+]], 1
; CHECK-VSX: stfd [[REG]],
; CHECK-VSX: ld 3,
; CHECK-VSX: blr

; CHECK-P9-LABEL: @foo3
; CHECK-P9: xscvdpuxds [[REG:[0-9]+]], 1
; CHECK-P9: stfd [[REG]],
; CHECK-P9: ld 3,
; CHECK-P9: blr
}

define i64 @foo4(double %a) nounwind {
  %x = fptoui double %a to i64
  ret i64 %x

; CHECK: @foo4
; CHECK: fctiduz [[REG:[0-9]+]], 1
; CHECK: stfd [[REG]],
; CHECK: ld 3,
; CHECK: blr

; CHECK-VSX: @foo4
; CHECK-VSX: xscvdpuxds [[REG:[0-9]+]], 1
; CHECK-VSX: stfd [[REG]],
; CHECK-VSX: ld 3,
; CHECK-VSX: blr

; CHECK-P9-LABEL: @foo4
; CHECK-P9: xscvdpuxds [[REG:[0-9]+]], 1
; CHECK-P9: stfd [[REG]],
; CHECK-P9: ld 3,
; CHECK-P9: blr
}

define i32 @goo(float %a) nounwind {
  %x = fptosi float %a to i32
  ret i32 %x

; CHECK: @goo
; CHECK: fctiwz [[REG:[0-9]+]], 1
; CHECK: stfiwx [[REG]],
; CHECK: lwz 3,
; CHECK: blr

; CHECK-VSX: @goo
; CHECK-VSX: xscvdpsxws [[REG:[0-9]+]], 1
; CHECK-VSX: stfiwx [[REG]],
; CHECK-VSX: lwz 3,
; CHECK-VSX: blr
}

define i32 @goo2(double %a) nounwind {
  %x = fptosi double %a to i32
  ret i32 %x

; CHECK: @goo2
; CHECK: fctiwz [[REG:[0-9]+]], 1
; CHECK: stfiwx [[REG]],
; CHECK: lwz 3,
; CHECK: blr

; CHECK-VSX: @goo2
; CHECK-VSX: xscvdpsxws [[REG:[0-9]+]], 1
; CHECK-VSX: stfiwx [[REG]],
; CHECK-VSX: lwz 3,
; CHECK-VSX: blr
}

define i32 @goo3(float %a) nounwind {
  %x = fptoui float %a to i32
  ret i32 %x

; CHECK: @goo3
; CHECK: fctiwuz [[REG:[0-9]+]], 1
; CHECK: stfiwx [[REG]],
; CHECK: lwz 3,
; CHECK: blr

; CHECK-VSX: @goo3
; CHECK-VSX: xscvdpuxws [[REG:[0-9]+]], 1
; CHECK-VSX: stfiwx [[REG]],
; CHECK-VSX: lwz 3,
; CHECK-VSX: blr
}

define i32 @goo4(double %a) nounwind {
  %x = fptoui double %a to i32
  ret i32 %x

; CHECK: @goo4
; CHECK: fctiwuz [[REG:[0-9]+]], 1
; CHECK: stfiwx [[REG]],
; CHECK: lwz 3,
; CHECK: blr

; CHECK-VSX: @goo4
; CHECK-VSX: xscvdpuxws [[REG:[0-9]+]], 1
; CHECK-VSX: stfiwx [[REG]],
; CHECK-VSX: lwz 3,
; CHECK-VSX: blr
}

