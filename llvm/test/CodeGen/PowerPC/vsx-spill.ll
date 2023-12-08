; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=+vsx \
; RUN:     -ppc-vsr-nums-as-vr -ppc-asm-full-reg-names < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=+vsx \
; RUN:     -ppc-vsr-nums-as-vr -ppc-asm-full-reg-names < %s | FileCheck \
; RUN:   -check-prefix=CHECK-REG %s
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=+vsx -fast-isel -O0 \
; RUN:     -ppc-vsr-nums-as-vr -ppc-asm-full-reg-names < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=+vsx -fast-isel -O0 \
; RUN:     -ppc-vsr-nums-as-vr -ppc-asm-full-reg-names < %s | \
; RUN:   FileCheck -check-prefix=CHECK-FISL %s
; RUN: llc -verify-machineinstrs -mcpu=pwr9 -ppc-vsr-nums-as-vr \
; RUN:     -ppc-asm-full-reg-names < %s | FileCheck -check-prefix=CHECK-P9-REG %s
; RUN: llc -verify-machineinstrs -mcpu=pwr9 -fast-isel -O0 \
; RUN:     -ppc-vsr-nums-as-vr -ppc-asm-full-reg-names < %s | FileCheck \
; RUN:   -check-prefix=CHECK-P9-FISL %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define double @foo1(double %a) nounwind {
entry:
  call void asm sideeffect "", "~{f0},~{f1},~{f2},~{f3},~{f4},~{f5},~{f6},~{f7},~{f8},~{f9},~{f10},~{f11},~{f12},~{f13},~{f14},~{f15},~{f16},~{f17},~{f18},~{f19},~{f20},~{f21},~{f22},~{f23},~{f24},~{f25},~{f26},~{f27},~{f28},~{f29},~{f30},~{f31}"() nounwind
  br label %return

; CHECK-REG: @foo1
; CHECK-REG: xxlor v2, f1, f1
; CHECK-REG: xxlor f1, v2, v2
; CHECK-REG: blr

; CHECK-FISL: @foo1
; CHECK-FISL-NOT: lis
; CHECK-FISL-NOT: ori
; CHECK-FISL: li r3, -152
; CHECK-FISL-NOT: lis
; CHECK-FISL-NOT: ori
; CHECK-FISL: stxsdx f1, r1, r3
; CHECK-FISL: blr

; CHECK-P9-REG: @foo1
; CHECK-P9-REG: xscpsgndp v2, f1, f1
; CHECK-P9-REG: xscpsgndp f1, v2, v2
; CHECK-P9-REG: blr

; CHECK-P9-FISL: @foo1
; CHECK-P9-FISL: stfd f31, -8(r1)
; CHECK-P9-FISL: blr

return:                                           ; preds = %entry
  ret double %a
}

define double @foo2(double %a) nounwind {
entry:
  %b = fadd double %a, %a
  call void asm sideeffect "", "~{f0},~{f1},~{f2},~{f3},~{f4},~{f5},~{f6},~{f7},~{f8},~{f9},~{f10},~{f11},~{f12},~{f13},~{f14},~{f15},~{f16},~{f17},~{f18},~{f19},~{f20},~{f21},~{f22},~{f23},~{f24},~{f25},~{f26},~{f27},~{f28},~{f29},~{f30},~{f31}"() nounwind
  br label %return

; CHECK-REG: @foo2
; CHECK-REG: {{xxlor|xsadddp}} v2, f1, f1
; CHECK-REG: {{xxlor|xsadddp}} f1, f0, f0
; CHECK-REG: blr

; CHECK-FISL: @foo2
; CHECK-FISL: xsadddp [[REG0:f[0-9]+]], f1, f1
; CHECK-FISL: stxsdx [[REG0]], r1, r3
; CHECK-FISL: lxsdx f1, r1, r3
; CHECK-FISL: blr

; CHECK-P9-REG: @foo2
; CHECK-P9-REG: {{xscpsgndp|xsadddp}} v2, f1, f1
; CHECK-P9-REG: {{xscpsgndp|xsadddp}} f1, v2, v2
; CHECK-P9-REG: blr

; CHECK-P9-FISL: @foo2
; CHECK-P9-FISL: xsadddp [[REG0:f[0-9]+]], f1, f1
; CHECK-P9-FISL: stfd [[REG0]], -152(r1)
; CHECK-P9-FISL: lfd f1, -152(r1)
; CHECK-P9-FISL: blr

return:                                           ; preds = %entry
  ret double %b
}

define double @foo3(double %a) nounwind {
entry:
  call void asm sideeffect "", "~{f0},~{f1},~{f2},~{f3},~{f4},~{f5},~{f6},~{f7},~{f8},~{f9},~{f10},~{f11},~{f12},~{f13},~{f14},~{f15},~{f16},~{f17},~{f18},~{f19},~{f20},~{f21},~{f22},~{f23},~{f24},~{f25},~{f26},~{f27},~{f28},~{f29},~{f30},~{f31},~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31}"() nounwind
  br label %return

; CHECK: @foo3
; CHECK: stxsdx f1, r1, r3
; CHECK: lxsdx f0, r1, r3
; CHECK: xsadddp f1, f0, f0
; CHECK: blr

; CHECK-P9-REG-LABEL: foo3
; CHECK-P9-REG: stdu r1, -400(r1)
; CHECK-P9-REG-DAG: lfd f30, 384(r1)
; CHECK-P9-REG-DAG: xsadddp f1, f0, f0

; CHECK-P9-FISL-LABEL: foo3
; CHECK-P9-FISL: stdu r1, -400(r1)
; CHECK-P9-FISL: lfd f0, 56(r1)
; CHECK-P9-FISL: xsadddp f1, f0, f0
return:                                           ; preds = %entry
  %b = fadd double %a, %a
  ret double %b
}

