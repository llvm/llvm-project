; RUN: llc -o - %s -mtriple=aarch64-unknown -aarch64-enable-cond-br-tune=false -mcpu=cortex-a57 -mattr=+arith-cbz-fusion | FileCheck %s
; RUN: llc -o - %s -mtriple=aarch64-unknown -aarch64-enable-cond-br-tune=false -mcpu=cyclone | FileCheck %s
; RUN: llc -o - %s -mtriple=aarch64-unknown -aarch64-enable-cond-br-tune=false -mcpu=apple-m5 | FileCheck %s

target triple = "aarch64-unknown"

declare void @fi32(i32, i32)
declare void @fi64(i64, i64)

; CHECK-LABEL: addwri_cbz:
; CHECK:      add  [[R:w[0-9]+]], {{w[0-9]+}}, #13
; CHECK-NEXT: cbz  [[R]], {{.?LBB[0-9_]+}}
define void @addwri_cbz(i32 %a) {
entry:
  %v0 = add i32 %a, 13
  %v1 = add i32 %a, 7
  %cond = icmp ne i32 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: addwri_cbnz:
; CHECK:      add  [[R:w[0-9]+]], {{w[0-9]+}}, #13
; CHECK-NEXT: cbnz [[R]], {{.?LBB[0-9_]+}}
define void @addwri_cbnz(i32 %a) {
entry:
  %v0 = add i32 %a, 13
  %v1 = add i32 %a, 7
  %cond = icmp eq i32 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: addwri_tbz:
; CHECK:      add  [[R:w[0-9]+]], {{w[0-9]+}}, #13
; CHECK-NEXT: tbz  [[R]], #8, {{.?LBB[0-9_]+}}
define void @addwri_tbz(i32 %a) {
entry:
  %v0 = add i32 %a, 13
  %v1 = add i32 %a, 7
  %t = and i32 %v0, 256
  %cond = icmp ne i32 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: addwri_tbnz:
; CHECK:      add  [[R:w[0-9]+]], {{w[0-9]+}}, #13
; CHECK-NEXT: tbnz [[R]], #8, {{.?LBB[0-9_]+}}
define void @addwri_tbnz(i32 %a) {
entry:
  %v0 = add i32 %a, 13
  %v1 = add i32 %a, 7
  %t = and i32 %v0, 256
  %cond = icmp eq i32 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: addwrr_cbz:
; CHECK:      add  [[R:w[0-9]+]], {{w[0-9]+}}, {{w[0-9]+}}
; CHECK-NEXT: cbz  [[R]], {{.?LBB[0-9_]+}}
define void @addwrr_cbz(i32 %a, i32 %b) {
entry:
  %v0 = add i32 %a, %b
  %v1 = add i32 %a, 7
  %cond = icmp ne i32 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: addwrr_cbnz:
; CHECK:      add  [[R:w[0-9]+]], {{w[0-9]+}}, {{w[0-9]+}}
; CHECK-NEXT: cbnz [[R]], {{.?LBB[0-9_]+}}
define void @addwrr_cbnz(i32 %a, i32 %b) {
entry:
  %v0 = add i32 %a, %b
  %v1 = add i32 %a, 7
  %cond = icmp eq i32 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: addwrr_tbz:
; CHECK:      add  [[R:w[0-9]+]], {{w[0-9]+}}, {{w[0-9]+}}
; CHECK-NEXT: tbz  [[R]], #8, {{.?LBB[0-9_]+}}
define void @addwrr_tbz(i32 %a, i32 %b) {
entry:
  %v0 = add i32 %a, %b
  %v1 = add i32 %a, 7
  %t = and i32 %v0, 256
  %cond = icmp ne i32 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: addwrr_tbnz:
; CHECK:      add  [[R:w[0-9]+]], {{w[0-9]+}}, {{w[0-9]+}}
; CHECK-NEXT: tbnz [[R]], #8, {{.?LBB[0-9_]+}}
define void @addwrr_tbnz(i32 %a, i32 %b) {
entry:
  %v0 = add i32 %a, %b
  %v1 = add i32 %a, 7
  %t = and i32 %v0, 256
  %cond = icmp eq i32 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: andwri_cbz:
; CHECK:      and  [[R:w[0-9]+]], {{w[0-9]+}}, #0xff
; CHECK-NEXT: cbz  [[R]], {{.?LBB[0-9_]+}}
define void @andwri_cbz(i32 %a) {
entry:
  %v0 = and i32 %a, 255
  %v1 = add i32 %a, 7
  %cond = icmp ne i32 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: andwri_cbnz:
; CHECK:      and  [[R:w[0-9]+]], {{w[0-9]+}}, #0xff
; CHECK-NEXT: cbnz [[R]], {{.?LBB[0-9_]+}}
define void @andwri_cbnz(i32 %a) {
entry:
  %v0 = and i32 %a, 255
  %v1 = add i32 %a, 7
  %cond = icmp eq i32 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; NOTE: no andwri_tbz/andwri_tbnz: a single-bit test of an immediate-AND result folds
; through the AND, so no fused ANDri+TB(N)Z pair is expressible at the IR level.
; The .mir test covers ANDWri + TB(N)Z directly.

; CHECK-LABEL: andwrr_cbz:
; CHECK:      and  [[R:w[0-9]+]], {{w[0-9]+}}, {{w[0-9]+}}
; CHECK-NEXT: cbz  [[R]], {{.?LBB[0-9_]+}}
define void @andwrr_cbz(i32 %a, i32 %b) {
entry:
  %v0 = and i32 %a, %b
  %v1 = add i32 %a, 7
  %cond = icmp ne i32 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: andwrr_cbnz:
; CHECK:      and  [[R:w[0-9]+]], {{w[0-9]+}}, {{w[0-9]+}}
; CHECK-NEXT: cbnz [[R]], {{.?LBB[0-9_]+}}
define void @andwrr_cbnz(i32 %a, i32 %b) {
entry:
  %v0 = and i32 %a, %b
  %v1 = add i32 %a, 7
  %cond = icmp eq i32 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: andwrr_tbz:
; CHECK:      and  [[R:w[0-9]+]], {{w[0-9]+}}, {{w[0-9]+}}
; CHECK-NEXT: tbz  [[R]], #8, {{.?LBB[0-9_]+}}
define void @andwrr_tbz(i32 %a, i32 %b) {
entry:
  %v0 = and i32 %a, %b
  %v1 = add i32 %a, 7
  %t = and i32 %v0, 256
  %cond = icmp ne i32 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: andwrr_tbnz:
; CHECK:      and  [[R:w[0-9]+]], {{w[0-9]+}}, {{w[0-9]+}}
; CHECK-NEXT: tbnz [[R]], #8, {{.?LBB[0-9_]+}}
define void @andwrr_tbnz(i32 %a, i32 %b) {
entry:
  %v0 = and i32 %a, %b
  %v1 = add i32 %a, 7
  %t = and i32 %v0, 256
  %cond = icmp eq i32 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: eorwri_cbz:
; CHECK:      eor  [[R:w[0-9]+]], {{w[0-9]+}}, #0xff
; CHECK-NEXT: cbz  [[R]], {{.?LBB[0-9_]+}}
define void @eorwri_cbz(i32 %a) {
entry:
  %v0 = xor i32 %a, 255
  %v1 = add i32 %a, 7
  %cond = icmp ne i32 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: eorwri_cbnz:
; CHECK:      eor  [[R:w[0-9]+]], {{w[0-9]+}}, #0xff
; CHECK-NEXT: cbnz [[R]], {{.?LBB[0-9_]+}}
define void @eorwri_cbnz(i32 %a) {
entry:
  %v0 = xor i32 %a, 255
  %v1 = add i32 %a, 7
  %cond = icmp eq i32 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: eorwri_tbz:
; CHECK:      eor  [[R:w[0-9]+]], {{w[0-9]+}}, #0xff
; CHECK-NEXT: tbz  [[R]], #8, {{.?LBB[0-9_]+}}
define void @eorwri_tbz(i32 %a) {
entry:
  %v0 = xor i32 %a, 255
  %v1 = add i32 %a, 7
  %t = and i32 %v0, 256
  %cond = icmp ne i32 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: eorwri_tbnz:
; CHECK:      eor  [[R:w[0-9]+]], {{w[0-9]+}}, #0xff
; CHECK-NEXT: tbnz [[R]], #8, {{.?LBB[0-9_]+}}
define void @eorwri_tbnz(i32 %a) {
entry:
  %v0 = xor i32 %a, 255
  %v1 = add i32 %a, 7
  %t = and i32 %v0, 256
  %cond = icmp eq i32 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: eorwrr_cbz:
; CHECK:      eor  [[R:w[0-9]+]], {{w[0-9]+}}, {{w[0-9]+}}
; CHECK-NEXT: cbz  [[R]], {{.?LBB[0-9_]+}}
define void @eorwrr_cbz(i32 %a, i32 %b) {
entry:
  %v0 = xor i32 %a, %b
  %v1 = add i32 %a, 7
  %cond = icmp ne i32 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: eorwrr_cbnz:
; CHECK:      eor  [[R:w[0-9]+]], {{w[0-9]+}}, {{w[0-9]+}}
; CHECK-NEXT: cbnz [[R]], {{.?LBB[0-9_]+}}
define void @eorwrr_cbnz(i32 %a, i32 %b) {
entry:
  %v0 = xor i32 %a, %b
  %v1 = add i32 %a, 7
  %cond = icmp eq i32 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: eorwrr_tbz:
; CHECK:      eor  [[R:w[0-9]+]], {{w[0-9]+}}, {{w[0-9]+}}
; CHECK-NEXT: tbz  [[R]], #8, {{.?LBB[0-9_]+}}
define void @eorwrr_tbz(i32 %a, i32 %b) {
entry:
  %v0 = xor i32 %a, %b
  %v1 = add i32 %a, 7
  %t = and i32 %v0, 256
  %cond = icmp ne i32 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: eorwrr_tbnz:
; CHECK:      eor  [[R:w[0-9]+]], {{w[0-9]+}}, {{w[0-9]+}}
; CHECK-NEXT: tbnz [[R]], #8, {{.?LBB[0-9_]+}}
define void @eorwrr_tbnz(i32 %a, i32 %b) {
entry:
  %v0 = xor i32 %a, %b
  %v1 = add i32 %a, 7
  %t = and i32 %v0, 256
  %cond = icmp eq i32 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: orrwri_cbz:
; CHECK:      orr  [[R:w[0-9]+]], {{w[0-9]+}}, #0xff
; CHECK-NEXT: cbz  [[R]], {{.?LBB[0-9_]+}}
define void @orrwri_cbz(i32 %a) {
entry:
  %v0 = or i32 %a, 255
  %v1 = add i32 %a, 7
  %cond = icmp ne i32 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: orrwri_cbnz:
; CHECK:      orr  [[R:w[0-9]+]], {{w[0-9]+}}, #0xff
; CHECK-NEXT: cbnz [[R]], {{.?LBB[0-9_]+}}
define void @orrwri_cbnz(i32 %a) {
entry:
  %v0 = or i32 %a, 255
  %v1 = add i32 %a, 7
  %cond = icmp eq i32 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: orrwri_tbz:
; CHECK:      orr  [[R:w[0-9]+]], {{w[0-9]+}}, #0xff
; CHECK-NEXT: tbz  [[R]], #8, {{.?LBB[0-9_]+}}
define void @orrwri_tbz(i32 %a) {
entry:
  %v0 = or i32 %a, 255
  %v1 = add i32 %a, 7
  %t = and i32 %v0, 256
  %cond = icmp ne i32 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: orrwri_tbnz:
; CHECK:      orr  [[R:w[0-9]+]], {{w[0-9]+}}, #0xff
; CHECK-NEXT: tbnz [[R]], #8, {{.?LBB[0-9_]+}}
define void @orrwri_tbnz(i32 %a) {
entry:
  %v0 = or i32 %a, 255
  %v1 = add i32 %a, 7
  %t = and i32 %v0, 256
  %cond = icmp eq i32 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: orrwrr_cbz:
; CHECK:      orr  [[R:w[0-9]+]], {{w[0-9]+}}, {{w[0-9]+}}
; CHECK-NEXT: cbz  [[R]], {{.?LBB[0-9_]+}}
define void @orrwrr_cbz(i32 %a, i32 %b) {
entry:
  %v0 = or i32 %a, %b
  %v1 = add i32 %a, 7
  %cond = icmp ne i32 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: orrwrr_cbnz:
; CHECK:      orr  [[R:w[0-9]+]], {{w[0-9]+}}, {{w[0-9]+}}
; CHECK-NEXT: cbnz [[R]], {{.?LBB[0-9_]+}}
define void @orrwrr_cbnz(i32 %a, i32 %b) {
entry:
  %v0 = or i32 %a, %b
  %v1 = add i32 %a, 7
  %cond = icmp eq i32 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: orrwrr_tbz:
; CHECK:      orr  [[R:w[0-9]+]], {{w[0-9]+}}, {{w[0-9]+}}
; CHECK-NEXT: tbz  [[R]], #8, {{.?LBB[0-9_]+}}
define void @orrwrr_tbz(i32 %a, i32 %b) {
entry:
  %v0 = or i32 %a, %b
  %v1 = add i32 %a, 7
  %t = and i32 %v0, 256
  %cond = icmp ne i32 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: orrwrr_tbnz:
; CHECK:      orr  [[R:w[0-9]+]], {{w[0-9]+}}, {{w[0-9]+}}
; CHECK-NEXT: tbnz [[R]], #8, {{.?LBB[0-9_]+}}
define void @orrwrr_tbnz(i32 %a, i32 %b) {
entry:
  %v0 = or i32 %a, %b
  %v1 = add i32 %a, 7
  %t = and i32 %v0, 256
  %cond = icmp eq i32 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: ornwrr_cbz:
; CHECK:      orn  [[R:w[0-9]+]], {{w[0-9]+}}, {{w[0-9]+}}
; CHECK-NEXT: cbz  [[R]], {{.?LBB[0-9_]+}}
define void @ornwrr_cbz(i32 %a, i32 %b) {
entry:
  %n = xor i32 %b, -1
  %v0 = or i32 %a, %n
  %v1 = add i32 %a, 7
  %cond = icmp ne i32 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: ornwrr_cbnz:
; CHECK:      orn  [[R:w[0-9]+]], {{w[0-9]+}}, {{w[0-9]+}}
; CHECK-NEXT: cbnz [[R]], {{.?LBB[0-9_]+}}
define void @ornwrr_cbnz(i32 %a, i32 %b) {
entry:
  %n = xor i32 %b, -1
  %v0 = or i32 %a, %n
  %v1 = add i32 %a, 7
  %cond = icmp eq i32 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: ornwrr_tbz:
; CHECK:      orn  [[R:w[0-9]+]], {{w[0-9]+}}, {{w[0-9]+}}
; CHECK-NEXT: tbz  [[R]], #8, {{.?LBB[0-9_]+}}
define void @ornwrr_tbz(i32 %a, i32 %b) {
entry:
  %n = xor i32 %b, -1
  %v0 = or i32 %a, %n
  %v1 = add i32 %a, 7
  %t = and i32 %v0, 256
  %cond = icmp ne i32 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: ornwrr_tbnz:
; CHECK:      orn  [[R:w[0-9]+]], {{w[0-9]+}}, {{w[0-9]+}}
; CHECK-NEXT: tbnz [[R]], #8, {{.?LBB[0-9_]+}}
define void @ornwrr_tbnz(i32 %a, i32 %b) {
entry:
  %n = xor i32 %b, -1
  %v0 = or i32 %a, %n
  %v1 = add i32 %a, 7
  %t = and i32 %v0, 256
  %cond = icmp eq i32 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: bicwrr_cbz:
; CHECK:      bic  [[R:w[0-9]+]], {{w[0-9]+}}, {{w[0-9]+}}
; CHECK-NEXT: cbz  [[R]], {{.?LBB[0-9_]+}}
define void @bicwrr_cbz(i32 %a, i32 %b) {
entry:
  %n = xor i32 %b, -1
  %v0 = and i32 %a, %n
  %v1 = add i32 %a, 7
  %cond = icmp ne i32 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: bicwrr_cbnz:
; CHECK:      bic  [[R:w[0-9]+]], {{w[0-9]+}}, {{w[0-9]+}}
; CHECK-NEXT: cbnz [[R]], {{.?LBB[0-9_]+}}
define void @bicwrr_cbnz(i32 %a, i32 %b) {
entry:
  %n = xor i32 %b, -1
  %v0 = and i32 %a, %n
  %v1 = add i32 %a, 7
  %cond = icmp eq i32 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: bicwrr_tbz:
; CHECK:      bic  [[R:w[0-9]+]], {{w[0-9]+}}, {{w[0-9]+}}
; CHECK-NEXT: tbz  [[R]], #8, {{.?LBB[0-9_]+}}
define void @bicwrr_tbz(i32 %a, i32 %b) {
entry:
  %n = xor i32 %b, -1
  %v0 = and i32 %a, %n
  %v1 = add i32 %a, 7
  %t = and i32 %v0, 256
  %cond = icmp ne i32 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: bicwrr_tbnz:
; CHECK:      bic  [[R:w[0-9]+]], {{w[0-9]+}}, {{w[0-9]+}}
; CHECK-NEXT: tbnz [[R]], #8, {{.?LBB[0-9_]+}}
define void @bicwrr_tbnz(i32 %a, i32 %b) {
entry:
  %n = xor i32 %b, -1
  %v0 = and i32 %a, %n
  %v1 = add i32 %a, 7
  %t = and i32 %v0, 256
  %cond = icmp eq i32 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: subwri_cbz:
; CHECK:      sub  [[R:w[0-9]+]], {{w[0-9]+}}, #13
; CHECK-NEXT: cbz  [[R]], {{.?LBB[0-9_]+}}
define void @subwri_cbz(i32 %a) {
entry:
  %v0 = sub i32 %a, 13
  %v1 = add i32 %a, 7
  %cond = icmp ne i32 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: subwri_cbnz:
; CHECK:      sub  [[R:w[0-9]+]], {{w[0-9]+}}, #13
; CHECK-NEXT: cbnz [[R]], {{.?LBB[0-9_]+}}
define void @subwri_cbnz(i32 %a) {
entry:
  %v0 = sub i32 %a, 13
  %v1 = add i32 %a, 7
  %cond = icmp eq i32 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: subwri_tbz:
; CHECK:      sub  [[R:w[0-9]+]], {{w[0-9]+}}, #13
; CHECK-NEXT: tbz  [[R]], #8, {{.?LBB[0-9_]+}}
define void @subwri_tbz(i32 %a) {
entry:
  %v0 = sub i32 %a, 13
  %v1 = add i32 %a, 7
  %t = and i32 %v0, 256
  %cond = icmp ne i32 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: subwri_tbnz:
; CHECK:      sub  [[R:w[0-9]+]], {{w[0-9]+}}, #13
; CHECK-NEXT: tbnz [[R]], #8, {{.?LBB[0-9_]+}}
define void @subwri_tbnz(i32 %a) {
entry:
  %v0 = sub i32 %a, 13
  %v1 = add i32 %a, 7
  %t = and i32 %v0, 256
  %cond = icmp eq i32 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: subwrr_cbz:
; CHECK:      sub  [[R:w[0-9]+]], {{w[0-9]+}}, {{w[0-9]+}}
; CHECK-NEXT: cbz  [[R]], {{.?LBB[0-9_]+}}
define void @subwrr_cbz(i32 %a, i32 %b) {
entry:
  %v0 = sub i32 %a, %b
  %v1 = add i32 %a, 7
  %cond = icmp ne i32 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: subwrr_cbnz:
; CHECK:      sub  [[R:w[0-9]+]], {{w[0-9]+}}, {{w[0-9]+}}
; CHECK-NEXT: cbnz [[R]], {{.?LBB[0-9_]+}}
define void @subwrr_cbnz(i32 %a, i32 %b) {
entry:
  %v0 = sub i32 %a, %b
  %v1 = add i32 %a, 7
  %cond = icmp eq i32 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: subwrr_tbz:
; CHECK:      sub  [[R:w[0-9]+]], {{w[0-9]+}}, {{w[0-9]+}}
; CHECK-NEXT: tbz  [[R]], #8, {{.?LBB[0-9_]+}}
define void @subwrr_tbz(i32 %a, i32 %b) {
entry:
  %v0 = sub i32 %a, %b
  %v1 = add i32 %a, 7
  %t = and i32 %v0, 256
  %cond = icmp ne i32 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: subwrr_tbnz:
; CHECK:      sub  [[R:w[0-9]+]], {{w[0-9]+}}, {{w[0-9]+}}
; CHECK-NEXT: tbnz [[R]], #8, {{.?LBB[0-9_]+}}
define void @subwrr_tbnz(i32 %a, i32 %b) {
entry:
  %v0 = sub i32 %a, %b
  %v1 = add i32 %a, 7
  %t = and i32 %v0, 256
  %cond = icmp eq i32 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: addxri_cbz:
; CHECK:      add  [[R:x[0-9]+]], {{x[0-9]+}}, #13
; CHECK-NEXT: cbz  [[R]], {{.?LBB[0-9_]+}}
define void @addxri_cbz(i64 %a) {
entry:
  %v0 = add i64 %a, 13
  %v1 = add i64 %a, 7
  %cond = icmp ne i64 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: addxri_cbnz:
; CHECK:      add  [[R:x[0-9]+]], {{x[0-9]+}}, #13
; CHECK-NEXT: cbnz [[R]], {{.?LBB[0-9_]+}}
define void @addxri_cbnz(i64 %a) {
entry:
  %v0 = add i64 %a, 13
  %v1 = add i64 %a, 7
  %cond = icmp eq i64 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: addxri_tbz:
; CHECK:      add  [[R:x[0-9]+]], {{x[0-9]+}}, #13
; CHECK-NEXT: tbz  [[R]], #32, {{.?LBB[0-9_]+}}
define void @addxri_tbz(i64 %a) {
entry:
  %v0 = add i64 %a, 13
  %v1 = add i64 %a, 7
  %t = and i64 %v0, 4294967296
  %cond = icmp ne i64 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: addxri_tbnz:
; CHECK:      add  [[R:x[0-9]+]], {{x[0-9]+}}, #13
; CHECK-NEXT: tbnz [[R]], #32, {{.?LBB[0-9_]+}}
define void @addxri_tbnz(i64 %a) {
entry:
  %v0 = add i64 %a, 13
  %v1 = add i64 %a, 7
  %t = and i64 %v0, 4294967296
  %cond = icmp eq i64 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: addxrr_cbz:
; CHECK:      add  [[R:x[0-9]+]], {{x[0-9]+}}, {{x[0-9]+}}
; CHECK-NEXT: cbz  [[R]], {{.?LBB[0-9_]+}}
define void @addxrr_cbz(i64 %a, i64 %b) {
entry:
  %v0 = add i64 %a, %b
  %v1 = add i64 %a, 7
  %cond = icmp ne i64 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: addxrr_cbnz:
; CHECK:      add  [[R:x[0-9]+]], {{x[0-9]+}}, {{x[0-9]+}}
; CHECK-NEXT: cbnz [[R]], {{.?LBB[0-9_]+}}
define void @addxrr_cbnz(i64 %a, i64 %b) {
entry:
  %v0 = add i64 %a, %b
  %v1 = add i64 %a, 7
  %cond = icmp eq i64 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: addxrr_tbz:
; CHECK:      add  [[R:x[0-9]+]], {{x[0-9]+}}, {{x[0-9]+}}
; CHECK-NEXT: tbz  [[R]], #32, {{.?LBB[0-9_]+}}
define void @addxrr_tbz(i64 %a, i64 %b) {
entry:
  %v0 = add i64 %a, %b
  %v1 = add i64 %a, 7
  %t = and i64 %v0, 4294967296
  %cond = icmp ne i64 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: addxrr_tbnz:
; CHECK:      add  [[R:x[0-9]+]], {{x[0-9]+}}, {{x[0-9]+}}
; CHECK-NEXT: tbnz [[R]], #32, {{.?LBB[0-9_]+}}
define void @addxrr_tbnz(i64 %a, i64 %b) {
entry:
  %v0 = add i64 %a, %b
  %v1 = add i64 %a, 7
  %t = and i64 %v0, 4294967296
  %cond = icmp eq i64 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: andxri_cbz:
; CHECK:      and  [[R:x[0-9]+]], {{x[0-9]+}}, #0xff
; CHECK-NEXT: cbz  [[R]], {{.?LBB[0-9_]+}}
define void @andxri_cbz(i64 %a) {
entry:
  %v0 = and i64 %a, 255
  %v1 = add i64 %a, 7
  %cond = icmp ne i64 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: andxri_cbnz:
; CHECK:      and  [[R:x[0-9]+]], {{x[0-9]+}}, #0xff
; CHECK-NEXT: cbnz [[R]], {{.?LBB[0-9_]+}}
define void @andxri_cbnz(i64 %a) {
entry:
  %v0 = and i64 %a, 255
  %v1 = add i64 %a, 7
  %cond = icmp eq i64 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; NOTE: no andxri_tbz/andxri_tbnz: a single-bit test of an immediate-AND result folds
; through the AND, so no fused ANDri+TB(N)Z pair is expressible at the IR level.
; The .mir test covers ANDXri + TB(N)Z directly.

; CHECK-LABEL: andxrr_cbz:
; CHECK:      and  [[R:x[0-9]+]], {{x[0-9]+}}, {{x[0-9]+}}
; CHECK-NEXT: cbz  [[R]], {{.?LBB[0-9_]+}}
define void @andxrr_cbz(i64 %a, i64 %b) {
entry:
  %v0 = and i64 %a, %b
  %v1 = add i64 %a, 7
  %cond = icmp ne i64 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: andxrr_cbnz:
; CHECK:      and  [[R:x[0-9]+]], {{x[0-9]+}}, {{x[0-9]+}}
; CHECK-NEXT: cbnz [[R]], {{.?LBB[0-9_]+}}
define void @andxrr_cbnz(i64 %a, i64 %b) {
entry:
  %v0 = and i64 %a, %b
  %v1 = add i64 %a, 7
  %cond = icmp eq i64 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: andxrr_tbz:
; CHECK:      and  [[R:x[0-9]+]], {{x[0-9]+}}, {{x[0-9]+}}
; CHECK-NEXT: tbz  [[R]], #32, {{.?LBB[0-9_]+}}
define void @andxrr_tbz(i64 %a, i64 %b) {
entry:
  %v0 = and i64 %a, %b
  %v1 = add i64 %a, 7
  %t = and i64 %v0, 4294967296
  %cond = icmp ne i64 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: andxrr_tbnz:
; CHECK:      and  [[R:x[0-9]+]], {{x[0-9]+}}, {{x[0-9]+}}
; CHECK-NEXT: tbnz [[R]], #32, {{.?LBB[0-9_]+}}
define void @andxrr_tbnz(i64 %a, i64 %b) {
entry:
  %v0 = and i64 %a, %b
  %v1 = add i64 %a, 7
  %t = and i64 %v0, 4294967296
  %cond = icmp eq i64 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: eorxri_cbz:
; CHECK:      eor  [[R:x[0-9]+]], {{x[0-9]+}}, #0xff
; CHECK-NEXT: cbz  [[R]], {{.?LBB[0-9_]+}}
define void @eorxri_cbz(i64 %a) {
entry:
  %v0 = xor i64 %a, 255
  %v1 = add i64 %a, 7
  %cond = icmp ne i64 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: eorxri_cbnz:
; CHECK:      eor  [[R:x[0-9]+]], {{x[0-9]+}}, #0xff
; CHECK-NEXT: cbnz [[R]], {{.?LBB[0-9_]+}}
define void @eorxri_cbnz(i64 %a) {
entry:
  %v0 = xor i64 %a, 255
  %v1 = add i64 %a, 7
  %cond = icmp eq i64 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: eorxri_tbz:
; CHECK:      eor  [[R:x[0-9]+]], {{x[0-9]+}}, #0xff
; CHECK-NEXT: tbz  [[R]], #32, {{.?LBB[0-9_]+}}
define void @eorxri_tbz(i64 %a) {
entry:
  %v0 = xor i64 %a, 255
  %v1 = add i64 %a, 7
  %t = and i64 %v0, 4294967296
  %cond = icmp ne i64 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: eorxri_tbnz:
; CHECK:      eor  [[R:x[0-9]+]], {{x[0-9]+}}, #0xff
; CHECK-NEXT: tbnz [[R]], #32, {{.?LBB[0-9_]+}}
define void @eorxri_tbnz(i64 %a) {
entry:
  %v0 = xor i64 %a, 255
  %v1 = add i64 %a, 7
  %t = and i64 %v0, 4294967296
  %cond = icmp eq i64 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: eorxrr_cbz:
; CHECK:      eor  [[R:x[0-9]+]], {{x[0-9]+}}, {{x[0-9]+}}
; CHECK-NEXT: cbz  [[R]], {{.?LBB[0-9_]+}}
define void @eorxrr_cbz(i64 %a, i64 %b) {
entry:
  %v0 = xor i64 %a, %b
  %v1 = add i64 %a, 7
  %cond = icmp ne i64 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: eorxrr_cbnz:
; CHECK:      eor  [[R:x[0-9]+]], {{x[0-9]+}}, {{x[0-9]+}}
; CHECK-NEXT: cbnz [[R]], {{.?LBB[0-9_]+}}
define void @eorxrr_cbnz(i64 %a, i64 %b) {
entry:
  %v0 = xor i64 %a, %b
  %v1 = add i64 %a, 7
  %cond = icmp eq i64 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: eorxrr_tbz:
; CHECK:      eor  [[R:x[0-9]+]], {{x[0-9]+}}, {{x[0-9]+}}
; CHECK-NEXT: tbz  [[R]], #32, {{.?LBB[0-9_]+}}
define void @eorxrr_tbz(i64 %a, i64 %b) {
entry:
  %v0 = xor i64 %a, %b
  %v1 = add i64 %a, 7
  %t = and i64 %v0, 4294967296
  %cond = icmp ne i64 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: eorxrr_tbnz:
; CHECK:      eor  [[R:x[0-9]+]], {{x[0-9]+}}, {{x[0-9]+}}
; CHECK-NEXT: tbnz [[R]], #32, {{.?LBB[0-9_]+}}
define void @eorxrr_tbnz(i64 %a, i64 %b) {
entry:
  %v0 = xor i64 %a, %b
  %v1 = add i64 %a, 7
  %t = and i64 %v0, 4294967296
  %cond = icmp eq i64 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: orrxri_cbz:
; CHECK:      orr  [[R:x[0-9]+]], {{x[0-9]+}}, #0xff
; CHECK-NEXT: cbz  [[R]], {{.?LBB[0-9_]+}}
define void @orrxri_cbz(i64 %a) {
entry:
  %v0 = or i64 %a, 255
  %v1 = add i64 %a, 7
  %cond = icmp ne i64 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: orrxri_cbnz:
; CHECK:      orr  [[R:x[0-9]+]], {{x[0-9]+}}, #0xff
; CHECK-NEXT: cbnz [[R]], {{.?LBB[0-9_]+}}
define void @orrxri_cbnz(i64 %a) {
entry:
  %v0 = or i64 %a, 255
  %v1 = add i64 %a, 7
  %cond = icmp eq i64 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: orrxri_tbz:
; CHECK:      orr  [[R:x[0-9]+]], {{x[0-9]+}}, #0xff
; CHECK-NEXT: tbz  [[R]], #32, {{.?LBB[0-9_]+}}
define void @orrxri_tbz(i64 %a) {
entry:
  %v0 = or i64 %a, 255
  %v1 = add i64 %a, 7
  %t = and i64 %v0, 4294967296
  %cond = icmp ne i64 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: orrxri_tbnz:
; CHECK:      orr  [[R:x[0-9]+]], {{x[0-9]+}}, #0xff
; CHECK-NEXT: tbnz [[R]], #32, {{.?LBB[0-9_]+}}
define void @orrxri_tbnz(i64 %a) {
entry:
  %v0 = or i64 %a, 255
  %v1 = add i64 %a, 7
  %t = and i64 %v0, 4294967296
  %cond = icmp eq i64 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: orrxrr_cbz:
; CHECK:      orr  [[R:x[0-9]+]], {{x[0-9]+}}, {{x[0-9]+}}
; CHECK-NEXT: cbz  [[R]], {{.?LBB[0-9_]+}}
define void @orrxrr_cbz(i64 %a, i64 %b) {
entry:
  %v0 = or i64 %a, %b
  %v1 = add i64 %a, 7
  %cond = icmp ne i64 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: orrxrr_cbnz:
; CHECK:      orr  [[R:x[0-9]+]], {{x[0-9]+}}, {{x[0-9]+}}
; CHECK-NEXT: cbnz [[R]], {{.?LBB[0-9_]+}}
define void @orrxrr_cbnz(i64 %a, i64 %b) {
entry:
  %v0 = or i64 %a, %b
  %v1 = add i64 %a, 7
  %cond = icmp eq i64 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: orrxrr_tbz:
; CHECK:      orr  [[R:x[0-9]+]], {{x[0-9]+}}, {{x[0-9]+}}
; CHECK-NEXT: tbz  [[R]], #32, {{.?LBB[0-9_]+}}
define void @orrxrr_tbz(i64 %a, i64 %b) {
entry:
  %v0 = or i64 %a, %b
  %v1 = add i64 %a, 7
  %t = and i64 %v0, 4294967296
  %cond = icmp ne i64 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: orrxrr_tbnz:
; CHECK:      orr  [[R:x[0-9]+]], {{x[0-9]+}}, {{x[0-9]+}}
; CHECK-NEXT: tbnz [[R]], #32, {{.?LBB[0-9_]+}}
define void @orrxrr_tbnz(i64 %a, i64 %b) {
entry:
  %v0 = or i64 %a, %b
  %v1 = add i64 %a, 7
  %t = and i64 %v0, 4294967296
  %cond = icmp eq i64 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: ornxrr_cbz:
; CHECK:      orn  [[R:x[0-9]+]], {{x[0-9]+}}, {{x[0-9]+}}
; CHECK-NEXT: cbz  [[R]], {{.?LBB[0-9_]+}}
define void @ornxrr_cbz(i64 %a, i64 %b) {
entry:
  %n = xor i64 %b, -1
  %v0 = or i64 %a, %n
  %v1 = add i64 %a, 7
  %cond = icmp ne i64 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: ornxrr_cbnz:
; CHECK:      orn  [[R:x[0-9]+]], {{x[0-9]+}}, {{x[0-9]+}}
; CHECK-NEXT: cbnz [[R]], {{.?LBB[0-9_]+}}
define void @ornxrr_cbnz(i64 %a, i64 %b) {
entry:
  %n = xor i64 %b, -1
  %v0 = or i64 %a, %n
  %v1 = add i64 %a, 7
  %cond = icmp eq i64 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: ornxrr_tbz:
; CHECK:      orn  [[R:x[0-9]+]], {{x[0-9]+}}, {{x[0-9]+}}
; CHECK-NEXT: tbz  [[R]], #32, {{.?LBB[0-9_]+}}
define void @ornxrr_tbz(i64 %a, i64 %b) {
entry:
  %n = xor i64 %b, -1
  %v0 = or i64 %a, %n
  %v1 = add i64 %a, 7
  %t = and i64 %v0, 4294967296
  %cond = icmp ne i64 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: ornxrr_tbnz:
; CHECK:      orn  [[R:x[0-9]+]], {{x[0-9]+}}, {{x[0-9]+}}
; CHECK-NEXT: tbnz [[R]], #32, {{.?LBB[0-9_]+}}
define void @ornxrr_tbnz(i64 %a, i64 %b) {
entry:
  %n = xor i64 %b, -1
  %v0 = or i64 %a, %n
  %v1 = add i64 %a, 7
  %t = and i64 %v0, 4294967296
  %cond = icmp eq i64 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: bicxrr_cbz:
; CHECK:      bic  [[R:x[0-9]+]], {{x[0-9]+}}, {{x[0-9]+}}
; CHECK-NEXT: cbz  [[R]], {{.?LBB[0-9_]+}}
define void @bicxrr_cbz(i64 %a, i64 %b) {
entry:
  %n = xor i64 %b, -1
  %v0 = and i64 %a, %n
  %v1 = add i64 %a, 7
  %cond = icmp ne i64 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: bicxrr_cbnz:
; CHECK:      bic  [[R:x[0-9]+]], {{x[0-9]+}}, {{x[0-9]+}}
; CHECK-NEXT: cbnz [[R]], {{.?LBB[0-9_]+}}
define void @bicxrr_cbnz(i64 %a, i64 %b) {
entry:
  %n = xor i64 %b, -1
  %v0 = and i64 %a, %n
  %v1 = add i64 %a, 7
  %cond = icmp eq i64 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: bicxrr_tbz:
; CHECK:      bic  [[R:x[0-9]+]], {{x[0-9]+}}, {{x[0-9]+}}
; CHECK-NEXT: tbz  [[R]], #32, {{.?LBB[0-9_]+}}
define void @bicxrr_tbz(i64 %a, i64 %b) {
entry:
  %n = xor i64 %b, -1
  %v0 = and i64 %a, %n
  %v1 = add i64 %a, 7
  %t = and i64 %v0, 4294967296
  %cond = icmp ne i64 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: bicxrr_tbnz:
; CHECK:      bic  [[R:x[0-9]+]], {{x[0-9]+}}, {{x[0-9]+}}
; CHECK-NEXT: tbnz [[R]], #32, {{.?LBB[0-9_]+}}
define void @bicxrr_tbnz(i64 %a, i64 %b) {
entry:
  %n = xor i64 %b, -1
  %v0 = and i64 %a, %n
  %v1 = add i64 %a, 7
  %t = and i64 %v0, 4294967296
  %cond = icmp eq i64 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: subxri_cbz:
; CHECK:      sub  [[R:x[0-9]+]], {{x[0-9]+}}, #13
; CHECK-NEXT: cbz  [[R]], {{.?LBB[0-9_]+}}
define void @subxri_cbz(i64 %a) {
entry:
  %v0 = sub i64 %a, 13
  %v1 = add i64 %a, 7
  %cond = icmp ne i64 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: subxri_cbnz:
; CHECK:      sub  [[R:x[0-9]+]], {{x[0-9]+}}, #13
; CHECK-NEXT: cbnz [[R]], {{.?LBB[0-9_]+}}
define void @subxri_cbnz(i64 %a) {
entry:
  %v0 = sub i64 %a, 13
  %v1 = add i64 %a, 7
  %cond = icmp eq i64 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: subxri_tbz:
; CHECK:      sub  [[R:x[0-9]+]], {{x[0-9]+}}, #13
; CHECK-NEXT: tbz  [[R]], #32, {{.?LBB[0-9_]+}}
define void @subxri_tbz(i64 %a) {
entry:
  %v0 = sub i64 %a, 13
  %v1 = add i64 %a, 7
  %t = and i64 %v0, 4294967296
  %cond = icmp ne i64 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: subxri_tbnz:
; CHECK:      sub  [[R:x[0-9]+]], {{x[0-9]+}}, #13
; CHECK-NEXT: tbnz [[R]], #32, {{.?LBB[0-9_]+}}
define void @subxri_tbnz(i64 %a) {
entry:
  %v0 = sub i64 %a, 13
  %v1 = add i64 %a, 7
  %t = and i64 %v0, 4294967296
  %cond = icmp eq i64 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: subxrr_cbz:
; CHECK:      sub  [[R:x[0-9]+]], {{x[0-9]+}}, {{x[0-9]+}}
; CHECK-NEXT: cbz  [[R]], {{.?LBB[0-9_]+}}
define void @subxrr_cbz(i64 %a, i64 %b) {
entry:
  %v0 = sub i64 %a, %b
  %v1 = add i64 %a, 7
  %cond = icmp ne i64 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: subxrr_cbnz:
; CHECK:      sub  [[R:x[0-9]+]], {{x[0-9]+}}, {{x[0-9]+}}
; CHECK-NEXT: cbnz [[R]], {{.?LBB[0-9_]+}}
define void @subxrr_cbnz(i64 %a, i64 %b) {
entry:
  %v0 = sub i64 %a, %b
  %v1 = add i64 %a, 7
  %cond = icmp eq i64 %v0, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: subxrr_tbz:
; CHECK:      sub  [[R:x[0-9]+]], {{x[0-9]+}}, {{x[0-9]+}}
; CHECK-NEXT: tbz  [[R]], #32, {{.?LBB[0-9_]+}}
define void @subxrr_tbz(i64 %a, i64 %b) {
entry:
  %v0 = sub i64 %a, %b
  %v1 = add i64 %a, 7
  %t = and i64 %v0, 4294967296
  %cond = icmp ne i64 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: subxrr_tbnz:
; CHECK:      sub  [[R:x[0-9]+]], {{x[0-9]+}}, {{x[0-9]+}}
; CHECK-NEXT: tbnz [[R]], #32, {{.?LBB[0-9_]+}}
define void @subxrr_tbnz(i64 %a, i64 %b) {
entry:
  %v0 = sub i64 %a, %b
  %v1 = add i64 %a, 7
  %t = and i64 %v0, 4294967296
  %cond = icmp eq i64 %t, 0
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}
