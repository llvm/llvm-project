; RUN: llc -o - %s -mtriple=aarch64-unknown -mcpu=cortex-a57 -mattr=+arith-bcc-fusion | FileCheck %s
; RUN: llc -o - %s -mtriple=aarch64-unknown -mcpu=cyclone | FileCheck %s
; RUN: llc -o - %s -mtriple=aarch64-unknown -mcpu=apple-m5 | FileCheck %s

target triple = "aarch64-unknown"

declare void @fi32(i32, i32)
declare void @fi64(i64, i64)

; CHECK-LABEL: subswri:
; CHECK:      cmp  {{w[0-9]+}}, #13
; CHECK-NEXT: b.{{[a-z][a-z]}}
define void @subswri(i32 %a) {
entry:
  %cond = icmp eq i32 %a, 13
  %v1 = add i32 %a, 7
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %a)
  br label %exit
exit:
  call void @fi32(i32 %a, i32 %v1)
  ret void
}

; CHECK-LABEL: subswrr:
; CHECK:      cmp  {{w[0-9]+}}, {{w[0-9]+}}
; CHECK-NEXT: b.{{[a-z][a-z]}}
define void @subswrr(i32 %a, i32 %b) {
entry:
  %cond = icmp eq i32 %a, %b
  %v1 = add i32 %a, 7
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %a)
  br label %exit
exit:
  call void @fi32(i32 %a, i32 %v1)
  ret void
}

; CHECK-LABEL: addswri:
; CHECK:      cmn  {{w[0-9]+}}, #13
; CHECK-NEXT: b.{{[a-z][a-z]}}
define void @addswri(i32 %a) {
entry:
  %cond = icmp eq i32 %a, -13
  %v1 = add i32 %a, 7
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %a)
  br label %exit
exit:
  call void @fi32(i32 %a, i32 %v1)
  ret void
}

; CHECK-LABEL: addswrr:
; CHECK:      cmn  {{w[0-9]+}}, {{w[0-9]+}}
; CHECK-NEXT: b.{{[a-z][a-z]}}
define void @addswrr(i32 %a, i32 %b) {
entry:
  %nb = sub i32 0, %b
  %cond = icmp eq i32 %a, %nb
  %v1 = add i32 %a, 7
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %a)
  br label %exit
exit:
  call void @fi32(i32 %a, i32 %v1)
  ret void
}

; CHECK-LABEL: andswri:
; CHECK:      ands {{w[0-9]+}}, {{w[0-9]+}}, #0x1000
; CHECK-NEXT: b.{{[a-z][a-z]}}
define void @andswri(i32 %a) {
entry:
  %v0 = and i32 %a, 4096
  %cond = icmp slt i32 %v0, 0
  %v1 = add i32 %a, 7
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: andswrr:
; CHECK:      ands {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK-NEXT: b.{{[a-z][a-z]}}
define void @andswrr(i32 %a, i32 %b) {
entry:
  %v0 = and i32 %a, %b
  %cond = icmp slt i32 %v0, 0
  %v1 = add i32 %a, 7
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: bicswrr:
; CHECK:      bics {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECK-NEXT: b.{{[a-z][a-z]}}
define void @bicswrr(i32 %a, i32 %b) {
entry:
  %n = xor i32 %b, -1
  %v0 = and i32 %a, %n
  %cond = icmp slt i32 %v0, 0
  %v1 = add i32 %a, 7
  br i1 %cond, label %if, label %exit
if:
  call void @fi32(i32 %v1, i32 %v0)
  br label %exit
exit:
  call void @fi32(i32 %v0, i32 %v1)
  ret void
}

; CHECK-LABEL: subsxri:
; CHECK:      cmp  {{x[0-9]+}}, #13
; CHECK-NEXT: b.{{[a-z][a-z]}}
define void @subsxri(i64 %a) {
entry:
  %cond = icmp eq i64 %a, 13
  %v1 = add i64 %a, 7
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %a)
  br label %exit
exit:
  call void @fi64(i64 %a, i64 %v1)
  ret void
}

; CHECK-LABEL: subsxrr:
; CHECK:      cmp  {{x[0-9]+}}, {{x[0-9]+}}
; CHECK-NEXT: b.{{[a-z][a-z]}}
define void @subsxrr(i64 %a, i64 %b) {
entry:
  %cond = icmp eq i64 %a, %b
  %v1 = add i64 %a, 7
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %a)
  br label %exit
exit:
  call void @fi64(i64 %a, i64 %v1)
  ret void
}

; CHECK-LABEL: addsxri:
; CHECK:      cmn  {{x[0-9]+}}, #13
; CHECK-NEXT: b.{{[a-z][a-z]}}
define void @addsxri(i64 %a) {
entry:
  %cond = icmp eq i64 %a, -13
  %v1 = add i64 %a, 7
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %a)
  br label %exit
exit:
  call void @fi64(i64 %a, i64 %v1)
  ret void
}

; CHECK-LABEL: addsxrr:
; CHECK:      cmn  {{x[0-9]+}}, {{x[0-9]+}}
; CHECK-NEXT: b.{{[a-z][a-z]}}
define void @addsxrr(i64 %a, i64 %b) {
entry:
  %nb = sub i64 0, %b
  %cond = icmp eq i64 %a, %nb
  %v1 = add i64 %a, 7
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %a)
  br label %exit
exit:
  call void @fi64(i64 %a, i64 %v1)
  ret void
}

; CHECK-LABEL: andsxri:
; CHECK:      ands {{x[0-9]+}}, {{x[0-9]+}}, #0x2000
; CHECK-NEXT: b.{{[a-z][a-z]}}
define void @andsxri(i64 %a) {
entry:
  %v0 = and i64 %a, 8192
  %cond = icmp slt i64 %v0, 0
  %v1 = add i64 %a, 7
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: andsxrr:
; CHECK:      ands {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
; CHECK-NEXT: b.{{[a-z][a-z]}}
define void @andsxrr(i64 %a, i64 %b) {
entry:
  %v0 = and i64 %a, %b
  %cond = icmp slt i64 %v0, 0
  %v1 = add i64 %a, 7
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

; CHECK-LABEL: bicsxrr:
; CHECK:      bics {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
; CHECK-NEXT: b.{{[a-z][a-z]}}
define void @bicsxrr(i64 %a, i64 %b) {
entry:
  %n = xor i64 %b, -1
  %v0 = and i64 %a, %n
  %cond = icmp slt i64 %v0, 0
  %v1 = add i64 %a, 7
  br i1 %cond, label %if, label %exit
if:
  call void @fi64(i64 %v1, i64 %v0)
  br label %exit
exit:
  call void @fi64(i64 %v0, i64 %v1)
  ret void
}

