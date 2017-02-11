; RUN: llc -march=hexagon -O0 < %s | FileCheck %s
; RUN: llc -march=hexagon -O0 < %s | FileCheck -check-prefix=CHECK-CALL %s
; Hexagon Programmer's Reference Manual 11.1.1 ALU32/ALU

; CHECK-CALL-NOT: call

; Add
declare i32 @llvm.hexagon.A2.addi(i32, i32)
define i32 @A2_addi(i32 %a) {
  %z = call i32 @llvm.hexagon.A2.addi(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: = add({{.*}},#0)

declare i32 @llvm.hexagon.A2.add(i32, i32)
define i32 @A2_add(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A2.add(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = add({{.*}},{{.*}})

declare i32 @llvm.hexagon.A2.addsat(i32, i32)
define i32 @A2_addsat(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A2.addsat(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = add({{.*}},{{.*}}):sat

; Logical operations
declare i32 @llvm.hexagon.A2.and(i32, i32)
define i32 @A2_and(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A2.and(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = and({{.*}},{{.*}})

declare i32 @llvm.hexagon.A2.or(i32, i32)
define i32 @A2_or(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A2.or(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = or({{.*}},{{.*}})

declare i32 @llvm.hexagon.A2.xor(i32, i32)
define i32 @A2_xor(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A2.xor(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = xor({{.*}},{{.*}})

declare i32 @llvm.hexagon.A4.andn(i32, i32)
define i32 @A4_andn(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A4.andn(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = and({{.*}},~{{.*}})

declare i32 @llvm.hexagon.A4.orn(i32, i32)
define i32 @A4_orn(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A4.orn(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = or({{.*}},~{{.*}})

; Subtract
declare i32 @llvm.hexagon.A2.sub(i32, i32)
define i32 @A2_sub(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A2.sub(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = sub({{.*}},{{.*}})

declare i32 @llvm.hexagon.A2.subsat(i32, i32)
define i32 @A2_subsat(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A2.subsat(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = sub({{.*}},{{.*}}):sat

; Sign extend
declare i32 @llvm.hexagon.A2.sxtb(i32)
define i32 @A2_sxtb(i32 %a) {
  %z = call i32 @llvm.hexagon.A2.sxtb(i32 %a)
  ret i32 %z
}
; CHECK: = sxtb({{.*}})

declare i32 @llvm.hexagon.A2.sxth(i32)
define i32 @A2_sxth(i32 %a) {
  %z = call i32 @llvm.hexagon.A2.sxth(i32 %a)
  ret i32 %z
}
; CHECK: = sxth({{.*}})

; Transfer immediate
declare i32 @llvm.hexagon.A2.tfril(i32, i32)
define i32 @A2_tfril(i32 %a) {
  %z = call i32 @llvm.hexagon.A2.tfril(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: = #0

declare i32 @llvm.hexagon.A2.tfrih(i32, i32)
define i32 @A2_tfrih(i32 %a) {
  %z = call i32 @llvm.hexagon.A2.tfrih(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: = #0

declare i32 @llvm.hexagon.A2.tfrsi(i32)
define i32 @A2_tfrsi() {
  %z = call i32 @llvm.hexagon.A2.tfrsi(i32 0)
  ret i32 %z
}
; CHECK: = #0

; Transfer register
declare i32 @llvm.hexagon.A2.tfr(i32)
define i32 @A2_tfr(i32 %a) {
  %z = call i32 @llvm.hexagon.A2.tfr(i32 %a)
  ret i32 %z
}
; CHECK: =

; Vector add halfwords
declare i32 @llvm.hexagon.A2.svaddh(i32, i32)
define i32 @A2_svaddh(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A2.svaddh(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = vaddh({{.*}},{{.*}})

declare i32 @llvm.hexagon.A2.svaddhs(i32, i32)
define i32 @A2_svaddhs(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A2.svaddhs(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = vaddh({{.*}},{{.*}}):sat

declare i32 @llvm.hexagon.A2.svadduhs(i32, i32)
define i32 @A2_svadduhs(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A2.svadduhs(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = vadduh({{.*}},{{.*}}):sat

; Vector average halfwords
declare i32 @llvm.hexagon.A2.svavgh(i32, i32)
define i32 @A2_svavgh(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A2.svavgh(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = vavgh({{.*}},{{.*}})

declare i32 @llvm.hexagon.A2.svavghs(i32, i32)
define i32 @A2_svavghs(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A2.svavghs(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = vavgh({{.*}},{{.*}}):rnd

declare i32 @llvm.hexagon.A2.svnavgh(i32, i32)
define i32 @A2_svnavgh(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A2.svnavgh(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = vnavgh({{.*}},{{.*}})

; Vector subtract halfwords
declare i32 @llvm.hexagon.A2.svsubh(i32, i32)
define i32 @A2_svsubh(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A2.svsubh(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = vsubh({{.*}},{{.*}})

declare i32 @llvm.hexagon.A2.svsubhs(i32, i32)
define i32 @A2_svsubhs(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A2.svsubhs(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = vsubh({{.*}},{{.*}}):sat

declare i32 @llvm.hexagon.A2.svsubuhs(i32, i32)
define i32 @A2_svsubuhs(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A2.svsubuhs(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = vsubuh({{.*}},{{.*}}):sat

; Zero extend
declare i32 @llvm.hexagon.A2.zxth(i32)
define i32 @A2_zxth(i32 %a) {
  %z = call i32 @llvm.hexagon.A2.zxth(i32 %a)
  ret i32 %z
}
; CHECK: = zxth({{.*}})
