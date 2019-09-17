; RUN: llc -O0 < %s | FileCheck %s

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define dso_local i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %a = alloca fp128, align 16
  %b = alloca fp128, align 16
  %c = alloca fp128, align 16
  %d = alloca fp128, align 16
  %e = alloca fp128, align 16
  %f = alloca fp128, align 16
  %g = alloca fp128, align 16
  %h = alloca float, align 4
  %i = alloca float, align 4
  %j = alloca float, align 4
  %k = alloca float, align 4
  %l = alloca double, align 8
  %m = alloca double, align 8
  %n = alloca double, align 8
  %o = alloca double, align 8
  store i32 0, i32* %retval, align 4

  store fp128 0xLBB2C11D0AE2E087D73E717A35985531C, fp128* %a, align 16
  store fp128 0xLBB2C11D0AE2E087D73E717A35985531C, fp128* %b, align 16
  store fp128 0xL00000000000000004002000000000000, fp128* %c, align 16
  store fp128 0xL00000000000000007FFF800000000000, fp128* %d, align 16
  store fp128 0xL00000000000000007FFF000000000000, fp128* %e, align 16
  store fp128 0xL00000000000000007FFF000000000000, fp128* %f, align 16
  store fp128 0xL10000000000000003F66244CE242C556, fp128* %g, align 16
  store float 0x3E212E0BE0000000, float* %h, align 4
  store float 8.000000e+00, float* %i, align 4
  store float 0x7FF8000000000000, float* %j, align 4
  store float 0x7FF0000000000000, float* %k, align 4
  store double 1.000000e+00, double* %l, align 8
  store double 8.000000e+00, double* %m, align 8
  store double 0x7FF8000000000000, double* %n, align 8
  store double 0x7FF0000000000000, double* %o, align 8

  ret i32 0
}

attributes #0 = { "target-features"="+mmx" }

; %o
; CHECK: .globl	__real@7ff0000000000000
; CHECK: .section	.rdata,"dr",discard,__real@7ff0000000000000

; %n
; CHECK: .globl	__real@7ff8000000000000
; CHECK: .section	.rdata,"dr",discard,__real@7ff8000000000000

; %m
; CHECK: .globl	__real@4020000000000000
; CHECK: .section	.rdata,"dr",discard,__real@4020000000000000

; %l
; CHECK: .globl	__real@3ff0000000000000
; CHECK: .section	.rdata,"dr",discard,__real@3ff0000000000000

; %j
; CHECK: .globl	__real@7f800000
; CHECK: .section	.rdata,"dr",discard,__real@7f800000

; %k
; CHECK: .globl	__real@7fc00000
; CHECK: .section	.rdata,"dr",discard,__real@7fc00000

; %i
; CHECK: .globl	__real@41000000
; CHECK: .section	.rdata,"dr",discard,__real@41000000

; %h
; CHECK: .globl	__real@3109705f
; CHECK: .section	.rdata,"dr",discard,__real@3109705f

; %a, %b
; CHECK: .globl	__xmm@73e717a35985531cbb2c11d0ae2e087d
; CHECK: .section	.rdata,"dr",discard,__xmm@73e717a35985531cbb2c11d0ae2e087d

; %c
; CHECK: .globl	__xmm@40020000000000000000000000000000
; CHECK: .section	.rdata,"dr",discard,__xmm@40020000000000000000000000000000

; %d
; CHECK: .globl	__xmm@7fff8000000000000000000000000000
; CHECK: .section	.rdata,"dr",discard,__xmm@7fff8000000000000000000000000000

; %e, %f
; CHECK: .globl	__xmm@7fff0000000000000000000000000000
; CHECK: .section	.rdata,"dr",discard,__xmm@7fff0000000000000000000000000000

; %g
; CHECK: .globl	__xmm@3f66244ce242c5561000000000000000
; CHECK: .section	.rdata,"dr",discard,__xmm@3f66244ce242c5561000000000000000
