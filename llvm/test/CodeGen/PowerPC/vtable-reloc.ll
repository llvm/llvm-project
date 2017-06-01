; RUN: llc -verify-machineinstrs -O0 < %s | FileCheck %s

target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

@_ZTV3foo = linkonce_odr unnamed_addr constant [1 x i8*] [i8* bitcast (void ()* @__cxa_pure_virtual to i8*)]
declare void @__cxa_pure_virtual()

; CHECK: .section .data.rel.ro
; CHECK: .quad __cxa_pure_virtual

