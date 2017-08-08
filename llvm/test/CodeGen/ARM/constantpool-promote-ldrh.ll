; RUN: llc < %s -O0 -fast-isel=false -arm-promote-constant | FileCheck %s
; RUN: llc < %s -O0 -fast-isel=false -filetype=obj -arm-promote-constant
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv6m-arm-linux-gnueabi"

@fn1.a = private unnamed_addr constant [4 x i16] [i16 6, i16 0, i16 0, i16 0], align 2

; We must not try and emit this bad instruction: "ldrh r1, .LCPI0_0"
; CHECK-LABEL: fn1:
; CHECK: adr [[base:r[0-9]+]], .LCPI0_0
; CHECK-NOT: ldrh {{r[0-9]+}}, .LCPI0_0
; CHECK: ldrh r{{[0-9]+}}, {{\[}}[[base]]]
define hidden i32 @fn1() #0 {
entry:
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* undef, i8* bitcast ([4 x i16]* @fn1.a to i8*), i32 8, i32 2, i1 false)
  ret i32 undef
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture writeonly, i8* nocapture readonly, i32, i32, i1)
attributes #0 = { "target-features"="+strict-align" }
