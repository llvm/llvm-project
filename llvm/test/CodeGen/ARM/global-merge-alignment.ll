; RUN: llc < %s -arm-global-merge -data-sections | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7em-arm-none-eabi"

@f = dso_local local_unnamed_addr global [4 x i32*] zeroinitializer, align 4
@d = dso_local local_unnamed_addr global i64 0, align 8

;CHECK: .section	.bss..L_MergedGlobals,"aw",%nobits
;CHECK-NEXT: .p2align	3
;CHECK-NEXT: .L_MergedGlobals:
;CHECK-NEXT: .zero	24
;CHECK-NEXT: .size	.L_MergedGlobals, 24


define dso_local i32 @func_1() {
  %1 = load i64, i64* @d, align 8
  %2 = load i32*, i32** getelementptr inbounds ([4 x i32*], [4 x i32*]* @f, i32 0, i32 0), align 4
  %3 = load i32, i32* %2, align 4
  %4 = trunc i64 %1 to i32
  %5 = add i32 %3, %4
  ret i32 %5
}

