; RUN: llc -O3 < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "armv7--linux-gnu"

@a = dso_local global i32 0, align 4
@b = dso_local global i32 0, align 4
@c = dso_local global i32 0, align 4

; CHECK-LABEL: bar:
; CHECK: ldm r{{[0-9]}}!, {r0, r{{[0-9]}}, r{{[0-9]}}}
define dso_local void @bar(i32 %a1, i32 %b1, i32 %c1) minsize optsize {
  %1 = load i32, i32* @a, align 4
  %2 = load i32, i32* @b, align 4
  %3 = load i32, i32* @c, align 4
  %4 = tail call i32 @baz(i32 %1, i32 %3) minsize optsize
  %5 = tail call i32 @baz(i32 %2, i32 %3) minsize optsize
  ret void
}

declare i32 @baz(i32,i32) minsize optsize
