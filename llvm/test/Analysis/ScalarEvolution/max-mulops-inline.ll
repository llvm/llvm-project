; RUN: opt -disable-output "-passes=print<scalar-evolution>" -scev-mulops-inline-threshold=1 < %s 2>&1 | FileCheck --check-prefix=CHECK1 %s
; RUN: opt -disable-output "-passes=print<scalar-evolution>" -scev-mulops-inline-threshold=10 < %s 2>&1 | FileCheck --check-prefix=CHECK10 %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = local_unnamed_addr global i32 0, align 4
@b = local_unnamed_addr global i32 0, align 4

define i32 @main() local_unnamed_addr {

; CHECK1: %mul.1 = mul nsw i32 %mul, %mul
; CHECK1: -->  ((%a.promoted * %a.promoted) * (%a.promoted * %a.promoted))

; CHECK10: %mul.1 = mul nsw i32 %mul, %mul
; CHECK10: -->  (%a.promoted * %a.promoted * %a.promoted * %a.promoted)

entry:
  %a.promoted = load i32, ptr @a, align 4
  %mul = mul nsw i32 %a.promoted, %a.promoted
  %mul.1 = mul nsw i32 %mul, %mul
  %mul.2 = mul nsw i32 %mul.1, %mul.1
  %mul.3 = mul nsw i32 %mul.2, %mul.2
  %mul.4 = mul nsw i32 %mul.3, %mul.3
  %mul.5 = mul nsw i32 %mul.4, %mul.4
  store i32 %mul.5, ptr @a, align 4
  store i32 31, ptr @b, align 4
  ret i32 0
}
