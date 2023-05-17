; RUN: llc < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64--linux"

@g = internal constant ptr @f, section "gsection", align 8
@h = constant ptr @f, section "hsection", align 8
@llvm.used = appending global [2 x ptr] [ptr @g, ptr @h], section "llvm.metadata"

; Function Attrs: nounwind uwtable
define internal void @f() {
entry:
  ret void
}

; CHECK: .section	gsection,"awR",@progbits
; CHECK: .section	hsection,"awR",@progbits
