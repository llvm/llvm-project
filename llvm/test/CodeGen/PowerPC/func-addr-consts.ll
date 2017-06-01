; RUN: llc < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64--linux"

@g = internal constant i8* bitcast (void ()* @f to i8*), section "gsection", align 8
@h = constant i8* bitcast (void ()* @f to i8*), section "hsection", align 8
@llvm.used = appending global [2 x i8*] [i8* bitcast (i8** @g to i8*), i8* bitcast (i8** @h to i8*)], section "llvm.metadata"

; Function Attrs: nounwind uwtable
define internal void @f() {
entry:
  ret void
}

; CHECK: .section	gsection,"aw",@progbits
; CHECK: .section	hsection,"aw",@progbits
