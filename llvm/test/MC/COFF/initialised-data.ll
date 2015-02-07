; RUN: llc -mtriple i686-windows %s -o - | FileCheck %s
; RUN: llc -mtriple x86_64-windows %s -o - | FileCheck %s

@data = dllexport constant [5 x i8] c"data\00", align 1

; CHECK: .section	.rdata,"dr"

