; RUN: llc < %s -mtriple=i386-apple-darwin | FileCheck %s
; rdar://6479858

; CHECK-NOT: comm
@str1 = internal constant [1 x i8] zeroinitializer
