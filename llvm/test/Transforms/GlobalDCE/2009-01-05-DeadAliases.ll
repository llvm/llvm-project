; RUN: opt < %s -passes=globaldce -S > %t
; RUN: FileCheck %s < %t
; RUN: FileCheck --check-prefix=DEAD %s < %t

@A = global i32 0
; CHECK: @A = global i32 0

@D = internal alias i32, i32* @A
; DEAD-NOT: @D

@L1 = alias i32, i32* @A
; CHECK: @L1 = alias i32, i32* @A

@L2 = internal alias i32, i32* @L1
; CHECK: @L2 = internal alias i32, i32* @L1

@L3 = alias i32, i32* @L2
; CHECK: @L3 = alias i32, i32* @L2
