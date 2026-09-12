; RUN: opt < %s -passes=globalopt -S | FileCheck %s

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

; CHECK: $x
$x = comdat any

; CHECK-NOT: $trulydeadg
$trulydeadg = comdat any

; CHECK: @g
@g = internal global i32 0, comdat($x)
; CHECK: @g2
@g2 = global i32 0, comdat($x)
; CHECK: @x
@x = internal alias i32, ptr @g

; CHECK-NOT: @trulydeadg
@trulydeadg = internal alias i32, ptr @deadg
; CHECK-NOT: @deadg
@deadg = internal global i32 0, comdat($trulydeadg)