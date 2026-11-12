; RUN: llvm-link %s %p/Inputs/visibility.ll -S | FileCheck %s
; RUN: llvm-link %p/Inputs/visibility.ll %s -S | FileCheck %s

; The values in this file are strong, the ones in Inputs/visibility.ll are weak,
; but we should still get the visibility from them.


$c1 = comdat any

; Variables
; CHECK-DAG: @v1 = hidden global i32 0
@v1 = global i32 0

; CHECK-DAG: @v2 = protected  global i32 0
@v2 = global i32 0

; CHECK-DAG: @v3 = hidden global i32 0
@v3 = protected global i32 0

; CHECK-DAG: @v4 = hidden global i32 1, comdat($c1)
@v4 = global i32 1, comdat($c1)

; Aliases
; CHECK: @a1 = hidden alias i32, ptr @v1
@a1 = alias i32, ptr @v1

; CHECK: @a2 = protected alias i32, ptr @v2
@a2 = alias i32, ptr @v2

; CHECK: @a3 = hidden alias i32, ptr @v3
@a3 = protected alias i32, ptr @v3


; Functions
; CHECK: define hidden void @f1()
define void @f1()  {
entry:
  ret void
}

; CHECK: define protected void @f2()
define void @f2()  {
entry:
  ret void
}

; CHECK: define hidden void @f3()
define protected void @f3()  {
entry:
  ret void
}
