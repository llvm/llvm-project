; RUN: llvm-link %s %p/visibility2.ll -S | FileCheck %s
; RUN: llvm-link %p/visibility2.ll %s -S | FileCheck %s

; The values in this file are strong, the ones in visibility2.ll are weak,
; but we should still get the visibility from them.

; Variables
; CHECK: @v1 = hidden global i32 0
@v1 = global i32 0

; CHECK: @v2 = protected  global i32 0
@v2 = global i32 0

; CHECK: @v3 = hidden global i32 0
@v3 = protected global i32 0


; Aliases
; CHECK: @a1 = hidden alias i32* @v1
@a1 = alias i32* @v1

; CHECK: @a2 = protected alias i32* @v2
@a2 = alias i32* @v2

; CHECK: @a3 = hidden alias i32* @v3
@a3 = protected alias i32* @v3


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
