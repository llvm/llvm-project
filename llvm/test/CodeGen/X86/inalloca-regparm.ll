; RUN: llc -mtriple=i686-windows-msvc < %s -o /dev/null
; RUN: not llc -mtriple=x86_64-windows-msvc %s -o /dev/null 2>&1 | FileCheck %s

; This will compile successfully on x86 but not x86_64, because %b will become a
; register parameter.

declare x86_thiscallcc i32 @f(i32 %a, i32* inalloca %b)
define void @g() {
  %b = alloca inalloca i32
  store i32 2, i32* %b
  call x86_thiscallcc i32 @f(i32 0, i32* inalloca %b)
  ret void
}

; CHECK: cannot use inalloca attribute on a register parameter
