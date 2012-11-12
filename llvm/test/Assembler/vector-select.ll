; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; Rudimentary test of select on vectors returning vector of bool

; CHECK: @foo
; CHECK: select <4 x i1> %cond, <4 x i32> %a, <4 x i32> %b
define <4 x i32> @foo(<4 x i32> %a, <4 x i32> %b, <4 x i1> %cond) nounwind  {
entry:
  %cmp = select <4 x i1>  %cond, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %cmp
}

