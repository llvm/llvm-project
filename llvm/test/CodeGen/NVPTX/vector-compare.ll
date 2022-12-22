; RUN: llc < %s -march=nvptx -mcpu=sm_20 %if ptxas %{ | %ptxas-verify %}
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 %if ptxas %{ | %ptxas-verify %}

; This test makes sure that the result of vector compares are properly
; scalarized.  If codegen fails, then the type legalizer incorrectly
; tried to promote <2 x i1> to <2 x i8> and instruction selection failed.

define void @foo(ptr %a, ptr %b, ptr %r1, ptr %r2) {
  %aval = load <2 x i32>, ptr %a
  %bval = load <2 x i32>, ptr %b
  %res = icmp slt <2 x i32> %aval, %bval
  %t1 = extractelement <2 x i1> %res, i32 0
  %t2 = extractelement <2 x i1> %res, i32 1
  %t1a = zext i1 %t1 to i32
  %t2a = zext i1 %t2 to i32
  store i32 %t1a, ptr %r1
  store i32 %t2a, ptr %r2
  ret void
}
