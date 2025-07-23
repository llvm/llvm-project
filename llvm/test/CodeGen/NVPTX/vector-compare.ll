; RUN: llc < %s -mtriple=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas && !ptxas-12.0 %{ llc < %s -mtriple=nvptx -mcpu=sm_20 | %ptxas-verify -m32 %}
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; This test makes sure that the result of vector compares are properly
; scalarized.  If codegen fails, then the type legalizer incorrectly
; tried to promote <2 x i1> to <2 x i8> and instruction selection failed.

; CHECK-LABEL: .visible .func foo(
define void @foo(ptr %a, ptr %b, ptr %r1, ptr %r2) {
; CHECK: ld.v2.b32
  %aval = load <2 x i32>, ptr %a
; CHECK: ld.v2.b32
  %bval = load <2 x i32>, ptr %b
; CHECK: setp.lt.s32
; CHECK: setp.lt.s32
  %res = icmp slt <2 x i32> %aval, %bval
  %t1 = extractelement <2 x i1> %res, i32 0
  %t2 = extractelement <2 x i1> %res, i32 1
; CHECK: selp.b32        %r{{[0-9]+}}, 1, 0
; CHECK: selp.b32        %r{{[0-9]+}}, 1, 0
  %t1a = zext i1 %t1 to i32
  %t2a = zext i1 %t2 to i32
; CHECK: st.b32
; CHECK: st.b32
  store i32 %t1a, ptr %r1
  store i32 %t2a, ptr %r2
  ret void
}
