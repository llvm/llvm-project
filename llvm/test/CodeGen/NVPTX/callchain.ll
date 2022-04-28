; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx -mcpu=sm_20 | %ptxas-verify %}

target triple = "nvptx"

define void @foo(i8* %ptr) {
  %fnptr = bitcast i8* %ptr to void ()*
; CHECK: prototype_0 : .callprototype ()_ ()
  tail call void %fnptr()
  ret void
}
