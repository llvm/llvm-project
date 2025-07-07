; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

target triple = "nvptx"

define void @foo(ptr %ptr) {
; CHECK: prototype_0 : .callprototype ()_ ()
  tail call void %ptr()
  ret void
}
