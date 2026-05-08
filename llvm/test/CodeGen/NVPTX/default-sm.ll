; RUN: llc < %s -mtriple=nvptx | FileCheck %s
; RUN: llc < %s -mtriple=nvptx64 | FileCheck %s
; RUN: %if ptxas-ptr32 %{ llc < %s -mtriple=nvptx | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 | %ptxas-verify %}

; CHECK: .target sm_75

define void @foo() {
  ret void
}
