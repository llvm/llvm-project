; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; PTX does not support .hidden or .protected.
; Make sure we do not emit them.

define hidden void @f_hidden() {
       ret void
}
; CHECK-NOT: .hidden
; CHECK: .visible .func f_hidden

define protected void @f_protected() {
       ret void
}
; CHECK-NOT: .protected
; CHECK: .visible .func f_protected
