; RUN: not opt -S -verify < %s 2>&1 | FileCheck %s

define void @alloca() {
; CHECK: error: Cannot allocate unsized type
  %a = alloca { i32, <vscale x 1 x i32> }
  ret void
}
