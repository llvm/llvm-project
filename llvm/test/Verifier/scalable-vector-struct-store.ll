; RUN: not opt -S -passes=verify < %s 2>&1 | FileCheck %s

define void @store({ i32, <vscale x 1 x i32> }* %x, i32 %y, <vscale x 1 x i32> %z) {
; CHECK: error: storing unsized types is not allowed
  %a = insertvalue { i32, <vscale x 1 x i32> } undef, i32 %y, 0
  %b = insertvalue { i32, <vscale x 1 x i32> } %a,  <vscale x 1 x i32> %z, 1
  store { i32, <vscale x 1 x i32> } %b, { i32, <vscale x 1 x i32> }* %x
  ret void
}
