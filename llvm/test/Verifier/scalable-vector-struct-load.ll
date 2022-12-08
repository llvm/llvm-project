; RUN: not opt -S -passes=verify < %s 2>&1 | FileCheck %s

define <vscale x 1 x i32> @load({ i32, <vscale x 1 x i32> }* %x) {
; CHECK: error: loading unsized types is not allowed
  %a = load { i32, <vscale x 1 x i32> }, { i32, <vscale x 1 x i32> }* %x
  %b = extractvalue { i32, <vscale x 1 x i32> } %a, 1
  ret <vscale x 1 x i32> %b
}
