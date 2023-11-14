; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: @foo
; CHECK: shl
define <4 x i32> @foo(<4 x i32> %a, <4 x i32> %b) nounwind  {
entry:
  %cmp = shl <4 x i32> %a, %b		; <4 x i32> [#uses=1]
  ret <4 x i32> %cmp
}

; CHECK: @bar
; CHECK: lshr
define <4 x i32> @bar(<4 x i32> %a, <4 x i32> %b) nounwind  {
entry:
  %cmp = lshr <4 x i32> %a, %b		; <4 x i32> [#uses=1]
  ret <4 x i32> %cmp
}

; CHECK: @baz
; CHECK: ashr 
define <4 x i32> @baz(<4 x i32> %a, <4 x i32> %b) nounwind  {
entry:
  %cmp = ashr <4 x i32> %a, %b		; <4 x i32> [#uses=1]
  ret <4 x i32> %cmp
}
