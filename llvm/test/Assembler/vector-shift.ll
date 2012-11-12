; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

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

; Constant expressions: these should be folded.

; CHECK: @foo_ce
; CHECK: ret <2 x i64> <i64 40, i64 192>
define <2 x i64> @foo_ce() nounwind {
  ret <2 x i64> shl (<2 x i64> <i64 5, i64 6>, <2 x i64> <i64 3, i64 5>)
}

; CHECK: @bar_ce
; CHECK: ret <2 x i64> <i64 42, i64 11>
define <2 x i64> @bar_ce() nounwind {
  ret <2 x i64> lshr (<2 x i64> <i64 340, i64 380>, <2 x i64> <i64 3, i64 5>)
}

; CHECK: baz_ce
; CHECK: ret <2 x i64> <i64 71, i64 12>
define <2 x i64> @baz_ce() nounwind {
  ret <2 x i64> ashr (<2 x i64> <i64 573, i64 411>, <2 x i64> <i64 3, i64 5>)
}
