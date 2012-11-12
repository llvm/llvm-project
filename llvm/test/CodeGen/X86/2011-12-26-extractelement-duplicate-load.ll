; RUN: llc -march=x86-64 -mattr=-sse42,+sse41 < %s | FileCheck %s
; Make sure we don't load from the location pointed to by %p
; twice: it has non-obvious performance implications, and
; the relevant transformation doesn't know how to update
; the chains correctly.
; PR10747

; CHECK: test:
; CHECK: pextrd $2, %xmm
define <4 x i32> @test(<4 x i32>* %p) {
  %v = load <4 x i32>* %p
  %e = extractelement <4 x i32> %v, i32 2
  %cmp = icmp eq i32 %e, 3
  %sel = select i1 %cmp, <4 x i32> %v, <4 x i32> zeroinitializer
  ret <4 x i32> %sel
}
