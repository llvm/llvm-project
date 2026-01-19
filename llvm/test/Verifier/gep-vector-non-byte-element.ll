; RUN: not llvm-as < %s 2>&1 | FileCheck %s
; CHECK: GEP into vector with non-byte-addressable element type

; This testcase is invalid because we are indexing into a vector
; with non-byte-addressable element type (i1).

define <2 x i1> @test(<2 x i1> %a) {
  %a.priv = alloca <2 x i1>, align 2
  store <2 x i1> %a, ptr %a.priv, align 2
  %a.priv.0.1 = getelementptr <2 x i1>, ptr %a.priv, i64 0, i64 1
  store <2 x i1> %a, ptr %a.priv.0.1, align 2
  ret <2 x i1> zeroinitializer
}
