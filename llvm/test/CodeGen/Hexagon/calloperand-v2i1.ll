;RUN: llc -mtriple=hexagon  < %s | FileCheck %s

; Check that v2i1 type is promoted to v2i32.
; CHECK: call f
; CHECK: r{{[0-9]+}}:{{[0-9]+}} = memd(r29+#8)

define  <2 x i1> @test(<2 x i1> %1) {
Entry:
   %2 = call <2 x i1> @f(<2 x i1> %1)
  ret <2 x i1> %2

  }

declare <2 x i1> @f(<2 x i1>)
