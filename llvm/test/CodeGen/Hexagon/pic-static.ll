; RUN: llc -mtriple=hexagon-- -mcpu=hexagonv5 -relocation-model=pic < %s | FileCheck %s

; CHECK-DAG: r{{[0-9]+}} = add({{pc|PC}},##_GLOBAL_OFFSET_TABLE_@PCREL)
; CHECK-DAG: r{{[0-9]+}} = add({{pc|PC}},##x@PCREL)
; CHECK: r{{[0-9]+}} = memw(r{{[0-9]+}}+##bar@GOT)

@x = internal global i32 9, align 4
@bar = external global i32*

define i32 @foo(i32 %y) nounwind {
entry:
  store i32* @x, i32** @bar, align 4, !tbaa !0
  %0 = load i32, i32* @x, align 4, !tbaa !3
  %add = add nsw i32 %0, %y
  ret i32 %add
}

!0 = !{!"any pointer", !1}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
!3 = !{!"int", !1}
