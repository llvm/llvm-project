; RUN: llc -march=hexagon -mcpu=hexagonv5 -relocation-model=pic < %s | FileCheck %s

; CHECK: r{{[0-9]+}} = add({{pc|PC}}, ##_GLOBAL_OFFSET_TABLE_@PCREL)
; CHECK: r{{[0-9]+}} = memw(r{{[0-9]+}}{{.*}}+{{.*}}##src@GOT)
; CHECK: r{{[0-9]+}} = memw(r{{[0-9]+}}{{.*}}+{{.*}}##dst@GOT)

@dst = external global i32
@src = external global i32

define i32 @foo() nounwind {
entry:
  %0 = load i32, i32* @src, align 4, !tbaa !0
  store i32 %0, i32* @dst, align 4, !tbaa !0
  %call = tail call i32 @baz(i32 %0) nounwind
  ret i32 0
}

declare i32 @baz(i32)

!0 = !{!"int", !1}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
