; RUN: not llvm-as -disable-output <%s 2>&1 | FileCheck %s

define void @foo() {
entry:
  br label %exit, !dbg !DILocation(scope: !1, inlinedAt: !{})
; CHECK: inlined-at should be a location
; CHECK-NEXT: !{{[0-9]+}} = !DILocation(line: 0, scope: !{{[0-9]+}}, inlinedAt: ![[IA:[0-9]+]])
; CHECK-NEXT: ![[IA]] = !{}

exit:
  ret void, !dbg !{}
; CHECK: invalid !dbg metadata attachment
; CHECK-NEXT: ret void, !dbg ![[LOC:[0-9]+]]
; CHECK-NEXT: ![[LOC]] = !{}
}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DISubprogram()
