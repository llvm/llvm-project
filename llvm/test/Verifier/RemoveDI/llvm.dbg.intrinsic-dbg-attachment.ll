; RUN: llvm-as -disable-output <%s 2>&1 | FileCheck %s
define void @foo() {
entry:
    #dbg_value(
      ptr undef,
      !DILocalVariable(scope: !1),
      !DIExpression(),
      !{})
; CHECK-LABEL: invalid #dbg record DILocation
; CHECK-NEXT: #dbg_value({{.*}})

    #dbg_declare(
      ptr undef,
      !DILocalVariable(scope: !1),
      !DIExpression(),
      !{})
; CHECK-LABEL: invalid #dbg record DILocation
; CHECK-NEXT: #dbg_declare({{.*}})

    #dbg_value(
      ptr undef,
      !DILocalVariable(scope: !1),
      !DIExpression(),
      !DILocation(scope: !2))
; CHECK-LABEL: mismatched subprogram between #dbg record variable and DILocation
; CHECK-NEXT: #dbg_value({{[^,]+}}, ![[VAR:[0-9]+]], {{[^,]+}}, ![[LOC:[0-9]+]]
; CHECK-NEXT: label %entry
; CHECK-NEXT: ptr @foo
; CHECK-NEXT: ![[VAR]] = !DILocalVariable({{.*}}scope: ![[VARSP:[0-9]+]]
; CHECK-NEXT: ![[VARSP]] = distinct !DISubprogram(
; CHECK-NEXT: ![[LOC]] = !DILocation({{.*}}scope: ![[LOCSP:[0-9]+]]
; CHECK-NEXT: ![[LOCSP]] = distinct !DISubprogram(

    #dbg_declare(
      ptr undef,
      !DILocalVariable(scope: !1),
      !DIExpression(),
      !DILocation(scope: !2))
; CHECK-LABEL: mismatched subprogram between #dbg record variable and DILocation
; CHECK-NEXT: #dbg_declare({{[^,]+}}, ![[VAR:[0-9]+]], {{.*[^,]+}}, ![[LOC:[0-9]+]]
; CHECK-NEXT: label %entry
; CHECK-NEXT: ptr @foo
; CHECK-NEXT: ![[VAR]] = !DILocalVariable({{.*}}scope: ![[VARSP:[0-9]+]]
; CHECK-NEXT: ![[VARSP]] = distinct !DISubprogram(
; CHECK-NEXT: ![[LOC]] = !DILocation({{.*}}scope: ![[LOCSP:[0-9]+]]
; CHECK-NEXT: ![[LOCSP]] = distinct !DISubprogram(

  ret void
}


!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DISubprogram(name: "foo")
!2 = distinct !DISubprogram(name: "bar")
