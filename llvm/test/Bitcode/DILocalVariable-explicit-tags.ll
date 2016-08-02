; Bitcode compiled with 3.7_rc2.  3.7 had redundant (but mandatory) tag fields
; on DILocalVariable.
;
; RUN: llvm-dis < %s.bc -o - | llvm-as | llvm-dis | FileCheck %s

; CHECK: ![[SP:[0-9]+]] = distinct !DISubprogram(name: "foo",{{.*}} variables: ![[VARS:[0-9]+]]
; CHECK: ![[VARS]] = !{![[PARAM:[0-9]+]], ![[AUTO:[0-9]+]]}
; CHECK: ![[PARAM]] = !DILocalVariable(name: "param", arg: 1, scope: ![[SP]])
; CHECK: ![[AUTO]]  = !DILocalVariable(name: "auto", scope: ![[SP]])

!named = !{!0}

!0 = distinct !DISubprogram(name: "foo", variables: !1)
!1 = !{!2, !3}
!2 = !DILocalVariable(tag: DW_TAG_arg_variable, name: "param", arg: 1, scope: !0)
!3 = !DILocalVariable(tag: DW_TAG_auto_variable, name: "auto", scope: !0)
