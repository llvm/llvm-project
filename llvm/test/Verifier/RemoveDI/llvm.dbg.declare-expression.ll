; RUN: llvm-as -disable-output <%s 2>&1 | FileCheck %s
; CHECK: invalid #dbg record expression
; CHECK-NEXT: #dbg_declare({{.*}})
; CHECK-NEXT: !{}
; CHECK: warning: ignoring invalid debug info

define void @foo(i32 %a) {
entry:
  %s = alloca i32
    #dbg_declare(ptr %s, !DILocalVariable(scope: !1), !{}, !DILocation(scope: !1))
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DISubprogram()
