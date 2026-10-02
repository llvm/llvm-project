; RUN: not llvm-as -disable-output <%s 2>&1 | FileCheck %s

define void @foo() {
  ret void, !dbg !{}
; CHECK: invalid !dbg metadata
}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DISubprogram(type: !3)
!2 = !{null}
!3 = !DISubroutineType(types: !2)
