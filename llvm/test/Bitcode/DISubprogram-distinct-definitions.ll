; RUN: llvm-dis < %s.bc | FileCheck %s
; Check that subprogram definitions are encoded as 'distinct'.

define void @f() !dbg !3 { ret void }

!llvm.module.flags = !{!4}
!llvm.dbg.cu = !{!0}
!0 = distinct !DICompileUnit(language: 12, file: !1)
!1 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")

; CHECK: = distinct !DISubprogram({{.*}} type: ![[TYPE:[0-9]+]],{{.*}} DISPFlagDefinition
; CHECK: ![[TYPE]] = !DISubroutineType(types: ![[ARGS:[0-9]+]])
; CHECK: ![[ARGS]] = !{null}
!3 = distinct !DISubprogram(name: "foo", type: !5, isDefinition: true, unit: !0)
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !DISubroutineType(types: !6)
!6 = !{null}
