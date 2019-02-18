; RUN: llvm-dis < %s.bc | FileCheck %s
; Check that subprogram definitions are correctly upgraded to 'distinct'.
; Bitcode compiled with llvm-as version 3.7.

define void @f() !dbg !3 { ret void }

!llvm.module.flags = !{!4}
!llvm.dbg.cu = !{!0}
!0 = distinct !DICompileUnit(language: 12, file: !1, subprograms: !{!3})
!1 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")

; CHECK: = distinct !DISubprogram({{.*}} DISPFlagDefinition
!3 = !DISubprogram(name: "foo", isDefinition: true)
!4 = !{i32 2, !"Debug Info Version", i32 3}
