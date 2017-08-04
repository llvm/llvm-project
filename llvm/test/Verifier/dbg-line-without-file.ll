; RUN: not llvm-as -disable-output <%s 2>&1 | FileCheck %s
; CHECK: assembly parsed, but does not verify
; CHECK: line specified with no file

define void @foo() !dbg !3 {
  ret void
}

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C, file: !2)
!2 = !DIFile(filename: "foo.c", directory: "")
!3 = distinct !DISubprogram(name: "foo", scope: !1, line: 1, unit: !1)
