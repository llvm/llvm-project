; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

define void @foo() !dbg !4 !dbg !4 {
  unreachable
}

; CHECK-NOT:  !dbg
; CHECK:      function !dbg attachment must be a subprogram
; CHECK-NEXT: void ()* @bar
; CHECK-NEXT: !{{[0-9]+}} = !{}
define void @bar() !dbg !6 {
  unreachable
}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}

!llvm.dbg.cu = !{!1}
!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, subprograms: !3)
!2 = !DIFile(filename: "t.c", directory: "/path/to/dir")
!3 = !{!4}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !2)
!6 = !{}
