; RUN: llvm-dis < %s.bc | FileCheck %s
; RUN: verify-uselistorder < %s.bc

; CHECK: define void @foo() !dbg [[SP:![0-9]+]]
define void @foo() {
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}

!llvm.dbg.cu = !{!1}
!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, subprograms: !{!3}, emissionKind: 1)
!2 = !DIFile(filename: "foo.c", directory: "/path/to/dir")
; CHECK: [[SP]] = distinct !DISubprogram
!3 = distinct !DISubprogram(file: !2, scope: !2, line: 51, name: "foo", function: void ()* @foo, type: !4)
!4 = !DISubroutineType(types: !{})
