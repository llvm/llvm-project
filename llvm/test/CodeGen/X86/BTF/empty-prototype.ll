; RUN: llc -mtriple=x86_64-linux-gnu %s -o - | FileCheck %s
; CHECK: .section .BTF,
; CHECK-NOT: BTF_KIND_FUNC
; CHECK: .section .BTF.ext,
; CHECK: .long 8 # FuncInfo
; CHECK-NEXT: .long 16 # LineInfo

define void @f() !dbg !4 {
  ret void, !dbg !7
}
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9, !10}
!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{}
!3 = !DISubroutineType(types: !2)
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !3, unit: !0, spFlags: DISPFlagDefinition)
!7 = !DILocation(line: 1, column: 1, scope: !4)
!8 = !{i32 7, !"Dwarf Version", i32 5}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 7, !"BTF", i32 1}
