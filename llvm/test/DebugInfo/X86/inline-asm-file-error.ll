;; An unassigned file number leaves a hole in the file table. Don't emit the
;; line table after the error.
; RUN: not llc -mtriple=x86_64 -filetype=obj %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: warning: inconsistent use of MD5 checksums
; CHECK: error: unassigned file number: 1 for .file directives
; CHECK: error: unassigned file number: 2 for .file directives

define void @f() !dbg !4 {
  call void asm sideeffect ".file 3 \22b\22", ""(), !dbg !5
  ret void, !dbg !5
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "a.c", directory: "/tmp", checksumkind: CSK_MD5, checksum: "00000000000000000000000000000000")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !6, spFlags: DISPFlagDefinition, unit: !0)
!5 = !DILocation(line: 1, scope: !4)
!6 = !DISubroutineType(types: !{})
