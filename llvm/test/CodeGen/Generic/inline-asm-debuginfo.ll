; RUN: not llc %s 2>&1 | FileCheck %s

define void @bad_asm() !dbg !8 {
entry:
  ; CHECK: inline-asm-debuginfo.c:3:{{[0-9]+}}: <inline asm>:1:14: error: unknown token in expression
  call void asm sideeffect "BAD SYNTAX$$%", ""(), !dbg !11, !srcloc !12
  ret void, !dbg !13
}

define void @bad_multi_asm() !dbg !14 {
entry:
  ; CHECK: inline-asm-debuginfo.c:7:{{[0-9]+}}: <inline asm>:1:3: error: invalid instruction mnemonic 'bad'
  call void asm sideeffect ";BAD SYNTAX;;", ""(), !dbg !15, !srcloc !16
  ret void, !dbg !17
}

define void @bad_multi_asm_linechg() !dbg !18 {
entry:
  ; CHECK: inline-asm-debuginfo.c:16:{{[0-9]+}}: <inline asm>:4:1: error: invalid instruction mnemonic 'bad'
  call void asm sideeffect ";\0A\0A\0ABAD SYNTAX;\0A;\0A", ""(), !dbg !19, !srcloc !20
  ret void, !dbg !21
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1)
!1 = !DIFile(filename: "inline-asm-debuginfo.c", directory: ".")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 2}
!6 = !{i32 7, !"frame-pointer", i32 2}
!7 = !{!"clang version 20.0.0git"}
!8 = distinct !DISubprogram(name: "bad_asm", scope: !1, file: !1, line: 2, type: !9, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !DILocation(line: 3, column: 3, scope: !8)
!12 = !{i64 74}
!13 = !DILocation(line: 4, column: 1, scope: !8)
!14 = distinct !DISubprogram(name: "bad_multi_asm", scope: !1, file: !1, line: 6, type: !9, scopeLine: 6, spFlags: DISPFlagDefinition, unit: !0)
!15 = !DILocation(line: 7, column: 3, scope: !14)
!16 = !{i64 226}
!17 = !DILocation(line: 10, column: 1, scope: !14)
!18 = distinct !DISubprogram(name: "bad_multi_asm_linechg", scope: !1, file: !1, line: 12, type: !9, scopeLine: 12, spFlags: DISPFlagDefinition, unit: !0)
!19 = !DILocation(line: 13, column: 3, scope: !18)
!20 = !{i64 418, i64 422, i64 424, i64 437, i64 569}
!21 = !DILocation(line: 16, column: 1, scope: !18)
