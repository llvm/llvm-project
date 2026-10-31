; The begin symbols of the debug sections on z/OS GOFF are created with
; EXE_DATA as code/data type, but no ADA is set for them (see
; MCObjectFileInfo::initGOFFMCObjectFileInfo). If such a symbol is referenced
; without its section ever being emitted, it stays undefined and
; SystemZHLASMAsmStreamer::finishImpl() dereferences its null ADA.
;
; Using sections as references makes DW_AT_stmt_list refer to the begin symbol
; of the .debug_line section (D_LINE), which is not emitted in this module.
;
; This test documents the current (crashing) behavior. Once the ADA is set for
; the debug sections, the crash goes away and this test should be turned into
; a regular test.
;
; RUN: llc < %s -mtriple=s390x-ibm-zos -dwarf-sections-as-references=Enable

@fortytwo = hidden global i32 42, align 4, !dbg !0

define hidden signext i32 @getFortyTwo() !dbg !8 {
entry:
  %0 = load i32, ptr @fortytwo, align 4, !dbg !11
  ret i32 %0, !dbg !12
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!6, !7}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "fortytwo", scope: !2, file: !3, line: 2, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 22.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "fortytwo.c", directory: "")
!4 = !{!0}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !{i32 7, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = distinct !DISubprogram(name: "getFortyTwo", scope: !3, file: !3, line: 4, type: !9, scopeLine: 4, spFlags: DISPFlagDefinition, unit: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{!5}
!11 = !DILocation(line: 4, column: 28, scope: !8)
!12 = !DILocation(line: 4, column: 21, scope: !8)
