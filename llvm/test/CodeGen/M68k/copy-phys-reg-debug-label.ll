; RUN: llc -mtriple=m68k -O0 -verify-machineinstrs < %s -o /dev/null

; This used to crash in M68kInstrInfo::copyPhysReg. Its backward liveness walk
; from MBB.end() to the COPY instruction passed a DBG_LABEL to
; LiveRegUnits::stepBackward, which rejects debug instructions.

declare void @consume(i32, i32, i32, i32, i32, i32, i32, i32,
                      i32, i32, i32, i32, i32, i32, i32, i32)

define void @f() !dbg !4 {
entry:
  call void @consume(i32 0, i32 0, i32 0, i32 0,
                     i32 0, i32 0, i32 0, i32 0,
                     i32 0, i32 0, i32 0, i32 0,
                     i32 0, i32 0, i32 0, i32 0)
    #dbg_label(!7, !8)
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1,
                             isOptimized: false, emissionKind: FullDebug)
!1 = !DIFile(filename: "reduced.c", directory: ".")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1,
                            type: !5, scopeLine: 1,
                            spFlags: DISPFlagDefinition, unit: !0,
                            retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !DILabel(scope: !4, name: "label", file: !1, line: 2)
!8 = !DILocation(line: 2, column: 1, scope: !4)
