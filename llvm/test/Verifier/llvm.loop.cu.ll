; RUN: llvm-as -disable-output < %s -o /dev/null 2>&1 | FileCheck %s

define void @f()  #0 !dbg !6 {
  br label %1, !dbg !9, !llvm.loop !10
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1)
!1 = !DIFile(filename: "f.c", directory: "./")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 7, !"PIC Level", i32 2}
!6 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 7, type: !7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!7 = !DISubroutineType(types: !8)
!8 = !{}
!9 = !DILocation(line: 18, column: 2, scope: !6)
!10 = distinct !{!10, !11}
!11 = !DILocation(line: 18, column: 2, scope: !12)
!12 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 7, type: !7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !13)

; CHECK: warning: ignoring invalid debug info
; This CU isn't listed in llvm.dbg.cu
!13 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1)
