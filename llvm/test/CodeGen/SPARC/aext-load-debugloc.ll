; RUN: llc -mtriple=sparcv9 -stop-after=finalize-isel %s -o - | FileCheck %s

; An i32 or i8 return without signext or zeroext is any-extended to i64 by
; return lowering, and the any_extend gets the ret's location. When the DAG
; combiner folds the any_extend into the load, the new load does nothing more
; than the original one, so it keeps the load's location.

; CHECK-DAG: [[LOAD:![0-9]+]] = !DILocation(line: 8, column: 7,
; CHECK-DAG: [[BYTE:![0-9]+]] = !DILocation(line: 14, column: 10,

; CHECK-LABEL: name: get
; CHECK:       LDri {{.*}}, debug-location [[LOAD]]

; CHECK-LABEL: name: get_byte
; CHECK:       LDUBrr {{.*}}, debug-location [[BYTE]]

define i32 @get(ptr %p) !dbg !4 {
entry:
  %a = getelementptr inbounds i8, ptr %p, i64 4, !dbg !11
  %v = load i32, ptr %a, align 4, !dbg !11
  ret i32 %v, !dbg !12
}

define i8 @get_byte(ptr %p) !dbg !13 {
entry:
  %v = load i8, ptr %p, align 1, !dbg !14
  ret i8 %v, !dbg !15
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "blk.c", directory: "/tmp")
!4 = distinct !DISubprogram(name: "get", scope: !1, file: !1, line: 7, type: !5, scopeLine: 7, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !DILocation(line: 8, column: 7, scope: !4)
!12 = !DILocation(line: 10, column: 2, scope: !4)
!13 = distinct !DISubprogram(name: "get_byte", scope: !1, file: !1, line: 13, type: !5, scopeLine: 13, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!14 = !DILocation(line: 14, column: 10, scope: !13)
!15 = !DILocation(line: 15, column: 3, scope: !13)
