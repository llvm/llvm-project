; RUN: opt -analyze -module-debuginfo -enable-new-pm=0 < %s | FileCheck %s
; RUN: opt -passes='print<module-debuginfo>' -disable-output 2>&1 < %s \
; RUN:   | FileCheck %s

; Produced from linking:
;   /tmp/test1.c containing f()
;   /tmp/test2.c containing g()

; Verify that both compile units and both their contained functions are
; listed by DebugInfoFinder:
;CHECK: Compile unit: DW_LANG_C99 from /tmp/test1.c
;CHECK: Compile unit: DW_LANG_C99 from /tmp/test2.c
;CHECK: Subprogram: f from /tmp/test1.c:1
;CHECK: Subprogram: g from /tmp/test2.c:1

define void @f() !dbg !4 {
  ret void, !dbg !14
}

define void @g() !dbg !11 {
  ret void, !dbg !15
}

!llvm.dbg.cu = !{!0, !8}
!llvm.module.flags = !{!13, !16}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.4 (192092)", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "test1.c", directory: "/tmp")
!2 = !{}
!4 = distinct !DISubprogram(name: "f", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !0, scopeLine: 1, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: "test1.c", directory: "/tmp")
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.4 (192092)", isOptimized: false, emissionKind: FullDebug, file: !9, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!9 = !DIFile(filename: "test2.c", directory: "/tmp")
!11 = distinct !DISubprogram(name: "g", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !8, scopeLine: 1, file: !9, scope: !12, type: !6, retainedNodes: !2)
!12 = !DIFile(filename: "test2.c", directory: "/tmp")
!13 = !{i32 2, !"Dwarf Version", i32 4}
!14 = !DILocation(line: 1, scope: !4)
!15 = !DILocation(line: 1, scope: !11)
!16 = !{i32 1, !"Debug Info Version", i32 3}
