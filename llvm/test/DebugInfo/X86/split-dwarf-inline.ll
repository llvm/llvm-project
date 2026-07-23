; RUN: llc -split-dwarf-file=foo.dwo %s -filetype=obj -o - | llvm-dwarfdump -debug-info - | FileCheck %s

; CHECK: .debug_info contents:
; CHECK: DW_TAG_subprogram
; CHECK: caller_func
; CHECK: DW_TAG_inlined_subroutine
; CHECK: inlined_func

; CHECK: .debug_info.dwo contents
; CHECK: DW_TAG_subprogram
; CHECK: caller_func
; CHECK: DW_TAG_inlined_subroutine
; CHECK: inlined_func

target triple = "x86_64-unknown-linux-gnu"

define void @caller_func() !dbg !9 {
entry:
  ret void, !dbg !13
}

!llvm.dbg.cu = !{!0, !3}
!llvm.module.flags = !{!5, !6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang trunk", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false)
!1 = !DIFile(filename: "a.cpp", directory: "/tmp")
!3 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "clang trunk", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: true)
!4 = !DIFile(filename: "b.cpp", directory: "/tmp")
!5 = !{i32 2, !"Dwarf Version", i32 4}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!9 = distinct !DISubprogram(name: "caller_func", scope: !4, file: !4, line: 2, type: !10, isLocal: false, isDefinition: true, scopeLine: 2, unit: !3)
!10 = !DISubroutineType(types: !11)
!11 = !{null}
!12 = distinct !DISubprogram(name: "inlined_func", scope: !1, file: !1, line: 1, type: !10, isLocal: false, isDefinition: true, scopeLine: 1, unit: !0)
!13 = !DILocation(line: 1, column: 5, scope: !12, inlinedAt: !14)
!14 = distinct !DILocation(line: 3, column: 3, scope: !9)
