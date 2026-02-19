; RUN: llvm-as < %s | llvm-dis > %t.orig
; RUN: llvm-as < %s | llvm-c-test --echo > %t.echo
; RUN: diff -w %t.orig %t.echo
;
; Test multiple named metadata nodes to ensure iteration works correctly.

source_filename = "multiple_metadata.ll"

define void @func1() !dbg !4 {
  ret void, !dbg !7
}

define void @func2() !dbg !8 {
  ret void, !dbg !9
}

define void @func3() !dbg !10 {
  ret void, !dbg !11
}

; Multiple named metadata entries
!llvm.dbg.cu = !{!0, !2}
!llvm.module.flags = !{!3}
!custom.metadata.1 = !{!0, !2}
!custom.metadata.2 = !{!3}
!custom.metadata.3 = !{!4, !8, !10}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "test", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "multiple_metadata.ll", directory: "/test")
!2 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "test", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "func1", scope: null, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 1, scope: !4)
!8 = distinct !DISubprogram(name: "func2", scope: null, file: !1, line: 2, type: !5, isLocal: false, isDefinition: true, unit: !0)
!9 = !DILocation(line: 2, scope: !8)
!10 = distinct !DISubprogram(name: "func3", scope: null, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true, unit: !2)
!11 = !DILocation(line: 3, scope: !10)
