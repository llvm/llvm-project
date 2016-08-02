; RUN: opt -S -simplifycfg < %s | FileCheck %s
; Radar 9342286
; Assign DebugLoc to trap instruction.
define void @foo() nounwind ssp !dbg !0 {
; CHECK: call void @llvm.trap(), !dbg
  store i32 42, i32* null, !dbg !5
  ret void, !dbg !7
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10}
!llvm.dbg.sp = !{!0}

!0 = distinct !DISubprogram(name: "foo", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, file: !8, scope: !1, type: !3)
!1 = !DIFile(filename: "foo.c", directory: "/private/tmp")
!2 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "Apple clang version 3.0 (tags/Apple/clang-206.1) (based on LLVM 3.0svn)", isOptimized: true, emissionKind: 0, file: !8, enums: !{}, retainedTypes: !{}, subprograms: !9)
!3 = !DISubroutineType(types: !4)
!4 = !{null}
!5 = !DILocation(line: 4, column: 2, scope: !6)
!6 = distinct !DILexicalBlock(line: 3, column: 12, file: !8, scope: !0)
!7 = !DILocation(line: 5, column: 1, scope: !6)
!8 = !DIFile(filename: "foo.c", directory: "/private/tmp")
!9 = !{!0}
!10 = !{i32 1, !"Debug Info Version", i32 3}
