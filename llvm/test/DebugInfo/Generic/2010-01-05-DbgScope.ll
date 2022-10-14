; RUN: llc < %s -o /dev/null
; PR 5942
define ptr @foo() nounwind {
entry:
  %0 = load i32, ptr undef, align 4, !dbg !0          ; <i32> [#uses=1]
  %1 = inttoptr i32 %0 to ptr, !dbg !0            ; <ptr> [#uses=1]
  ret ptr %1, !dbg !10

}

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!14}

!0 = !DILocation(line: 571, column: 3, scope: !1)
!1 = distinct !DILexicalBlock(line: 1, column: 1, file: !11, scope: !2)
!2 = distinct !DISubprogram(name: "foo", linkageName: "foo", file: !11, line: 561, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !3, scope: !3, type: !4)
!3 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang 1.1", isOptimized: true, emissionKind: FullDebug, file: !11, enums: !12, retainedTypes: !12)
!4 = !DISubroutineType(types: !5)
!5 = !{!6}
!6 = !DIBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!10 = !DILocation(line: 588, column: 1, scope: !2)
!11 = !DIFile(filename: "hashtab.c", directory: "/usr/src/gnu/usr.bin/cc/cc_tools/../../../../contrib/gcclibs/libiberty")
!12 = !{}
!14 = !{i32 1, !"Debug Info Version", i32 3}
