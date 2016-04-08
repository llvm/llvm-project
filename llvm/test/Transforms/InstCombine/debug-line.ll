; RUN: opt -instcombine -S < %s | FileCheck %s


@.str = private constant [3 x i8] c"%c\00"

define void @foo() nounwind ssp !dbg !0 {
;CHECK: call i32 @putchar{{.+}} !dbg
  %1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), i32 97), !dbg !5
  ret void, !dbg !7
}

declare i32 @printf(i8*, ...)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10}

!0 = distinct !DISubprogram(name: "foo", line: 4, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, file: !8, scope: !1, type: !3)
!1 = !DIFile(filename: "m.c", directory: "/private/tmp")
!2 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang", isOptimized: true, emissionKind: FullDebug, file: !8, enums: !{}, retainedTypes: !{}, subprograms: !9)
!3 = !DISubroutineType(types: !4)
!4 = !{null}
!5 = !DILocation(line: 5, column: 2, scope: !6)
!6 = distinct !DILexicalBlock(line: 4, column: 12, file: !8, scope: !0)
!7 = !DILocation(line: 6, column: 1, scope: !6)
!8 = !DIFile(filename: "m.c", directory: "/private/tmp")
!9 = !{!0}
!10 = !{i32 1, !"Debug Info Version", i32 3}
