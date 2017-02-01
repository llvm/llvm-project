; RUN: llc -filetype=obj -o - %s | llvm-objdump -section-headers - | FileCheck %s
; This should not trigger the "Creating regular section after DWARF" assert.
; CHECK: __text
; CHECK: __thread_ptr  00000004
target triple = "thumbv7-apple-ios9.0.0"

@b = external thread_local global i32
define i32* @func(i32 %a) !dbg !9 {
  ret i32* @b
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6, !7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "r.ii", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 1, !"min_enum_size", i32 4}
!7 = !{i32 1, !"PIC Level", i32 2}
!9 = distinct !DISubprogram(name: "func", scope: !1, file: !1, line: 4, type: !10, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{null, !12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocalVariable(name: "a", arg: 1, scope: !9, file: !1, line: 4, type: !12)
!14 = !DIExpression()
