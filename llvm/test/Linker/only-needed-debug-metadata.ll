; RUN: llvm-as %s -o %t.bc
; RUN: llvm-as %p/Inputs/only-needed-debug-metadata.ll -o %t2.bc

; Without -only-needed, we need to link in both DISubprogram.
; RUN: llvm-link -S %t2.bc %t.bc | FileCheck %s
; CHECK: distinct !DISubprogram(name: "foo"
; CHECK: distinct !DISubprogram(name: "unused"

; With -only-needed, we only need to link in foo's DISubprogram.
; RUN: llvm-link -S -only-needed %t2.bc %t.bc | FileCheck %s -check-prefix=ONLYNEEDED
; ONLYNEEDED: distinct !DISubprogram(name: "foo"
; ONLYNEEDED-NOT: distinct !DISubprogram(name: "unused"

@X = global i32 5
@U = global i32 6
@U_linkonce = linkonce_odr hidden global i32 6
define i32 @foo() !dbg !4 {
    ret i32 7, !dbg !20
}
define i32 @unused() !dbg !10 {
    ret i32 8, !dbg !21
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!16, !17}
!llvm.ident = !{!18}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.0 (trunk 251407) (llvm/trunk 251401)", isOptimized: true, runtimeVersion: 0, emissionKind: 1, enums: !2, subprograms: !3, globals: !13)
!1 = !DIFile(filename: "linkused2.c", directory: "/usr/local/google/home/tejohnson/llvm/tmp")
!2 = !{}
!3 = !{!4, !10}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 4, type: !5, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: true, variables: !8)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{!9}
!9 = !DILocalVariable(name: "x", arg: 1, scope: !4, file: !1, line: 4, type: !7)
!10 = distinct !DISubprogram(name: "unused", scope: !1, file: !1, line: 8, type: !11, isLocal: false, isDefinition: true, scopeLine: 8, isOptimized: true, variables: !2)
!11 = !DISubroutineType(types: !12)
!12 = !{!7}
!13 = !{!14, !15}
!14 = !DIGlobalVariable(name: "X", scope: !0, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, variable: i32* @X)
!15 = !DIGlobalVariable(name: "U", scope: !0, file: !1, line: 2, type: !7, isLocal: false, isDefinition: true, variable: i32* @U)
!16 = !{i32 2, !"Dwarf Version", i32 4}
!17 = !{i32 2, !"Debug Info Version", i32 3}
!18 = !{!"clang version 3.8.0 (trunk 251407) (llvm/trunk 251401)"}
!19 = !DIExpression()
!20 = !DILocation(line: 4, column: 13, scope: !4)
!21 = !DILocation(line: 9, column: 3, scope: !10)
