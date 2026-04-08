; RUN: llc -mtriple=x86_64-linux-gnu -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Verify BTF_KIND_VAR, BTF_KIND_DATASEC, and .BTF.ext FuncInfo/LineInfo
; for a complete translation unit on x86_64.
;
; Source:
;   int g = 5;
;   int test(void) { return g; }
; Compilation flag:
;   clang -target x86_64-linux-gnu -g -gbtf -S -emit-llvm t.c

@g = dso_local global i32 5, align 4, !dbg !0

define dso_local i32 @test() !dbg !11 {
  %1 = load i32, ptr @g, align 4, !dbg !14
  ret i32 %1, !dbg !15
}

; .BTF section with types
; CHECK:             .section        .BTF,"",@progbits
; CHECK-NEXT:        .short  60319                           # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK:             .long   0                               # BTF_KIND_FUNC_PROTO(id = 1)
; CHECK-NEXT:        .long   218103808                       # 0xd000000
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   1                               # BTF_KIND_INT(id = 2)
; CHECK-NEXT:        .long   16777216                        # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16777248                        # 0x1000020
; CHECK:             .long   5                               # BTF_KIND_FUNC(id = 3)
; CHECK:             .long   25                              # BTF_KIND_VAR(id = 4)
; CHECK-NEXT:        .long   234881024                       # 0xe000000
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   1
; CHECK:             .long   27                              # BTF_KIND_DATASEC(id = 5)
; CHECK-NEXT:        .long   251658241                       # 0xf000001
; CHECK:             .ascii  "int"
; CHECK:             .ascii  "test"
; CHECK:             .ascii  ".data"

; .BTF.ext section with FuncInfo and LineInfo
; CHECK:             .section        .BTF.ext,"",@progbits
; CHECK-NEXT:        .short  60319                           # 0xeb9f
; CHECK:             .long   8                               # FuncInfo
; CHECK:             .long   .Lfunc_begin0
; CHECK-NEXT:        .long   3
; CHECK:             .long   16                              # LineInfo

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: "/tmp")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{i32 4, !"BTF", i32 1}
!11 = distinct !DISubprogram(name: "test", scope: !3, file: !3, line: 2, type: !12, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !4)
!12 = !DISubroutineType(types: !13)
!13 = !{!6}
!14 = !DILocation(line: 2, column: 25, scope: !11)
!15 = !DILocation(line: 2, column: 18, scope: !11)
