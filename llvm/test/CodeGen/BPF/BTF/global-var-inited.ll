; RUN: llc -mtriple=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -mtriple=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Source code:
;   int a = 3;
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm test.c

@a = dso_local local_unnamed_addr global i32 3, align 4, !dbg !0

; CHECK:             .section        .BTF,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   56
; CHECK-NEXT:        .long   56
; CHECK-NEXT:        .long   13
; CHECK-NEXT:        .long   1                       # BTF_KIND_INT(id = 1)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16777248                # 0x1000020
; CHECK-NEXT:        .long   5                       # BTF_KIND_VAR(id = 2)
; CHECK-NEXT:        .long   234881024               # 0xe000000
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   7                       # BTF_KIND_DATASEC(id = 3)
; CHECK-NEXT:        .long   251658241               # 0xf000001
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   a
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .byte   0                       # string offset=0
; CHECK-NEXT:        .ascii  "int"                   # string offset=1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .byte   97                      # string offset=5
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  ".data"                 # string offset=7
; CHECK-NEXT:        .byte   0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 8.0.20181009 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "/home/yhs/work/tests/llvm/bug")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{!"clang version 8.0.20181009 "}
