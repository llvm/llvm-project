; RUN: llc -mtriple=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -mtriple=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Source code:
;   int gv1 __attribute__((section("maps")));
;   int gv2 __attribute__((section("maps"))) = 5;
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm test.c

@gv2 = dso_local local_unnamed_addr global i32 5, section "maps", align 4, !dbg !0
@gv1 = dso_local local_unnamed_addr global i32 0, section "maps", align 4, !dbg !6

; CHECK:             .section        .BTF,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   84
; CHECK-NEXT:        .long   84
; CHECK-NEXT:        .long   18
; CHECK-NEXT:        .long   1                       # BTF_KIND_INT(id = 1)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16777248                # 0x1000020
; CHECK-NEXT:        .long   5                       # BTF_KIND_VAR(id = 2)
; CHECK-NEXT:        .long   234881024               # 0xe000000
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   9                       # BTF_KIND_VAR(id = 3)
; CHECK-NEXT:        .long   234881024               # 0xe000000
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   13                      # BTF_KIND_DATASEC(id = 4)
; CHECK-NEXT:        .long   251658242               # 0xf000002
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   gv2
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   gv1
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .byte   0                       # string offset=0
; CHECK-NEXT:        .ascii  "int"                   # string offset=1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "gv2"                   # string offset=5
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "gv1"                   # string offset=9
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "maps"                  # string offset=13
; CHECK-NEXT:        .byte   0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!9, !10, !11}
!llvm.ident = !{!12}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "gv2", scope: !2, file: !3, line: 2, type: !8, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 8.0.20181009 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "/home/yhs/work/tests/llvm/bug")
!4 = !{}
!5 = !{!0, !6}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "gv1", scope: !2, file: !3, line: 1, type: !8, isLocal: false, isDefinition: true)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{!"clang version 8.0.20181009 "}
