; RUN: llc -mtriple=x86_64-linux-gnu -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Verify BTF_KIND_ENUM emission on x86_64.
;
; Source:
;   enum color { RED, GREEN, BLUE };
;   enum color a;
; Compilation flag:
;   clang -target x86_64-linux-gnu -g -gbtf -S -emit-llvm t.c

@a = dso_local local_unnamed_addr global i32 0, align 4, !dbg !0

; CHECK:             .section        .BTF,"",@progbits
; CHECK-NEXT:        .short  60319                           # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   76
; CHECK-NEXT:        .long   76
; CHECK-NEXT:        .long   29
; CHECK-NEXT:        .long   1                               # BTF_KIND_ENUM(id = 1)
; CHECK-NEXT:        .long   100663299                       # 0x6000003
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   7
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   11
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   17
; CHECK-NEXT:        .long   2
; CHECK:             .byte   0                               # string offset=0
; CHECK-NEXT:        .ascii  "color"                         # string offset=1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "RED"                           # string offset=7
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "GREEN"                         # string offset=11
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "BLUE"                          # string offset=17
; CHECK-NEXT:        .byte   0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!12, !13, !14, !15}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 2, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: "/tmp")
!4 = !{!6}
!5 = !{!0}
!6 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "color", file: !3, line: 1, baseType: !7, size: 32, elements: !8)
!7 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!8 = !{!9, !10, !11}
!9 = !DIEnumerator(name: "RED", value: 0)
!10 = !DIEnumerator(name: "GREEN", value: 1)
!11 = !DIEnumerator(name: "BLUE", value: 2)
!12 = !{i32 2, !"Dwarf Version", i32 4}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = !{i32 4, !"BTF", i32 1}
