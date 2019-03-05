; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Source code:
;   enum { A = -1, B = 2 } a;
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm t.c

@a = common dso_local local_unnamed_addr global i32 0, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!11, !12, !13}
!llvm.ident = !{!14}

; CHECK:             .section        .BTF,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   28
; CHECK-NEXT:        .long   28
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   0                       # BTF_KIND_ENUM(id = 1)
; CHECK-NEXT:        .long   100663298               # 0x6000002
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   -1
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .byte   0                       # string offset=0
; CHECK-NEXT:        .byte   65                      # string offset=1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .byte   66                      # string offset=3
; CHECK-NEXT:        .byte   0

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 8.0.0 (trunk 344789) (llvm/trunk 344782)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !10, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: "/home/yhs/tmp")
!4 = !{!5}
!5 = !DICompositeType(tag: DW_TAG_enumeration_type, file: !3, line: 1, baseType: !6, size: 32, elements: !7)
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{!8, !9}
!8 = !DIEnumerator(name: "A", value: -1)
!9 = !DIEnumerator(name: "B", value: 2)
!10 = !{!0}
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = !{!"clang version 8.0.0 (trunk 344789) (llvm/trunk 344782)"}
