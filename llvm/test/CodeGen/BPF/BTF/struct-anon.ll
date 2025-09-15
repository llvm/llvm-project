; RUN: llc -mtriple=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -mtriple=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Source code:
;   struct { struct {int m;}; } a;
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm t.c

%struct.anon = type { %struct.anon.0 }
%struct.anon.0 = type { i32 }

@a = common dso_local local_unnamed_addr global %struct.anon zeroinitializer, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!13, !14, !15}
!llvm.ident = !{!16}

; CHECK:             .section        .BTF,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   64
; CHECK-NEXT:        .long   64
; CHECK-NEXT:        .long   7
; CHECK-NEXT:        .long   0                       # BTF_KIND_STRUCT(id = 1)
; CHECK-NEXT:        .long   67108865                # 0x4000001
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   0                       # BTF_KIND_STRUCT(id = 2)
; CHECK-NEXT:        .long   67108865                # 0x4000001
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   3                       # BTF_KIND_INT(id = 3)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16777248                # 0x1000020
; CHECK-NEXT:        .byte   0                       # string offset=0
; CHECK-NEXT:        .byte   109                     # string offset=1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "int"                   # string offset=3
; CHECK-NEXT:        .byte   0

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 8.0.0 (trunk 344789) (llvm/trunk 344782)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: "/home/yhs/tmp")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !3, line: 1, size: 32, elements: !7)
!7 = !{!8}
!8 = !DIDerivedType(tag: DW_TAG_member, scope: !6, file: !3, line: 1, baseType: !9, size: 32)
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, scope: !6, file: !3, line: 1, size: 32, elements: !10)
!10 = !{!11}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "m", scope: !9, file: !3, line: 1, baseType: !12, size: 32)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{i32 2, !"Dwarf Version", i32 4}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 1, !"wchar_size", i32 4}
!16 = !{!"clang version 8.0.0 (trunk 344789) (llvm/trunk 344782)"}
