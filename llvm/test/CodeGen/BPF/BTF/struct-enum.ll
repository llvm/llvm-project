; RUN: llc -mtriple=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -mtriple=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Source code:
;   enum t1 { A , B };
;   struct t2 { enum t1 m:2; enum t1 n; } a;
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm t.c

%struct.t2 = type { i8, i32 }

@a = common dso_local local_unnamed_addr global %struct.t2 zeroinitializer, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!15, !16, !17}
!llvm.ident = !{!18}

; CHECK:             .section        .BTF,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   64
; CHECK-NEXT:        .long   64
; CHECK-NEXT:        .long   15
; CHECK-NEXT:        .long   1                       # BTF_KIND_STRUCT(id = 1)
; CHECK-NEXT:        .long   2214592514              # 0x84000002
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   33554432                # 0x2000000
; CHECK-NEXT:        .long   6
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   32                      # 0x20
; CHECK-NEXT:        .long   8                       # BTF_KIND_ENUM(id = 2)
; CHECK-NEXT:        .long   100663298               # 0x6000002
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   11
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   13
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .byte   0                       # string offset=0
; CHECK-NEXT:        .ascii  "t2"                    # string offset=1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .byte   109                     # string offset=4
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .byte   110                     # string offset=6
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "t1"                    # string offset=8
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .byte   65                      # string offset=11
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .byte   66                      # string offset=13
; CHECK-NEXT:        .byte   0

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 2, type: !11, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 8.0.0 (trunk 344789) (llvm/trunk 344782)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !10, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: "/home/yhs/tmp")
!4 = !{!5}
!5 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "t1", file: !3, line: 1, baseType: !6, size: 32, elements: !7)
!6 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!7 = !{!8, !9}
!8 = !DIEnumerator(name: "A", value: 0, isUnsigned: true)
!9 = !DIEnumerator(name: "B", value: 1, isUnsigned: true)
!10 = !{!0}
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t2", file: !3, line: 2, size: 64, elements: !12)
!12 = !{!13, !14}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "m", scope: !11, file: !3, line: 2, baseType: !5, size: 2, flags: DIFlagBitField, extraData: i64 0)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "n", scope: !11, file: !3, line: 2, baseType: !5, size: 32, offset: 32)
!15 = !{i32 2, !"Dwarf Version", i32 4}
!16 = !{i32 2, !"Debug Info Version", i32 3}
!17 = !{i32 1, !"wchar_size", i32 4}
!18 = !{!"clang version 8.0.0 (trunk 344789) (llvm/trunk 344782)"}
