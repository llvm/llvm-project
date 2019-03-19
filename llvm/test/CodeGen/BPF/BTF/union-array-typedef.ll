; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Source code:
;   typedef int _int;
;   union t {char m[4]; _int n;} a;
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm t.c

%union.t = type { i32 }

@a = common dso_local local_unnamed_addr global %union.t zeroinitializer, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!16, !17, !18}
!llvm.ident = !{!19}

; CHECK:             .section        .BTF,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   120
; CHECK-NEXT:        .long   120
; CHECK-NEXT:        .long   41
; CHECK-NEXT:        .long   1                       # BTF_KIND_UNION(id = 1)
; CHECK-NEXT:        .long   83886082                # 0x5000002
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   0                       # BTF_KIND_ARRAY(id = 2)
; CHECK-NEXT:        .long   50331648                # 0x3000000
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   7                       # BTF_KIND_INT(id = 3)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   32                      # 0x20
; CHECK-NEXT:        .long   27                      # BTF_KIND_INT(id = 4)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   16777224                # 0x1000008
; CHECK-NEXT:        .long   32                      # BTF_KIND_TYPEDEF(id = 5)
; CHECK-NEXT:        .long   134217728               # 0x8000000
; CHECK-NEXT:        .long   6
; CHECK-NEXT:        .long   37                      # BTF_KIND_INT(id = 6)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16777248                # 0x1000020
; CHECK-NEXT:        .byte   0                       # string offset=0
; CHECK-NEXT:        .byte   116                     # string offset=1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .byte   109                     # string offset=3
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .byte   110                     # string offset=5
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "__ARRAY_SIZE_TYPE__"   # string offset=7
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "char"                  # string offset=27
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "_int"                  # string offset=32
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "int"                   # string offset=37
; CHECK-NEXT:        .byte   0

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 2, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 8.0.0 (trunk 345296) (llvm/trunk 345297)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: "/home/yhs/tmp")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "t", file: !3, line: 2, size: 32, elements: !7)
!7 = !{!8, !13}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "m", scope: !6, file: !3, line: 2, baseType: !9, size: 32)
!9 = !DICompositeType(tag: DW_TAG_array_type, baseType: !10, size: 32, elements: !11)
!10 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!11 = !{!12}
!12 = !DISubrange(count: 4)
!13 = !DIDerivedType(tag: DW_TAG_member, name: "n", scope: !6, file: !3, line: 2, baseType: !14, size: 32)
!14 = !DIDerivedType(tag: DW_TAG_typedef, name: "_int", file: !3, line: 1, baseType: !15)
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = !{i32 2, !"Dwarf Version", i32 4}
!17 = !{i32 2, !"Debug Info Version", i32 3}
!18 = !{i32 1, !"wchar_size", i32 4}
!19 = !{!"clang version 8.0.0 (trunk 345296) (llvm/trunk 345297)"}
