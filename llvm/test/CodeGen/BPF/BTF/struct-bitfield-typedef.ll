; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Source code:
;   typedef int _int;
;   typedef _int __int;
;   struct {char m:2; __int n:3; char p;} a;
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm t.c

%struct.anon = type { i8, i8, [2 x i8] }

@a = common dso_local local_unnamed_addr global %struct.anon zeroinitializer, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!15, !16, !17}
!llvm.ident = !{!18}

; CHECK:             .section        .BTF,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   104
; CHECK-NEXT:        .long   104
; CHECK-NEXT:        .long   27
; CHECK-NEXT:        .long   0                       # BTF_KIND_STRUCT(id = 1)
; CHECK-NEXT:        .long   2214592515              # 0x84000003
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   33554432                # 0x2000000
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   50331650                # 0x3000002
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   8                       # 0x8
; CHECK-NEXT:        .long   7                       # BTF_KIND_INT(id = 2)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   16777224                # 0x1000008
; CHECK-NEXT:        .long   12                      # BTF_KIND_TYPEDEF(id = 3)
; CHECK-NEXT:        .long   134217728               # 0x8000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   18                      # BTF_KIND_TYPEDEF(id = 4)
; CHECK-NEXT:        .long   134217728               # 0x8000000
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   23                      # BTF_KIND_INT(id = 5)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16777248                # 0x1000020
; CHECK-NEXT:        .byte   0                       # string offset=0
; CHECK-NEXT:        .byte   109                     # string offset=1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .byte   110                     # string offset=3
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .byte   112                     # string offset=5
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "char"                  # string offset=7
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "__int"                 # string offset=12
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "_int"                  # string offset=18
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "int"                   # string offset=23
; CHECK-NEXT:        .byte   0

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 3, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 8.0.0 (trunk 345296) (llvm/trunk 345297)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: "/home/yhs/tmp")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !3, line: 3, size: 32, elements: !7)
!7 = !{!8, !10, !14}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "m", scope: !6, file: !3, line: 3, baseType: !9, size: 2, flags: DIFlagBitField, extraData: i64 0)
!9 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "n", scope: !6, file: !3, line: 3, baseType: !11, size: 3, offset: 2, flags: DIFlagBitField, extraData: i64 0)
!11 = !DIDerivedType(tag: DW_TAG_typedef, name: "__int", file: !3, line: 2, baseType: !12)
!12 = !DIDerivedType(tag: DW_TAG_typedef, name: "_int", file: !3, line: 1, baseType: !13)
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "p", scope: !6, file: !3, line: 3, baseType: !9, size: 8, offset: 8)
!15 = !{i32 2, !"Dwarf Version", i32 4}
!16 = !{i32 2, !"Debug Info Version", i32 3}
!17 = !{i32 1, !"wchar_size", i32 4}
!18 = !{!"clang version 8.0.0 (trunk 345296) (llvm/trunk 345297)"}
