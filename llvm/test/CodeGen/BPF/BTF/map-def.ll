; RUN: llc -mtriple=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -mtriple=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Source code:
;   struct key_type {
;     int a;
;     int b;
;   };
;   struct map_type {
;     struct key_ptr key;
;     unsigned *value;
;   };
;   struct map_type __attribute__((section(".maps"))) hash_map;
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm t.c

%struct.map_type = type { ptr, ptr }
%struct.key_type = type { i32, i32 }

@hash_map = dso_local local_unnamed_addr global %struct.map_type zeroinitializer, section ".maps", align 8, !dbg !0

; CHECK:             .section        .BTF,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   168
; CHECK-NEXT:        .long   168
; CHECK-NEXT:        .long   65
; CHECK-NEXT:        .long   0                       # BTF_KIND_PTR(id = 1)
; CHECK-NEXT:        .long   33554432                # 0x2000000
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   1                       # BTF_KIND_STRUCT(id = 2)
; CHECK-NEXT:        .long   67108866                # 0x4000002
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .long   10
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   0                       # 0x0
; CHECK-NEXT:        .long   12
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   32                      # 0x20
; CHECK-NEXT:        .long   14                      # BTF_KIND_INT(id = 3)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16777248                # 0x1000020
; CHECK-NEXT:        .long   0                       # BTF_KIND_PTR(id = 4)
; CHECK-NEXT:        .long   33554432                # 0x2000000
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   18                      # BTF_KIND_INT(id = 5)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   32                      # 0x20
; CHECK-NEXT:        .long   31                      # BTF_KIND_STRUCT(id = 6)
; CHECK-NEXT:        .long   67108866                # 0x4000002
; CHECK-NEXT:        .long   16
; CHECK-NEXT:        .long   40
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   0                       # 0x0
; CHECK-NEXT:        .long   44
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   64                      # 0x40
; CHECK-NEXT:        .long   50                      # BTF_KIND_VAR(id = 7)
; CHECK-NEXT:        .long   234881024               # 0xe000000
; CHECK-NEXT:        .long   6
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   59                      # BTF_KIND_DATASEC(id = 8)
; CHECK-NEXT:        .long   251658241               # 0xf000001
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   7
; CHECK-NEXT:        .long   hash_map
; CHECK-NEXT:        .long   16
; CHECK-NEXT:        .byte   0                       # string offset=0
; CHECK-NEXT:        .ascii  "key_type"              # string offset=1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .byte   97                      # string offset=10
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .byte   98                      # string offset=12
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "int"                   # string offset=14
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "unsigned int"          # string offset=18
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "map_type"              # string offset=31
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "key"                   # string offset=40
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "value"                 # string offset=44
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "hash_map"              # string offset=50
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  ".maps"                 # string offset=59
; CHECK-NEXT:        .byte   0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!18, !19, !20}
!llvm.ident = !{!21}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "hash_map", scope: !2, file: !3, line: 9, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 9.0.0 (trunk 364157) (llvm/trunk 364156)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: "/tmp/home/yhs/work/tests/llvm")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "map_type", file: !3, line: 5, size: 128, elements: !7)
!7 = !{!8, !15}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "key", scope: !6, file: !3, line: 6, baseType: !9, size: 64)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "key_type", file: !3, line: 1, size: 64, elements: !11)
!11 = !{!12, !14}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !10, file: !3, line: 2, baseType: !13, size: 32)
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !10, file: !3, line: 3, baseType: !13, size: 32, offset: 32)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "value", scope: !6, file: !3, line: 7, baseType: !16, size: 64, offset: 64)
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !17, size: 64)
!17 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!18 = !{i32 2, !"Dwarf Version", i32 4}
!19 = !{i32 2, !"Debug Info Version", i32 3}
!20 = !{i32 1, !"wchar_size", i32 4}
!21 = !{!"clang version 9.0.0 (trunk 364157) (llvm/trunk 364156)"}
