; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
;
; Source code:
;   struct key_type {
;     int a1;
;   };
;   typedef struct map_type {
;     struct key_type *key;
;   } _map_type;
;   typedef _map_type __map_type;
;   __map_type __attribute__((section(".maps"))) hash_map;
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm t2.c

%struct.map_type = type { %struct.key_type* }
%struct.key_type = type { i32 }

@hash_map = dso_local local_unnamed_addr global %struct.map_type zeroinitializer, section ".maps", align 8, !dbg !0

; CHECK:             .long   0                               # BTF_KIND_PTR(id = 1)
; CHECK-NEXT:        .long   33554432                        # 0x2000000
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   1                               # BTF_KIND_STRUCT(id = 2)
; CHECK-NEXT:        .long   67108865                        # 0x4000001
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   10
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   0                               # 0x0
; CHECK-NEXT:        .long   13                              # BTF_KIND_INT(id = 3)
; CHECK-NEXT:        .long   16777216                        # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16777248                        # 0x1000020
; CHECK-NEXT:        .long   17                              # BTF_KIND_TYPEDEF(id = 4)
; CHECK-NEXT:        .long   134217728                       # 0x8000000
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   28                              # BTF_KIND_TYPEDEF(id = 5)
; CHECK-NEXT:        .long   134217728                       # 0x8000000
; CHECK-NEXT:        .long   6
; CHECK-NEXT:        .long   38                              # BTF_KIND_STRUCT(id = 6)
; CHECK-NEXT:        .long   67108865                        # 0x4000001
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .long   47
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   0                               # 0x0
; CHECK-NEXT:        .long   51                              # BTF_KIND_VAR(id = 7)
; CHECK-NEXT:        .long   234881024                       # 0xe000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   60                              # BTF_KIND_DATASEC(id = 8)
; CHECK-NEXT:        .long   251658241                       # 0xf000001
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   7
; CHECK-NEXT:        .long   hash_map
; CHECK-NEXT:        .long   8

; CHECK:             .ascii  "key_type"                      # string offset=1
; CHECK:             .ascii  "a1"                            # string offset=10
; CHECK:             .ascii  "int"                           # string offset=13
; CHECK:             .ascii  "__map_type"                    # string offset=17
; CHECK:             .ascii  "_map_type"                     # string offset=28
; CHECK:             .ascii  "map_type"                      # string offset=38
; CHECK:             .ascii  "key"                           # string offset=47
; CHECK:             .ascii  "hash_map"                      # string offset=51
; CHECK:             .ascii  ".maps"                         # string offset=60

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!16, !17, !18}
!llvm.ident = !{!19}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "hash_map", scope: !2, file: !3, line: 8, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 11.0.0 (https://github.com/llvm/llvm-project.git b8409c03ed90807f3d49c7d98dceea98cf461f7a)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "t2.c", directory: "/tmp/home/yhs/tmp1")
!4 = !{}
!5 = !{!0}
!6 = !DIDerivedType(tag: DW_TAG_typedef, name: "__map_type", file: !3, line: 7, baseType: !7)
!7 = !DIDerivedType(tag: DW_TAG_typedef, name: "_map_type", file: !3, line: 6, baseType: !8)
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "map_type", file: !3, line: 4, size: 64, elements: !9)
!9 = !{!10}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "key", scope: !8, file: !3, line: 5, baseType: !11, size: 64)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "key_type", file: !3, line: 1, size: 32, elements: !13)
!13 = !{!14}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "a1", scope: !12, file: !3, line: 2, baseType: !15, size: 32)
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = !{i32 7, !"Dwarf Version", i32 4}
!17 = !{i32 2, !"Debug Info Version", i32 3}
!18 = !{i32 1, !"wchar_size", i32 4}
!19 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git b8409c03ed90807f3d49c7d98dceea98cf461f7a)"}
