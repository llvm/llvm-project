; RUN: llc -mtriple=bpfel -mcpu=v3 -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck -check-prefixes=CHECK-BTF %s
;
; Source code:
;   struct key_type {
;     int a1;
;   };
;   typedef struct map_type {
;     struct key_ptr key;
;   } _map_type;
;   typedef _map_type __map_type;
;   __map_type __attribute__((section(".maps"))) hash_map;
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm t2.c

%struct.map_type = type { ptr }
%struct.key_type = type { i32 }

@hash_map = dso_local local_unnamed_addr global %struct.map_type zeroinitializer, section ".maps", align 8, !dbg !0

; CHECK-BTF: [1] PTR '(anon)' type_id=2
; CHECK-BTF-NEXT: [2] STRUCT 'key_type' size=4 vlen=1
; CHECK-BTF-NEXT:         'a1' type_id=3 bits_offset=0
; CHECK-BTF-NEXT: [3] INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK-BTF-NEXT: [4] STRUCT 'map_type' size=8 vlen=1
; CHECK-BTF-NEXT:         'key' type_id=1 bits_offset=0
; CHECK-BTF-NEXT: [5] TYPEDEF '_map_type' type_id=4
; CHECK-BTF-NEXT: [6] TYPEDEF '__map_type' type_id=5
; CHECK-BTF-NEXT: [7] VAR 'hash_map' type_id=6, linkage=global
; CHECK-BTF-NEXT: [8] DATASEC '.maps' size=0 vlen=1
; CHECK-BTF-NEXT:         type_id=7 offset=0 size=8

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
