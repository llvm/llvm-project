; RUN: llc -mtriple=bpfel -mcpu=v3 -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck -check-prefixes=CHECK-BTF-SHORT %s
; RUN: %python %p/print_btf.py %t2 | FileCheck -check-prefixes=CHECK-BTF %s

; Source code:
;   struct key { int i; };
;   struct val { int j; };
;   
;   #define __uint(name, val) int (*name)[val]
;   #define __type(name, val) typeof(val) *name
;   
;   struct {
;      struct {
;           __uint(type, 1);
;           __uint(max_entries, 1337);
;           __type(key, struct key);
;           __type(value, struct val);
;       } map_def;
;   } map __attribute__((section(".maps")));
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm t.c

; ModuleID = 'bpf.c'
source_filename = "bpf.c"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "bpf"

%struct.anon = type { %struct.anon.0 }
%struct.anon.0 = type { ptr, ptr, ptr, ptr }

@map = dso_local local_unnamed_addr global %struct.anon zeroinitializer, section ".maps", align 8, !dbg !0

; We expect exactly 4 structs:
; * key
; * val
; * inner map type (the actual definition)
; * outer map type (the wrapper)
;
; CHECK-BTF-SHORT-COUNT-4: STRUCT
; CHECK-BTF-SHORT-NOT:     STRUCT

; We expect no forward declarations.
;
; CHECK-BTF-SHORT-NOT: FWD

; Assert the whole BTF.
;
; CHECK-BTF:      [1] PTR '(anon)' type_id=3
; CHECK-BTF-NEXT: [2] INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK-BTF-NEXT: [3] ARRAY '(anon)' type_id=2 index_type_id=4 nr_elems=1
; CHECK-BTF-NEXT: [4] INT '__ARRAY_SIZE_TYPE__' size=4 bits_offset=0 nr_bits=32 encoding=(none)
; CHECK-BTF-NEXT: [5] PTR '(anon)' type_id=6
; CHECK-BTF-NEXT: [6] ARRAY '(anon)' type_id=2 index_type_id=4 nr_elems=1337
; CHECK-BTF-NEXT: [7] PTR '(anon)' type_id=8
;
; Before bug https://github.com/llvm/llvm-project/issues/143361 was fixed, the
; BTF kind of MyKey (#6) and MyValue (#9) would be FWD instead of STRUCT. The
; main goal of this test is making sure that the full STRUCT BTF is generated
; for these types.
;
; CHECK-BTF-NEXT: [8] STRUCT 'key' size=4 vlen=1
; CHECK-BTF-NEXT:         'i' type_id=2 bits_offset=0
; CHECK-BTF-NEXT: [9] PTR '(anon)' type_id=10
; CHECK-BTF-NEXT: [10] STRUCT 'val' size=4 vlen=1
; CHECK-BTF-NEXT:         'j' type_id=2 bits_offset=0
; CHECK-BTF-NEXT: [11] STRUCT '(anon)' size=32 vlen=4
; CHECK-BTF-NEXT:         'type' type_id=1 bits_offset=0
; CHECK-BTF-NEXT:         'max_entries' type_id=5 bits_offset=64
; CHECK-BTF-NEXT:         'key' type_id=7 bits_offset=128
; CHECK-BTF-NEXT:         'value' type_id=9 bits_offset=192
; CHECK-BTF-NEXT: [12] STRUCT '(anon)' size=32 vlen=1
; CHECK-BTF-NEXT:         'map_def' type_id=11 bits_offset=0
; CHECK-BTF-NEXT: [13] VAR 'map' type_id=12, linkage=global
; CHECK-BTF-NEXT: [14] DATASEC '.maps' size=0 vlen=1
; CHECK-BTF-NEXT:         type_id=13 offset=0 size=32

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!31, !32, !33, !34}
!llvm.ident = !{!35}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "map", scope: !2, file: !3, line: 14, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 21.0.0git (git@github.com:llvm/llvm-project.git c935bd3798b39330aab2c9ca29a519457d5e5245)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "bpf.c", directory: "/tmp", checksumkind: CSK_MD5, checksum: "2330cce6d83c72ef5335abc3016de28e")
!4 = !{!0}
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !3, line: 7, size: 256, elements: !6)
!6 = !{!7}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "map_def", scope: !5, file: !3, line: 13, baseType: !8, size: 256)
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, scope: !5, file: !3, line: 8, size: 256, elements: !9)
!9 = !{!10, !16, !21, !26}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "type", scope: !8, file: !3, line: 9, baseType: !11, size: 64)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 32, elements: !14)
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{!15}
!15 = !DISubrange(count: 1)
!16 = !DIDerivedType(tag: DW_TAG_member, name: "max_entries", scope: !8, file: !3, line: 10, baseType: !17, size: 64, offset: 64)
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !18, size: 64)
!18 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 42784, elements: !19)
!19 = !{!20}
!20 = !DISubrange(count: 1337)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "key", scope: !8, file: !3, line: 11, baseType: !22, size: 64, offset: 128)
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !23, size: 64)
!23 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "key", file: !3, line: 1, size: 32, elements: !24)
!24 = !{!25}
!25 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !23, file: !3, line: 1, baseType: !13, size: 32)
!26 = !DIDerivedType(tag: DW_TAG_member, name: "value", scope: !8, file: !3, line: 12, baseType: !27, size: 64, offset: 192)
!27 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !28, size: 64)
!28 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "val", file: !3, line: 2, size: 32, elements: !29)
!29 = !{!30}
!30 = !DIDerivedType(tag: DW_TAG_member, name: "j", scope: !28, file: !3, line: 2, baseType: !13, size: 32)
!31 = !{i32 7, !"Dwarf Version", i32 5}
!32 = !{i32 2, !"Debug Info Version", i32 3}
!33 = !{i32 1, !"wchar_size", i32 4}
!34 = !{i32 7, !"frame-pointer", i32 2}
!35 = !{!"clang version 21.0.0git (git@github.com:llvm/llvm-project.git c935bd3798b39330aab2c9ca29a519457d5e5245)"}
