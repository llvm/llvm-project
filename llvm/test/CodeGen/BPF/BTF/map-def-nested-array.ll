; RUN: llc -mtriple=bpfel -mcpu=v3 -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck -check-prefixes=CHECK-BTF-SHORT %s
; RUN: %python %p/print_btf.py %t2 | FileCheck -check-prefixes=CHECK-BTF %s
; Source:
;  struct nested_value_type {
;  	int a1;
;  };
;  struct map_type {
;  	struct {
;  		struct nested_value_type *value;
;  	} *values[];
;  };
; Compilation flags:
;   clang -target bpf -g -O2 -S -emit-llvm prog.c

; ModuleID = 'prog.c'
source_filename = "prog.c"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "bpf"

%struct.map_type = type { [0 x ptr] }

@array_of_maps = dso_local local_unnamed_addr global %struct.map_type zeroinitializer, section ".maps", align 8, !dbg !0

; We expect no forward declarations.
;
; CHECK-BTF-SHORT-NOT: FWD

; Assert the whole BTF.
;
; CHECK-BTF: [1] PTR '(anon)' type_id=2
; CHECK-BTF-NEXT: [2] STRUCT 'nested_value_type' size=4 vlen=1
; CHECK-BTF-NEXT:         'a1' type_id=3 bits_offset=0
; CHECK-BTF-NEXT: [3] INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK-BTF-NEXT: [4] STRUCT '(anon)' size=8 vlen=1
; CHECK-BTF-NEXT:         'value' type_id=1 bits_offset=0
; CHECK-BTF-NEXT: [5] PTR '(anon)' type_id=4
; CHECK-BTF-NEXT: [6] ARRAY '(anon)' type_id=5 index_type_id=7 nr_elems=0
; CHECK-BTF-NEXT: [7] INT '__ARRAY_SIZE_TYPE__' size=4 bits_offset=0 nr_bits=32 encoding=(none)
; CHECK-BTF-NEXT: [8] STRUCT 'map_type' size=0 vlen=1
; CHECK-BTF-NEXT:         'values' type_id=6 bits_offset=0
; CHECK-BTF-NEXT: [9] VAR 'array_of_maps' type_id=8, linkage=global
; CHECK-BTF-NEXT: [10] DATASEC '.maps' size=0 vlen=1
; CHECK-BTF-NEXT:         type_id=9 offset=0 size=0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!20, !21, !22, !23}
!llvm.ident = !{!24}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "array_of_maps", scope: !2, file: !3, line: 9, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 22.0.0git (git@github.com:llvm/llvm-project.git ed93eaa421b714028b85cc887d80c45991d7207f)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "prog.c", directory: "/home/mtardy/llvm-bug-repro", checksumkind: CSK_MD5, checksum: "9381d9e83e9c0b235a14704224815e96")
!4 = !{!0}
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "map_type", file: !3, line: 4, elements: !6)
!6 = !{!7}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "values", scope: !5, file: !3, line: 7, baseType: !8)
!8 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, elements: !18)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, scope: !5, file: !3, line: 5, size: 64, elements: !11)
!11 = !{!12}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "value", scope: !10, file: !3, line: 6, baseType: !13, size: 64)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!14 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "nested_value_type", file: !3, line: 1, size: 32, elements: !15)
!15 = !{!16}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "a1", scope: !14, file: !3, line: 2, baseType: !17, size: 32)
!17 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!18 = !{!19}
!19 = !DISubrange(count: -1)
!20 = !{i32 7, !"Dwarf Version", i32 5}
!21 = !{i32 2, !"Debug Info Version", i32 3}
!22 = !{i32 1, !"wchar_size", i32 4}
!23 = !{i32 7, !"frame-pointer", i32 2}
!24 = !{!"clang version 22.0.0git (git@github.com:llvm/llvm-project.git ed93eaa421b714028b85cc887d80c45991d7207f)"}
