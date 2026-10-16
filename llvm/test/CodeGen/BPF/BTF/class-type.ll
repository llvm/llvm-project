; RUN: llc -mtriple=bpfel -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck %s
; RUN: %python %p/print_btf.py %t2 | FileCheck %s --check-prefix=FULL

; A C++ record declared with the class keyword has DW_TAG_class_type and is
; a BTF struct like one declared with struct, also when only declared, and a
; map definition declared as a class is handled like a struct one: the types
; its members point to, here Val, are emitted in full rather than as FWD.
;
; class C { public: int x; int y; };
; class Fwd;
; struct Holder { C c; Fwd *f; };
; Holder h;
; class Map { public: int *key; Val *value; };   // struct Val { long a; };
; Map map __attribute__((section(".maps")));

; CHECK:      [1] PTR '(anon)' type_id=2
; CHECK-NEXT: [2] INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK-NEXT: [3] PTR '(anon)' type_id=4
; CHECK-NEXT: [4] STRUCT 'Val' size=8 vlen=1
; CHECK-NEXT:         'a' type_id=5 bits_offset=0
; CHECK-NEXT: [5] INT 'long' size=8 bits_offset=0 nr_bits=64 encoding=SIGNED
; CHECK-NEXT: [6] STRUCT 'Map' size=16 vlen=2
; CHECK-NEXT:         'key' type_id=1 bits_offset=0
; CHECK-NEXT:         'value' type_id=3 bits_offset=64
; CHECK-NEXT: [7] VAR 'map' type_id=6, linkage=global
; CHECK-NEXT: [8] STRUCT 'Holder' size=16 vlen=2
; CHECK-NEXT:         'c' type_id=9 bits_offset=0
; CHECK-NEXT:         'f' type_id=10 bits_offset=64
; CHECK-NEXT: [9] STRUCT 'C' size=8 vlen=2
; CHECK-NEXT:         'x' type_id=2 bits_offset=0
; CHECK-NEXT:         'y' type_id=2 bits_offset=32
; CHECK-NEXT: [10] PTR '(anon)' type_id=11
; CHECK-NEXT: [11] FWD 'Fwd' fwd_kind=struct
; CHECK-NEXT: [12] VAR 'h' type_id=8, linkage=global

; FULL-NOT: FWD 'Val'
; FULL:     STRUCT 'Val' size=8 vlen=1
; FULL-NOT: FWD 'Val'

%struct.Holder = type { %class.C, ptr }
%class.C = type { i32, i32 }

@h = global %struct.Holder zeroinitializer, align 8, !dbg !0
@map = global [2 x ptr] zeroinitializer, section ".maps", align 8, !dbg !30

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!20, !21}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "h", scope: !2, file: !3, line: 4, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4)
!3 = !DIFile(filename: "test.cpp", directory: "/")
!4 = !{!0, !30}
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Holder", file: !3, line: 3, size: 128, elements: !6)
!6 = !{!7, !13}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !5, file: !3, line: 3, baseType: !8, size: 64)
!8 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "C", file: !3, line: 1, size: 64, flags: DIFlagTypePassByValue, elements: !9)
!9 = !{!10, !12}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !8, file: !3, line: 1, baseType: !11, size: 32, flags: DIFlagPublic)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !8, file: !3, line: 1, baseType: !11, size: 32, offset: 32, flags: DIFlagPublic)
!13 = !DIDerivedType(tag: DW_TAG_member, name: "f", scope: !5, file: !3, line: 3, baseType: !14, size: 64, offset: 64)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = !DICompositeType(tag: DW_TAG_class_type, name: "Fwd", file: !3, line: 2, flags: DIFlagFwdDecl)
!20 = !{i32 7, !"Dwarf Version", i32 5}
!21 = !{i32 2, !"Debug Info Version", i32 3}
!30 = !DIGlobalVariableExpression(var: !31, expr: !DIExpression())
!31 = distinct !DIGlobalVariable(name: "map", scope: !2, file: !3, line: 6, type: !32, isLocal: false, isDefinition: true)
!32 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "Map", file: !3, line: 5, size: 128, elements: !33)
!33 = !{!34, !36}
!34 = !DIDerivedType(tag: DW_TAG_member, name: "key", scope: !32, file: !3, line: 5, baseType: !35, size: 64, flags: DIFlagPublic)
!35 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!36 = !DIDerivedType(tag: DW_TAG_member, name: "value", scope: !32, file: !3, line: 5, baseType: !37, size: 64, offset: 64, flags: DIFlagPublic)
!37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !38, size: 64)
!38 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Val", file: !3, line: 5, size: 64, elements: !39)
!39 = !{!40}
!40 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !38, file: !3, line: 5, baseType: !41, size: 64)
!41 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
