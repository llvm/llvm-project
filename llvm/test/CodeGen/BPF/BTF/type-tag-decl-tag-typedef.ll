; RUN: llc -march=bpfel -filetype=obj -o - %s \
; RUN: | llvm-objcopy --dump-section .BTF=- - | %python %S/print_btf.py - | FileCheck %s
; RUN: llc -march=bpfeb -filetype=obj -o - %s \
; RUN: | llvm-objcopy --dump-section .BTF=- - | %python %S/print_btf.py - | FileCheck %s
;
; Source:
;   #define __tag1 __attribute__((btf_type_tag("tag1")))
;   #define __tag2 __attribute__((btf_type_tag("tag2")))
;   #define __dtag1 __attribute__((btf_decl_tag("dtag1")))
;   #define __dtag2 __attribute__((btf_decl_tag("dtag2")))
;   
;   typedef int __tag1 __dtag1 foo;
;   typedef foo __tag2 __dtag2 bar;
;   struct buz {
;     bar a;
;   } g;
;
; Compilation flag:
;   clang -mllvm -btf-type-tag-v2 -S -g -emit-llvm test.c -o test.ll

; Verify that for both 'foo' and 'bar' btf_decl_tag applies to 'typedef' ID.

; CHECK: [1] STRUCT 'buz' size=4 vlen=1
; CHECK:    'a' type_id=2 bits_offset=0
; CHECK: [2] TYPEDEF 'bar' type_id=5
; CHECK: [3] DECL_TAG 'dtag2' type_id=2 component_idx=-1
; CHECK: [4] TYPEDEF 'foo' type_id=8
; CHECK: [5] TYPE_TAG 'tag2' type_id=4
; CHECK: [6] DECL_TAG 'dtag1' type_id=4 component_idx=-1
; CHECK: [7] INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK: [8] TYPE_TAG 'tag1' type_id=7
; CHECK: [9] VAR 'g' type_id=1, linkage=global
; CHECK: [10] DATASEC '.bss' size=0 vlen=1
; CHECK:    type_id=9 offset=0 size=4

%struct.buz = type { i32 }

@g = dso_local global %struct.buz zeroinitializer, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!18, !19, !20, !21}
!llvm.ident = !{!22}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 10, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang, some version", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "some-file.c", directory: "/some/dir/")
!4 = !{!0}
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "buz", file: !3, line: 8, size: 32, elements: !6)
!6 = !{!7}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !5, file: !3, line: 9, baseType: !8, size: 32)
!8 = !DIDerivedType(tag: DW_TAG_typedef, name: "bar", file: !3, line: 7, baseType: !9, annotations: !16)
!9 = !DIDerivedType(tag: DW_TAG_typedef, name: "foo", file: !3, line: 6, baseType: !10, annotations: !13)
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed, annotations: !11)
!11 = !{!12}
!12 = !{!"btf:type_tag", !"tag1"}
!13 = !{!14, !15}
!14 = !{!"btf:type_tag", !"tag2"}
!15 = !{!"btf_decl_tag", !"dtag1"}
!16 = !{!17}
!17 = !{!"btf_decl_tag", !"dtag2"}
!18 = !{i32 7, !"Dwarf Version", i32 5}
!19 = !{i32 2, !"Debug Info Version", i32 3}
!20 = !{i32 1, !"wchar_size", i32 4}
!21 = !{i32 7, !"frame-pointer", i32 2}
!22 = !{!"clang, some version"}
