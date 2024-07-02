; RUN: llc -march=bpfel -filetype=obj -o - %s \
; RUN: | llvm-objcopy --dump-section .BTF=- - | %python %S/print_btf.py - | FileCheck %s
; RUN: llc -march=bpfeb -filetype=obj -o - %s \
; RUN: | llvm-objcopy --dump-section .BTF=- - | %python %S/print_btf.py - | FileCheck %s
;
; Source:
;   #define __tag1 __attribute__((btf_type_tag("tag1")))
;   #define __tag2 __attribute__((btf_type_tag("tag2")))
;   
;   const volatile __tag1 __tag2 int a;
;   const volatile __tag1 int b;
;
; Compilation flag:
;   clang -mllvm -btf-type-tag-v2 -S -g -emit-llvm test.c -o test.ll

; Check that const->volatile->int chain is reused for variables 'a' and 'b'.

; CHECK: [1] TYPE_TAG 'tag2' type_id=14
; CHECK: [2] TYPE_TAG 'tag2' type_id=15
; CHECK: [3] INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK: [4] TYPE_TAG 'tag1' type_id=3
; CHECK: [5] TYPE_TAG 'tag2' type_id=4
; CHECK: [6] VAR 'a' type_id=1, linkage=global
; CHECK: [7] TYPE_TAG 'tag1' type_id=13
; CHECK: [8] TYPE_TAG 'tag1' type_id=12
; CHECK: [9] TYPE_TAG 'tag1' type_id=3
; CHECK: [10] VAR 'b' type_id=7, linkage=global
; CHECK: [11] DATASEC '.rodata' size=0 vlen=2
; CHECK:    type_id=6 offset=0 size=4
; CHECK:    type_id=10 offset=0 size=4
; CHECK: [12] VOLATILE '(anon)' type_id=3
; CHECK: [13] CONST '(anon)' type_id=12
; CHECK: [14] TYPE_TAG 'tag1' type_id=13
; CHECK: [15] TYPE_TAG 'tag1' type_id=12

@a = dso_local constant i32 0, align 4, !dbg !0
@b = dso_local constant i32 0, align 4, !dbg !5

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!17, !18, !19, !20}
!llvm.ident = !{!21}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 4, type: !12, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang, some version", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "some-file.c", directory: "/some/dir/")
!4 = !{!0, !5}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "b", scope: !2, file: !3, line: 5, type: !7, isLocal: false, isDefinition: true)
!7 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !8)
!8 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !9)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed, annotations: !10)
!10 = !{!11}
!11 = !{!"btf:type_tag", !"tag1"}
!12 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !13)
!13 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !14)
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed, annotations: !15)
!15 = !{!11, !16}
!16 = !{!"btf:type_tag", !"tag2"}
!17 = !{i32 7, !"Dwarf Version", i32 5}
!18 = !{i32 2, !"Debug Info Version", i32 3}
!19 = !{i32 1, !"wchar_size", i32 4}
!20 = !{i32 7, !"frame-pointer", i32 2}
!21 = !{!"clang, some version"}
