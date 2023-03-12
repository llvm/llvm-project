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
;
; Compilation flag:
;   clang -mllvm -btf-type-tag-v2 -S -g -emit-llvm test.c -o test.ll

; Check generation of type tags in chain of CVR modifiers.

; CHECK: [1] TYPE_TAG 'tag2' type_id=10
; CHECK: [2] TYPE_TAG 'tag2' type_id=11
; CHECK: [3] INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK: [4] TYPE_TAG 'tag1' type_id=3
; CHECK: [5] TYPE_TAG 'tag2' type_id=4
; CHECK: [6] VAR 'a' type_id=1, linkage=global
; CHECK: [7] DATASEC '.rodata' size=0 vlen=1
; CHECK:    type_id=6 offset=0 size=4
; CHECK: [8] VOLATILE '(anon)' type_id=3
; CHECK: [9] CONST '(anon)' type_id=8
; CHECK: [10] TYPE_TAG 'tag1' type_id=9
; CHECK: [11] TYPE_TAG 'tag1' type_id=8

@a = dso_local constant i32 0, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!11, !12, !13, !14}
!llvm.ident = !{!15}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 4, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang, some version", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "some-file.c", directory: "/some/dir/")
!4 = !{!0}
!5 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !6)
!6 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !7)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed, annotations: !8)
!8 = !{!9, !10}
!9 = !{!"btf:type_tag", !"tag1"}
!10 = !{!"btf:type_tag", !"tag2"}
!11 = !{i32 7, !"Dwarf Version", i32 5}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = !{i32 7, !"frame-pointer", i32 2}
!15 = !{!"clang, some version"}
