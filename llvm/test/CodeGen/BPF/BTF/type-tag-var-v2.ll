; RUN: llc -march=bpfel -filetype=obj -o - %s \
; RUN: | llvm-objcopy --dump-section .BTF=- - | %python %S/print_btf.py - | FileCheck %s
; RUN: llc -march=bpfeb -filetype=obj -o - %s \
; RUN: | llvm-objcopy --dump-section .BTF=- - | %python %S/print_btf.py - | FileCheck %s
;
; Source:
;   #define __tag1 __attribute__((btf_type_tag("tag1")))
;   #define __tag2 __attribute__((btf_type_tag("tag2")))
;   int __tag1 * __tag1 __tag2 *g;
; Compilation flag:
;   clang -mllvm -btf-type-tag-v2 -target bpf -O2 -g -S -emit-llvm test.c

; CHECK: [1] PTR '(anon)' type_id=4
; CHECK: [2] PTR '(anon)' type_id=6
; CHECK: [3] TYPE_TAG 'tag1' type_id=2
; CHECK: [4] TYPE_TAG 'tag2' type_id=3
; CHECK: [5] INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK: [6] TYPE_TAG 'tag1' type_id=5
; CHECK: [7] VAR 'g' type_id=1, linkage=global
; CHECK: [8] DATASEC '.bss' size=0 vlen=1
; CHECK: 	type_id=7 offset=0 size=8

@g = dso_local local_unnamed_addr global ptr null, align 8, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!12, !13, !14, !15}
!llvm.ident = !{!16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 3, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang, some version", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "some-file.c", directory: "/some/dir")
!4 = !{!0}
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64)
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64, annotations: !10)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed, annotations: !8)
!8 = !{!9}
!9 = !{!"btf:type_tag", !"tag1"}
!10 = !{!9, !11}
!11 = !{!"btf:type_tag", !"tag2"}

!12 = !{i32 7, !"Dwarf Version", i32 5}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = !{i32 7, !"frame-pointer", i32 2}
!16 = !{!"clang, some version"}
