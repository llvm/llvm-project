; RUN: llc -march=bpfel -filetype=obj -o - %s \
; RUN: | llvm-objcopy --dump-section .BTF=- - | %python %S/print_btf.py - | FileCheck %s
; RUN: llc -march=bpfeb -filetype=obj -o - %s \
; RUN: | llvm-objcopy --dump-section .BTF=- - | %python %S/print_btf.py - | FileCheck %s
;
; Source:
;   #define __tag1 __attribute__((btf_type_tag("tag1")))
;   int __tag1 *aa;
;   int __tag1  bb;
; Compilation flag:
;   clang -mllvm -btf-type-tag-v2 -S -g -emit-llvm test.c -o test.ll
;
; BTF VAR entry for `b` should be stripped of type tag.

; CHECK: [1] PTR '(anon)' type_id=3
; CHECK: [2] INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK: [3] TYPE_TAG 'tag1' type_id=2
; CHECK: [4] VAR 'aa' type_id=1, linkage=global
; CHECK: [5] VAR 'bb' type_id=2, linkage=global
; CHECK: [6] DATASEC '.bss' size=0 vlen=2
; CHECK: 	type_id=4 offset=0 size=8
; CHECK: 	type_id=5 offset=0 size=4

@aa = dso_local global ptr null, align 8, !dbg !0
@bb = dso_local global i32 0, align 4, !dbg !5

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!11, !12, !13, !14, !15, !16, !17}
!llvm.ident = !{!18}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "aa", scope: !2, file: !3, line: 2, type: !10, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang, some version", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "some-file.c", directory: "/some/dir")
!4 = !{!0, !5}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "bb", scope: !2, file: !3, line: 3, type: !7, isLocal: false, isDefinition: true)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed, annotations: !8)
!8 = !{!9}
!9 = !{!"btf:type_tag", !"tag1"}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!11 = !{i32 7, !"Dwarf Version", i32 5}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = !{i32 8, !"PIC Level", i32 2}
!15 = !{i32 7, !"PIE Level", i32 2}
!16 = !{i32 7, !"uwtable", i32 2}
!17 = !{i32 7, !"frame-pointer", i32 2}
!18 = !{!"clang, some version"}
