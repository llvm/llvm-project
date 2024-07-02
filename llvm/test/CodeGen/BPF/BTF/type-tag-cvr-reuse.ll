; RUN: llc -march=bpfel -filetype=obj -o - %s \
; RUN: | llvm-objcopy --dump-section .BTF=- - | %python %S/print_btf.py - | FileCheck %s
; RUN: llc -march=bpfeb -filetype=obj -o - %s \
; RUN: | llvm-objcopy --dump-section .BTF=- - | %python %S/print_btf.py - | FileCheck %s
;
; Source:
;   #define __tag1 __attribute__((btf_type_tag("tag1")))
;   
;   const volatile __tag1 int a;
;   volatile int b;
;
; Compilation flag:
;   clang -mllvm -btf-type-tag-v2 -S -g -emit-llvm test.c -o test.ll

; Check that volatile->int chain is reused for variables 'a' and 'b'.

; CHECK: [1] TYPE_TAG 'tag1' type_id=10
; CHECK: [2] TYPE_TAG 'tag1' type_id=6
; CHECK: [3] INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK: [4] TYPE_TAG 'tag1' type_id=3
; CHECK: [5] VAR 'a' type_id=1, linkage=global
; CHECK: [6] VOLATILE '(anon)' type_id=3
; CHECK: [7] VAR 'b' type_id=6, linkage=global
; CHECK: [8] DATASEC '.bss' size=0 vlen=1
; CHECK:    type_id=7 offset=0 size=4
; CHECK: [9] DATASEC '.rodata' size=0 vlen=1
; CHECK:    type_id=5 offset=0 size=4
; CHECK: [10] CONST '(anon)' type_id=6

@a = dso_local constant i32 0, align 4, !dbg !0
@b = dso_local global i32 0, align 4, !dbg !5

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!14, !15, !16, !17}
!llvm.ident = !{!18}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 3, type: !9, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang, some version", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "some-file.c", directory: "/some/dir/")
!4 = !{!0, !5}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "b", scope: !2, file: !3, line: 4, type: !7, isLocal: false, isDefinition: true)
!7 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !8)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !10)
!10 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !11)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed, annotations: !12)
!12 = !{!13}
!13 = !{!"btf:type_tag", !"tag1"}
!14 = !{i32 7, !"Dwarf Version", i32 5}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{i32 1, !"wchar_size", i32 4}
!17 = !{i32 7, !"frame-pointer", i32 2}
!18 = !{!"clang, some version"}
