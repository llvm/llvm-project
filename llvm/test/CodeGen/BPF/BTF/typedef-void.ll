; RUN: llc -mtriple=bpfel -filetype=obj %s -o %t
; RUN: llvm-objcopy --dump-section .BTF=- %t | %python %p/print_btf.py - | FileCheck %s
;
; Source code:
;   typedef void alias;
;   alias *p;
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm alias.c
;
; CHECK: [[PTR:[0-9]+]]] PTR '(anon)' type_id=[[ALIAS:[0-9]+]]
; CHECK: {{\[}}[[ALIAS]]] TYPEDEF 'alias' type_id=0
; CHECK: VAR 'p' type_id=[[PTR]],

@p = global ptr null, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!6}
!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "p", scope: !2, file: !3, line: 2, type: !4, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, emissionKind: FullDebug, globals: !{!0})
!3 = !DIFile(filename: "alias.c", directory: "/")
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
!5 = !DIDerivedType(tag: DW_TAG_typedef, name: "alias", file: !3, line: 1, baseType: null)
!6 = !{i32 2, !"Debug Info Version", i32 3}
