; RUN: llc -mtriple=bpfel -filetype=obj %s -o %t.el
; RUN: llvm-objcopy --dump-section .BTF=- %t.el | %python %p/print_btf.py - | FileCheck %s
; RUN: llc -mtriple=bpfeb -filetype=obj %s -o %t.eb
; RUN: llvm-objcopy --dump-section .BTF=- %t.eb | %python %p/print_btf.py - | FileCheck %s

; Source code:
;   void *a;
;   typedef void alias;
;   alias *b;
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm t.c

; CHECK: [[VOID_PTR:[0-9]+]]] PTR '(anon)' type_id=0
; CHECK: VAR 'a' type_id=[[VOID_PTR]],
; CHECK: [[ALIAS_PTR:[0-9]+]]] PTR '(anon)' type_id=[[ALIAS:[0-9]+]]
; CHECK: {{\[}}[[ALIAS]]] TYPEDEF 'alias' type_id=0
; CHECK: VAR 'b' type_id=[[ALIAS_PTR]],

@a = dso_local local_unnamed_addr global ptr null, align 8, !dbg !0
@b = dso_local local_unnamed_addr global ptr null, align 8, !dbg !5

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!9}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 1, type: !4, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, emissionKind: FullDebug, globals: !{!0, !5})
!3 = !DIFile(filename: "t.c", directory: "/")
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "b", scope: !2, file: !3, line: 3, type: !7, isLocal: false, isDefinition: true)
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!8 = !DIDerivedType(tag: DW_TAG_typedef, name: "alias", file: !3, line: 2, baseType: null)
!9 = !{i32 2, !"Debug Info Version", i32 3}
